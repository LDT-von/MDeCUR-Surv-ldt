"""Full-batch research trainer with patient split and train-only RNA scaling."""
from dataclasses import dataclass
from copy import deepcopy
import math
import torch
from cancer_model import CancerDeCUR, piecewise_nll
from model import decur_auxiliary, stability_weighted_common, DegenerateCorrelation


@dataclass(frozen=True)
class CancerConfig:
    hidden: int = 16
    common: int = 8
    epochs: int = 80
    patience: int = 15
    lr: float = 0.003
    weight_decay: float = 0.001
    lambda_within: float = 0.001
    lambda_common: float = 0.001
    lambda_unique: float = 0.001
    common_mode: str = 'stability'  # off, uniform, stability, shuffled
    strength: float = 2.0
    min_weight: float = 0.1
    patch_keep: float = 0.8
    rna_noise: float = 0.05
    seed: int = 17


def validate_cohort(data, starts, require_labels=True):
    required = ('patches','rna','mask','ids')
    if any(k not in data for k in required):
        raise ValueError('Missing patient cohort fields')
    x,g,m,ids = (data[k] for k in required)
    n = len(ids)
    if not n or len(set(ids)) != n or x.ndim != 3 or g.ndim != 2 or m.dtype != torch.bool:
        raise ValueError('One unique ID per nonempty patient row, valid feature shapes and boolean masks required')
    if x.shape[0] != n or g.shape[0] != n or m.shape != x.shape[:2] or not m.any(1).all():
        raise ValueError('Cohort fields have inconsistent lengths or empty bags')
    if not torch.isfinite(x[m]).all() or not torch.isfinite(g).all():
        raise ValueError('Missing/nonfinite features require explicit train-only preprocessing')
    if require_labels:
        if 'time' not in data or 'event' not in data:
            raise ValueError('Missing survival labels')
        t,e = data['time'],data['event']
        piecewise_nll(torch.zeros(n,len(starts),device=t.device),t,e,starts)


def view_batch(x, g, mask, cfg):
    views=[]
    for _ in range(2):
        chosen = mask & (torch.rand(mask.shape,device=mask.device)<cfg.patch_keep)
        empty = ~chosen.any(1)
        chosen[empty] = mask[empty]
        views.append((x,g+cfg.rna_noise*torch.randn_like(g),chosen))
    return views


def train_cancer_model(train, validation, starts, cfg):
    """Fit on train; select the validation-NLL-best checkpoint; no test input."""
    if cfg.common_mode not in ('off','uniform','stability','shuffled'):
        raise ValueError('Invalid common mode')
    if cfg.epochs < 1 or cfg.patience < 1 or cfg.lr <= 0 or not 0 < cfg.patch_keep <= 1 or cfg.rna_noise < 0:
        raise ValueError('Invalid training configuration')
    if min(cfg.lambda_within,cfg.lambda_common,cfg.lambda_unique,cfg.strength) < 0:
        raise ValueError('Negative loss or weight strength')
    validate_cohort(train,starts)
    validate_cohort(validation,starts)
    if set(train['ids']) & set(validation['ids']):
        raise ValueError('Patient overlap between training and validation')
    if train['rna'].shape[1] != validation['rna'].shape[1] or train['patches'].shape[-1] != validation['patches'].shape[-1]:
        raise ValueError('Feature dimensions differ')
    torch.manual_seed(cfg.seed)
    mean = train['rna'].mean(0)
    scale = train['rna'].std(0,unbiased=False).clamp_min(1e-6)
    x, g, m = train['patches'], (train['rna']-mean)/scale, train['mask']
    vx, vg, vm = validation['patches'], (validation['rna']-mean)/scale, validation['mask']
    model = CancerDeCUR(x.shape[-1],g.shape[-1],cfg.hidden,cfg.common,len(starts)).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    best = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history=[]
    skipped = 0
    for epoch in range(1,cfg.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x,g,m)
        nll = piecewise_nll(out['log_rates'],train['time'],train['event'],starts)
        loss = nll
        stats = {'ess':None,'weight_mean':None,'weight_min':None,'weight_std':None,'common_loss':None,'skipped':False}
        if cfg.lambda_within or cfg.lambda_unique or (cfg.lambda_common and cfg.common_mode != 'off'):
            (x1,g1,m1),(x2,g2,m2) = view_batch(x,g,m,cfg)
            p1,r1,_ = model.encode(x1,g1,m1)
            p2,r2,_ = model.encode(x2,g2,m2)
            aux = decur_auxiliary(p1,p2,r1,r2,cfg.common)
            loss = loss + cfg.lambda_within*aux['within'] + cfg.lambda_unique*aux['unique']
            if cfg.lambda_common and cfg.common_mode != 'off':
                try:
                    common, w = stability_weighted_common(
                        p1[:,:cfg.common],p2[:,:cfg.common],r1[:,:cfg.common],r2[:,:cfg.common],
                        strength=0 if cfg.common_mode == 'uniform' else cfg.strength,
                        min_weight=cfg.min_weight,shuffle_weights=cfg.common_mode == 'shuffled')
                    loss = loss + cfg.lambda_common*common
                    stats.update(ess=w['ess'].item(),weight_mean=w['weights'].mean().item(),
                                 weight_min=w['weights'].min().item(),
                                 weight_std=w['weights'].std(unbiased=False).item(),common_loss=common.item())
                except DegenerateCorrelation:
                    stats['skipped'] = True
                    skipped += 1
        if not torch.isfinite(loss):
            raise FloatingPointError('Training loss is not finite')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = piecewise_nll(model(vx,vg,vm)['log_rates'],validation['time'],validation['event'],starts).item()
        if not math.isfinite(val):
            raise FloatingPointError('Validation loss is not finite')
        history.append({'epoch':epoch,'train_nll':nll.item(),'validation_nll':val,**stats})
        if val < best:
            best, best_epoch, best_state, stale = val, epoch, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return {'model':model,'rna_mean':mean,'rna_scale':scale,'starts':starts.clone(),
            'best_epoch':best_epoch,'validation_nll':best,'history':history,'skipped_common':skipped}


def predict(fitted, cohort):
    validate_cohort(cohort,fitted['starts'],require_labels=False)
    model = fitted['model']
    model.eval()
    with torch.no_grad():
        return model(cohort['patches'],(cohort['rna']-fitted['rna_mean'])/fitted['rna_scale'],cohort['mask'])
