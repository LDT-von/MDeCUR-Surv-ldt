"""Full-batch research trainer with patient split and train-only RNA scaling."""
from dataclasses import dataclass
from copy import deepcopy
import math
import torch
from cancer_model import CancerDeCUR, piecewise_nll
from cancer_auxiliary import patient_balanced_local_correlation
from cancer_rna import fit_rna_scaler, transform_rna, rna_noise_view
from model import correlation_loss, stability_weighted_common, DegenerateCorrelation


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
    path_views: str = 'legacy_patch_dropout'
    prediction_space: str = 'projected'  # historical path; encoder separates prediction/auxiliary
    lambda_local: float = 0.0
    local_dim: int = 16
    lambda_path_within: float | None = None
    lambda_rna_within: float | None = None
    rna_scale_policy: str = 'clamp'
    rna_min_std: float = 1e-6


def effective_loss_weights(cfg):
    """Explicit per-modality overrides; absent overrides preserve the old mean."""
    return {
        'path_within': cfg.lambda_within / 2 if cfg.lambda_path_within is None else cfg.lambda_path_within,
        'rna_within': cfg.lambda_within / 2 if cfg.lambda_rna_within is None else cfg.lambda_rna_within,
        'local': cfg.lambda_local,
        'common': 0.0 if cfg.common_mode == 'off' else cfg.lambda_common,
        'unique': cfg.lambda_unique,
    }


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
    keys=('patches_view1','patches_view2')
    if any(k in data for k in keys):
        if not all(k in data for k in keys):
            raise ValueError('Both cached pathology views are required')
        for key in keys:
            view=data[key]
            if view.shape != x.shape or view.device != x.device or not torch.isfinite(view[m]).all():
                raise ValueError('Cached views must be finite and match the complete base bag')
    if require_labels:
        if 'time' not in data or 'event' not in data:
            raise ValueError('Missing survival labels')
        t,e = data['time'],data['event']
        piecewise_nll(torch.zeros(n,len(starts),device=t.device),t,e,starts)


def view_batch(x, g, mask, cfg, cached_views=None, rna_mask=None):
    if cfg.path_views == 'cached_uni':
        if cfg.patch_keep != 1 or cached_views is None or len(cached_views) != 2:
            raise ValueError('cached_uni requires two full cached views and patch_keep=1')
        if any(v.shape != x.shape for v in cached_views):
            raise ValueError('Cached views must preserve every input patch row')
        return [(v,rna_noise_view(g,cfg.rna_noise,rna_mask),mask) for v in cached_views]
    if cfg.path_views != 'legacy_patch_dropout' or cached_views is not None:
        raise ValueError('Choose an explicit compatible pathology view source')
    views=[]
    for _ in range(2):
        chosen = mask & (torch.rand(mask.shape,device=mask.device)<cfg.patch_keep)
        empty = ~chosen.any(1)
        chosen[empty] = mask[empty]
        views.append((x,rna_noise_view(g,cfg.rna_noise,rna_mask),chosen))
    return views


def train_cancer_model(train, validation, starts, cfg):
    """Fit on train; select the validation-NLL-best checkpoint; no test input."""
    if cfg.common_mode not in ('off','uniform','stability','shuffled'):
        raise ValueError('Invalid common mode')
    if cfg.path_views not in ('legacy_patch_dropout','cached_uni'):
        raise ValueError('Invalid pathology view source')
    if cfg.path_views == 'cached_uni' and cfg.patch_keep != 1:
        raise ValueError('cached_uni must preserve all patches: patch_keep=1')
    if cfg.epochs < 1 or cfg.patience < 1 or cfg.lr <= 0 or not 0 < cfg.patch_keep <= 1 or cfg.rna_noise < 0:
        raise ValueError('Invalid training configuration')
    weights = effective_loss_weights(cfg)
    if any(not math.isfinite(v) or v < 0 for v in (*weights.values(), cfg.lambda_within, cfg.lambda_common, cfg.strength)):
        raise ValueError('Loss coefficients and weight strength must be finite and nonnegative')
    if not 0 < cfg.min_weight <= 1:
        raise ValueError('min_weight must be in (0,1]')
    if cfg.prediction_space not in ('projected', 'encoder'):
        raise ValueError('Invalid prediction space')
    if not isinstance(cfg.local_dim, int) or cfg.local_dim < 1:
        raise ValueError('local_dim must be a positive integer')
    if cfg.lambda_local and cfg.path_views != 'cached_uni':
        raise ValueError('Local consistency requires aligned cached_uni image views')
    validate_cohort(train,starts)
    validate_cohort(validation,starts)
    cached_views=None
    if cfg.path_views == 'cached_uni':
        if not all(k in train for k in ('patches_view1','patches_view2')):
            raise ValueError('Missing cached UNI training views')
        cached_views=(train['patches_view1'],train['patches_view2'])
    elif any(k in train for k in ('patches_view1','patches_view2')):
        raise ValueError('Cached views supplied but legacy patch dropping was selected')
    if set(train['ids']) & set(validation['ids']):
        raise ValueError('Patient overlap between training and validation')
    if train['rna'].shape[1] != validation['rna'].shape[1] or train['patches'].shape[-1] != validation['patches'].shape[-1]:
        raise ValueError('Feature dimensions differ')
    if any(weights.values()) and len(train['ids']) < 2:
        raise ValueError('Auxiliary correlation requires at least two training patients')
    torch.manual_seed(cfg.seed)
    rna_fitted = fit_rna_scaler(train['rna'], cfg.rna_scale_policy, cfg.rna_min_std)
    x, g, m = train['patches'], transform_rna(train['rna'],rna_fitted), train['mask']
    vx, vg, vm = validation['patches'], transform_rna(validation['rna'],rna_fitted), validation['mask']
    rna_fitted['rna_preprocessing']['validation_max_abs'] = vg.abs().max().item()
    model_kwargs = {'path_dim': x.shape[-1], 'rna_dim': g.shape[-1], 'hidden': cfg.hidden,
                    'common': cfg.common, 'bins': len(starts), 'prediction_space': cfg.prediction_space,
                    'local_dim': cfg.local_dim if cfg.lambda_local else None}
    model = CancerDeCUR(**model_kwargs).to(x.device)
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
        stats = {'ess':None,'weight_mean':None,'weight_min':None,'weight_std':None,'common_loss':None,'skipped':False,
                 'path_within_loss':None,'rna_within_loss':None,'unique_loss':None,
                 'local_loss':None,'local_diagnostics':None}
        if any(weights.values()):
            (x1,g1,m1),(x2,g2,m2) = view_batch(x,g,m,cfg,cached_views,rna_fitted['rna_mask'])
            v1 = model.encode_auxiliary(x1,g1,m1,include_local=bool(weights['local']))
            v2 = model.encode_auxiliary(x2,g2,m2,include_local=bool(weights['local']))
            p1,r1,p2,r2 = v1['path'],v1['rna'],v2['path'],v2['rna']
            if weights['path_within']:
                within_path = correlation_loss(p1,p2,1)
                loss = loss + weights['path_within']*within_path
                stats['path_within_loss'] = within_path.item()
            if weights['rna_within']:
                within_rna = correlation_loss(r1,r2,1)
                loss = loss + weights['rna_within']*within_rna
                stats['rna_within_loss'] = within_rna.item()
            if weights['unique']:
                unique = correlation_loss(p1[:,cfg.common:],r1[:,cfg.common:],0)
                loss = loss + weights['unique']*unique
                stats['unique_loss'] = unique.item()
            if weights['local']:
                if not torch.equal(m1,m2):
                    raise ValueError('Local view masks must preserve identical patch rows')
                local, diagnostics = patient_balanced_local_correlation(v1['local'],v2['local'],m1)
                loss = loss + weights['local']*local
                stats.update(local_loss=local.item(),local_diagnostics=diagnostics)
            if weights['common']:
                try:
                    common, w = stability_weighted_common(
                        p1[:,:cfg.common],p2[:,:cfg.common],r1[:,:cfg.common],r2[:,:cfg.common],
                        strength=0 if cfg.common_mode == 'uniform' else cfg.strength,
                        min_weight=cfg.min_weight,shuffle_weights=cfg.common_mode == 'shuffled')
                    loss = loss + weights['common']*common
                    stats.update(ess=w['ess'].item(),weight_mean=w['weights'].mean().item(),
                                 weight_min=w['weights'].min().item(),
                                 weight_std=w['weights'].std(unbiased=False).item(),common_loss=common.item())
                except DegenerateCorrelation:
                    stats['skipped'] = True
                    skipped += 1
        if not torch.isfinite(loss):
            raise FloatingPointError('Training loss is not finite')
        stats['total_loss'] = loss.item()
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
    return {'model':model,**rna_fitted,'starts':starts.clone(),
            'best_epoch':best_epoch,'validation_nll':best,'history':history,'skipped_common':skipped,
            'model_kwargs':model_kwargs,'effective_loss_weights':weights}


def predict(fitted, cohort):
    validate_cohort(cohort,fitted['starts'],require_labels=False)
    model = fitted['model']
    model.eval()
    with torch.no_grad():
        return model(cohort['patches'],transform_rna(cohort['rna'],fitted),cohort['mask'])
