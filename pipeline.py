"""Fixed-configuration research pipeline, intended for synthetic validation first.

All selection, time discretization and cohort construction must happen outside
this module. A row is exactly one patient. No test set is used in fitting.
"""
from dataclasses import dataclass
import numpy as np
import torch
from model import ConditionalDeCURSurv, decur_auxiliary, survival_nll, utility_weighted_loss


@dataclass(frozen=True)
class Config:
    hidden: int = 16
    common: int = 8
    rank: int = 4
    epochs: int = 100
    teacher_epochs: int = 100
    learning_rate: float = 0.01
    weight_decay: float = 0.001
    alpha: float = 0.0
    lambda_within: float = 0.0
    lambda_common: float = 0.0
    lambda_unique: float = 0.0
    interaction_l2: float = 0.0
    patch_keep: float = 0.8
    omic_noise: float = 0.05
    folds: int = 3
    seed: int = 17


def check_data(x, g, t, e, mask, ids, n_intervals):
    if x.ndim != 3 or g.ndim != 2 or len(x) != len(g) or len(x) == 0:
        raise ValueError('Expected nonempty aligned [patients,patches,dim] and [patients,genes]')
    if mask.shape != x.shape[:2] or mask.dtype != torch.bool or not mask.any(1).all():
        raise ValueError('Boolean mask must select at least one patch per patient')
    if not torch.isfinite(x[mask]).all() or not torch.isfinite(g).all():
        raise ValueError('Observed features must be finite; imputation is not implemented')
    if len(ids) != len(x) or len(set(ids)) != len(ids):
        raise ValueError('IDs must uniquely identify patients')
    survival_nll(torch.zeros(len(x), n_intervals, device=x.device), t, e)


def make_views(x, g, mask, cfg):
    if not 0 < cfg.patch_keep <= 1 or cfg.omic_noise < 0:
        raise ValueError('Invalid augmentation parameters')
    views = []
    for _ in range(2):
        m = mask & (torch.rand(mask.shape, device=mask.device) < cfg.patch_keep)
        empty = ~m.any(1)
        m[empty] = mask[empty]  # Keep all patches if the sampled view is empty.
        views.append((x, g + torch.randn_like(g) * cfg.omic_noise, m))
    return views


def fit(x, g, t, e, mask, n_intervals, cfg, *, interaction=True, teacher_loss=None, seed=None, validation=None):
    if cfg.epochs < 1 or cfg.learning_rate <= 0 or cfg.alpha < 0 or cfg.interaction_l2 < 0:
        raise ValueError('Invalid training configuration')
    if cfg.alpha > 0 and teacher_loss is None:
        raise ValueError('Nonzero alpha requires cross-fitted teacher losses')
    torch.manual_seed(cfg.seed if seed is None else seed)
    model = ConditionalDeCURSurv(x.shape[-1], g.shape[-1], cfg.hidden, cfg.common,
                                n_intervals, cfg.rank, interaction=interaction).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x, g, mask)
        loss = (survival_nll(out['logits'], t, e) if cfg.alpha == 0 else
                utility_weighted_loss(out['logits'], t, e, teacher_loss, cfg.alpha))
        loss = loss + cfg.interaction_l2 * out['interaction'].square().mean()
        if cfg.lambda_within or cfg.lambda_common or cfg.lambda_unique:
            (x1,g1,m1),(x2,g2,m2) = make_views(x,g,mask,cfg)
            p1, q1 = model.encode(x1,g1,m1)
            p2, q2 = model.encode(x2,g2,m2)
            aux = decur_auxiliary(p1,p2,q1,q2,cfg.common)
            loss = loss + cfg.lambda_within*aux['within'] + cfg.lambda_common*aux['common'] + cfg.lambda_unique*aux['unique']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        optimizer.step()
        if validation is not None:
            vx, vg, vt, ve, vm = validation
            model.eval()
            with torch.no_grad():
                vout = model(vx, vg, vm)
                score = survival_nll(vout['logits'], vt, ve).item()
                magnitude = vout['interaction'].square().mean().sqrt().item()
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite validation NLL')
            history.append({'epoch':epoch+1, 'validation_nll':score, 'interaction_rms':magnitude})
    model.validation_history = history
    return model


def cross_fit_teacher(x, g, t, e, mask, ids, n_intervals, cfg):
    """Return OOF losses, fold IDs and per-fold train statistics.

    Teacher configuration is fixed in advance: no OOF or test driven selection.
    Omics standardization is separately fitted inside each teacher train fold.
    """
    from dataclasses import replace
    check_data(x,g,t,e,mask,ids,n_intervals)
    if not 2 <= cfg.folds <= len(ids):
        raise ValueError('Invalid number of folds')
    rng = np.random.default_rng(cfg.seed)
    folds = np.array_split(rng.permutation(len(ids)), cfg.folds)
    losses = torch.full((len(ids),), float('nan'), device=x.device)
    fold_id = np.full(len(ids), -1, dtype=int)
    audits = []
    teacher_cfg = replace(cfg, epochs=cfg.teacher_epochs, alpha=0, lambda_within=0,
                          lambda_common=0, lambda_unique=0)
    for k, held in enumerate(folds):
        train = np.concatenate([f for j,f in enumerate(folds) if j != k])
        mu = g[train].mean(0)
        sd = g[train].std(0, unbiased=False).clamp_min(1e-6)
        standardized = (g-mu)/sd
        model = fit(x[train], standardized[train], t[train], e[train], mask[train],
                    n_intervals, teacher_cfg, interaction=False, seed=cfg.seed+k+1)
        model.eval()
        with torch.no_grad():
            logits = model(x[held], standardized[held], mask[held])['logits']
            losses[held] = survival_nll(logits, t[held], e[held], reduction='none')
        fold_id[held] = k
        audits.append({'fold':k, 'train_ids':[ids[i] for i in train],
                       'heldout_ids':[ids[i] for i in held],
                       'omic_mean':mu.cpu().tolist(), 'omic_scale':sd.cpu().tolist()})
    if not torch.isfinite(losses).all() or (fold_id < 0).any():
        raise RuntimeError('Incomplete cross fitting')
    return losses, fold_id, audits


def fit_outer_train(x, g, t, e, mask, ids, n_intervals, cfg, *, interaction=True):
    """Fit only on supplied training rows; returns model and training normalization."""
    check_data(x,g,t,e,mask,ids,n_intervals)
    teacher, folds, audit = (None, None, [])
    if cfg.alpha:
        teacher, folds, audit = cross_fit_teacher(x,g,t,e,mask,ids,n_intervals,cfg)
    mu = g.mean(0)
    sd = g.std(0, unbiased=False).clamp_min(1e-6)
    model = fit(x, (g-mu)/sd, t, e, mask, n_intervals, cfg,
                interaction=interaction, teacher_loss=teacher)
    return model, mu, sd, teacher, folds, audit


def select_and_fit(x, g, t, e, mask, ids, n_intervals, cfg, *, interaction=True):
    """Select epoch inside supplied training cohort, then refit on all of it.

    All inner preprocessing and teacher predictions are fitted on inner-train
    only. Hyperparameters, maximum epoch and validation fraction are fixed.
    """
    from dataclasses import replace
    check_data(x,g,t,e,mask,ids,n_intervals)
    if len(ids) < max(8, cfg.folds + 2):
        raise ValueError('Insufficient patients for inner selection and cross-fitting')
    order = np.random.default_rng(cfg.seed+999).permutation(len(ids))
    count = max(2, len(ids)//4)
    val, train = order[:count], order[count:]
    train_ids = [ids[i] for i in train]
    teacher = None
    if cfg.alpha:
        teacher, _, _ = cross_fit_teacher(x[train],g[train],t[train],e[train],
                                           mask[train],train_ids,n_intervals,cfg)
    mu = g[train].mean(0)
    sd = g[train].std(0,unbiased=False).clamp_min(1e-6)
    inner = fit(x[train],(g[train]-mu)/sd,t[train],e[train],mask[train],n_intervals,cfg,
                interaction=interaction,teacher_loss=teacher,
                validation=(x[val],(g[val]-mu)/sd,t[val],e[val],mask[val]))
    history = inner.validation_history
    selected = min(history, key=lambda row: row['validation_nll'])['epoch']
    refit = fit_outer_train(x,g,t,e,mask,ids,n_intervals,replace(cfg,epochs=selected),
                            interaction=interaction)
    audit = {'inner_train_ids':train_ids, 'inner_validation_ids':[ids[i] for i in val],
             'selected_epoch':selected, 'history':history}
    return (*refit, audit)
