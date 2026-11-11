"""Research prototype; inputs are already extracted, paired patient features."""
import torch
from torch import nn
from torch.nn import functional as F


class DegenerateCorrelation(ValueError):
    """An auxiliary correlation is not estimable for this batch."""


class ConditionalDeCURSurv(nn.Module):
    def __init__(self, path_dim, omic_dim, hidden=64, common=32, intervals=4, rank=16, interaction=True):
        super().__init__()
        if hidden <= common or common < 1:
            raise ValueError('Require 0 < common < hidden')
        self.common = common
        self.use_interaction = interaction
        self.path = nn.Sequential(nn.Linear(path_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.omic = nn.Sequential(nn.Linear(omic_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.path_head = nn.Linear(hidden, intervals, bias=False)
        self.omic_head = nn.Linear(hidden, intervals, bias=False)
        self.bias = nn.Parameter(torch.zeros(intervals))
        self.path_factor = nn.Linear(hidden, rank, bias=False)
        self.omic_factor = nn.Linear(hidden, rank, bias=False)
        self.interaction_head = nn.Linear(rank, intervals, bias=False)

    def encode(self, patches, omics, mask=None):
        if patches.ndim != 3 or omics.ndim != 2 or patches.shape[0] != omics.shape[0]:
            raise ValueError('Expected patches [B,N,D], omics [B,G]')
        if mask is None:
            mask = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
        if mask.shape != patches.shape[:2] or (~mask.bool()).all(dim=1).any():
            raise ValueError('Each patient must have at least one valid patch')
        # Zero padded values before pooling, including potential NaNs.
        valid = patches.masked_fill(~mask.bool().unsqueeze(-1), 0)
        pooled = valid.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.path(pooled), self.omic(omics)

    def forward(self, patches, omics, mask=None):
        p, g = self.encode(patches, omics, mask)
        main_p, main_g = self.path_head(p), self.omic_head(g)
        interaction = (self.interaction_head(self.path_factor(p) * self.omic_factor(g))
                       if self.use_interaction else torch.zeros_like(main_p))
        logits = self.bias + main_p + main_g + interaction
        return {'logits': logits, 'path': p, 'omic': g, 'interaction': interaction}


def survival_nll(logits, interval, event, reduction='mean'):
    """Event in bin k: survive bins < k, event in k; censored: survive through k.

    Discretization must follow this end-of-bin censoring convention.
    """
    if logits.ndim != 2 or interval.shape != logits.shape[:1] or event.shape != interval.shape:
        raise ValueError('Incompatible label shapes')
    if interval.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('Interval labels must be integers')
    if ((interval < 0) | (interval >= logits.shape[1])).any() or not torch.all((event == 0) | (event == 1)):
        raise ValueError('Invalid interval or event')
    idx = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    before = idx < interval.unsqueeze(1)
    at = idx == interval.unsqueeze(1)
    log_survive = F.logsigmoid(-logits)
    log_event = F.logsigmoid(logits)
    ll = (log_survive * before).sum(1)
    ll = ll + (at * (event.unsqueeze(1) * log_event + (1-event.unsqueeze(1)) * log_survive)).sum(1)
    if reduction == 'none':
        return -ll
    if reduction == 'mean':
        return -ll.mean()
    raise ValueError('reduction must be mean or none')


def correlation_loss(a, b, target, off_weight=0.005, eps=1e-4):
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] < 2:
        raise ValueError('Correlation requires matched [B,D] tensors and B >= 2')
    a = (a - a.mean(0)) / torch.sqrt(a.var(0, unbiased=False) + eps)
    b = (b - b.mean(0)) / torch.sqrt(b.var(0, unbiased=False) + eps)
    c = a.T @ b / a.shape[0]
    diag = torch.diagonal(c)
    off = c - torch.diag_embed(diag)
    return (diag - target).square().sum() + off_weight * off.square().sum()


def decur_auxiliary(p1, p2, g1, g2, common, off_weight=0.005):
    """DeCUR-style view consistency and common/unique cross correlations."""
    if not 0 < common < p1.shape[1]:
        raise ValueError('Invalid common dimension')
    within = (correlation_loss(p1, p2, 1, off_weight) + correlation_loss(g1, g2, 1, off_weight)) / 2
    shared = correlation_loss(p1[:, :common], g1[:, :common], 1, off_weight)
    unique = correlation_loss(p1[:, common:], g1[:, common:], 0, off_weight)
    return {'within': within, 'common': shared, 'unique': unique}


def stability_weighted_common(p1, p2, g1, g2, strength=2.0, min_weight=0.1,
                              off_weight=0.005, eps=1e-4, min_ess_fraction=0.5,
                              shuffle_weights=False, shuffle_generator=None):
    """Candidate only: view-stability weighted common correlation.

    The weights are detached and measure sensitivity to the chosen views only.
    Degenerate batches fail explicitly; callers must record and handle them.
    """
    if p1.ndim != 2 or any(x.shape != p1.shape for x in (p2,g1,g2)) or p1.shape[0] < 2:
        raise ValueError('Expected matching [B,D] views with B >= 2')
    if strength < 0 or not 0 < min_weight <= 1 or not 0 < min_ess_fraction <= 1:
        raise ValueError('Invalid weighting parameters')
    if not all(torch.isfinite(x).all() for x in (p1,p2,g1,g2)):
        raise ValueError('Nonfinite views')
    # Cosine distance normalized to [0,1], bounded for numerical error.
    with torch.no_grad():
        dp = ((1-F.cosine_similarity(p1,p2,dim=1,eps=eps))/2).clamp(0,1)
        dg = ((1-F.cosine_similarity(g1,g2,dim=1,eps=eps))/2).clamp(0,1)
        weights = torch.exp(-strength*(dp+dg)).clamp(min=min_weight,max=1)
        if shuffle_weights:
            weights = weights[torch.randperm(len(weights),device=weights.device,
                                            generator=shuffle_generator)]
        ess = weights.sum().square()/weights.square().sum()
        if ess < min_ess_fraction*p1.shape[0]:
            raise DegenerateCorrelation('Insufficient effective sample size')
    weights_col = weights[:,None]
    def standardize(x):
        centered = x - (weights_col*x).sum(0)/weights.sum()
        variance = (weights_col*centered.square()).sum(0)/weights.sum()
        if (variance < eps).any():
            raise DegenerateCorrelation('Degenerate weighted common feature variance')
        return centered / torch.sqrt(variance + eps)
    p = standardize(p1)
    g = standardize(g1)
    corr = (p*weights_col).T @ g / weights.sum()
    diagonal = torch.diagonal(corr)
    off = corr - torch.diag_embed(diagonal)
    loss = (diagonal-1).square().sum() + off_weight*off.square().sum()
    return loss, {'weights':weights, 'ess':ess}


def utility_weighted_loss(logits, interval, event, teacher_additive_nll, weight=1.0):
    """Optional bounded residual-priority loss; teacher NLL must be out-of-fold.

    Detaching the teacher prevents learning the weights. Set weight=0 for uniform
    training. Out-of-fold scores come from an independently fitted additive model.
    """
    if teacher_additive_nll.shape != interval.shape or not torch.isfinite(teacher_additive_nll).all():
        raise ValueError('Need one finite out-of-fold teacher loss per patient')
    if weight < 0:
        raise ValueError('weight must be nonnegative')
    per_patient = survival_nll(logits, interval, event, reduction='none')
    # Bounded and positive; higher teacher error means greater attention, not proof of interaction.
    w = 1 + weight * torch.sigmoid(teacher_additive_nll.detach())
    return (w * per_patient).sum() / w.sum()
