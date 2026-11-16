"""Shared DeCUR auxiliary losses used by the current WSI/RNA framework."""
import torch
from torch.nn import functional as F


class DegenerateCorrelation(ValueError):
    """An auxiliary correlation is not estimable for this batch."""


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
