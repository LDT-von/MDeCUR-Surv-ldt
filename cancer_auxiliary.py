"""Local pathology-view objective; no patch-to-bulk-RNA matching is inferred."""
import math
import torch


def patient_balanced_local_correlation(a, b, mask, off_weight=0.005, eps=1e-4):
    """Centered, patient-balanced Barlow-style loss on aligned [B,N,D] views.

    Each valid patch in patient i has weight 1/(B*N_i). Means, variances and
    cross-correlations all use these weights. Patch count is not patient count.
    Cache provenance, rather than feature similarity, must establish pairing.
    A finite epsilon keeps constant dimensions measurable; diagnostics flag
    them instead of claiming this objective guarantees prevention of collapse.
    """
    if a.ndim != 3 or a.shape != b.shape or a.shape[-1] < 1:
        raise ValueError('Local views require matching [B,N,D] tensors')
    if (mask.dtype != torch.bool or mask.shape != a.shape[:2]
            or mask.device != a.device or b.device != a.device or b.dtype != a.dtype):
        raise ValueError('Local views require a matching boolean mask, device and dtype')
    if a.shape[0] < 2 or not mask.any(1).all():
        raise ValueError('Local correlation requires at least two patients with nonempty bags')
    if not math.isfinite(off_weight) or off_weight < 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError('Invalid local correlation coefficients')
    av, bv = a[mask], b[mask]
    if not torch.isfinite(av).all() or not torch.isfinite(bv).all():
        raise ValueError('Nonfinite valid local embeddings')
    counts = mask.sum(1, keepdim=True)
    weights = (mask.to(a.dtype) / counts / a.shape[0])[mask].unsqueeze(1)
    ac = av - (weights * av).sum(0)
    bc = bv - (weights * bv).sum(0)
    va = (weights * ac.square()).sum(0)
    vb = (weights * bc.square()).sum(0)
    az, bz = ac / (va + eps).sqrt(), bc / (vb + eps).sqrt()
    corr = (weights * az).T @ bz
    diagonal = corr.diagonal()
    off_diagonal = corr - torch.diag_embed(diagonal)
    loss = (diagonal - 1).square().sum() + off_weight * off_diagonal.square().sum()
    return loss, {
        'patients': a.shape[0], 'matched_patches': int(mask.sum()),
        'variance_min': torch.minimum(va.min(), vb.min()).detach().item(),
        'low_variance_dimensions': int(((va < eps) | (vb < eps)).sum()),
        'diagonal_mean': diagonal.detach().mean().item(),
    }
