"""Train-fitted RNA scaling and an optional fixed low-variance mask."""
import math
import torch


def fit_rna_scaler(train, policy='clamp', min_std=1e-6):
    """Fit per-gene statistics without validation/test observations or labels.

    mask preserves the original feature dimension and sets excluded coordinates
    to zero everywhere. clamp retains the historical scale-floor behavior.
    none keeps the raw expression values unchanged, which is the convention of
    the reference comparison pipeline.
    """
    if policy not in ('clamp', 'mask', 'none') or not math.isfinite(min_std) or min_std <= 0:
        raise ValueError('RNA policy must be clamp/mask/none with a finite positive min_std')
    if (not isinstance(train, torch.Tensor) or train.ndim != 2 or min(train.shape) < 1
            or not train.is_floating_point() or not torch.isfinite(train).all()):
        raise ValueError('RNA scaler requires a finite nonempty floating [patients,genes] tensor')
    mean = train.mean(0)
    std = train.std(0, unbiased=False)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise FloatingPointError('Training RNA mean/std is not finite')
    active = torch.ones_like(std, dtype=torch.bool)
    if policy == 'none':
        # No centering and no rescaling: the raw values are the model input.
        mean = torch.zeros_like(mean)
        std = torch.ones_like(std)
    if policy == 'mask':
        # Exact range also detects constants if reduction rounding gives a tiny std.
        active = (std >= min_std) & (train.amax(0) > train.amin(0))
        if not active.any():
            raise ValueError('No training-variable RNA genes remain under the selected threshold')
    scale = std.clamp_min(min_std)
    scale = torch.where(active, scale, torch.ones_like(scale))
    metadata = dict(policy=policy, min_std=float(min_std), input_genes=train.shape[1],
                    active_genes=int(active.sum()), masked_genes=int((~active).sum()),
                    masked_gene_indices=(~active).nonzero().flatten().tolist())
    fitted = dict(rna_mean=mean, rna_scale=scale, rna_mask=active,
                  rna_train_std=std, rna_preprocessing=metadata)
    metadata['train_max_abs'] = transform_rna(train, fitted).abs().max().item()
    return fitted


def transform_rna(rna, fitted):
    """Use a saved train-only transform; never refit or clip held-out values."""
    if (not isinstance(rna, torch.Tensor) or rna.ndim != 2 or not rna.is_floating_point()
            or not torch.isfinite(rna).all()):
        raise ValueError('RNA transform requires a finite floating matrix')
    mean, scale = fitted['rna_mean'], fitted['rna_scale']
    if mean.shape != (rna.shape[1],) or scale.shape != mean.shape:
        raise ValueError('Saved RNA statistics do not match input gene dimensions')
    mean, scale = (value.to(device=rna.device, dtype=rna.dtype) for value in (mean, scale))
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Saved RNA statistics must be finite with positive scales')
    mask = fitted.get('rna_mask')
    if mask is None:
        if fitted.get('rna_preprocessing', {}).get('policy') == 'mask':
            raise ValueError('Masked RNA checkpoint requires its saved rna_mask')
        # Backward compatibility with checkpoints saved before RNA masking existed.
        mask = torch.ones_like(mean, dtype=torch.bool)
    if mask.dtype != torch.bool or mask.shape != mean.shape:
        raise ValueError('Saved RNA mask must be boolean and match gene dimensions')
    mask = mask.to(rna.device)
    centered = torch.where(mask, rna-mean, torch.zeros_like(rna))
    normalized = centered/scale
    if not torch.isfinite(normalized).all():
        raise FloatingPointError('RNA normalization produced nonfinite active values')
    return normalized


def rna_noise_view(normalized, strength, mask=None):
    """Preserve excluded genes at zero in auxiliary views too."""
    noise = strength*torch.randn_like(normalized)
    if mask is not None:
        noise = noise*mask.to(device=normalized.device, dtype=normalized.dtype)
    return normalized+noise
