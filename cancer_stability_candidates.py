"""Research-only view sensitivity candidates; not enabled by any trainer.

Scales must be fitted from the training reference representations only and
frozen for the comparison. View sensitivity is not clinical reliability.
"""
import math

import torch


def scaled_difference_weights(p1, p2, r1, r2, path_scale, rna_scale,
                              strength=2., min_weight=.1):
    """Detached, translation-invariant weights using fixed coordinate scales.

    The mean squared standardized view difference is averaged across the two
    modalities. Its coefficient is a research choice, not an estimated error
    probability. Pathology and RNA may have different dimensions.
    """
    if (p1.ndim != 2 or p2.shape != p1.shape or r1.ndim != 2 or r2.shape != r1.shape
            or len(p1) != len(r1) or not len(p1) or min(p1.shape[1], r1.shape[1]) < 1
            or path_scale.shape != (p1.shape[1],) or rna_scale.shape != (r1.shape[1],)):
        raise ValueError('Expected paired nonempty views and one positive scale per coordinate')
    if (not math.isfinite(strength) or strength < 0 or not math.isfinite(min_weight)
            or not 0 < min_weight <= 1):
        raise ValueError('Invalid weight strength or lower bound')
    values = (p1, p2, r1, r2, path_scale, rna_scale)
    if (any(not x.is_floating_point() or not torch.isfinite(x).all() for x in values)
            or any(x.device != p1.device for x in values)
            or (path_scale <= 0).any() or (rna_scale <= 0).any()):
        raise ValueError('Expected finite floating values on one device and strictly positive scales')
    with torch.no_grad():
        # Float64 accumulation protects the comparison of small view changes.
        dp = ((p1.double()-p2.double())/path_scale.double()).square().mean(1)
        dr = ((r1.double()-r2.double())/rna_scale.double()).square().mean(1)
        energy = (dp+dr)/2
        weights = torch.exp(-strength*energy).clamp(min=min_weight, max=1.)
        ess = weights.sum().square()/weights.square().sum()
    return weights, {'ess': ess, 'standardized_difference_energy': energy}
