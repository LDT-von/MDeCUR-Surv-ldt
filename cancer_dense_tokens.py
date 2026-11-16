"""DenseTwins-inspired token objectives for one WSI/bulk-RNA patient pair.

This is an adaptation, not the authors' implementation. RNA pathway tokens
have no measured spatial correspondence to pathology patches. Token rows are
not independent patients. Correlations are centered and epsilon-stabilized,
consistent with this project's objectives, unlike the uncentered formula in
DenseTwins equation (1). There is no synergy/PID objective here.

The caller controls token selection/maximum count before invoking this module.
Nearest-neighbor search chunks both axes; autograd follows gathered values,
but never the discrete matching indices. Low-variance objectives are returned
with an explicit invalid diagnostic, not interpreted as successful decoupling.
"""

import math

import torch
from torch.nn import functional as F


def _matrix(value, name, minimum=1):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2
            or value.shape[0] < minimum or value.shape[1] < 1
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError(f'{name} must be a finite floating [tokens, dimensions] tensor '
                         f'with at least {minimum} token(s)')


def _same_space(a, b):
    if a.shape[1] != b.shape[1] or a.device != b.device or a.dtype != b.dtype:
        raise ValueError('Matched tensors require equal feature dimensions, device and dtype')


@torch.no_grad()
def chunked_cosine_nearest_neighbor(source, target, chunk_size=256, eps=1e-8):
    """Return source->target indices; ties choose the first target index.

    Many source tokens may select one target. Zero/near-zero norm vectors are
    rejected because their cosine direction is not estimable.
    """
    _matrix(source, 'source')
    _matrix(target, 'target')
    _same_space(source, target)
    if type(chunk_size) is not int or chunk_size < 1 or not math.isfinite(eps) or eps <= 0:
        raise ValueError('Positive chunk_size and finite positive eps are required')
    if (source.norm(dim=1) <= eps).any() or (target.norm(dim=1) <= eps).any():
        raise ValueError('Cosine matching is undefined for zero/near-zero norm tokens')
    left, right = F.normalize(source, dim=1, eps=eps), F.normalize(target, dim=1, eps=eps)
    result = torch.empty(len(left), dtype=torch.long, device=left.device)
    for start in range(0, len(left), chunk_size):
        rows = left[start:start+chunk_size]
        best = rows.new_full((len(rows),), -torch.inf)
        indices = torch.zeros(len(rows), dtype=torch.long, device=rows.device)
        for offset in range(0, len(right), chunk_size):
            values, local = (rows @ right[offset:offset+chunk_size].T).max(dim=1)
            improved = values > best
            indices = torch.where(improved, local+offset, indices)
            best = torch.maximum(best, values)
        result[start:start+len(rows)] = indices
    return result


def token_correlation_penalty(left, right, target, eps=1e-4, off_weight=.005):
    """Centered BT (target=1) or decorrelation (target=0) on matched tokens.

    A finite zero decorrelation penalty with low variance is marked invalid.
    This diagnostic cannot turn within-patient tokens into independent samples.
    """
    _matrix(left, 'left', minimum=2)
    _matrix(right, 'right', minimum=2)
    _same_space(left, right)
    if left.shape != right.shape:
        raise ValueError('Correlation requires equally sized matched token matrices')
    if target not in (0, 1) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('Expected target 0/1 and finite positive eps')
    if not math.isfinite(off_weight) or off_weight < 0:
        raise ValueError('off_weight must be finite and nonnegative')
    a, b = left-left.mean(0), right-right.mean(0)
    va, vb = a.square().mean(0), b.square().mean(0)
    corr = (a/(va+eps).sqrt()).T @ (b/(vb+eps).sqrt())/len(a)
    diag = corr.diagonal()
    off = corr-torch.diag_embed(diag)
    loss = (diag-target).square().sum()+off_weight*off.square().sum()
    low_a, low_b = va < eps, vb < eps
    low = low_a | low_b
    diagnostic = {'matched_tokens': len(a), 'dimensions': a.shape[1],
                  'low_variance_left': int(low_a.sum().item()),
                  'low_variance_right': int(low_b.sum().item()),
                  'low_variance_coordinates': int(low.sum().item()),
                  'mean_diagonal': float(diag.detach().mean().item()),
                  'valid': not bool(low.any().item())}
    return loss, diagnostic


def _views(values, name):
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f'{name} must contain exactly two view tensors')
    for index, value in enumerate(values):
        _matrix(value, f'{name}[{index}]', minimum=2)
    _same_space(*values)


def _match_info(indices, source_count, target_count):
    return {'source_tokens': source_count, 'target_tokens': target_count,
            'unique_matched_target_tokens': int(indices.unique().numel()),
            'direction': 'source_to_target', 'many_to_one_allowed': True}


def dense_token_objective(path_features, rna_features, path_projected, rna_projected,
                          common_dim, *, match_chunk_size=256, eps=1e-4, off_weight=.005):
    """Return four raw losses and diagnostics for a single patient's two views.

    Inputs are tuples of two [tokens, D] matrices. Each modality's feature and
    projected rows must refer to the same tokens, but view sizes may differ.
    Within-modality matching uses backbone features A->B, followed by a full-Z
    BT penalty. Cross-modality matching uses shared projected coordinates and
    averages all four pathology/RNA view combinations. Cross blocks Csu/Cus
    are not directly penalized, following the shared/specific formulation.
    """
    for values, name in ((path_features, 'path_features'), (rna_features, 'rna_features'),
                         (path_projected, 'path_projected'), (rna_projected, 'rna_projected')):
        _views(values, name)
    dimensions = path_projected[0].shape[1]
    if type(common_dim) is not int or not 0 < common_dim < dimensions:
        raise ValueError('common_dim must split nonempty common and unique coordinates')
    for feature, projected in zip((*path_features, *rna_features),
                                  (*path_projected, *rna_projected)):
        if len(feature) != len(projected):
            raise ValueError('Each projected view must preserve its backbone token rows')
        if feature.device != projected.device or feature.dtype != projected.dtype:
            raise ValueError('Features and projections must share device and dtype')
    for z in (*path_projected, *rna_projected):
        _same_space(path_projected[0], z)
    losses, within = {}, {}
    for name, features, projected in (('path', path_features, path_projected),
                                      ('rna', rna_features, rna_projected)):
        indices = chunked_cosine_nearest_neighbor(*features, chunk_size=match_chunk_size)
        term, diag = token_correlation_penalty(projected[0], projected[1].index_select(0, indices),
                                              1, eps=eps, off_weight=off_weight)
        losses[f'dense_{name}_within'] = term
        within[name] = {**_match_info(indices, len(features[0]), len(features[1])), **diag}
    cross, common, unique = [], [], []
    for pview in range(2):
        for rview in range(2):
            p, r = path_projected[pview], rna_projected[rview]
            indices = chunked_cosine_nearest_neighbor(p[:, :common_dim], r[:, :common_dim],
                                                       chunk_size=match_chunk_size)
            matched = r.index_select(0, indices)
            c, dc = token_correlation_penalty(p[:, :common_dim], matched[:, :common_dim], 1,
                                              eps=eps, off_weight=off_weight)
            u, du = token_correlation_penalty(p[:, common_dim:], matched[:, common_dim:], 0,
                                              eps=eps, off_weight=off_weight)
            common.append(c)
            unique.append(u)
            cross.append({'path_view': pview, 'rna_view': rview,
                          **_match_info(indices, len(p), len(r)), 'common': dc, 'unique': du})
    losses['dense_common'], losses['dense_unique'] = torch.stack(common).mean(), torch.stack(unique).mean()
    diagnostics = {'within': within, 'cross': cross,
                   'valid': all(x['valid'] for x in within.values()) and
                            all(x['common']['valid'] and x['unique']['valid'] for x in cross),
                   'cross_view_pairs': 4, 'common_dimensions': common_dim,
                   'unique_dimensions': dimensions-common_dim,
                   'token_axis_is_independent_patients': False,
                   'limits': ['DenseTwins-inspired WSI/bulk-RNA adaptation, not an official reproduction.',
                              'RNA pathway tokens have no observed spatial correspondence to WSI patches.',
                              'Within-patient token rows are not independent patient observations.',
                              'Low variance is flagged; a finite or zero loss does not establish decoupling.',
                              'No synergy or biological independence is identified by these objectives.']}
    return losses, diagnostics
