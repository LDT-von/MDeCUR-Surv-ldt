"""Full auxiliary pathway model with dense-token constraints and task interaction.

DenseTwins-inspired patch/pathway matching is an adaptation to bulk RNA,
not a spatial-transcriptomic correspondence or an information-theoretic PID.
The legacy prediction route is retained; local [common, private] coordinates
also enter prediction directly, rather than being auxiliary-only projectors.
"""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn

from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import (PathwayConfig, LOSS_NAMES, effective_weights,
                                   feature_noise_pair, pathway_objective)
from cancer_rna import rna_noise_view


DENSE_LOSS_NAMES = ('dense_path_within', 'dense_rna_within', 'dense_common', 'dense_unique')


@dataclass(frozen=True)
class DensePathwayConfig(PathwayConfig):
    common_mode: str = 'stability'
    dense_path_within: float = .0005
    dense_rna_within: float = .0005
    dense_common: float = .001
    dense_unique: float = .001
    dense_max_tokens: int = 256
    dense_match_chunk_size: int = 128
    dense_warmup_epochs: int = 5


def dense_weights(cfg, epoch=1):
    if type(epoch) is not int or epoch < 1 or cfg.dense_warmup_epochs < 1:
        raise ValueError('Positive epoch and dense warmup are required')
    weights = {name: getattr(cfg, name) for name in DENSE_LOSS_NAMES}
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Dense coefficients must be finite and nonnegative')
    ramp = min(1., epoch/cfg.dense_warmup_epochs)
    for name in ('dense_common', 'dense_unique'):
        weights[name] *= ramp
    return weights


def disable_all_auxiliary(cfg):
    return DensePathwayConfig(**{**asdict(cfg), **{name: 0. for name in (*LOSS_NAMES, *DENSE_LOSS_NAMES)},
                                  'common_mode': 'off'})


def require_full_auxiliary(cfg):
    if any(w <= 0 for w in (*effective_weights(cfg).values(), *dense_weights(cfg).values())):
        raise ValueError('The full model requires every legacy and dense auxiliary coefficient > 0')
    if cfg.common_mode != 'stability':
        raise ValueError('Full reference includes stability-weighted global common objective')
    if cfg.dense_max_tokens < max(cfg.local_dim+1, 2) or cfg.dense_match_chunk_size < 1:
        raise ValueError('Dense token budget must support the local within-correlation dimension')


class CenteredTokenInteraction(nn.Module):
    """Train-reference-centered bilinear token interaction; not a synergy estimator.

    Buffers update only on training forwards, and remain frozen for validation.
    In the full model the pathology argument is RNA-conditioned, so even this
    centered product must not be described as an identified pure interaction.
    """
    def __init__(self, dimension, hidden, bins, momentum=.05):
        super().__init__()
        self.path = nn.Linear(dimension, hidden, bias=False)
        self.rna = nn.Linear(dimension, hidden, bias=False)
        self.head = nn.Linear(hidden, bins, bias=False)
        self.momentum = momentum
        self.register_buffer('path_reference', torch.zeros(dimension))
        self.register_buffer('rna_reference', torch.zeros(dimension))
        self.register_buffer('training_reference_updates', torch.tensor(0, dtype=torch.long))

    def forward(self, path, rna):
        if path.ndim != 3 or path.shape != rna.shape:
            raise ValueError('Matching [patients,tokens,private_dimension] inputs required')
        if self.training:
            with torch.no_grad():
                self.path_reference.lerp_(path.detach().mean((0, 1)), self.momentum)
                self.rna_reference.lerp_(rna.detach().mean((0, 1)), self.momentum)
                self.training_reference_updates.add_(1)
        # clone protects saved backward values from later buffer mutation.
        p = self.path(path-self.path_reference.detach().clone())
        r = self.rna(rna-self.rna_reference.detach().clone())
        return self.head((p*r).mean(1))


class DensePathwaySurvival(PathwayLocalSurvival):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bins = self.interaction_head.out_features
        self.decomposition_head = nn.Linear(2*self.local_dim-self.local_common, bins, bias=False)
        self.token_cooperation = CenteredTokenInteraction(self.local_dim-self.local_common, self.hidden, bins)

    def prediction(self, pooled_path, global_rna, matched_pathways, rna_tokens):
        legacy, interaction = super().prediction(pooled_path, global_rna, matched_pathways, rna_tokens)
        p, r = self.local_projection(matched_pathways, rna_tokens)
        c = self.local_common
        common = (p[..., :c]+r[..., :c])/2
        direct = self.decomposition_head(torch.cat((common, p[..., c:], r[..., c:]), dim=-1).mean(1))
        cooperative = self.token_cooperation(p[..., c:], r[..., c:])
        return legacy+direct+self.interaction_strength*cooperative, interaction+cooperative

    def adaptation_metadata(self):
        return {'label': 'MDeCUR-full-dense-token-v2',
                'prediction': ['legacy global and RNA-conditioned pathway interaction',
                               'local common/private coordinates directly enter hazard logits',
                               'train-reference-centered private-token bilinear cooperation'],
                'dense_matching': 'independently encoded patch/pathway tokens, shared-space nearest neighbor',
                'dense_max_tokens_scope': 'sampled valid pathology tokens per patient; all RNA pathway tokens',
                'synergy_estimator': False, 'spatial_rna_ground_truth': False,
                'limits': ['DenseTwins-inspired WSI/bulk-RNA adaptation, not author-code replication.',
                           'Nearest-neighbor selection can induce correlation under independent modalities.',
                           'Conditional prediction tokens are not independent common-information evidence.',
                           'Patient-level generalization and interaction gains require matched controls.']}


def valid_dense_rows(bag, maximum):
    """Deterministic subset of real rows; parity padding is never a dense token."""
    valid = bag.n_patches-int(getattr(bag, 'padding', 0))
    if valid < 2 or type(maximum) is not int or maximum < 2:
        raise ValueError('Dense objectives need at least two real pathology tokens')
    indices = torch.linspace(0, valid-1, min(maximum, valid)).round().long()
    pieces, cursor = [], 0
    for block in bag.blocks(4096):
        length = block[2]-block[1]
        selected = indices[(indices >= cursor) & (indices < cursor+length)]-cursor
        if len(selected):
            rows, = bag.read_block(block, ('features_reference',))
            pieces.append(rows.index_select(0, selected))
        cursor += length
    result = torch.cat(pieces)
    if len(result) != len(indices):
        raise ValueError('Dense token selection failed to cover requested valid rows')
    return result, {'available_real_tokens': valid, 'selected_tokens': len(result),
                    'excluded_padding': int(getattr(bag, 'padding', 0))}


def dense_pathway_objective(model, batch, cfg, chunk_size, view_seed, epoch=1, *,
                            survival_loss=None, return_components=False):
    """All legacy objectives plus patient-equally-weighted dense token terms."""
    if survival_loss is None:
        raise ValueError('Dense objective requires an explicit survival_loss callback; no implicit time grid')
    loss, stats, components = pathway_objective(model, batch, None, cfg, chunk_size,
                                               'feature_noise', view_seed,
                                               survival_loss=survival_loss, return_components=True)
    weights = dense_weights(cfg, epoch)
    stats['dense_effective_weights'] = weights
    stats['dense_diagnostics'] = []
    stats['auxiliary_enabled'] = {name: value > 0 for name, value in
                                  {**effective_weights(cfg), **weights}.items()}
    if any(weights.values()):
        from cancer_dense_tokens import dense_token_objective
        rv = tuple(rna_noise_view(batch['rna'], cfg.rna_noise, batch.get('rna_mask')) for _ in range(2))
        rfeatures = tuple(model.pathways(g) for g in rv)
        rprojected = tuple(model.local_rna_projector(g) for g in rfeatures)
        terms = {name: [] for name in DENSE_LOSS_NAMES}
        for i, bag in enumerate(batch['bags']):
            raw, selection = valid_dense_rows(bag, cfg.dense_max_tokens)
            h = model.base.patch_encoder(raw.to(device=batch['rna'].device, dtype=batch['rna'].dtype))
            pv = feature_noise_pair(h, bag.patient_id, ('<dense-valid-tokens>', 0, len(h)), view_seed,
                                    cfg.feature_noise)
            pp = tuple(model.local_path_projector(x) for x in pv)
            values, info = dense_token_objective(
                path_features=pv, rna_features=tuple(x[i] for x in rfeatures),
                path_projected=pp, rna_projected=tuple(x[i] for x in rprojected),
                common_dim=cfg.local_common, match_chunk_size=cfg.dense_match_chunk_size)
            stats['dense_diagnostics'].append({**selection, **info})
            for name in DENSE_LOSS_NAMES:
                terms[name].append(values[name])
        for name in DENSE_LOSS_NAMES:
            if weights[name]:
                term = torch.stack(terms[name]).mean()
                components[name] = term
                loss = loss+weights[name]*term
                stats['terms'][name] = float(term.detach())
    diagnostics = stats['dense_diagnostics']
    stats['dense_estimability'] = {
        'patients': len(diagnostics),
        'all_directions_estimable_patients': sum(bool(info['valid']) for info in diagnostics),
        'nonestimable_patients': sum(not info['valid'] for info in diagnostics),
        'caution': 'Enabled coefficients do not guarantee token variance or identified common/private semantics.'}
    stats['total_loss'] = float(loss.detach())
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite dense-full objective')
    return (loss, stats, components) if return_components else (loss, stats)
