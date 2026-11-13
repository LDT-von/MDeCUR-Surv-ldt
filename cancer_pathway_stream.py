"""Full-patch streaming for the pathway/local pathology-RNA research model.

Two explicit pathology view sources are supported: independently sampled
feature noise on a single stored tensor, or verified same-patch UNI image views.
The reference prediction always uses the original, unperturbed features.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from cancer_model import piecewise_nll
from cancer_pathway import PathwayLocalSurvival, pathway_correlation_loss
from cancer_rna import fit_rna_scaler, rna_noise_view, transform_rna
from cancer_stream import _local_from_patients, _merge_moments, _validate_stream_cohort
from model import DegenerateCorrelation, correlation_loss, stability_weighted_common


@dataclass(frozen=True)
class PathwayConfig:
    hidden: int = 16
    common: int = 8
    local_dim: int = 8
    local_common: int = 4
    interaction_strength: float = .2
    match_temperature: float = .2
    epochs: int = 80
    patience: int = 15
    lr: float = .003
    weight_decay: float = .001
    seed: int = 17
    rna_noise: float = .05
    feature_noise: float = .05
    rna_scale_policy: str = 'mask'
    rna_min_std: float = 1e-6
    common_mode: str = 'uniform'
    strength: float = 2.
    min_weight: float = .1
    global_path_within: float = .0005
    global_rna_within: float = .0005
    global_common: float = .001
    global_unique: float = .001
    patch_within: float = .001
    pathway_path_within: float = .0005
    pathway_rna_within: float = .0005
    pathway_common: float = .001
    pathway_unique: float = .001
    matched_common: float = .0005
    matched_unique: float = .0005


LOSS_NAMES = ('global_path_within', 'global_rna_within', 'global_common', 'global_unique',
              'patch_within', 'pathway_path_within', 'pathway_rna_within',
              'pathway_common', 'pathway_unique', 'matched_common', 'matched_unique')


def effective_weights(cfg):
    weights = {name: getattr(cfg, name) for name in LOSS_NAMES}
    if cfg.common_mode == 'off':
        weights['global_common'] = 0.
    return weights


def minimum_batch_size(cfg):
    """Necessary rank bound for active identity-correlation targets.

    This is a necessary condition only, not a claim of stable estimation.
    """
    w = effective_weights(cfg)
    required = 2 if any(w.values()) else 1
    for enabled, dim in ((w['global_path_within'] or w['global_rna_within'], cfg.hidden),
                         (w['global_common'], cfg.common),
                         (w['pathway_path_within'] or w['pathway_rna_within'], cfg.local_dim),
                         (w['pathway_common'] or w['matched_common'], cfg.local_common)):
        if enabled:
            required = max(required, dim+1)
    return required


def validate_config(cfg, view_source, patient_batch_size, chunk_size):
    if view_source not in ('feature_noise', 'image_d4'):
        raise ValueError('Choose explicit feature_noise or image_d4 pathology view source')
    if cfg.common_mode not in ('off', 'uniform', 'stability', 'shuffled'):
        raise ValueError('Invalid global common mode')
    all_weights = [getattr(cfg, name) for name in LOSS_NAMES]
    if any(not math.isfinite(x) or x < 0 for x in all_weights):
        raise ValueError('All auxiliary weights must be finite and nonnegative')
    if (type(cfg.hidden) is not int or cfg.hidden < 2 or type(cfg.common) is not int
            or not 0 < cfg.common < cfg.hidden or type(cfg.local_dim) is not int
            or type(cfg.local_common) is not int or not 0 < cfg.local_common < cfg.local_dim
            or type(cfg.epochs) is not int or cfg.epochs < 1 or type(cfg.patience) is not int or cfg.patience < 1
            or type(patient_batch_size) is not int or type(chunk_size) is not int or chunk_size < 1
            or not math.isfinite(cfg.lr) or cfg.lr <= 0 or not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0
            or not math.isfinite(cfg.rna_noise) or cfg.rna_noise < 0
            or not math.isfinite(cfg.feature_noise) or cfg.feature_noise < 0
            or not math.isfinite(cfg.interaction_strength) or cfg.interaction_strength < 0
            or not math.isfinite(cfg.match_temperature) or cfg.match_temperature <= 0
            or not math.isfinite(cfg.rna_min_std) or cfg.rna_min_std <= 0
            or cfg.rna_scale_policy not in ('mask', 'clamp', 'none')
            or not math.isfinite(cfg.strength) or cfg.strength < 0
            or not math.isfinite(cfg.min_weight) or not 0 < cfg.min_weight <= 1):
        raise ValueError('Invalid pathway training configuration')
    required = minimum_batch_size(cfg)
    if patient_batch_size < required:
        raise ValueError(f'Active identity correlations require patient_batch_size >= {required}; '
                         'the bound is necessary, not sufficient for stable estimation')
    if view_source == 'feature_noise' and any(effective_weights(cfg).values()) and cfg.feature_noise <= 0:
        raise ValueError('feature_noise views require a positive noise level when auxiliary losses are active')
    return required


def patient_batches(order, size, minimum):
    if len(order) < minimum:
        raise ValueError(f'The training split needs at least {minimum} patients for active correlation terms')
    batches = list(order.split(size))
    if len(batches[-1]) < minimum:
        if len(batches) == 1:
            raise ValueError('Not enough patients for active correlation terms')
        batches[-2] = torch.cat((batches[-2], batches[-1]))
        batches.pop()
    return batches


def feature_noise_pair(reference, patient_id, block, seed, sigma):
    """Stateless per-block perturbations of encoded patch features.

    Noise is scaled by the per-patch feature RMS; it is NOT an image transform.
    Seeds depend on patient, slide basename, row bounds, epoch/batch seed and
    view. Identical config/replay yields identical values. A changed chunk size
    changes the sampled perturbations and is therefore recorded as an input.
    """
    if reference.ndim != 2 or not reference.is_floating_point():
        raise ValueError('Feature-noise generation requires floating encoded [N,D] patch features')
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError('Feature-noise standard deviation must be finite and nonnegative')
    rms = reference.detach().float().square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
    views = []
    for view in ('A', 'B'):
        key = json.dumps([int(seed), str(patient_id), Path(block[0]).name,
                          int(block[1]), int(block[2]), view], ensure_ascii=False).encode('utf-8')
        noise_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**63-1)
        generator = torch.Generator(device=reference.device).manual_seed(noise_seed)
        noise = torch.randn(reference.shape, device=reference.device, dtype=torch.float32, generator=generator)
        views.append(reference+sigma*rms.to(reference.dtype)*noise.to(reference.dtype))
    return tuple(views)


def _pool_summary(features, scores):
    if scores.ndim != 2 or len(scores) != len(features) or not len(features):
        raise ValueError('Expected nonempty patch features and score columns')
    weights = scores.softmax(0)
    # Extra moments are diagnostics only; no alternative loss depends on them.
    expected_score = (weights.detach()*scores.detach()).sum(0)
    max_score = scores.detach().max(0).values
    return scores.logsumexp(0), weights.T @ features, expected_score, max_score


def _merge_pool(left, right):
    lse = torch.logaddexp(left[0], right[0])
    mean = (left[0]-lse).exp()[:, None]*left[1]+(right[0]-lse).exp()[:, None]*right[1]
    expected = (left[0].detach()-lse.detach()).exp()*left[2]+(right[0].detach()-lse.detach()).exp()*right[2]
    return lse, mean, expected, torch.maximum(left[3], right[3])


def _feature_scores(model, features, rna_queries=None):
    global_score = model.base.attention(features)
    pathology_queries = features @ model.pathway_queries.T/math.sqrt(model.hidden)
    if rna_queries is None:
        return torch.cat((global_score, pathology_queries), dim=1)
    keys = F.normalize(model.match_path(features), dim=-1, eps=1e-6)
    conditional = (keys @ rna_queries.T)/model.match_temperature
    return torch.cat((global_score, pathology_queries, conditional), dim=1)


def _chunk_summary(model, bag, block, view_source, include_views, include_patch,
                   feature_noise, view_seed, rna_queries):
    parameter = next(model.parameters())
    if view_source == 'image_d4' and include_views:
        values = bag.read_block(block, ('features_reference', 'features_a', 'features_b'))
        embedded = [model.base.patch_encoder(x.to(device=parameter.device, dtype=parameter.dtype)) for x in values]
    else:
        reference, = bag.read_block(block, ('features_reference',))
        h0 = model.base.patch_encoder(reference.to(device=parameter.device, dtype=parameter.dtype))
        if include_views and view_source == 'feature_noise':
            embedded = (h0, *feature_noise_pair(h0, bag.patient_id, block, view_seed, feature_noise))
        else:
            embedded = (h0,)
    outputs = list(_pool_summary(embedded[0], _feature_scores(model, embedded[0], rna_queries[0])))
    if include_views:
        for index, features in enumerate(embedded[1:], 1):
            outputs.extend(_pool_summary(features, _feature_scores(model, features, rna_queries[index])))
    if include_patch:
        left, right = (model.base.local_projector(x).double() for x in embedded[1:])
        ma, mb = left.mean(0), right.mean(0)
        ac, bc = left-ma, right-mb
        outputs.extend((ma, mb, ac.square().mean(0), bc.square().mean(0), ac.T@bc/len(left)))
    return tuple(outputs)


def _stream_patient(model, bag, rna_queries, chunk_size, view_source,
                    include_views, include_patch, feature_noise, view_seed, replay):
    pools = None
    moments = None
    seen = 0
    for block in bag.blocks(chunk_size):
        fn = partial(_chunk_summary, model, bag, block, view_source, include_views,
                     include_patch, feature_noise, view_seed)
        values = checkpoint(fn, rna_queries, use_reentrant=False, preserve_rng_state=True) \
                 if replay and torch.is_grad_enabled() else fn(rna_queries)
        count = block[2]-block[1]
        if count < 1:
            raise ValueError('Patch chunks must be nonempty')
        nviews = 3 if include_views else 1
        current = [tuple(values[4*i:4*(i+1)]) for i in range(nviews)]
        pools = current if pools is None else [_merge_pool(a, b) for a, b in zip(pools, current)]
        if include_patch:
            chunk_moments = values[4*nviews:]
            moments = chunk_moments if moments is None else _merge_moments(moments, chunk_moments, count/(seen+count))
        seen += count
    if pools is None or seen != bag.n_patches:
        raise ValueError('Patch chunks must cover every patient patch exactly once')
    reference = pools[0]
    entropy = (reference[0].detach()-reference[2]).clamp_min(0)
    max_weight = (reference[3]-reference[0].detach()).exp().clamp(max=1)
    return [x[1] for x in pools], moments, entropy, max_weight


def pathway_stream_forward(model, bags, rna, chunk_size=2048, view_source='feature_noise',
                           include_views=False, include_patch=False, rna_views=None,
                           feature_noise=.05, view_seed=0, checkpoint_chunks=True):
    if (type(chunk_size) is not int or chunk_size < 1 or not bags or rna.ndim != 2
            or len(bags) != len(rna) or view_source not in ('feature_noise', 'image_d4')):
        raise ValueError('Expected valid view source, positive chunk size, and one RNA row per patient')
    expected_source = 'single_view' if view_source == 'feature_noise' else 'cached_uni'
    if any(getattr(bag, 'view_source', expected_source) != expected_source for bag in bags):
        raise ValueError('View source does not match the feature cache')
    if include_patch and not include_views:
        raise ValueError('Patch correlation requires two explicit pathology views')
    if include_views and (rna_views is None or len(rna_views) != 2 or any(v.shape != rna.shape for v in rna_views)):
        raise ValueError('Two matching RNA views are required when auxiliaries are active')
    parameter = next(model.parameters())
    rna = rna.to(device=parameter.device, dtype=parameter.dtype)
    tokens = model.pathways(rna)
    tokens_by_view = [tokens]
    if include_views:
        g1, g2 = [view.to(device=parameter.device, dtype=parameter.dtype) for view in rna_views]
        token_r1, token_r2 = model.pathways(g1), model.pathways(g2)
        tokens_by_view.extend((token_r1, token_r2))
    queries = torch.stack([model.match_queries(view) for view in tokens_by_view], dim=1)
    pooled, moments, entropies, max_weights = [], [], [], []
    for i, bag in enumerate(bags):
        summary, local, entropy, max_weight = _stream_patient(model, bag, queries[i], chunk_size, view_source,
                                                               include_views, include_patch, feature_noise, view_seed,
                                                               checkpoint_chunks)
        pooled.append(summary)
        entropies.append(entropy)
        max_weights.append(max_weight)
        if include_patch:
            moments.append(local)
    k = model.pathway_count
    global_path = torch.stack([item[0][0] for item in pooled])
    matched_path = torch.stack([item[0][1+k:1+2*k] for item in pooled])
    global_rna = model.base.rna_encoder(rna)
    logits, interaction = model.prediction(global_path, global_rna, matched_path, tokens)
    output = {'log_rates': logits, 'interaction_logits': interaction,
              'path_hidden': global_path, 'rna_hidden': global_rna,
              'matched_path': matched_path, 'rna_pathways': tokens}
    match_entropy = torch.stack([h[1+k:1+2*k] for h in entropies])
    normalizers = match_entropy.new_tensor([math.log(b.n_patches) if b.n_patches > 1 else 1. for b in bags])[:, None]
    output['matching_entropy'] = match_entropy
    output['matching_normalized_entropy'] = (match_entropy/normalizers).clamp(0, 1)
    output['matching_max_weight'] = torch.stack([m[1+k:1+2*k] for m in max_weights])
    if include_views:
        global_p1 = torch.stack([item[1][0] for item in pooled])
        global_p2 = torch.stack([item[2][0] for item in pooled])
        region_p1 = torch.stack([item[1][1:1+k] for item in pooled])
        region_p2 = torch.stack([item[2][1:1+k] for item in pooled])
        matched_p1 = torch.stack([item[1][1+k:1+2*k] for item in pooled])
        matched_p2 = torch.stack([item[2][1+k:1+2*k] for item in pooled])
        local_p1, local_r1 = model.local_projection(region_p1, token_r1)
        local_p2, local_r2 = model.local_projection(region_p2, token_r2)
        match_p1, match_r1 = model.local_projection(matched_p1, token_r1)
        match_p2, match_r2 = model.local_projection(matched_p2, token_r2)
        output['auxiliary'] = {
            'global_p1': model.base.path_projector(global_p1),
            'global_p2': model.base.path_projector(global_p2),
            'global_r1': model.base.rna_projector(model.base.rna_encoder(g1)),
            'global_r2': model.base.rna_projector(model.base.rna_encoder(g2)),
            'local_p1': local_p1, 'local_p2': local_p2,
            'local_r1': local_r1, 'local_r2': local_r2,
            'matched_p1': match_p1, 'matched_p2': match_p2,
            'matched_r1': match_r1, 'matched_r2': match_r2,
            'pathology_regions1': region_p1, 'pathology_regions2': region_p2,
        }
    if include_patch:
        output['patch_loss'], output['patch_diagnostics'] = _local_from_patients(moments, bags)
    return output


def pathway_objective(model, batch, starts, cfg, chunk_size, view_source, view_seed, *,
                      return_components=False, survival_loss=None):
    """Compute the pathway objective.

    ``survival_loss`` overrides only the survival likelihood term; it receives
    the raw model output and the batch and must return a scalar.  The default
    is the continuous-time piecewise likelihood.  The auxiliary terms, their
    weights and their diagnostics are identical in both cases.
    """
    weights = effective_weights(cfg)
    use_views = any(weights.values())
    rna = batch['rna']
    rna_views = tuple(rna_noise_view(rna, cfg.rna_noise, batch.get('rna_mask')) for _ in range(2)) if use_views else None
    output = pathway_stream_forward(model, batch['bags'], rna, chunk_size, view_source,
                                    include_views=use_views, include_patch=bool(weights['patch_within']),
                                    rna_views=rna_views, feature_noise=cfg.feature_noise,
                                    view_seed=view_seed)
    loss = (piecewise_nll(output['log_rates'], batch['time'], batch['event'], starts)
            if survival_loss is None else survival_loss(output['log_rates'], batch))
    components = {'survival': loss}
    stats = {'patients': len(rna), 'train_nll': loss.detach().item(),
             'common_skipped': False, 'common_skip_reason': None,
             'terms': {name: None for name in LOSS_NAMES},
             'pathway_diagnostics': {}, 'patch_diagnostics': None,
             'interaction_abs_mean': output['interaction_logits'].detach().abs().mean().item(),
             'interaction_contribution_abs_mean': (cfg.interaction_strength*output['interaction_logits']).detach().abs().mean().item(),
             'matching_normalized_entropy_mean': output['matching_normalized_entropy'].mean().item(),
             'matching_max_weight_mean': output['matching_max_weight'].mean().item()}
    if use_views:
        aux = output['auxiliary']
        for name, x, y, target in (
                ('global_path_within', aux['global_p1'], aux['global_p2'], 1),
                ('global_rna_within', aux['global_r1'], aux['global_r2'], 1),
                ('global_unique', aux['global_p1'][:, cfg.common:], aux['global_r1'][:, cfg.common:], 0)):
            if weights[name]:
                term = correlation_loss(x, y, target)
                loss = loss+weights[name]*term
                stats['terms'][name] = term.detach().item()
                components[name] = term
        if weights['global_common']:
            try:
                if cfg.common_mode == 'uniform':
                    p, r = aux['global_p1'][:, :cfg.common], aux['global_r1'][:, :cfg.common]
                    term = correlation_loss(p, r, 1)
                    stats['global_common_low_variance_dimensions'] = int(
                        ((p.var(0, unbiased=False) < 1e-4) | (r.var(0, unbiased=False) < 1e-4)).sum().item())
                    stats['global_common_ess'] = float(len(rna))
                else:
                    # Shuffling the weights must not alter subsequent RNA views.
                    shuffle_generator = None
                    if cfg.common_mode == 'shuffled':
                        shuffle_generator = torch.Generator(device=aux['global_p1'].device)
                        shuffle_generator.manual_seed((int(view_seed)+0x5A17ED) % (2**63-1))
                    term, diagnostic = stability_weighted_common(
                        aux['global_p1'][:, :cfg.common], aux['global_p2'][:, :cfg.common],
                        aux['global_r1'][:, :cfg.common], aux['global_r2'][:, :cfg.common],
                        strength=cfg.strength, min_weight=cfg.min_weight,
                        shuffle_weights=cfg.common_mode == 'shuffled',
                        shuffle_generator=shuffle_generator)
                    stats['global_common_ess'] = diagnostic['ess'].item()
                loss = loss+weights['global_common']*term
                stats['terms']['global_common'] = term.detach().item()
                components['global_common'] = term
            except DegenerateCorrelation as error:
                stats['common_skipped'] = True
                stats['common_skip_reason'] = str(error)
        if weights['patch_within']:
            loss = loss+weights['patch_within']*output['patch_loss']
            stats['terms']['patch_within'] = output['patch_loss'].detach().item()
            components['patch_within'] = output['patch_loss']
            stats['patch_diagnostics'] = output['patch_diagnostics']
        for name, x, y, target in (
                ('pathway_path_within', aux['local_p1'], aux['local_p2'], 1),
                ('pathway_rna_within', aux['local_r1'], aux['local_r2'], 1),
                ('pathway_common', aux['local_p1'][..., :cfg.local_common], aux['local_r1'][..., :cfg.local_common], 1),
                ('pathway_unique', aux['local_p1'][..., cfg.local_common:], aux['local_r1'][..., cfg.local_common:], 0)):
            if weights[name]:
                term, diagnostic = pathway_correlation_loss(x, y, target)
                loss = loss+weights[name]*term
                stats['terms'][name] = term.detach().item()
                components[name] = term
                stats['pathway_diagnostics'][name] = diagnostic
        for name, target, selection in (
                ('matched_common', 1, slice(None, cfg.local_common)),
                ('matched_unique', 0, slice(cfg.local_common, None))):
            if weights[name]:
                a, diag_a = pathway_correlation_loss(aux['matched_p1'][..., selection],
                                                     aux['matched_r1'][..., selection], target)
                b, diag_b = pathway_correlation_loss(aux['matched_p2'][..., selection],
                                                     aux['matched_r2'][..., selection], target)
                term = (a+b)/2
                loss = loss+weights[name]*term
                stats['terms'][name] = term.detach().item()
                components[name] = term
                stats['pathway_diagnostics'][name] = {'view_a': diag_a, 'view_b': diag_b}
    if not torch.isfinite(loss):
        raise FloatingPointError('Pathway model loss is not finite')
    stats['total_loss'] = loss.detach().item()
    if return_components:
        return loss, stats, components
    return loss, stats


def _gradient_norm(module):
    values = [p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
    return math.sqrt(sum(v.item() for v in values)) if values else 0.


def train_pathway_model(train, validation, starts, cfg, pathway_mapping, patient_batch_size=20,
                        patch_chunk_size=2048, view_source='feature_noise', device='cpu'):
    minimum = validate_config(cfg, view_source, patient_batch_size, patch_chunk_size)
    _validate_stream_cohort(train)
    _validate_stream_cohort(validation)
    if set(train['ids']) & set(validation['ids']):
        raise ValueError('Patient overlap between train and validation')
    if pathway_mapping['rna_dim'] != train['rna'].shape[1] or train['rna'].shape[1] != validation['rna'].shape[1]:
        raise ValueError('Ordered gene list and cohort RNA dimensions differ')
    dimension = train['bags'][0].feature_dim
    expected_source = 'single_view' if view_source == 'feature_noise' else 'cached_uni'
    if any(b.feature_dim != dimension or b.n_patches < 1 or getattr(b, 'view_source', expected_source) != expected_source
           for b in train['bags']+validation['bags']):
        raise ValueError('Incompatible feature dimensions, empty bags, or wrong view source')
    for data in (train, validation):
        piecewise_nll(torch.zeros(len(data['ids']), len(starts)), data['time'], data['event'], starts)
    patient_batches(torch.arange(len(train['ids'])), patient_batch_size, minimum)
    scaler = fit_rna_scaler(train['rna'], cfg.rna_scale_policy, cfg.rna_min_std)
    normalized = transform_rna(train['rna'], scaler)
    scaler['rna_preprocessing']['validation_max_abs'] = transform_rna(validation['rna'], scaler).abs().max().item()
    for item in pathway_mapping['pathways']:
        item['active_training_genes'] = int(scaler['rna_mask'][item['indices']].sum().item())
        if item['active_training_genes'] == 0:
            raise ValueError(f"Pathway {item['id']} contains no active training RNA genes")
    model_kwargs = dict(path_dim=dimension, rna_dim=train['rna'].shape[1],
                        pathway_indices=[item['indices'] for item in pathway_mapping['pathways']],
                        hidden=cfg.hidden, common=cfg.common, local_dim=cfg.local_dim,
                        local_common=cfg.local_common, bins=len(starts),
                        interaction_strength=cfg.interaction_strength, match_temperature=cfg.match_temperature)
    torch.manual_seed(cfg.seed)
    model = PathwayLocalSurvival(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    fitted = dict(model=model, **scaler, starts=starts.clone(), model_kwargs=model_kwargs,
                  pathway_mapping=deepcopy(pathway_mapping), effective_loss_weights=effective_weights(cfg),
                  stream_config={'patient_batch_size': patient_batch_size, 'patch_chunk_size': patch_chunk_size,
                                 'view_source': view_source, 'device': str(device),
                                 'feature_noise': cfg.feature_noise if view_source == 'feature_noise' else None,
                                 'rna_noise': cfg.rna_noise, 'minimum_patient_batch': minimum,
                                 'auxiliary_rng_scheme': 'rna_global__weight_shuffle_isolated_v1',
                                 'correlation_scope': 'per_pathway_across_patient_minibatch',
                                 'checkpoint_chunks': True})
    generator = torch.Generator().manual_seed(cfg.seed+1009)
    history, best, best_state, best_epoch, stale = [], math.inf, None, 0, 0
    for epoch in range(1, cfg.epochs+1):
        epoch_start = time.perf_counter()
        model.train()
        records = []
        order = torch.randperm(len(train['ids']), generator=generator)
        for batch_number, indices in enumerate(patient_batches(order, patient_batch_size, minimum)):
            optimizer.zero_grad(set_to_none=True)
            batch = {'bags': [train['bags'][i] for i in indices.tolist()],
                     'rna': normalized[indices].to(device), 'time': train['time'][indices].to(device),
                     'event': train['event'][indices].to(device), 'rna_mask': scaler['rna_mask']}
            # This seed is explicitly recorded through config, epoch and batch order.
            view_seed = cfg.seed*1_000_003+epoch*100_003+batch_number
            loss, stats = pathway_objective(model, batch, starts.to(device), cfg, patch_chunk_size,
                                            view_source, view_seed)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Pathway model parameter gradient is not finite')
            stats['gradient_norms'] = {
                'patch_encoder': _gradient_norm(model.base.patch_encoder),
                'rna_encoder': _gradient_norm(model.base.rna_encoder),
                'global_path_projector': _gradient_norm(model.base.path_projector),
                'global_rna_projector': _gradient_norm(model.base.rna_projector),
                'patch_local_projector': _gradient_norm(model.base.local_projector),
                'pathway_tokenizer': _gradient_norm(model.pathways),
                'local_path_projector': _gradient_norm(model.local_path_projector),
                'local_rna_projector': _gradient_norm(model.local_rna_projector),
                'matching_path': _gradient_norm(model.match_path),
                'matching_rna': _gradient_norm(model.match_rna),
                'interaction_head': _gradient_norm(model.interaction_head),
                'pathway_queries': 0. if model.pathway_queries.grad is None else
                                   model.pathway_queries.grad.detach().double().norm().item(),
            }
            optimizer.step()
            records.append(stats)
        predicted = predict_pathway_model(fitted, validation)['log_rates']
        val = piecewise_nll(predicted, validation['time'], validation['event'], starts).item()
        if not math.isfinite(val):
            raise FloatingPointError('Pathway validation NLL is not finite')
        n = len(train['ids'])
        history.append({'epoch': epoch, 'validation_nll': val,
                        'train_nll': sum(r['patients']*r['train_nll'] for r in records)/n,
                        'total_loss': sum(r['patients']*r['total_loss'] for r in records)/n,
                        'batch_patient_counts': [r['patients'] for r in records], 'batches': records,
                        'epoch_seconds': time.perf_counter()-epoch_start})
        if val < best:
            best, best_state, best_epoch, stale = val, deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        print(json.dumps({'epoch': epoch, 'train_nll': history[-1]['train_nll'],
                          'total_loss': history[-1]['total_loss'], 'validation_nll': val,
                          'best_epoch': best_epoch, 'best_validation_nll': best,
                          'seconds': history[-1]['epoch_seconds'],
                          'batch_patient_counts': history[-1]['batch_patient_counts'],
                          'global_common_skipped_batches': sum(bool(r['common_skipped']) for r in records)},
                         ensure_ascii=False, allow_nan=False), flush=True)
        if stale >= cfg.patience:
            break
    model.load_state_dict(best_state, strict=True)
    model.eval()
    fitted.update(best_epoch=best_epoch, validation_nll=best, history=history, config=asdict(cfg))
    return fitted


def predict_pathway_model(fitted, cohort):
    """Reference-only predictions with a saved train-fitted RNA transform."""
    model = fitted['model']
    model.eval()
    if not cohort['bags'] or len(cohort['bags']) != len(cohort['rna']):
        raise ValueError('One RNA row per patient bag is required')
    rna = transform_rna(cohort['rna'], fitted)
    outputs = []
    with torch.no_grad():
        for i, bag in enumerate(cohort['bags']):
            output = pathway_stream_forward(model, [bag], rna[i:i+1],
                                            fitted['stream_config']['patch_chunk_size'],
                                            fitted['stream_config']['view_source'],
                                            checkpoint_chunks=False)
            outputs.append({name: output[name].cpu() for name in
                            ('log_rates', 'interaction_logits', 'matching_normalized_entropy', 'matching_max_weight')})
    return {name: torch.cat([item[name] for item in outputs]) for name in outputs[0]}
