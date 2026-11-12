"""Per-loss gradients from ONE production forward; diagnostic, not an optimizer.

Cosines describe local Euclidean geometry on a specified parameter space.
They do not establish validation benefit or the direction of an AdamW update.
"""
import math

import torch

from cancer_pathway_stream import LOSS_NAMES, effective_weights, pathway_objective, validate_config


GROUPS = {'patch_encoder': 'base.patch_encoder', 'patient_attention': 'base.attention',
          'rna_encoder': 'base.rna_encoder', 'survival_head': 'base.head',
          'global_path_projector': 'base.path_projector', 'global_rna_projector': 'base.rna_projector',
          'patch_local_projector': 'base.local_projector', 'pathway_tokenizer': 'pathways',
          'pathology_anchor_queries': 'pathway_queries', 'matching_path': 'match_path',
          'matching_rna': 'match_rna', 'local_path_projector': 'local_path_projector',
          'local_rna_projector': 'local_rna_projector', 'interaction_path': 'interaction_path',
          'interaction_rna': 'interaction_rna', 'interaction_head': 'interaction_head'}


def norm(grads, indices=None):
    indices = range(len(grads)) if indices is None else indices
    return math.sqrt(sum(float(grads[i].double().square().sum()) for i in indices if grads[i] is not None))


def relation(survival, auxiliary, weight=1., indices=None, norm_floor=1e-12, cosine_tolerance=1e-8):
    """Compare on parameters connected to the survival graph, keeping None distinct.

    A connected zero tensor is NOT a missing graph dependency. Coordinates of
    a survival-connected tensor with zero current derivative stay in the space.
    """
    if len(survival) != len(auxiliary) or not math.isfinite(weight) or weight < 0:
        raise ValueError('Paired gradient lists and finite nonnegative weight required')
    indices = list(range(len(survival))) if indices is None else list(indices)
    support = [i for i in indices if survival[i] is not None]
    overlap = [i for i in support if auxiliary[i] is not None]
    gs, ga = norm(survival, support), norm(auxiliary, support)
    dot = sum(float((survival[i].double()*auxiliary[i].double()).sum()) for i in overlap)
    if not support:
        status = 'no_survival_dependency'
    elif not overlap:
        status = 'no_auxiliary_dependency_in_survival_space'
    elif gs <= norm_floor:
        status = 'zero_survival_gradient'
    elif ga <= norm_floor:
        status = 'zero_auxiliary_gradient'
    else:
        status = 'defined'
    cosine = max(-1., min(1., dot/(gs*ga))) if status == 'defined' else None
    direction = None if cosine is None else ('aligned' if cosine > cosine_tolerance else
                                            'conflicting' if cosine < -cosine_tolerance else 'orthogonal')
    return {'status': status, 'direction': direction, 'cosine': cosine,
            'survival_norm': gs, 'raw_auxiliary_norm_in_survival_space': ga,
            'weighted_auxiliary_norm_in_survival_space': weight*ga,
            'weighted_norm_ratio_to_survival': weight*ga/gs if gs > norm_floor else None,
            'raw_dot': dot, 'weighted_dot': weight*dot,
            'first_order_survival_change_per_unit_sgd_lr': -weight*dot,
            'all_parameter_raw_auxiliary_norm': norm(auxiliary, indices),
            'outside_survival_graph_raw_auxiliary_norm': norm(auxiliary, [i for i in indices if i not in support]),
            'survival_connected_tensors': len(support), 'both_connected_tensors': len(overlap),
            'norm_floor': norm_floor, 'cosine_tolerance': cosine_tolerance}


def parameter_groups(named):
    groups = {key: [] for key in GROUPS}
    for i, (name, _) in enumerate(named):
        matches = [key for key, prefix in GROUPS.items() if name == prefix or name.startswith(prefix+'.')]
        if len(matches) != 1:
            raise ValueError(f'Parameter must belong to one explicit group: {name}, {matches}')
        groups[matches[0]].append(i)
    return {key: indices for key, indices in groups.items() if indices}


def audit_batch(model, batch, starts, cfg, chunk_size, view_source, view_seed, rna_noise_seed):
    """No step, no .grad accumulation; preserve RNG and the caller's train/eval mode.

    All raw terms are returned by the production objective from one graph.
    Checkpointed chunk recomputation replays that graph's same perturbations.
    """
    validate_config(cfg, view_source, len(batch['rna']), chunk_size)
    for name in ('hidden', 'common', 'local_dim', 'local_common', 'interaction_strength', 'match_temperature'):
        if getattr(model, name) != getattr(cfg, name):
            raise ValueError(f'Model and config disagree on {name}')
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named:
        raise ValueError('At least one trainable parameter is required for gradient diagnosis')
    groups, params = parameter_groups(named), tuple(p for _, p in named)
    devices = sorted({p.device.index if p.device.index is not None else torch.cuda.current_device()
                      for p in params if p.device.type == 'cuda'})
    def gradients(term, retain=True):
        if not term.requires_grad:
            return (None,)*len(params)
        result = torch.autograd.grad(term, params, allow_unused=True, retain_graph=retain)
        if any(g is not None and not torch.isfinite(g).all() for g in result):
            raise FloatingPointError('Nonfinite per-term gradient')
        return tuple(None if g is None else g.detach() for g in result)
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator().manual_seed(rna_noise_seed).get_state())
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(rna_noise_seed)
        total, stats, components = pathway_objective(model, batch, starts, cfg, chunk_size,
                                                     view_source, view_seed, return_components=True)
        survival = gradients(components['survival'])
        summed_auxiliary = [None]*len(params)
        sum_individual_norms = 0.
        terms, weights = {}, effective_weights(cfg)
        for name in LOSS_NAMES:
            if not weights[name]:
                terms[name] = {'status': 'disabled', 'weight': weights[name]}
                continue
            if name not in components:
                if name != 'global_common' or not stats['common_skipped']:
                    raise RuntimeError(f'Active term unexpectedly missing: {name}')
                terms[name] = {'status': 'skipped', 'weight': weights[name], 'reason': stats['common_skip_reason']}
                continue
            raw = gradients(components[name])
            comparison = relation(survival, raw, weights[name])
            terms[name] = {'status': 'computed', 'weight': weights[name],
                           'raw_loss': float(components[name].detach()),
                           'weighted_loss': weights[name]*float(components[name].detach()),
                           'survival_graph_space': comparison,
                           'modules': {key: relation(survival, raw, weights[name], indices) for key, indices in groups.items()}}
            sum_individual_norms += comparison['weighted_auxiliary_norm_in_survival_space']
            for i, value in enumerate(raw):
                if value is not None:
                    current = weights[name]*value.double()
                    summed_auxiliary[i] = current if summed_auxiliary[i] is None else summed_auxiliary[i]+current
        direct = gradients(total, retain=False)
    reconstructed = [((torch.zeros_like(p, dtype=torch.float64) if s is None else s.double())
                      +(torch.zeros_like(p, dtype=torch.float64) if a is None else a))
                     for p, s, a in zip(params, survival, summed_auxiliary)]
    residual = [r-(torch.zeros_like(p, dtype=torch.float64) if g is None else g.double())
                for p, r, g in zip(params, reconstructed, direct)]
    error, total_norm = norm(residual), norm(direct)
    if error > 1e-6+1e-5*total_norm:
        raise RuntimeError('Weighted component gradients do not reconstruct the total gradient')
    aggregate = relation(survival, summed_auxiliary)
    return {'scope': 'one batch, one forward graph, current parameterization, no optimizer step',
            'patients': len(batch['rna']), 'model_training_mode': model.training,
            'view_seed': view_seed, 'rna_noise_seed': rna_noise_seed,
            'survival_loss': stats['train_nll'], 'total_loss': stats['total_loss'], 'terms': terms,
            'sum_weighted_auxiliary': {'survival_graph_space': aggregate,
                                     'modules': {key: relation(survival, summed_auxiliary, indices=indices) for key, indices in groups.items()},
                                     'norm_of_sum_over_sum_of_norms': aggregate['raw_auxiliary_norm_in_survival_space']/sum_individual_norms if sum_individual_norms else None},
            'total_descent_direction': relation(survival, reconstructed),
            'gradient_reconstruction': {'absolute_l2_error': error, 'direct_total_norm': total_norm,
                                        'relative_l2_error': error/total_norm if total_norm else None},
            'parameter_groups': {key: {'names': [named[i][0] for i in indices],
                                       'elements': sum(params[i].numel() for i in indices)} for key, indices in groups.items()}}
