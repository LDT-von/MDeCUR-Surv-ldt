"""Synthetic-only, independently split rotated multimodal interaction data.

Returns model inputs separately from audit-only generative truth.  No local
patient files, paths, or external services are opened by this module.
"""
import math

import numpy as np
import torch


SCENARIOS = ('xor', 'shared', 'additive', 'rna_only', 'path_only', 'null',
             'redundant_nonlinear')


def _orthogonal(rng, dim):
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    # Fix the arbitrary QR column signs to make the generated basis explicit.
    signs = np.where(np.diag(r) < 0., -1., 1.)
    return q*signs[None, :]


def make_interaction_splits(seed, scenario, *, n_train=80, n_val=80, n_test=240,
                            min_patches=16, max_patches=32, patch_noise=.4):
    """Create ``(splits, audit_truth, metadata)`` for one fixed seed/scenario.

    Each patient has a pathology bag and two-dimensional RNA. Two latent
    morphology components are present in each bag; their mixture is drawn
    independently of the patient-level pathology and RNA latent variables.
    One pair of fixed random orthogonal bases rotates the four-dimensional
    pathology features and two-dimensional RNA features across all splits.

    The returned split dictionaries contain only the inputs/observed outcomes;
    latent factors, generating risk, cluster labels and the bases are stored
    separately under ``audit_truth`` and must never be inputs to model fitting.
    Right censoring occurs at the end of the observed label bin (1=censored).
    """
    if type(seed) is not int or seed < 0 or scenario not in SCENARIOS:
        raise ValueError('Provide a nonnegative integer seed and a known scenario')
    sizes = (n_train, n_val, n_test, min_patches, max_patches)
    if any(type(n) is not int or n < 1 for n in sizes) or min_patches > max_patches:
        raise ValueError('Positive integer sample and patch sizes are required')
    if not math.isfinite(patch_noise) or patch_noise <= 0:
        raise ValueError('Patch noise must be finite and positive')

    basis_seed, train_seed, val_seed, test_seed = np.random.SeedSequence(seed).spawn(4)
    basis_rng = np.random.default_rng(basis_seed)
    path_basis, rna_basis = _orthogonal(basis_rng, 4), _orthogonal(basis_rng, 2)
    # The shared and redundant cases have correlation for the first latent
    # coordinate. All other cases have independent P and G by construction.
    rho = .8 if scenario == 'shared' else .98 if scenario == 'redundant_nonlinear' else 0.
    outputs, truths = {}, {'basis_path_audit_only': path_basis,
                           'basis_rna_audit_only': rna_basis}
    children = (('train', n_train, train_seed), ('validation', n_val, val_seed),
                ('test', n_test, test_seed))
    for name, n, child in children:
        rng = np.random.default_rng(child)
        p, e = rng.normal(size=(n, 2)), rng.normal(size=(n, 2))
        g = e.copy()
        g[:, 0] = rho*p[:, 0] + math.sqrt(1-rho*rho)*e[:, 0]
        rna = g @ rna_basis
        if scenario == 'xor':
            risk = 1.4*np.sign(p[:, 0]*g[:, 0])
        elif scenario == 'shared':
            risk = 1.2*p[:, 0]
        elif scenario == 'additive':
            risk = .8*(p[:, 0]+g[:, 0])
        elif scenario == 'rna_only':
            risk = 1.2*g[:, 0]
        elif scenario == 'path_only':
            risk = 1.2*p[:, 0]
        elif scenario == 'redundant_nonlinear':
            risk = 1.4*(p[:, 0]**2-1.)
        else:
            risk = np.zeros(n, dtype=np.float64)

        sizes_i = rng.integers(min_patches, max_patches+1, size=n)
        composition = rng.uniform(.3, .7, size=n)
        bags, clusters = [], []
        for i, size in enumerate(sizes_i):
            membership = (rng.random(size) < composition[i]).astype(np.int64)
            # Latent coordinate 2 separates two morphology prototypes; this is
            # not supplied to the model, and their ratio has no direct effect
            # on the outcome in these interaction-specific scenarios.
            tile = rng.normal(scale=patch_noise, size=(size, 4))
            tile[:, :2] += p[i]
            tile[:, 2] += 2.*(2*membership-1)
            bags.append(torch.as_tensor(tile @ path_basis, dtype=torch.float64))
            clusters.append(membership)

        base_logits = np.asarray([-1.6, -1.4, -1.2, -1.0], dtype=np.float64)
        logits = base_logits[None, :]+risk[:, None]
        # Branch-stable sigmoid avoids overflow on extreme quadratic effects.
        hazard = np.empty_like(logits)
        nonnegative = logits >= 0
        hazard[nonnegative] = 1./(1.+np.exp(-logits[nonnegative]))
        exp_z = np.exp(logits[~nonnegative])
        hazard[~nonnegative] = exp_z/(1.+exp_z)
        draws = rng.random((n, 4)) < hazard
        event_time = np.where(draws.any(axis=1), draws.argmax(axis=1), 4)
        censor_time = rng.integers(0, 4, size=n)
        observed = event_time <= censor_time
        observed_bin = np.where(observed, event_time, censor_time)
        ids = [f'interaction-synthetic-{seed}-{name}-{i:05d}' for i in range(n)]
        outputs[name] = {'bags': bags, 'rna': torch.as_tensor(rna, dtype=torch.float64),
                         'ids': ids, 'labels': torch.as_tensor(observed_bin, dtype=torch.long),
                         'censorship': torch.as_tensor(~observed, dtype=torch.float64)}
        truths[name] = {'latent_path_audit_only': p, 'latent_rna_audit_only': g,
                        'generating_risk_audit_only': risk,
                        'true_patch_clusters_audit_only': clusters,
                        'event_bin_audit_only': event_time,
                        'censor_bin_audit_only': censor_time}

    metadata = {'source': 'in_memory_synthetic_only', 'seed': seed, 'scenario': scenario,
                'split_sizes': {'train': n_train, 'validation': n_val, 'test': n_test},
                'split_rng': 'independent SeedSequence children; common seed-specific orthogonal bases',
                'path_dimensions': 4, 'rna_dimensions': 2,
                'path_patch_count_range': [min_patches, max_patches],
                'latent_first_coordinate_correlation': rho,
                'hazard_logits': '[-1.6,-1.4,-1.2,-1.0] + generating risk',
                'censorship_convention': '1=censored; observed event when event_bin <= censor_bin',
                'audit_truth_never_supplied_to_model': True}
    return outputs, truths, metadata
