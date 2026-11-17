"""Independent, synthetic-only survival fixtures for Method 3.1 mechanism checks.

Patients are generated in memory; this module never opens real features, labels,
or remote paths. Split dictionaries contain *only* model-observable data.
Latents, true patch groups, event draws and generating risks are returned in a
separate audit dictionary and must never be provided to a fitted model.

The phenotype mixture and its relation with RNA differ from the earlier toy
generators; fresh seeds, masks and outcome definitions should be declared by
the calling experimental protocol before looking at held-out outcomes.
"""
import math

import numpy as np
import torch


SCENARIOS = ('xor', 'xor_smooth', 'xor_time_varying', 'shared', 'additive',
             'rna_only', 'path_only', 'null', 'redundant_nonlinear',
             'composition_only', 'composition_plus_rna',
             'composition_interaction', 'heteroscedastic_null')


def _basis(rng, dim):
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q*np.where(np.diag(r) < 0, -1., 1.)[None, :]


def make_independent_splits(seed, scenario, *, n_train=160, n_val=120, n_test=360,
                            min_patches=12, max_patches=36, patch_noise=.5,
                            mismatch_test=False):
    """Generate three patient-disjoint outcome-bearing datasets and audit truth.

    ``mismatch_test=True`` cyclically permutes *only* test RNA, keeping the
    original test outcome fixed. It is a controlled pair-mismatch experiment,
    not a valid paired patient evaluation or a universal independence test.
    The same seed without this flag yields the exact matched training and
    validation inputs and the same test labels/bags.

    Censoring is 1 when censored at the end of the observed discrete bin.
    An event tied with censoring in the same bin is observed. All risk values
    and unrotated generating factors belong solely to ``audit_truth``.
    """
    sizes = (n_train, n_val, n_test, min_patches, max_patches)
    if type(seed) is not int or seed < 0 or scenario not in SCENARIOS:
        raise ValueError('Use a nonnegative integer seed and a known scenario')
    if any(type(v) is not int or v < 1 for v in sizes) or min_patches > max_patches:
        raise ValueError('All split and patch sizes must be positive integers')
    if (not math.isfinite(patch_noise) or patch_noise <= 0 or type(mismatch_test) is not bool
            or (mismatch_test and n_test < 2)):
        raise ValueError('Positive patch noise and at least two mismatched test patients are required')

    basis_seed, train_seed, val_seed, test_seed = np.random.SeedSequence(seed).spawn(4)
    basis_rng = np.random.default_rng(basis_seed)
    pathology_basis, rna_basis = _basis(basis_rng, 4), _basis(basis_rng, 2)
    observations, audit = {}, {'pathology_basis_audit_only': pathology_basis,
                               'rna_basis_audit_only': rna_basis}
    split_specs = (('train', n_train, train_seed), ('validation', n_val, val_seed),
                   ('test', n_test, test_seed))
    for name, n, child in split_specs:
        rng = np.random.default_rng(child)
        pathology = rng.normal(size=(n, 2))
        rna = rng.normal(size=(n, 2))
        composition_latent = rng.normal(size=n)
        if scenario == 'shared':
            rna[:, 0] = .75*pathology[:, 0]+math.sqrt(1-.75**2)*rna[:, 0]
        elif scenario == 'redundant_nonlinear':
            rna[:, 0] = .95*pathology[:, 0]+math.sqrt(1-.95**2)*rna[:, 0]
        elif scenario == 'heteroscedastic_null':
            # E[P0|G0]=0, but Var(P0|G0)=G0^2; no outcome signal.
            pathology[:, 0] = rna[:, 0]*pathology[:, 0]
        elif scenario == 'composition_only':
            # RNA is an imperfect proxy for composition in this one scenario.
            rna[:, 0] = .60*composition_latent+math.sqrt(1-.60**2)*rna[:, 0]

        mixture_probability = 1/(1+np.exp(-.95*composition_latent))
        lengths = rng.integers(min_patches, max_patches+1, size=n)
        bags, memberships, fractions = [], [], []
        for i, length in enumerate(lengths):
            cluster = (rng.random(length) < mixture_probability[i]).astype(np.int64)
            patches = rng.normal(scale=patch_noise, size=(length, 4))
            patches[:, :2] += pathology[i]
            patches[:, 2] += 2.3*(2*cluster-1)
            # A second phenotype direction makes the mixtures non-collinear.
            patches[:, 3] += .7*(2*cluster-1) + .15*composition_latent[i]
            bags.append(torch.as_tensor(patches @ pathology_basis, dtype=torch.float64))
            memberships.append(cluster)
            fractions.append(float(cluster.mean()))
        fractions = np.asarray(fractions)
        clr_proxy = 2*fractions-1
        p0, g0 = pathology[:, 0], rna[:, 0]
        if scenario in ('xor', 'xor_time_varying'):
            risk = 1.4*np.sign(p0*g0)
        elif scenario == 'xor_smooth':
            risk = 2.2*np.tanh(p0)*np.tanh(g0)
        elif scenario == 'shared':
            risk = 1.15*p0
        elif scenario == 'additive':
            risk = .8*(p0+g0)
        elif scenario == 'rna_only':
            risk = 1.15*g0
        elif scenario == 'path_only':
            risk = 1.15*p0
        elif scenario == 'redundant_nonlinear':
            risk = 2.2*(p0*p0/(1+p0*p0)-.35)
        elif scenario == 'composition_only':
            risk = 2.4*clr_proxy
        elif scenario == 'composition_plus_rna':
            risk = 1.8*clr_proxy+1.0*g0
        elif scenario == 'composition_interaction':
            risk = 1.5*clr_proxy+1.8*clr_proxy*g0
        else:
            risk = np.zeros(n, dtype=np.float64)
        bin_multiplier = np.asarray([1., 1., -1., -1.]) if scenario == 'xor_time_varying' else np.ones(4)
        risk_by_bin = risk[:, None]*bin_multiplier[None, :]
        baseline = np.asarray([-1.7, -1.35, -1.05, -.8], dtype=np.float64)
        logits = baseline[None, :]+risk_by_bin
        # Stable sigmoid keeps even extreme finite input values well-defined.
        hazard = np.empty_like(logits)
        positive = logits >= 0
        hazard[positive] = 1/(1+np.exp(-logits[positive]))
        exp_neg = np.exp(logits[~positive])
        hazard[~positive] = exp_neg/(1+exp_neg)
        event_hits = rng.random((n, 4)) < hazard
        event_bin = np.where(event_hits.any(axis=1), event_hits.argmax(axis=1), 4)
        censor_bin = rng.integers(0, 4, size=n)
        observed = event_bin <= censor_bin
        observed_bin = np.where(observed, event_bin, censor_bin)
        rotated_rna = rna @ rna_basis
        permutation = np.arange(n)
        if mismatch_test and name == 'test':
            permutation = np.roll(permutation, 1)
            rotated_rna = rotated_rna[permutation]
        ids = [f'independent-{seed}-{name}-{i:06d}' for i in range(n)]
        observations[name] = {'bags': bags, 'rna': torch.as_tensor(rotated_rna, dtype=torch.float64),
                              'ids': ids, 'labels': torch.as_tensor(observed_bin, dtype=torch.long),
                              'censorship': torch.as_tensor(~observed, dtype=torch.float64)}
        audit[name] = {'pathology_latent_audit_only': pathology,
                       'rna_latent_audit_only': rna,
                       'composition_latent_audit_only': composition_latent,
                       'true_patch_groups_audit_only': memberships,
                       'realized_group_fraction_audit_only': fractions,
                       'risk_by_bin_audit_only': risk_by_bin,
                       'event_bin_audit_only': event_bin,
                       'censor_bin_audit_only': censor_bin,
                       'rna_row_permutation_audit_only': permutation}

    id_sets = [set(v['ids']) for v in observations.values()]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError('Synthetic patient split collision')
    metadata = {'synthetic_only': True, 'seed': seed, 'scenario': scenario,
                'split_sizes': {'train': n_train, 'validation': n_val, 'test': n_test},
                'patch_count_range': [min_patches, max_patches], 'patch_noise': float(patch_noise),
                'pathology_dim': 4, 'rna_dim': 2,
                'split_randomness': 'independent SeedSequence children; common fixed seed-specific rotations',
                'mismatch_test_rna_only': mismatch_test,
                'truth_in_model_inputs': False,
                'hazard': 'sigmoid([-1.7,-1.35,-1.05,-0.8] + generating risk by bin)',
                'event_rule': 'event_bin <= censor_bin; censorship 1 means censor at end of observed bin',
                'meaning': {'composition_only': 'RNA is a proxy for composition; risk is composition only',
                            'composition_plus_rna': 'composition and independent RNA have additive direct effects',
                            'composition_interaction': 'risk includes composition by RNA interaction',
                            'redundant_nonlinear': 'RNA proxy; outcome depends on pathology alone',
                            'heteroscedastic_null': 'P0=G0*epsilon; zero risk, dependent conditional variance',
                            'xor_time_varying': 'XOR risk changes sign across time bins'}.get(scenario, scenario)}
    return observations, audit, metadata
