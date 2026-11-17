"""Independent synthetic splits with known patient-level morphology mixtures.

No patient data, local patient files, or remote resources are accessed.  The
true per-patch classes and proportions are returned in a separate audit-only
object: they must not be passed into morphology fitting or model prediction.
"""
import math

import numpy as np
import torch


def make_composition_splits(seed, *, n_train=80, n_val=80, n_test=120,
                            min_patches=16, max_patches=32, noise=.35,
                            rna_correlation=.65, composition_effect=2.5,
                            rna_effect=0., interaction_effect=0.):
    """Return ``(splits, truth, metadata)`` for three disjoint generated cohorts.

    Each patient has two latent morphological phenotypes. Their mixture is
    patient-specific, and each patch is sampled from its phenotype's noisy
    feature distribution. RNA correlates with the latent mixture but does not
    contain the true per-patch labels. The risk is a configurable function of
    empirical (not log-ratio) composition and RNA. Hazard has four bins and
    right-censoring is independently drawn before the event time.

    ``splits[split]`` contains *only* bags, RNA, patient IDs, outcome bin, and
    censorship (1 = censored). ``truth[split]`` has latent labels, proportions,
    generating risk, and event/censoring bins for audit only. Ground truth must
    never be used as model features or to fit the clustering library.
    """
    counts = (n_train, n_val, n_test, min_patches, max_patches)
    if type(seed) is not int or any(type(x) is not int or x < 1 for x in counts):
        raise ValueError('Integer seed and strictly positive integer sizes required')
    if min_patches > max_patches:
        raise ValueError('min_patches cannot exceed max_patches')
    settings = (noise, rna_correlation, composition_effect, rna_effect, interaction_effect)
    if not all(math.isfinite(x) for x in settings) or noise <= 0 or not 0 <= rna_correlation <= 1:
        raise ValueError('Invalid noise, correlation or generating risk coefficients')
    # SeedSequence isolates split RNGs; no validation/test sample enters a fit.
    split_seeds = np.random.SeedSequence(seed).spawn(3)
    sizes = {'train': n_train, 'validation': n_val, 'test': n_test}
    splits, truth = {}, {}
    centers = np.asarray([[-3., 0., 0., 0.], [3., 0., 0., 0.]])
    for (split, n), child_seed in zip(sizes.items(), split_seeds):
        rng = np.random.default_rng(child_seed)
        latent = rng.normal(size=n)
        ratio = 1. / (1. + np.exp(-latent))
        rna0 = rna_correlation*latent + math.sqrt(1-rna_correlation**2)*rng.normal(size=n)
        rna = np.stack((rna0, rng.normal(size=n)), axis=1)
        lengths = rng.integers(min_patches, max_patches+1, size=n)
        bags, class_ids, fractions = [], [], []
        for i, length in enumerate(lengths):
            labels = (rng.random(length) < ratio[i]).astype(np.int64)
            patches = centers[labels] + rng.normal(0., noise, size=(length, 4))
            # A small sample-specific feature shift lets patient-paired SSL
            # explore individual variation without supplying cluster labels.
            patches[:, 1] += .2*latent[i]
            bags.append(torch.as_tensor(patches, dtype=torch.float64))
            class_ids.append(labels)
            fractions.append(float(labels.mean()))
        fractions = np.asarray(fractions)
        contrast = 2*fractions-1
        risk = (composition_effect*contrast + rna_effect*rna0
                + interaction_effect*contrast*rna0)
        bins = np.asarray([-1.6, -1.4, -1.2, -1.0])[None, :]+risk[:, None]
        hazard = 1/(1+np.exp(-bins))
        event_draw = rng.random((n, 4)) < hazard
        event_time = np.where(event_draw.any(axis=1), event_draw.argmax(axis=1), 4)
        censor_time = rng.integers(0, 4, size=n)
        event_observed = event_time <= censor_time
        observed_time = np.where(event_observed, event_time, censor_time)
        ids = [f'synthetic-{seed}-{split}-{i:05d}' for i in range(n)]
        splits[split] = {'bags': bags, 'rna': torch.as_tensor(rna, dtype=torch.float64),
                         'ids': ids, 'labels': torch.as_tensor(observed_time, dtype=torch.long),
                         'censorship': torch.as_tensor(~event_observed, dtype=torch.float64)}
        truth[split] = {'true_cluster_labels_audit_only': class_ids,
                        'true_cluster1_fraction_audit_only': fractions,
                        'generating_risk_audit_only': risk,
                        'uncensored_event_bin_audit_only': event_time,
                        'censor_bin_audit_only': censor_time}
    ids = [set(v['ids']) for v in splits.values()]
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i))
    metadata = {'source': 'synthetic_only', 'seed': seed, 'n': sizes,
                'bags': 'two latent prototypes, variable patch counts, phenotype labels held out for audit',
                'rna_correlation_with_latent_mixture': rna_correlation,
                'outcome': {'composition_effect': composition_effect, 'rna_effect': rna_effect,
                            'interaction_effect': interaction_effect,
                            'risk': 'b_c*(2*fraction1-1)+b_r*rna0+b_i*(2*fraction1-1)*rna0',
                            'baseline_hazard_logits': [-1.6, -1.4, -1.2, -1.0],
                            'event_observed_rule': 'event_bin <= censor_bin',
                            'censorship': '1 denotes censoring'},
                'split_rng': 'independent children of numpy SeedSequence'}
    return splits, truth, metadata
