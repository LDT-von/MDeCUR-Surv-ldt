"""Method 3.2 synthetic experiments (CPU only).

Fixed arms, fixed scenarios, fixed seeds.  Every data-driven step (atlas, PCA,
k-means, contrastive encoder, retrieval rate, bin boundaries, head weights) is
fitted on training patients only.  Validation patients are used for nothing but
the predefined checkpoint selection; the frozen model is evaluated once on the
test patients.

Usage:
    python scripts/method3_2_synthetic.py --out results/method3_2_synthetic.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from method3_2_model import (  # noqa: E402
    METHOD3_2_VERSION, Method32Config, fit_method3_2, harrell_c, predict_method3_2,
    survival_bin_labels,
)
from method3_2_synthetic_data import CLUSTERS, generate_cohort  # noqa: E402


SCENARIOS = ['redundant', 'path_only', 'rna_only', 'no_signal', 'xor',
              'xor_additive', 'nonlinear', 'composition']
SEEDS = [11, 23, 37]
N_TRAIN, N_VAL, N_TEST = 160, 60, 120
EPOCHS_SSL, EPOCHS_HEAD = 40, 120

ARMS = {
    'full':            {},
    'uniform_route':   {'route': 'uniform'},
    'no_ssl':          {'route': 'uniform', 'epochs_ssl': 0},
    'no_composition':  {'use_composition': False},
    'no_interaction':  {'use_interaction': False},
    'generic_inter':   {'use_interaction': False, 'use_generic_interaction': True},
    'no_cluster_cond': {'cluster_conditioned': False},
    'shuffle_pairs':   {'shuffle_pairs': True},
    # explicit common / unique decomposition
    'decouple':        {'decouple': True, 'decouple_mode': 'symmetric',
                         'lambda_common': 0.1, 'lambda_unique': 0.1},
    'decouple_noregu': {'decouple': True, 'decouple_mode': 'symmetric'},
    'decouple_asym':   {'decouple': True, 'decouple_mode': 'asymmetric',
                         'lambda_common': 0.1, 'lambda_unique': 0.1},
    'decouple_nocommon': {'decouple': True, 'decouple_mode': 'symmetric',
                           'lambda_unique': 0.1},
    # input-masking negative controls (NOT clinical single-modality baselines:
    # ``block_pathology`` also removes the composition block, and ``block_rna``
    # leaves the interaction term intact, so it is still an RNA-dependent model)
    'path_block_off':  {'block_pathology': True},
    'rna_block_off':   {'block_rna': True},
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state():
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.splitlines()
        return {'head': head, 'dirty_entries': len(status),
                'dirty_files': sorted(status)[:40]}
    except Exception as error:  # pragma: no cover - environment dependent
        return {'error': str(error)}


def _metrics(fitted, cohort, labels, censorship):
    prediction = predict_method3_2(fitted, cohort.bags, cohort.rna, cohort.ids)
    from cancer_parity import discrete_nll
    per_patient = discrete_nll(prediction['logits'], labels, censorship, alpha=0.0,
                                reduction='none')
    risk = prediction['risk']
    c_index = harrell_c(risk, cohort.times, cohort.events)
    return {
        'nll': float(per_patient.mean()),
        'harrell_c': None if c_index is None else float(c_index),
        'risk_mean': float(risk.mean()),
        'n_patients': len(cohort.ids),
        'n_events': int(cohort.events.sum()),
    }


_COHORT_CACHE = {}
_ATLAS_CACHE = {}
_SSL_CACHE = {}


def _cohorts(scenario, seed):
    key = (scenario, seed)
    if key not in _COHORT_CACHE:
        prefix = f'{scenario}_{seed}'
        train = generate_cohort(scenario, N_TRAIN, seed=1000 + seed, prefix=f'train_{prefix}')
        val = generate_cohort(scenario, N_VAL, seed=2000 + seed, prefix=f'val_{prefix}')
        test = generate_cohort(scenario, N_TEST, seed=3000 + seed, prefix=f'test_{prefix}')
        assert not (set(train.ids) & set(val.ids))
        assert not (set(train.ids) & set(test.ids))
        assert not (set(val.ids) & set(test.ids))
        _COHORT_CACHE[key] = (train, val, test)
    return _COHORT_CACHE[key]


def _atlas(scenario, seed):
    key = (scenario, seed)
    if key not in _ATLAS_CACHE:
        from method3_2_model import MorphologyAtlas
        train, _, _ = _cohorts(scenario, seed)
        _ATLAS_CACHE[key] = MorphologyAtlas.fit(
            train.bags, train.ids, clusters=CLUSTERS, latent_dim=8,
            per_patient_cap=64, seed=seed)
    return _ATLAS_CACHE[key]


def _ssl(scenario, seed, config):
    """Shared stage-1 result for arms that differ only in the survival head.

    The atlas, the label-free encoder and the cross-fitted retrieval rate do
    not depend on the head configuration, so arms that keep the same
    ``cluster_conditioned``/``shuffle_pairs``/``epochs_ssl`` reuse exactly the
    same frozen stage-1 objects.  This is a runtime optimisation only; the
    values are identical to training them separately.
    """
    key = (scenario, seed, config.cluster_conditioned, config.shuffle_pairs,
           config.epochs_ssl, config.min_patients, config.folds, config.seed)
    if key in _SSL_CACHE:
        return _SSL_CACHE[key]
    from method3_2_model import (ClusterConditionalEncoder, cross_fitted_association,
                                  train_cluster_contrastive)
    atlas = _atlas(scenario, seed)
    train, _, _ = _cohorts(scenario, seed)
    patients = atlas.encode(train.bags, train.ids)
    rna = train.rna

    def build_encoder():
        return ClusterConditionalEncoder(atlas.latent_dim, rna.shape[1], atlas.clusters,
                                         hidden=config.hidden,
                                         cluster_conditioned=config.cluster_conditioned,
                                         seed=config.seed)

    if config.epochs_ssl <= 0:
        encoder = build_encoder()
        a = torch.zeros(atlas.clusters, dtype=torch.float64)
        stats = {'attempts': [0] * atlas.clusters, 'successes': [0] * atlas.clusters,
                 'a': a.tolist(), 'cross_fitted': False}
        history = []
    else:
        history = train_cluster_contrastive(
            encoder=build_encoder(), patients=patients, rna=rna,
            epochs=config.epochs_ssl, batch_size=config.batch_size,
            temperature=config.temperature, min_patients=config.min_patients,
            lr=config.lr_ssl, seed=config.seed, shuffle_pairs=config.shuffle_pairs,
            patch_fraction=config.patch_fraction, rna_noise=config.rna_noise,
            min_mass=config.min_mass)
        a, stats = cross_fitted_association(
            build_encoder, patients, rna, folds=config.folds, seed=config.seed,
            epochs=config.epochs_ssl, batch_size=config.batch_size,
            temperature=config.temperature, min_patients=config.min_patients,
            lr=config.lr_ssl, shuffle_pairs=config.shuffle_pairs,
            patch_fraction=config.patch_fraction, rna_noise=config.rna_noise,
            min_mass=config.min_mass)
        encoder = build_encoder()
        train_cluster_contrastive(
            encoder=encoder, patients=patients, rna=rna, epochs=config.epochs_ssl,
            batch_size=config.batch_size, temperature=config.temperature,
            min_patients=config.min_patients, lr=config.lr_ssl, seed=config.seed + 101,
            shuffle_pairs=config.shuffle_pairs, patch_fraction=config.patch_fraction,
            rna_noise=config.rna_noise, min_mass=config.min_mass)
    _SSL_CACHE[key] = (encoder, a, stats, history)
    return _SSL_CACHE[key]


def run_arm(scenario, seed, arm_name, overrides):
    settings = {'clusters': CLUSTERS, 'bins': 4, 'hidden': 8, 'seed': seed,
                'epochs_ssl': EPOCHS_SSL, 'epochs_head': EPOCHS_HEAD,
                'folds': 4, 'batch_size': 32}
    settings.update(overrides)
    config = Method32Config(**settings)
    train, val, test = _cohorts(scenario, seed)
    atlas = _atlas(scenario, seed)
    encoder, a, retrieval_stats, ssl_history = _ssl(scenario, seed, config)

    from method3_2_model import (ClusterRoutedSurvivalHead, fit_survival_head,
                                  head_features, survival_bin_labels)
    if config.route == 'info':
        route = a.clone()
    elif config.route == 'uniform':
        route = torch.ones(atlas.clusters, dtype=torch.float64)
    else:
        raise ValueError('route must be "info" or "uniform"')

    head = ClusterRoutedSurvivalHead(config)
    train_features = head_features(encoder, atlas.encode(train.bags, train.ids),
                                   train.rna, route=route)
    labels, censorship, _ = survival_bin_labels(train.times, train.events, bins=config.bins)
    val_patients = atlas.encode(val.bags, val.ids)
    val_features = head_features(encoder, val_patients, val.rna, route=route)
    val_labels, val_censorship, _ = survival_bin_labels(val.times, val.events,
                                                        bins=config.bins)
    history = fit_survival_head(
        head, train_features, labels, censorship, epochs=config.epochs_head,
        batch_size=config.batch_size, lr=config.lr_head, alpha=config.alpha,
        seed=config.seed, val_features=val_features, val_labels=val_labels,
        val_censorship=val_censorship)

    fitted = {'atlas': atlas, 'encoder': encoder, 'head': head, 'route': route,
              'retrieval_a': a, 'retrieval_stats': retrieval_stats,
              'ssl_history': ssl_history, 'head_history': history}
    test_labels, test_censorship, _ = survival_bin_labels(test.times, test.events,
                                                          bins=config.bins)
    metrics = _metrics(fitted, test, test_labels, test_censorship)
    metrics['route'] = fitted['route'].tolist()
    metrics['retrieval_a'] = fitted['retrieval_a'].tolist()
    metrics['retrieval_attempts'] = fitted['retrieval_stats']['attempts']
    metrics['atlas_fingerprint'] = atlas.fingerprint()
    metrics['val_nll_last'] = history[-1].get('val_nll')
    metrics['val_nll_best'] = min((e['val_nll'] for e in history
                                    if e.get('val_nll') is not None), default=None)
    metrics['n_head_parameters'] = sum(p.numel() for p in head.parameters())
    metrics['n_encoder_parameters'] = sum(p.numel() for p in encoder.parameters())
    metrics['cluster_usage'] = encoder.cluster_usage.tolist()
    metrics['cluster_trained'] = encoder.cluster_trained_mask().tolist()
    metrics['rna_valid_fraction'] = float(train_features['rna_valid'].to(torch.float64).mean())
    if config.decouple:
        with torch.no_grad():
            energy = head.common_energy(train_features['u'], train_features['rna'])
            mask = train_features['rna_valid'].to(torch.float64)
            denom = max(1.0, float(mask.sum()))
            metrics['path_common_fraction'] = float((energy['path_common_fraction'] * mask).sum() / denom)
            metrics['rna_common_fraction'] = float((energy['rna_common_fraction'] * mask).sum() / denom)
    return metrics


def scenario_leakage_check(scenario='redundant', seed=11, epochs=15):
    """Changing validation/test data must not change any fitted training state."""
    config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=seed,
                             epochs_ssl=epochs, epochs_head=40)
    from method3_2_model import MorphologyAtlas

    def build(val_seed, test_seed):
        train = generate_cohort(scenario, 80, seed=1000 + seed, prefix='leak_train')
        val = generate_cohort(scenario, 30, seed=val_seed, prefix='leak_val')
        test = generate_cohort(scenario, 30, seed=test_seed, prefix='leak_test')
        atlas = MorphologyAtlas.fit(train.bags, train.ids, clusters=CLUSTERS, latent_dim=8,
                                    per_patient_cap=64, seed=seed)
        fitted = fit_method3_2(atlas, train.bags, train.rna, train.ids, config=config,
                                times=train.times, events=train.events,
                                val_bags=val.bags, val_rna=val.rna, val_ids=val.ids,
                                val_times=val.times, val_events=val.events)
        state = hashlib.sha256()
        state.update(atlas.fingerprint().encode())
        for name, tensor in sorted(fitted['encoder'].state_dict().items()):
            state.update(name.encode())
            state.update(tensor.detach().numpy().tobytes())
        for name, tensor in sorted(fitted['head'].state_dict().items()):
            state.update(name.encode())
            state.update(tensor.detach().numpy().tobytes())
        state.update(fitted['route'].numpy().tobytes())
        return state.hexdigest()

    first = build(2000, 3000)
    second = build(2000, 9999)
    third = build(8888, 3000)
    return {'baseline': first, 'changed_test': second, 'changed_val': third,
            'test_change_keeps_state': first == second,
            'val_change_keeps_state': first == third}


def composition_audit(scenario='redundant', seed=11):
    """Known-answer audit of the atlas: composition, CLR closure, cluster recovery."""
    from method3_2_model import MorphologyAtlas, multiplicative_replacement_clr
    from sklearn.metrics import adjusted_rand_score
    train = generate_cohort(scenario, 120, seed=1000 + seed, prefix='audit_train')
    atlas = MorphologyAtlas.fit(train.bags, train.ids, clusters=CLUSTERS, latent_dim=8,
                                per_patient_cap=64, seed=seed)
    encoded = atlas.encode(train.bags, train.ids)
    rows = []
    predicted, true = [], []
    for patient, bag, labels in zip(encoded, train.bags, train.truth['cluster_labels']):
        rows.append({'composition_sum': float(patient.composition.sum()),
                     'clr_sum': float(patient.clr.sum()),
                     'mass_sum': float(patient.mass.sum()),
                     'valid_count': int(patient.valid.sum())})
        raw = bag if isinstance(bag, torch.Tensor) else bag[0]
        z = ((raw - atlas.patch_mean) / atlas.patch_scale) @ atlas.components
        predicted.append(torch.cdist(z, atlas.centers).argmin(1).numpy())
        true.append(np.asarray(labels))
    clr_matrix = multiplicative_replacement_clr(
        torch.stack([p.composition for p in encoded]))
    ari = adjusted_rand_score(np.concatenate(true), np.concatenate(predicted))
    composition = np.stack([p.composition.numpy() for p in encoded])
    true_composition = np.stack([np.bincount(labels, minlength=CLUSTERS) / len(labels)
                                 for labels in true])
    composition_mae = float(np.abs(composition - true_composition).mean())
    return {'composition_rows': rows[:5],
            'composition_sum_max_deviation': max(abs(r['composition_sum'] - 1.0) for r in rows),
            'clr_sum_max_deviation': max(abs(r['clr_sum']) for r in rows),
            'mass_sum_min': min(r['mass_sum'] for r in rows),
            'clr_matrix_row_sum_max_deviation': float(clr_matrix.sum(1).abs().max()),
            'cluster_ari': float(ari),
            'composition_mae_vs_truth': composition_mae,
            'empty_cluster_repairs': atlas.metadata.get('empty_cluster_repairs'),
            'atlas_fingerprint': atlas.fingerprint()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/method3_2_synthetic_20261116.json')
    parser.add_argument('--scenarios', default=','.join(SCENARIOS))
    parser.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    scenarios = args.scenarios.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    arms = args.arms.split(',')

    started = time.time()
    results = {'version': METHOD3_2_VERSION,
               'generated_utc': datetime.now(timezone.utc).isoformat(),
               'command': ' '.join(sys.argv),
               'environment': {'python': platform.python_version(),
                               'torch': torch.__version__,
                               'numpy': np.__version__,
                               'platform': platform.platform()},
               'git': _git_state(),
               'source_sha256': {name: _sha256(os.path.join(ROOT, name)) for name in (
                   'method3_2_model.py', 'method3_2_synthetic_data.py',
                   'scripts/method3_2_synthetic.py')},
               'scale': {'n_train': N_TRAIN, 'n_val': N_VAL, 'n_test': N_TEST,
                         'epochs_ssl': EPOCHS_SSL, 'epochs_head': EPOCHS_HEAD,
                         'seeds': seeds, 'arms': arms, 'scenarios': scenarios},
               'leakage_check': scenario_leakage_check(),
               'atlas_audit': composition_audit(),
               'runs': {}}

    for scenario in scenarios:
        for seed in seeds:
            for arm in arms:
                key = f'{scenario}|{seed}|{arm}'
                started_arm = time.time()
                try:
                    metrics = run_arm(scenario, seed, arm, ARMS[arm])
                    metrics['status'] = 'ok'
                except Exception as error:  # failures are recorded, never hidden
                    metrics = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
                metrics['seconds'] = round(time.time() - started_arm, 2)
                results['runs'][key] = metrics
                print(key, metrics.get('status'), metrics.get('nll'), metrics.get('harrell_c'),
                      flush=True)

    results['elapsed_seconds'] = round(time.time() - started, 2)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print('written', out_path)


if __name__ == '__main__':
    main()
