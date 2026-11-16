"""Known-answer audits for frozen MorphID conditional-mean residuals.

Only generated arrays are used. These checks concern the probe/residual
mechanism, not prognosis, patient generalization, biological uniqueness,
statistical independence, or HPL phenotype reproducibility. The small linear
least-squares reference is a mathematical control, not a survival baseline.

All data sizes, seeds, probe settings, and gates are fixed before execution.
An output path must be new. A failed gate is recorded and returns exit code 2;
the script never launches another experiment or patient-training stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import morphid_probe
from morphid_probe import CrossFittedRBFProbe, FrozenRBFProbe, r2_score


AUDIT_VERSION = 'morphid_frozen_probe_known_answers_v1'
SEED = 20261116
FOLDS = 4


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _correlation(left, right):
    left, right = _array(left).reshape(-1), _array(right).reshape(-1)
    left, right = left-left.mean(), right-right.mean()
    denominator = np.linalg.norm(left)*np.linalg.norm(right)
    if denominator == 0:
        return None
    return float(np.dot(left, right)/denominator)


def _r2(target, prediction):
    """Use the production scorer; negative values must remain negative."""
    return r2_score(target, prediction)['uniform_average']


def _data_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(_array(value))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _fit_probes(x, y, name, *, alpha=1e-3, seed=SEED):
    settings = {'alpha': alpha, 'bandwidth': None, 'pca_dim': None}
    full = FrozenRBFProbe(**settings).fit(x, y)
    ids = [f'{name}-generated-{i:04d}' for i in range(len(x))]
    cross = CrossFittedRBFProbe(**settings).fit(x, y, ids, folds=FOLDS, seed=seed)
    fold_ids = _array(cross.fold_ids).astype(np.int64)
    summary = {'settings': settings, 'folds': FOLDS,
               'heldout_fold_sizes': [int(np.sum(fold_ids == k)) for k in range(FOLDS)],
               'oof_r2': _r2(y, cross.oof_predictions),
               'in_sample_r2': _r2(y, full.predict(x)),
               'fresh_prediction_rule': 'mean of the four held-out-fold models',
               'finite_sample_caution': 'OOF rows each use one held-out model; new rows use the fold ensemble.'}
    return full, cross, summary


def nonlinear_mean_case():
    rng = np.random.default_rng(SEED+1)
    x = np.linspace(-2., 2., 160).reshape(-1, 1)
    y = x*x-4./3.
    fresh_x = rng.uniform(-2., 2., size=(2048, 1))
    fresh_y = fresh_x*fresh_x-4./3.
    _, cross, fit = _fit_probes(x, y, 'S1', seed=SEED+1)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    linear_prediction = np.column_stack((np.ones(len(fresh_x)), fresh_x)) @ coefficients
    linear_r2 = _r2(fresh_y, linear_prediction)
    nonlinear_r2 = _r2(fresh_y, cross.predict(fresh_x))
    correlation = _correlation(x, y)
    checks = {'symmetric_training_linear_correlation_zero': abs(correlation) < 1e-12,
              'linear_fresh_r2_below_0p05': linear_r2 < .05,
              'nonlinear_oof_r2_above_0p80': fit['oof_r2'] > .80,
              'nonlinear_fresh_r2_above_0p90': nonlinear_r2 > .90}
    return {'scenario': 'S1_nonlinear_conditional_mean',
            'truth': 'p=h^2-4/3, h uniform[-2,2]; deterministic dependence with zero population linear covariance.',
            'generated_patients': {'train': len(x), 'fresh': len(fresh_x)},
            'data_sha256': _data_hash(x, y, fresh_x, fresh_y), 'fit': fit,
            'training_linear_correlation': correlation, 'linear_fresh_r2': linear_r2,
            'nonlinear_fresh_r2': nonlinear_r2, 'checks': checks,
            'interpretation': 'A nonlinear predictor can find this conditional-mean relation; a linear probe cannot.'}


def overlapping_direction_case():
    rng = np.random.default_rng(SEED+2)
    x = rng.uniform(-2., 2., size=(256, 1))
    epsilon = .7*rng.normal(size=(256, 1))
    y = x+epsilon
    fresh_x = rng.uniform(-2., 2., size=(2048, 1))
    fresh_epsilon = .7*rng.normal(size=(2048, 1))
    fresh_y = fresh_x+fresh_epsilon
    _, cross, fit = _fit_probes(x, y, 'S2', alpha=1e-2, seed=SEED+2)
    residual = fresh_y-_array(cross.predict(fresh_x))
    correlation = _correlation(residual, fresh_epsilon)
    residual_variance = float(residual.var())
    # The scalar population map W=1 spans the entire feature coordinate.
    hard_span_residual = (1.-1.)*fresh_y
    checks = {'prediction_residual_preserves_epsilon_corr_above_0p85': correlation > .85,
              'prediction_residual_variance_above_0p25': residual_variance > .25,
              'hard_span_projection_erases_everything': bool(np.array_equal(hard_span_residual, np.zeros_like(fresh_y)))}
    return {'scenario': 'S2_shared_and_unpredictable_signal_overlap',
            'truth': 'p=h+epsilon, epsilon independent; both contributions occupy the same scalar coordinate.',
            'generated_patients': {'train': len(x), 'fresh': len(fresh_x)},
            'data_sha256': _data_hash(x, y, fresh_x, fresh_y, fresh_epsilon), 'fit': fit,
            'residual_to_true_epsilon_correlation': correlation,
            'residual_variance': residual_variance,
            'true_epsilon_variance': float(fresh_epsilon.var()),
            'hard_span_residual_variance': float(hard_span_residual.var()), 'checks': checks,
            'interpretation': 'Subtract the predicted value, not the entire feature direction that contains it.'}


def independent_noise_overfit_case():
    rng = np.random.default_rng(SEED+3)
    x = rng.normal(size=(64, 96))
    y = rng.normal(size=(64, 1))
    fresh_x = rng.normal(size=(1024, 96))
    fresh_y = rng.normal(size=(1024, 1))
    _, cross, fit = _fit_probes(x, y, 'S3', alpha=1e-10, seed=SEED+3)
    fresh_r2 = _r2(fresh_y, cross.predict(fresh_x))
    known_negative = _r2(np.array([[-1.], [1.]]), np.array([[3.], [3.]]))
    checks = {'kernel_can_fit_training_noise': fit['in_sample_r2'] > .95,
              'oof_does_not_support_predictability': fit['oof_r2'] < .20,
              'fresh_test_does_not_support_predictability': fresh_r2 < .20,
              'negative_r2_is_preserved': known_negative < 0.}
    return {'scenario': 'S3_high_dimensional_independent_noise',
            'truth': 'RNA and pathology targets are generated independently; the weak ridge is deliberately allowed to overfit.',
            'generated_patients': {'train': len(x), 'fresh': len(fresh_x)},
            'rna_dimensions': x.shape[1], 'data_sha256': _data_hash(x, y, fresh_x, fresh_y),
            'fit': fit, 'fresh_r2': fresh_r2, 'known_negative_r2': known_negative,
            'checks': checks,
            'interpretation': 'In-sample fit is not an identification criterion. Low held-out R2 is not a proof of independence.'}


def conditional_variance_case():
    rng = np.random.default_rng(SEED+4)
    base_x = np.linspace(-2., 2., 80).reshape(-1, 1)
    base_epsilon = rng.normal(size=(80, 1))
    x = np.repeat(base_x, 2, axis=0)
    # Antithetic pairs give exactly zero empirical conditional means on the
    # training grid while preserving nonconstant conditional variance.
    y = np.column_stack((base_x[:, 0]*base_epsilon[:, 0],
                         -base_x[:, 0]*base_epsilon[:, 0])).reshape(-1, 1)
    fresh_x = rng.uniform(-2., 2., size=(4096, 1))
    fresh_y = fresh_x*rng.normal(size=(4096, 1))
    full, cross, fit = _fit_probes(x, y, 'S4', alpha=1e-2, seed=SEED+4)
    full_prediction = _array(full.predict(fresh_x))
    residual = fresh_y-full_prediction
    squared_correlation = _correlation(residual*residual, fresh_x*fresh_x)
    full_r2 = _r2(fresh_y, full_prediction)
    ensemble_r2 = _r2(fresh_y, cross.predict(fresh_x))
    checks = {'full_fit_conditional_mean_near_zero': float(np.max(np.abs(full_prediction))) < 1e-8,
              'mean_probe_fresh_r2_near_zero': abs(full_r2) < .05,
              'residual_variance_still_depends_on_rna': squared_correlation > .30}
    return {'scenario': 'S4_conditional_variance_limit',
            'truth': 'p=h*epsilon, E[epsilon]=0; E[p|h]=0 but Var(p|h)=h^2.',
            'generated_patients': {'train': len(x), 'fresh': len(fresh_x)},
            'data_sha256': _data_hash(x, y, fresh_x, fresh_y), 'fit': fit,
            'full_fit_fresh_r2': full_r2, 'fold_ensemble_fresh_r2': ensemble_r2,
            'squared_residual_to_squared_rna_correlation': squared_correlation,
            'independence_established': False, 'checks': checks,
            'interpretation': 'The mean probe does not identify conditional-variance dependence. This is an explicit limitation, not a claimed detection mechanism.'}


def run(output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    sources = {'morphid_probe.py': Path(morphid_probe.__file__),
               'scripts/check_morphid_synthetic.py': Path(__file__)}
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    before = {name: digest(path) for name, path in sources.items()}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cases = []
        for function in (nonlinear_mean_case, overlapping_direction_case,
                         independent_noise_overfit_case, conditional_variance_case):
            case = function()
            case['passed'] = all(case['checks'].values())
            cases.append(case)
            print(json.dumps({'scenario': case['scenario'], 'passed': case['passed']}), flush=True)
        source_unchanged = before == {name: digest(path) for name, path in sources.items()}
        receipt = {'audit_version': AUDIT_VERSION,
                   'scope': 'CPU generated mathematical mechanism audits only; no patient/survival training or prognosis claim.',
                   'status': 'passed' if source_unchanged and all(c['passed'] for c in cases) else 'failed',
                   'seed': SEED, 'threads': 1, 'source_sha256': before,
                   'sources_unchanged_during_audit': source_unchanged,
                   'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                               'torch': str(torch.__version__), 'device': 'cpu'},
                   'cases': cases, 'independence_established': False,
                   'limits': ['Frozen probe inputs are supplied as generated coordinates, not patient RNA or image features.',
                              'No threshold on R2 is used to label a residual biologically unique.',
                              'Fresh generated samples validate these constructed cases, not patient generalization.',
                              'The conditional-variance example demonstrates a mean-probe limitation.',
                              'Cross-fit training rows and new-row fold ensembles have different finite-sample prediction rules.']}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as handle:
            json.dump(receipt, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        return receipt
    finally:
        torch.set_num_threads(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='New compact JSON receipt; existing outputs are refused')
    arguments = parser.parse_args()
    result = run(arguments.out)
    print(json.dumps({'status': result['status'], 'out': str(Path(arguments.out).resolve())}), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 2)
