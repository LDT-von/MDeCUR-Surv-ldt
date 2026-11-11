"""Paired validation-only comparison of one-interval pathway experiments.

This analyzes an explicit train/val manifest and two completed output folders;
it never loads feature or RNA files. See --help for assumptions and restrictions.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from cancer_metrics import harrell_c, ipcw_brier


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    groups = {'train': [], 'val': []}
    seen = set()
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            identity, split = row['patient_id'], row['split']
            if not identity or identity in seen or split not in groups:
                raise ValueError('Manifest must have unique IDs and contain train/val only')
            seen.add(identity)
            # The original loader converts these labels to float32 tensors.
            t, e = float(np.float32(row['time'])), float(np.float32(row['event']))
            if not math.isfinite(t) or t < 0 or e not in (0., 1.):
                raise ValueError('Invalid survival labels')
            groups[split].append((identity, t, e))
    if not all(groups.values()):
        raise ValueError('Both training and validation patients are required')
    return groups


def read_predictions(path):
    result = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            identity = row['patient_id']
            s = float(row['survival_at_horizon'])
            if not identity or identity in result or not math.isfinite(s) or not 0 < s < 1:
                raise ValueError('Predictions require unique IDs and 0 < survival < 1')
            result[identity] = s
    return result


def load_run(directory, manifest_hash):
    d = Path(directory)
    report_path, prediction_path = d / 'report.json', d / 'val_predictions.csv'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report['evaluation_split'] != 'val' or report['test_evaluated'] is not False
            or report['manifest_sha256'] != manifest_hash):
        raise ValueError('Run must be validation only and use the provided manifest')
    if (report['interval_starts'] != [0] or not math.isfinite(report['prediction_horizon'])
            or report['prediction_horizon'] <= 0):
        raise ValueError('Rate recovery is supported only for a single interval starting at zero')
    return report, read_predictions(prediction_path), {
        'report_sha256': digest(report_path), 'predictions_sha256': digest(prediction_path)}


def check_comparable(a, b, comparison):
    if a.get('metric_conventions') != b.get('metric_conventions'):
        raise ValueError('Runs differ in metric_conventions; reconcile protocols before paired comparison')
    for key in ('architecture', 'manifest_sha256', 'cohort_input_sha256',
                'prediction_horizon', 'interval_starts', 'time_unit', 'endpoint',
                'view_source', 'minimum_patient_batch', 'stream_config',
                'pathway_mapping', 'rna_preprocessing'):
        if a[key] != b[key]:
            raise ValueError(f'Runs differ in {key}; paired ablation assumptions not met')
    ca, cb = a['config'], b['config']
    ma, mb = a['model_kwargs'], b['model_kwargs']
    if set(ca) != set(cb) or set(ma) != set(mb):
        raise ValueError('Runs have different config or model_kwargs keys')
    if ('interaction_strength' not in ca or 'interaction_strength' not in ma
            or ca['interaction_strength'] != ma['interaction_strength']
            or cb['interaction_strength'] != mb['interaction_strength']):
        raise ValueError('Config and model interaction strengths must agree in each run')
    changes = {k: [ca[k], cb[k]] for k in sorted(ca) if ca[k] != cb[k]}
    model_changes = {k: [ma[k], mb[k]] for k in sorted(ma) if ma[k] != mb[k]}
    allowed = {'matched-off': {'matched_common', 'matched_unique'},
               'uniform-common': {'common_mode'},
               'interaction-off': {'interaction_strength'}}
    if comparison not in allowed:
        raise ValueError(f'Unknown comparison: {comparison}')
    if set(changes) != allowed[comparison]:
        raise ValueError(f'Unexpected configuration differences: {changes}')
    expected_model_changes = {'interaction_strength'} if comparison == 'interaction-off' else set()
    if set(model_changes) != expected_model_changes:
        raise ValueError(f'Unexpected model_kwargs differences: {model_changes}')
    if comparison == 'uniform-common':
        if changes['common_mode'] != ['stability', 'uniform']:
            raise ValueError('Expected stability reference and uniform candidate')
    else:
        for key in allowed[comparison]:
            before, after = changes[key]
            if (type(before) not in (int, float) or not math.isfinite(before)
                    or before <= 0 or type(after) not in (int, float) or after != 0):
                raise ValueError(f'Expected positive reference and zero candidate for {key}')
    return changes, model_changes


def evaluate(times, events, survival, train_times, train_events, horizon):
    # Recovered rate is valid only under the checked single-interval model.
    rate = -np.log(survival) / horizon
    nll = np.mean(rate * times - events * np.log(rate))
    c = harrell_c(times, events, 1 - survival)
    brier = ipcw_brier(train_times, train_events, times, events, survival, horizon)
    return {'nll': float(nll), 'harrell_c': c['value'], 'comparable_pairs': c['comparable_pairs'],
            'ipcw_brier': brier['value'], 'brier_info': brier}


def compare(manifest, reference, candidate, seed, replicates, comparison):
    manifest_hash = digest(manifest)
    groups = read_manifest(manifest)
    ar, ap, ah = load_run(reference, manifest_hash)
    br, bp, bh = load_run(candidate, manifest_hash)
    changes, model_changes = check_comparable(ar, br, comparison)
    ids, times, events = zip(*groups['val'])
    if set(ids) != set(ap) or set(ids) != set(bp):
        raise ValueError('Prediction IDs must match the complete validation set exactly')
    t, e = np.array(times), np.array(events)
    train_t = np.array([r[1] for r in groups['train']])
    train_e = np.array([r[2] for r in groups['train']])
    av, bv = (np.array([p[i] for i in ids]) for p in (ap, bp))
    h = float(ar['prediction_horizon'])
    left, right = (evaluate(t, e, s, train_t, train_e, h) for s in (av, bv))
    for computed, report in ((left, ar), (right, br)):
        for key, orig in (('nll', report['evaluation_nll']),
                          ('harrell_c', report['evaluation_c_index_risk_at_horizon']['value']),
                          ('ipcw_brier', report['evaluation_ipcw_brier_at_horizon']['value'])):
            if (computed[key] is None or orig is None or not math.isfinite(computed[key])
                    or not math.isfinite(orig) or abs(computed[key] - orig) > 2e-5):
                raise ValueError(f'Recomputed {key} fails original report check: {computed[key]} vs {orig}')
    metrics = ('nll', 'harrell_c', 'ipcw_brier')
    delta = {k: right[k] - left[k] for k in metrics}
    samples = {k: [] for k in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        i = rng.integers(0, len(t), size=len(t))
        x, y = (evaluate(t[i], e[i], s[i], train_t, train_e, h) for s in (av, bv))
        for k in metrics:
            if x[k] is not None and y[k] is not None:
                samples[k].append(y[k] - x[k])
    intervals = {k: {'valid_replicates': len(v), 'requested_replicates': replicates,
                     'percentile_95': [float(x) for x in np.percentile(v, [2.5, 97.5])] if v else None}
                 for k, v in samples.items()}
    return {'analysis': 'candidate_minus_reference', 'comparison': comparison, 'validation_patients': len(t),
            'training_patients': len(train_t), 'horizon': h,
            'config_changes': changes, 'model_kwargs_changes': model_changes,
            'reference': left, 'candidate': right, 'difference': delta,
            'bootstrap': {'seed': seed, 'intervals': intervals,
                          'method': 'paired validation-patient percentile bootstrap; training censor KM fixed'},
            'inputs': {'manifest_sha256': manifest_hash, 'reference': ah, 'candidate': bh,
                       'analysis_script_sha256': digest(__file__),
                       'cancer_metrics_sha256': digest(importlib.util.find_spec('cancer_metrics').origin)},
            'runtime': {'numpy': np.__version__, 'bootstrap_rng': 'numpy default_rng (PCG64)'},
            'limits': ['NLL reconstructed from saved horizon survival under single-interval constant hazard; checked against reports.',
                       'Confidence intervals condition on fitted models, training sample and censoring KM; they exclude training/selection uncertainty.',
                       'Validation selected both checkpoint epochs and is used for development; this is not an independent test.',
                       'Harrell C uses full follow-up and risk at the horizon; it is not a time-dependent C-index.',
                       'RNA upstream provenance and original feature encoder require separate verification.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, help='train/val-only manifest used by both runs')
    p.add_argument('--reference', required=True, help='full-model reference output directory')
    p.add_argument('--candidate', required=True, help='candidate output directory')
    p.add_argument('--comparison', required=True, choices=['matched-off', 'uniform-common', 'interaction-off'],
                   help='predefined, strictly checked change from reference to candidate')
    p.add_argument('--out', required=True, help='new JSON output file; never overwritten')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261110)
    args = p.parse_args()
    if args.bootstrap < 1:
        p.error('--bootstrap must be positive')
    result = compare(args.manifest, args.reference, args.candidate, args.seed, args.bootstrap, args.comparison)
    with Path(args.out).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'out': args.out, 'difference': result['difference'],
                      'bootstrap': result['bootstrap']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
