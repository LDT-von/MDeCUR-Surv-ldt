"""Read-only rescoring of the two-arm V1 archive, best and last checkpoints.

Reads CSV/JSON and hashes opaque checkpoint bytes. Never loads model objects,
features or RNA arrays. Original reports and predictions are not rewritten.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cancer_parity import parity_metrics, dct_bin_edges
from cancer_parity_audit import audit_parity_metrics
from scripts.audit_parity_saved_predictions import (
    AuditInputError, _sha256, _read_manifest, _read_reference, _read_report,
    _read_predictions, _json_safe_metrics, C_ATOL)


MODELS = ('mdecur_aux_off', 'camr_2m')
V1 = 'blca_legacy_fold0_matched_inputs_v1'


def _require(condition, message):
    if not condition:
        raise AuditInputError(message)


def _equal_metric(left, right):
    if left is None or right is None:
        return left is None and right is None
    try:
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        return a.shape == b.shape and bool(np.allclose(a, b, atol=1e-8, rtol=1e-8, equal_nan=False))
    except (ValueError, TypeError):
        return False


def audit_comparison(manifest, survival_reference, comparison_root, plan_path=None):
    root = Path(comparison_root)
    paths = {'manifest': Path(manifest), 'survival_reference': Path(survival_reference),
             'plan': Path(plan_path) if plan_path else root/'plan.json',
             'comparison': root/'comparison.json'}
    for model in MODELS:
        paths[f'{model}.report'] = root/model/'report.json'
        for selection in ('best', 'last'):
            paths[f'{model}.{selection}.predictions'] = root/model/f'val_predictions_{selection}.csv'
            paths[f'{model}.{selection}.checkpoint_bytes'] = root/model/f'checkpoint_{selection}.pt'
    result = {'protocol': 'v1_comparison_saved_prediction_audit', 'status': 'failed',
              'counts': {}, 'rows': [], 'errors': [], 'input_sha256': {},
              'input_sha256_after': {},
              'limits': ['Supplied CSV/report consistency only; original Git ancestry is not established.',
                         'No features, RNA arrays or checkpoint objects are loaded.',
                         'No training or independent test; validation selection bias remains.',
                         'NLL is checked against history/summary, not reconstructed without logits.',
                         'A matching plan hash is not proof all upstream preprocessing was leakage-free.']}
    try:
        result['runtime'] = {'python': platform.python_version(), 'numpy': np.__version__,
                             'sksurv': importlib.metadata.version('scikit-survival')}
        result['audit_source_sha256'] = {name: _sha256(Path(__file__).resolve().parents[1]/name)
            for name in ('scripts/audit_v1_comparison_predictions.py', 'scripts/audit_parity_saved_predictions.py',
                         'cancer_parity.py', 'cancer_parity_audit.py')}
        for key, path in paths.items():
            result['input_sha256'][key] = _sha256(path)
        plan = json.loads(paths['plan'].read_text(encoding='utf-8-sig'))
        _require(plan['protocol'] == V1, 'Expected archived V1 plan, not a new full-dense V2 plan')
        _require(plan['settings']['patient_batch_size'] == 8 and plan['settings']['epochs'] == 30,
                 'Expected the archived batch8/30-epoch V1 settings')
        _require(all(plan['settings'][key] == value for key, value in
                     {'endpoint': 'DSS', 'time_unit': 'months', 'seed': 3,
                      'n_bins': 4, 'num_patches': 4096, 'alpha_surv': .5}.items()),
                 'Unexpected endpoint/seed/likelihood/patch settings for the reported V1 batch')
        for name in ('manifest', 'survival_reference'):
            _require(Path(plan[name]).resolve() == paths[name].resolve(), 'Explicit input path differs from plan')
            _require(plan['inputs']['file_sha256'][plan[name]] == result['input_sha256'][name],
                     'Manifest/reference hash differs from archived plan')
        patients, train_ids, val_ids = _read_manifest(paths['manifest'], result['counts'])
        reference = _read_reference(paths['survival_reference'], patients, result['counts'])
        _require(set(reference) == set(patients), 'Reference contains identities outside supplied train/val cohort')
        train_t, train_e = (np.array([reference[i][j] for i in train_ids]) for j in (0, 1))
        val_t, val_e = (np.array([reference[i][j] for i in val_ids]) for j in (0, 1))
        expected_counts = {'train': len(train_ids), 'val': len(val_ids)}
        expected_events = {'train': int(train_e.sum()), 'val': int(val_e.sum())}
        comparison = json.loads(paths['comparison'].read_text(encoding='utf-8-sig'))
        _require(comparison['plan_sha256'] == result['input_sha256']['plan'], 'Summary plan hash mismatch')
        summary_rows = comparison['rows']
        _require(len(summary_rows) == 4 and
                 {(r['model'], r['selection']) for r in summary_rows} ==
                 {(m, s) for m in MODELS for s in ('best', 'last')}, 'Expected exactly four V1 summary rows')
        sampling = set()
        for model in MODELS:
            report, edges = _read_report(paths[f'{model}.report'])
            _require(report['protocol'] == V1 and report['model_id'] == model, 'Unexpected report method/protocol')
            _require(report['plan_sha256'] == result['input_sha256']['plan'], 'Run plan hash mismatch')
            _require(report['settings'] == plan['settings'], 'Run settings differ from archived plan')
            _require(report['source_sha256'] == plan['source_sha256'], 'Report/plan source fingerprints differ')
            _require(report['epochs_completed'] == 30 and not report['smoke_only'] and not report['test_evaluated'],
                     'Incomplete, smoke or test-evaluated run cannot enter this V1 audit')
            _require(report['patients'] == expected_counts and report['events'] == expected_events,
                     'Report cohort counts differ from input labels')
            _require(np.array_equal(edges, np.array([float(x) for x in plan['inputs']['bin_edges']])),
                     'Report bin edges differ from plan')
            fitted_edges = dct_bin_edges(train_t, train_e, len(edges)-1)
            _require(np.array_equal(edges, fitted_edges), 'Archived edges do not equal train-only quantiles')
            history = report['history']
            _require([r['epoch'] for r in history] == list(range(1, 31)), 'Expected complete ordered 30-epoch history')
            valid = [(r['epoch'], r['c_index']) for r in history if r['c_index'] is not None]
            _require(bool(valid), 'No finite history C to select')
            peak = max(value for _, value in valid)
            _require(report['best_epoch'] == min(epoch for epoch, value in valid if value == peak),
                     'Best checkpoint is not the earliest maximum validation C')
            _require(report['last_epoch'] == 30, 'Last checkpoint must be epoch30')
            sampling.add(report['training_sampling_sha256'])
            for selection in ('best', 'last'):
                counts = {}
                survival, _, risk_delta = _read_predictions(
                    paths[f'{model}.{selection}.predictions'], val_ids, reference, edges, counts)
                metrics = _json_safe_metrics(parity_metrics(train_t, train_e, val_t, val_e, survival, edges))
                aligned = audit_parity_metrics(train_t, train_e, val_t, val_e, survival, edges)
                stored = report[f'{selection}_validation']
                epoch = report[f'{selection}_epoch']
                record = history[epoch-1]
                for name in ('c_index', 'c_index_ipcw', 'brier', 'ibs', 'iauc', 'metric_times'):
                    _require(_equal_metric(metrics[name], stored[name]),
                             f'{model}/{selection}: saved predictions do not reproduce {name}')
                _require(metrics['c_index'] is not None and
                         abs(metrics['c_index']-record['c_index']) <= C_ATOL,
                         'Selected/last history C differs from recomputed predictions')
                _require(_equal_metric(stored['validation_nll'], record['validation_nll']),
                         'Recorded validation NLL differs from selected/last history')
                summary = next(r for r in summary_rows if r['model'] == model and r['selection'] == selection)
                _require(summary['epoch'] == epoch and _equal_metric(summary['Harrell_C'], metrics['c_index'])
                         and _equal_metric(summary['IPCW_C'], metrics['c_index_ipcw'])
                         and _equal_metric(summary['NLL'], stored['validation_nll']), 'Summary/report mismatch')
                for column, metric in (('Brier', 'brier'), ('IBS_legacy_grid', 'ibs'),
                                       ('AUC_legacy_times', 'iauc'), ('metric_times', 'metric_times')):
                    _require(_equal_metric(summary[column], metrics[metric]), 'Summary/report metric mismatch')
                result['rows'].append({'model': model, 'selection': selection, 'epoch': epoch,
                    'risk_reconstruction_max_abs_error': risk_delta, 'legacy_metrics': metrics,
                    'finite_edge_audit': aligned, 'reported_validation_nll': stored['validation_nll'],
                    'nll_checked_against_history_and_summary_only': True})
        _require(len(sampling) == 1 and all(isinstance(x, str) and len(x) == 64 for x in sampling),
                 'The two arms do not have a valid common training-sampling fingerprint')
        result['plan_sha256'] = result['input_sha256']['plan']
        result['training_sampling_sha256'] = next(iter(sampling))
        result['legacy_scorer_bytes_match_plan'] = (
            _sha256(Path(__file__).resolve().parents[1]/'cancer_parity.py') == plan['source_sha256'].get('cancer_parity'))
        result['status'] = 'passed'
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        for key, old_hash in result['input_sha256'].items():
            try:
                new_hash = _sha256(paths[key])
                result['input_sha256_after'][key] = new_hash
                if new_hash != old_hash:
                    result['status'] = 'failed'
                    result['errors'].append(f'Input changed during audit: {key}')
            except OSError:
                result['status'] = 'failed'
                result['errors'].append(f'Input unavailable after audit: {key}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--survival-reference', required=True)
    parser.add_argument('--comparison-root', required=True)
    parser.add_argument('--plan', help='Defaults to comparison-root/plan.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    result = audit_comparison(args.manifest, args.survival_reference, args.comparison_root, args.plan)
    with Path(args.out).open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'rows': len(result['rows']), 'out': args.out}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
