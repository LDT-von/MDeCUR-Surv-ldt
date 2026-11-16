"""Merge the fold0 project arms with the upstream comparator arms into one table.

Two runners produce the two groups of arms:

* ``comparison_fold0.py`` runs the project arms (``mdecur_full_dense``,
  ``mdecur_dense_aux_off``, ``camr_2m``, ``camr_2m_official``,
  ``morphid``, ``morphid_no_interaction``) under protocol
  ``blca_fold0_full_dense_patientwise_v3``.
* ``comparison_5fold.py`` runs the upstream comparator arms (``pibd_2m``,
  ``camr_2m``, ``deref_2m``, ``dimaf_2m``) under protocol
  ``blca_1fold_dss_uni_v1_p4096_dnll_v1``.  The upstream arms are run on the
  **same single fold 0** as the project arms; the runner name and the protocol
  identifier are historical, the fold scope is one fold.

The two runners share the endpoint, the unit, the binning function, the patch
budget, the patch resampling seed, the training seed, the survival likelihood
version and the metric implementation, but they do **not** share every detail:
the upstream runner does not merge a short tail batch and it computes its own
validation NLL from ``log(1 - S)`` instead of from the hazard logits.

This module therefore never reads a metric from one runner and places it next
to a metric from the other one.  It recomputes every comparable number from
the saved validation survival matrices with one shared scoring call, so the
merged table is scored by exactly one implementation.

``camr_2m`` exists in both runners.  It is used as a bridge arm: the two rows
of the same method under the same data quantify the runner-to-runner offset
that the merged table inherits.

The merge refuses to publish when the two plans do not describe the same
patients, the same labels or the same bins.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from cancer_parity import (discrete_nll, discretize_labels, dct_bin_edges,
                           parity_metrics, DISCRETE_NLL_VERSION)
from comparison_5fold import PROTOCOL as FIVEFOLD_PROTOCOL
from comparison_5fold import PROTOCOL_VERSION as FIVEFOLD_PROTOCOL_VERSION
from comparison_fold0 import PROTOCOL as FOLD0_PROTOCOL

FOLD0_RUNS = ('mdecur_full_dense', 'mdecur_dense_aux_off', 'camr_2m',
               'camr_2m_official', 'morphid', 'morphid_no_interaction')
UPSTREAM_RUNS = ('pibd_2m', 'camr_2m', 'deref_2m', 'dimaf_2m')
BRIDGE_ARM = 'camr_2m'

METRIC_TOLERANCE = 1e-6
NLL_TOLERANCE = 1e-3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalize_edges(edges):
    """Map either runner's edge encoding onto plain floats.

    The fold0 runner serializes an infinite edge as the string ``Infinity``
    and the upstream runner serializes every infinite edge as ``None``.  In
    the upstream encoding the first edge is the open lower bound and the last
    edge is the open upper bound, so the position decides the sign.
    """
    values = list(edges)
    result = []
    for index, value in enumerate(values):
        if value is None:
            result.append(-math.inf if index == 0 else math.inf)
        elif value == 'Infinity':
            result.append(math.inf)
        elif value == '-Infinity':
            result.append(-math.inf)
        else:
            result.append(float(value))
    if len(result) < 2:
        raise ValueError('A survival bin specification needs at least two edges')
    if not math.isinf(result[0]) or result[0] > 0:
        raise ValueError('The first bin edge must be the open lower bound')
    if not math.isinf(result[-1]) or result[-1] < 0:
        raise ValueError('The last bin edge must be the open upper bound')
    return result


def edge_signature(edges):
    return [('inf' if math.isinf(x) else round(float(x), 12)) for x in normalize_edges(edges)]


def plan_identity(plan):
    """The comparable identity of one plan.

    Only the fields that describe the data and the labelling are used.  The
    protocol strings, the code hashes and the plan file hashes differ between
    the runners by design and are deliberately excluded.
    """
    inputs = plan['inputs']
    labels = inputs.get('double_labels_sha256') or inputs.get('labels_sha256')
    return {'patients': inputs['patients'], 'events': inputs['events'],
            'order_sha256': inputs['order_sha256'], 'labels_sha256': labels,
            'bin_edges': edge_signature(inputs['bin_edges'])}


def compare_identities(fold0_plan, fivefold_plan):
    """Refuse a merge whose two plans do not describe the same cohort."""
    left, right = plan_identity(fold0_plan), plan_identity(fivefold_plan)
    differences = []
    for key in ('patients', 'events', 'order_sha256', 'labels_sha256', 'bin_edges'):
        if left[key] != right[key]:
            differences.append({'field': key, 'fold0': left[key], 'fivefold': right[key]})
    if differences:
        raise ValueError('Refuse to merge two plans that describe different data: '
                         + json.dumps(differences, ensure_ascii=False))
    return left


def read_survival_reference(path):
    values = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            identity = (row.get('patient_id') or '').strip()
            split = (row.get('split') or 'train').strip().lower()
            if not identity or identity in values:
                raise ValueError('Survival reference requires unique nonempty patient IDs')
            if split not in ('train', 'val'):
                raise ValueError('Survival reference must contain train/val rows only')
            time, event = float(row['time']), float(row['event'])
            if not math.isfinite(time) or time < 0 or event not in (0., 1.):
                raise ValueError('Invalid double-precision survival reference')
            values[identity] = (split, time, event)
    return values


def read_predictions(path):
    """Read a saved validation prediction file.

    Returns the patient identities in file order and the survival matrix.
    """
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        try:
            first = header.index('patient_id')
        except ValueError:
            raise ValueError('Prediction file must carry a patient_id column')
        columns = [index for index, name in enumerate(header) if name.startswith('survival_bin_')]
        if not columns:
            raise ValueError('Prediction file must carry survival_bin_ columns')
        ids, rows = [], []
        for row in reader:
            if not row:
                continue
            ids.append(row[first])
            rows.append([float(row[index]) for index in columns])
    survival = np.asarray(rows, dtype=np.float64)
    if survival.ndim != 2 or not np.isfinite(survival).all():
        raise ValueError('Survival matrix must be finite')
    if np.any((survival < 0) | (survival > 1)):
        raise ValueError('Survival probabilities must lie in [0, 1]')
    if np.any(np.diff(survival, axis=1) > 1e-9):
        raise ValueError('Cumulative survival must be nonincreasing across columns')
    return ids, survival


def logits_from_survival(survival):
    """Recover the hazard logits that produced a survival matrix.

    ``discrete_survival`` is a cumulative product of ``1 - sigmoid(logit)``, so
    the inversion is exact up to the precision of the stored file.  The
    recovered logits let the shared likelihood be applied to every arm,
    including the arms whose own runner scored a different NLL.
    """
    survival = np.asarray(survival, dtype=np.float64)
    previous = np.concatenate([np.ones((survival.shape[0], 1)), survival[:, :-1]], axis=1)
    hazards = 1.0 - survival/np.clip(previous, 1e-300, None)
    hazards = np.clip(hazards, 1e-7, 1.0-1e-7)
    logits = np.log(hazards) - np.log1p(-hazards)
    return torch.as_tensor(logits, dtype=torch.float64)


def score_arm(ids, survival, reference, edges, train_ids):
    """Score one arm with the shared metric call.

    The scoring is identical for every arm of both runners, so the merged
    table never mixes two metric implementations.
    """
    labels = {}
    for identity in ids:
        if identity not in reference:
            raise ValueError(f'Prediction patient {identity} is not in the survival reference')
        labels[identity] = reference[identity]
    val_time = np.asarray([labels[i][1] for i in ids], dtype=np.float64)
    val_event = np.asarray([labels[i][2] for i in ids], dtype=np.float64)
    train_time = np.asarray([reference[i][1] for i in train_ids], dtype=np.float64)
    train_event = np.asarray([reference[i][2] for i in train_ids], dtype=np.float64)
    metrics = parity_metrics(train_time, train_event, val_time, val_event, survival, edges)
    bins = discretize_labels(val_time, edges)
    logits = logits_from_survival(survival)
    censorship = torch.as_tensor(1.0-val_event, dtype=torch.float64)
    metrics['validation_nll'] = float(discrete_nll(logits, torch.as_tensor(bins),
                                                     censorship, alpha=.5))
    metrics['nll_definition'] = ('patientwise discrete NLL recomputed from the saved '
                                 'survival matrix through the recovered hazard logits')
    return metrics


def stored_metrics(report, selection):
    key = 'best_validation' if selection == 'best' else 'last_validation'
    block = report.get(key) or {}
    return {'c_index': block.get('c_index'), 'c_index_ipcw': block.get('c_index_ipcw'),
            'validation_nll': block.get('validation_nll'), 'brier': block.get('brier'),
            'ibs': block.get('ibs'), 'iauc': block.get('iauc'),
            'errors': block.get('errors', []), 'metric_times': block.get('metric_times', [])}


def compare_metrics(recomputed, stored):
    """Report every disagreement between the shared scoring and the runner.

    A disagreement in a ranking metric means the two scorers did not see the
    same prediction, so the merge must not publish a merged table silently.
    """
    problems = []
    for key in ('c_index', 'c_index_ipcw', 'iauc'):
        left, right = recomputed.get(key), stored.get(key)
        if left is None or right is None:
            continue
        if abs(float(left)-float(right)) > METRIC_TOLERANCE:
            problems.append({'metric': key, 'recomputed': float(left),
                             'runner': float(right)})
    for key in ('brier', 'ibs'):
        left, right = recomputed.get(key), stored.get(key)
        if left is None or right is None:
            continue
        if isinstance(left, list) or isinstance(right, list):
            if np.max(np.abs(np.asarray(left, dtype=float)
                             -np.asarray(right, dtype=float))) > METRIC_TOLERANCE:
                problems.append({'metric': key, 'recomputed': left, 'runner': right})
            continue
        if abs(float(left)-float(right)) > METRIC_TOLERANCE:
            problems.append({'metric': key, 'recomputed': float(left),
                             'runner': float(right)})
    return problems


def collect_arm(directory, group, reference, edges, train_ids):
    """Collect every recomputed row of one run directory."""
    directory = Path(directory)
    report = read_json(directory/'report.json')
    if report.get('smoke_only') or report.get('test_evaluated'):
        raise ValueError(f'Refuse smoke or test-evaluated run: {directory}')
    if report.get('outer_test_used'):
        raise ValueError(f'Refuse a run that used the outer test partition: {directory}')
    if report.get('loss_version') != DISCRETE_NLL_VERSION:
        raise ValueError(f'Refuse a run under another likelihood version: {directory}')
    model_id = report['model_id']
    rows = []
    for selection in ('best', 'last'):
        predictions = directory/f'val_predictions_{selection}.csv'
        if not predictions.is_file():
            continue
        ids, survival = read_predictions(predictions)
        recomputed = score_arm(ids, survival, reference, edges, train_ids)
        stored = stored_metrics(report, selection)
        problems = compare_metrics(recomputed, stored)
        rows.append({'model': model_id, 'group': group, 'selection': selection,
                     'epoch': report.get(f'{selection}_epoch'),
                     'epochs_completed': report.get('epochs_completed'),
                     'Harrell_C': recomputed['c_index'],
                     'IPCW_C': recomputed['c_index_ipcw'],
                     'NLL': recomputed['validation_nll'],
                     'NLL_definition': recomputed['nll_definition'],
                     'Brier': recomputed['brier'],
                     'IBS_legacy_grid': recomputed['ibs'],
                     'AUC_legacy_times': recomputed['iauc'],
                     'metric_times': recomputed['metric_times'],
                     'errors': recomputed['errors'],
                     'runner_protocol': report.get('protocol'),
                     'runner_loss_version': report.get('loss_version'),
                     'runner_reported_NLL': stored['validation_nll'],
                     'runner_metric_mismatch': problems,
                     'parameters': report.get('parameters'),
                     'wall_seconds': report.get('wall_seconds'),
                     'evaluation_patients': len(ids)})
    return rows


def merge(fold0_plan_path, fold0_runs, fivefold_plan_path, fivefold_runs,
          survival_reference_path, out, required_epochs=30):
    """Build the merged ranking table."""
    fold0_plan = read_json(fold0_plan_path)
    fivefold_plan = read_json(fivefold_plan_path)
    identity = compare_identities(fold0_plan, fivefold_plan)
    reference = read_survival_reference(survival_reference_path)
    edges = dct_bin_edges(*_train_outcomes(reference), 4)
    if edge_signature(edges) != identity['bin_edges']:
        raise ValueError('The recomputed train-fitted bin edges differ from the plans')
    train_ids = sorted(p for p, value in reference.items() if value[0] == 'train')
    rows = []
    for directory in fold0_runs:
        rows.extend(collect_arm(directory, 'project', reference, edges, train_ids))
    for directory in fivefold_runs:
        rows.extend(collect_arm(directory, 'upstream', reference, edges, train_ids))
    if not rows:
        raise ValueError('No arms were collected')
    seen = {}
    for row in rows:
        key = (row['group'], row['model'], row['selection'])
        if key in seen:
            raise ValueError(f'Duplicate arm row: {key}')
        seen[key] = row
    # The bridge arm is produced by both runners, so every row carries a unique
    # label while ``model`` keeps the method name.
    for row in rows:
        row['arm'] = row['model'] if row['group'] == 'project' else f"{row['model']} [upstream runner]"
    epoch_counts = {row['epochs_completed'] for row in rows}
    if len(epoch_counts) != 1:
        raise ValueError('Refuse to merge arms that ran a different number of epochs: '
                         + json.dumps(sorted(epoch_counts, key=lambda v: (v is None, v)),
                                      ensure_ascii=False))
    if epoch_counts != {required_epochs}:
        raise ValueError(f'Every merged arm must have run {required_epochs} epochs, '
                         f'not {sorted(epoch_counts)}')
    mismatches = [row for row in rows if row['runner_metric_mismatch']]
    if mismatches:
        raise ValueError('Refuse to publish: the shared scoring disagrees with a '
                         'runner on ' + json.dumps(
                             [{'model': r['model'], 'selection': r['selection'],
                               'problems': r['runner_metric_mismatch']} for r in mismatches],
                             ensure_ascii=False))
    rankings = {}
    for selection in ('best', 'last'):
        subset = [row for row in rows if row['selection'] == selection
                  and row['Harrell_C'] is not None]
        rankings[selection] = [row['arm'] for row in
                                sorted(subset, key=lambda r: -float(r['Harrell_C']))]
    bridge = _bridge_offsets(rows)
    result = {
        'merged_table_version': 'fold0_plus_upstream_merged_v1',
        'fold0_protocol': FOLD0_PROTOCOL,
        'upstream_protocol': FIVEFOLD_PROTOCOL,
        'upstream_protocol_version': FIVEFOLD_PROTOCOL_VERSION,
        'upstream_runner_note': ('The upstream arms are run by '
                                 '``comparison_5fold.py``. The runner name and '
                                 'the protocol identifier are historical: the '
                                 'upstream arms run on the same single fold 0 '
                                 'as the project arms.'),
        'loss_version': DISCRETE_NLL_VERSION,
        'identity': identity,
        'rows': rows,
        'Harrell_C_rankings': rankings,
        'bridge_arm': bridge,
        'metric_directions': {'Harrell_C': 'higher', 'IPCW_C': 'higher',
                              'AUC_legacy_times': 'higher', 'NLL': 'lower',
                              'Brier': 'lower', 'IBS_legacy_grid': 'lower'},
        'scoring_note': (
            'Every row was rescored from the saved validation survival matrix '
            'with one shared metric call, so the two runners never contribute '
            'two different metric implementations to the same column. The NLL '
            'column is the patientwise discrete NLL recomputed from the '
            'recovered hazard logits; a runner-native NLL is kept beside it '
            'for auditing.'),
        'bridge_arm_note': (
            'camr_2m was run by both runners on the same patients, so the two '
            'rows of that arm measure the runner-to-runner offset that every '
            'cross-group comparison inherits.'),
        'comparability_limits': [
            'The two runners share the endpoint, unit, binning function, patch '
            'budget, resampling seed, training seed, likelihood version and '
            'metric implementation, but not the batch tail rule and not the '
            'runner-native NLL definition.',
            'The upstream runner does not merge a short tail batch, so its '
            'arms take one optimizer step more per epoch than the fold0 arms.',
            'Every number is a validation-selected point estimate on one fold '
            'with no significance test and no claim of superiority.',
            'Adapted comparator arms are not reproductions of their original '
            'papers, and no arm score may be compared with a published table.',
            'The archived DCT five-fold reference is not part of this table: '
            'it was trained under the pre-fix broadcast likelihood.'],
    }
    write_new(out, result)
    return result


def _train_outcomes(reference):
    train = sorted(p for p, v in reference.items() if v[0] == 'train')
    times = np.asarray([reference[p][1] for p in train], dtype=np.float64)
    events = np.asarray([reference[p][2] for p in train], dtype=np.float64)
    return times, events


def _bridge_offsets(rows):
    """Quantify the runner-to-runner offset using the shared bridge arm."""
    result = {'arm': BRIDGE_ARM, 'available': False}
    for selection in ('best', 'last'):
        entries = [row for row in rows
                   if row['model'] == BRIDGE_ARM and row['selection'] == selection]
        if len(entries) != 2:
            continue
        project = next(r for r in entries if r['group'] == 'project')
        upstream = next(r for r in entries if r['group'] == 'upstream')
        if project['Harrell_C'] is None or upstream['Harrell_C'] is None:
            continue
        result[selection] = {
            'project_arm': project['arm'], 'upstream_arm': upstream['arm'],
            'project_Harrell_C': project['Harrell_C'],
            'upstream_Harrell_C': upstream['Harrell_C'],
            'delta_Harrell_C': float(project['Harrell_C'])-float(upstream['Harrell_C']),
            'project_NLL': project['NLL'], 'upstream_NLL': upstream['NLL'],
            'delta_NLL': float(project['NLL'])-float(upstream['NLL']),
            'project_epoch': project['epoch'], 'upstream_epoch': upstream['epoch']}
        result['available'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold0-plan', required=True)
    parser.add_argument('--fivefold-plan', required=True)
    parser.add_argument('--fold0-runs', nargs='+', required=True)
    parser.add_argument('--fivefold-runs', nargs='+', required=True)
    parser.add_argument('--survival-reference', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--required-epochs', type=int, default=30,
                        help='Every merged arm must have completed this many epochs')
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    result = merge(args.fold0_plan, args.fold0_runs, args.fivefold_plan,
                   args.fivefold_runs, args.survival_reference, args.out,
                   args.required_epochs)
    print(json.dumps({'out': args.out,
                      'arms': len({row['model'] for row in result['rows']}),
                      'rankings_best': result['Harrell_C_rankings']['best']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
