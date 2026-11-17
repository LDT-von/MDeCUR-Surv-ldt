"""Read-only audit of the finished one-fold BLCA comparison batch.

The audit answers the questions that decide whether a ranking may be read as a
result, and it separates the three effects that the published table mixes:

1. **Statistical resolution.**  The validation partition holds 77 patients and
   27 events.  The audit bootstraps the Harrell C over those patients and
   reports the interval width, so a gap between two arms can be compared with
   the width of the noise it is measured in.
2. **Selection inflation.**  The published numbers are the maximum over 30
   validation epochs.  The audit recomputes a **pure-noise null** for the same
   cohort and the same number of epochs: random risk vectors are scored with
   the same metric call.  The maximum over 30 such epochs is not 0.5, and the
   audit reports where each arm's epoch mean and epoch maximum sit against
   that null.  An arm whose epoch mean is inside the null carries no signal;
   an arm whose epoch maximum is high but whose epoch mean is not has a score
   that is mostly selection.
3. **Reproducibility.**  The upstream arms were run twice with the same seed.
   The audit compares the two runs and reports the per-arm agreement, so a
   number that moves between identical runs is flagged instead of published.

The script never trains, never writes into a run directory and never opens
the outer test partition.  It only reads finished artefacts.

Usage:
    python scripts/audit_onefold_comparison.py \
        --batch /data1/decur_runs/BLCA_FOLD0_RANKING_20261116 \
        --survival-reference /data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv \
        --out results/onefold_comparison_audit_20261117.json
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import comparison_merge as merge
import comparison_sources as sources
from cancer_parity import dct_bin_edges, dct_risk


def read_reference(path):
    reference = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            reference[row['patient_id']] = (row['split'], float(row['time']),
                                            float(row['event']))
    return reference


def bootstrap_c_index(ids, survival, reference, trials=2000, seed=20261117):
    """Bootstrap the Harrell C over the validation patients.

    The patients are resampled with replacement; the censoring reference and
    the bin edges stay fixed, because they are fitted on the training rows and
    do not depend on the validation sample.
    """
    from sksurv.metrics import concordance_index_censored

    labels = [reference[i] for i in ids]
    times = np.asarray([row[1] for row in labels], dtype=np.float64)
    events = (np.asarray([row[2] for row in labels], dtype=np.float64) > .5).astype(bool)
    risk = dct_risk(survival)
    generator = np.random.default_rng(seed)
    values = []
    n = len(ids)
    for _ in range(trials):
        picks = generator.integers(0, n, size=n)
        picked_events = events[picks]
        if picked_events.sum() < 2 or picked_events.all():
            continue
        try:
            value = concordance_index_censored(picked_events, times[picks],
                                                risk[picks], tied_tol=1e-8)[0]
        except Exception:
            continue
        values.append(float(value))
    if not values:
        return None
    values.sort()
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    return {'trials': len(values), 'mean': statistics.fmean(values),
            'sd': statistics.pstdev(values), 'ci95': [lower, upper]}


def noise_null(n_patients, events, times, trials=1000, epochs=30, seed=20261117):
    """The null of the whole protocol: random risk, the same metric, the same
    number of validation-selected epochs.

    Two statistics are reported, because the published table uses the second
    one and the first one is the honest summary of the run:

    * ``epoch_mean``: the mean Harrell C over the epochs.  A model that
      carries no signal sits at the chance level here.
    * ``epoch_max``: the maximum over the epochs, which is what the
      validation-selected score is.  Even a pure random risk reaches a high
      maximum, because the maximum of 30 noisy draws is biased upwards.
    """
    from sksurv.metrics import concordance_index_censored

    generator = np.random.default_rng(seed)
    means, maxes = [], []
    for _ in range(trials):
        scores = []
        for _ in range(epochs):
            risk = generator.random(n_patients)
            try:
                scores.append(float(concordance_index_censored(events, times, risk,
                                                                tied_tol=1e-8)[0]))
            except Exception:
                continue
        if not scores:
            continue
        means.append(statistics.fmean(scores))
        maxes.append(max(scores))
    if not means:
        return None
    means = np.asarray(means)
    maxes = np.asarray(maxes)
    return {'trials': len(means), 'epochs': epochs,
            'epoch_mean': {'mean': float(means.mean()), 'sd': float(means.std()),
                           'p95': float(np.percentile(means, 95)),
                           'p99': float(np.percentile(means, 99))},
            'epoch_max': {'mean': float(maxes.mean()), 'sd': float(maxes.std()),
                          'p95': float(np.percentile(maxes, 95)),
                          'p99': float(np.percentile(maxes, 99))}}


def epoch_statistics(history):
    """The epoch mean, spread and maximum of one arm's validation C-index."""
    values = [entry.get('c_index') for entry in history or []
              if entry.get('c_index') is not None]
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return {'epochs': len(values), 'mean_c_index': mean,
            'sd_c_index': statistics.pstdev(values),
            'min_c_index': min(values), 'max_c_index': max(values),
            'last_c_index': values[-1],
            'selection_inflation': max(values) - mean}


def reproducibility(batch, arms, reference, edges, train_ids):
    """Compare two identical-seed runs of the same upstream arm."""
    from scipy.stats import spearmanr

    repeat_root = batch/'upstream_v1_only_best'
    if not repeat_root.is_dir():
        return None
    result = {}
    for directory in sorted(batch.glob('upstream_*')):
        if not directory.is_dir() or not (directory/'val_predictions_best.csv').is_file():
            continue
        arm = directory.name
        first = repeat_root/directory.name/'val_predictions_best.csv'
        if not first.is_file():
            continue
        try:
            ids_a, survival_a = merge.read_predictions(first)
            ids_b, survival_b = merge.read_predictions(directory/'val_predictions_best.csv')
        except Exception as error:
            result[arm] = {'error': f'{type(error).__name__}: {error}'}
            continue
        if ids_a != ids_b:
            result[arm] = {'error': 'the two runs scored different patients'}
            continue
        difference = float(np.abs(np.asarray(survival_a) - np.asarray(survival_b)).max())
        rho = float(spearmanr(dct_risk(survival_a), dct_risk(survival_b)).statistic)
        result[arm] = {'max_abs_survival_difference': difference,
                       'risk_rank_spearman': rho,
                       'bit_identical': difference == 0.0}
    return result or None


def fold_event_profile(splits_root):
    """The event profile of every legacy split's validation partition.

    The IPCW C and the dynamic AUC of a fold are decided by the late events:
    the inverse-probability-of-censoring weight grows with the follow-up
    time, so the two or three latest validation events can flip the sign of
    the metric.  This profile says how many late events each fold has, so a
    fold whose IPCW C is unusual can be checked against the others instead of
    being read as a model property.
    """
    import pandas as pd

    clinical = Path('/data1/dataset_csv/clinical/all/blca.csv')
    if not clinical.is_file() or not Path(splits_root).is_dir():
        return None
    table = pd.read_csv(clinical).rename(columns={'case id': 'pid'})
    table['event_dss'] = 1-table['censorship_dss']
    table['pid'] = table['pid'].astype(str).str.slice(0, 12)
    table = table.drop_duplicates(subset=['pid'])
    profile = {}
    for path in sorted(Path(splits_root).glob('fold_*.csv')):
        split = pd.read_csv(path)
        validation = set(split['val'].astype(str).str.slice(0, 12))
        rows = table[table['pid'].isin(validation)]
        if rows.empty:
            continue
        late = int(((rows['event_dss'] == 1) & (rows['survival_months_dss'] > 60)).sum())
        profile[path.stem] = {
            'validation_patients': int(len(rows)),
            'events': int(rows['event_dss'].sum()),
            'event_rate': float(rows['event_dss'].mean()),
            'median_dss_months': float(rows['survival_months_dss'].median()),
            'events_beyond_60_months': late,
        }
    return profile or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--survival-reference', required=True)
    parser.add_argument('--merged', default='merged_ranking.json')
    parser.add_argument('--splits-root', default='/data1/dataset_splits_summary/uni/blca')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    batch = Path(args.batch)
    reference = read_reference(args.survival_reference)
    train_ids = [k for k, v in reference.items() if v[0] == 'train']
    val_ids = [k for k, v in reference.items() if v[0] == 'val']
    edges = dct_bin_edges(np.asarray([reference[i][1] for i in train_ids]),
                           np.asarray([reference[i][2] for i in train_ids]), 4)
    val_times = np.asarray([reference[i][1] for i in val_ids], dtype=np.float64)
    val_events = (np.asarray([reference[i][2] for i in val_ids], dtype=np.float64) > .5).astype(bool)

    merged_path = batch/args.merged
    merged = json.loads(merged_path.read_text(encoding='utf-8')) if merged_path.is_file() else None

    null = noise_null(len(val_ids), val_events, val_times)

    arms = {}
    for directory in sorted(batch.iterdir()):
        report_path = directory/'report.json'
        if not report_path.is_file() or not directory.is_dir():
            continue
        report = json.loads(report_path.read_text(encoding='utf-8'))
        entry = {'model_id': report.get('model_id'), 'group': 'project',
                 'epochs_completed': report.get('epochs_completed'),
                 'epoch_statistics': epoch_statistics(report.get('history'))}
        for selection in ('best', 'last'):
            predictions = directory/f'val_predictions_{selection}.csv'
            if not predictions.is_file():
                continue
            ids, survival = merge.read_predictions(predictions)
            recomputed = merge.score_arm(ids, survival, reference, edges, train_ids)
            entry[selection] = {
                'Harrell_C': recomputed['c_index'],
                'IPCW_C': recomputed['c_index_ipcw'],
                'NLL': recomputed['validation_nll'],
                'IBS': recomputed['ibs'], 'AUC': recomputed['iauc'],
                'epoch': report.get(f'{selection}_epoch'),
                'events_in_validation': int(sum(1 for i in ids if reference[i][2] > .5)),
                'patients_in_validation': len(ids),
                'bootstrap_Harrell_C': bootstrap_c_index(ids, survival, reference),
            }
        stats = entry.get('epoch_statistics')
        if stats and null:
            entry['epoch_mean_vs_null_p99'] = (
                stats['mean_c_index'] > null['epoch_mean']['p99'])
            entry['epoch_max_vs_null_p99'] = (
                stats['max_c_index'] > null['epoch_max']['p99'])
        arms[directory.name] = entry

    widths = []
    for entry in arms.values():
        for selection in ('best', 'last'):
            block = entry.get(selection) or {}
            boot = block.get('bootstrap_Harrell_C')
            if boot:
                widths.append(boot['ci95'][1] - boot['ci95'][0])
    resolution = {'widest_bootstrap_CI_width': max(widths) if widths else None,
                  'mean_bootstrap_CI_width': statistics.fmean(widths) if widths else None}

    gaps = {}
    for selection in ('best', 'last'):
        scored = [(name, (entry.get(selection) or {}).get('Harrell_C'))
                  for name, entry in arms.items()
                  if (entry.get(selection) or {}).get('Harrell_C') is not None]
        scored.sort(key=lambda item: -item[1])
        if len(scored) > 1:
            gaps[selection] = {'top': scored[0][0], 'runner_up': scored[1][0],
                               'gap': scored[0][1] - scored[1][1]}

    result = {
        'audit_version': 'onefold_comparison_audit_v2',
        'batch': str(batch.resolve()),
        'survival_reference': str(Path(args.survival_reference).resolve()),
        'validation_events': int(val_events.sum()),
        'validation_patients': len(val_ids),
        'train_patients': len(train_ids),
        'bin_edges': [None if (isinstance(x, float) and np.isinf(x)) else x
                      for x in edges.tolist()],
        'resolution': resolution,
        'top_gaps': gaps,
        'pure_noise_null': null,
        'fold_event_profile': fold_event_profile(args.splits_root),
        'arms': arms,
        'reproducibility': reproducibility(batch, arms, reference, edges, train_ids),
        'upstream_source_verification': {
            name: sources.verify_upstream(name)[name] for name in sources.UPSTREAM},
        'merged_table_present': merged is not None,
        'limits': [
            'This audit only reads the finished artefacts; it trains nothing.',
            'The outer test partition is never opened.',
            'The bootstrap resamples the validation patients only; the censoring '
            'reference and the bin edges stay fixed because they are fitted on the '
            'training rows.',
            'A bootstrap interval wider than the gap between two arms means the '
            'table cannot rank those two arms.',
            'The published arm scores are validation-selected maxima over 30 '
            'epochs. The pure-noise null shows how large that selection bias is '
            'for this cohort: a random risk vector already reaches the reported '
            'epoch-maximum p95 after 30 epochs.',
            'The epoch mean over the 30 epochs is the selection-free summary of '
            'one arm and is the statistic to compare against the null.',
            'The IPCW C and the dynamic AUC are decided by the few latest '
            'validation events, because the inverse-probability-of-censoring '
            'weight grows with the follow-up time. The fold event profile '
            'reports how many events lie beyond 60 months in every legacy '
            'split, so an unusual IPCW C can be checked against the other '
            'folds instead of being read as a model property.',
            'No arm score may be compared with a published table.',
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                   encoding='utf-8')
    print(json.dumps({'action': 'audit', 'out': str(out),
                      'validation_events': result['validation_events'],
                      'noise_null': null,
                      'resolution': resolution}, ensure_ascii=False))


if __name__ == '__main__':
    main()
