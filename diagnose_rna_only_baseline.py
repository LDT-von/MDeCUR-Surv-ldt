"""Historical observed-follow-up-bin ridge exploration, NOT a survival NLL model.

RNA is standardized using training rows, then ridge least squares predicts a
one-hot observed follow-up bin, including censored rows. Those scores are mapped
to hazard-shaped outputs only for historical scoring compatibility. This is not
censoring-aware fitting or the parity hazard likelihood. Alpha is selected on
validation C, so the reported maximum is a selected development score.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cancer_parity import dct_bin_edges, discretize_labels, parity_metrics


RIDGE_SEMANTICS = {
    'model': 'historical_observed_followup_bin_ridge',
    'training_objective': 'L2-penalized least squares on observed-followup-bin one-hot labels',
    'censoring_aware_training': False,
    'nll_semantics': 'No survival NLL is fitted or reported; censored follow-up bins are historical regression targets.',
    'rna_preprocessing': 'Training-row mean/std standardization; std below 1e-8 replaced by one; no intercept.',
    'output_mapping': 'Historical regression scores treated as hazard logits; not calibrated by survival likelihood.',
    'selection': 'Maximum validation C over seven fixed alpha values; not an independent test result.',
    'historical_score_caution': 'An earlier 0.7472 from this procedure must not be called an NLL survival baseline.',
}


def metric_text(value):
    return 'NA' if value is None or not np.isfinite(value) else f'{value:.4f}'


def dual_ridge_logits(train_x, targets, val_x, alpha):
    """Equivalent to X(X'X+aI)^-1 without a genes-by-genes matrix."""
    x, y, v = map(lambda a: np.asarray(a, dtype=np.float64), (train_x, targets, val_x))
    if (x.ndim != 2 or y.ndim != 2 or v.ndim != 2 or len(x) != len(y)
            or x.shape[1] != v.shape[1] or not len(x) or not len(v)
            or not all(np.isfinite(a).all() for a in (x, y, v))
            or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError('Finite paired matrices and positive ridge alpha required')
    return (v @ x.T) @ np.linalg.solve(x @ x.T+alpha*np.eye(len(x)), y)


def write_result(destination, result):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--survival-reference', required=True)
    parser.add_argument('--n-bins', type=int, default=4)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    if args.out and Path(args.out).exists():
        raise FileExistsError(args.out)
    if args.n_bins < 2:
        raise ValueError('At least two bins required')
    manifest_path = Path(args.manifest)
    with manifest_path.open(encoding='utf-8-sig', newline='') as handle:
        manifest = list(csv.DictReader(handle))
    if any(row['split'].strip() not in ('train', 'val') for row in manifest):
        raise ValueError('This diagnostic accepts explicit train/val rows only')
    ids = [row['patient_id'].strip() for row in manifest]
    if any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Unique nonempty patient IDs required')
    reference = {}
    with Path(args.survival_reference).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            ident = row['patient_id'].strip()
            t, e = float(row['time']), float(row['event'])
            if (not ident or ident in reference or not np.isfinite(t) or t < 0 or e not in (0., 1.)
                    or row.get('split', 'train').strip() not in ('train', 'val')):
                raise ValueError('Invalid or duplicated train/val survival reference')
            reference[ident] = (t, e)
    if set(reference) != set(ids):
        raise ValueError('Survival reference must exactly cover the manifest')
    for row in manifest:
        t, e = reference[row['patient_id'].strip()]
        if (float(row['event']) != e or
                not np.isclose(float(row['time']), t, rtol=2e-7, atol=2e-6)):
            raise ValueError('Survival reference disagrees with manifest outcomes')
    train_rows = [row for row in manifest if row['split'].strip() == 'train']
    val_rows = [row for row in manifest if row['split'].strip() == 'val']
    if not train_rows or not val_rows:
        raise ValueError('Both train and validation must be nonempty')
    def labels(rows, column):
        return np.array([reference[r['patient_id'].strip()][column] for r in rows])
    train_time, train_event = labels(train_rows, 0), labels(train_rows, 1)
    val_time, val_event = labels(val_rows, 0), labels(val_rows, 1)
    edges = dct_bin_edges(train_time, train_event, args.n_bins)
    train_onehot = np.eye(args.n_bins)[discretize_labels(train_time, edges)]
    def load(rows):
        values = np.stack([np.load(manifest_path.parent/row['rna_npy'], allow_pickle=False) for row in rows])
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError('Finite one-dimensional RNA vectors required')
        return values.astype(np.float64)
    train_x, val_x = load(train_rows), load(val_rows)
    mean, scale = train_x.mean(0), train_x.std(0)
    scale[scale < 1e-8] = 1.0
    train_x, val_x = (train_x-mean)/scale, (val_x-mean)/scale
    best, sweep = None, []
    print('Historical observed-bin ridge; NOT a censoring-aware survival NLL baseline.')
    for alpha in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8):
        logits = dual_ridge_logits(train_x, train_onehot, val_x, alpha)
        hazards = np.exp(-np.logaddexp(0., -logits))
        survival = np.cumprod(1.0-hazards, axis=1)
        metrics = parity_metrics(train_time, train_event, val_time, val_event, survival, edges)
        sweep.append({'alpha': alpha, **metrics})
        print(f"alpha {alpha:.0f}  C={metric_text(metrics['c_index'])}  "
              f"IPCW={metric_text(metrics['c_index_ipcw'])}  IBS={metric_text(metrics['ibs'])}  "
              f"IAUC={metric_text(metrics['iauc'])}")
        value = metrics['c_index']
        if value is not None and np.isfinite(value) and (best is None or value > best[1]):
            best = (alpha, value)
    print('Selected validation C='+metric_text(None if best is None else best[1])+
          '; alpha='+('NA' if best is None else str(best[0])))
    result = {**RIDGE_SEMANTICS, 'sweep': sweep,
              'best_alpha': None if best is None else best[0],
              'best_c_index': None if best is None else best[1],
              'linear_solver': 'dual ridge (patients-by-patients)',
              'train_patients': len(train_rows), 'validation_patients': len(val_rows)}
    if args.out:
        write_result(args.out, result)


if __name__ == '__main__':
    main()
