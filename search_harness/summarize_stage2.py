"""Stage 2 summary: per-fold metrics and pooled out-of-fold C-index.

The pooled out-of-fold C-index is computed by concatenating the five folds'
validation predictions, so every one of the 304 development patients is scored
exactly once by a model that never saw her during training.  This is the only
quantity in the search with a standard error small enough (~0.011) to separate
configurations; the single-fold Stage 1 C-index (n=61, SE ~0.03) cannot.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from search_config import (REPO as _REPO, RUN as _RUN, FOLD as _FOLD,
                           FOLD_DIR as _FOLD_DIR, GENES as _GENES, PY as _PY,
                           MEASURE as _MEASURE, LD_LIBRARY as _LD_LIBRARY,
                           PATCH_CHUNK_SIZE as _PATCH_CHUNK_SIZE)
_sys.path.insert(0, str(_REPO))

import csv
import json
import os
from pathlib import Path

import numpy as np

from cancer_metrics import harrell_c, ipcw_brier

RUN = _RUN
STAGE2 = RUN / 'stage2'
BASELINE_RUN = Path(os.environ.get('SEARCH_BASELINE_RUN', '/data1/decur_runs/DEV_FIVE_FOLD_CINDEX_20261112'))


def read_fold(path):
    report = path / 'report.json'
    predictions = path / 'val_predictions.csv'
    if not report.exists() or not predictions.exists():
        return None
    data = json.loads(report.read_text(encoding='utf-8'))
    rows = list(csv.DictReader(predictions.open()))
    return {
        'fold': int(path.name.split('_')[-1]),
        'validation_nll': data['validation_nll'],
        'c_index': data['evaluation_c_index_risk_at_horizon']['value'],
        'ipcw_brier': data['evaluation_ipcw_brier_at_horizon']['value'],
        'best_epoch': data['best_epoch'],
        'epochs_run': len(data['history']),
        'val_patients': data['patients']['val'],
        'val_events': data['events']['val'],
        'train_patients': data['patients']['train'],
        'wall_seconds': data['runtime']['wall_seconds'],
        'rows': rows,
    }


def load_labels():
    labels = {}
    for fold in range(5):
        manifest = Path(os.environ.get('SEARCH_FOLD_DIR', str(_FOLD_DIR))) \
            / f'fold_{fold}.csv'
        for row in csv.DictReader(manifest.open()):
            labels[row['patient_id']] = (float(row['time']), int(row['event']))
    return labels


def pooled(folds, labels):
    times, events, risks, survivals, ids = [], [], [], [], []
    for fold in folds:
        for row in fold['rows']:
            identity = row['patient_id']
            times.append(labels[identity][0])
            events.append(labels[identity][1])
            survivals.append(float(row['survival_at_horizon']))
            risks.append(1-float(row['survival_at_horizon']))
            ids.append(identity)
    times = np.asarray(times)
    events = np.asarray(events)
    risks = np.asarray(risks)
    survivals = np.asarray(survivals)
    c = harrell_c(times, events, risks)
    return {
        'patients': len(ids),
        'events': int(events.sum()),
        'unique_patients': len(set(ids)),
        'c_index': c['value'],
        'comparable_pairs': c['comparable_pairs'],
        'c_index_standard_error': float(np.sqrt(c['value'] * (1-c['value']) / max(c['comparable_pairs'], 1))),
    }


def baseline_folds():
    folds = []
    for fold in range(5):
        item = read_fold(BASELINE_RUN / f'FOLD_{fold}')
        if item is not None:
            folds.append(item)
    return folds


def main():
    labels = load_labels()
    results = {}

    base = baseline_folds()
    if len(base) == 5:
        results['BASELINE'] = {
            'source': str(BASELINE_RUN),
            'folds': [{k: v for k, v in f.items() if k != 'rows'} for f in base],
            'pooled': pooled(base, labels),
        }

    for tag_dir in sorted(STAGE2.iterdir()) if STAGE2.exists() else []:
        if not tag_dir.is_dir():
            continue
        folds = [read_fold(tag_dir / f'FOLD_{f}') for f in range(5)]
        present = [f for f in folds if f is not None]
        entry = {'source': str(tag_dir),
                 'folds': [{k: v for k, v in f.items() if k != 'rows'} for f in present],
                 'folds_complete': len(present) == 5}
        if present:
            entry['pooled'] = pooled(present, labels)
            entry['mean_c_index'] = float(np.mean([f['c_index'] for f in present]))
            entry['std_c_index'] = float(np.std([f['c_index'] for f in present], ddof=1)) if len(present) > 1 else None
        results[tag_dir.name] = entry

    out = RUN / 'stage2_summary.json'
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'{"config":30s} {"fold":>5s} {"C":>7s} {"Brier":>7s} {"NLL":>9s} {"ep":>4s}')
    for name, entry in results.items():
        for fold in entry['folds']:
            print(f'{name:30s} {fold["fold"]:5d} {fold["c_index"]:7.4f} {fold["ipcw_brier"]:7.4f} '
                  f'{fold["validation_nll"]:9.4f} {fold["best_epoch"]:4d}')
        if 'pooled' in entry:
            p = entry['pooled']
            extra = ''
            if 'mean_c_index' in entry:
                extra = f'  mean_fold_C={entry["mean_c_index"]:.4f}'
            print(f'{name:30s} {"POOLED":>5s} {p["c_index"]:7.4f} {"":7s} {"":9s} {"":4s} '
                  f' n={p["patients"]} pairs={p["comparable_pairs"]} SE~{p["c_index_standard_error"]:.4f}{extra}')
        if not entry.get('folds_complete', True):
            print(f'{name:30s}  (incomplete: {len(entry["folds"])}/5 folds)')
        print()


if __name__ == '__main__':
    main()
