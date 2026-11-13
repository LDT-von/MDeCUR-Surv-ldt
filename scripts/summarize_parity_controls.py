"""Summarize the auxiliary-loss comparison and the modality controls.

Every run is a fold-0 BLCA parity run on the same 77-patient validation
cohort.  The table reports the validation C-index per run together with the
auxiliary diagnostics, so that the effect of the decoupling objectives can be
read off directly.
"""
import json
import os
from pathlib import Path

import numpy as np

BASE = Path('/data1/decur_runs/PARITY_DCT_FOLD0')
RUNS = [
    ('full_aux_off_bs17', 'aux_off.log', 'auxiliary off, batch 17'),
    ('full_aux_on_bs17', 'aux_on.log', 'auxiliary on, batch 17'),
    ('aux_off_bs8', 'run_fold0_30ep/report.json', 'auxiliary off, batch 8 (reference run)'),
    ('null_labels', 'ctrl_null.log', 'training labels permuted'),
    ('rna_only', 'ctrl_rna_only.log', 'pathology zeroed'),
    ('path_only', 'ctrl_path_only.log', 'RNA zeroed'),
]


def read_epochs(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('{') and '"epoch"' in line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def summarize(name, path, note):
    rows = read_epochs(path)
    if not rows:
        return {'name': name, 'note': note, 'status': 'not available'}
    scores = [row['c_index'] for row in rows if row.get('c_index') is not None]
    if not scores:
        return {'name': name, 'note': note, 'status': 'no C-index'}
    best = max(rows, key=lambda row: row['c_index'])
    record = {
        'name': name, 'note': note, 'status': 'ok',
        'epochs': len(rows), 'mean_c_index': float(np.mean(scores)),
        'median_c_index': float(np.median(scores)), 'max_c_index': float(np.max(scores)),
        'last_c_index': float(scores[-1]), 'best_epoch': best.get('epoch'),
        'best_c_index': float(best['c_index']),
    }
    terms = best.get('auxiliary_terms')
    if terms:
        record['auxiliary_terms_at_best'] = {k: v['mean'] for k, v in terms.items()}
    diagnostics = best.get('pathway_diagnostics')
    if diagnostics:
        record['pathway_diagnostics_at_best'] = diagnostics
    return record


def main():
    table = []
    for name, filename, note in RUNS:
        path = BASE/filename
        if filename.endswith('report.json'):
            report = json.loads(path.read_text(encoding='utf-8'))
            history = report.get('history', [])
            scores = [row['c_index'] for row in history if row.get('c_index') is not None]
            best = max(history, key=lambda row: row['c_index'])
            table.append({'name': name, 'note': note, 'status': 'ok',
                          'epochs': len(history), 'mean_c_index': float(np.mean(scores)),
                          'median_c_index': float(np.median(scores)),
                          'max_c_index': float(np.max(scores)),
                          'last_c_index': float(scores[-1]),
                          'best_epoch': best.get('epoch'),
                          'best_c_index': float(best['c_index']),
                          'batch_size': report.get('minimum_patient_batch')})
        else:
            table.append(summarize(name, path, note))
    destination = BASE/'aux_comparison.summary.json'
    destination.write_text(json.dumps(table, indent=2), encoding='utf-8')
    header = f"{'run':<22}{'note':<34}{'ep':>4}{'mean':>8}{'peak':>8}{'bestEp':>8}"
    print(header)
    print('-'*len(header))
    for row in table:
        if row.get('status') != 'ok':
            print(f"{row['name']:<22}{row['note']:<34}{row.get('status',''):>8}")
            continue
        print(f"{row['name']:<22}{row['note']:<34}{row['epochs']:>4}"
              f"{row['mean_c_index']:>8.4f}{row['max_c_index']:>8.4f}{row['best_epoch']:>8}")
    print(f'\nwritten to {destination}')


if __name__ == '__main__':
    main()
