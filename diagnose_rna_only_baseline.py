"""How prognostic is the RNA vector on its own, under the reference protocol?

A deliberately simple baseline: a ridge regression from the raw expression
vector onto the discrete survival bins, evaluated with the reference metric
implementations.  The ridge output is used as the reference discrete hazard
logits, so the predicted survival is the cumulative product of the hazards.

This answers whether a high validation C-index is reachable from the RNA
table alone, without any pathology input and without the pathway model.

No patient data are bundled with the repository.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cancer_parity import dct_bin_edges, discretize_labels, parity_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--survival-reference', required=True)
    parser.add_argument('--n-bins', type=int, default=4)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = list(csv.DictReader(manifest_path.open(encoding='utf-8-sig', newline='')))
    reference = {}
    with Path(args.survival_reference).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            reference[row['patient_id'].strip()] = (float(row['time']), float(row['event']))

    train_rows = [row for row in manifest if row['split'] == 'train']
    val_rows = [row for row in manifest if row['split'] == 'val']
    train_ids = [row['patient_id'] for row in train_rows]
    val_ids = [row['patient_id'] for row in val_rows]
    train_time = np.array([reference[i][0] for i in train_ids])
    train_event = np.array([reference[i][1] for i in train_ids])
    val_time = np.array([reference[i][0] for i in val_ids])
    val_event = np.array([reference[i][1] for i in val_ids])

    edges = dct_bin_edges(train_time, train_event, args.n_bins)
    train_label = discretize_labels(train_time, edges)
    train_onehot = np.eye(args.n_bins)[train_label]

    def load(rows):
        return np.stack([np.load(manifest_path.parent/row['rna_npy']) for row in rows])

    train_x = load(train_rows).astype(np.float64)
    val_x = load(val_rows).astype(np.float64)

    # Center and scale on the training rows only.
    mean = train_x.mean(0)
    scale = train_x.std(0)
    scale[scale < 1e-8] = 1.0
    train_x = (train_x-mean)/scale
    val_x = (val_x-mean)/scale

    gram = train_x.T @ train_x
    rhs = train_x.T @ train_onehot
    identity = np.eye(train_x.shape[1])
    best = None
    sweep = []
    for alpha in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8):
        weights = np.linalg.solve(gram+alpha*identity, rhs)
        logits = val_x @ weights
        # The reference hazard parameterization: hazards are sigmoid(logits)
        # and the survival is the cumulative product of one minus the hazard.
        hazards = 1.0/(1.0+np.exp(-logits))
        survival = np.cumprod(1.0-hazards, axis=1)
        metrics = parity_metrics(train_time, train_event, val_time, val_event, survival, edges)
        sweep.append({'alpha': alpha, 'c_index': metrics['c_index'],
                      'c_index_ipcw': metrics['c_index_ipcw'], 'ibs': metrics['ibs'],
                      'iauc': metrics['iauc']})
        print('alpha %-10.0f  C=%.4f  IPCW=%.4f  IBS=%.4f  IAUC=%.4f'
              % (alpha, metrics['c_index'], metrics['c_index_ipcw'],
                 metrics['ibs'], metrics['iauc']))
        if metrics['c_index'] is not None and (best is None or metrics['c_index'] > best[1]):
            best = (alpha, metrics['c_index'])
    print()
    print('RNA-only ridge, best C over the penalty sweep: %.4f at alpha=%.0f' % best)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {'model': 'ridge_on_raw_expression', 'sweep': sweep,
             'best_alpha': best[0], 'best_c_index': best[1]}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
