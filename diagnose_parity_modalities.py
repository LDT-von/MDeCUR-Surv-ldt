"""Inference-time ablations for a trained parity checkpoint.

The questions answered here are:

* does the score survive when the RNA of every patient is replaced by another
  patient's RNA?  A genuine cross-modal match must collapse;
* how much of the score comes from the pathology branch alone?
* how much of the score comes from the RNA branch alone?

Everything is computed on the validation cohort with the reference metric
implementations.  The checkpoint is never modified.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cancer_parity import discrete_survival, dct_risk, parity_metrics
from cancer_pathway import PathwayLocalSurvival, read_pathway_mapping
from cancer_pathway_stream import PathwayConfig, pathway_stream_forward
from cancer_parity_bag import BudgetedPatientBag
from cancer_rna import fit_rna_scaler, transform_rna
from cancer_stream_data import load_stream_manifest


def _predict(model, bags, rna, chunk_size):
    with torch.no_grad():
        output = pathway_stream_forward(model, bags, rna, chunk_size, 'feature_noise',
                                        include_views=False, include_patch=False)
    return output['log_rates']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--survival-reference', required=True)
    parser.add_argument('--genes', required=True)
    parser.add_argument('--pathways', required=True)
    parser.add_argument('--num-patches', type=int, required=True)
    parser.add_argument('--n-bins', type=int, default=4)
    parser.add_argument('--patch-chunk-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = PathwayLocalSurvival(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    cohort = load_stream_manifest(args.manifest, path_views='single_view', include_test=False)
    reference = {}
    with Path(args.survival_reference).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            reference[row['patient_id'].strip()] = (float(row['time']), float(row['event']))
    train_ids = cohort['train']['ids']
    val_ids = cohort['val']['ids']
    train_time = np.array([reference[i][0] for i in train_ids])
    train_event = np.array([reference[i][1] for i in train_ids])
    val_time = np.array([reference[i][0] for i in val_ids])
    val_event = np.array([reference[i][1] for i in val_ids])

    scaler = fit_rna_scaler(cohort['train']['rna'], checkpoint['rna_preprocessing']['policy'],
                             checkpoint['rna_preprocessing']['min_std'])
    normalized = transform_rna(cohort['val']['rna'], scaler)
    bags = [BudgetedPatientBag(bag, args.num_patches, 'eval', args.seed)
            for bag in cohort['val']['bags']]
    edges = np.asarray(checkpoint['bin_edges'], dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf

    def score(logits, tag):
        survival = discrete_survival(logits.detach().cpu()).numpy()
        metrics = parity_metrics(train_time, train_event, val_time, val_event, survival, edges)
        metrics['case'] = tag
        return metrics

    results = []
    baseline = _predict(model, bags, normalized.to(device), args.patch_chunk_size)
    results.append(score(baseline, 'baseline'))

    # Replace the RNA of every validation patient with another patient's RNA.
    # The pathology bags are untouched, so any survival of the score would mean
    # the cross-modal match is not carrying the prediction.
    generator = np.random.default_rng(0)
    permutation = generator.permutation(len(val_ids))
    shuffled = normalized[permutation].to(device)
    results.append(score(_predict(model, bags, shuffled, args.patch_chunk_size),
                          'rna_permuted_across_patients'))

    # Zero the RNA: the score now comes from the pathology branch alone.
    zeros = torch.zeros_like(normalized).to(device)
    results.append(score(_predict(model, bags, zeros, args.patch_chunk_size),
                          'rna_zeroed'))

    # Zero the pathology: the score now comes from the RNA branch alone.
    empty = [BudgetedPatientBag(bag, 0, 'eval', args.seed) for bag in []] or None
    del empty
    zeros_patches = []
    for bag in bags:
        stub = BudgetedPatientBag(bag, args.num_patches, 'eval', args.seed)
        stub._rows = []
        stub.padding = args.num_patches
        zeros_patches.append(stub)
    results.append(score(_predict(model, zeros_patches, normalized.to(device),
                                   args.patch_chunk_size),
                          'pathology_zeroed'))

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'checkpoint': args.checkpoint,
                                       'cases': results}, indent=2), encoding='utf-8')
    for item in results:
        print('%-34s C=%.4f  IPCW=%.4f  IBS=%.4f  IAUC=%.4f'
              % (item['case'], item['c_index'], item['c_index_ipcw'],
                 item['ibs'], item['iauc']))


if __name__ == '__main__':
    main()
