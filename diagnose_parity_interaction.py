"""Is the interaction branch doing anything, and what does each branch contribute?

The trained checkpoint is re-evaluated under three controlled changes:

* ``interaction_strength = 0`` removes the cross-modal interaction term from
  the prediction.  If the score is unchanged, the interaction branch is
  decoration.
* the interaction term is reported on its own scale, so that its magnitude
  can be compared with the base logits.
* the matching entropy and the maximum matching weight are reported, so that
  the sharpness of the RNA-conditioned patch matching is visible.

The checkpoint is never modified.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cancer_parity import discrete_survival, parity_metrics
from cancer_parity_bag import BudgetedPatientBag
from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import pathway_stream_forward
from cancer_rna import fit_rna_scaler, transform_rna
from cancer_stream_data import load_stream_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--survival-reference', required=True)
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
    train_ids, val_ids = cohort['train']['ids'], cohort['val']['ids']
    train_time = np.array([reference[i][0] for i in train_ids])
    train_event = np.array([reference[i][1] for i in train_ids])
    val_time = np.array([reference[i][0] for i in val_ids])
    val_event = np.array([reference[i][1] for i in val_ids])

    scaler = fit_rna_scaler(cohort['train']['rna'], checkpoint['rna_preprocessing']['policy'],
                             checkpoint['rna_preprocessing']['min_std'])
    normalized = transform_rna(cohort['val']['rna'], scaler).to(device)
    bags = [BudgetedPatientBag(bag, args.num_patches, 'eval', args.seed)
            for bag in cohort['val']['bags']]
    edges = np.asarray(checkpoint['bin_edges'], dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf

    def run(strength):
        original = model.interaction_strength
        model.interaction_strength = strength
        with torch.no_grad():
            output = pathway_stream_forward(model, bags, normalized, args.patch_chunk_size,
                                            'feature_noise', include_views=False,
                                            include_patch=False)
        model.interaction_strength = original
        logits = output['log_rates'].detach().cpu()
        survival = discrete_survival(logits).numpy()
        metrics = parity_metrics(train_time, train_event, val_time, val_event, survival, edges)
        extra = output['interaction_logits'].detach().cpu()
        base = logits-original*extra
        metrics['interaction_abs_mean'] = float(extra.abs().mean())
        metrics['base_abs_mean'] = float(base.abs().mean())
        metrics['interaction_to_base_ratio'] = float(extra.abs().mean()/(base.abs().mean()+1e-12))
        metrics['matching_normalized_entropy_mean'] = float(
            output['matching_normalized_entropy'].mean())
        metrics['matching_max_weight_mean'] = float(output['matching_max_weight'].mean())
        return metrics

    cases = [('interaction_strength_0.2', run(0.2)),
             ('interaction_strength_0', run(0.0)),
             ('interaction_strength_1.0', run(1.0))]
    for tag, item in cases:
        print('%-26s C=%.4f  IPCW=%.4f  IBS=%.4f  IAUC=%.4f  |inter|/|base|=%.3f'
              % (tag, item['c_index'], item['c_index_ipcw'], item['ibs'], item['iauc'],
                 item['interaction_to_base_ratio']))
    print()
    print('matching normalized entropy %.4f (1.0 = uniform over patches, 0 = one patch)'
          % cases[0][1]['matching_normalized_entropy_mean'])
    print('matching max weight %.4f' % cases[0][1]['matching_max_weight_mean'])

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'checkpoint': args.checkpoint, 'cases': cases},
                                      indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
