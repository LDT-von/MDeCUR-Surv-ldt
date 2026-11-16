"""Frozen legacy parity input interventions, not retrained single-modality models.

Full-RNA derangement and input-zeroing probe this checkpoint under distribution
changes. Remaining predictive signal does not disprove a cross-modal mechanism;
a drop does not establish causal synergy. Saved training input controls are
replayed before any additional intervention. Dense checkpoints are rejected.
"""
import hashlib

import numpy as np
import torch

from cancer_pathway_stream import pathway_stream_forward
from diagnose_parity_interaction import common_parser, loaded_legacy_inputs, score_output
from diagnose_rna_only_baseline import metric_text, write_result


class ZeroInputBag:
    """Explicit zero rows with exactly the source budget and block lengths."""
    def __init__(self, source):
        self.patient_id = source.patient_id
        self.feature_dim = source.feature_dim
        self.view_source = source.view_source
        self.n_patches = source.n_patches
        self.padding = self.n_patches

    def blocks(self, chunk_size):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('Positive chunk size required')
        for start in range(0, self.n_patches, chunk_size):
            yield ('<diagnostic-zero-input>', start, min(start+chunk_size, self.n_patches))

    def read_block(self, token, names):
        path, start, stop = token
        if (path != '<diagnostic-zero-input>' or tuple(names) != ('features_reference',)
                or not 0 <= start < stop <= self.n_patches):
            raise ValueError('Invalid zero-input block')
        return (torch.zeros(stop-start, self.feature_dim, dtype=torch.float32),)


def _predict(model, bags, rna, chunk_size):
    with torch.no_grad():
        return pathway_stream_forward(model, bags, rna, chunk_size, 'feature_noise',
                                      include_views=False, include_patch=False, checkpoint_chunks=False)


def full_rna_derangement(count, seed):
    if count < 2:
        raise ValueError('At least two validation patients required')
    order = np.random.default_rng(seed).permutation(count)
    permutation = np.empty_like(order)
    permutation[order] = np.roll(order, 1)
    if np.any(permutation == np.arange(count)):
        raise AssertionError('Expected no fixed points')
    return permutation


def main():
    parser = common_parser()
    parser.description = __doc__
    parser.add_argument('--genes', help='Optional ordered-gene schema assertion; requires --pathways')
    parser.add_argument('--pathways', help='Optional pathway schema assertion; requires --genes')
    args = parser.parse_args()
    with loaded_legacy_inputs(args) as (model, bags, normalized, observed, edges, evidence):
        permutation = full_rna_derangement(len(normalized), args.seed)
        indices = torch.as_tensor(permutation, device=normalized.device)
        cases = []
        for tag, case_bags, rna in (
                ('checkpoint_native_inputs', bags, normalized),
                ('whole_rna_deranged_across_patients', bags, normalized[indices]),
                ('rna_input_zeroed', bags, torch.zeros_like(normalized)),
                ('pathology_input_zeroed', [ZeroInputBag(bag) for bag in bags], normalized)):
            output = _predict(model, case_bags, rna, args.patch_chunk_size)
            cases.append({'case': tag, **score_output(output, observed, edges)})
        result = {'checkpoint': args.checkpoint, 'evidence': evidence, 'cases': cases,
                  'rna_permutation': {'seed': args.seed, 'fixed_points': 0,
                                      'algorithm': 'random single-cycle derangement; not uniform over all derangements',
                                      'sha256': hashlib.sha256(permutation.astype('<i8').tobytes()).hexdigest()},
                  'scope': 'Inference-only input interventions on a frozen legacy checkpoint; no retraining or selection.',
                  'limits': ['Zeroing an input is not deleting its biased encoder/interaction path or retraining a single-modality model.',
                             'When a modality was already frozen during training, some interventions are intentionally identical.',
                             'A retained score after RNA permutation is possible through pathology signal and does not prove leakage or ineffective matching.',
                             'No strict information-theoretic synergy, causal effect, or generalization gain is estimated.']}
    write_result(args.out, result)
    for item in cases:
        print(f"{item['case']:38} C={metric_text(item['c_index'])} IPCW={metric_text(item['c_index_ipcw'])} "
              f"IBS={metric_text(item['ibs'])} IAUC={metric_text(item['iauc'])}")


if __name__ == '__main__':
    main()
