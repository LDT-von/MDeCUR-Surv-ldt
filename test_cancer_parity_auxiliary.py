"""Checks for the auxiliary-weight overrides of the parity trainer.

The auxiliary path has two properties that the overrides must respect:

* every override names a real auxiliary term and carries a finite
  nonnegative weight;
* the auxiliary path needs a patient batch that satisfies the rank bound, so
  the training batches must merge a too-small tail instead of dropping it.
"""
import unittest

import torch

from cancer_pathway_stream import LOSS_NAMES, PathwayConfig, minimum_batch_size, patient_batches
from train_pathway_parity import _parse_auxiliary_overrides


class AuxiliaryOverrideTest(unittest.TestCase):
    def test_override_is_parsed_into_a_weight_mapping(self):
        overrides = _parse_auxiliary_overrides(['global_common=0.002', 'patch_within=0'])
        self.assertEqual(overrides, {'global_common': 0.002, 'patch_within': 0.0})

    def test_unknown_term_is_rejected(self):
        with self.assertRaises(ValueError):
            _parse_auxiliary_overrides(['not_a_term=1'])

    def test_malformed_value_is_rejected(self):
        with self.assertRaises(ValueError):
            _parse_auxiliary_overrides(['global_common'])
        with self.assertRaises(ValueError):
            _parse_auxiliary_overrides(['global_common=abc'])

    def test_negative_and_nonfinite_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            _parse_auxiliary_overrides(['global_common=-1'])
        with self.assertRaises(ValueError):
            _parse_auxiliary_overrides(['global_common=nan'])
        with self.assertRaises(ValueError):
            _parse_auxiliary_overrides(['global_common=inf'])

    def test_empty_override_set_is_allowed(self):
        self.assertEqual(_parse_auxiliary_overrides([]), {})

    def test_every_term_name_is_covered_by_the_parser(self):
        overrides = _parse_auxiliary_overrides([f'{name}=0.5' for name in LOSS_NAMES])
        self.assertEqual(set(overrides), set(LOSS_NAMES))


class AuxiliaryBatchBoundTest(unittest.TestCase):
    def test_rank_bound_requires_more_patients_than_the_hidden_width(self):
        cfg = PathwayConfig(hidden=16, common=8, local_dim=8, local_common=4,
                            global_path_within=.0005)
        self.assertEqual(minimum_batch_size(cfg), 17)

    def test_disabled_auxiliaries_need_only_one_patient(self):
        cfg = PathwayConfig(**{name: 0. for name in LOSS_NAMES}, common_mode='off')
        self.assertEqual(minimum_batch_size(cfg), 1)

    def test_tail_batch_is_merged_instead_of_dropped(self):
        order = torch.arange(35)
        batches = patient_batches(order, 17, 17)
        self.assertEqual([len(batch) for batch in batches], [17, 18])
        self.assertEqual(sum(len(batch) for batch in batches), 35)

    def test_a_single_short_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            patient_batches(torch.arange(5), 17, 17)


if __name__ == '__main__':
    unittest.main()
