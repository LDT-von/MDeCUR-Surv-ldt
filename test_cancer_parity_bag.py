"""Checks for the reference-protocol patch budget wrapper.

The reference pipeline concatenates the slides of a patient, samples a fixed
budget of rows and zero-pads a short bag.  These checks pin down the row
selection, the padding and the streaming contract.
"""
import unittest

import torch

from cancer_parity_bag import BudgetedPatientBag


class _FakeBag:
    """A minimal bag that satisfies the streaming contract."""

    view_source = 'single_view'

    def __init__(self, patient_id, slides, dim=4):
        self.patient_id = patient_id
        self.feature_dim = dim
        self._slides = slides
        self.n_patches = sum(slides.values())

    def blocks(self, chunk_size):
        for path, rows in self._slides.items():
            for start in range(0, rows, chunk_size):
                yield (path, start, min(start + chunk_size, rows))

    def read_rows(self, rows):
        offsets, cursor = {}, 0
        for path, count in self._slides.items():
            offsets[path] = (cursor, cursor+count)
            cursor += count
        pieces = []
        for row in rows:
            for path, (base, stop) in offsets.items():
                if base <= row < stop:
                    pieces.append(torch.full((self.feature_dim,), float(row)))
                    break
        return torch.stack(pieces)

    def read_block(self, token, names):
        path, start, stop = token
        return (self.read_rows(list(range(start, stop))),)

    @property
    def provenance(self):
        return []


class BudgetedBagTest(unittest.TestCase):
    def test_eval_rows_are_evenly_spaced_over_the_concatenated_bag(self):
        bag = _FakeBag('P1', {'a.pt': 30, 'b.pt': 20})
        budgeted = BudgetedPatientBag(bag, 10, 'eval', 0)
        self.assertEqual(budgeted._rows, [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        self.assertEqual(budgeted.padding, 0)
        self.assertEqual(budgeted.n_patches, 10)

    def test_short_bag_is_zero_padded_to_the_budget(self):
        bag = _FakeBag('P1', {'a.pt': 4})
        budgeted = BudgetedPatientBag(bag, 8, 'eval', 0)
        self.assertEqual(budgeted.padding, 4)
        rows = []
        for token in budgeted.blocks(8):
            array, = budgeted.read_block(token, ('features_reference',))
            rows.append(array)
        stacked = torch.cat(rows)
        self.assertEqual(stacked.shape, (8, bag.feature_dim))
        self.assertTrue(torch.all(stacked[4:] == 0))
        self.assertTrue(torch.all(stacked[:4, 0] == torch.tensor([0., 1., 2., 3.])))

    def test_train_rows_are_a_subset_without_replacement(self):
        bag = _FakeBag('P1', {'a.pt': 100})
        budgeted = BudgetedPatientBag(bag, 20, 'train', 5)
        self.assertEqual(len(set(budgeted._rows)), 20)
        self.assertEqual(budgeted._rows, sorted(budgeted._rows))
        self.assertTrue(all(0 <= row < 100 for row in budgeted._rows))

    def test_resample_changes_the_training_rows(self):
        bag = _FakeBag('P1', {'a.pt': 100})
        budgeted = BudgetedPatientBag(bag, 20, 'train', 5)
        first = list(budgeted._rows)
        budgeted.resample(6)
        self.assertNotEqual(first, budgeted._rows)
        self.assertEqual(len(set(budgeted._rows)), 20)

    def test_blocks_cover_the_whole_budget(self):
        bag = _FakeBag('P1', {'a.pt': 30, 'b.pt': 20})
        budgeted = BudgetedPatientBag(bag, 16, 'eval', 0)
        covered = 0
        for token in budgeted.blocks(4):
            self.assertEqual(token[0], '<parity-budget>')
            array, = budgeted.read_block(token, ('features_reference',))
            self.assertEqual(array.shape[0], token[2]-token[1])
            covered += token[2]-token[1]
        self.assertEqual(covered, 16)

    def test_multi_slide_rows_are_resolved_to_the_right_slide(self):
        bag = _FakeBag('P1', {'a.pt': 10, 'b.pt': 10})
        budgeted = BudgetedPatientBag(bag, 10, 'eval', 0)
        array, = budgeted.read_block(('<parity-budget>', 0, 10), ('features_reference',))
        # The evenly spaced rows of a twenty-row bag are 0, 2, 4, ... 18.
        expected = torch.tensor([0., 2., 4., 6., 8., 10., 12., 14., 16., 18.])
        self.assertTrue(torch.all(array[:, 0] == expected))

    def test_invalid_arguments_are_rejected(self):
        bag = _FakeBag('P1', {'a.pt': 10})
        with self.assertRaises(ValueError):
            BudgetedPatientBag(bag, 0, 'eval', 0)
        with self.assertRaises(ValueError):
            BudgetedPatientBag(bag, 10, 'unknown', 0)

    def test_the_source_bag_is_never_mutated(self):
        bag = _FakeBag('P1', {'a.pt': 10})
        BudgetedPatientBag(bag, 5, 'train', seed=1)
        self.assertEqual(bag.n_patches, 10)


if __name__ == '__main__':
    unittest.main()
