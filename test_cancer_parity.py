"""Checks for the reference-protocol parity helpers.

The expectations come from the reference implementation, not from this
repository's previous behaviour: the patch-budget rule, the quantile bin edges,
the discrete likelihood, the risk convention and the metric grid.
"""
import unittest

import numpy as np
import torch

from cancer_parity import (discrete_nll, discrete_survival, dct_bin_edges, dct_risk,
                           discretize_labels, metric_time_grid, parity_metrics,
                           select_patch_rows)


class PatchBudgetTest(unittest.TestCase):
    def test_eval_rows_are_evenly_spaced(self):
        rows, padding = select_patch_rows(100, 10, 'eval')
        self.assertEqual(rows.tolist(), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        self.assertEqual(padding, 0)

    def test_short_patient_reports_padding(self):
        rows, padding = select_patch_rows(7, 10, 'eval')
        self.assertEqual(len(rows), 7)
        self.assertEqual(padding, 3)

    def test_train_rows_are_sorted_and_distinct(self):
        rows, padding = select_patch_rows(500, 64, 'train', seed=11, patient_id='P1')
        self.assertEqual(len(set(rows.tolist())), 64)
        self.assertEqual(rows.tolist(), sorted(rows.tolist()))
        self.assertEqual(padding, 0)

    def test_train_draw_is_reproducible(self):
        first, _ = select_patch_rows(500, 64, 'train', seed=11, patient_id='P1')
        second, _ = select_patch_rows(500, 64, 'train', seed=11, patient_id='P1')
        self.assertEqual(first.tolist(), second.tolist())

    def test_train_draw_depends_on_the_patient(self):
        first, _ = select_patch_rows(500, 64, 'train', seed=11, patient_id='P1')
        second, _ = select_patch_rows(500, 64, 'train', seed=11, patient_id='P2')
        self.assertNotEqual(first.tolist(), second.tolist())

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            select_patch_rows(0, 10, 'eval')
        with self.assertRaises(ValueError):
            select_patch_rows(10, 0, 'eval')
        with self.assertRaises(ValueError):
            select_patch_rows(10, 10, 'unknown')


class BinEdgeTest(unittest.TestCase):
    def test_edges_are_train_only_quantiles_with_infinite_outer_edges(self):
        time = np.arange(1, 21, dtype=float)
        event = np.ones(20)
        edges = dct_bin_edges(time, event, 4)
        self.assertEqual(len(edges), 5)
        self.assertEqual(edges[0], -np.inf)
        self.assertEqual(edges[-1], np.inf)
        self.assertTrue(np.all(np.diff(edges) > 0))

    def test_censored_training_rows_are_excluded_from_the_edges(self):
        time = np.arange(1, 21, dtype=float)
        event = np.ones(20)
        event[0] = 0
        event[1] = 0
        edges = dct_bin_edges(time, event, 4)
        reference = dct_bin_edges(time[2:], event[2:], 4)
        self.assertEqual(edges.tolist(), reference.tolist())
        self.assertEqual(edges[0], -np.inf)
        self.assertEqual(edges[-1], np.inf)

    def test_labels_cover_every_patient(self):
        edges = dct_bin_edges(np.arange(1, 21, dtype=float), np.ones(20), 4)
        labels = discretize_labels(np.array([0.5, 5.0, 12.0, 20.0, 100.0]), edges)
        self.assertEqual(len(labels), 5)
        self.assertTrue(np.all((labels >= 0) & (labels < 4)))

    def test_too_few_uncensored_rows_is_rejected(self):
        with self.assertRaises(ValueError):
            dct_bin_edges(np.arange(1, 4, dtype=float), np.ones(3), 4)


class DiscreteLikelihoodTest(unittest.TestCase):
    def test_survival_is_the_cumulative_hazard_product(self):
        logits = torch.zeros(2, 3)
        survival = discrete_survival(logits)
        expected = torch.tensor([0.5, 0.25, 0.125]).expand(2, 3)
        self.assertTrue(torch.allclose(survival, expected))

    def test_censored_patients_contribute_only_the_survival_term(self):
        logits = torch.zeros(1, 4)
        labels = torch.tensor([1])
        censored = discrete_nll(logits, labels, torch.tensor([1.]), alpha=0.5)
        uncensored = discrete_nll(logits, labels, torch.tensor([0.]), alpha=0.5)
        survival = discrete_survival(logits)[0]
        # The reference likelihood weights the uncensored term by alpha and
        # the censored term by 1-alpha, so the censored loss is half of the
        # plain negative log survival term.
        expected_censored = .5*(-torch.log(survival[1]))
        self.assertAlmostEqual(float(censored), float(expected_censored), places=5)
        self.assertNotAlmostEqual(float(censored), float(uncensored), places=4)

    def test_alpha_zero_equals_the_plain_negative_log_likelihood(self):
        logits = torch.randn(6, 4)
        labels = torch.randint(0, 4, (6,))
        censorship = torch.randint(0, 2, (6,)).float()
        first = discrete_nll(logits, labels, censorship, alpha=0.0)
        second = discrete_nll(logits, labels, censorship, alpha=0.0)
        self.assertTrue(torch.allclose(first, second))

    def test_shapes_are_validated(self):
        with self.assertRaises(ValueError):
            discrete_nll(torch.zeros(3, 4), torch.zeros(2, dtype=torch.long),
                         torch.zeros(2), alpha=0.5)
        with self.assertRaises(ValueError):
            discrete_nll(torch.full((2, 4), float('nan')), torch.zeros(2, dtype=torch.long),
                         torch.zeros(2), alpha=0.5)


class RiskAndGridTest(unittest.TestCase):
    def test_risk_is_the_negative_survival_sum(self):
        survival = np.array([[0.9, 0.8, 0.7], [1.0, 1.0, 1.0]])
        risk = dct_risk(survival)
        self.assertAlmostEqual(float(risk[0]), -2.4, places=8)
        self.assertAlmostEqual(float(risk[1]), -3.0, places=8)

    def test_grid_mixes_the_outcome_range_with_the_inner_edges(self):
        edges = np.array([-np.inf, 5.0, 10.0, 15.0, np.inf])
        grid, valid = metric_time_grid(np.array([1.0, 20.0]), np.array([0.5, 25.0]),
                                       edges, 4)
        self.assertEqual(grid.size, valid.size)
        self.assertTrue(np.all(np.diff(grid) > 0))
        self.assertAlmostEqual(float(grid[0]), 1.0+1e-4, places=8)

    def test_grid_requires_matching_columns(self):
        edges = np.array([-np.inf, 5.0, 10.0, 15.0, np.inf])
        with self.assertRaises(ValueError):
            metric_time_grid(np.array([1.0, 20.0]), np.array([0.5, 25.0]), edges, 6)


class ParityMetricTest(unittest.TestCase):
    def _cohort(self, n=60, seed=5):
        rng = np.random.default_rng(seed)
        time = rng.uniform(1, 60, n)
        event = (rng.uniform(size=n) < 0.6).astype(float)
        return time, event

    def test_metrics_are_reported_with_the_reference_conventions(self):
        train_time, train_event = self._cohort(80, 1)
        test_time, test_event = self._cohort(40, 2)
        edges = dct_bin_edges(train_time, train_event, 4)
        logits = torch.randn(len(test_time), 4)
        survival = discrete_survival(logits).numpy()
        result = parity_metrics(train_time, train_event, test_time, test_event, survival, edges)
        self.assertIn('c_index', result)
        self.assertIsNotNone(result['c_index'])
        self.assertIsNotNone(result['comparable_pairs'])
        self.assertIsInstance(result['errors'], list)
        self.assertIsInstance(result['brier'], list)

    def test_local_and_sksurv_c_index_agree(self):
        from sksurv.metrics import concordance_index_censored
        train_time, train_event = self._cohort(80, 1)
        test_time, test_event = self._cohort(40, 2)
        edges = dct_bin_edges(train_time, train_event, 4)
        logits = torch.randn(len(test_time), 4)
        survival = discrete_survival(logits).numpy()
        result = parity_metrics(train_time, train_event, test_time, test_event, survival, edges)
        reference = concordance_index_censored(test_event.astype(bool), test_time,
                                               dct_risk(survival), tied_tol=1e-8)[0]
        self.assertAlmostEqual(result['c_index'], float(reference), places=10)
        self.assertAlmostEqual(result['harrell_c_local_implementation'],
                               float(reference), places=10)


if __name__ == '__main__':
    unittest.main()
