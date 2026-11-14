"""Known-answer synthetic checks for the independent metric-semantics audit."""
import importlib.util
import json
import unittest

import numpy as np

from cancer_parity_audit import aligned_grid, audit_parity_metrics


@unittest.skipUnless(importlib.util.find_spec('sksurv'), 'scikit-survival is required')
class ParitySemanticAuditTest(unittest.TestCase):
    def test_finite_edges_match_known_survival_columns_and_brier(self):
        edges = [-np.inf, 10., 20., 30., np.inf]
        grid, columns = aligned_grid([.5, 40.], [1, 1], [5., 35.], edges, 4)
        np.testing.assert_array_equal(grid, [10., 20., 30.])
        np.testing.assert_array_equal(columns, [0, 1, 2])
        # Hazards [.2, .25, .5, .5] imply these cumulative survival values.
        survival = np.array([[.8, .6, .3, .15], [.8, .6, .3, .15]])
        audit = audit_parity_metrics([.5, 40.], [1, 1], [5., 35.], [1, 1], survival, edges)
        np.testing.assert_allclose(audit['brier'], [.34, .26, .29], atol=1e-14)
        self.assertAlmostEqual(audit['ibs'], .2875)
        self.assertEqual(audit['survival_column_indices'], [0, 1, 2])
        self.assertEqual(audit['protocol'], 'corrected/time_semantic_audit')
        self.assertFalse(audit['legacy_results_overwritten'])
        self.assertEqual(audit['auc_by_time'], [.5, .5, .5])
        json.dumps(audit, allow_nan=False)

    def test_grid_filters_support_without_shifting_column_ids(self):
        grid, columns = aligned_grid([1., 15., 25.], [1, 0, 0], [12., 35.],
                                     [-np.inf, 10., 20., 30., np.inf], 4)
        np.testing.assert_array_equal(grid, [20.])
        np.testing.assert_array_equal(columns, [1])

    def test_real_sksurv_pair_count_includes_same_time_censored_patient(self):
        audit = audit_parity_metrics([.5, 1.5, 2.5, 3.5], [1, 1, 1, 1],
                                    [1., 1., 2.], [1, 0, 1],
                                    [[.3, .2], [.2, .1], [.8, .7]],
                                    [-np.inf, 1.5, np.inf])
        self.assertEqual(audit['c_index'], .5)
        self.assertEqual(audit['comparable_pairs'], 2)
        self.assertEqual(audit['harrell_counts'], {'concordant': 1, 'discordant': 1,
                                                  'tied_risk': 0, 'tied_time': 1,
                                                  'comparable_pairs': 2})
        self.assertEqual(audit['legacy_comparable_pairs'], 1)
        self.assertEqual(audit['legacy_harrell_c'], 1.)
        self.assertEqual(audit['pair_count_difference_sksurv_minus_legacy'], 1)

    def test_harrell_risk_direction_stays_unchanged(self):
        predictions = np.array([[.1, .05, .02, .01], [.9, .8, .7, .6]])
        arguments = ([.5, 3.], [1, 1], [1., 2.], [1, 1])
        edges = [-np.inf, 1.1, 1.4, 1.6, np.inf]
        self.assertEqual(audit_parity_metrics(*arguments, predictions, edges)['c_index'], 1.)
        self.assertEqual(audit_parity_metrics(*arguments, predictions[::-1], edges)['c_index'], 0.)

    def test_ipcw_weights_and_actual_event_contributions_have_known_values(self):
        predictions = [[.95, .85, .75, .65], [.4, .2, .1, .05], [.8, .5, .3, .1]]
        audit = audit_parity_metrics([1., 2., 3., 4., 5., 6.], [1, 0, 0, 0, 0, 1],
                                    [1.5, 4.5, 5.5], [1, 1, 1], predictions,
                                    [-np.inf, 2., 3., 4., np.inf])
        self.assertAlmostEqual(audit['c_index'], 1 / 3)
        self.assertAlmostEqual(audit['c_index_ipcw'], 6.25 / 8.25)
        diagnostics = audit['censoring_audit']
        self.assertEqual(diagnostics['n_events_with_comparable_pairs'], 2)
        self.assertEqual(diagnostics['n_events_without_comparable_pairs'], 1)
        for key, expected in [('min', 1.), ('median', 6.25), ('max', 25.)]:
            self.assertAlmostEqual(diagnostics['ipcw_squared_event_weights'][key], expected)
        self.assertAlmostEqual(diagnostics['g_at_event_times']['min'], .2)
        self.assertAlmostEqual(diagnostics['weighted_pair_denominator'], 8.25)
        self.assertAlmostEqual(diagnostics['largest_event_pair_weight_fraction'], 6.25 / 8.25)
        self.assertAlmostEqual(diagnostics['contributing_event_weights']['max'], 6.25)
        self.assertIsNone(audit['ipcw_tau'])

    def test_unsupported_ipcw_remains_none_with_explicit_errors(self):
        audit = audit_parity_metrics([1., 2., 3.], [1, 0, 0], [2.5, 3.5], [1, 1],
                                    [[.4, .2], [.8, .5]], [-np.inf, 2.7, np.inf])
        self.assertIsNone(audit['c_index_ipcw'])
        self.assertTrue(any(message.startswith('c_index_ipcw:') for message in audit['errors']))
        self.assertEqual(audit['censoring_audit']['zero_g_event_times'], 1)
        self.assertIsNone(audit['censoring_audit']['weighted_pair_denominator'])
        self.assertEqual(audit['c_index'], 1.)
        json.dumps(audit, allow_nan=False)

    def test_no_finite_grid_does_not_destroy_harrell_result(self):
        audit = audit_parity_metrics([.5, 3.], [1, 1], [1., 2.], [1, 1],
                                    [[.2], [.8]], [-np.inf, np.inf])
        self.assertEqual(audit['c_index'], 1.)
        self.assertIsNone(audit['brier'])
        self.assertEqual(audit['metric_times'], [])
        self.assertTrue(any(message.startswith('time_grid:') for message in audit['errors']))

    def test_invalid_probabilities_and_nonbinary_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonincreasing'):
            audit_parity_metrics([.5, 3.], [1, 1], [1., 2.], [1, 1],
                                 [[.2, .8], [.8, .5]], [-np.inf, 1.5, np.inf])
        with self.assertRaisesRegex(ValueError, 'zero or one'):
            audit_parity_metrics([.5, 3.], [1, 2], [1., 2.], [1, 1],
                                 [[.4, .2], [.8, .5]], [-np.inf, 1.5, np.inf])
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            aligned_grid([.5, 3.], [1, 1], [1., 2.], [-np.inf, -np.inf], 1)


if __name__ == '__main__':
    unittest.main()
