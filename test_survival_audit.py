import unittest

import torch

from cancer_metrics import harrell_c, ipcw_brier
from cancer_survival_diagnostics import cumulative_hazard_at
from research_survival_audit import crossing_case, rng_case, saturation_case


class SurvivalAuditTest(unittest.TestCase):
    def test_probability_rounding_changes_c_without_survival_underflow(self):
        result = saturation_case()
        self.assertEqual(result['1.0']['legacy_probability_risk_c']['value'], 1.)
        self.assertEqual(result['12.0']['legacy_probability_risk_c']['value'], .5)
        self.assertEqual(result['12.0']['cumulative_hazard_risk_c']['value'], 1.)
        self.assertEqual(result['12.0']['comparable_pairs_tied_only_in_probability_risk'], 1)

    def test_two_interval_known_crossing(self):
        values = crossing_case()['survival']
        self.assertLess(values[0][0], values[1][0])
        self.assertAlmostEqual(values[0][1], values[1][1], places=12)
        self.assertGreater(values[0][2], values[1][2])
        self.assertAlmostEqual(values[0][0], .38289288597511206, places=12)

    def test_legacy_ipcw_ties_and_full_evaluation_denominator(self):
        got = ipcw_brier([1, 1, 2], [1, 0, 1], [1, 2], [1, 1], [.2, .7], 1.5)
        self.assertAlmostEqual(got['value'], .0875)
        # An early-censored subject has zero contribution but remains in n.
        third = ipcw_brier([1, 1, 2], [1, 0, 1], [1, 2, .5], [1, 1, 0], [.2, .7, .99], 1.5)
        self.assertAlmostEqual(third['value'], .0875*2/3)

    def test_reject_nonfinite_metric_horizon_and_invalid_floor(self):
        for horizon in (float('nan'), float('inf'), -1.):
            with self.assertRaises(ValueError):
                ipcw_brier([1, 3], [1, 1], [1, 3], [1, 1], [.2, .8], horizon)
        for eps in (0., float('nan'), 1.):
            with self.assertRaises(ValueError):
                ipcw_brier([1, 3], [1, 1], [1, 3], [1, 1], [.2, .8], 2., eps=eps)
        with self.assertRaises(ValueError):
            harrell_c([-1, 2], [1, 1], [2, 1])

    def test_hazard_rejects_bad_shapes_and_handles_time_zero(self):
        with self.assertRaises(ValueError):
            cumulative_hazard_at(torch.ones(2, 1), 1., torch.tensor([0., 2.]))
        actual = cumulative_hazard_at(torch.ones(2, 1), 0., torch.tensor([0.]))
        torch.testing.assert_close(actual, torch.zeros(2, dtype=torch.float64))

    def test_shuffle_isolated_from_following_view_noise(self):
        values = rng_case()
        self.assertFalse(values['legacy_next_noise_matches_reference'])
        self.assertTrue(values['isolated_next_noise_matches_reference'])


if __name__ == '__main__':
    unittest.main()
