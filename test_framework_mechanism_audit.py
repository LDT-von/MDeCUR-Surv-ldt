"""Known-answer checks for research nulls and candidate sensitivity weights."""
import unittest

import torch

from cancer_stability_candidates import scaled_difference_weights
from research_framework_audit import matching_null, stability_shift, unique_nonlinear_null


class FrameworkMechanismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_actual_matching_admits_non_degenerate_independent_modality_shortcut(self):
        result = matching_null(17)
        self.assertTrue(result['raw_pathology_independent_of_rna_by_construction'])
        self.assertLess(result['closed_form_pool_max_error'], 1e-12)
        self.assertGreater(result['own_query']['diagonal_mean'], .99)
        self.assertLess(result['own_query']['identity_loss'], 1e-3)
        self.assertEqual(result['own_query']['path_centered_rank'], 4)
        for name in ('independent_query', 'pathology_only_anchor'):
            self.assertLess(abs(result[name]['diagonal_mean']), 1e-12)
            self.assertEqual(result[name]['low_variance_coordinates'], 0)
            self.assertAlmostEqual(result[name]['identity_loss'], 4.)
        self.assertEqual(result['detach_query_max_difference'], 0.)
        self.assertEqual(result['pathology_only_anchor']['path_centered_rank'], 1)
        self.assertFalse(result['independent_patient_sample'])

    def test_zero_correlation_does_not_identify_unique_information(self):
        result = unique_nonlinear_null()
        self.assertLess(result['zero_target_loss'], 1e-24)
        self.assertGreater(result['path_variance'], .01)
        self.assertEqual(result['low_variance_coordinates'], 0)

    def test_shift_changes_cosine_weights_and_candidate_is_invariant(self):
        result = stability_shift(17)
        self.assertGreater(result['legacy_shift_max_weight_change'], .5)
        self.assertLess(result['candidate_shift_max_weight_change'], 1e-12)
        self.assertLess(result['candidate_joint_rescale_max_weight_change'], 1e-12)
        self.assertTrue(result['candidate_detached'])

    def test_candidate_detaches_and_rejects_invalid_training_scales(self):
        p = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
        r = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
        ps, rs = torch.ones(3), torch.ones(2)
        weights, info = scaled_difference_weights(p, p+.1, r, r+.2, ps, rs)
        self.assertFalse(weights.requires_grad)
        self.assertFalse(info['standardized_difference_energy'].requires_grad)
        with self.assertRaises(ValueError):
            scaled_difference_weights(p, p, r, r, ps*0, rs)
        with self.assertRaises(ValueError):
            scaled_difference_weights(p, p, r, r, ps, rs, strength=float('nan'))


if __name__ == '__main__':
    unittest.main()
