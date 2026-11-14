"""Deterministic mathematical checks; no patient files or training."""
import unittest

import torch

from cancer_dense_tokens import (chunked_cosine_nearest_neighbor, dense_token_objective,
                                  token_correlation_penalty)


class DenseTokenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_known_matches_many_to_one_ties_and_chunk_equivalence(self):
        a = torch.tensor([[1., 0.], [2., .1], [-1., 0.]], requires_grad=True)
        b = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [1., 0.]], requires_grad=True)
        for size in (1, 2, 20):
            indices = chunked_cosine_nearest_neighbor(a, b, chunk_size=size)
            self.assertEqual(indices.tolist(), [0, 0, 1])
            self.assertFalse(indices.requires_grad)

    def test_exact_bt_and_decorrelation_values(self):
        x = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]], dtype=torch.float64)
        # Both variances are exactly one; epsilon shrinks each correlation to 1/(1+eps).
        eps = .25
        bt, db = token_correlation_penalty(x, x, 1, eps=eps)
        dec, dd = token_correlation_penalty(x, x, 0, eps=eps)
        self.assertAlmostEqual(bt.item(), 2*(1-1/(1+eps))**2, places=12)
        self.assertAlmostEqual(dec.item(), 2/(1+eps)**2, places=12)
        self.assertTrue(db['valid'] and dd['valid'])
        # Swapping coordinates yields off-diagonal entries only.
        swapped, _ = token_correlation_penalty(x, x.flip(1), 1, eps=eps, off_weight=.1)
        self.assertAlmostEqual(swapped.item(), 2+.1*2/(1+eps)**2, places=12)

    def test_mismatched_values_raise_bt_loss_and_collapsed_dec_is_invalid(self):
        x = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
        good, _ = token_correlation_penalty(x, x, 1)
        bad, _ = token_correlation_penalty(x, -x, 1)
        self.assertGreater(bad.item(), good.item()+1)
        zero, diag = token_correlation_penalty(torch.ones_like(x), torch.ones_like(x), 0)
        self.assertEqual(zero.item(), 0)
        self.assertFalse(diag['valid'])
        self.assertEqual(diag['low_variance_coordinates'], 2)

    def test_objective_four_view_pairs_and_backprop_to_projected_values(self):
        f = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]], dtype=torch.float64)
        views = (f, f[[2, 0, 3, 1]])
        z = tuple(torch.cat((v, v*.5), dim=1).clone().requires_grad_() for v in views)
        r = tuple((v*.7+.1).detach().clone().requires_grad_() for v in z)
        losses, diag = dense_token_objective(views, views, z, r, 2, match_chunk_size=2)
        self.assertEqual(set(losses), {'dense_path_within', 'dense_rna_within',
                                       'dense_common', 'dense_unique'})
        self.assertEqual(diag['cross_view_pairs'], 4)
        self.assertEqual(len(diag['cross']), 4)
        self.assertTrue(diag['valid'])
        self.assertFalse(diag['token_axis_is_independent_patients'])
        sum(losses.values()).backward()
        for value in (*z, *r):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_shape_empty_and_zero_direction_rejections(self):
        f = torch.ones(3, 2)
        with self.assertRaises(ValueError):
            chunked_cosine_nearest_neighbor(f[:0], f)
        with self.assertRaisesRegex(ValueError, 'zero/near-zero'):
            chunked_cosine_nearest_neighbor(torch.zeros_like(f), f)
        with self.assertRaises(ValueError):
            chunked_cosine_nearest_neighbor(f, torch.ones(3, 3))
        with self.assertRaises(ValueError):
            token_correlation_penalty(f[:1], f[:1], 1)
        with self.assertRaisesRegex(ValueError, 'preserve'):
            dense_token_objective((f, f), (f, f), (f[:2], f), (f, f), 1)
        with self.assertRaisesRegex(ValueError, 'common_dim'):
            dense_token_objective((f, f), (f, f), (f, f), (f, f), 2)


if __name__ == '__main__':
    unittest.main()
