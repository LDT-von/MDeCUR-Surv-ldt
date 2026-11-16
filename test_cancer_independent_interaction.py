"""Known-answer checks for the frozen-reference token interaction candidate."""
from io import BytesIO
import unittest

import torch

from cancer_dense_model import CenteredTokenInteraction
from cancer_independent_interaction import IndependentCenteredInteraction


class IndependentInteractionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1403)
        self.model = IndependentCenteredInteraction(2, 3, 2).double()
        self.p = torch.randn(5, 4, 2, dtype=torch.double)
        self.g = torch.randn(5, 4, 2, dtype=torch.double)

    def test_reference_is_patient_weighted_and_padding_is_excluded(self):
        p = torch.tensor([[[1., 2.], [float('nan'), 0.]],
                          [[5., 6.], [7., 8.]]], dtype=torch.double)
        g = torch.tensor([[[4., 8.], [float('inf'), 0.]],
                          [[0., 4.], [2., 6.]]], dtype=torch.double)
        mask = torch.tensor([[True, False], [True, True]])
        self.model.fit_reference(p, g, mask, mask)
        # First average within patient, then across patients (not across patches).
        torch.testing.assert_close(self.model.path_reference, torch.tensor([3.5, 4.5], dtype=torch.double))
        torch.testing.assert_close(self.model.rna_reference, torch.tensor([2.5, 6.5], dtype=torch.double))
        self.assertEqual(int(self.model.reference_patients), 2)
        self.assertTrue(torch.isfinite(self.model(p, g, mask, mask)).all())

    def test_one_token_equals_existing_module_with_identical_fixed_references(self):
        self.model.fit_reference(self.p, self.g)
        old = CenteredTokenInteraction(2, 3, 2).double().eval()
        with torch.no_grad():
            old.path.weight.copy_(self.model.path.weight)
            old.rna.weight.copy_(self.model.rna.weight)
            old.head.weight.copy_(self.model.head.weight)
            old.path_reference.copy_(self.model.path_reference)
            old.rna_reference.copy_(self.model.rna_reference)
        p, g = self.p[:, :1].clone().requires_grad_(), self.g[:, :1].clone().requires_grad_()
        newer = self.model(p, g)
        older = old(p, g)
        torch.testing.assert_close(newer, older, rtol=0, atol=0)
        newgrad = torch.autograd.grad(newer.sum(), (p, g), retain_graph=True)
        oldgrad = torch.autograd.grad(older.sum(), (p, g))
        for new, prev in zip(newgrad, oldgrad):
            torch.testing.assert_close(new, prev, rtol=0, atol=0)

    def test_all_pairs_equals_explicit_token_cartesian_product(self):
        m = IndependentCenteredInteraction(2, 3, 2, pairing='all_pairs').double()
        g = self.g[:, :3]
        pm = torch.tensor([[True, True, False, False]]*5)
        gm = torch.tensor([[True, False, True]]*5)
        m.fit_reference(self.p, g, pm, gm)
        actual = m(self.p, g, pm, gm)
        expected = []
        for i in range(len(self.p)):
            pp = m.path(self.p[i, pm[i]]-m.path_reference)
            gg = m.rna(g[i, gm[i]]-m.rna_reference)
            expected.append(m.head((pp[:, None, :]*gg[None, :, :]).mean((0, 1))))
        torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-12, atol=1e-12)

    def test_paired_intersection_and_mask_are_not_pairwise_cartesian(self):
        self.model.fit_reference(self.p, self.g)
        pm = torch.tensor([[True, False, True, True]]*5)
        gm = torch.tensor([[False, True, True, True]]*5)
        actual = self.model(self.p, self.g, pm, gm)
        pp = self.model.path(self.p[:, 2:]-self.model.path_reference)
        gg = self.model.rna(self.g[:, 2:]-self.model.rna_reference)
        torch.testing.assert_close(actual, self.model.head((pp*gg).mean(1)))

    def test_paired_reference_uses_intersection_and_rejects_empty_pairs(self):
        pm = torch.tensor([[True, False, True, True]]*5)
        gm = torch.tensor([[False, True, True, True]]*5)
        self.model.fit_reference(self.p, self.g, pm, gm)
        torch.testing.assert_close(self.model.path_reference, self.p[:, 2:].mean((0, 1)))
        torch.testing.assert_close(self.model.rna_reference, self.g[:, 2:].mean((0, 1)))
        empty = IndependentCenteredInteraction(2, 3, 2).double()
        bad = torch.tensor([[True, False, False, False]]*5)
        other = torch.tensor([[False, True, False, False]]*5)
        with self.assertRaisesRegex(ValueError, 'token pair'):
            empty.fit_reference(self.p, self.g, bad, other)
        self.assertFalse(bool(empty.reference_fitted))
        with self.assertRaisesRegex(ValueError, 'equal token'):
            empty.fit_reference(self.p, self.g[:, :3])

    def test_product_reference_double_centering_identity(self):
        # Exact empirical double centering of the whole separable h, evaluated
        # on a Cartesian reference distribution with equal patient weights.
        p, g = self.p[:, :1], self.g[:, :1]
        self.model.fit_reference(p, g)
        def h(x, y):
            return self.model.head(self.model.path(x)*self.model.rna(y))
        for i in range(len(p)):
            pi, gi = p[i, 0], g[i, 0]
            all_p, all_g = p[:, 0], g[:, 0]
            joint = h(pi, gi)
            marginal_p = h(pi.expand_as(all_g), all_g).mean(0)
            marginal_g = h(all_p, gi.expand_as(all_p)).mean(0)
            all_pairs = h(all_p[:, None, :], all_g[None, :, :]).mean((0, 1))
            expected = joint-marginal_p-marginal_g+all_pairs
            torch.testing.assert_close(self.model(p[i:i+1], g[i:i+1])[0], expected,
                                       rtol=1e-12, atol=1e-12)

    def test_zero_residual_means_zero_interaction_and_gradients_flow(self):
        self.model.fit_reference(self.p, self.g)
        p = self.model.path_reference.expand_as(self.p).clone()
        self.assertTrue(torch.equal(self.model(p, self.g), torch.zeros(5, 2, dtype=torch.double)))
        p = self.p.clone().requires_grad_()
        g = self.g.clone().requires_grad_()
        self.model(p, g).square().mean().backward()
        for x in (p, g, self.model.path.weight, self.model.rna.weight, self.model.head.weight):
            self.assertIsNotNone(x.grad)
            self.assertGreater(float(x.grad.abs().sum()), 0.)

    def test_references_frozen_in_both_modes_and_saved_checkpoint_replays(self):
        self.model.fit_reference(self.p, self.g)
        references = (self.model.path_reference.clone(), self.model.rna_reference.clone())
        self.model.train()
        before = self.model(self.p, self.g).detach()
        self.model(self.p+100, self.g-30).sum().backward()
        self.model.eval()
        self.model(self.p-40, self.g+80)
        for x, y in zip(references, (self.model.path_reference, self.model.rna_reference)):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        with BytesIO() as stream:
            torch.save(self.model.state_dict(), stream)
            stream.seek(0)
            reloaded = IndependentCenteredInteraction(2, 3, 2).double()
            reloaded.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(reloaded(self.p, self.g), before, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            IndependentCenteredInteraction(2, 3, 2, pairing='all_pairs').double().load_state_dict(self.model.state_dict())

    def test_cannot_refit_or_predict_before_reference(self):
        with self.assertRaisesRegex(RuntimeError, 'fit_reference'):
            self.model(self.p, self.g)
        self.model.fit_reference(self.p, self.g)
        with self.assertRaisesRegex(RuntimeError, 'already fitted'):
            self.model.fit_reference(self.p+100, self.g)

    def test_invalid_masks_nonfinite_values_and_different_token_counts(self):
        bad = self.p.clone()
        bad[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.model.fit_reference(bad, self.g)
        self.assertFalse(bool(self.model.reference_fitted))
        self.model.fit_reference(self.p, self.g)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.model(self.p, self.g, torch.ones(5, 4))
        with self.assertRaisesRegex(ValueError, 'equal token'):
            self.model(self.p, self.g[:, :3])
        pm = torch.tensor([[True, False, False, False]]*5)
        gm = torch.tensor([[False, True, False, False]]*5)
        with self.assertRaisesRegex(ValueError, 'token pair'):
            self.model(self.p, self.g, pm, gm)

    def test_product_centering_does_not_identify_patient_synergy(self):
        # The marginal-product interaction has zero single-source expectation
        # under independent reference draws, but p=g gives a constant on the
        # correlated observed data; this does not establish patient synergy.
        m = IndependentCenteredInteraction(1, 1, 1).double()
        p = torch.tensor([[[-1.]], [[1.]]], dtype=torch.double)
        m.fit_reference(p, p)
        with torch.no_grad():
            m.path.weight.fill_(1)
            m.rna.weight.fill_(1)
            m.head.weight.fill_(1)
        torch.testing.assert_close(m(p, p), torch.ones(2, 1, dtype=torch.double))
        torch.testing.assert_close(m(p, -p), -torch.ones(2, 1, dtype=torch.double))


if __name__ == '__main__':
    unittest.main()
