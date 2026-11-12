"""Known-answer gradients, unchanged training API, and same-graph audits."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

import cancer_pathway_stream as stream
from pathway_gradient_audit import audit_batch, relation
from research_gradient_study import setup


class PathwayGradientAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_analytic_aligned_conflicting_and_actual_small_sgd_step(self):
        survival = (torch.tensor([1., 0.]),)
        for sign, direction in ((1., 'aligned'), (-1., 'conflicting')):
            auxiliary = (torch.tensor([sign, 1.]),)
            result = relation(survival, auxiliary, weight=.1)
            self.assertEqual(result['direction'], direction)
            self.assertAlmostEqual(result['cosine'], sign/2**.5)
            self.assertAlmostEqual(result['weighted_norm_ratio_to_survival'], .1*2**.5)
            actual_delta = .5*(1-1e-5*.1*sign)**2-.5
            self.assertEqual(actual_delta < 0, result['weighted_dot'] > 0)
        orthogonal = relation(survival, (torch.tensor([0., 1.]),))
        self.assertEqual(orthogonal['direction'], 'orthogonal')
        # A globally aligned direction may conflict inside a particular module.
        gs = (torch.tensor([10.]), torch.tensor([1.]))
        ga = (torch.tensor([1.]), torch.tensor([-5.]))
        self.assertEqual(relation(gs, ga)['direction'], 'aligned')
        self.assertEqual(relation(gs, ga, indices=[1])['direction'], 'conflicting')

    def test_missing_dependencies_and_zero_tensors_are_distinct(self):
        one, zero = torch.ones(2), torch.zeros(2)
        self.assertEqual(relation((None,), (one,))['status'], 'no_survival_dependency')
        self.assertEqual(relation((one,), (None,))['status'], 'no_auxiliary_dependency_in_survival_space')
        self.assertEqual(relation((zero,), (one,))['status'], 'zero_survival_gradient')
        self.assertEqual(relation((one,), (zero,))['status'], 'zero_auxiliary_gradient')
        mixed = relation((one, None), (one, 1000*one))
        self.assertAlmostEqual(mixed['cosine'], 1.)
        self.assertGreater(mixed['outside_survival_graph_raw_auxiliary_norm'], 1000)

    def test_optional_components_preserve_default_loss_rng_and_backward(self):
        first, batch, cfg = setup(17, n=6, small=True)
        second = deepcopy(first)
        torch.manual_seed(231)
        old_loss, old_stats = stream.pathway_objective(first, batch, torch.tensor([0.]), cfg, 4, 'feature_noise', 92)
        old_loss.backward()
        after_old = torch.get_rng_state()
        torch.manual_seed(231)
        total, new_stats, components = stream.pathway_objective(second, batch, torch.tensor([0.]), cfg,
                                                               4, 'feature_noise', 92, return_components=True)
        total.backward()
        torch.testing.assert_close(old_loss, total, rtol=0, atol=0)
        self.assertEqual(old_stats, new_stats)
        self.assertTrue(torch.equal(after_old, torch.get_rng_state()))
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertEqual(a.grad is None, b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        self.assertIn('survival', components)

    def test_one_forward_reconstruction_and_caller_state_preserved(self):
        model, batch, cfg = setup(17, n=6, small=True)
        cfg = replace(cfg, common_mode='uniform')
        model.base.head.weight.grad = torch.ones_like(model.base.head.weight)
        state = deepcopy(model.state_dict())
        grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        with patch('cancer_pathway_stream.pathway_stream_forward', wraps=stream.pathway_stream_forward) as forward:
            result = audit_batch(model, batch, torch.tensor([0.]), cfg, 4, 'feature_noise', 92, 31)
            self.assertEqual(forward.call_count, 1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        for p, previous in zip(model.parameters(), grads):
            self.assertEqual(p.grad is None, previous is None)
            if previous is not None:
                torch.testing.assert_close(p.grad, previous, rtol=0, atol=0)
        self.assertLess(result['gradient_reconstruction']['relative_l2_error'], 1e-5)
        self.assertEqual(len(result['terms']), 11)
        common = result['terms']['global_common']
        self.assertEqual(common['modules']['global_path_projector']['status'], 'no_survival_dependency')
        self.assertEqual(common['modules']['survival_head']['status'], 'no_auxiliary_dependency_in_survival_space')

    def test_skipped_disabled_and_connected_zero_loss_gradient(self):
        model, batch, cfg = setup(17, n=6, small=True)
        cfg = replace(cfg, matched_common=0., matched_unique=0.)
        with torch.no_grad():
            model.base.path_projector.weight.zero_()
            model.base.path_projector.bias.zero_()
        result = audit_batch(model, batch, torch.tensor([0.]), cfg, 4, 'feature_noise', 92, 31)
        self.assertEqual(result['terms']['global_common']['status'], 'skipped')
        self.assertEqual(result['terms']['matched_common']['status'], 'disabled')
        within = result['terms']['global_path_within']
        self.assertEqual(within['status'], 'computed')
        self.assertEqual(within['survival_graph_space']['status'], 'zero_auxiliary_gradient')

    def test_zero_interaction_has_connected_but_zero_survival_gradient(self):
        model, batch, cfg = setup(29, n=6, small=True)
        model.interaction_strength = 0.
        cfg = replace(cfg, interaction_strength=0., common_mode='uniform')
        result = audit_batch(model, batch, torch.tensor([0.]), cfg, 4, 'feature_noise', 92, 31)
        matching = result['terms']['matched_common']['modules']['matching_path']
        self.assertEqual(matching['status'], 'zero_survival_gradient')
        self.assertIsNone(matching['cosine'])


if __name__ == '__main__':
    unittest.main()
