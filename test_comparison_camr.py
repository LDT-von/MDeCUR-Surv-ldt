"""Known-answer and gradient ownership checks for explicit CAMR-2M adaptation."""
import unittest

import torch

from comparison_camr import CAMR2M, TwoWayLowRankFusion, difference_loss, METHOD_NAME


class CAMRComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setup_model(self):
        torch.manual_seed(17)
        model = CAMR2M(6, 8, bins=4, dropout=0.)
        return model, torch.randn(5, 11, 6), torch.randn(5, 8)

    def test_two_way_low_rank_known_answer_and_single_batch_shape(self):
        model = TwoWayLowRankFusion(2, rank=1).double()
        with torch.no_grad():
            model.rna_factor.fill_(1.)
            model.path_factor.fill_(2.)
            model.fusion_weights.fill_(3.)
            model.fusion_bias.fill_(4.)
        rna, path = torch.tensor([[1., 2.]], dtype=torch.float64), torch.tensor([[3., 4.]], dtype=torch.float64)
        output = model(rna, path)
        # (1+1+2) * 2*(1+3+4) * 3 + 4 for each output coordinate.
        torch.testing.assert_close(output, torch.full((1, 2), 196., dtype=torch.float64))

    def test_original_difference_loss_known_orthogonal_and_correlated(self):
        a = torch.tensor([[-1., 0.], [1., 0.]])
        orthogonal = torch.tensor([[0., -1.], [0., 1.]])
        # The upstream matrix energy penalizes cross-dimension association too.
        self.assertGreater(difference_loss(a, orthogonal).item(), .9)
        torch.testing.assert_close(difference_loss(a, a), difference_loss(a, orthogonal))
        zero = difference_loss(a, torch.ones_like(a))
        self.assertEqual(zero.item(), 0.)

    def test_tensor_and_list_inputs_agree_shapes_and_no_cna_parameters(self):
        model, patches, rna = self.setup_model()
        model.eval()
        first, second = model(patches, rna), model(list(patches), rna)
        self.assertEqual(first['logits'].shape, (5, 4))
        self.assertEqual(first['rna_reconstruction'].shape, (5, 80))
        self.assertEqual(first['path_common'].shape, (5, 60))
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
            self.assertTrue(torch.isfinite(first[key]).all())
        self.assertFalse(any('cna' in name.lower() for name, _ in model.named_parameters()))
        self.assertEqual(model.adaptation_metadata()['method'], METHOD_NAME)
        self.assertFalse(model.adaptation_metadata()['official_model_reproduction'])

    def test_all_three_auxiliary_terms_have_encoder_gradients_and_no_d_gradients(self):
        model, patches, rna = self.setup_model()
        output = model(patches, rna)
        losses = model.auxiliary_losses(output)
        self.assertEqual(set(losses), {'adversarial', 'reconstruction', 'orthogonality'})
        for name, loss in losses.items():
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            self.assertGreater(loss.item(), 0., name)
            for module in (model.patch_encoder, model.rna_adapter, model.common):
                norm = sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)
                self.assertGreater(norm, 0., (name, type(module).__name__))
            self.assertTrue(all(p.grad is None for p in model.discriminator.parameters()), name)

    def test_discriminator_step_detaches_encoders_and_optimizer_sets_are_disjoint(self):
        model, patches, rna = self.setup_model()
        main = list(model.main_parameters())
        discriminator = list(model.discriminator_parameters())
        self.assertFalse(set(map(id, main)) & set(map(id, discriminator)))
        self.assertEqual(set(map(id, main+discriminator)), set(map(id, model.parameters())))
        output = model(patches, rna)
        loss = model.discriminator_loss(output)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in main))
        self.assertGreater(sum(float(p.grad.square().sum()) for p in discriminator), 0.)

    def test_alternating_discriminator_then_main_step_is_finite(self):
        model, patches, rna = self.setup_model()
        main = torch.optim.AdamW(model.main_parameters(), lr=5e-4)
        discriminator = torch.optim.AdamW(model.discriminator_parameters(), lr=5e-4)
        initial_head = model.head.weight.detach().clone()
        output = model(patches, rna)
        discriminator.zero_grad(set_to_none=True)
        model.discriminator_loss(output).backward()
        discriminator.step()
        discriminator.zero_grad(set_to_none=True)
        # Auxiliary D forward uses the updated D weights without creating D grads.
        terms = model.auxiliary_losses(output)
        loss = output['logits'].square().mean()+sum(model.auxiliary_weights[name]*value
                                                  for name, value in terms.items())
        main.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.discriminator.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in main.param_groups[0]['params']))
        main.step()
        self.assertFalse(torch.equal(initial_head, model.head.weight))

    def test_padding_rows_visible_and_invalid_inputs_rejected(self):
        model, patches, rna = self.setup_model()
        model.eval()
        first = model(patches[:1], rna[:1])['path_input']
        padded = torch.cat((patches[:1], torch.zeros(1, 9, 6)), 1)
        second = model(padded, rna[:1])['path_input']
        self.assertGreater((first-second).abs().max().item(), 0.)
        for bad in (torch.zeros(5, 0, 6), torch.zeros(5, 11, 7), torch.full((5, 11, 6), float('nan'))):
            with self.assertRaises(ValueError):
                model(bad, rna)
        with self.assertRaises(ValueError):
            model(patches, torch.zeros(5, 7))


if __name__ == '__main__':
    unittest.main()
