"""Upstream-execution checks for the two-modality CAMR arm.

These tests assert that the CAMR arm really runs the untouched upstream
classes from ``papers/comparators/CAMR`` and that the protocol adaptations
(placeholder slot, discrete-hazard head, discriminator ownership) behave as
documented.
"""
import unittest

import torch

import comparison_sources as sources
from comparison_camr import CAMR2M, LATENT_DIM, METHOD_NAME


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

    def test_the_arm_executes_the_unmodified_upstream_classes(self):
        model, _, _ = self.setup_model()
        upstream = sources.load_camr_modules()
        self.assertIsInstance(model.upstream_model, upstream['networks'].MisaLMFGatedRec)
        self.assertIsInstance(model.discriminator, upstream['fusion'].Discriminator)
        self.assertIsInstance(model.diff_loss, upstream['utils'].DiffLoss)
        self.assertIsInstance(model.mse, upstream['utils'].MSE)
        root = str(sources.UPSTREAM['CAMR']['root'])
        self.assertIn(root, upstream['networks'].__file__)
        self.assertIn(root, upstream['fusion'].__file__)
        self.assertIn(root, upstream['utils'].__file__)

    def test_upstream_checkout_is_still_byte_identical(self):
        sources.verify_upstream('CAMR')

    def test_forward_uses_the_upstream_thirteen_outputs_and_discrete_logits(self):
        model, patches, rna = self.setup_model()
        model.eval()
        output = model(patches, rna)
        self.assertEqual(output['logits'].shape, (5, 4))
        self.assertEqual(output['rna_common'].shape, (5, LATENT_DIM))
        self.assertEqual(output['path_common'].shape, (5, LATENT_DIM))
        self.assertEqual(output['rna_reconstruction'].shape, (5, 80))
        self.assertEqual(output['path_reconstruction'].shape, (5, 80))
        for value in output.values():
            self.assertTrue(torch.isfinite(value).all())
        # The upstream module returns 13 tensors; the adapter keeps all of them.
        raw = model.upstream_model(output['rna_input'], output['path_input'],
                                   model.placeholder.expand(5, -1))
        self.assertEqual(len(raw), 13)

    def test_placeholder_slot_carries_no_patient_information(self):
        model, patches, rna = self.setup_model()
        model.eval()
        first = model(patches, rna)
        second = model(patches*0., rna*0.+1.)
        torch.testing.assert_close(first['placeholder_common'], second['placeholder_common'],
                                   atol=0, rtol=0)
        torch.testing.assert_close(first['placeholder_input'], second['placeholder_input'],
                                   atol=0, rtol=0)

    def test_auxiliary_terms_have_encoder_gradients_and_no_discriminator_gradient(self):
        model, patches, rna = self.setup_model()
        output = model(patches, rna)
        losses = model.auxiliary_losses(output)
        self.assertEqual(set(losses),
                         {'adversarial', 'reconstruction', 'orthogonality',
                          'weight_regularization'})
        for name, loss in losses.items():
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            self.assertTrue(torch.isfinite(loss).all(), name)
            modules = ((model.patch_encoder, model.rna_adapter, model.upstream_model.common)
                       if name != 'weight_regularization' else (model.upstream_model,))
            for module in modules:
                norm = sum(float(p.grad.square().sum())
                           for p in module.parameters() if p.grad is not None)
                self.assertGreater(norm, 0., (name, type(module).__name__))
            self.assertTrue(all(p.grad is None for p in model.discriminator.parameters()), name)

    def test_discriminator_step_detaches_the_encoders(self):
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
        initial_head = model.upstream_model.classifier.weight.detach().clone()
        output = model(patches, rna)
        discriminator.zero_grad(set_to_none=True)
        model.discriminator_loss(output).backward()
        discriminator.step()
        discriminator.zero_grad(set_to_none=True)
        terms = model.auxiliary_losses(output)
        loss = output['logits'].square().mean()+sum(model.auxiliary_weights[name]*value
                                                    for name, value in terms.items())
        main.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.discriminator.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in main.param_groups[0]['params']))
        main.step()
        self.assertFalse(torch.equal(initial_head, model.upstream_model.classifier.weight))

    def test_padding_rows_stay_visible_and_invalid_inputs_are_rejected(self):
        model, patches, rna = self.setup_model()
        model.eval()
        first = model(patches[:1], rna[:1])['path_input']
        padded = torch.cat((patches[:1], torch.zeros(1, 9, 6)), 1)
        second = model(padded, rna[:1])['path_input']
        self.assertGreater((first-second).abs().max().item(), 0.)
        for bad in (torch.zeros(5, 0, 6), torch.zeros(5, 11, 7),
                    torch.full((5, 11, 6), float('nan'))):
            with self.assertRaises(ValueError):
                model(bad, rna)
        with self.assertRaises(ValueError):
            model(patches, torch.zeros(5, 7))

    def test_metadata_declares_the_deviations(self):
        model, _, _ = self.setup_model()
        metadata = model.adaptation_metadata()
        self.assertEqual(metadata['method'], METHOD_NAME)
        self.assertFalse(metadata['official_model_reproduction'])
        self.assertFalse(metadata['fake_CNA_used'])
        self.assertIn('MisaLMFGatedRec', metadata['upstream_modules_executed'])


if __name__ == '__main__':
    unittest.main()
