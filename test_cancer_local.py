"""Synthetic checks of the local/global adaptation, never patient efficacy."""
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from cancer_auxiliary import patient_balanced_local_correlation
from cancer_data import load_manifest
from cancer_model import CancerDeCUR, piecewise_nll, survival_at
from cancer_train import CancerConfig, effective_loss_weights, predict, train_cancer_model
from model import correlation_loss, decur_auxiliary
from test_cancer_model import cohort
from test_uni_views import create_cache, write_csv


def local_fixture():
    a = torch.tensor([[[0., 1.], [2., 3.], [0., 0.]],
                      [[-2., 2.], [1., 0.], [4., -1.]]], dtype=torch.float64)
    b = torch.tensor([[[1., 2.], [3., 0.], [0., 0.]],
                      [[0., 3.], [2., -1.], [5., 1.]]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    return a, b, mask


def cached_cohort(seed=2, n=8, prefix='train'):
    data = cohort(seed, n, prefix)
    data['mask'][0, 2:] = False
    data['patches_view1'] = data['patches'] + .1 * torch.randn_like(data['patches'])
    data['patches_view2'] = data['patches'] + .1 * torch.randn_like(data['patches'])
    return data


class CancerLocalTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_encoder_prediction_skips_all_auxiliary_modules(self):
        data = cohort(5, 6, 'fake')
        model = CancerDeCUR(5, 6, hidden=8, common=4, bins=2,
                            prediction_space='encoder', local_dim=3)
        with (patch.object(model.path_projector, 'forward', side_effect=AssertionError('path auxiliary used')),
              patch.object(model.rna_projector, 'forward', side_effect=AssertionError('RNA auxiliary used')),
              patch.object(model.local_projector, 'forward', side_effect=AssertionError('local auxiliary used'))):
            out = model(data['patches'], data['rna'], data['mask'])
            piecewise_nll(out['log_rates'], data['time'], data['event'], torch.tensor([0., 1.])).backward()
        for module in (model.patch_encoder, model.rna_encoder, model.head):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
        for module in (model.path_projector, model.rna_projector, model.local_projector):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertTrue(torch.equal(out['path'], out['path_hidden']))
        self.assertTrue(torch.equal(out['rna'], out['rna_hidden']))

    def test_legacy_projection_route_and_checkpoint_remain_supported(self):
        data = cohort(6, 4, 'fake')
        model = CancerDeCUR(5, 6, hidden=8, common=4, bins=2)
        p, r, _ = model.encode(data['patches'], data['rna'], data['mask'])
        out = model(data['patches'], data['rna'], data['mask'])
        self.assertTrue(torch.equal(out['log_rates'], model.head(torch.cat((p, r), 1))))
        self.assertFalse(any(k.startswith('local_projector') for k in model.state_dict()))
        restored = CancerDeCUR(5, 6, hidden=8, common=4, bins=2)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(out['log_rates'], restored(data['patches'], data['rna'], data['mask'])['log_rates']))

    def test_local_branch_preserves_shared_initialization_and_rng(self):
        torch.manual_seed(18)
        legacy = CancerDeCUR(5, 6, hidden=8, common=4, bins=2)
        expected_random = torch.randn(7)
        torch.manual_seed(18)
        local = CancerDeCUR(5, 6, hidden=8, common=4, bins=2,
                           prediction_space='encoder', local_dim=3)
        self.assertTrue(torch.equal(expected_random, torch.randn(7)))
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, local.state_dict()[key]), key)

    def test_local_and_global_gradients_reach_prediction_backbones(self):
        data = cached_cohort()
        model = CancerDeCUR(5, 6, hidden=8, common=4, bins=2,
                            prediction_space='encoder', local_dim=3)
        a = model.encode_auxiliary(data['patches_view1'], data['rna'], data['mask'], include_local=True)
        b = model.encode_auxiliary(data['patches_view2'], data['rna'] + .02, data['mask'], include_local=True)
        local, _ = patient_balanced_local_correlation(a['local'], b['local'], data['mask'])
        local.backward(retain_graph=True)
        self.assertGreater(model.patch_encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.local_projector.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.attention.parameters()))
        self.assertTrue(all(p.grad is None for p in model.rna_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.head.parameters()))
        model.zero_grad(set_to_none=True)
        global_loss = correlation_loss(a['path'], b['path'], 1) + correlation_loss(a['rna'], b['rna'], 1)
        global_loss.backward()
        for module in (model.patch_encoder, model.attention, model.rna_encoder,
                       model.path_projector, model.rna_projector):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

    def test_patient_balanced_moments_match_independent_numpy_answer(self):
        a, b, mask = local_fixture()
        aa, bb = a[mask].numpy(), b[mask].numpy()
        q = np.array([1/4, 1/4, 1/6, 1/6, 1/6])
        ac, bc = aa - np.average(aa, axis=0, weights=q), bb - np.average(bb, axis=0, weights=q)
        av, bv = np.average(ac**2, axis=0, weights=q), np.average(bc**2, axis=0, weights=q)
        corr = (ac / np.sqrt(av + 1e-4)).T @ (q[:, None] * bc / np.sqrt(bv + 1e-4))
        expected = sum((corr[k, k] - 1)**2 for k in range(2)) + .005 * (corr[0, 1]**2 + corr[1, 0]**2)
        loss, stats = patient_balanced_local_correlation(a, b, mask)
        self.assertAlmostEqual(loss.item(), expected, places=12)
        self.assertEqual(stats['patients'], 2)
        self.assertEqual(stats['matched_patches'], 5)

    def test_duplicating_one_patients_patches_does_not_increase_its_weight(self):
        a, b, mask = local_fixture()
        repeated_a = pad_sequence([a[0, :2].repeat_interleave(3, dim=0), a[1]], batch_first=True)
        repeated_b = pad_sequence([b[0, :2].repeat_interleave(3, dim=0), b[1]], batch_first=True)
        repeated_mask = torch.arange(6)[None] < torch.tensor([6, 3])[:, None]
        expected = patient_balanced_local_correlation(a, b, mask)[0]
        actual = patient_balanced_local_correlation(repeated_a, repeated_b, repeated_mask)[0]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=0))
        flat_old = correlation_loss(a[mask], b[mask], 1)
        flat_new = correlation_loss(repeated_a[repeated_mask], repeated_b[repeated_mask], 1)
        self.assertFalse(torch.allclose(flat_old, flat_new))

    def test_padding_nan_and_feature_offsets_do_not_change_local_loss(self):
        a, b, mask = local_fixture()
        expected = patient_balanced_local_correlation(a, b, mask)[0]
        a = a + torch.tensor([100., -31.])
        b = b + torch.tensor([-83., 42.])
        a[~mask] = float('nan')
        b[~mask] = float('nan')
        a.requires_grad_()
        b.requires_grad_()
        actual = patient_balanced_local_correlation(a, b, mask)[0]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=0))
        actual.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())
        self.assertTrue((a.grad[~mask] == 0).all())
        self.assertTrue((b.grad[~mask] == 0).all())

    def test_local_loss_requires_corresponding_rows(self):
        torch.manual_seed(22)
        a = torch.randn(3, 9, 4, dtype=torch.float64)
        mask = torch.ones(3, 9, dtype=torch.bool)
        order = torch.arange(8, -1, -1)
        paired = patient_balanced_local_correlation(a, a, mask)[0]
        both_reordered = patient_balanced_local_correlation(a[:, order], a[:, order], mask)[0]
        mismatched = patient_balanced_local_correlation(a, a[:, order], mask)[0]
        self.assertTrue(torch.allclose(paired, both_reordered, atol=1e-12, rtol=0))
        self.assertGreater(mismatched.item(), paired.item() + .1)

    def test_local_gradients_match_finite_differences(self):
        a, b, mask = local_fixture()
        self.assertTrue(torch.autograd.gradcheck(
            lambda aa, bb: patient_balanced_local_correlation(aa, bb, mask)[0],
            (a.requires_grad_(), b.requires_grad_()), eps=1e-6, atol=1e-5, rtol=1e-3))

    def test_collapse_is_finite_and_reported_without_claiming_a_guarantee(self):
        a = torch.ones(2, 3, 4, requires_grad=True)
        loss, stats = patient_balanced_local_correlation(a, a, torch.ones(2, 3, dtype=torch.bool))
        self.assertEqual(loss.item(), 4.)
        self.assertEqual(stats['low_variance_dimensions'], 4)
        self.assertEqual(stats['variance_min'], 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_local_validation_rejects_empty_single_patient_and_nonfinite_rows(self):
        a, b, mask = local_fixture()
        with self.assertRaisesRegex(ValueError, 'two patients'):
            patient_balanced_local_correlation(a[:1], b[:1], mask[:1])
        empty = mask.clone()
        empty[0] = False
        with self.assertRaises(ValueError):
            patient_balanced_local_correlation(a, b, empty)
        b[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            patient_balanced_local_correlation(a, b, mask)

    def test_legacy_within_average_and_independent_overrides(self):
        cfg = CancerConfig(lambda_within=.3)
        weights = effective_loss_weights(cfg)
        self.assertEqual(weights['path_within'], .15)
        self.assertEqual(weights['rna_within'], .15)
        torch.manual_seed(24)
        p1, p2, r1, r2 = torch.randn(4, 7, 6, dtype=torch.float64)
        old = cfg.lambda_within * decur_auxiliary(p1, p2, r1, r2, 3)['within']
        new = weights['path_within'] * correlation_loss(p1, p2, 1) + weights['rna_within'] * correlation_loss(r1, r2, 1)
        self.assertTrue(torch.allclose(old, new, atol=1e-12, rtol=0))
        overrides = effective_loss_weights(replace(cfg, lambda_path_within=0., lambda_rna_within=.7))
        self.assertEqual(overrides['path_within'], 0.)
        self.assertEqual(overrides['rna_within'], .7)

    def test_training_logs_each_active_term_and_preserves_validation_selection(self):
        train, valid = cached_cohort(), cohort(3, 5, 'val')
        base = CancerConfig(hidden=8, common=4, epochs=3, patience=3,
                            prediction_space='encoder', path_views='cached_uni', patch_keep=1,
                            lambda_path_within=.003, lambda_rna_within=0.,
                            lambda_local=.002, local_dim=3, common_mode='uniform')
        for mode in ('off', 'uniform', 'stability', 'shuffled'):
            fitted = train_cancer_model(train, valid, torch.tensor([0., 1.]), replace(base, common_mode=mode))
            self.assertEqual(fitted['best_epoch'], min(fitted['history'], key=lambda h: h['validation_nll'])['epoch'])
            weights = fitted['effective_loss_weights']
            for h in fitted['history']:
                self.assertIsNone(h['rna_within_loss'])
                self.assertIsNotNone(h['path_within_loss'])
                self.assertIsNotNone(h['local_loss'])
                self.assertEqual(h['local_diagnostics']['patients'], 8)
                self.assertEqual(h['local_diagnostics']['matched_patches'], 30)
                expected = h['train_nll'] + sum(weights[key] * (h[field] or 0.) for key, field in (
                    ('path_within', 'path_within_loss'), ('rna_within', 'rna_within_loss'),
                    ('local', 'local_loss'), ('common', 'common_loss'), ('unique', 'unique_loss')))
                self.assertAlmostEqual(h['total_loss'], expected, places=5)
            with patch.object(fitted['model'], 'encode_auxiliary', side_effect=AssertionError('validation auxiliary used')):
                outputs = predict(fitted, {k: v for k, v in valid.items() if k not in ('time', 'event')})
            self.assertTrue(torch.isfinite(outputs['log_rates']).all())
        rna_only = train_cancer_model(train, valid, torch.tensor([0., 1.]), replace(
            base, lambda_local=0., lambda_path_within=0., lambda_rna_within=.003,
            lambda_unique=0., common_mode='off'))
        self.assertIsNone(rna_only['history'][0]['path_within_loss'])
        self.assertIsNotNone(rna_only['history'][0]['rna_within_loss'])

    def test_disabled_auxiliary_terms_use_only_reference_path(self):
        train, valid = cached_cohort(), cohort(3, 5, 'val')
        cfg = CancerConfig(hidden=8, common=4, epochs=2, prediction_space='encoder',
                           path_views='cached_uni', patch_keep=1, common_mode='off',
                           lambda_path_within=0., lambda_rna_within=0., lambda_unique=0.)
        with patch.object(CancerDeCUR, 'encode_auxiliary', side_effect=AssertionError('disabled auxiliary used')):
            first = train_cancer_model(train, valid, torch.tensor([0., 1.]), cfg)
            changed_views = dict(train, patches_view1=train['patches_view1'] + 50.,
                                 patches_view2=train['patches_view2'] - 50.)
            second = train_cancer_model(changed_views, valid, torch.tensor([0., 1.]), cfg)
        self.assertTrue(torch.equal(predict(first, valid)['log_rates'], predict(second, valid)['log_rates']))

    def test_configuration_rejects_unaligned_local_and_invalid_coefficients(self):
        train, valid = cached_cohort(), cohort(3, 5, 'val')
        base = CancerConfig(epochs=1, path_views='cached_uni', patch_keep=1)
        for cfg in (replace(base, path_views='legacy_patch_dropout', lambda_local=.1),
                    replace(base, patch_keep=.8, lambda_local=.1),
                    replace(base, lambda_path_within=-1), replace(base, lambda_rna_within=float('nan')),
                    replace(base, lambda_local=float('inf')), replace(base, local_dim=0),
                    replace(base, prediction_space='unknown')):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                train_cancer_model(train, valid, torch.tensor([0., 1.]), cfg)

    def test_cached_cli_roundtrip_with_local_encoder_and_independent_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for i in range(6):
                cache, _ = create_cache(root, f'slide{i}', f'fake{i}')
                np.save(root / f'rna{i}.npy', np.arange(5, dtype=np.float32) + i)
                rows.append({'patient_id': f'fake{i}', 'split': 'train' if i < 3 else 'val' if i < 5 else 'test',
                             'wsi_h5': cache.name, 'rna_npy': f'rna{i}.npy', 'time': .5 + i*.2, 'event': i % 2})
            manifest = root / 'patients.csv'
            write_csv(manifest, rows)
            command = [sys.executable, '-B', 'train_cancer.py', '--manifest', str(manifest),
                       '--out', str(root / 'out'), '--starts', '0', '1', '--horizon', '1',
                       '--time-unit', 'years', '--endpoint', 'synthetic', '--epochs', '2',
                       '--mode', 'uniform', '--path-views', 'cached_uni', '--prediction-space', 'encoder',
                       '--lambda-local', '.002', '--local-dim', '3', '--lambda-path-within', '.003',
                       '--lambda-rna-within', '0', '--rna-noise', '0']
            result = subprocess.run(command, cwd=Path(__file__).parent,
                                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'out' / 'report.json').read_text(encoding='utf-8'))
            checkpoint = torch.load(root / 'out' / 'checkpoint.pt', map_location='cpu', weights_only=True)
            self.assertEqual(report['model_kwargs']['prediction_space'], 'encoder')
            self.assertEqual(report['effective_loss_weights']['rna_within'], 0.)
            self.assertTrue(all(h['local_loss'] is not None for h in report['history']))
            model = CancerDeCUR(**checkpoint['model_kwargs'])
            model.load_state_dict(checkpoint['model_state'], strict=True)
            data = load_manifest(manifest, path_views='cached_uni')
            fitted = dict(model=model, rna_mean=checkpoint['rna_mean'], rna_scale=checkpoint['rna_scale'],
                          starts=checkpoint['starts'])
            with (root / 'out' / 'test_predictions.csv').open(encoding='utf-8', newline='') as handle:
                predictions = list(csv.DictReader(handle))
            logits = predict(fitted, {k: v for k, v in data['test'].items() if k not in ('time', 'event')})['log_rates']
            expected = survival_at(logits, torch.tensor([1.]), checkpoint['starts'])[:, 0]
            self.assertTrue(torch.allclose(expected, torch.tensor([float(r['survival_at_horizon']) for r in predictions])))
            self.assertNotIn('patches_view1', data['test'])
            self.assertEqual(checkpoint['model_kwargs'], report['model_kwargs'])


if __name__ == '__main__':
    unittest.main()
