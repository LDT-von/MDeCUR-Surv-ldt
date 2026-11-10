"""Known-answer RNA scale/mask and saved-inference checks, synthetic only."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from cancer_model import CancerDeCUR, survival_at
from cancer_rna import fit_rna_scaler, transform_rna, rna_noise_view
from cancer_train import CancerConfig, train_cancer_model, predict, view_batch
from cancer_stream import train_stream_model, predict_stream
from cancer_stream_data import load_stream_manifest
from test_cancer_model import cohort
from test_cancer_single_view import fixture, baseline_config


class CancerRNATest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_constant_and_low_variance_mask_prevents_heldout_floor_explosion(self):
        train = torch.tensor([[-218.3546, 0., 0.], [-218.3546, 1., 1e-8], [-218.3546, 2., 2e-8]])
        heldout = torch.tensor([[-1., 3., 1.]])
        before = train.clone()
        legacy = fit_rna_scaler(train)
        self.assertGreater(transform_rna(heldout, legacy).abs().max().item(), 1e8)
        fixed = fit_rna_scaler(train, 'mask', 1e-6)
        self.assertEqual(fixed['rna_mask'].tolist(), [False, True, False])
        expected = torch.tensor([[0., 2./np.sqrt(2./3.), 0.]], dtype=torch.float32)
        torch.testing.assert_close(transform_rna(heldout, fixed), expected)
        self.assertTrue(torch.equal(before, train))
        self.assertEqual(fixed['rna_preprocessing']['masked_gene_indices'], [0, 2])
        self.assertEqual(fixed['rna_preprocessing']['input_genes'], 3)

    def test_transform_never_refits_or_clips_active_values(self):
        train = torch.tensor([[5., 0.], [5., 1.], [5., 2.]])
        fitted = fit_rna_scaler(train, 'mask')
        saved = deepcopy(fitted)
        output = transform_rna(torch.tensor([[1e30, 100.], [-1e30, -100.]]), fitted)
        self.assertTrue(torch.equal(output[:, 0], torch.zeros(2)))
        self.assertGreater(output[:, 1].abs().max().item(), 100.)
        for name in ('rna_mean', 'rna_scale', 'rna_mask', 'rna_train_std'):
            self.assertTrue(torch.equal(fitted[name], saved[name]))
        self.assertEqual(fitted['rna_preprocessing'], saved['rna_preprocessing'])

    def test_legacy_formula_and_old_checkpoint_remain_identical(self):
        torch.manual_seed(42)
        train = torch.randn(5, 7)
        train[:, 0] = 3.
        heldout = torch.randn(3, 7)
        fitted = fit_rna_scaler(train)
        mean, scale = train.mean(0), train.std(0, unbiased=False).clamp_min(1e-6)
        expected = (heldout-mean)/scale
        self.assertTrue(torch.equal(transform_rna(heldout, fitted), expected))
        self.assertTrue(torch.equal(transform_rna(heldout, dict(rna_mean=mean, rna_scale=scale)), expected))

    def test_invalid_configuration_and_incomplete_masked_checkpoint_fail(self):
        train = torch.tensor([[0., 1.], [0., 2.]])
        for policy, threshold in (('unknown', 1e-6), ('mask', 0.), ('mask', -1.), ('mask', float('nan'))):
            with self.assertRaises(ValueError):
                fit_rna_scaler(train, policy, threshold)
        with self.assertRaisesRegex(ValueError, 'No training-variable'):
            fit_rna_scaler(torch.ones(3, 4), 'mask')
        fitted = fit_rna_scaler(train, 'mask')
        for mask in (None, torch.ones(2), torch.ones(3, dtype=torch.bool)):
            broken = deepcopy(fitted)
            if mask is None:
                del broken['rna_mask']
            else:
                broken['rna_mask'] = mask
            with self.assertRaisesRegex(ValueError, 'mask'):
                transform_rna(train, broken)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            transform_rna(torch.ones(2, 3), fitted)
        with self.assertRaises(ValueError):
            transform_rna(torch.tensor([[float('nan'), 1.]]), fitted)

    def test_auxiliary_noise_keeps_excluded_genes_zero(self):
        g = torch.zeros(3, 4)
        active = torch.tensor([False, True, True, False])
        torch.manual_seed(71)
        view = rna_noise_view(g, .5, active)
        self.assertTrue(torch.equal(view[:, ~active], torch.zeros(3, 2)))
        self.assertTrue((view[:, active] != 0).any())
        x, mask = torch.ones(3, 2, 5), torch.ones(3, 2, dtype=torch.bool)
        cfg = CancerConfig(rna_noise=.5)
        for _, rna, _ in view_batch(x, g, mask, cfg, rna_mask=active):
            self.assertTrue(torch.equal(rna[:, ~active], torch.zeros(3, 2)))
        cfg = CancerConfig(path_views='cached_uni', patch_keep=1, rna_noise=.5)
        for _, rna, _ in view_batch(x, g, mask, cfg, (x, x), rna_mask=active):
            self.assertTrue(torch.equal(rna[:, ~active], torch.zeros(3, 2)))

    def test_padded_training_and_inference_use_the_same_fixed_mask(self):
        train, val = cohort(32, 6, 'train'), cohort(33, 3, 'val')
        train['rna'][:, 0] = -218.3546
        val['rna'][:, 0] = -1.
        cfg = CancerConfig(hidden=8, common=4, epochs=2, common_mode='off',
                           lambda_within=0., lambda_unique=0., rna_scale_policy='mask')
        fitted = train_cancer_model(train, val, torch.tensor([0., 1.]), cfg)
        self.assertFalse(fitted['rna_mask'][0])
        first = predict(fitted, val)['log_rates']
        changed = dict(val, rna=val['rna'].clone())
        changed['rna'][:, 0] = 1e30
        self.assertTrue(torch.equal(first, predict(fitted, changed)['log_rates']))
        self.assertEqual(fitted['model_kwargs']['rna_dim'], 6)

    def test_stream_mask_and_selected_checkpoint_ignore_masked_heldout_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = fixture(root)
            data = load_stream_manifest(manifest, 'single_view', include_test=False)
            try:
                data['train']['rna'][:, 0] = -218.3546
                data['val']['rna'][:, 0] = -1.
                cfg = baseline_config(rna_scale_policy='mask')
                a = train_stream_model(data['train'], data['val'], torch.tensor([0., 1.]), cfg, 2, 2)
                predicted = predict_stream(a, data['val'])['log_rates']
                data['val']['rna'][:, 0] = 1e30
                b = train_stream_model(data['train'], data['val'], torch.tensor([0., 1.]), cfg, 2, 2)
                self.assertEqual(a['history'], b['history'])
                self.assertTrue(torch.equal(predicted, predict_stream(a, data['val'])['log_rates']))
                self.assertFalse(a['rna_mask'][0])
                for name, value in a['model'].state_dict().items():
                    self.assertTrue(torch.equal(value, b['model'].state_dict()[name]))
            finally:
                for partition in data.values():
                    for bag in partition['bags']:
                        bag.close()

    def test_cli_saves_mask_diagnostics_and_reloads_validation_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = fixture(root)
            for i in range(5):
                rna = np.load(root/f'rna{i}.npy')
                rna[0] = -218.3546 if i < 3 else -1.
                np.save(root/f'rna{i}.npy', rna)
            destination = root/'masked_run'
            command = [sys.executable, '-B', 'train_cancer.py', '--manifest', str(manifest), '--out', str(destination),
                       '--training-backend', 'stream', '--path-views', 'single_view', '--validation-only',
                       '--mode', 'off', '--lambda-within', '0', '--lambda-common', '0', '--lambda-unique', '0',
                       '--rna-noise', '0', '--rna-scale-policy', 'mask', '--rna-min-std', '0.000001',
                       '--epochs', '2', '--patience', '2', '--hidden', '8', '--common', '4',
                       '--patient-batch-size', '2', '--patch-chunk-size', '2', '--prediction-space', 'encoder',
                       '--starts', '0', '1', '--horizon', '1', '--time-unit', 'months', '--endpoint', 'synthetic']
            result = subprocess.run(command, cwd=Path(__file__).parent,
                                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((destination/'report.json').read_text())
            self.assertFalse(report['test_evaluated'])
            self.assertEqual(report['config']['patience'], 2)
            self.assertEqual(report['rna_preprocessing']['masked_gene_indices'], [0])
            self.assertLess(report['rna_preprocessing']['validation_max_abs'], 10.)
            saved = torch.load(destination/'checkpoint.pt', map_location='cpu', weights_only=True)
            self.assertEqual(saved['rna_mask'].dtype, torch.bool)
            self.assertEqual(saved['rna_mask'].tolist(), [False, True, True, True, True])
            model = CancerDeCUR(**saved['model_kwargs'])
            model.load_state_dict(saved['model_state'], strict=True)
            fitted = dict(model=model, starts=saved['starts'], stream_config=saved['stream_config'],
                          **{k:saved[k] for k in ('rna_mean','rna_scale','rna_mask','rna_preprocessing')})
            data = load_stream_manifest(manifest, 'single_view', include_test=False)
            try:
                actual = predict_stream(fitted, data['val'])
                expected = survival_at(actual['log_rates'], torch.tensor([1.]), saved['starts'])[:, 0]
                with (destination/'validation_predictions.csv').open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                torch.testing.assert_close(expected, torch.tensor([float(row['survival_at_horizon']) for row in rows]))
            finally:
                for partition in data.values():
                    for bag in partition['bags']:
                        bag.close()


if __name__ == '__main__':
    unittest.main()
