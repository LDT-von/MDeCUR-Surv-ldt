"""Synthetic unimodal controls, ignored-input invariance and branch skipping."""
from copy import deepcopy
from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from cancer_model import CancerDeCUR, piecewise_nll, survival_at
from cancer_train import CancerConfig, train_cancer_model, predict, validate_modality_config
from cancer_stream import stream_forward, train_stream_model, predict_stream, validate_stream_views
from cancer_stream_data import load_stream_manifest
from cancer_single_view import PTPatientBag
from test_cancer_model import cohort
from test_cancer_single_view import fixture, baseline_config


class TensorBag:
    def __init__(self, values):
        self.values = values
        self.n_patches, self.feature_dim = values.shape

    def blocks(self, size):
        for start in range(0, self.n_patches, size):
            yield ('synthetic', start, min(start+size, self.n_patches))

    def read_block(self, token, names):
        if tuple(names) != ('features_reference',):
            raise AssertionError('Unexpected views in a unimodal control')
        return (self.values[token[1]:token[2]],)


class CancerModalityTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_default_both_matches_legacy_formula_state_and_rng(self):
        for space in ('projected', 'encoder'):
            torch.manual_seed(101)
            old_kwargs = dict(path_dim=5, rna_dim=6, hidden=8, common=4, bins=2, prediction_space=space)
            default = CancerDeCUR(**old_kwargs)
            rng = torch.random.get_rng_state().clone()
            torch.manual_seed(101)
            explicit = CancerDeCUR(**old_kwargs, modality='both')
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            for key, value in default.state_dict().items():
                self.assertTrue(torch.equal(value, explicit.state_dict()[key]))
            data = cohort(102, 3, 'fake')
            features = default.patch_encoder(data['patches'])
            attention = default.attention(features).squeeze(-1).softmax(1)
            hp = (attention[:, :, None]*features).sum(1)
            hr = default.rna_encoder(data['rna'])
            p, r = (default.path_projector(hp), default.rna_projector(hr)) if space == 'projected' else (hp, hr)
            expected = default.head(torch.cat((p, r), 1))
            self.assertTrue(torch.equal(default(data['patches'], data['rna'], data['mask'])['log_rates'], expected))
            # An old checkpoint with no modality constructor key remains strict-loadable.
            restored = CancerDeCUR(**old_kwargs)
            restored.load_state_dict(default.state_dict(), strict=True)
            self.assertTrue(torch.equal(restored(data['patches'], data['rna'], data['mask'])['log_rates'], expected))

    def test_dense_controls_skip_ignored_modules_and_ignore_their_values(self):
        data = cohort(103, 3, 'fake')
        def forbidden(*args):
            raise AssertionError('An omitted branch was executed')
        for modality in ('pathology', 'rna'):
            for space in ('projected', 'encoder'):
                torch.manual_seed(104)
                model = CancerDeCUR(5, 6, hidden=8, common=4, bins=2, modality=modality, prediction_space=space)
                ignored = [model.rna_encoder, model.rna_projector] if modality == 'pathology' else [
                    model.patch_encoder, model.attention, model.path_projector]
                handles = [module.register_forward_pre_hook(forbidden) for module in ignored]
                try:
                    out = model(data['patches'], data['rna'], data['mask'])
                    changed_x, changed_rna = data['patches'].clone(), data['rna'].clone()
                    if modality == 'pathology':
                        changed_rna.fill_(float('nan'))
                        self.assertTrue(torch.equal(out['rna_hidden'], torch.zeros_like(out['rna_hidden'])))
                    else:
                        changed_x.fill_(float('nan'))
                        self.assertTrue(torch.equal(out['path_hidden'], torch.zeros_like(out['path_hidden'])))
                        self.assertTrue(torch.equal(out['attention'], torch.zeros_like(out['attention'])))
                    changed = model(changed_x, changed_rna, data['mask'])
                    self.assertTrue(torch.equal(out['log_rates'], changed['log_rates']))
                    out['log_rates'].square().mean().backward()
                    for module in ignored:
                        self.assertTrue(all(p.grad is None for p in module.parameters()))
                    selected = model.patch_encoder if modality == 'pathology' else model.rna_encoder
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in selected.parameters()))
                    inactive = slice(8, None) if modality == 'pathology' else slice(None, 8)
                    self.assertTrue(torch.equal(model.head.weight.grad[:, inactive], torch.zeros_like(model.head.weight.grad[:, inactive])))
                finally:
                    for handle in handles:
                        handle.remove()

    def test_stream_rna_only_never_enumerates_or_reads_patch_blocks(self):
        model = CancerDeCUR(5, 6, hidden=8, common=4, bins=2, modality='rna', prediction_space='encoder')
        bag = TensorBag(torch.randn(7, 5))
        rna = torch.randn(1, 6)
        with patch.object(bag, 'blocks', side_effect=AssertionError('PT blocks were enumerated')):
            out = stream_forward(model, [bag], rna, chunk_size=2)
        expected = model.head(torch.cat((torch.zeros(1, 8), model.rna_encoder(rna)), 1))
        self.assertTrue(torch.equal(out['log_rates'], expected))
        out['log_rates'].sum().backward()
        self.assertTrue(all(p.grad is None for p in model.patch_encoder.parameters()))

    def test_dense_stream_unimodal_outputs_losses_and_gradients_match(self):
        torch.manual_seed(105)
        arrays = [torch.randn(n, 5, dtype=torch.float64) for n in (4, 7)]
        bags = [TensorBag(x) for x in arrays]
        x = torch.zeros(2, 7, 5, dtype=torch.float64)
        mask = torch.zeros(2, 7, dtype=torch.bool)
        for i, array in enumerate(arrays):
            x[i, :len(array)] = array
            mask[i, :len(array)] = True
        rna = torch.randn(2, 6, dtype=torch.float64)
        time, event, starts = torch.tensor([.4, 1.4]), torch.tensor([1., 0.]), torch.tensor([0., 1.])
        for modality in ('pathology', 'rna'):
            for space in ('encoder', 'projected'):
                base = CancerDeCUR(5, 6, hidden=8, common=4, bins=2, modality=modality, prediction_space=space).double()
                dense = deepcopy(base)
                expected = dense(x, rna, mask)['log_rates']
                expected_loss = piecewise_nll(expected, time, event, starts)
                expected_loss.backward()
                for size, replay in ((1, True), (3, True), (99, False)):
                    stream = deepcopy(base)
                    actual = stream_forward(stream, bags, rna, size, checkpoint_chunks=replay)['log_rates']
                    loss = piecewise_nll(actual, time, event, starts)
                    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
                    torch.testing.assert_close(loss, expected_loss, atol=1e-10, rtol=1e-8)
                    loss.backward()
                    for a, b in zip(stream.parameters(), dense.parameters()):
                        if b.grad is None:
                            self.assertIsNone(a.grad)
                        else:
                            torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-8)

    def test_configuration_and_auxiliary_calls_reject_unimodal_partners(self):
        for modality in ('pathology', 'rna'):
            cfg = baseline_config(modality=modality)
            self.assertFalse(any(validate_stream_views(cfg).values()))
            for changes in (dict(lambda_within=.01), dict(lambda_unique=.01), dict(lambda_local=.01),
                            dict(common_mode='uniform'), dict(rna_noise=.01)):
                with self.assertRaisesRegex(ValueError, 'Single-modality'):
                    validate_modality_config(replace(cfg, **changes))
            model = CancerDeCUR(5, 6, hidden=8, common=4, bins=2, modality=modality)
            with self.assertRaisesRegex(ValueError, 'modality=both'):
                stream_forward(model, [TensorBag(torch.randn(3, 5))], torch.randn(1, 6), include_views=True)
            with self.assertRaisesRegex(ValueError, 'modality=both'):
                model.encode_auxiliary(torch.randn(1, 3, 5), torch.randn(1, 6), torch.ones(1, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            CancerDeCUR(5, 6, modality='unknown')
        with self.assertRaises(ValueError):
            validate_modality_config(baseline_config(modality='unknown'))

    def test_padded_training_saves_modality_and_ignored_input_does_not_change_prediction(self):
        train, val = cohort(106, 6, 'train'), cohort(107, 3, 'val')
        for modality in ('pathology', 'rna'):
            cfg = CancerConfig(hidden=8, common=4, epochs=2, modality=modality, common_mode='off',
                               lambda_within=0, lambda_unique=0, rna_noise=0, rna_scale_policy='mask')
            fitted = train_cancer_model(train, val, torch.tensor([0., 1.]), cfg)
            self.assertEqual(fitted['model_kwargs']['modality'], modality)
            first = predict(fitted, val)['log_rates']
            changed = dict(val)
            field = 'rna' if modality == 'pathology' else 'patches'
            changed[field] = val[field]*10000
            self.assertTrue(torch.equal(first, predict(fitted, changed)['log_rates']))

    def test_stream_rna_only_trains_without_feature_block_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = fixture(root)
            data = load_stream_manifest(manifest, 'single_view', include_test=False)
            try:
                with patch.object(PTPatientBag, 'read_block', side_effect=AssertionError('Unused features read')):
                    fitted = train_stream_model(data['train'], data['val'], torch.tensor([0., 1.]),
                                                baseline_config(modality='rna', rna_scale_policy='mask'), 2, 2)
                    prediction = predict_stream(fitted, data['val'])
                self.assertEqual(fitted['model_kwargs']['modality'], 'rna')
                self.assertEqual(prediction['log_rates'].shape, (2, 2))
                self.assertTrue(all(p.grad is None for p in fitted['model'].patch_encoder.parameters()))
            finally:
                for split in data.values():
                    for bag in split['bags']:
                        bag.close()

    def test_cli_controls_share_cohort_fingerprint_and_reload_selected_modality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = fixture(root)
            reports = []
            data = load_stream_manifest(manifest, 'single_view', include_test=False)
            try:
                for modality in ('both', 'pathology', 'rna'):
                    destination = root/modality
                    command = [sys.executable, '-B', 'train_cancer.py', '--manifest', str(manifest), '--out', str(destination),
                               '--training-backend', 'stream', '--path-views', 'single_view', '--validation-only',
                               '--mode', 'off', '--lambda-within', '0', '--lambda-common', '0', '--lambda-unique', '0',
                               '--rna-noise', '0', '--rna-scale-policy', 'mask', '--modality', modality,
                               '--epochs', '2', '--patience', '2', '--hidden', '8', '--common', '4',
                               '--patient-batch-size', '2', '--patch-chunk-size', '2', '--prediction-space', 'encoder',
                               '--starts', '0', '1', '--horizon', '1', '--time-unit', 'months', '--endpoint', 'synthetic']
                    result = subprocess.run(command, cwd=Path(__file__).parent,
                                            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads((destination/'report.json').read_text())
                    reports.append(report)
                    self.assertEqual(report['modality'], modality)
                    self.assertFalse(report['test_evaluated'])
                    self.assertFalse((destination/'test_predictions.csv').exists())
                    saved = torch.load(destination/'checkpoint.pt', map_location='cpu', weights_only=True)
                    self.assertEqual(saved['model_kwargs']['modality'], modality)
                    model = CancerDeCUR(**saved['model_kwargs'])
                    model.load_state_dict(saved['model_state'], strict=True)
                    fitted = dict(model=model, starts=saved['starts'], stream_config=saved['stream_config'],
                                  **{k:saved[k] for k in ('rna_mean','rna_scale','rna_mask','rna_preprocessing')})
                    prediction = predict_stream(fitted, data['val'])
                    expected = survival_at(prediction['log_rates'], torch.tensor([1.]), saved['starts'])[:, 0]
                    with (destination/'validation_predictions.csv').open(newline='') as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual([r['patient_id'] for r in rows], data['val']['ids'])
                    torch.testing.assert_close(torch.tensor([float(r['survival_at_horizon']) for r in rows]), expected)
                self.assertEqual(len({r['cohort_input_sha256'] for r in reports}), 1)
                self.assertEqual(len({json.dumps(r['rna_preprocessing'], sort_keys=True) for r in reports}), 1)
            finally:
                for split in data.values():
                    for bag in split['bags']:
                        bag.close()

    def test_cli_rejects_unimodal_auxiliaries_before_input_io(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, '-B', 'train_cancer.py', '--manifest', str(root/'absent.csv'),
                       '--out', str(root/'out'), '--training-backend', 'stream', '--path-views', 'cached_uni',
                       '--modality', 'rna', '--mode', 'uniform', '--starts', '0', '--horizon', '1',
                       '--time-unit', 'months', '--endpoint', 'synthetic']
            result = subprocess.run(command, cwd=Path(__file__).parent, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Single-modality', result.stderr)


if __name__ == '__main__':
    unittest.main()
