"""Streaming CLI and replay guards on synthetic H5 data, never real patients."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

from cancer_model import CancerDeCUR, survival_at
from cancer_stream import predict_stream, stream_forward
from cancer_stream_data import H5PatientBag, load_stream_manifest, stream_cohort_fingerprint
from test_uni_views import create_cache, write_csv


class CancerStreamCLITest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_stream_cli_reports_batches_fingerprint_and_reload_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(8):
                cache, _ = create_cache(root, f'slide{i}', f'fake{i}')
                np.save(root/f'rna{i}.npy', np.arange(5, dtype=np.float32) + i)
                rows.append({'patient_id': f'fake{i}',
                             'split': 'train' if i < 5 else 'val' if i < 7 else 'test',
                             'wsi_h5': cache.name, 'rna_npy': f'rna{i}.npy',
                             'time': .5 + .2*i, 'event': i % 2})
            manifest = root/'patients.csv'
            write_csv(manifest, rows)
            destination = root/'stream_output'
            command = [sys.executable, '-B', 'train_cancer.py', '--manifest', str(manifest),
                       '--out', str(destination), '--starts', '0', '1', '--horizon', '1',
                       '--time-unit', 'years', '--endpoint', 'synthetic', '--mode', 'uniform',
                       '--training-backend', 'stream', '--path-views', 'cached_uni',
                       '--prediction-space', 'encoder', '--lambda-local', '.001',
                       '--patient-batch-size', '2', '--patch-chunk-size', '2',
                       '--hidden', '8', '--common', '4', '--epochs', '2']
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            result = subprocess.run(command, cwd=Path(__file__).parent, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report_path = destination/'report.json'
            report_before = report_path.read_bytes()
            report = json.loads(report_before)
            self.assertEqual(report['training_backend'], 'stream')
            self.assertEqual(report['cohort_fingerprint_scheme'], 'verified-cache-stream-v1')
            self.assertIsNone(report['cohort_tensor_sha256'])
            self.assertEqual(report['patients'], {'train': 5, 'val': 2, 'test': 1})
            self.assertEqual(report['stream_config']['correlation_scope'], 'patient_minibatch')
            for history in report['history']:
                # The final singleton joins the preceding batch: no patient is discarded.
                self.assertEqual(history['batch_patient_counts'], [2, 3])
                self.assertEqual(sum(history['batch_patient_counts']), 5)
                for batch in history['batches']:
                    self.assertEqual(batch['local_diagnostics']['patients'], batch['patients'])
                    self.assertEqual(batch['local_diagnostics']['matched_patches'], 4*batch['patients'])
                    self.assertIsNotNone(batch['local_loss'])
            data = load_stream_manifest(manifest)
            self.assertEqual(report['cohort_input_sha256'], stream_cohort_fingerprint(data))
            checkpoint = torch.load(destination/'checkpoint.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['training_backend'], 'stream')
            self.assertEqual(checkpoint['stream_config']['patch_chunk_size'], 2)
            model = CancerDeCUR(**checkpoint['model_kwargs'])
            model.load_state_dict(checkpoint['model_state'], strict=True)
            fitted = dict(model=model, rna_mean=checkpoint['rna_mean'], rna_scale=checkpoint['rna_scale'],
                          starts=checkpoint['starts'], stream_config=checkpoint['stream_config'])
            unlabelled = {k: v for k, v in data['test'].items() if k not in ('time', 'event')}
            predicted = predict_stream(fitted, unlabelled)
            expected = survival_at(predicted['log_rates'], torch.tensor([1.]), checkpoint['starts'])[:, 0]
            with (destination/'test_predictions.csv').open(encoding='utf-8', newline='') as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual([row['patient_id'] for row in saved], data['test']['ids'])
            self.assertTrue(torch.allclose(expected, torch.tensor([float(row['survival_at_horizon']) for row in saved])))
            repeated = subprocess.run(command, cwd=Path(__file__).parent, env=env,
                                      capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(report_path.read_bytes(), report_before)

    def test_checkpoint_replay_detects_h5_change_after_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [create_cache(root, f'slide{i}', f'fake{i}')[0] for i in range(2)]
            bags = [H5PatientBag(f'fake{i}', path.name, root) for i, path in enumerate(paths)]
            model = CancerDeCUR(1024, 5, hidden=8, common=4, bins=2,
                                prediction_space='encoder', local_dim=3)
            outputs = stream_forward(model, bags, torch.randn(2, 5), chunk_size=2,
                                     include_local=True, include_views=True, checkpoint_chunks=True)
            loss = outputs['log_rates'].square().mean() + .001*outputs['local_loss']
            before = paths[0].stat()
            with h5py.File(paths[0], 'r+') as handle:
                handle.attrs['external_change_during_training'] = 1
            os.utime(paths[0], ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
            with self.assertRaisesRegex(ValueError, 'changed'):
                loss.backward()


if __name__ == '__main__':
    unittest.main()
