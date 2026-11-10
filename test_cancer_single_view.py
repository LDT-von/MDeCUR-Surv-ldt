"""Synthetic single-view mmap, complete-bag math and test-isolation checks."""
from copy import deepcopy
from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cancer_model import CancerDeCUR, piecewise_nll, survival_at
from cancer_single_view import PTPatientBag
from cancer_stream import stream_forward, train_stream_model, predict_stream, validate_stream_views
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
from cancer_train import CancerConfig
from test_uni_views import create_cache, write_csv


def fixture(root):
    rows = []
    generator = torch.Generator().manual_seed(31)
    for i in range(6):
        torch.save(torch.randn(3+i, 6, generator=generator), root/f'slide{i}.pt')
        np.save(root/f'rna{i}.npy', np.arange(5, dtype=np.float32)+i)
        rows.append(dict(patient_id=f'fake{i}', split='train' if i < 3 else 'val' if i < 5 else 'test',
                         wsi_pt=f'slide{i}.pt', rna_npy=f'rna{i}.npy',
                         time=0. if i == 0 else .5+.2*i, event=i % 2))
    manifest = root/'patients.csv'
    write_csv(manifest, rows)
    return manifest, rows


def baseline_config(**kwargs):
    values = dict(path_views='single_view', patch_keep=1., common_mode='off',
                  lambda_within=0., lambda_unique=0., lambda_common=0., rna_noise=0.,
                  prediction_space='encoder', hidden=8, common=4, epochs=2)
    values.update(kwargs)
    return CancerConfig(**values)


class CancerSingleViewTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bag(self, patient, cell, root=None):
        bag = PTPatientBag(patient, cell, root or self.root)
        self.addCleanup(bag.close)
        return bag

    def load(self, manifest, **kwargs):
        data = load_stream_manifest(manifest, path_views='single_view', **kwargs)
        for cohort in data.values():
            for bag in cohort['bags']:
                self.addCleanup(bag.close)
        return data

    def test_complete_ordered_multislide_reads_are_bounded_copies(self):
        a = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        b = -torch.arange(18, dtype=torch.float32).reshape(3, 6)
        torch.save(a, self.root/'a.pt')
        torch.save(b, self.root/'b.pt')
        original = torch.load
        with patch('cancer_single_view.torch.load', wraps=original) as load:
            bag = self.bag('fake', json.dumps(['b.pt', 'a.pt']))
        for call in load.call_args_list:
            self.assertEqual(call.kwargs, dict(map_location='cpu', weights_only=True, mmap=True))
        tokens = list(bag.blocks(2))
        self.assertEqual([t[2]-t[1] for t in tokens], [2, 1, 2, 2, 1])
        result = torch.cat([bag.read_block(t, ('features_reference',))[0] for t in tokens])
        self.assertTrue(torch.equal(result, torch.cat([b, a])))
        self.assertEqual((bag.n_patches, bag.feature_dim), (8, 6))
        result.fill_(999)
        self.assertTrue(torch.equal(bag.read_block(tokens[0], ('features_reference',))[0], b[:2]))
        self.assertEqual(bag.provenance[0]['encoder_status'], 'unverified')
        modified = bag.provenance
        modified[0]['rows'] = -1
        self.assertEqual(bag.provenance[0]['rows'], 3)

    def test_no_fake_views_bad_blocks_and_closed_bags(self):
        torch.save(torch.ones(4, 6), self.root/'a.pt')
        bag = self.bag('fake', 'a.pt')
        token = next(bag.blocks(2))
        for names in ((), ('features_a',), ('features_reference', 'features_reference')):
            with self.assertRaisesRegex(ValueError, 'reference only'):
                bag.read_block(token, names)
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                list(bag.blocks(size))
        for bad in (['a.pt', 0, 1], ('missing', 0, 1), (token[0], -1, 1),
                    (token[0], 0, 5), (token[0], True, 2)):
            with self.assertRaises(ValueError):
                bag.read_block(bad, ('features_reference',))
        bag.close()
        with self.assertRaisesRegex(ValueError, 'closed'):
            bag.read_block(token, ('features_reference',))

    def test_unsupported_serialization_and_objects_have_no_eager_fallback(self):
        torch.save(torch.ones(2, 6), self.root/'old.pt', _use_new_zipfile_serialization=False)
        with self.assertRaisesRegex(ValueError, 'no eager fallback'):
            self.bag('fake', 'old.pt')
        for i, value in enumerate(({'features': torch.ones(2, 6)}, torch.ones(2, 6, dtype=torch.int64),
                                   torch.ones(2, 6, dtype=torch.complex64), torch.empty(0, 6), torch.ones(6))):
            torch.save(value, self.root/f'bad{i}.pt')
            with self.assertRaisesRegex(ValueError, 'one nonempty real floating'):
                self.bag('fake', f'bad{i}.pt')

    def test_last_chunk_nonfinite_and_float32_overflow_are_rejected(self):
        for i, value in enumerate((float('nan'), 1e100)):
            data = torch.ones(5, 6, dtype=torch.float64)
            data[-1, -1] = value
            torch.save(data, self.root/f'bad{i}.pt')
            bag = self.bag('fake', f'bad{i}.pt')
            tokens = list(bag.blocks(2))
            self.assertEqual(bag.read_block(tokens[0], ('features_reference',))[0].dtype, torch.float32)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                bag.read_block(tokens[-1], ('features_reference',))

    def test_source_change_is_detected_during_checkpoint_backward(self):
        torch.save(torch.randn(7, 6), self.root/'a.pt')
        bag = self.bag('fake', 'a.pt')
        model = CancerDeCUR(6, 5, hidden=8, common=4, bins=2, prediction_space='encoder')
        output = stream_forward(model, [bag], torch.ones(1, 5), chunk_size=3)
        previous = (self.root/'a.pt').stat()
        os.utime(self.root/'a.pt', ns=(previous.st_atime_ns, previous.st_mtime_ns+1_000_000))
        with self.assertRaisesRegex(ValueError, 'changed'):
            output['log_rates'].square().sum().backward()

    def test_validation_only_does_not_open_test_features_rna_or_parse_labels(self):
        manifest, rows = fixture(self.root)
        rows[-1].update(wsi_pt='missing.pt', rna_npy='missing.npy', time='invalid', event='invalid')
        write_csv(manifest, rows)
        data = self.load(manifest, include_test=False)
        self.assertEqual(set(data), {'train', 'val'})
        self.assertEqual(data['train']['time'][0], 0)
        self.assertNotIn('patches', data['train'])
        with self.assertRaises(FileNotFoundError):
            self.load(manifest)
        write_csv(manifest, rows[:-1])
        self.assertEqual(set(self.load(manifest, include_test=False)), {'train', 'val'})

    def test_manifest_rejects_patient_slide_and_dimension_conflicts(self):
        manifest, rows = fixture(self.root)
        for key, value in (('patient_id', 'fake0'), ('wsi_pt', 'slide0.pt'), ('event', 'nan'),
                           ('time', '-1'), ('split', 'unknown')):
            changed = deepcopy(rows)
            changed[-1][key] = value
            write_csv(manifest, changed)
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.load(manifest)
        write_csv(manifest, rows)
        torch.save(torch.ones(5, 7), self.root/'different.pt')
        rows[-1]['wsi_pt'] = 'different.pt'
        write_csv(manifest, rows)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.load(manifest)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.bag('fake', json.dumps(['slide0.pt', 'slide0.pt']))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.bag('fake', json.dumps(['slide0.pt', 'different.pt']))

    def test_fingerprint_tracks_source_content_and_ignores_directory_moves(self):
        source = self.root/'source'
        source.mkdir()
        manifest, _ = fixture(source)
        original = self.load(manifest)
        expected = stream_cohort_fingerprint(original, path_views='single_view')
        destination = self.root/'moved'
        shutil.copytree(source, destination)
        moved = self.load(destination/'patients.csv')
        self.assertEqual(stream_cohort_fingerprint(moved, path_views='single_view'), expected)
        moved['train']['rna'][0, 0] += 1
        self.assertNotEqual(stream_cohort_fingerprint(moved, path_views='single_view'), expected)
        for data in moved.values():
            for bag in data['bags']:
                bag.close()
        torch.save(torch.full((3, 6), 99.), destination/'slide0.pt')
        changed = self.load(destination/'patients.csv')
        self.assertNotEqual(stream_cohort_fingerprint(changed, path_views='single_view'), expected)

    def test_full_bag_prediction_nll_gradients_and_update_match_dense(self):
        generator = torch.Generator().manual_seed(67)
        arrays = [torch.randn(n, 6, generator=generator) for n in (5, 8)]
        bags = []
        for i, array in enumerate(arrays):
            torch.save(array, self.root/f'{i}.pt')
            bags.append(self.bag(f'fake{i}', f'{i}.pt'))
        x = torch.zeros(2, 8, 6, dtype=torch.float64)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        for i, array in enumerate(arrays):
            x[i, :len(array)] = array
            mask[i, :len(array)] = True
        g = torch.randn(2, 5, generator=generator).double()
        starts = torch.tensor([0., 1.], dtype=torch.float64)
        time, event = torch.tensor([0., 1.4]), torch.tensor([0., 1.])
        torch.manual_seed(89)
        base = CancerDeCUR(6, 5, hidden=8, common=4, bins=2, prediction_space='encoder').double()
        dense = deepcopy(base)
        target = dense(x, g, mask)['log_rates']
        target_loss = piecewise_nll(target, time, event, starts)
        target_loss.backward()
        expected_gradients = {n: None if p.grad is None else p.grad.clone() for n, p in dense.named_parameters()}
        torch.optim.AdamW(dense.parameters(), lr=.002).step()
        for size, replay in ((1, True), (3, True), (99, False)):
            stream = deepcopy(base)
            actual = stream_forward(stream, bags, g, size, checkpoint_chunks=replay)['log_rates']
            loss = piecewise_nll(actual, time, event, starts)
            torch.testing.assert_close(actual, target, atol=1e-10, rtol=1e-8)
            torch.testing.assert_close(loss, target_loss, atol=1e-10, rtol=1e-8)
            loss.backward()
            for name, parameter in stream.named_parameters():
                expected = expected_gradients[name]
                if expected is None:
                    self.assertIsNone(parameter.grad)
                else:
                    torch.testing.assert_close(parameter.grad, expected, atol=1e-10, rtol=1e-8)
            torch.optim.AdamW(stream.parameters(), lr=.002).step()
            for actual_parameter, expected_parameter in zip(stream.parameters(), dense.parameters()):
                torch.testing.assert_close(actual_parameter, expected_parameter, atol=1e-10, rtol=1e-8)

    def test_baseline_objectives_are_explicit_and_time_zero_censoring_is_valid(self):
        cfg = baseline_config()
        self.assertFalse(any(validate_stream_views(cfg).values()))
        for changes in (dict(common_mode='uniform'), dict(lambda_local=.01), dict(lambda_unique=.01),
                        dict(lambda_within=.01), dict(lambda_path_within=.01), dict(lambda_rna_within=.01),
                        dict(rna_noise=.01), dict(patch_keep=.5)):
            with self.assertRaises(ValueError):
                validate_stream_views(replace(cfg, **changes))
        logits = torch.zeros(1, 2, requires_grad=True)
        loss = piecewise_nll(logits, torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0., 1.]))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_training_normalizes_from_train_only_and_never_requests_views(self):
        manifest, _ = fixture(self.root)
        data = self.load(manifest, include_test=False)
        data['val']['rna'] += 1000
        fitted = train_stream_model(data['train'], data['val'], torch.tensor([0., 1.]),
                                    baseline_config(), patient_batch_size=1, patch_chunk_size=2)
        torch.testing.assert_close(fitted['rna_mean'], data['train']['rna'].mean(0))
        torch.testing.assert_close(fitted['rna_scale'], data['train']['rna'].std(0, unbiased=False).clamp_min(1e-6))
        self.assertEqual(fitted['stream_config']['correlation_scope'], 'disabled')
        for history in fitted['history']:
            self.assertEqual(history['batch_patient_counts'], [1, 1, 1])
            self.assertTrue(np.isfinite(history['validation_nll']))
            for batch in history['batches']:
                self.assertEqual(batch['total_loss'], batch['train_nll'])
        output = predict_stream(fitted, {k: v for k, v in data['val'].items() if k not in ('time', 'event')})
        self.assertEqual(tuple(output['log_rates'].shape), (2, 2))

    def command(self, manifest, destination, validation_only=False):
        command = [sys.executable, '-B', 'train_cancer.py', '--manifest', str(manifest),
                   '--out', str(destination), '--starts', '0', '1', '--horizon', '1',
                   '--time-unit', 'months', '--endpoint', 'synthetic OS', '--mode', 'off',
                   '--training-backend', 'stream', '--path-views', 'single_view',
                   '--prediction-space', 'encoder', '--lambda-within', '0', '--lambda-common', '0',
                   '--lambda-unique', '0', '--rna-noise', '0', '--patient-batch-size', '2',
                   '--patch-chunk-size', '2', '--hidden', '8', '--common', '4', '--epochs', '2']
        if validation_only:
            command.append('--validation-only')
        return command

    def run_cli(self, command):
        return subprocess.run(command, cwd=Path(__file__).parent,
                              env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                              capture_output=True, text=True)

    def test_validation_only_cli_skips_invalid_test_inputs_and_checkpoint_reloads(self):
        manifest, rows = fixture(self.root)
        rows[-1].update(wsi_pt='ABSENT.pt', rna_npy='ABSENT.npy', time='INVALID', event='INVALID')
        write_csv(manifest, rows)
        destination = self.root/'validation_run'
        command = self.command(manifest, destination, True)
        result = self.run_cli(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((destination/'report.json').read_text())
        self.assertFalse(report['test_evaluated'])
        self.assertEqual(report['evaluation_split'], 'val')
        self.assertEqual(report['patients'], {'train': 3, 'val': 2})
        self.assertEqual(report['cohort_fingerprint_scheme'], 'single-view-pt-stream-v1')
        self.assertNotIn('test_nll', report)
        self.assertNotIn('c_index_risk_at_horizon', report)
        self.assertFalse((destination/'test_predictions.csv').exists())
        self.assertTrue(all(value == 0 for value in report['effective_loss_weights'].values()))
        self.assertIn('unverified', report['feature_source'])
        data = self.load(manifest, include_test=False)
        self.assertEqual(report['cohort_input_sha256'], stream_cohort_fingerprint(data, 'single_view'))
        saved = torch.load(destination/'checkpoint.pt', map_location='cpu', weights_only=True)
        self.assertTrue(saved['validation_only'])
        model = CancerDeCUR(**saved['model_kwargs'])
        model.load_state_dict(saved['model_state'], strict=True)
        fitted = dict(model=model, rna_mean=saved['rna_mean'], rna_scale=saved['rna_scale'],
                      stream_config=saved['stream_config'], starts=saved['starts'])
        predicted = predict_stream(fitted, data['val'])
        expected = survival_at(predicted['log_rates'], torch.tensor([1.]), saved['starts'])[:, 0]
        with (destination/'validation_predictions.csv').open(newline='') as handle:
            output = list(csv.DictReader(handle))
        self.assertEqual([row['patient_id'] for row in output], data['val']['ids'])
        torch.testing.assert_close(torch.tensor([float(row['survival_at_horizon']) for row in output]), expected)
        before = (destination/'report.json').read_bytes()
        self.assertNotEqual(self.run_cli(command).returncode, 0)
        self.assertEqual((destination/'report.json').read_bytes(), before)

    def test_full_cli_still_evaluates_explicit_test(self):
        manifest, _ = fixture(self.root)
        destination = self.root/'full_run'
        result = self.run_cli(self.command(manifest, destination))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((destination/'report.json').read_text())
        self.assertTrue(report['test_evaluated'])
        self.assertEqual(report['evaluation_split'], 'test')
        self.assertIn('test_nll', report)
        self.assertTrue((destination/'test_predictions.csv').exists())

    def test_cli_rejects_auxiliary_and_padded_misconfiguration_before_manifest_io(self):
        command = self.command(self.root/'absent.csv', self.root/'out', True)
        command[command.index('--lambda-within')+1] = '.001'
        result = self.run_cli(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('survival-only baseline', result.stderr)
        command[command.index('--lambda-within')+1] = '0'
        command[command.index('--training-backend')+1] = 'padded'
        result = self.run_cli(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('require --training-backend stream', result.stderr)

    def test_cached_validation_only_keeps_strict_verification_and_skips_test(self):
        manifest, rows = fixture(self.root)
        for i, row in enumerate(rows):
            del row['wsi_pt']
            if row['split'] == 'test':
                row.update(wsi_h5='ABSENT.h5', time='INVALID', rna_npy='ABSENT.npy')
            else:
                path, _ = create_cache(self.root, f'cache{i}', row['patient_id'])
                row['wsi_h5'] = path.name
        write_csv(manifest, rows)
        data = load_stream_manifest(manifest, include_test=False)
        self.assertEqual(set(data), {'train', 'val'})
        self.assertTrue(all(b.feature_dim == 1024 for d in data.values() for b in d['bags']))
        with self.assertRaises(FileNotFoundError):
            load_stream_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
