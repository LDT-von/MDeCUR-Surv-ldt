"""Same-input plans, real synthetic training, and best/last comparison guards."""
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch

from cancer_parity import dct_bin_edges, _json_float
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
import comparison_fold0 as compare


class Fold0ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'PYTHONHASHSEED': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)
        g = torch.Generator().manual_seed(18)
        rows = []
        for i in range(16):
            torch.save(torch.randn(4+i % 3, 6, generator=g), self.root/f'p{i}.pt')
            np.save(self.root/f'g{i}.npy', torch.randn(8, generator=g).numpy())
            split = 'train' if i < 12 else 'val'
            t = i+1 if i < 12 else [2.5, 4.5, 7.5, 10.5][i-12]
            rows.append([f'p{i}', split, f'p{i}.pt', f'g{i}.npy', t, 1])
        self.manifest, self.reference = self.root/'manifest.csv', self.root/'survival_reference.csv'
        with self.manifest.open('w', newline='', encoding='utf-8') as handle:
            w = csv.writer(handle)
            w.writerow(['patient_id', 'split', 'wsi_pt', 'rna_npy', 'time', 'event'])
            w.writerows(rows)
        with self.reference.open('w', newline='', encoding='utf-8') as handle:
            w = csv.writer(handle)
            w.writerow(['patient_id', 'split', 'time', 'event'])
            w.writerows([r[0], r[1], r[4], r[5]] for r in rows)
        cohort = load_stream_manifest(self.manifest, 'single_view', include_test=False)
        fingerprint = stream_cohort_fingerprint(cohort, 'single_view')
        compare.close(cohort)
        self.ref_run = self.root/'old_reference'
        self.ref_run.mkdir()
        # Checkpoint bytes are pinned only; the plan never unpickles them.
        (self.ref_run/'checkpoint.pt').write_bytes(b'synthetic-reference-file')
        self.original = {'endpoint': 'DSS', 'time_unit': 'months', 'n_bins': 4, 'alpha_surv': .5,
                         'num_patches': 4096, 'config': {'seed': 3, 'epochs': 30, 'rna_scale_policy': 'none'},
                         'history': [{}]*30, 'effective_loss_weights': {'any': 0.},
                         'manifest_sha256': compare.digest(self.manifest), 'cohort_input_sha256': fingerprint,
                         'patients': {'train': 12, 'val': 4}, 'events': {'train': 12, 'val': 4},
                         'bin_edges': [_json_float(x) for x in dct_bin_edges(np.arange(1, 13), np.ones(12))],
                         'model_kwargs': dict(path_dim=6, rna_dim=8, pathway_indices=[[0, 1, 2], [3, 4, 5]],
                                              hidden=16, common=8, local_dim=8, local_common=4, bins=4,
                                              interaction_strength=.2, match_temperature=.2)}
        self.save_report()

    def save_report(self):
        (self.ref_run/'report.json').write_text(json.dumps(self.original), encoding='utf-8')

    def plan(self):
        result = compare.make_plan(self.manifest, self.reference, self.ref_run)
        path = self.root/'plan.json'
        compare.write_new(path, result)
        return path, compare.digest(path)

    def test_plan_uses_completed_reference_and_exact_train_bins(self):
        path, _ = self.plan()
        plan = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(plan['settings']['fold'], 0)
        self.assertEqual(plan['settings']['epochs'], 30)
        self.assertEqual(plan['inputs']['patients'], {'train': 12, 'val': 4})
        self.assertEqual(plan['inputs']['bin_edges'], self.original['bin_edges'])
        self.assertEqual(plan['available_models'], ['mdecur_aux_off', 'camr_2m'])
        self.assertIn('murrenet', plan['unavailable_models'])

    def test_test_rows_rejected_before_opening_missing_feature(self):
        with self.manifest.open('a', encoding='utf-8') as handle:
            handle.write('sealed,test,DO_NOT_OPEN.pt,DO_NOT_OPEN.npy,SEALED,SEALED\n')
        with self.assertRaisesRegex(ValueError, 'test rejected BEFORE'):
            compare.make_plan(self.manifest, self.reference, self.ref_run)

    def test_both_models_train_on_identical_sampling_and_save_best_last(self):
        path, sha = self.plan()
        reports = []
        for model_id in compare.MODELS:
            out = self.root/model_id
            with redirect_stdout(io.StringIO()):
                result = compare.run_model(path, sha, model_id, out, smoke=True)
            reports.append(result)
            self.assertEqual(result['epochs_completed'], 1)
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(result['last_epoch'], 1)
            self.assertEqual(result['best_validation'], result['last_validation'])
            self.assertTrue((out/'checkpoint_best.pt').is_file())
            self.assertTrue((out/'checkpoint_last.pt').is_file())
            self.assertEqual((out/'val_predictions_best.csv').read_bytes(), (out/'val_predictions_last.csv').read_bytes())
        self.assertEqual(reports[0]['training_sampling_sha256'], reports[1]['training_sampling_sha256'])
        self.assertIn('adversarial', reports[1]['history'][0]['train_terms'])
        self.assertTrue(reports[1]['metadata']['fake_CNA_used'] is False)
        with self.assertRaisesRegex(ValueError, 'incomplete/smoke'):
            compare.summarize([self.root/m for m in compare.MODELS], self.root/'summary.json')

    def test_input_changes_and_unavailable_models_refused(self):
        path, sha = self.plan()
        for model_id in ('murrenet', 'mohd'):
            with self.assertRaisesRegex(ValueError, 'implementation'):
                compare.run_model(path, sha, model_id, self.root/model_id)
        np.save(self.root/'g0.npy', np.zeros(8, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'inputs differ'):
            compare.run_model(path, sha, 'camr_2m', self.root/'bad')

    def test_report_mismatch_and_reference_label_mismatch_refused(self):
        self.original['config']['seed'] = 4
        self.save_report()
        with self.assertRaisesRegex(ValueError, 'seed3'):
            compare.make_plan(self.manifest, self.reference, self.ref_run)
        self.original['config']['seed'] = 3
        self.save_report()
        text = self.reference.read_text(encoding='utf-8').replace('p0,train,1,1', 'p0,train,100,1')
        self.reference.write_text(text, encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            compare.make_plan(self.manifest, self.reference, self.ref_run)

    def test_second_training_load_must_still_match_locked_plan(self):
        path, sha = self.plan()
        original = compare.load_inputs
        calls = 0

        def change_between_reads(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.original['model_kwargs']['interaction_strength'] = .7
                self.save_report()
            return original(*args)

        with patch.object(compare, 'load_inputs', side_effect=change_between_reads):
            with self.assertRaisesRegex(ValueError, 'between plan verification'):
                compare.run_model(path, sha, 'mdecur_aux_off', self.root/'raced')
        self.assertFalse((self.root/'raced').exists())

    def test_summary_reports_both_selections_and_does_not_choose_per_metric_peak(self):
        runs = []
        for model, best_c, last_c in [('mdecur_aux_off', .73, .70), ('camr_2m', .72, .71)]:
            out = self.root/model
            out.mkdir()
            metric = lambda c: {'c_index': c, 'c_index_ipcw': .51, 'validation_nll': .9,
                                 'brier': [.1, .2], 'ibs': .15, 'iauc': .8,
                                 'metric_times': [8., 15.], 'errors': []}
            compare.write_new(out/'report.json', {'model_id': model, 'plan_sha256': 'same', 'smoke_only': False,
                'epochs_completed': 30, 'test_evaluated': False, 'training_sampling_sha256': 'equal',
                'best_epoch': 20, 'last_epoch': 30, 'best_validation': metric(best_c), 'last_validation': metric(last_c)})
            runs.append(out)
        result = compare.summarize(runs, self.root/'summary.json')
        self.assertEqual(result['Harrell_C_rankings']['best'], ['mdecur_aux_off', 'camr_2m'])
        self.assertEqual(result['Harrell_C_rankings']['last'], ['camr_2m', 'mdecur_aux_off'])
        self.assertEqual([r['epoch'] for r in result['rows']], [20, 30, 20, 30])
        with self.assertRaises(FileExistsError):
            compare.summarize(runs, self.root/'summary.json')
        damaged = json.loads((runs[0]/'report.json').read_text(encoding='utf-8'))
        damaged['last_epoch'] = 29
        (runs[0]/'report.json').write_text(json.dumps(damaged), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'incomplete/smoke'):
            compare.summarize(runs, self.root/'bad_summary.json')

    def test_cli_plan_and_separate_process_run_share_hash_seed(self):
        script = Path(compare.__file__)
        plan_path, output = self.root/'cli_plan.json', self.root/'cli_camr'
        env = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        p = subprocess.run([sys.executable, str(script), 'plan', '--manifest', str(self.manifest),
                            '--survival-reference', str(self.reference), '--reference-run', str(self.ref_run),
                            '--out', str(plan_path)], capture_output=True, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        p = subprocess.run([sys.executable, str(script), 'run', '--plan', str(plan_path),
                            '--plan-sha256', compare.digest(plan_path), '--model', 'camr_2m',
                            '--device', 'cpu', '--smoke', '--out', str(output)],
                           capture_output=True, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue((output/'report.json').exists())


if __name__ == '__main__':
    unittest.main()
