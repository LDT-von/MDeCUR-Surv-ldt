"""Synthetic tests for the fold0 plus upstream merged ranking table.

The tests build two synthetic cohorts that describe the same patients, run the
real fold0 and five-fold drivers on CPU, and assert that the merged table:

* refuses to merge two plans that describe different patients or bins;
* rescores every arm with one shared metric call;
* reports the runner-to-runner offset through the shared bridge arm;
* never lets a runner-native NLL pass as the shared NLL;
* refuses smoke or test-evaluated runs.
"""
import csv
import io
import json
import os
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import comparison_merge as merge
import comparison_5fold as fivefold
import comparison_fold0 as fold0
from cancer_parity import DISCRETE_NLL_VERSION, discrete_survival, dct_bin_edges

# The merge refuses arms that ran a different number of epochs, so the tests
# use a short but identical, non-smoke epoch count for every arm.
TEST_EPOCHS = 2


class MergeTest(unittest.TestCase):
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
        os.environ['PYTHONHASHSEED'] = '0'
        generator = torch.Generator().manual_seed(77)
        rows, train_ids, val_ids = [], [], []
        for index in range(24):
            patches = (torch.randn(4100, 8, generator=generator)
                       + 3*torch.randn(1, 8, generator=generator))
            torch.save(patches, self.root/f'p{index}.pt')
            np.save(self.root/f'g{index}.npy', torch.randn(12, generator=generator).numpy())
            split = 'train' if index < 18 else 'val'
            if split == 'train':
                time = float(index+1)
                event = int(index % 3 != 0)
                train_ids.append(f'p{index}')
            else:
                time = [3.5, 9.5, 15.5, 21.5, 27.5, 30.5][index-18]
                event = [1, 0, 1, 1, 0, 1][index-18]
                val_ids.append(f'p{index}')
            rows.append([f'p{index}', split, f'p{index}.pt', f'g{index}.npy', time, event])
        self.train_ids, self.val_ids = train_ids, val_ids
        self.manifest = self.root/'manifest.csv'
        self.reference = self.root/'survival_reference.csv'
        with self.manifest.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['patient_id', 'split', 'wsi_pt', 'rna_npy', 'time', 'event'])
            writer.writerows(rows)
        with self.reference.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['patient_id', 'split', 'time', 'event'])
            writer.writerows([[r[0], r[1], r[4], r[5]] for r in rows])
        self.genes = self.root/'genes.txt'
        self.genes.write_text('\n'.join(f'GENE{i}' for i in range(12)), encoding='utf-8')
        self.gmt = self.root/'signatures.gmt'
        self.gmt.write_text('\n'.join(
            f'PATH{i}\tdesc{i}\t' + '\t'.join(f'GENE{(i*2+j) % 12}' for j in range(6))
            for i in range(4)), encoding='utf-8')
        self.deref_signatures = self.root/'deref_signatures.csv'
        with self.deref_signatures.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow([f'family{i}' for i in range(6)])
            for row in range(10):
                writer.writerow([f'GENE{(row+i) % 12}' for i in range(6)])

    # ------------------------------------------------------------------ plans
    def make_fivefold_plan(self):
        plan = fivefold.make_plan(0, self.manifest, self.reference, self.genes,
                                  self.gmt, self.deref_signatures, device='cpu')
        path = self.root/'fivefold_plan.json'
        fivefold.write_new(path, plan)
        return path, fivefold.digest(path)

    def make_fold0_plan(self):
        """Build a fold0 plan from the same cohort, with a matching reference run."""
        cohort = fold0.load_stream_manifest(self.manifest, 'single_view', include_test=False)
        try:
            train_time = np.asarray([t for t in cohort['train']['time']], dtype=float)
            train_event = np.asarray([t for t in cohort['train']['event']], dtype=float)
            edges = dct_bin_edges(train_time, train_event, 4)
            report = {'endpoint': 'DSS', 'time_unit': 'months', 'n_bins': 4,
                      'alpha_surv': .5, 'num_patches': 4096,
                      'config': {'seed': 3, 'epochs': 30, 'rna_scale_policy': 'none'},
                      'history': [{}]*30, 'effective_loss_weights': {'any': 0.},
                      'manifest_sha256': fold0.digest(self.manifest),
                      'cohort_input_sha256': fold0.stream_cohort_fingerprint(
                          cohort, 'single_view'),
                      'patients': {s: len(cohort[s]['ids']) for s in ('train', 'val')},
                      'events': {s: int(cohort[s]['event'].sum()) for s in ('train', 'val')},
                      'bin_edges': [fold0._json_float(x) for x in edges],
                      'model_kwargs': {'hidden': 16, 'common': 8, 'local_dim': 8,
                                       'local_common': 4, 'interaction_strength': .2,
                                       'match_temperature': 4.}}
            reference_run = self.root/'reference_run'
            reference_run.mkdir()
            (reference_run/'report.json').write_text(json.dumps(report), encoding='utf-8')
            (reference_run/'checkpoint.pt').write_bytes(b'synthetic-reference-file')
            reference = fold0.read_reference(self.reference, cohort)
            plan = fold0.make_plan(self.manifest, self.reference, str(reference_run))
        finally:
            fold0.close(cohort)
        path = self.root/'fold0_plan.json'
        fold0.write_new(path, plan)
        return path, fold0.digest(path)

    def run_fivefold_arms(self, plan_path, plan_sha, models=('camr_2m', 'pibd_2m')):
        directories = []
        for model_id in models:
            out = self.root/f'ff_{model_id}'
            with redirect_stdout(io.StringIO()):
                fivefold.run_model(plan_path, plan_sha, model_id, out, device='cpu',
                                   epochs=TEST_EPOCHS)
            directories.append(out)
        return directories

    def run_fold0_arms(self, plan_path, plan_sha, models=('camr_2m',)):
        directories = []
        for model_id in models:
            out = self.root/f'f0_{model_id}'
            with redirect_stdout(io.StringIO()):
                fold0.run_model(plan_path, plan_sha, model_id, out, device='cpu',
                                epochs=TEST_EPOCHS)
            directories.append(out)
        return directories

    # ------------------------------------------------------------------ tests
    def test_the_two_plans_describe_the_same_cohort(self):
        fold0_path, _ = self.make_fold0_plan()
        fivefold_path, _ = self.make_fivefold_plan()
        identity = merge.compare_identities(merge.read_json(fold0_path),
                                            merge.read_json(fivefold_path))
        self.assertEqual(identity['patients'], {'train': 18, 'val': 6})
        self.assertEqual(len(identity['bin_edges']), 5)

    def test_a_different_cohort_is_refused(self):
        fold0_path, _ = self.make_fold0_plan()
        fivefold_path, _ = self.make_fivefold_plan()
        plan = merge.read_json(fivefold_path)
        plan['inputs']['patients'] = {'train': 17, 'val': 6}
        with self.assertRaisesRegex(ValueError, 'different data'):
            merge.compare_identities(merge.read_json(fold0_path), plan)

    def test_different_bin_edges_are_refused(self):
        fold0_path, _ = self.make_fold0_plan()
        fivefold_path, _ = self.make_fivefold_plan()
        plan = merge.read_json(fivefold_path)
        edges = list(plan['inputs']['bin_edges'])
        edges[2] = 7.5
        plan['inputs']['bin_edges'] = edges
        with self.assertRaisesRegex(ValueError, 'different data'):
            merge.compare_identities(merge.read_json(fold0_path), plan)

    def test_the_merge_rescores_every_arm_with_one_metric_call(self):
        fold0_path, fold0_sha = self.make_fold0_plan()
        fivefold_path, fivefold_sha = self.make_fivefold_plan()
        fold0_runs = self.run_fold0_arms(fold0_path, fold0_sha)
        fivefold_runs = self.run_fivefold_arms(fivefold_path, fivefold_sha)
        out = self.root/'merged.json'
        result = merge.merge(fold0_path, fold0_runs, fivefold_path, fivefold_runs,
                             self.reference, out, required_epochs=TEST_EPOCHS)
        self.assertTrue(out.is_file())
        models = {row['arm'] for row in result['rows']}
        self.assertIn('camr_2m', models)
        self.assertIn('camr_2m [upstream runner]', models)
        self.assertIn('pibd_2m [upstream runner]', models)
        # Every row carries a metric that was recomputed here.
        for row in result['rows']:
            self.assertIsNotNone(row['Harrell_C'])
            self.assertIn('runner_metric_mismatch', row)
            self.assertEqual(row['runner_metric_mismatch'], [])
            self.assertTrue(row['NLL_definition'].startswith('patientwise discrete NLL'))
        # The bridge arm appears once per group, so the offset is measurable.
        self.assertTrue(result['bridge_arm']['available'])
        self.assertIn('delta_Harrell_C', result['bridge_arm']['best'])

    def test_the_bridge_arm_reports_the_runner_offset(self):
        fold0_path, fold0_sha = self.make_fold0_plan()
        fivefold_path, fivefold_sha = self.make_fivefold_plan()
        fold0_runs = self.run_fold0_arms(fold0_path, fold0_sha)
        fivefold_runs = self.run_fivefold_arms(fivefold_path, fivefold_sha)
        result = merge.merge(fold0_path, fold0_runs, fivefold_path, fivefold_runs,
                             self.reference, self.root/'merged.json',
                             required_epochs=TEST_EPOCHS)
        bridge = result['bridge_arm']
        self.assertEqual(bridge['arm'], 'camr_2m')
        self.assertIn('project_Harrell_C', bridge['best'])
        self.assertIn('upstream_Harrell_C', bridge['best'])

    def test_the_shared_nll_is_not_the_runner_native_nll(self):
        """The upstream runner scores its NLL from log(1-S); the shared NLL is
        recomputed from the hazard logits, so the two must be told apart."""
        fold0_path, fold0_sha = self.make_fold0_plan()
        fivefold_path, fivefold_sha = self.make_fivefold_plan()
        fold0_runs = self.run_fold0_arms(fold0_path, fold0_sha)
        fivefold_runs = self.run_fivefold_arms(fivefold_path, fivefold_sha)
        result = merge.merge(fold0_path, fold0_runs, fivefold_path, fivefold_runs,
                             self.reference, self.root/'merged.json',
                             required_epochs=TEST_EPOCHS)
        upstream = [row for row in result['rows'] if row['group'] == 'upstream']
        self.assertTrue(upstream)
        for row in upstream:
            self.assertIsNotNone(row['NLL'])
            self.assertIn('runner_reported_NLL', row)
        self.assertIn('runner-native NLL', result['scoring_note'])

    def test_a_runner_metric_mismatch_refuses_the_merge(self):
        fold0_path, fold0_sha = self.make_fold0_plan()
        fivefold_path, fivefold_sha = self.make_fivefold_plan()
        fold0_runs = self.run_fold0_arms(fold0_path, fold0_sha)
        fivefold_runs = self.run_fivefold_arms(fivefold_path, fivefold_sha)
        # Corrupt one arm's saved prediction so the shared scoring disagrees
        # with the metric the runner stored.
        target = fivefold_runs[0]/'val_predictions_best.csv'
        rows = target.read_text(encoding='utf-8').splitlines()
        header = rows[0].split(',')
        index = header.index('survival_bin_0')
        fields = rows[1].split(',')
        fields[index] = '0.5'
        rows[1] = ','.join(fields)
        target.write_text('\n'.join(rows)+'\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'shared scoring disagrees'):
            merge.merge(fold0_path, fold0_runs, fivefold_path, fivefold_runs,
                        self.reference, self.root/'merged_bad.json',
                        required_epochs=TEST_EPOCHS)

    def test_smoke_runs_are_refused_by_the_merge(self):
        fold0_path, fold0_sha = self.make_fold0_plan()
        fivefold_path, fivefold_sha = self.make_fivefold_plan()
        fold0_runs = self.run_fold0_arms(fold0_path, fold0_sha)
        fivefold_runs = self.run_fivefold_arms(fivefold_path, fivefold_sha)
        report_path = fivefold_runs[0]/'report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report['smoke_only'] = True
        report_path.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'smoke'):
            merge.merge(fold0_path, fold0_runs, fivefold_path, fivefold_runs,
                        self.reference, self.root/'merged_smoke.json',
                        required_epochs=TEST_EPOCHS)

    def test_the_recovered_logits_reproduce_the_survival_matrix(self):
        survival = discrete_survival(torch.randn(5, 4)).numpy()
        logits = merge.logits_from_survival(survival)
        recovered = discrete_survival(logits).numpy()
        np.testing.assert_allclose(recovered, survival, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
