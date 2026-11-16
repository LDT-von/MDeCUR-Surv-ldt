"""Contracts for the MorphID fold-0 comparison adapter.

These are generated-data checks.  They verify the properties that make the
arm a fair participant in the locked one-fold comparison: the frozen fit sees
the training fold only, the head is the only trainable object, the batch
composition and optimizer step count match the other arms, and the summary
guards still reject mismatched runs.
"""
import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from cancer_parity import _json_float, dct_bin_edges
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
import comparison_fold0 as compare
from comparison_morphid import MorphIDArm, MORPHID_ADAPTER_VERSION


class MorphIDAdapterTest(unittest.TestCase):
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
        generator = torch.Generator().manual_seed(31)
        rows = []
        for i in range(24):
            # Between-patient structure so the generated bags carry a real
            # patient signal rather than a common Gaussian bag mean.
            patches = torch.randn(4096+i % 3, 6, generator=generator) \
                + 3*torch.randn(1, 6, generator=generator)
            torch.save(patches, self.root/f'p{i}.pt')
            np.save(self.root/f'g{i}.npy', torch.randn(8, generator=generator).numpy())
            split = 'train' if i < 20 else 'val'
            time = i+1 if i < 20 else [2.5, 7.5, 12.5, 18.5][i-20]
            event = int(i % 3 != 0) if split == 'train' else [1, 0, 1, 1][i-20]
            rows.append([f'p{i}', split, f'p{i}.pt', f'g{i}.npy', time, event])
        self.manifest = self.root/'manifest.csv'
        with self.manifest.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['patient_id', 'split', 'wsi_pt', 'rna_npy', 'time', 'event'])
            writer.writerows(rows)
        self.reference = self.root/'survival_reference.csv'
        with self.reference.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['patient_id', 'split', 'time', 'event'])
            writer.writerows([[r[0], r[1], r[4], r[5]] for r in rows])
        cohort = load_stream_manifest(self.manifest, 'single_view', include_test=False)
        fingerprint = stream_cohort_fingerprint(cohort, 'single_view')
        compare.close(cohort)
        self.ref_run = self.root/'old_reference'
        self.ref_run.mkdir()
        (self.ref_run/'checkpoint.pt').write_bytes(b'synthetic-reference-file')
        self.original = {'endpoint': 'DSS', 'time_unit': 'months', 'n_bins': 4, 'alpha_surv': .5,
                         'num_patches': 4096, 'config': {'seed': 3, 'epochs': 30, 'rna_scale_policy': 'none'},
                         'history': [{}]*30, 'effective_loss_weights': {'any': 0.},
                         'manifest_sha256': compare.digest(self.manifest), 'cohort_input_sha256': fingerprint,
                         'patients': {'train': 20, 'val': 4}, 'events': {'train': 13, 'val': 3},
                         'bin_edges': [_json_float(x) for x in dct_bin_edges(
                             np.arange(1, 21), np.array([int(i % 3 != 0) for i in range(20)]))],
                         'model_kwargs': dict(path_dim=6, rna_dim=8, pathway_indices=[
                             [j, (j+1) % 8, (j+3) % 8] for j in range(8)]+[[0, 2, 4], [1, 3, 5]],
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

    def test_morphid_arms_are_registered_and_the_plan_covers_their_sources(self):
        self.assertIn('morphid', compare.MODELS)
        self.assertIn('morphid_no_interaction', compare.MODELS)
        self.assertEqual(set(compare.MORPHID_MODELS),
                         {'morphid', 'morphid_no_interaction'})
        path, _ = self.plan()
        plan = json.loads(path.read_text(encoding='utf-8'))
        self.assertIn('comparison_morphid', plan['source_sha256'])
        self.assertIn('cancer_morphid_model', plan['source_sha256'])
        self.assertIn('morphid_probe', plan['source_sha256'])
        self.assertIn('morphid', plan['available_models'])
        self.assertIn('training fold only', plan['settings']['morphid_fit_scope'])

    def test_both_morphid_arms_train_and_produce_reports_and_predictions(self):
        path, sha = self.plan()
        reports = []
        for model_id in compare.MORPHID_MODELS:
            out = self.root/model_id
            with redirect_stdout(io.StringIO()):
                result = compare.run_model(path, sha, model_id, out, smoke=True)
            reports.append(result)
            self.assertEqual(result['model_id'], model_id)
            self.assertEqual(result['epochs_completed'], 1)
            self.assertTrue((out/'checkpoint_best.pt').is_file())
            self.assertTrue((out/'checkpoint_last.pt').is_file())
            self.assertTrue((out/'val_predictions_best.csv').is_file())
            self.assertTrue((out/'val_predictions_last.csv').is_file())
            self.assertIsNone(result['native_model_kwargs'])
            self.assertFalse(result['metadata']['r2_used_as_gate'])
            self.assertFalse(result['metadata']['independence_established'])
            self.assertTrue(result['metadata']['frozen_state_unchanged_after_training'])
            self.assertEqual(result['metadata']['trainable_scope'], 'discrete survival head only')
        self.assertEqual(reports[0]['metadata']['interaction_enabled'], True)
        self.assertEqual(reports[1]['metadata']['interaction_enabled'], False)
        # Every arm must share the schedule and the optimizer step count.
        self.assertEqual(len({r['training_sampling_sha256'] for r in reports}), 1)
        self.assertEqual(len({r['training_schedule_sha256'] for r in reports}), 1)
        self.assertEqual(len({r['optimizer_steps'] for r in reports}), 1)

    def test_morphid_schedule_matches_the_other_arms(self):
        path, sha = self.plan()
        reports = []
        for model_id in ('mdecur_dense_aux_off', 'morphid'):
            with redirect_stdout(io.StringIO()):
                reports.append(compare.run_model(path, sha, model_id, self.root/model_id, smoke=True))
        self.assertEqual(reports[0]['training_sampling_sha256'], reports[1]['training_sampling_sha256'])
        self.assertEqual(reports[0]['training_schedule_sha256'], reports[1]['training_schedule_sha256'])
        self.assertEqual(reports[0]['optimizer_steps'], reports[1]['optimizer_steps'])
        self.assertEqual(reports[0]['history'][0]['patient_batches'],
                         reports[1]['history'][0]['patient_batches'])

    def test_morphid_probe_diagnostics_report_cluster_evidence(self):
        path, sha = self.plan()
        with redirect_stdout(io.StringIO()):
            result = compare.run_model(path, sha, 'morphid', self.root/'morphid', smoke=True)
        diagnostics = result['metadata']['probe_diagnostics']
        self.assertIn('probe_clusters', diagnostics)
        self.assertTrue(diagnostics['frozen_state_unchanged'])
        self.assertTrue(diagnostics['fit_patient_ids'])
        # Every fitting patient must be a training patient, never a validation one.
        validation_ids = {f'p{i}' for i in range(20, 24)}
        self.assertFalse(set(diagnostics['fit_patient_ids']) & validation_ids)
        for entry in diagnostics['probe_clusters']:
            self.assertIn('cluster', entry)
            self.assertIn('fitted', entry)

    def test_frozen_fit_refuses_validation_patients_and_reuse(self):
        arm = MorphIDArm(6, 8, bins=4, clusters=2, latent_dim=4)
        bags = [torch.randn(64, 6, dtype=torch.float64) for _ in range(12)]
        rna = torch.randn(12, 8, dtype=torch.float64)
        ids = [f'p{i}' for i in range(12)]
        arm.fit_frozen(bags, rna, ids)
        self.assertTrue(arm.fitted)
        # Re-fitting the same object is refused rather than silently repeated.
        with self.assertRaises(RuntimeError):
            arm.fit_frozen(bags, rna, ids)
        # A fitting patient cannot be used for inference.
        with self.assertRaises(ValueError):
            arm.predict(bags[:2], rna[:2], ids[:2])

    def test_summary_accepts_a_morphid_arm_and_ranks_it(self):
        path, sha = self.plan()
        runs = []
        for model_id in ('mdecur_dense_aux_off', 'morphid'):
            out = self.root/f'run_{model_id}'
            with redirect_stdout(io.StringIO()):
                compare.run_model(path, sha, model_id, out, smoke=True)
            runs.append(out)
        # The smoke runs are refused by the summary; that guard must still hold
        # once a MorphID arm is present.
        with self.assertRaisesRegex(ValueError, 'incomplete/smoke'):
            compare.summarize(runs, self.root/'smoke_summary.json')
        # Build a complete pair by rewriting only the guard fields.
        for model_id, best_c, last_c in (('mdecur_dense_aux_off', .73, .70), ('morphid', .72, .71)):
            out = self.root/f'full_{model_id}'
            out.mkdir()
            metric = lambda c: {'c_index': c, 'c_index_ipcw': .51, 'validation_nll': .9,
                                 'brier': [.1, .2], 'ibs': .15, 'iauc': .8,
                                 'metric_times': [8., 15.], 'errors': []}
            compare.write_new(out/'report.json', {
                'model_id': model_id, 'plan_sha256': 'same', 'smoke_only': False,
                'protocol': compare.PROTOCOL, 'loss_version': compare.SETTINGS['loss_version'],
                'settings': compare.SETTINGS, 'epochs_completed': 30, 'test_evaluated': False,
                'training_sampling_sha256': 'equal', 'training_schedule_sha256': 'same-order',
                'optimizer_steps': 450, 'best_epoch': 20, 'last_epoch': 30,
                'best_validation': metric(best_c), 'last_validation': metric(last_c)})
        result = compare.summarize([self.root/'full_mdecur_dense_aux_off',
                                     self.root/'full_morphid'], self.root/'summary.json')
        self.assertEqual(result['Harrell_C_rankings']['best'],
                         ['mdecur_dense_aux_off', 'morphid'])
        self.assertEqual(result['Harrell_C_rankings']['last'], ['morphid', 'mdecur_dense_aux_off'])

    def test_metadata_states_the_frozen_design_limits(self):
        arm = MorphIDArm(6, 8, bins=4, clusters=2, latent_dim=4)
        bags = [torch.randn(64, 6, dtype=torch.float64) for _ in range(12)]
        rna = torch.randn(12, 8, dtype=torch.float64)
        arm.fit_frozen(bags, rna, [f'p{i}' for i in range(12)])
        metadata = arm.adaptation_metadata()
        self.assertEqual(metadata['adapter'], MORPHID_ADAPTER_VERSION)
        self.assertFalse(metadata['cluster_count_selected_with_outcomes'])
        self.assertTrue(metadata['composition_is_closed_with_k_minus_1_freedom'])
        self.assertFalse(metadata['per_epoch_patch_resampling_affects_this_arm'])
        joined = ' '.join(metadata['limits'])
        self.assertIn('not a fully nested cross-validation', joined)
        self.assertIn('not evidence of independence', joined)
        self.assertIn('not a PID synergy estimate', joined)

    def test_head_parameters_exclude_the_frozen_components(self):
        arm = MorphIDArm(6, 8, bins=4, clusters=2, latent_dim=4)
        bags = [torch.randn(64, 6, dtype=torch.float64) for _ in range(12)]
        rna = torch.randn(12, 8, dtype=torch.float64)
        arm.fit_frozen(bags, rna, [f'p{i}' for i in range(12)])
        names = arm.named_head_parameters()
        self.assertTrue(names)
        for name in names:
            self.assertTrue(name.startswith('head.'), name)
        # The frozen components must not be trainable through this list.
        trainable = {id(parameter) for parameter in arm.head_parameters()}
        frozen_ids = {id(parameter) for parameter in arm.model.morphology.parameters()}
        self.assertFalse(trainable & frozen_ids)


if __name__ == '__main__':
    unittest.main()
