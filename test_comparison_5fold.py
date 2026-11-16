"""Synthetic five-fold comparison regression tests.

The tests build a small synthetic cohort (patient bags plus RNA vectors), run
the whole plan/run/summary chain on CPU, and assert the properties that make
the comparison trustworthy:

* the plan is reproducible and refuses any input change;
* every arm trains, scores and reports under the shared protocol;
* the outer test partition is never opened;
* the train-only fitting boundary holds (bins, RNA scaler, prototypes, ridge
  alpha are fitted on training rows only);
* the summary refuses incomplete, duplicated or mismatched folds.
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

import comparison_5fold as runner
from cancer_parity import DISCRETE_NLL_VERSION


class FiveFoldComparisonTest(unittest.TestCase):
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
        generator = torch.Generator().manual_seed(41)
        rows = []
        self.train_ids, self.val_ids = [], []
        for index in range(30):
            # Patient-level offsets give the bags real between-patient signal.
            patches = (torch.randn(4100, 8, generator=generator)
                       + 3*torch.randn(1, 8, generator=generator))
            torch.save(patches, self.root/f'p{index}.pt')
            np.save(self.root/f'g{index}.npy', torch.randn(12, generator=generator).numpy())
            split = 'train' if index < 24 else 'val'
            if split == 'train':
                time = float(index+1)
                event = int(index % 3 != 0)
                self.train_ids.append(f'p{index}')
            else:
                time = [3.5, 9.5, 15.5, 21.5, 27.5, 30.5][index-24]
                event = [1, 0, 1, 1, 0, 1][index-24]
                self.val_ids.append(f'p{index}')
            rows.append([f'p{index}', split, f'p{index}.pt', f'g{index}.npy', time, event])
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
        # Ordered gene names and a GMT whose pathways overlap the RNA schema.
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

    def make_plan(self):
        plan = runner.make_plan(0, self.manifest, self.reference, self.genes,
                                self.gmt, self.deref_signatures, device='cpu')
        path = self.root/'plan.json'
        runner.write_new(path, plan)
        return path, runner.digest(path)

    # ------------------------------------------------------------------ plan
    def test_plan_is_reproducible_and_records_the_fit_boundary(self):
        path, sha = self.make_plan()
        plan = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(plan['fold'], 0)
        self.assertEqual(plan['inputs']['patients'], {'train': 24, 'val': 6})
        self.assertEqual(plan['settings']['num_patches'], 4096)
        self.assertEqual(plan['settings']['n_bins'], 4)
        self.assertEqual(plan['settings']['alpha_surv'], .5)
        self.assertIn('never opened', plan['fit_boundary']['outer_test'])
        self.assertEqual(len(plan['upstream_sources']), 4)
        for name, files in plan['upstream_sources'].items():
            self.assertTrue(files, name)
        self.assertEqual(plan['pathway_mapping']['pathway_count'], 4)
        self.assertEqual(len(plan['deref_groups']['groups']), 6)
        # Rebuilding the plan must reproduce it byte-for-byte.
        again = runner.make_plan(0, self.manifest, self.reference, self.genes,
                                 self.gmt, self.deref_signatures, device='cpu')
        self.assertEqual(again, plan)

    def test_outer_test_rows_are_refused(self):
        with self.manifest.open('a', encoding='utf-8') as handle:
            handle.write('sealed,test,DO_NOT_OPEN.pt,DO_NOT_OPEN.npy,SEALED,SEALED\n')
        with self.assertRaises(ValueError):
            runner.make_plan(0, self.manifest, self.reference, self.genes,
                             self.gmt, self.deref_signatures, device='cpu')

    def test_bin_edges_come_from_training_rows_only(self):
        cohort = runner.load_fold(self.manifest, self.reference, 0)
        expected = runner.dct_bin_edges(cohort.train_time, cohort.train_event, 4)
        self.assertTrue(np.allclose(cohort.bin_edges, expected, equal_nan=True))
        self.assertEqual(len(cohort.bin_edges), 5)
        for bag in cohort.train_bags + cohort.val_bags:
            bag.close()

    # ------------------------------------------------------- upstream integrity
    def test_upstream_checkouts_are_byte_identical_and_stay_clean(self):
        """The third-party trees must not be edited, and importing them must
        not leave generated caches inside them."""
        import comparison_sources as sources

        # The recorded hashes still match the files on disk.
        hashes = sources.verify_upstream()
        self.assertEqual(set(hashes), set(sources.UPSTREAM))

        # Importing the upstream modules must not write into the checkouts.
        sources.load_pibd_modules()
        sources.load_deref_modules()
        sources.load_dimaf_modules()
        for name, record in sources.UPSTREAM.items():
            root = Path(record['root'])
            caches = [str(p.relative_to(root)) for p in root.rglob('__pycache__')]
            self.assertEqual(caches, [], f'{name} checkout was polluted: {caches}')

    def test_the_upstream_loader_disables_bytecode_writing(self):
        import comparison_sources as sources
        import sys
        original = sys.dont_write_bytecode
        with self.assertRaises((ImportError, FileNotFoundError)):
            sources._load_module_from_path(
                'does_not_exist_module', Path(self.root/'missing.py'), [])
        self.assertEqual(sys.dont_write_bytecode, original)

    # ------------------------------------------------------------------- run
    def test_every_arm_runs_and_scores_on_cpu(self):
        path, sha = self.make_plan()
        for model_id in runner.MODELS:
            out = self.root/model_id
            with redirect_stdout(io.StringIO()):
                result = runner.run_model(path, sha, model_id, out, device='cpu',
                                          smoke=True, epochs=1)
            self.assertEqual(result['model_id'], model_id)
            self.assertEqual(result['fold'], 0)
            self.assertFalse(result['test_evaluated'])
            self.assertFalse(result['outer_test_used'])
            self.assertIn('c_index', result['best_validation'])
            self.assertTrue((out/'report.json').is_file())
            self.assertTrue((out/'val_predictions_best.csv').is_file())
            self.assertEqual(result['metadata']['official_model_reproduction'], False)

    def test_plan_mismatch_refuses_to_train(self):
        path, sha = self.make_plan()
        np.save(self.root/'g0.npy', np.zeros(12, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'no longer matches'):
            runner.run_model(path, sha, 'wsi_only', self.root/'bad', device='cpu')
        self.assertFalse((self.root/'bad').exists())

    def test_rna_ridge_selects_alpha_on_training_rows(self):
        path, sha = self.make_plan()
        with redirect_stdout(io.StringIO()):
            result = runner.run_model(path, sha, 'rna_ridge', self.root/'ridge',
                                      device='cpu', smoke=True)
        metadata = result['metadata']
        self.assertEqual(metadata['selected_alpha'], result['history'][0]['selected_alpha'])
        self.assertIn('training rows', metadata['selection'])

    # ---------------------------------------------------------- DCT reference
    def test_dct_reference_is_loaded_and_never_rerun(self):
        from comparison_baselines import load_dct_reference
        archive = load_dct_reference()
        self.assertFalse(archive['re_run'])
        self.assertTrue(archive['official_model_reproduction'])
        self.assertEqual([record['fold'] for record in archive['folds']], list(range(5)))
        self.assertTrue(all(record['c_index'] is not None for record in archive['folds']))

    def test_dct_reference_rejects_an_incomplete_archive(self):
        from comparison_baselines import load_dct_reference
        partial = self.root/'partial_archive.json'
        partial.write_text(json.dumps({'folds': [{'fold': 0, 'c_index': .7}]}),
                           encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'must cover folds'):
            load_dct_reference(partial)

    def test_summary_places_the_dct_reference_on_the_same_table(self):
        from comparison_baselines import DCT_REFERENCE_ARCHIVE
        runs = [self._fake_report('wsi_only', fold, .60) for fold in range(5)]
        without = runner.summarize(runs, self.root/'summary_without_dct.json')
        self.assertNotIn('dct_reference', without['models'])
        result = runner.summarize(runs, self.root/'summary_with_dct.json',
                                  dct_reference=str(DCT_REFERENCE_ARCHIVE))
        self.assertIn('dct_reference', result['models'])
        row = [r for r in result['rows']
               if r['model'] == 'dct_reference' and r['metric'] == 'c_index'][0]
        self.assertEqual(row['folds'], [0, 1, 2, 3, 4])
        self.assertIsNotNone(row['mean'])
        self.assertIsNotNone(row['std'])
        self.assertFalse(result['dct_reference']['re_run'])
        self.assertEqual(len(result['dct_reference']['limits']), 5)

    def test_the_dct_archive_declares_its_older_likelihood_version(self):
        """The archive must declare that it used a different objective.

        The archived DCT numbers were produced before the discrete_nll
        cross-patient broadcast was fixed.  The arms compared against them
        now use the patientwise likelihood, so the archive has to say so,
        otherwise a reader could take it for a like-for-like baseline.
        """
        from comparison_baselines import load_dct_reference
        archive = load_dct_reference()
        protocol = archive['protocol']
        self.assertIn('loss_version', protocol)
        self.assertNotEqual(protocol['loss_version'], DISCRETE_NLL_VERSION)
        self.assertTrue(any('broadcast' in limit or 'patientwise' in limit
                            for limit in archive['limits']))

    # ---------------------------------------------------------------- summary
    def _fake_report(self, model, fold, c_index):
        metric = {'c_index': c_index, 'c_index_ipcw': .55, 'validation_nll': .9,
                  'brier': [.1, .2], 'ibs': .15, 'iauc': .8,
                  'metric_times': [8., 15.], 'errors': []}
        directory = self.root/f'{model}_f{fold}'
        directory.mkdir(exist_ok=True)
        report = {'model_id': model, 'fold': fold, 'plan_sha256': 'same',
                  'smoke_only': False, 'test_evaluated': False, 'epochs_completed': 30,
                  'best_epoch': 20, 'last_epoch': 30, 'best_validation': metric,
                  'last_validation': metric, 'history': []}
        (directory/'report.json').write_text(json.dumps(report), encoding='utf-8')
        return directory

    def test_summary_reports_mean_and_std_over_five_folds(self):
        runs = []
        for fold in range(5):
            runs.append(self._fake_report('wsi_only', fold, .60 + .01*fold))
        result = runner.summarize(runs, self.root/'summary.json')
        row = [r for r in result['rows'] if r['metric'] == 'c_index'][0]
        self.assertEqual(row['folds'], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(row['mean'], .62, places=6)
        self.assertAlmostEqual(row['std'], float(np.std([.60, .61, .62, .63, .64])), places=6)
        self.assertEqual(len(row['per_fold']), 5)

    def test_summary_refuses_incomplete_and_duplicate_folds(self):
        runs = [self._fake_report('wsi_only', fold, .6) for fold in range(4)]
        with self.assertRaisesRegex(ValueError, 'does not cover folds'):
            runner.summarize(runs, self.root/'partial.json')
        runs = [self._fake_report('wsi_only', fold, .6) for fold in range(5)]
        runs.append(self._fake_report('wsi_only', 2, .6))
        with self.assertRaisesRegex(ValueError, 'Duplicate fold'):
            runner.summarize(runs, self.root/'dup.json')

    def test_summary_refuses_smoke_runs(self):
        runs = []
        for fold in range(5):
            directory = self._fake_report('wsi_only', fold, .6)
            report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
            report['smoke_only'] = True
            (directory/'report.json').write_text(json.dumps(report), encoding='utf-8')
            runs.append(directory)
        with self.assertRaisesRegex(ValueError, 'smoke'):
            runner.summarize(runs, self.root/'smoke.json')


if __name__ == '__main__':
    unittest.main()
