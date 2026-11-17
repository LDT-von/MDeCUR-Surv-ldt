"""Synthetic five-fold comparison regression tests.

The tests build a small synthetic cohort (patient bags plus RNA vectors), run
the whole plan/run/summary chain on CPU, and assert the properties that make
the comparison trustworthy:

* the plan is reproducible and refuses any input change;
* every arm trains, scores and reports under the shared protocol;
* the outer test partition is never opened;
* the train-only fitting boundary holds (bins, RNA scaler, prototypes are fitted on training rows only);
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
    def test_the_prototype_sampling_is_bounded_and_reproducible(self):
        """The prototype sample must be bounded and drawn only from the bags.

        The upstream prototype step samples a fixed number of rows from the
        training slides.  The adaptation keeps that rule but must never
        materialise a whole slide, and the draw must be reproducible from the
        recorded seed.
        """
        from comparison_dimaf import sample_prototype_rows

        class FakeBag:
            """One slide of ``rows`` rows that records every read."""

            feature_dim = 4

            def __init__(self, rows, seed):
                self.n_patches = rows
                self._tensor = torch.randn(rows, 4, generator=torch.Generator().manual_seed(seed))
                self.reads = 0

            def blocks(self, chunk):
                for start in range(0, self.n_patches, chunk):
                    yield ('fake', start, min(start + chunk, self.n_patches))

            def read_rows(self, rows):
                self.reads += 1
                index = torch.as_tensor(list(rows), dtype=torch.long)
                return self._tensor.index_select(0, index)

        bags = [FakeBag(500, 1), FakeBag(300, 2)]
        sample = sample_prototype_rows(bags, n_proto=4, n_patches_per_proto=100,
                                       feature_dim=4, seed=3)
        self.assertEqual(sample.shape, (400, 4))
        self.assertTrue(torch.isfinite(sample).all())
        # Only the requested rows were read, never a whole slide copy.
        for bag in bags:
            self.assertLessEqual(bag.reads, 1)
        # The draw is reproducible from the seed.
        again = sample_prototype_rows(bags, n_proto=4, n_patches_per_proto=100,
                                      feature_dim=4, seed=3)
        self.assertTrue(torch.equal(sample, again))
        # A different seed gives a different draw.
        other = sample_prototype_rows(bags, n_proto=4, n_patches_per_proto=100,
                                      feature_dim=4, seed=5)
        self.assertFalse(torch.equal(sample, other))

    def test_the_prototype_sampling_refuses_an_empty_cohort(self):
        from comparison_dimaf import sample_prototype_rows

        with self.assertRaises(ValueError):
            sample_prototype_rows([], n_proto=4, n_patches_per_proto=10,
                                  feature_dim=4, seed=3)

    def test_the_dimaf_prototypes_do_not_depend_on_the_ambient_numpy_state(self):
        """The upstream KMeans reads NumPy's global state, so the prototypes
        must be pinned to the recorded seed.

        Measured defect: the DIMAF arm was not bit-reproducible across two
        identical-seed runs, while the two upstream arms that do not cluster
        were.  The upstream ``kmeans_clustering`` calls scikit-learn's
        ``KMeans`` without a ``random_state``, so its initialisation is drawn
        from NumPy's global random state, which ``torch.manual_seed`` does not
        control.
        """
        import numpy as np
        from comparison_dimaf import fit_prototypes

        features = torch.randn(400, 8, generator=torch.Generator().manual_seed(0))

        # Perturb the ambient NumPy global state between the two calls.
        np.random.seed(1234)
        np.random.random(17)
        first = fit_prototypes(features, n_proto=4, n_iter=10, feature_dim=8, seed=3)

        np.random.seed(9999)
        np.random.random(23)
        second = fit_prototypes(features, n_proto=4, n_iter=10, feature_dim=8, seed=3)

        self.assertTrue(np.allclose(first, second),
                        'The prototypes changed with the ambient NumPy state')

    def test_the_dimaf_prototype_call_restores_the_ambient_numpy_state(self):
        """The adaptation must not leave the global NumPy state disturbed.

        The upstream clustering consumes the global random state.  The
        adaptation pins it for the call and restores it afterwards, so the
        rest of the run sees the same stream it would have seen anyway.
        """
        import numpy as np
        from comparison_dimaf import fit_prototypes

        features = torch.randn(200, 8, generator=torch.Generator().manual_seed(0))

        np.random.seed(7)
        before = np.random.random(5)
        np.random.seed(7)
        fit_prototypes(features, n_proto=4, n_iter=10, feature_dim=8, seed=3)
        after = np.random.random(5)
        self.assertTrue(np.allclose(before, after),
                        'The ambient NumPy stream was disturbed by the clustering')

    def test_the_deref_upstream_reorganisation_is_pinned_to_the_seed(self):
        """The upstream feature reorganisation draws from Python's global
        ``random`` state, so the adapter must pin it.

        Measured defect: the DeReF arm was not bit-reproducible across two
        identical-seed runs.  The upstream network calls ``random.choice`` on
        the segment length, which ``torch.manual_seed`` does not control.
        """
        import random
        from comparison_deref import DeReF2M

        self.assertEqual(DeReF2M.method_name, 'DeReF-2M-P4096-DNLL')
        source = Path('comparison_deref.py')
        self.assertTrue(source.is_file())
        text = source.read_text(encoding='utf-8')
        # The adapter must pin the global ``random`` state around the upstream
        # call and restore it afterwards.
        self.assertIn('random.getstate()', text)
        self.assertIn('random.setstate(', text)
        self.assertIn('random.seed(self.seed', text)

    def test_the_deref_forward_is_reproducible_under_a_perturbed_ambient_state(self):
        """The DeReF forward must not depend on the ambient ``random`` stream.

        The upstream network picks the segment length with ``random.choice``.
        Two calls that reach that point with different ambient states used to
        produce different logits.  The adapter pins the state, so the same
        model and the same inputs must give the same logits.

        The two calls deliberately use **different** ambient states.  Resetting
        the ambient state identically before each call would make even the
        unfixed adapter deterministic inside one process and would hide the
        defect.
        """
        import random
        from comparison_deref import DeReF2M

        def forward(ambient):
            random.seed(ambient)
            random.random()
            np.random.seed(ambient)
            np.random.random(11)
            torch.manual_seed(3)
            model = DeReF2M(path_dim=8, rna_dim=24,
                            omic_indices=[[0, 1, 2, 3]]*6, bins=4, seed=3)
            model.eval()
            bags = [torch.randn(20, 8) for _ in range(2)]
            rna = torch.randn(2, 24)
            with torch.no_grad():
                return model(bags, rna)['logits'].clone()

        first = forward(999)
        second = forward(20261117)
        self.assertTrue(torch.equal(first, second),
                        'The DeReF forward changed with the ambient random state')

    def test_the_dimaf_prototypes_are_reproducible_across_processes(self):
        """The upstream clustering must give the same prototypes in two processes.

        The upstream ``kmeans_clustering`` calls scikit-learn's ``KMeans``
        without a ``random_state``, so its initialisation follows NumPy's
        global state.  The adapter pins that state, so two separate processes
        must produce identical prototypes.
        """
        import subprocess
        import sys
        import tempfile

        from comparison_dimaf import fit_prototypes

        script = (
            'import sys, numpy as np, torch\n'
            'sys.path.insert(0, %r)\n'
            'from comparison_dimaf import fit_prototypes\n'
            'np.random.seed(12345); np.random.random(100)\n'
            'features = torch.randn(400, 16, generator=torch.Generator().manual_seed(0))\n'
            'np.save(sys.argv[1], fit_prototypes(features, n_proto=8, n_iter=20,\n'
            '                                      feature_dim=16, seed=3))\n'
            % str(Path(__file__).resolve().parent)
        )
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for index in range(2):
                target = Path(folder)/f'proto_{index}.npy'
                result = subprocess.run([sys.executable, '-c', script, str(target)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                paths.append(target)
            first = np.load(paths[0])
            second = np.load(paths[1])
        self.assertTrue(np.array_equal(first, second),
                        'The prototypes differ between two identical processes')
        # The in-process call must agree with the cross-process one.
        features = torch.randn(400, 16, generator=torch.Generator().manual_seed(0))
        in_process = fit_prototypes(features, n_proto=8, n_iter=20, feature_dim=16, seed=3)
        self.assertTrue(np.allclose(first, in_process))

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

    def test_the_upstream_verifier_accepts_either_line_ending(self):
        """A fresh clone has LF files; the recorded hashes came from CRLF.

        Both forms represent the same upstream commit, so the verifier must
        accept either one, and it must report the digest of the bytes that
        are actually on disk rather than the recorded one.
        """
        import comparison_sources as sources

        for name, record in sources.UPSTREAM.items():
            root = Path(record['root'])
            for relative, expected in record['files'].items():
                path = root / relative
                if not path.is_file():
                    continue
                variants = sources.line_ending_variants(path)
                self.assertIn(expected, variants.values(),
                              f'{name}/{relative} matches no line-ending form')
                self.assertEqual(variants['raw'], sources.digest(path))
                break

    def test_a_modified_upstream_file_is_still_refused(self):
        """Accepting a second line-ending form must not weaken the check."""
        import comparison_sources as sources

        record = sources.UPSTREAM['CAMR']
        relative = next(iter(record['files']))
        path = Path(record['root'])/relative
        if not path.is_file():
            self.skipTest('CAMR checkout is not available')
        original = path.read_bytes()
        try:
            path.write_bytes(original.replace(b'\r\n', b'\n') + b'\n# tampered\n')
            with self.assertRaisesRegex(ValueError, 'Upstream file modified'):
                sources.verify_upstream('CAMR')
        finally:
            path.write_bytes(original)

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
            runner.run_model(path, sha, 'camr_2m', self.root/'bad', device='cpu')
        self.assertFalse((self.root/'bad').exists())

    def test_retired_single_modality_arms_refused_before_input_loading(self):
        for model_id in ('rna_ridge', 'wsi_only'):
            with self.subTest(model=model_id):
                with self.assertRaisesRegex(ValueError, 'Unknown model'):
                    runner.run_model(self.root/'DO_NOT_OPEN.json', 'unread', model_id,
                                     self.root/model_id, device='cpu')
                self.assertFalse((self.root/model_id).exists())

    # ---------------------------------------------------------- DCT reference
    def test_dct_reference_is_loaded_and_never_rerun(self):
        from comparison_5fold import load_dct_reference
        from comparison_protocol import FOLDS
        archive = load_dct_reference()
        self.assertFalse(archive['re_run'])
        self.assertTrue(archive['official_model_reproduction'])
        # The committed archive still holds the five-fold schedule, but only
        # the protocol folds are read.
        self.assertEqual(archive['archive_fold_count'], 5)
        self.assertEqual([record['fold'] for record in archive['folds']], list(FOLDS))
        self.assertEqual(archive['selected_folds'], list(FOLDS))
        self.assertTrue(all(record['c_index'] is not None for record in archive['folds']))

    def test_dct_reference_rejects_an_incomplete_archive(self):
        from comparison_5fold import load_dct_reference
        from comparison_protocol import FOLDS
        missing = [fold for fold in FOLDS if fold != 0] or [1]
        partial = self.root/'partial_archive.json'
        partial.write_text(json.dumps({'folds': [{'fold': fold, 'c_index': .7}
                                                 for fold in missing]}),
                           encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'must cover folds'):
            load_dct_reference(partial)

    def test_summary_places_the_dct_reference_on_the_same_table(self):
        from comparison_5fold import DCT_REFERENCE_ARCHIVE
        from comparison_protocol import FOLDS
        runs = [self._fake_report('camr_2m', fold, .60) for fold in FOLDS]
        without = runner.summarize(runs, self.root/'summary_without_dct.json')
        self.assertNotIn('dct_reference', without['models'])
        result = runner.summarize(runs, self.root/'summary_with_dct.json',
                                  dct_reference=str(DCT_REFERENCE_ARCHIVE))
        self.assertIn('dct_reference', result['models'])
        row = [r for r in result['rows']
               if r['model'] == 'dct_reference' and r['metric'] == 'c_index'][0]
        self.assertEqual(row['folds'], list(FOLDS))
        self.assertEqual(row['fold_count'], len(FOLDS))
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
        from comparison_5fold import load_dct_reference
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

    def test_summary_reports_mean_and_std_over_the_protocol_folds(self):
        from comparison_protocol import FOLDS
        runs = []
        for fold in FOLDS:
            runs.append(self._fake_report('camr_2m', fold, .60 + .01*fold))
        result = runner.summarize(runs, self.root/'summary.json')
        row = [r for r in result['rows'] if r['metric'] == 'c_index'][0]
        self.assertEqual(row['folds'], list(FOLDS))
        self.assertEqual(row['fold_count'], len(FOLDS))
        self.assertAlmostEqual(row['mean'], .60, places=6)
        values = [.60 + .01*fold for fold in FOLDS]
        self.assertAlmostEqual(row['std'], float(np.std(values)), places=6)
        self.assertEqual(len(row['per_fold']), len(FOLDS))

    def test_summary_refuses_incomplete_and_duplicate_folds(self):
        from comparison_protocol import FOLDS
        # A fold outside the protocol scope is refused, not silently dropped.
        runs = [self._fake_report('camr_2m', fold, .6) for fold in (0, 1)]
        with self.assertRaisesRegex(ValueError, 'does not cover folds'):
            runner.summarize(runs, self.root/'partial.json')
        runs = [self._fake_report('camr_2m', fold, .6) for fold in FOLDS]
        runs.append(self._fake_report('camr_2m', FOLDS[0], .6))
        with self.assertRaisesRegex(ValueError, 'Duplicate fold'):
            runner.summarize(runs, self.root/'dup.json')

    def test_summary_refuses_smoke_runs(self):
        from comparison_protocol import FOLDS
        runs = []
        for fold in FOLDS:
            directory = self._fake_report('camr_2m', fold, .6)
            report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
            report['smoke_only'] = True
            (directory/'report.json').write_text(json.dumps(report), encoding='utf-8')
            runs.append(directory)
        with self.assertRaisesRegex(ValueError, 'smoke'):
            runner.summarize(runs, self.root/'smoke.json')


if __name__ == '__main__':
    unittest.main()
