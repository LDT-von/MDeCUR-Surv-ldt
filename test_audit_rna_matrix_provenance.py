"""Contract tests for the read-only RNA matrix provenance audit.

The audit itself is a data-level probe, so its tests are about *behaviour under
adversarial inputs* rather than about a fixed number: the point is that the
audit cannot silently mislabel a matrix as clean, and that it never writes
into the data tree.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import audit_rna_matrix_provenance as audit  # noqa: E402


def _clean_matrix(n_genes=40, n_samples=30, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_genes, n_samples))
    frame = pd.DataFrame(values, index=[f'G{i}' for i in range(n_genes)],
                         columns=[f'P{i}' for i in range(n_samples)])
    return frame


class NumericSemanticsTest(unittest.TestCase):
    def test_population_zscore_is_recognised(self):
        frame = _clean_matrix()
        z = frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1, ddof=0), axis=0)
        report = audit.numeric_semantics(z)
        self.assertEqual(report['genes_exactly_z_scored_population_sd'],
                         frame.shape[0])
        self.assertIn('per-gene z-scoring', report['interpretation'])

    def test_raw_expression_is_not_called_z_scored(self):
        report = audit.numeric_semantics(_clean_matrix())
        self.assertEqual(report['genes_exactly_z_scored_population_sd'], 0)
        self.assertIn('mixed', report['interpretation'])

    def test_unbiased_standard_deviation_is_distinguished(self):
        frame = _clean_matrix()
        z = frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1, ddof=1), axis=0)
        report = audit.numeric_semantics(z)
        self.assertEqual(report['genes_ddof1_std_one_within_5e-6'],
                         frame.shape[0])
        self.assertEqual(report['genes_exactly_z_scored_population_sd'], 0)


class ImputationStructureTest(unittest.TestCase):
    def test_imputed_gene_is_detected(self):
        frame = _clean_matrix(n_genes=10, n_samples=40)
        # Gene 0: observe the first 10 samples as an exact z-score and fill
        # the remaining 30 with a single constant.
        row = frame.iloc[0].to_numpy()
        observed = row[:10]
        z = (observed - observed.mean()) / observed.std(ddof=0)
        frame.iloc[0, :] = np.concatenate([z, np.full(30, -1.5)])
        report = audit.imputation_structure(frame)
        self.assertIn('G0', report['imputed_genes'])
        self.assertEqual(report['n_imputed_genes_whose_observed_part_is_exactly_z_scored'], 1)
        self.assertEqual(report['n_genes_fully_observed'], 9)
    def test_fully_observed_matrix_reports_no_imputation(self):
        report = audit.imputation_structure(_clean_matrix())
        self.assertEqual(report['n_genes_imputed'], 0)
        self.assertEqual(report['fraction_genes_imputed'], 0.0)

    def test_imputation_is_not_confused_with_a_constant_gene(self):
        frame = _clean_matrix(n_genes=4, n_samples=20)
        frame.iloc[0, :] = 3.0
        report = audit.imputation_structure(frame)
        # A fully constant gene has no observed part at all, so it is not an
        # imputed gene; it is reported as constant in part 1 instead.
        self.assertNotIn('G0', report['imputed_genes'])


class CenteringPopulationTest(unittest.TestCase):
    def test_extra_samples_are_reported(self):
        frame = _clean_matrix(n_genes=5, n_samples=10)
        # Only P0..P2 have a clinical row; the other seven RNA samples are
        # still part of the centering population.
        report = audit.centering_population(frame, ['P0', 'P1'], ['P2'],
                                            ['P0', 'P1', 'P2'])
        self.assertEqual(report['n_extra_samples_without_clinical_row'], 7)
        self.assertEqual(report['n_clinical_rows_found_in_matrix'], 3)

    def test_train_only_difference_is_measured(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=(6, 20))
        frame = pd.DataFrame(values, index=[f'G{i}' for i in range(6)],
                             columns=[f'P{i}' for i in range(20)])
        report = audit.centering_population(frame, [f'P{i}' for i in range(10)],
                                            [f'P{i}' for i in range(10, 15)],
                                            [f'P{i}' for i in range(15)])
        self.assertTrue(report['transductive_input_leak_detected'])

    def test_leak_flag_is_false_when_train_covers_everything(self):
        frame = _clean_matrix(n_genes=6, n_samples=12)
        report = audit.centering_population(frame, list(frame.columns), [], list(frame.columns))
        self.assertFalse(report['transductive_input_leak_detected'])


class OutcomeProbeTest(unittest.TestCase):
    def test_a_deterministic_outcome_column_is_flagged(self):
        frame = _clean_matrix(n_genes=20, n_samples=40, seed=5)
        times = np.linspace(1, 40, 40)
        events = (np.arange(40) % 3 == 0).astype(float)
        # Plant an exact function of the outcome.
        frame.iloc[0, :] = events * 7.0 + times * 0.0
        ids = [f'P{i}' for i in range(40)]
        time_by_id = {p: float(times[i]) for i, p in enumerate(ids)}
        event_by_id = {p: float(events[i]) for i, p in enumerate(ids)}
        report = audit.outcome_derived_probe(frame, ids, time_by_id, event_by_id)
        self.assertTrue(report['deterministic_outcome_column_detected'])

    def test_independent_matrix_is_not_flagged(self):
        frame = _clean_matrix(n_genes=20, n_samples=40, seed=6)
        ids = [f'P{i}' for i in range(40)]
        time_by_id = {p: float(i + 1) for i, p in enumerate(ids)}
        event_by_id = {p: float(i % 2) for i, p in enumerate(ids)}
        report = audit.outcome_derived_probe(frame, ids, time_by_id, event_by_id)
        self.assertFalse(report['deterministic_outcome_column_detected'])


class SignalSurvivalTest(unittest.TestCase):
    def test_label_permutation_null_is_near_random_and_real_signal_is_above_it(self):
        rng = np.random.default_rng(11)
        n_genes, n_train, n_val = 60, 80, 40
        n_total = n_train + n_val
        latent = rng.normal(size=n_total)
        values = rng.normal(size=(n_genes, n_total))
        # Gene 0 carries the signal in every patient, train and validation.
        values[0, :] += 3.0 * latent
        frame = pd.DataFrame(values, index=[f'G{i}' for i in range(n_genes)],
                             columns=[f'T{i}' for i in range(n_train)]
                             + [f'V{i}' for i in range(n_val)])
        train_ids = [f'T{i}' for i in range(n_train)]
        val_ids = [f'V{i}' for i in range(n_val)]
        # The outcome depends on the planted signal for every patient.
        time_by_id, event_by_id = {}, {}
        for i, p in enumerate(train_ids + val_ids):
            time_by_id[p] = float(np.clip(30.0 + 8.0 * latent[i], 1.0, 59.0))
            event_by_id[p] = 1.0
        report = audit.signal_survival_analysis(
            frame, train_ids, val_ids, time_by_id, event_by_id,
            list(frame.index), [], null_repeats=3)
        self.assertEqual(report['status'], 'ok')
        self.assertTrue(report['signal_above_label_permuted_null'])
        self.assertIn('shipped_matrix', report['variants'])

    def test_a_pure_noise_matrix_does_not_beat_its_own_null(self):
        rng = np.random.default_rng(13)
        n_genes, n_train, n_val = 40, 60, 30
        values = rng.normal(size=(n_genes, n_train + n_val))
        frame = pd.DataFrame(values, index=[f'G{i}' for i in range(n_genes)],
                             columns=[f'T{i}' for i in range(n_train)]
                             + [f'V{i}' for i in range(n_val)])
        train_ids = [f'T{i}' for i in range(n_train)]
        val_ids = [f'V{i}' for i in range(n_val)]
        time_by_id = {p: float(rng.uniform(1, 60)) for p in train_ids + val_ids}
        event_by_id = {p: float(rng.integers(0, 2)) for p in train_ids + val_ids}
        report = audit.signal_survival_analysis(
            frame, train_ids, val_ids, time_by_id, event_by_id,
            list(frame.index), [], null_repeats=3)
        self.assertEqual(report['status'], 'ok')
        # A pure noise matrix must not claim a signal above its own null.
        self.assertFalse(report['signal_above_label_permuted_null'])

    def test_insufficient_rows_are_reported_not_guessed(self):
        frame = _clean_matrix(n_genes=5, n_samples=10)
        report = audit.signal_survival_analysis(
            frame, ['P0'], ['P1'], {'P0': 1.0, 'P1': 2.0},
            {'P0': 1.0, 'P1': 0.0}, list(frame.index), [])
        self.assertEqual(report['status'], 'insufficient_rows')


class EndToEndTest(unittest.TestCase):
    """The CLI must run on a synthetic cohort and must not touch the source."""

    def test_cli_writes_receipt_and_leaves_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(21)
            ids = [f'P{i:03d}' for i in range(30)]
            times = rng.uniform(1, 60, 30)
            events = rng.integers(0, 2, 30)
            matrix = pd.DataFrame(rng.normal(size=(12, 30)),
                                  index=[f'G{i}' for i in range(12)], columns=ids)
            rna_csv = root / 'rna.csv'
            matrix.to_csv(rna_csv)
            clinical = pd.DataFrame({
                'case id': ids, 'patient_id': ids,
                'survival_months_dss': times, 'censorship_dss': 1 - events,
            })
            clinical_csv = root / 'clinical.csv'
            clinical.to_csv(clinical_csv, index=False)
            split_csv = root / 'split.csv'
            split_frame = pd.DataFrame({'train': pd.Series(ids[:20], dtype=object),
                                        'val': pd.Series(ids[20:] + [''] * 10,
                                                         dtype=object)})
            split_frame.to_csv(split_csv, index=False)
            manifest_csv = root / 'manifest.csv'
            with manifest_csv.open('w', encoding='utf-8', newline='') as handle:
                handle.write('patient_id,split,rna_npy,time,event\n')
                for i, p in enumerate(ids):
                    handle.write(f'{p},{"train" if i < 20 else "val"},rna/{p}.npy,'
                                 f'{times[i]:.6f},{events[i]}\n')
            rna_dir = root / 'rna'
            rna_dir.mkdir()
            for p in ids:
                np.save(rna_dir / f'{p}.npy', matrix[p].to_numpy())

            rna_before = rna_csv.read_bytes()
            manifest_before = manifest_csv.read_bytes()
            clinical_before = clinical_csv.read_bytes()

            out = root / 'receipt.json'
            result = subprocess.run(
                [sys.executable, str(ROOT / 'scripts' / 'audit_rna_matrix_provenance.py'),
                 '--rna-csv', str(rna_csv), '--manifest', str(manifest_csv),
                 '--split-csv', str(split_csv), '--clinical-csv', str(clinical_csv),
                 '--out', str(out)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(receipt['audit'], audit.AUDIT_VERSION)
            self.assertFalse(receipt['boundaries']['training_performed'])
            self.assertFalse(receipt['boundaries']['test_partition_read'])
            self.assertFalse(receipt['boundaries']['data_tree_modified'])
            self.assertEqual(receipt['cohort']['manifest_train'], 20)
            self.assertEqual(receipt['cohort']['manifest_val'], 10)

            self.assertEqual(rna_csv.read_bytes(), rna_before)
            self.assertEqual(manifest_csv.read_bytes(), manifest_before)
            self.assertEqual(clinical_csv.read_bytes(), clinical_before)

    def test_receipt_reports_the_source_hashes(self):
        self.assertTrue(hasattr(audit, 'sha256_file'))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'x.bin'
            path.write_bytes(b'abc')
            self.assertEqual(audit.sha256_file(path),
                             'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')


if __name__ == '__main__':
    unittest.main()
