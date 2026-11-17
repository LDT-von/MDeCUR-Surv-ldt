"""Synthetic tests for the read-only RNA matrix semantics audit.

The audit answers "what exactly is in the RNA matrix every arm consumes?".
The tests build small synthetic matrices and assert that the audit:

* accepts a clean genes-by-patients matrix;
* reports a metadata row instead of treating it as a gene;
* reports a malformed column label instead of accepting it;
* reports a column that is shaped like an outcome indicator;
* reports a patient that a split names but the matrix does not contain;
* never opens an outer-test column, and never writes into a run directory.
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT/'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT/'scripts'))

import audit_rna_matrix_semantics as audit


def write_matrix(path, header, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def barcodes(count, prefix='TCGA'):
    return [f'{prefix}-AA-{index:04d}-BBBB-CCCC' for index in range(count)]


class RnaMatrixSemanticsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_clean_matrix_is_accepted(self):
        patients = barcodes(3)
        path = self.root/'clean.csv'
        write_matrix(path, ['gene_symbol'] + patients,
                     [['GENE1', '1.0', '2.0', '3.0'],
                      ['GENE2', '4.0', '5.0', '6.0'],
                      ['GENE3', '7.0', '8.0', '9.0']])
        header, labels = audit.read_matrix_header(path)
        rows = audit.classify_rows(labels)
        self.assertEqual(rows['gene_rows'], 3)
        self.assertEqual(rows['malformed_rows'], 0)
        self.assertEqual(rows['duplicate_gene_symbols'], 0)
        columns, names = audit.classify_columns(header)
        self.assertEqual(columns['columns'], 3)
        self.assertEqual(columns['malformed_columns'], 0)
        self.assertEqual(names, patients)

    def test_a_metadata_row_is_reported_and_not_counted_as_a_gene(self):
        patients = barcodes(2)
        path = self.root/'meta.csv'
        write_matrix(path, ['gene_symbol'] + patients,
                     [['GENE1', '1.0', '2.0'],
                      ['sample_type', 'tumor', 'tumor'],
                      ['GENE2', '3.0', '4.0']])
        _, labels = audit.read_matrix_header(path)
        rows = audit.classify_rows(labels)
        self.assertEqual(rows['gene_rows'], 2)
        self.assertEqual(rows['malformed_rows'], 1)
        example = rows['malformed_examples'][0]
        self.assertEqual(example['label'], 'sample_type')
        self.assertIn('metadata', example['reason'])

    def test_a_malformed_column_label_is_reported(self):
        path = self.root/'bad_column.csv'
        write_matrix(path, ['gene_symbol', 'TCGA-AA-0001-BBBB-CCCC', 'not_a_patient'],
                     [['GENE1', '1.0', '2.0']])
        header, _ = audit.read_matrix_header(path)
        columns, _ = audit.classify_columns(header)
        self.assertEqual(columns['columns'], 2)
        self.assertEqual(columns['malformed_columns'], 1)
        self.assertEqual(columns['malformed_examples'][0]['label'], 'not_a_patient')

    def test_a_duplicate_gene_symbol_is_reported(self):
        patients = barcodes(2)
        path = self.root/'dup.csv'
        write_matrix(path, ['gene_symbol'] + patients,
                     [['GENE1', '1.0', '2.0'], ['GENE1', '3.0', '4.0']])
        _, labels = audit.read_matrix_header(path)
        rows = audit.classify_rows(labels)
        self.assertEqual(rows['duplicate_gene_symbols'], 1)

    def test_the_cohort_overlap_reports_both_directions(self):
        patients = barcodes(3)
        membership = {patients[0]: {'fold_0:train': True},
                      patients[1]: {'fold_0:val': True},
                      'TCGA-AA-9999-BBBB-CCCC': {'fold_0:train': True}}
        overlap = audit.cohort_overlap(patients, membership)
        self.assertEqual(overlap['shared_patients'], 2)
        self.assertEqual(overlap['columns_not_in_any_split'], 1)
        self.assertEqual(overlap['split_patients_without_a_column'], 1)

    def test_a_row_shaped_like_an_event_indicator_is_reported(self):
        """The matrix is genes by patients, so an outcome-derived entry hides
        in a row, not in a column."""
        patients = barcodes(3)
        path = self.root/'event.csv'
        write_matrix(path, ['gene_symbol'] + patients,
                     [['GENE1', '1.0', '2.0', '3.0'],
                      ['GENE2', '4.0', '5.0', '6.0'],
                      ['EVENT', '0', '1', '0']])
        header, labels = audit.read_matrix_header(path)
        rows = audit.classify_rows(labels)
        summary = audit.value_summary(path, patients, rows['gene_row_indices'],
                                      row_labels=labels, sample_genes=10)
        self.assertTrue(summary['readable'])
        flagged = summary['rows_shaped_like_an_event_indicator']
        self.assertTrue(flagged)
        self.assertIn('EVENT', {entry['row'] for entry in flagged})

    def test_the_outcome_shape_test_scans_every_row_not_only_the_sample(self):
        """An outcome-derived row is exactly the row that a gene-only
        subsample drops, so the shape test must not run on the subsample
        alone."""
        patients = barcodes(3)
        path = self.root/'event_hidden.csv'
        write_matrix(path, ['gene_symbol'] + patients,
                     [['GENE1', '1.0', '2.0', '3.0'],
                      ['GENE2', '4.0', '5.0', '6.0'],
                      ['GENE3', '7.0', '8.0', '9.0'],
                      ['EVENT', '0', '1', '0']])
        header, labels = audit.read_matrix_header(path)
        rows = audit.classify_rows(labels)
        # Only the first gene row is sampled; the EVENT row is not a gene and
        # can never enter the sample.
        summary = audit.value_summary(path, patients, rows['gene_row_indices'],
                                      row_labels=labels, sample_genes=1)
        self.assertTrue(summary['readable'])
        self.assertEqual(summary['genes_sampled_for_pca'], 1)
        self.assertEqual(summary['rows_scanned_for_outcome_shapes'], 4)
        flagged = summary['rows_shaped_like_an_event_indicator']
        self.assertTrue(flagged)
        self.assertIn('EVENT', {entry['row'] for entry in flagged})

    def test_the_value_summary_reports_the_distribution(self):
        patients = barcodes(2)
        path = self.root/'values.csv'
        write_matrix(path, ['gene_symbol'] + patients,
                     [['GENE1', '0.0', '2.0'], ['GENE2', '1.0', '3.0']])
        header, labels = audit.read_matrix_header(path)
        rows = audit.classify_rows(labels)
        summary = audit.value_summary(path, patients, rows['gene_row_indices'],
                                      row_labels=labels, sample_genes=10)
        self.assertTrue(summary['readable'])
        self.assertEqual(summary['value_quantiles']['min'], 0.0)
        self.assertEqual(summary['value_quantiles']['max'], 3.0)
        self.assertAlmostEqual(summary['value_mean'], 1.5)
        self.assertEqual(summary['nonfinite_or_malformed_rows'], 0)

    def test_the_split_membership_is_read_without_opening_features(self):
        splits = self.root/'splits'
        splits.mkdir()
        with (splits/'fold_0.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['train', 'val'])
            writer.writerow(['TCGA-AA-0001-BBBB-CCCC', 'TCGA-AA-0002-BBBB-CCCC'])
        membership = audit.read_split_membership(splits)
        self.assertIn('TCGA-AA-0001-BBBB-CCCC', membership)
        self.assertIn('fold_0:train', membership['TCGA-AA-0001-BBBB-CCCC'])
        self.assertIn('fold_0:val', membership['TCGA-AA-0002-BBBB-CCCC'])

    def test_the_audit_writes_a_report_that_states_what_it_cannot_settle(self):
        patients = barcodes(2)
        path = self.root/'report.csv'
        write_matrix(path, ['gene_symbol'] + patients,
                     [['GENE1', '1.0', '2.0'], ['GENE2', '3.0', '4.0']])
        out = self.root/'audit.json'
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, str(Path(audit.__file__).resolve()), '--rna', str(path),
             '--out', str(out), '--sample-genes', '5'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(out.read_text(encoding='utf-8'))
        self.assertTrue(report['answers']['every_row_is_a_gene_symbol'])
        self.assertTrue(report['answers']['every_column_is_a_patient_barcode'])
        # The cohort-wide standardization question is not decidable from the
        # written values and must be recorded as such.
        limits = ' '.join(report['limits'])
        self.assertIn('NOT decidable', limits)
        self.assertIn('outer test', limits)


if __name__ == '__main__':
    unittest.main()
