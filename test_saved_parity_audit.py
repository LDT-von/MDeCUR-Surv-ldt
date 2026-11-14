"""Tiny synthetic archived runs; no patient data, features or models are used."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_parity_saved_predictions import audit_saved_predictions, main


@unittest.skipUnless(importlib.util.find_spec('sksurv'), 'scikit-survival is required')
class SavedParityAuditTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='synthetic_saved_parity_')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.manifest = self.root / 'manifest.csv'
        self.reference = self.root / 'reference.csv'
        self.manifest_rows = [
            {'patient_id': 'SYNTHETIC_T1', 'split': 'train', 'time': .5, 'event': 1},
            {'patient_id': 'SYNTHETIC_T2', 'split': 'train', 'time': 40., 'event': 1},
            {'patient_id': 'SYNTHETIC_V1', 'split': 'val', 'time': 5., 'event': 1},
            {'patient_id': 'SYNTHETIC_V2', 'split': 'val', 'time': 35., 'event': 1},
        ]
        self.reference_rows = [{key: row[key] for key in ('patient_id', 'time', 'event')}
                               for row in self.manifest_rows]
        self.prediction_rows = [
            {'patient_id': 'SYNTHETIC_V1', 'risk': -.18, 'bin_label': 0,
             'survival_bin_0': .1, 'survival_bin_1': .05, 'survival_bin_2': .02, 'survival_bin_3': .01},
            {'patient_id': 'SYNTHETIC_V2', 'risk': -3., 'bin_label': 3,
             'survival_bin_0': .9, 'survival_bin_1': .8, 'survival_bin_2': .7, 'survival_bin_3': .6},
        ]
        self.report = {'bin_edges': ['-Infinity', 10., 20., 30., 'Infinity'], 'n_bins': 4,
                       'best_epoch': 2, 'best_validation_c_index': 1.,
                       'final_validation': {'c_index': 1.},
                       'history': [{'epoch': 1, 'c_index': .5}, {'epoch': 2, 'c_index': 1.},
                                   {'epoch': 3, 'c_index': 1.}],
                       'patients': {'train': 2, 'val': 2}, 'events': {'train': 2, 'val': 2}}
        self.write_csv(self.manifest, self.manifest_rows)
        self.write_csv(self.reference, self.reference_rows)
        self.write_csv(self.run / 'val_predictions.csv', self.prediction_rows)
        self.write_report()
        (self.run / 'checkpoint.pt').write_bytes(b'SYNTHETIC OPAQUE BYTES - NOT A PICKLE')

    @staticmethod
    def write_csv(path, rows):
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def write_report(self):
        (self.run / 'report.json').write_text(json.dumps(self.report), encoding='utf-8')

    def audit(self):
        return audit_saved_predictions(self.manifest, self.reference, self.run)

    def assert_failed(self, result, fragment):
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any(fragment in error for error in result['errors']), result['errors'])

    def test_valid_multislide_archive_reproduces_c_without_loading_checkpoint(self):
        self.write_csv(self.manifest, self.manifest_rows + [dict(self.manifest_rows[0])])
        with patch('torch.load', side_effect=AssertionError('Checkpoint must never be unpickled')):
            result = self.audit()
        self.assertEqual(result['status'], 'passed', result['errors'])
        self.assertEqual(result['counts']['duplicate_manifest_rows'], 1)
        self.assertEqual(result['legacy_metrics']['c_index'], 1.)
        self.assertEqual(result['finite_edge_audit']['metric_times'], [10., 20., 30.])
        self.assertEqual(result['input_sha256'], result['input_sha256_after'])
        self.assertEqual(len(result['input_sha256']['checkpoint_opaque_bytes']), 64)
        serialized = json.dumps(result, allow_nan=False)
        for row in self.manifest_rows:
            self.assertNotIn(row['patient_id'], serialized)

    def test_duplicate_reference_is_rejected(self):
        self.write_csv(self.reference, self.reference_rows + [dict(self.reference_rows[0])])
        self.assert_failed(self.audit(), 'duplicate patient')

    def test_same_patient_cross_split_is_rejected(self):
        self.write_csv(self.manifest, self.manifest_rows + [dict(self.manifest_rows[0], split='val')])
        self.assert_failed(self.audit(), 'both train and validation')

    def test_tcga_aliquot_cross_split_is_rejected(self):
        self.manifest_rows[0]['patient_id'] = 'TCGA-AB-1234-01A'
        self.manifest_rows[2]['patient_id'] = 'TCGA-AB-1234-02B'
        self.write_csv(self.manifest, self.manifest_rows)
        result = self.audit()
        self.assert_failed(result, 'TCGA case overlaps')
        self.assertNotIn('TCGA-AB-1234', json.dumps(result))

    def test_reference_manifest_disagreement_is_rejected(self):
        self.reference_rows[0]['time'] = 15.
        self.write_csv(self.reference, self.reference_rows)
        self.assert_failed(self.audit(), 'labels disagree')

    def test_prediction_membership_and_duplicate_fail(self):
        self.prediction_rows[0]['patient_id'] = 'SYNTHETIC_NOT_IN_VALIDATION'
        self.write_csv(self.run / 'val_predictions.csv', self.prediction_rows)
        self.assert_failed(self.audit(), 'do not exactly equal')
        self.prediction_rows[0] = dict(self.prediction_rows[1])
        self.write_csv(self.run / 'val_predictions.csv', self.prediction_rows)
        self.assert_failed(self.audit(), 'duplicate patient')

    def test_wrong_stored_risk_is_rejected(self):
        self.prediction_rows[0]['risk'] = .18
        self.write_csv(self.run / 'val_predictions.csv', self.prediction_rows)
        self.assert_failed(self.audit(), 'stored risk disagrees')

    def test_report_score_or_selected_epoch_disagreement_is_rejected(self):
        self.report['best_validation_c_index'] = .7361
        self.write_report()
        self.assert_failed(self.audit(), 'does not reproduce')
        self.report['best_validation_c_index'] = 1.
        self.report['best_epoch'] = 3
        self.write_report()
        self.assert_failed(self.audit(), 'earliest history maximum')

    def test_legacy_nan_metric_is_explicit_none_and_json_stays_valid(self):
        legacy = {'c_index': 1., 'iauc': float('nan'), 'errors': []}
        with patch('scripts.audit_parity_saved_predictions.parity_metrics', return_value=legacy):
            result = self.audit()
        self.assertEqual(result['status'], 'passed', result['errors'])
        self.assertIsNone(result['legacy_metrics']['iauc'])
        self.assertTrue(any('nonfinite' in error for error in result['legacy_metrics']['errors']))
        json.dumps(result, allow_nan=False)

    def test_cli_writes_failed_status_nonzero_and_never_overwrites(self):
        self.report['final_validation']['c_index'] = .7361
        self.write_report()
        output = self.root / 'audit.json'
        arguments = ['--manifest', str(self.manifest), '--survival-reference', str(self.reference),
                     '--reference-run', str(self.run), '--out', str(output)]
        self.assertEqual(main(arguments), 1)
        before = output.read_bytes()
        self.assertEqual(json.loads(before)['status'], 'failed')
        self.assertEqual(main(arguments), 2)
        self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
