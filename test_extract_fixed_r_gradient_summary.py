"""Read-only re-report of an existing fixed-R gradient audit JSON.

Uses a synthetic audit payload only.  No checkpoint, manifest, patient file or
model is loaded, and no gradient is ever computed here.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_fixed_r_gradient_summary as extract


def space(cosine=0.0, ratio=0.02, survival_norm=0.5, auxiliary_norm=0.01,
          status='defined', direction='aligned', outside=0.003):
    dot = None if cosine is None else cosine*auxiliary_norm*survival_norm
    return {'status': status, 'direction': direction, 'cosine': cosine,
            'survival_norm': survival_norm,
            'raw_auxiliary_norm_in_survival_space': auxiliary_norm,
            'weighted_auxiliary_norm_in_survival_space': auxiliary_norm,
            'weighted_norm_ratio_to_survival': ratio,
            'raw_dot': dot,
            'weighted_dot': dot,
            'outside_survival_graph_raw_auxiliary_norm': outside,
            'survival_connected_tensors': 3, 'both_connected_tensors': 3}


def payload():
    modules = {'patch_encoder': space(0.012, 0.215),
               'matching_rna': space(-0.157, 0.795),
               'survival_head': space(status='no_auxiliary_dependency_in_survival_space',
                                      direction=None, cosine=None)}
    records = []
    for index in range(3):
        records.append({
            'batch_index': index, 'repeat': 11, 'patient_ids': [f'p{index}_{j}' for j in range(20)],
            'audit': {
                'terms': {'global_common': {'status': 'computed', 'weight': 1e-3,
                                            'survival_graph_space': space(0.0019, 0.0207),
                                            'modules': modules}},
                'sum_weighted_auxiliary': {
                    'survival_graph_space': space(-0.026, 0.0262, direction='conflicting'),
                    'modules': modules, 'norm_of_sum_over_sum_of_norms': 0.5},
                'total_descent_direction': space(0.9998, 1.0003),
                'gradient_reconstruction': {'absolute_l2_error': 4.46e-8,
                                            'relative_l2_error': 6.9e-8, 'direct_total_norm': 0.5},
            }})
    return {'protocol': 'fixed_r_train_gradients_v1', 'training_performed': False, 'optimizer_steps': 0,
            'validation_loaded': False, 'test_loaded': False,
            'summary': {'global_common': {}, 'sum_weighted_auxiliary': {}},
            'records': records}


class ExtractFixedRGradientSummaryTest(unittest.TestCase):
    def test_reports_aggregate_and_descent_separately(self):
        result = extract.extract(payload())
        aggregate = result['auxiliary_sum_versus_survival_gradient']
        descent = result['survival_gradient_versus_total_gradient']
        self.assertAlmostEqual(aggregate['cosine']['median'], -0.026)
        self.assertEqual(aggregate['direction_counts'], {'conflicting': 3})
        self.assertAlmostEqual(descent['cosine']['median'], 0.9998)
        self.assertAlmostEqual(descent['weighted_norm_ratio_to_survival']['median'], 1.0003)

    def test_absolute_and_relative_module_numbers_are_both_kept(self):
        result = extract.extract(payload())
        modules = result['auxiliary_sum_versus_survival_gradient']['modules']
        self.assertAlmostEqual(modules['matching_rna']['weighted_norm_ratio_to_survival']['median'], 0.795)
        self.assertAlmostEqual(modules['matching_rna']['survival_norm']['median'], 0.5)
        self.assertAlmostEqual(modules['matching_rna']['weighted_auxiliary_norm_in_survival_space']['median'], 0.01)
        self.assertEqual(modules['survival_head']['cosine']['count'], 0)

    def test_distinct_batch_count_is_reported(self):
        result = extract.extract(payload())
        self.assertEqual(result['records'], 3)
        self.assertEqual(result['batches'], 3)
        self.assertEqual(result['patients_total'], 60)

    def test_cli_reads_existing_file_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'audit.json'
            source.write_text(json.dumps(payload()), encoding='utf-8')
            out = Path(folder)/'summary.json'
            result = subprocess.run([sys.executable, 'extract_fixed_r_gradient_summary.py',
                                     '--audit', str(source), '--out', str(out)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            written = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(written['summary']['records'], 3)
            self.assertFalse(written['training_performed'])
            self.assertFalse(written['validation_loaded'])
            self.assertFalse(written['test_loaded'])
            self.assertEqual(len(written['source_audit_sha256']), 64)
            again = subprocess.run([sys.executable, 'extract_fixed_r_gradient_summary.py',
                                    '--audit', str(source), '--out', str(out)],
                                   capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)

    def test_nonfinite_value_is_rejected(self):
        broken = payload()
        broken['records'][0]['audit']['terms']['global_common']['survival_graph_space']['cosine'] = float('nan')
        with self.assertRaises(ValueError):
            extract.check_finite(broken, 'audit')

    def test_missing_records_is_rejected(self):
        with self.assertRaises(ValueError):
            extract.extract({'summary': {}})


if __name__ == '__main__':
    unittest.main()
