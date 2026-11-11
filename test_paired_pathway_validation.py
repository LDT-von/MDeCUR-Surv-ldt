"""Small synthetic checks for validation-only paired comparison contracts."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from paired_pathway_validation import compare, digest


class PairedPathwayValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.csv'
        self.train = [('t1', 3, 1), ('t2', 20, 1), ('t3', 35, 1), ('t4', 40, 1)]
        self.val = [('v1', 5, 1), ('v2', 15, 1), ('v3', 25, 0), ('v4', 35, 1)]
        with self.manifest.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['patient_id', 'split', 'time', 'event'])
            for split, rows in [('train', self.train), ('val', self.val)]:
                for identity, t, e in rows:
                    writer.writerow([identity, split, t, e])
        self.reference = self.make_run('reference', 'full', [.70, .80, .85, .90])

    def make_run(self, name, mode, survival):
        directory = self.root / name
        directory.mkdir()
        config = {'matched_common': .0005, 'matched_unique': .0005,
                  'common_mode': 'stability', 'interaction_strength': .2,
                  'seed': 17, 'lr': .003}
        model = {'interaction_strength': .2, 'hidden': 16}
        if mode == 'matched-off':
            config.update(matched_common=0, matched_unique=0)
        elif mode == 'uniform-common':
            config['common_mode'] = 'uniform'
        elif mode == 'interaction-off':
            config['interaction_strength'] = model['interaction_strength'] = 0
        rates = [-math.log(s)/12 for s in survival]
        nll = sum(r*t-e*math.log(r) for r, (_, t, e) in zip(rates, self.val))/len(self.val)
        # Training patients have no censoring; G(t)=1. Only v1 has an event by 12 months.
        brier = (survival[0]**2 + sum((1-s)**2 for s in survival[1:]))/4
        # Each observed event precedes every subject with lower predicted risk.
        report = {'evaluation_split': 'val', 'test_evaluated': False,
                  'manifest_sha256': digest(self.manifest), 'interval_starts': [0],
                  'prediction_horizon': 12, 'architecture': 'PathwayLocalSurvival',
                  'cohort_input_sha256': 'same', 'model_kwargs': model,
                  'time_unit': 'months', 'endpoint': 'OS', 'view_source': 'feature_noise',
                  'minimum_patient_batch': 17, 'stream_config': {'patient_batch_size': 20},
                  'pathway_mapping': {'pathways': ['A']}, 'rna_preprocessing': {'mask': [1]},
                  'config': config, 'evaluation_nll': nll,
                  'evaluation_c_index_risk_at_horizon': {'value': 1.0},
                  'evaluation_ipcw_brier_at_horizon': {'value': brier}}
        (directory/'report.json').write_text(json.dumps(report), encoding='utf-8')
        with (directory/'val_predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['patient_id', 'survival_at_horizon'])
            # Reverse order to test ID alignment rather than row order.
            for i in reversed(range(len(self.val))):
                writer.writerow([self.val[i][0], survival[i]])
        return directory

    def run_comparison(self, candidate, mode):
        return compare(self.manifest, self.reference, candidate, 1234, 100, mode)

    def update_report(self, directory, mutate):
        path = directory/'report.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        mutate(data)
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_three_comparisons_and_paired_bootstrap(self):
        for mode in ('matched-off', 'uniform-common', 'interaction-off'):
            with self.subTest(mode=mode):
                candidate = self.make_run(mode, mode, [.72, .79, .83, .91])
                result = self.run_comparison(candidate, mode)
                self.assertEqual(result['comparison'], mode)
                self.assertAlmostEqual(result['candidate']['ipcw_brier'],
                                       (.72**2 + .21**2 + .17**2 + .09**2)/4)
                self.assertAlmostEqual(result['candidate']['harrell_c'], 1)
                self.assertEqual(result['bootstrap']['intervals']['nll']['valid_replicates'], 100)
                self.assertEqual(set(result['model_kwargs_changes']),
                                 {'interaction_strength'} if mode == 'interaction-off' else set())
        identical = self.make_run('identical_prediction', 'matched-off', [.70, .80, .85, .90])
        result = self.run_comparison(identical, 'matched-off')
        for metric in ('nll', 'harrell_c', 'ipcw_brier'):
            self.assertEqual(result['difference'][metric], 0)
            self.assertEqual(result['bootstrap']['intervals'][metric]['percentile_95'], [0, 0])

    def test_reject_unexpected_configuration_and_model_changes(self):
        for mode in ('matched-off', 'uniform-common', 'interaction-off'):
            with self.subTest(mode=mode):
                candidate = self.make_run('bad_'+mode, mode, [.72, .79, .83, .91])
                self.update_report(candidate, lambda r: r['config'].__setitem__('lr', .1))
                with self.assertRaisesRegex(ValueError, 'Unexpected configuration'):
                    self.run_comparison(candidate, mode)
        candidate = self.make_run('bad_model', 'interaction-off', [.72, .79, .83, .91])
        self.update_report(candidate, lambda r: r['model_kwargs'].__setitem__('hidden', 32))
        with self.assertRaisesRegex(ValueError, 'Unexpected model_kwargs'):
            self.run_comparison(candidate, 'interaction-off')
        candidate = self.make_run('bad_strength', 'interaction-off', [.72, .79, .83, .91])
        self.update_report(candidate, lambda r: r['model_kwargs'].__setitem__('interaction_strength', .1))
        with self.assertRaisesRegex(ValueError, 'strengths must agree'):
            self.run_comparison(candidate, 'interaction-off')

    def test_reject_input_and_metric_inconsistency(self):
        for field in ('pathway_mapping', 'rna_preprocessing', 'stream_config', 'metric_conventions'):
            with self.subTest(field=field):
                candidate = self.make_run('different_'+field, 'matched-off', [.72, .79, .83, .91])
                self.update_report(candidate, lambda r: r.__setitem__(field, {'unexpected': True}))
                with self.assertRaisesRegex(ValueError, field):
                    self.run_comparison(candidate, 'matched-off')
        candidate = self.make_run('bad_metric', 'matched-off', [.72, .79, .83, .91])
        self.update_report(candidate, lambda r: r.__setitem__('evaluation_nll', 100))
        with self.assertRaisesRegex(ValueError, 'nll fails original report check'):
            self.run_comparison(candidate, 'matched-off')
        candidate = self.make_run('duplicate', 'matched-off', [.72, .79, .83, .91])
        with (candidate/'val_predictions.csv').open('a', encoding='utf-8') as f:
            f.write('v1,0.7\n')
        with self.assertRaisesRegex(ValueError, 'unique IDs'):
            self.run_comparison(candidate, 'matched-off')

    def test_reject_wrong_ablation_direction_or_partial_change(self):
        partial = self.make_run('partial', 'matched-off', [.72, .79, .83, .91])
        self.update_report(partial, lambda r: r['config'].__setitem__('matched_unique', .0005))
        with self.assertRaisesRegex(ValueError, 'Unexpected configuration'):
            self.run_comparison(partial, 'matched-off')
        shuffled = self.make_run('shuffled', 'uniform-common', [.72, .79, .83, .91])
        self.update_report(shuffled, lambda r: r['config'].__setitem__('common_mode', 'shuffled'))
        with self.assertRaisesRegex(ValueError, 'stability reference and uniform candidate'):
            self.run_comparison(shuffled, 'uniform-common')
        partial_interaction = self.make_run('partial_interaction', 'interaction-off', [.72, .79, .83, .91])
        def partial_strength(r):
            r['config']['interaction_strength'] = .1
            r['model_kwargs']['interaction_strength'] = .1
        self.update_report(partial_interaction, partial_strength)
        with self.assertRaisesRegex(ValueError, 'positive reference and zero candidate'):
            self.run_comparison(partial_interaction, 'interaction-off')

    def test_reject_nonfinite_report_and_record_metric_dependency(self):
        candidate = self.make_run('nonfinite', 'matched-off', [.72, .79, .83, .91])
        self.update_report(candidate, lambda r: r.__setitem__('evaluation_nll', float('nan')))
        with self.assertRaisesRegex(ValueError, 'nll fails original report check'):
            self.run_comparison(candidate, 'matched-off')
        valid = self.make_run('dependency', 'matched-off', [.72, .79, .83, .91])
        result = self.run_comparison(valid, 'matched-off')
        self.assertEqual(result['inputs']['cancer_metrics_sha256'],
                         digest(Path(__file__).with_name('cancer_metrics.py')))
        self.assertIn('numpy', result['runtime'])

    def test_cli_from_project_root_and_no_overwrite(self):
        candidate = self.make_run('cli', 'uniform-common', [.72, .79, .83, .91])
        output = self.root/'result.json'
        cmd = [sys.executable, str(Path(__file__).with_name('paired_pathway_validation.py')),
               '--manifest', str(self.manifest), '--reference', str(self.reference),
               '--candidate', str(candidate), '--comparison', 'uniform-common',
               '--bootstrap', '10', '--out', str(output)]
        cwd = Path(__file__).resolve().parent
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['comparison'], 'uniform-common')
        repeat = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        self.assertNotEqual(repeat.returncode, 0)
        self.assertIn('FileExistsError', repeat.stderr)


if __name__ == '__main__':
    unittest.main()
