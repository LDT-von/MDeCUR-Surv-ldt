"""Synthetic checks for the validation-only query perturbation diagnostic."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import predict_pathway_model
from cancer_rna import fit_rna_scaler, transform_rna
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
from diagnose_pathway_query_permutation import (fixed_derangements, forward_patient, patient_survival_changes, run,
                                               sha256_file, validation_metrics)


class QueryPermutationTest(unittest.TestCase):
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
        self.manifest = self.root/'manifest.csv'
        generator = torch.Generator().manual_seed(22)
        rows = []
        for i in range(8):
            feature, rna = f'slide{i}.pt', f'rna{i}.npy'
            torch.save(torch.randn(5+i, 6, generator=generator), self.root/feature)
            np.save(self.root/rna, torch.randn(8, generator=generator).numpy())
            rows.append([f'p{i}', 'train' if i < 4 else 'val', feature, rna,
                         20+i if i < 4 else 5*(i-3), 1 if i != 6 else 0])
        with self.manifest.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['patient_id', 'split', 'wsi_pt', 'rna_npy', 'time', 'event'])
            writer.writerows(rows)
        cohort = load_stream_manifest(self.manifest, 'single_view', include_test=False)
        self.addCleanup(lambda: [bag.close() for split in cohort.values() for bag in split['bags']])
        self.cohort = cohort
        torch.manual_seed(100)
        kwargs = dict(path_dim=6, rna_dim=8, pathway_indices=[[0, 1, 2], [2, 3, 4]],
                      hidden=4, common=2, local_dim=4, local_common=2, bins=1,
                      interaction_strength=.5, match_temperature=.2)
        self.model = PathwayLocalSurvival(**kwargs).eval()
        scaler = fit_rna_scaler(cohort['train']['rna'], 'mask', 1e-6)
        config = {'seed': 17}
        stream = {'patch_chunk_size': 3, 'view_source': 'feature_noise', 'device': 'cpu'}
        mapping = {'rna_dim': 8, 'pathways': [{'id': 'A', 'indices': [0, 1, 2]},
                                              {'id': 'B', 'indices': [2, 3, 4]}]}
        checkpoint = {'architecture': 'PathwayLocalSurvival', 'model_state': self.model.state_dict(),
                      'model_kwargs': kwargs, 'pathway_mapping': mapping, **scaler,
                      'starts': torch.tensor([0.]), 'stream_config': stream, 'config': config,
                      'effective_loss_weights': {}, 'manifest_sha256': sha256_file(self.manifest),
                      'cohort_input_sha256': stream_cohort_fingerprint(cohort, 'single_view'),
                      'evaluation_split': 'val', 'test_evaluated': False}
        self.fitted = {**scaler, 'model': self.model, 'starts': checkpoint['starts'], 'stream_config': stream}
        predictions = predict_pathway_model(self.fitted, cohort['val'])
        metrics, survival = validation_metrics(predictions['log_rates'], cohort, checkpoint['starts'], 12)
        self.directory = self.root/'run'
        self.directory.mkdir()
        torch.save(checkpoint, self.directory/'checkpoint.pt')
        report = {key: checkpoint[key] for key in ('architecture', 'model_kwargs', 'pathway_mapping',
                  'rna_preprocessing', 'stream_config', 'config', 'effective_loss_weights',
                  'manifest_sha256', 'cohort_input_sha256', 'evaluation_split', 'test_evaluated')}
        report.update(view_source='feature_noise', interval_starts=[0], prediction_horizon=12,
                      evaluation_nll=metrics['nll'],
                      evaluation_c_index_risk_at_horizon={'value': metrics['harrell_c']},
                      evaluation_ipcw_brier_at_horizon={'value': metrics['ipcw_brier']})
        (self.directory/'report.json').write_text(json.dumps(report), encoding='utf-8')
        with (self.directory/'val_predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['patient_id', 'survival_at_horizon'])
            writer.writerows(zip(cohort['val']['ids'], survival))
        self.sha = sha256_file(self.directory/'checkpoint.pt')

    def test_query_only_and_own_query_reproduces_original(self):
        val = self.cohort['val']
        rna = transform_rna(val['rna'], self.fitted)
        with torch.inference_mode():
            original = forward_patient(self.model, val['bags'][0], rna[0:1], self.fitted['stream_config'])
            own = self.model.match_queries(self.model.pathways(rna[0:1]))
            donor = self.model.match_queries(self.model.pathways(rna[1:2]))
            identity = forward_patient(self.model, val['bags'][0], rna[0:1], self.fitted['stream_config'],
                                       own, original['rna_pathways'])
            changed = forward_patient(self.model, val['bags'][0], rna[0:1], self.fitted['stream_config'],
                                      donor, original['rna_pathways'])
            torch.testing.assert_close(original['log_rates'], identity['log_rates'], rtol=0, atol=0)
            for key in ('path_hidden', 'rna_hidden', 'rna_pathways'):
                torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
            self.assertGreater((original['matched_path']-changed['matched_path']).abs().max().item(), 1e-6)
            # The patch must be restored after the scoped inference.
            torch.testing.assert_close(self.model.match_queries(self.model.pathways(rna[0:1])), own)
            with self.assertRaisesRegex(ValueError, 'unexpected target-patient'):
                forward_patient(self.model, val['bags'][0], rna[0:1], self.fitted['stream_config'],
                                donor, original['rna_pathways']+10)
            torch.testing.assert_close(self.model.match_queries(self.model.pathways(rna[0:1])), own)

    def test_derangements_and_full_synthetic_run(self):
        perms = fixed_derangements(4, 5, 92)
        self.assertEqual(perms, fixed_derangements(4, 5, 92))
        for perm in perms:
            self.assertEqual(sorted(perm), list(range(4)))
            self.assertTrue(all(i != j for i, j in enumerate(perm)))
        result = run(self.manifest, self.directory, self.sha, torch.device('cpu'), 3, 92)
        self.assertFalse(result['test_loaded'])
        self.assertEqual([x['donor_indices_in_validation_order'] for x in result['results']], perms[:3])
        self.assertLess(result['original_max_absolute_survival_difference'], 1e-7)
        self.assertEqual(len(result['results']), 3)
        for item in result['results']:
            summary = item['paired_survival_change']
            self.assertEqual(summary['patients'], 4)
            self.assertGreaterEqual(summary['max_absolute'], summary['mean_absolute'])

    def test_patient_changes_capture_cancellation_and_reject_invalid_values(self):
        result = patient_survival_changes([0.2, 0.8], [0.3, 0.7])
        self.assertAlmostEqual(result['mean_signed'], 0.0)
        self.assertAlmostEqual(result['mean_absolute'], 0.1)
        self.assertAlmostEqual(result['max_absolute'], 0.1)
        with self.assertRaises(ValueError):
            patient_survival_changes([0.5], [float('nan')])
        with self.assertRaises(ValueError):
            patient_survival_changes([0.5], [1.1])

    def test_reject_original_prediction_mismatch(self):
        path = self.directory/'val_predictions.csv'
        lines = path.read_text(encoding='utf-8').splitlines()
        lines[1] = lines[1].split(',')[0]+',0.5'
        path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'disagree with stored predictions'):
            run(self.manifest, self.directory, self.sha, torch.device('cpu'), 1, 92)

    def test_script_outside_project_records_imported_source(self):
        script = self.root/'independent.py'
        shutil.copy2(Path(__file__).with_name('diagnose_pathway_query_permutation.py'), script)
        output = self.root/'output.json'
        project = Path(__file__).resolve().parent
        env = dict(os.environ, PYTHONPATH=str(project), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        result = subprocess.run([sys.executable, str(script), '--manifest', str(self.manifest),
                                 '--run-dir', str(self.directory), '--checkpoint-sha256', self.sha,
                                 '--device', 'cpu', '--permutations', '1', '--seed', '92', '--out', str(output)],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(data['inputs']['source_sha256']['cancer_pathway_stream'],
                         sha256_file(project/'cancer_pathway_stream.py'))
        self.assertEqual(data['inputs']['script_sha256'], sha256_file(script))

    def test_reject_nonfinite_report_metric(self):
        path = self.directory/'report.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        report['evaluation_nll'] = float('nan')
        path.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'original nll differs'):
            run(self.manifest, self.directory, self.sha, torch.device('cpu'), 1, 92)

    def test_reject_hash_preprocessing_and_test_rows(self):
        with self.assertRaisesRegex(ValueError, 'Checkpoint SHA256'):
            run(self.manifest, self.directory, '0'*64, torch.device('cpu'), 1, 92)
        checkpoint = torch.load(self.directory/'checkpoint.pt', map_location='cpu', weights_only=True)
        checkpoint['rna_mean'][0] += 1
        torch.save(checkpoint, self.directory/'checkpoint.pt')
        with self.assertRaisesRegex(ValueError, 'rna_mean'):
            run(self.manifest, self.directory, sha256_file(self.directory/'checkpoint.pt'),
                torch.device('cpu'), 1, 92)
        with self.manifest.open('a', encoding='utf-8') as f:
            f.write('outside,test,ABSENT.pt,ABSENT.npy,3,1\n')
        with self.assertRaisesRegex(ValueError, 'no test rows'):
            run(self.manifest, self.directory, self.sha, torch.device('cpu'), 1, 92)


if __name__ == '__main__':
    unittest.main()
