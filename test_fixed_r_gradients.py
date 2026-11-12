"""Train-only fixed-checkpoint audit contracts; synthetic files and small bags."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import PathwayConfig, effective_weights, minimum_batch_size
from cancer_rna import fit_rna_scaler
from cancer_stream_data import load_stream_manifest
import diagnose_fixed_r_gradients as diag


class FixedRGradientsTest(unittest.TestCase):
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
        self.directory = self.root/'model'
        self.directory.mkdir()
        generator = torch.Generator().manual_seed(42)
        rows, rna = [], []
        patches = 0
        for i in range(18):
            x = torch.randn(5+i % 3, 6, generator=generator)
            g = torch.randn(8, generator=generator)
            torch.save(x, self.root/f'p{i}.pt')
            np.save(self.root/f'g{i}.npy', g.numpy())
            rows.append([f'p{i}', 'train', f'p{i}.pt', f'g{i}.npy', 10+i, i % 2])
            rna.append(g)
            patches += len(x)
        # Held-out inputs do not exist and labels are deliberately unparsable.
        rows += [['sealed_val', 'val', 'DO_NOT_OPEN.pt', 'DO_NOT_OPEN.npy', 'SEALED', 'SEALED'],
                 ['sealed_test', 'test', 'NEVER_OPEN.pt', 'NEVER_OPEN.npy', 'SEALED', 'SEALED']]
        with self.manifest.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['patient_id', 'split', 'wsi_pt', 'rna_npy', 'time', 'event'])
            writer.writerows(rows)
        cfg = PathwayConfig(hidden=4, common=2, local_dim=4, local_common=2, common_mode='stability')
        kwargs = dict(path_dim=6, rna_dim=8, pathway_indices=[[0, 1, 2], [2, 3, 4]],
                      hidden=4, common=2, local_dim=4, local_common=2, bins=1,
                      interaction_strength=cfg.interaction_strength, match_temperature=cfg.match_temperature)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = PathwayLocalSurvival(**kwargs)
        scaler = fit_rna_scaler(torch.stack(rna), cfg.rna_scale_policy, cfg.rna_min_std)
        stream = dict(patient_batch_size=6, patch_chunk_size=3, view_source='feature_noise', device='cpu',
                      rna_noise=cfg.rna_noise, feature_noise=cfg.feature_noise,
                      minimum_patient_batch=minimum_batch_size(cfg), checkpoint_chunks=True,
                      auxiliary_rng_scheme='rna_global__weight_shuffle_isolated_v1')
        self.checkpoint = dict(architecture='PathwayLocalSurvival', model_state=model.state_dict(),
                               model_kwargs=kwargs, config=asdict(cfg), stream_config=stream, **scaler,
                               starts=torch.tensor([0.]), effective_loss_weights=effective_weights(cfg),
                               pathway_mapping={'rna_dim': 8, 'pathways': [{'id': 'A', 'indices': [0, 1, 2]},
                                                                         {'id': 'B', 'indices': [2, 3, 4]}]},
                               manifest_sha256=diag.sha256_file(self.manifest), cohort_input_sha256='a'*64,
                               evaluation_split='val', test_evaluated=False)
        self.report = {key: self.checkpoint[key] for key in (
            'architecture', 'model_kwargs', 'config', 'stream_config', 'effective_loss_weights',
            'pathway_mapping', 'rna_preprocessing', 'manifest_sha256', 'cohort_input_sha256',
            'evaluation_split', 'test_evaluated')}
        self.report.update(view_source='feature_noise', interval_starts=[0.], prediction_horizon=12.,
                           patients={'train': 18}, events={'train': 9}, patches={'train': patches})
        self.save_artifacts()

    def save_artifacts(self):
        torch.save(self.checkpoint, self.directory/'checkpoint.pt')
        (self.directory/'report.json').write_text(json.dumps(self.report), encoding='utf-8')
        self.sha = diag.sha256_file(self.directory/'checkpoint.pt')

    def save_plan(self):
        plan = diag.make_plan(self.manifest, self.directory, self.sha)
        path = self.root/'plan.json'
        path.write_text(json.dumps(plan, allow_nan=False), encoding='utf-8')
        return plan, path, diag.sha256_file(path)

    def test_train_only_skips_held_out_io_and_labels_and_plan_has_no_forward(self):
        with patch.object(diag, 'audit_batch', side_effect=AssertionError('no gradient in plan')), \
             patch.object(PathwayLocalSurvival, 'prediction', side_effect=AssertionError('no forward in plan')):
            plan, _, _ = self.save_plan()
        self.assertFalse(plan['validation_loaded'])
        self.assertFalse(plan['test_loaded'])
        self.assertFalse(plan['inputs']['historical_combined_cohort_recomputed'])
        self.assertEqual(len(plan['batches']), 3)
        self.assertEqual(len({p for b in plan['batches'] for p in b['patient_ids']}), 18)
        self.assertEqual(plan, diag.make_plan(self.manifest, self.directory, self.sha))
        with self.assertRaisesRegex(ValueError, 'include_test=False'):
            load_stream_manifest(self.manifest, 'single_view', train_only=True)

    def test_real_autograd_six_records_preserves_checkpoint_rng_and_no_optimizer(self):
        plan, path, digest = self.save_plan()
        original = (self.directory/'checkpoint.pt').read_bytes()
        rng = torch.get_rng_state().clone()
        with patch.object(torch.optim.AdamW, '__init__', side_effect=AssertionError('no optimizer')), \
             patch.object(torch.optim.SGD, '__init__', side_effect=AssertionError('no optimizer')):
            result = diag.run(self.manifest, self.directory, self.sha, path, digest, 'cpu')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(original, (self.directory/'checkpoint.pt').read_bytes())
        self.assertEqual(result['model_state_before_sha256'], result['model_state_after_sha256'])
        self.assertEqual(len(result['records']), 6)
        self.assertEqual(result['unique_diagnosed_patients'], 18)
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertTrue(result['parameter_grads_remain_none'])
        self.assertEqual(result['plan'], plan)
        for record in result['records']:
            self.assertEqual(len(record['audit']['terms']), 11)
            self.assertLess(record['audit']['gradient_reconstruction']['relative_l2_error'], 1e-5)
        self.assertEqual(result['summary']['matched_common']['term_status_counts'], {'computed': 6})
        json.dumps(result, allow_nan=False)

    def test_plan_digest_and_batch_schedule_tampering_rejected_before_audit(self):
        plan, path, digest = self.save_plan()
        with patch.object(diag, 'audit_batch', side_effect=AssertionError('must reject before audit')):
            with self.assertRaisesRegex(ValueError, 'Plan SHA256'):
                diag.run(self.manifest, self.directory, self.sha, path, '0'*64, 'cpu')
            plan['batches'][0]['views'][0]['view_seed'] += 1
            path.write_text(json.dumps(plan), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Plan no longer matches'):
                diag.run(self.manifest, self.directory, self.sha, path, diag.sha256_file(path), 'cpu')

    def test_changed_training_features_and_saved_config_rejected(self):
        _, path, digest = self.save_plan()
        torch.save(torch.zeros(5, 6), self.root/'p0.pt')
        with patch.object(diag, 'audit_batch', side_effect=AssertionError('must reject before audit')):
            with self.assertRaisesRegex(ValueError, 'Plan no longer matches'):
                diag.run(self.manifest, self.directory, self.sha, path, digest, 'cpu')
        self.checkpoint['effective_loss_weights']['matched_common'] = 0.
        self.save_artifacts()
        with self.assertRaisesRegex(ValueError, 'Expected full R'):
            diag.make_plan(self.manifest, self.directory, self.sha)

    def test_checkpoint_hash_and_scaler_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Checkpoint SHA256'):
            diag.make_plan(self.manifest, self.directory, '0'*64)
        self.checkpoint['rna_mean'] = self.checkpoint['rna_mean']+1.
        self.save_artifacts()
        with self.assertRaisesRegex(ValueError, 'rna_mean differs'):
            diag.make_plan(self.manifest, self.directory, self.sha)

    def test_state_and_rna_file_mutation_during_audit_rejected(self):
        _, path, digest = self.save_plan()
        def change_state(model, *args):
            with torch.no_grad():
                next(model.parameters()).add_(1.)
            return {}
        with patch.object(diag, 'audit_batch', side_effect=change_state):
            with self.assertRaisesRegex(RuntimeError, 'Fixed model state'):
                diag.run(self.manifest, self.directory, self.sha, path, digest, 'cpu')
        with self.assertRaisesRegex(RuntimeError, 'Training RNA files changed'):
            with diag.loaded_training_reference(self.manifest, self.directory, self.sha, 'cpu'):
                np.save(self.root/'g0.npy', np.zeros(8, dtype=np.float32))

    def test_fixed_batches_tail_merge_and_minimum_cohort(self):
        batches = diag.fixed_batches([str(i) for i in range(19)], 6, 5)
        self.assertEqual([len(b['patient_ids']) for b in batches], [6, 6, 7])
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            diag.fixed_batches([str(i) for i in range(12)], 6, 5)

    def test_cli_plan_then_audit_and_no_overwrite(self):
        script = Path(diag.__file__)
        plan_path, output = self.root/'cli_plan.json', self.root/'audit.json'
        base = [sys.executable, str(script), '--manifest', str(self.manifest), '--run-dir', str(self.directory),
                '--checkpoint-sha256', self.sha]
        command = base+['--plan-out', str(plan_path)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(base+['--plan', str(plan_path), '--plan-sha256', diag.sha256_file(plan_path),
                                     '--out', str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(len(saved['records']), 6)
        self.assertEqual(saved['plan']['source_sha256']['entrypoint'], diag.sha256_file(script))
        before = plan_path.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FileExistsError', result.stderr)
        self.assertEqual(plan_path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
