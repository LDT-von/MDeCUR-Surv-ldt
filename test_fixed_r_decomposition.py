"""Synthetic known-answer and end-to-end guards for frozen R decomposition."""
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cancer_pathway import PathwayLocalSurvival
from cancer_rna import transform_rna
from diagnose_fixed_r_decomposition import (paired_change, run, sha256_file,
                                           streaming_attention_l1, training_mean_query)
import test_diagnose_pathway_query_permutation as fixtures


class FixedRDecompositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        # Reuse only the synthetic fixture builder, without rediscovering the
        # other test class or inheriting/rerunning all its test methods.
        fixtures.QueryPermutationTest.setUp(self)

    def test_full_synthetic_run_preserves_state_and_isolates_components(self):
        original_bytes = (self.directory/'checkpoint.pt').read_bytes()
        result = run(self.manifest, self.directory, self.sha, torch.device('cpu'), count=2, seed=92)
        self.assertEqual(result['model_state_before_sha256'], result['model_state_after_sha256'])
        self.assertFalse(result['test_loaded'])
        self.assertFalse(result['training_performed'])
        self.assertLess(result['original_reproduction_max_survival_difference'], 1e-7)
        self.assertEqual(original_bytes, (self.directory/'checkpoint.pt').read_bytes())
        self.assertEqual(result['fixed_query']['training_patients'], 4)
        self.assertEqual(set(result['cases']), {'original', 'permuted_1', 'permuted_2',
                                              'training_fixed', 'uniform', 'interaction_off'})
        for name, case in result['cases'].items():
            self.assertEqual(case['changes']['base_logits']['patient_l2_change']['max'], 0.)
            self.assertLessEqual(case['attention_l1_patient_mean']['max'], 2.0000001)
        original, off = result['cases']['original'], result['cases']['interaction_off']
        self.assertEqual(original['survival_change']['patient_mean_absolute']['max'], 0.)
        self.assertEqual(off['changes']['matched_path']['patient_l2_change']['max'], 0.)
        self.assertEqual(off['changes']['interaction_raw']['patient_l2_change']['max'], 0.)
        self.assertGreater(off['changes']['interaction_applied']['patient_l2_change']['max'], 0.)
        self.assertEqual(off['absolute_levels']['interaction_applied']['max'], 0.)
        # At least one donor changes both the matching and applied interaction.
        self.assertGreater(result['cases']['permuted_1']['changes']['matched_path']['patient_l2_change']['max'], 0.)
        json.dumps(result, allow_nan=False)

    def test_training_fixed_query_is_train_mean_and_zero_mean_fallback(self):
        normalized = transform_rna(self.cohort['train']['rna'], self.fitted)
        fixed, info = training_mean_query(self.model, normalized)
        with torch.inference_mode():
            qs = [self.model.match_queries(self.model.pathways(row)).double() for row in normalized.split(1)]
            expected = F.normalize(torch.stack(qs).mean(0), dim=-1).to(fixed.dtype)
        torch.testing.assert_close(fixed, expected)
        self.assertEqual(info['training_patients'], 4)
        class FakeModel:
            def pathways(self, x):
                return x[:, None, :]
            def match_queries(self, x):
                return F.normalize(x, dim=-1)
        zero, metadata = training_mean_query(FakeModel(), torch.tensor([[1., 0.], [-1., 0.]]))
        torch.testing.assert_close(zero, torch.zeros(1, 1, 2))
        self.assertEqual(metadata['near_zero_mean_pathway_indices'], [0])

    def test_streaming_attention_matches_analytic_two_patch_answer(self):
        from test_cancer_pathway import SyntheticBag
        model = PathwayLocalSurvival(4, 4, [[0, 1]], hidden=4, common=2,
                                     local_dim=4, local_common=2, match_temperature=.2)
        model.base.patch_encoder = nn.Identity()
        with torch.no_grad():
            model.match_path.weight.copy_(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]))
        x = torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.]])
        bag = SyntheticBag('s', x, x, x)
        queries = [torch.tensor([[[1., 0.]]]), torch.zeros(1, 1, 2), torch.tensor([[[-1., 0.]]])]
        a = streaming_attention_l1(model, bag, queries, 1)
        b = streaming_attention_l1(model, bag, queries, 20)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(a.numpy()[:, 0], [0., math.tanh(5.), 2*math.tanh(5.)], atol=1e-12)

    def test_uniform_query_equals_unweighted_patch_mean(self):
        from diagnose_pathway_query_permutation import forward_patient
        rna = transform_rna(self.cohort['val']['rna'], self.fitted)[0:1]
        bag = self.cohort['val']['bags'][0]
        with torch.inference_mode():
            original = forward_patient(self.model, bag, rna, self.fitted['stream_config'])
            q = torch.zeros_like(self.model.match_queries(original['rna_pathways']))
            changed = forward_patient(self.model, bag, rna, self.fitted['stream_config'], q, original['rna_pathways'])
            features = []
            for block in bag.blocks(3):
                raw, = bag.read_block(block, ('features_reference',))
                features.append(self.model.base.patch_encoder(raw))
            expected = torch.cat(features).mean(0)
            for pathway in changed['matched_path'][0]:
                torch.testing.assert_close(pathway, expected, atol=1e-6, rtol=1e-6)

    def test_changed_matching_can_be_hidden_by_zero_interaction_head(self):
        from diagnose_pathway_query_permutation import forward_patient
        rna = transform_rna(self.cohort['val']['rna'], self.fitted)
        with torch.no_grad():
            self.model.interaction_head.weight.zero_()
        with torch.inference_mode():
            original = forward_patient(self.model, self.cohort['val']['bags'][0], rna[0:1], self.fitted['stream_config'])
            donor = self.model.match_queries(self.model.pathways(rna[1:2]))
            changed = forward_patient(self.model, self.cohort['val']['bags'][0], rna[0:1], self.fitted['stream_config'],
                                      donor, original['rna_pathways'])
            self.assertGreater((changed['matched_path']-original['matched_path']).abs().max().item(), 1e-6)
            torch.testing.assert_close(changed['log_rates'], original['log_rates'], rtol=0, atol=0)

    def test_cli_success_default_five_permutations_and_source_hash(self):
        out = self.root/'cli_result.json'
        script = Path(__file__).with_name('diagnose_fixed_r_decomposition.py')
        completed = subprocess.run([sys.executable, str(script), '--manifest', str(self.manifest),
                                   '--run-dir', str(self.directory), '--checkpoint-sha256', self.sha,
                                   '--device', 'cpu', '--out', str(out)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(len(result['permutations']), 5)
        self.assertEqual(len(result['cases']), 9)
        self.assertEqual(result['inputs']['script_sha256'], sha256_file(script))

    def test_reject_test_rows_before_loader_and_bad_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'Checkpoint SHA256'):
            run(self.manifest, self.directory, '0'*64, torch.device('cpu'), count=1)
        with self.manifest.open('a', encoding='utf-8') as handle:
            handle.write('outside,test,ABSENT.pt,ABSENT.npy,3,1\n')
        with patch('diagnose_fixed_r_decomposition.load_stream_manifest') as loader:
            with self.assertRaisesRegex(ValueError, 'no test rows'):
                run(self.manifest, self.directory, self.sha, torch.device('cpu'), count=1)
            loader.assert_not_called()

    def test_reproduction_failure_prevents_interventions(self):
        path = self.directory/'val_predictions.csv'
        rows = path.read_text(encoding='utf-8').splitlines()
        rows[1] = rows[1].split(',')[0]+',0.5'
        path.write_text('\n'.join(rows)+'\n', encoding='utf-8')
        with patch('diagnose_fixed_r_decomposition.streaming_attention_l1') as attention:
            with self.assertRaisesRegex(ValueError, 'disagree with stored predictions'):
                run(self.manifest, self.directory, self.sha, torch.device('cpu'), count=1)
            attention.assert_not_called()

    def test_zero_relative_denominator_and_cli_no_overwrite(self):
        result = paired_change(torch.zeros(3, 2), torch.ones(3, 2))
        self.assertIsNone(result['relative_l2'])
        self.assertEqual(result['relative_l2_near_zero_baseline_patients'], 3)
        out = self.root/'existing.json'
        out.write_text('preserve', encoding='utf-8')
        script = Path(__file__).with_name('diagnose_fixed_r_decomposition.py')
        completed = subprocess.run([sys.executable, str(script), '--manifest', 'ABSENT', '--run-dir', 'ABSENT',
                                    '--checkpoint-sha256', '0'*64, '--device', 'cpu', '--out', str(out)],
                                   capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('FileExistsError', completed.stderr)
        self.assertEqual(out.read_text(encoding='utf-8'), 'preserve')


if __name__ == '__main__':
    unittest.main()
