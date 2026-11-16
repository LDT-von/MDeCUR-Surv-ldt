"""Synthetic contract regressions for legacy parity diagnostic entry points."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_parity_interaction as interaction
import diagnose_parity_modalities as modalities
import diagnose_rna_only_baseline as ridge
from cancer_pathway import PathwayLocalSurvival


class SyntheticBag:
    view_source = 'single_view'
    feature_dim = 3
    n_patches = 5
    def __init__(self, ident):
        self.patient_id = ident
        self.closed = False
    def blocks(self, size):
        for start in range(0, self.n_patches, size):
            yield ('synthetic', start, min(start+size, self.n_patches))
    def read_rows(self, rows):
        return torch.ones(len(rows), self.feature_dim)
    def assert_unchanged(self):
        return None
    def close(self):
        self.closed = True


class ParityDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_dual_ridge_equals_primal_without_gene_gram(self):
        g = np.random.default_rng(31)
        x, v, y = g.normal(size=(7, 13)), g.normal(size=(4, 13)), g.normal(size=(7, 3))
        for alpha in (.1, 100., 1e8):
            expected = v @ np.linalg.solve(x.T @ x+alpha*np.eye(13), x.T @ y)
            observed = ridge.dual_ridge_logits(x, y, v, alpha)
            np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-10)
        self.assertFalse(ridge.RIDGE_SEMANTICS['censoring_aware_training'])
        self.assertIn('No survival NLL', ridge.RIDGE_SEMANTICS['nll_semantics'])

    def test_contributions_use_current_strength_and_patient_spans(self):
        base = torch.tensor([[2., 3.], [4., 8.]])
        raw = torch.tensor([[5., -2.], [7., 6.]])
        for strength in (0., .2, 1.):
            result = interaction.decompose_output({'log_rates': base+strength*raw,
                                                    'interaction_logits': raw}, strength)
            self.assertAlmostEqual(result['base_abs_mean'], float(base.abs().mean()), places=6)
            self.assertAlmostEqual(result['weighted_interaction_abs_mean'], float((strength*raw).abs().mean()), places=6)
            np.testing.assert_allclose(result['base_span_by_bin'], [2., 5.], atol=1e-6)
            np.testing.assert_allclose(result['weighted_interaction_span_by_bin'], np.array([2., 8.])*strength, atol=1e-6)
        zero = interaction.decompose_output({'log_rates': torch.zeros(2, 1),
                                              'interaction_logits': torch.zeros(2, 1)}, 0.)
        self.assertIsNone(zero['weighted_to_base_abs_ratio'])
        self.assertEqual(zero['weighted_to_base_span_ratio_by_bin'], [None])

    def test_intervention_restores_strength_on_success_and_exception(self):
        model = SimpleNamespace(interaction_strength=.37)
        seen = []
        def forward(*args, **kwargs):
            seen.append(model.interaction_strength)
            return {'ok': True}
        with patch.object(interaction, 'pathway_stream_forward', side_effect=forward):
            self.assertEqual(interaction.predict_at_strength(model, [], None, 2, 0.), {'ok': True})
        self.assertEqual(seen, [0.])
        self.assertEqual(model.interaction_strength, .37)
        with patch.object(interaction, 'pathway_stream_forward', side_effect=RuntimeError('toy')):
            with self.assertRaises(RuntimeError):
                interaction.predict_at_strength(model, [], None, 2, 1.)
        self.assertEqual(model.interaction_strength, .37)

    def test_dense_rejected_even_without_architecture_metadata(self):
        for checkpoint in ({'architecture': 'DensePathwaySurvival', 'model_state': {}},
                           {'model_state': {'token_cooperation.head.weight': torch.zeros(1)}}):
            with self.assertRaisesRegex(ValueError, 'legacy'):
                interaction.legacy_model(checkpoint, 'cpu')

    def test_zero_input_chunks_and_derangement(self):
        bag = modalities.ZeroInputBag(SyntheticBag('toy'))
        counts = [len(bag.read_block(t, ('features_reference',))[0]) for t in bag.blocks(2)]
        self.assertEqual(counts, [2, 2, 1])
        permutation = modalities.full_rna_derangement(60, 0)
        self.assertFalse(np.any(permutation == np.arange(60)))
        self.assertEqual(sorted(permutation.tolist()), list(range(60)))
        np.testing.assert_array_equal(permutation, modalities.full_rna_derangement(60, 0))

    def test_nullable_metrics_and_output_overwrite_refusal(self):
        self.assertEqual(ridge.metric_text(None), 'NA')
        self.assertEqual(ridge.metric_text(float('nan')), 'NA')
        self.assertEqual(ridge.metric_text(.75), '0.7500')
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'report.json'
            ridge.write_result(destination, {'metric': None})
            before = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                ridge.write_result(destination, {'changed': True})
            self.assertEqual(destination.read_bytes(), before)
            args = SimpleNamespace(out=str(destination))
            with patch.object(interaction.torch, 'load') as load:
                with self.assertRaises(FileExistsError):
                    with interaction.loaded_legacy_inputs(args):
                        pass
                load.assert_not_called()

    def fixture(self, root):
        kwargs = dict(path_dim=3, rna_dim=3, pathway_indices=[[0, 1], [1, 2]],
                      hidden=4, common=2, local_dim=4, local_common=2, bins=2)
        model = PathwayLocalSurvival(**kwargs)
        manifest = root/'manifest.csv'
        manifest.write_text('synthetic manifest fixture\n')
        reference = root/'reference.csv'
        reference.write_text('patient_id,time,event,split\nt0,1,1,train\nt1,2,0,train\nv0,3,1,val\nv1,4,0,val\n')
        checkpoint = {'architecture': 'PathwayLocalSurvival', 'model_kwargs': kwargs,
                      'model_state': model.state_dict(), 'bin_edges': ['-Infinity', 2.5, 'Infinity'],
                      'rna_mean': torch.tensor([1., 2., 3.]), 'rna_scale': torch.tensor([2., 3., 4.]),
                      'rna_mask': torch.ones(3, dtype=torch.bool),
                      'rna_preprocessing': {'policy': 'clamp', 'min_std': 1e-6},
                      'manifest_sha256': interaction.file_sha256(manifest),
                      'survival_reference_sha256': interaction.file_sha256(reference),
                      'cohort_input_sha256': 'synthetic_cohort_hash',
                      'patch_chunk_size': 2, 'num_patches': 5,
                      'loss_version': 'patientwise_clamped_discrete_v1',
                      'controls': {'frozen_modality': 'rna', 'input_control_version': 'parity_input_zero_v1'}}
        path = root/'checkpoint.pt'
        torch.save(checkpoint, path)
        cohort = {}
        for split, ids, times in (('train', ['t0', 't1'], [1., 2.]), ('val', ['v0', 'v1'], [3., 4.])):
            cohort[split] = {'ids': ids, 'bags': [SyntheticBag(i) for i in ids],
                             'rna': torch.tensor([[5., 8., 11.], [3., 5., 7.]]),
                             'time': torch.tensor(times), 'event': torch.tensor([1., 0.])}
        args = argparse.Namespace(checkpoint=str(path), manifest=str(manifest), survival_reference=str(reference),
                                  out=str(root/'out.json'), num_patches=5, patch_chunk_size=2,
                                  seed=3, n_bins=2, device='cpu')
        return args, checkpoint, cohort

    def test_saved_scaler_controls_and_input_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            args, checkpoint, cohort = self.fixture(Path(directory))
            calls = []
            def replay(ckpt, bags, rna):
                calls.append(rna.clone())
                return bags, torch.zeros_like(rna), {'input_contract_verified': True}
            helper = SimpleNamespace(apply_checkpoint_controls=replay)
            with patch.dict(sys.modules, {'parity_checkpoint_controls': helper}), \
                    patch.object(interaction, 'load_stream_manifest', return_value=cohort), \
                    patch.object(interaction, 'stream_cohort_fingerprint', return_value='synthetic_cohort_hash'):
                with interaction.loaded_legacy_inputs(args) as (_, _, normalized, _, _, evidence):
                    self.assertTrue(torch.equal(normalized, torch.zeros(2, 3)))
                    self.assertEqual(evidence['input_verification'], 'saved_input_contracts_verified')
                    self.assertFalse(evidence['baseline_prediction_reproduction_checked'])
                self.assertTrue(evidence['model_and_input_guards_passed'])
            self.assertTrue(torch.equal(calls[0], torch.tensor([[2., 2., 2.], [1., 1., 1.]])))
            self.assertTrue(all(b.closed for d in cohort.values() for b in d['bags']))
            checkpoint['manifest_sha256'] = 'wrong'
            torch.save(checkpoint, args.checkpoint)
            with self.assertRaisesRegex(ValueError, 'manifest_sha256'):
                with interaction.loaded_legacy_inputs(args):
                    pass

    def test_historical_frozen_control_omission_rejected_and_missing_hashes_labelled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, checkpoint, cohort = self.fixture(root)
            checkpoint.pop('controls')
            checkpoint.pop('manifest_sha256')
            checkpoint.pop('cohort_input_sha256')
            torch.save(checkpoint, args.checkpoint)
            helper = SimpleNamespace(apply_checkpoint_controls=lambda c, b, r: (b, r, {'status': 'not_recorded'}))
            with patch.dict(sys.modules, {'parity_checkpoint_controls': helper}), \
                    patch.object(interaction, 'load_stream_manifest', return_value=cohort), \
                    patch.object(interaction, 'stream_cohort_fingerprint', return_value='current_only'):
                with interaction.loaded_legacy_inputs(args) as (_, _, _, _, _, evidence):
                    self.assertEqual(evidence['input_verification'], 'historical_input_unverified')
                    self.assertIn('manifest_sha256', evidence['missing_historical_contracts'])
            (root/'report.json').write_text(json.dumps({'controls': {'frozen_modality': 'pathology'}}))
            with self.assertRaisesRegex(ValueError, 'report records a frozen modality'):
                with interaction.loaded_legacy_inputs(args):
                    pass

    def test_both_cli_entries_save_nullable_metrics_after_guarded_forward(self):
        metrics = {'c_index': .6, 'c_index_ipcw': None, 'ibs': None, 'iauc': None,
                   'errors': ['synthetic unavailable censoring metric']}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, cohort = self.fixture(root)
            for module in (interaction, modalities):
                out = root/(module.__name__+'.json')
                argv = [module.__name__, '--checkpoint', args.checkpoint, '--manifest', args.manifest,
                        '--survival-reference', args.survival_reference, '--num-patches', '5',
                        '--patch-chunk-size', '2', '--out', str(out)]
                with patch.object(interaction, 'load_stream_manifest', return_value=cohort), \
                        patch.object(interaction, 'stream_cohort_fingerprint', return_value='synthetic_cohort_hash'), \
                        patch.object(interaction, 'parity_metrics', return_value=metrics), \
                        patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()) as output:
                    module.main()
                result = json.loads(out.read_text())
                self.assertTrue(result['evidence']['model_and_input_guards_passed'])
                self.assertEqual(result['evidence']['controls_replay']['frozen_modality'], 'rna')
                self.assertTrue(result['evidence']['controls_replay']['input_contract_verified'])
                self.assertEqual(result['evidence']['input_verification'], 'saved_input_contracts_verified')
                self.assertIn('IPCW=NA', output.getvalue())
                self.assertTrue(all(row['c_index_ipcw'] is None for row in result['cases']))

    def test_real_replay_helper_rejects_saved_budget_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            args, checkpoint, cohort = self.fixture(Path(directory))
            checkpoint['num_patches'] = 6
            torch.save(checkpoint, args.checkpoint)
            with patch.object(interaction, 'load_stream_manifest', return_value=cohort), \
                    patch.object(interaction, 'stream_cohort_fingerprint', return_value='synthetic_cohort_hash'):
                with self.assertRaisesRegex(ValueError, 'patch budget differs'):
                    with interaction.loaded_legacy_inputs(args):
                        pass


if __name__ == '__main__':
    unittest.main()
