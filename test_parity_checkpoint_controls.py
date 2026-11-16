"""Generated-data regression checks for saved parity input-zero controls."""
import argparse
import csv
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import diagnose_parity_interaction as diagnostic
from cancer_parity import DISCRETE_NLL_VERSION
from cancer_parity_bag import BudgetedPatientBag
from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import pathway_stream_forward
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
from parity_checkpoint_controls import (INPUT_CONTROL_VERSION, ZeroPatientBag,
                                        apply_checkpoint_controls)


class GeneratedBag:
    view_source = 'single_view'

    def __init__(self, patient_id, features):
        self.patient_id = patient_id
        self.features = features
        self.n_patches, self.feature_dim = features.shape
        self.provenance = [{'path': 'generated-'+patient_id, 'rows': self.n_patches}]
        self.checked = 0
        self.closed = 0
        self.reads = 0

    def blocks(self, chunk_size):
        for start in range(0, self.n_patches, chunk_size):
            yield ('generated', start, min(start+chunk_size, self.n_patches))

    def read_rows(self, rows):
        self.reads += 1
        return self.features[rows]

    def assert_unchanged(self):
        self.checked += 1

    def close(self):
        self.closed += 1


def contract(frozen='none', budget=8, chunk=2):
    return {'controls': {'frozen_modality': frozen, 'input_control_version': INPUT_CONTROL_VERSION},
            'num_patches': budget, 'patch_chunk_size': chunk, 'loss_version': DISCRETE_NLL_VERSION}


def rows(bag, chunk):
    return torch.cat([bag.read_block(token, ('features_reference',))[0]
                      for token in bag.blocks(chunk)])


class CheckpointControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.rng_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, self.rng_state)

    def fixture(self):
        generator = torch.Generator().manual_seed(721)
        sources = [GeneratedBag('short', torch.randn(3, 3, generator=generator)),
                   GeneratedBag('long', torch.randn(11, 3, generator=generator))]
        bags = [BudgetedPatientBag(source, 8, mode='eval') for source in sources]
        rna = torch.tensor([[1., 2., -1.], [-2., .5, 3.]])
        return sources, bags, rna

    def test_saved_controls_reproduce_predictions_for_each_frozen_modality(self):
        sources, bags, rna = self.fixture()
        torch.manual_seed(91)
        kwargs = dict(path_dim=3, rna_dim=3, pathway_indices=[[0, 1], [1, 2]],
                      hidden=4, common=2, local_dim=4, local_common=2, bins=2)
        original_model = PathwayLocalSurvival(**kwargs).eval()
        for frozen in ('none', 'pathology', 'rna'):
            with self.subTest(frozen=frozen):
                checkpoint = dict(contract(frozen), model_kwargs=kwargs,
                                  model_state=original_model.state_dict())
                buffer = io.BytesIO()
                torch.save(checkpoint, buffer)
                buffer.seek(0)
                saved = torch.load(buffer, map_location='cpu', weights_only=True)
                replay_model = PathwayLocalSurvival(**saved['model_kwargs']).eval()
                replay_model.load_state_dict(saved['model_state'])
                replay_bags, replay_rna, evidence = apply_checkpoint_controls(saved, bags, rna)
                # Construct expected zero inputs independently of ZeroPatientBag.
                expected_sources = [GeneratedBag(s.patient_id, torch.zeros_like(s.features)
                                                 if frozen == 'pathology' else s.features.clone())
                                    for s in sources]
                expected_bags = [BudgetedPatientBag(source, 8, mode='eval')
                                 for source in expected_sources]
                expected_rna = torch.zeros_like(rna) if frozen == 'rna' else rna.clone()
                with torch.no_grad():
                    expected = pathway_stream_forward(original_model, expected_bags, expected_rna, 8,
                        'feature_noise', include_views=False, include_patch=False, checkpoint_chunks=False)
                    for chunk in (1, 2, 5, 8, 13):
                        observed = pathway_stream_forward(replay_model, replay_bags, replay_rna, chunk,
                            'feature_noise', include_views=False, include_patch=False, checkpoint_chunks=False)
                        torch.testing.assert_close(observed['log_rates'], expected['log_rates'],
                                                   rtol=2e-6, atol=2e-6)
                self.assertTrue(evidence['input_contract_verified'])
                self.assertEqual(evidence['frozen_modality'], frozen)
                self.assertEqual(evidence['num_patches'], 8)
                self.assertEqual(evidence['patch_chunk_size'], 2)

    def test_legacy_loader_replays_real_saved_inputs_and_scaler_for_every_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = torch.Generator().manual_seed(291)
            manifest, reference = root/'manifest.csv', root/'reference.csv'
            records = []
            for i, (split, count) in enumerate((('train', 5), ('train', 11), ('val', 3), ('val', 12))):
                torch.save(torch.randn(count, 3, generator=generator), root/f'p{i}.pt')
                np.save(root/f'g{i}.npy', np.array([i+1., i+3., i+5.], dtype=np.float32))
                records.append([f'p{i}', split, f'p{i}.pt', f'g{i}.npy', i+1., i % 2])
            with manifest.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['patient_id', 'split', 'wsi_pt', 'rna_npy', 'time', 'event'])
                writer.writerows(records)
            with reference.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['patient_id', 'split', 'time', 'event'])
                writer.writerows([[row[0], row[1], row[4], row[5]] for row in records])
            cohort = load_stream_manifest(manifest, 'single_view', include_test=False)
            try:
                fingerprint = stream_cohort_fingerprint(cohort, 'single_view')
            finally:
                for data in cohort.values():
                    for bag in data['bags']:
                        bag.close()
            kwargs = dict(path_dim=3, rna_dim=3, pathway_indices=[[0, 1], [1, 2]],
                          hidden=4, common=2, local_dim=4, local_common=2, bins=2)
            model = PathwayLocalSurvival(**kwargs).eval()
            args = argparse.Namespace(checkpoint=str(root/'checkpoint.pt'), manifest=str(manifest),
                survival_reference=str(reference), out=str(root/'unused_output.json'),
                num_patches=8, patch_chunk_size=2, seed=3, n_bins=2, device='cpu')
            for frozen in ('none', 'pathology', 'rna'):
                with self.subTest(frozen=frozen):
                    checkpoint = dict(contract(frozen), architecture='PathwayLocalSurvival',
                        model_kwargs=kwargs, model_state=model.state_dict(),
                        bin_edges=['-Infinity', 2.5, 'Infinity'],
                        rna_mean=torch.tensor([1., 1., 1.]), rna_scale=torch.tensor([2., 2., 2.]),
                        rna_mask=torch.ones(3, dtype=torch.bool),
                        rna_preprocessing={'policy': 'clamp', 'min_std': 1e-6},
                        manifest_sha256=diagnostic.file_sha256(manifest),
                        survival_reference_sha256=diagnostic.file_sha256(reference),
                        cohort_input_sha256=fingerprint)
                    torch.save(checkpoint, args.checkpoint)
                    with diagnostic.loaded_legacy_inputs(args) as (loaded, bags, normalized, _, _, evidence):
                        expected_rna = (torch.zeros(2, 3) if frozen == 'rna' else
                                        torch.tensor([[1., 2., 3.], [1.5, 2.5, 3.5]]))
                        torch.testing.assert_close(normalized, expected_rna, rtol=0, atol=0)
                        self.assertEqual([bag._source.n_patches for bag in bags], [3, 12])
                        if frozen == 'pathology':
                            self.assertTrue(all(torch.equal(rows(bag, 2), torch.zeros(8, 3)) for bag in bags))
                        result = diagnostic.predict_at_strength(loaded, bags, normalized, 2, .2)
                        self.assertTrue(torch.isfinite(result['log_rates']).all())
                        self.assertTrue(evidence['controls_replay']['input_contract_verified'])
                        self.assertEqual(evidence['controls_replay']['frozen_modality'], frozen)
                        self.assertEqual(evidence['input_verification'], 'saved_input_contracts_verified')
                        self.assertFalse(evidence['baseline_prediction_reproduction_checked'])
                    self.assertTrue(evidence['model_and_input_guards_passed'])

    def test_zero_pathology_preserves_source_counts_padding_and_provenance(self):
        sources, bags, rna = self.fixture()
        frozen, normalized, _ = apply_checkpoint_controls(contract('pathology'), bags, rna)
        self.assertIs(normalized, rna)
        self.assertEqual([b._source.n_patches for b in frozen], [3, 11])
        self.assertEqual([b.n_patches for b in frozen], [8, 8])
        self.assertEqual([b.padding for b in frozen], [5, 0])
        for bag, source in zip(frozen, sources):
            self.assertIs(bag._source, source)
            self.assertEqual(bag.provenance, source.provenance)
            changed = bag.provenance
            changed[0]['rows'] = -1
            self.assertEqual(source.provenance[0]['rows'], source.n_patches)
            for chunk in (1, 2, 3, 5, 13):
                self.assertTrue(torch.equal(rows(bag, chunk), torch.zeros(8, 3)))
            self.assertIs(bag.resample(99), bag)
            bag.assert_unchanged()
            bag.close()
            self.assertEqual(source.checked, 1)
            self.assertEqual(source.closed, 1)
            self.assertEqual(source.reads, 0)

    def test_rna_zero_control_does_not_modify_normalized_inputs(self):
        _, bags, rna = self.fixture()
        before = rna.clone()
        bag_rows = [rows(b, 3).clone() for b in bags]
        replay, zero_rna, _ = apply_checkpoint_controls(contract('rna'), bags, rna)
        self.assertIsNot(replay, bags)
        self.assertTrue(all(a is b for a, b in zip(replay, bags)))
        self.assertTrue(torch.equal(zero_rna, torch.zeros_like(rna)))
        zero_rna[0, 0] = 7.
        self.assertTrue(torch.equal(rna, before))
        for b, expected in zip(bags, bag_rows):
            self.assertTrue(torch.equal(rows(b, 3), expected))

    def test_none_control_and_historical_checkpoint_are_explicitly_distinguished(self):
        _, bags, rna = self.fixture()
        for checkpoint, explicit, verified in ((contract(), True, True), ({}, False, False),
                                               ({'controls': {}}, True, False)):
            replay, normalized, evidence = apply_checkpoint_controls(checkpoint, bags, rna)
            self.assertTrue(all(a is b for a, b in zip(replay, bags)))
            self.assertIs(normalized, rna)
            self.assertEqual(evidence['frozen_modality'], 'none')
            self.assertEqual(evidence['has_explicit_contract'], explicit)
            self.assertEqual(evidence['input_contract_verified'], verified)

    def test_incomplete_or_unknown_contracts_are_not_marked_verified(self):
        _, bags, rna = self.fixture()
        variants = []
        for key in ('num_patches', 'patch_chunk_size', 'loss_version'):
            checkpoint = contract()
            checkpoint.pop(key)
            variants.append(checkpoint)
        checkpoint = contract()
        checkpoint['controls']['input_control_version'] = 'unsupported_future_contract'
        variants.append(checkpoint)
        for checkpoint in variants:
            with self.subTest(checkpoint=checkpoint):
                _, _, evidence = apply_checkpoint_controls(checkpoint, bags, rna)
                self.assertFalse(evidence['input_contract_verified'])

    def test_bad_controls_budgets_and_rna_shape_are_refused(self):
        _, bags, rna = self.fixture()
        bad_contracts = [dict(contract(), controls=[]), dict(contract(), controls={'frozen_modality': 'other'}),
                         dict(contract(), num_patches=7), dict(contract(), num_patches=True),
                         dict(contract(), num_patches=0)]
        missing_budget = contract('pathology')
        missing_budget.pop('num_patches')
        bad_contracts.append(missing_budget)
        for checkpoint in bad_contracts:
            with self.subTest(checkpoint=checkpoint):
                with self.assertRaises(ValueError):
                    apply_checkpoint_controls(checkpoint, bags, rna)
        for invalid in (rna[0], rna[:1], rna.numpy()):
            with self.assertRaisesRegex(ValueError, 'RNA row'):
                apply_checkpoint_controls(contract(), bags, invalid)
        different = BudgetedPatientBag(bags[1]._source, 7, mode='eval')
        with self.assertRaisesRegex(ValueError, 'patch budget differs'):
            apply_checkpoint_controls(contract(), [bags[0], different], rna)

    def test_zero_bag_validates_chunk_and_reference_only_contract(self):
        _, bags, _ = self.fixture()
        zero = ZeroPatientBag(bags[0], 8)
        for chunk in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                list(zero.blocks(chunk))
        with self.assertRaisesRegex(ValueError, 'reference rows only'):
            zero.read_block(('<parity-budget>', 0, 2), ('features_a',))
        for token in (('wrong', 0, 2), ('<parity-budget>', -1, 2), ('<parity-budget>', 0, 9)):
            with self.assertRaisesRegex(ValueError, 'Invalid zero-bag block'):
                zero.read_block(token, ('features_reference',))


if __name__ == '__main__':
    unittest.main()
