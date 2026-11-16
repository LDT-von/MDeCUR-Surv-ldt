"""Generated-data regressions through the actual parity training entry point."""
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cancer_dense_model import DENSE_LOSS_NAMES, DensePathwaySurvival, dense_weights
from cancer_parity import DISCRETE_NLL_VERSION, discrete_survival
from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import LOSS_NAMES, effective_weights, patient_batches
from cancer_rna import transform_rna
from comparison_synthetic import write_synthetic_cohort
from parity_checkpoint_controls import apply_checkpoint_controls
import train_pathway_parity as runner


class ParityEntrypointControlsTest(unittest.TestCase):
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
        self.data = write_synthetic_cohort(self.root/'inputs', patients=28, feature_dim=6,
                                            rna_dim=30, pathway_groups=10, seed=81)
        # Enough real tokens for the dense within objectives, with real
        # patient variation. The budget deliberately includes padding.
        generator = torch.Generator().manual_seed(82)
        for i, path in enumerate(sorted((self.data['root']/'pt_files').glob('*.pt'))):
            torch.save(torch.randn(19+i % 2, 6, generator=generator)
                       + 3*torch.randn(1, 6, generator=generator), path)
        self.base = ['--manifest', str(self.data['manifest']), '--survival-reference',
                     str(self.data['survival_reference']), '--genes', str(self.data['genes']),
                     '--pathways', str(self.data['gmt']), '--out', str(self.root/'run'),
                     '--num-patches', '23', '--patch-chunk-size', '5', '--epochs', '1',
                     '--endpoint', 'SYNTHETIC', '--time-unit', 'months', '--split-name', 'generated']

    def configure(self, *extra):
        return runner._configure(runner.make_parser().parse_args([*self.base, *extra]))

    def run_training(self, name, *extra):
        output = self.root/name
        arguments = [*self.base, '--out', str(output), *extra]
        with patch.object(sys, 'argv', ['train_pathway_parity.py', *arguments]), \
                patch.object(runner, '_configure', wraps=runner._configure) as configure, \
                redirect_stdout(io.StringIO()):
            runner.main()
        configure.assert_called_once()
        report = json.loads((output/'report.json').read_text(encoding='utf-8'))
        checkpoint = torch.load(output/'checkpoint_best.pt', map_location='cpu', weights_only=True)
        return output, report, checkpoint

    def test_final_configuration_preserves_default_full_and_marks_explicit_changes(self):
        cfg, controls, minimum, tail = self.configure()
        self.assertTrue(controls['is_full_reference'])
        self.assertEqual(len({**effective_weights(cfg), **dense_weights(cfg)}), 15)
        self.assertTrue(all(value > 0 for value in {**effective_weights(cfg), **dense_weights(cfg)}.values()))
        self.assertEqual((minimum, tail), (17, 17))
        cfg, controls, _, _ = self.configure('--common-mode', 'off')
        self.assertFalse(controls['is_full_reference'])
        self.assertEqual(effective_weights(cfg)['global_common'], 0.)
        for name in (*LOSS_NAMES, *DENSE_LOSS_NAMES):
            with self.subTest(term=name):
                cfg, controls, _, _ = self.configure('--auxiliary-weight', f'{name}=0')
                self.assertFalse(controls['is_full_reference'])
                self.assertEqual({**effective_weights(cfg), **dense_weights(cfg)}[name], 0.)

    def test_invalid_controls_fail_before_opening_any_data(self):
        cases = [(['--freeze-modality', 'rna'], 'require --disable-auxiliary'),
                 (['--legacy-architecture', '--auxiliary-weight', 'dense_common=0'], 'Unknown auxiliary'),
                 (['--disable-auxiliary', '--common-mode', 'uniform'], 'cannot enable'),
                 (['--disable-auxiliary', '--auxiliary-weight', 'global_common=0'], 'contradict'),
                 (['--batch-tail-minimum', '1'], 'batch-tail-minimum')]
        for extra, message in cases:
            with self.subTest(arguments=extra), \
                    patch.object(sys, 'argv', ['runner', *self.base, *extra]), \
                    patch.object(runner, 'load_stream_manifest') as load:
                with self.assertRaisesRegex(ValueError, message):
                    runner.main()
                load.assert_not_called()

    def test_shared_tail_minimum_controls_aux_off_schedule(self):
        _, _, _, full_minimum = self.configure('--legacy-architecture', '--patient-batch-size', '17')
        _, _, active_minimum, tail = self.configure('--legacy-architecture', '--disable-auxiliary',
                                                   '--patient-batch-size', '17', '--batch-tail-minimum', '17')
        self.assertEqual(active_minimum, 1)
        self.assertEqual(tail, full_minimum)
        order = torch.arange(303)
        schedules = [patient_batches(order, 17, bound) for bound in (tail, full_minimum)]
        self.assertEqual([[len(batch) for batch in schedule] for schedule in schedules],
                         [[17]*16+[31]]*2)

    def test_actual_default_training_saves_full_control_contract(self):
        _, report, checkpoint = self.run_training('full')
        self.assertTrue(report['controls']['is_full_reference'])
        self.assertEqual(report['controls'], checkpoint['controls'])
        self.assertTrue(report['input_replay']['input_contract_verified'])
        self.assertEqual(report['loss_version'], DISCRETE_NLL_VERSION)
        self.assertEqual(checkpoint['loss_version'], DISCRETE_NLL_VERSION)
        self.assertEqual(checkpoint['num_patches'], 23)
        self.assertEqual(checkpoint['patch_chunk_size'], 5)
        self.assertEqual(report['history'][0]['patient_batches'], [21])
        self.assertEqual(report['optimizer_steps'], 1)
        self.assertEqual(len(report['training_sampling_sha256']), 64)
        self.assertEqual(set(report['history'][0]['auxiliary_terms']), {*LOSS_NAMES, *DENSE_LOSS_NAMES})
        self.assertTrue(all(value > 0 for value in report['effective_loss_weights'].values()))

    def test_actual_explicit_ablation_is_applied_before_final_guard(self):
        _, report, checkpoint = self.run_training('explicit', '--common-mode', 'off',
                                                  '--auxiliary-weight', 'dense_unique=0')
        self.assertFalse(report['controls']['is_full_reference'])
        self.assertEqual(report['controls']['role'], 'explicit_auxiliary_ablation_or_tuning')
        self.assertEqual(checkpoint['config']['common_mode'], 'off')
        weights = report['effective_loss_weights']
        self.assertEqual({name for name, value in weights.items() if value == 0},
                         {'global_common', 'dense_unique'})

    def test_actual_frozen_training_reloads_predictions_and_source_counts(self):
        for frozen, legacy in (('rna', False), ('pathology', False), ('pathology', True)):
            with self.subTest(frozen=frozen, legacy=legacy):
                options = ['--disable-auxiliary', '--freeze-modality', frozen,
                           '--batch-tail-minimum', '17']
                if legacy:
                    options.append('--legacy-architecture')
                output, report, checkpoint = self.run_training(f'{frozen}_{legacy}', *options)
                self.assertEqual(report['controls']['role'], 'input_zero_control')
                self.assertFalse(report['controls']['is_full_reference'])
                self.assertEqual(report['controls'], checkpoint['controls'])
                self.assertTrue(report['input_replay']['input_contract_verified'])
                self.assertTrue(all(value == 0 for value in report['effective_loss_weights'].values()))
                self.assertEqual(report['history'][0]['patient_batches'], [21])
                cohort = runner.load_stream_manifest(self.data['manifest'], 'single_view', include_test=False)
                try:
                    validation = runner._budget(cohort['val'], 23, 'eval', 3)
                    rna = transform_rna(validation['rna'], checkpoint)
                    validation['bags'], rna, metadata = apply_checkpoint_controls(
                        checkpoint, validation['bags'], rna)
                    self.assertTrue(metadata['input_contract_verified'])
                    cls = PathwayLocalSurvival if legacy else DensePathwaySurvival
                    model = cls(**checkpoint['model_kwargs'])
                    model.load_state_dict(checkpoint['model_state'], strict=True)
                    predictions = discrete_survival(runner._predict(model, validation, rna, 5)).numpy()
                    with (output/'val_predictions_best.csv').open(newline='', encoding='utf-8') as handle:
                        saved = list(csv.DictReader(handle))
                    expected = np.array([[float(row[f'survival_bin_{k}']) for k in range(4)] for row in saved])
                    np.testing.assert_allclose(predictions, expected, rtol=0, atol=1e-7)
                    available = sum(bag._source.n_patches for bag in validation['bags'])
                    self.assertEqual(report['patches_available']['val'], available)
                    self.assertLess(available, 23*len(validation['bags']))
                finally:
                    for split in cohort.values():
                        for bag in split['bags']:
                            bag.close()


if __name__ == '__main__':
    unittest.main()
