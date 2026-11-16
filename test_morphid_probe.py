"""Known-answer and held-out-patient contracts for the frozen RNA probe."""
import io
import math
import unittest

import torch

from morphid_probe import (CrossFittedRBFProbe, FrozenRBFProbe, patient_fold_ids,
                           r2_score)


class FrozenMorphIDProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_two_patient_closed_form_answer_and_effective_df(self):
        x = torch.tensor([[0.], [1.]], dtype=torch.float64)
        y = torch.tensor([1., 3.], dtype=torch.float64)
        alpha = .2
        model = FrozenRBFProbe(alpha=alpha, bandwidth=1.).fit(x, y)
        off = math.exp(-2.)  # Training-standardized inputs are -1 and +1.
        delta = (1-off)/(1+2*alpha-off)
        torch.testing.assert_close(model.predict(x), torch.tensor([2-delta, 2+delta], dtype=torch.float64))
        expected_df = sum(eigen/(eigen+2*alpha) for eigen in (1-off, 1+off))
        self.assertAlmostEqual(model.metadata['effective_degrees_of_freedom'], expected_df, places=12)
        self.assertEqual(model.metadata['input_rank'], 1)
        self.assertFalse(model.metadata['independence_established'])

    def test_nonlinear_signal_is_detected_without_linear_covariance(self):
        x = torch.linspace(-2., 2., 128, dtype=torch.float64)[:, None]
        y = x.square()-x.square().mean()
        # The grid is symmetric, so the optimal centered linear slope is zero.
        linear_slope = torch.linalg.lstsq(x, y).solution
        self.assertLess(float(linear_slope.abs().max()), 1e-12)
        model = CrossFittedRBFProbe(alpha=1e-3).fit(x, y, [f's{i}' for i in range(len(x))])
        self.assertGreater(r2_score(y, model.oof_predictions)['uniform_average'], .9)
        fresh = torch.linspace(-1.99, 1.99, 223, dtype=torch.float64)[:, None]
        truth = fresh.square()-x.square().mean()
        self.assertGreater(r2_score(truth, model.predict(fresh))['uniform_average'], .95)

    def test_prediction_residual_retains_independent_overlapping_signal(self):
        generator = torch.Generator().manual_seed(94)
        x = 4*torch.rand(256, 1, generator=generator, dtype=torch.float64)-2
        epsilon = .8*torch.randn(256, 1, generator=generator, dtype=torch.float64)
        y = x+epsilon
        model = CrossFittedRBFProbe(alpha=.01).fit(x, y, [f'o{i}' for i in range(len(x))])
        residual = y-model.oof_predictions
        correlation = torch.corrcoef(torch.stack((epsilon[:, 0], residual[:, 0])))[0, 1]
        self.assertGreater(float(correlation), .85)
        self.assertGreater(float(residual.var()), .3)
        # A feature-span projection in this one-dimensional case gives zero.
        self.assertEqual(float(((1-torch.ones(1))*y).abs().sum()), 0.)

    def test_own_target_and_held_out_preprocessing_do_not_leak(self):
        generator = torch.Generator().manual_seed(25)
        x = torch.randn(40, 5, generator=generator, dtype=torch.float64)
        y = torch.stack((x[:, 0].square(), x[:, 1]+x[:, 2]), 1)
        ids = [f'p{i}' for i in range(len(x))]
        first = CrossFittedRBFProbe(pca_dim=3).fit(x, y, ids)
        fold = int(first.fold_ids[0])
        held_out = first.fold_ids == fold
        changed_y = y.clone()
        changed_y[held_out] += 1000.
        second = CrossFittedRBFProbe(pca_dim=3).fit(x, changed_y, ids)
        self.assertTrue(torch.equal(first.oof_predictions[held_out], second.oof_predictions[held_out]))
        changed_x = x.clone()
        changed_x[held_out] *= 100.
        third = CrossFittedRBFProbe(pca_dim=3).fit(changed_x, y, ids)
        for name in ('x_mean', 'x_scale', 'pca_components', 'train_x', 'bandwidth_value', 'target_mean'):
            self.assertTrue(torch.equal(getattr(first.models[fold], name), getattr(third.models[fold], name)), name)
        for entry in first.split_audit:
            self.assertFalse(set(entry['train_ids']) & set(entry['validation_ids']))

    def test_patient_folds_are_balanced_identity_stable_and_rng_neutral(self):
        ids = [f'病人{i}' for i in range(19)]
        state = torch.random.get_rng_state().clone()
        assignments = patient_fold_ids(ids, folds=4, seed=17)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        counts = torch.bincount(assignments, minlength=4)
        self.assertLessEqual(int(counts.max()-counts.min()), 1)
        permutation = list(reversed(range(len(ids))))
        other = patient_fold_ids([ids[i] for i in permutation], folds=4, seed=17)
        self.assertEqual({identity: int(fold) for identity, fold in zip(ids, assignments)},
                         {ids[i]: int(fold) for i, fold in zip(permutation, other)})

    def test_serialized_buffer_only_state_restores_fresh_modules(self):
        generator = torch.Generator().manual_seed(43)
        x = torch.randn(24, 6, generator=generator, dtype=torch.float64)
        y = torch.randn(24, 2, 3, generator=generator, dtype=torch.float64)
        models = [FrozenRBFProbe(alpha=.2, pca_dim=3).fit(x, y),
                  CrossFittedRBFProbe(alpha=.2, pca_dim=3).fit(x, y, [f'z{i}' for i in range(len(x))])]
        for model in models:
            with self.subTest(kind=type(model).__name__):
                self.assertEqual(list(model.parameters()), [])
                stream = io.BytesIO()
                torch.save(model.state_dict(), stream)
                stream.seek(0)
                restored = type(model)()
                restored.load_state_dict(torch.load(stream, weights_only=True))
                self.assertTrue(torch.equal(model.predict(x[:3]), restored.predict(x[:3])))
                self.assertEqual(restored.predict(x[:3]).shape, (3, 2, 3))
                self.assertEqual(model.metadata, restored.metadata)
                if isinstance(model, CrossFittedRBFProbe):
                    self.assertTrue(torch.equal(model.oof_predictions, restored.oof_predictions))
                    self.assertEqual(model.split_audit, restored.split_audit)

    def test_repeated_fits_are_deterministic_and_do_not_consume_rng(self):
        x = torch.arange(90, dtype=torch.float64).reshape(30, 3)
        y = torch.sin(x[:, 0])
        ids = [str(i) for i in range(len(x))]
        state = torch.random.get_rng_state().clone()
        first = CrossFittedRBFProbe(pca_dim=2).fit(x, y, ids)
        second = CrossFittedRBFProbe(pca_dim=2).fit(x, y, ids)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first.oof_predictions, second.oof_predictions))
        self.assertTrue(torch.equal(first.predict(x), second.predict(x)))

    def test_negative_and_undefined_r2_are_not_hidden(self):
        y = torch.tensor([[0., 1.], [1., 1.], [2., 1.]], dtype=torch.float64)
        result = r2_score(y, torch.full_like(y, 10.))
        self.assertLess(result['per_output'][0], 0.)
        self.assertIsNone(result['per_output'][1])
        self.assertEqual(result['valid_outputs'], [True, False])
        self.assertIsNone(r2_score(torch.ones(4), torch.ones(4))['uniform_average'])
        tiny = 1e-10*torch.arange(5, dtype=torch.float64)
        self.assertEqual(r2_score(tiny, tiny)['uniform_average'], 1.)
        supplied = r2_score(torch.tensor([1., 2.]), torch.tensor([0., 0.]), reference_mean=0.)
        self.assertEqual(supplied['uniform_average'], 0.)

    def test_regularized_solution_does_not_impose_n_greater_than_output_parameter_count(self):
        generator = torch.Generator().manual_seed(41)
        x = torch.randn(8, 12, generator=generator, dtype=torch.float64)
        y = torch.randn(8, 30, generator=generator, dtype=torch.float64)
        model = FrozenRBFProbe().fit(x, y)
        self.assertTrue(torch.isfinite(model.predict(x)).all())
        self.assertLessEqual(model.metadata['effective_degrees_of_freedom'], 8.)

    def test_explicit_folds_and_invalid_inputs_are_checked(self):
        x = torch.arange(8, dtype=torch.float64)[:, None]
        y, ids = x.square(), [f'a{i}' for i in range(8)]
        folds = torch.arange(8) % 2
        model = CrossFittedRBFProbe().fit(x, y, ids, folds=2, fold_ids=folds)
        self.assertTrue(torch.equal(model.fold_ids, folds))
        with self.assertRaisesRegex(RuntimeError, 'refitted'):
            model.fit(x, y, ids)
        with self.assertRaisesRegex(ValueError, 'unique'):
            CrossFittedRBFProbe().fit(x, y, ['same']*8)
        with self.assertRaisesRegex(ValueError, 'fold_ids'):
            CrossFittedRBFProbe().fit(x, y, ids, fold_ids=torch.zeros(8))
        with self.assertRaisesRegex(ValueError, 'two training'):
            CrossFittedRBFProbe().fit(x[:2], y[:2], ids[:2], folds=2)
        with self.assertRaisesRegex(ValueError, 'finite'):
            FrozenRBFProbe().fit(torch.full((4, 2), float('nan')), torch.ones(4))
        with self.assertRaises(TypeError):
            FrozenRBFProbe().fit(x, y, survival_labels=torch.ones(8))
        with self.assertRaisesRegex(RuntimeError, 'not been fitted'):
            FrozenRBFProbe().predict(x)
        for config in ({'alpha': 0}, {'bandwidth': -1}, {'pca_dim': 0}, {'min_scale': float('nan')}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                FrozenRBFProbe(**config)


if __name__ == '__main__':
    unittest.main()
