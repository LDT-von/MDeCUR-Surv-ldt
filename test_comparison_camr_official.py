"""Contract tests for the CAMR upstream-survival-contract arm.

These are generated-data tests. They check the properties that make the arm
meaningful: the upstream bounded scalar head, a time-based Cox risk set, the
Breslow baseline projection and the arm's registration in the one-fold
comparison. They are not evidence of patient benefit.
"""
import unittest

import numpy as np
import torch

from comparison_camr_official import (CAMR2MOfficial, COX_PARTIAL_LIKELIHOOD_VERSION,
                                      cox_discrete_logits, cox_partial_likelihood,
                                      fit_brewslow_baseline)
from cancer_parity import discrete_survival


class CoxLikelihoodTest(unittest.TestCase):
    def test_risk_set_uses_times_not_row_positions(self):
        # Two patients, the first with the LONGER time.  If the risk set were
        # built from row indices, patient 0 (row 0) would be at risk for
        # patient 1 (row 1).  With a time-based risk set, patient 1 (t=5) is
        # at risk for patient 0 (t=10), but patient 0 is not at risk for
        # patient 1.
        risk = torch.tensor([1.0, 0.0], requires_grad=True)
        time = torch.tensor([10.0, 5.0])
        event = torch.tensor([1.0, 1.0])
        terms = cox_partial_likelihood(risk, time, event, reduction='none')
        # R[0] = {t>=10} = {0} -> log(e^1) = 1 -> contribution -(1 - 1) = 0
        # R[1] = {t>=5}  = {0,1} -> log(e^1+e^0) -> contribution log(1+e)
        self.assertAlmostEqual(float(terms[0]), 0.0, places=6)
        self.assertAlmostEqual(float(terms[1]), float(np.log(1. + np.exp(1.))), places=5)
        loss = cox_partial_likelihood(risk, time, event)
        self.assertAlmostEqual(float(loss),
                               float((terms[0] + terms[1])/2), places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(risk.grad).all())

    def test_shuffling_the_batch_keeps_the_loss_invariant(self):
        # A correct risk set depends on the times only, so permuting the
        # patients must not change the loss.  A row-index risk set would.
        generator = torch.Generator().manual_seed(7)
        risk = torch.randn(6, generator=generator, dtype=torch.float64)
        time = torch.tensor([3.0, 9.0, 1.0, 7.0, 4.0, 11.0], dtype=torch.float64)
        event = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        base = cox_partial_likelihood(risk, time, event)
        permutation = torch.tensor([4, 0, 5, 2, 1, 3])
        shuffled = cox_partial_likelihood(risk[permutation], time[permutation], event[permutation])
        self.assertAlmostEqual(float(base), float(shuffled), places=10)

    def test_a_censored_patient_contributes_no_term_of_its_own(self):
        generator = torch.Generator().manual_seed(11)
        risk = torch.randn(4, generator=generator, dtype=torch.float64)
        time = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        event = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
        terms = cox_partial_likelihood(risk, time, event, reduction='none')
        # The censored patient's own contribution is exactly zero.
        self.assertEqual(float(terms[3]), 0.)
        # Changing its EVENT flag must not move anybody else's term.  Its
        # risk value does enter other patients' risk sets, which is the
        # standard Cox behaviour and not a cross-patient leak.
        changed = event.clone()
        changed[3] = 1.
        other = cox_partial_likelihood(risk, time, changed, reduction='none')
        for index in range(3):
            self.assertEqual(float(terms[index]), float(other[index]))
        # Patient 3 has the longest follow-up, so its own risk set contains
        # only itself and its term stays zero whatever its event flag is.
        self.assertEqual(float(other[3]), 0.)

    def test_an_event_patient_with_a_later_risk_set_does_contribute(self):
        generator = torch.Generator().manual_seed(17)
        risk = torch.randn(4, generator=generator, dtype=torch.float64)
        time = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        event = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        terms = cox_partial_likelihood(risk, time, event, reduction='none')
        # Patient 2 is censored, so it contributes nothing.
        self.assertEqual(float(terms[2]), 0.)
        # Patient 0 (t=1) is at risk for everyone and contributes a real term.
        self.assertNotEqual(float(terms[0]), 0.)

    def test_one_patient_event_flag_does_not_move_another_patient_term(self):
        generator = torch.Generator().manual_seed(13)
        risk = torch.randn(5, generator=generator, dtype=torch.float64)
        time = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        event = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
        terms = cox_partial_likelihood(risk, time, event, reduction='none')
        changed = event.clone()
        changed[4] = 0.
        other = cox_partial_likelihood(risk, time, changed, reduction='none')
        # Patient 4's own term becomes zero; every other term is unchanged.
        self.assertEqual(float(other[4]), 0.)
        for index in range(4):
            self.assertEqual(float(terms[index]), float(other[index]))

    def test_invalid_inputs_are_rejected(self):
        risk = torch.zeros(3)
        with self.assertRaisesRegex(ValueError, 'one-dimensional'):
            cox_partial_likelihood(risk.reshape(3, 1), torch.zeros(3), torch.ones(3))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            cox_partial_likelihood(torch.tensor([float('nan'), 0., 0.]), torch.zeros(3), torch.ones(3))
        with self.assertRaisesRegex(ValueError, 'one value per patient'):
            cox_partial_likelihood(risk, torch.zeros(2), torch.ones(3))
        with self.assertRaisesRegex(ValueError, 'zero or one'):
            cox_partial_likelihood(risk, torch.zeros(3), torch.full((3,), 2.))
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            cox_partial_likelihood(risk, torch.tensor([-1., 0., 0.]), torch.ones(3))


class BreslowBaselineTest(unittest.TestCase):
    def test_baseline_is_fitted_from_training_outcomes_only(self):
        time = np.array([1., 2., 3., 4., 5.])
        event = np.array([1., 0., 1., 1., 0.])
        risk = np.array([0.2, -0.1, 0.5, 0.3, -0.4])
        levels = fit_brewslow_baseline(time, event, risk, np.array([2., 4., 5.]))
        self.assertEqual(levels.shape, (3,))
        self.assertTrue(np.all(np.diff(levels) >= -1e-12))
        self.assertTrue(np.all(levels >= 0))
        # H0 is nondecreasing and the first value must be positive because an
        # event occurred at time 1.
        self.assertGreater(levels[0], 0.)

    def test_no_event_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'no observed event'):
            fit_brewslow_baseline(np.array([1., 2.]), np.array([0., 0.]),
                                   np.array([0., 0.]), np.array([1.5]))

    def test_nonincreasing_column_times_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            fit_brewslow_baseline(np.array([1., 2., 3.]), np.array([1., 1., 1.]),
                                  np.array([0., 0., 0.]), np.array([2., 1.]))

    def test_projection_is_monotone_in_the_risk(self):
        levels = np.array([0.2, 0.5, 0.9, 1.4])
        risk = torch.tensor([-1.0, 0.0, 1.0])
        logits = cox_discrete_logits(risk, levels)
        self.assertEqual(logits.shape, (3, 4))
        survival = discrete_survival(logits)
        # A larger risk must give a smaller survival and a larger risk score.
        self.assertTrue(bool((survival[0] > survival[1]).all()))
        self.assertTrue(bool((survival[1] > survival[2]).all()))
        risk_score = -survival.sum(dim=1)
        self.assertTrue(bool((risk_score[0] < risk_score[1]) and (risk_score[1] < risk_score[2])))
        # Cumulative survival must be nonincreasing across the bins.
        self.assertTrue(bool((survival[:, 1:] <= survival[:, :-1] + 1e-9).all()))

    def test_projection_rejects_a_decreasing_baseline(self):
        with self.assertRaisesRegex(ValueError, 'nondecreasing'):
            cox_discrete_logits(torch.zeros(2), np.array([0.5, 0.2]))


class OfficialArmTest(unittest.TestCase):
    def test_head_is_the_upstream_bounded_scalar(self):
        model = CAMR2MOfficial(path_dim=6, rna_dim=8, bins=4)
        classifier = model.upstream_model.classifier
        self.assertEqual(len(classifier), 2)
        self.assertIsInstance(classifier[1], torch.nn.Sigmoid)
        self.assertEqual(classifier[0].out_features, 1)
        self.assertAlmostEqual(float(model.upstream_model.output_range), 8.0)
        self.assertAlmostEqual(float(model.upstream_model.output_shift), -4.0)

    def test_forward_emits_one_bounded_risk_per_patient(self):
        model = CAMR2MOfficial(path_dim=6, rna_dim=8, bins=4)
        generator = torch.Generator().manual_seed(3)
        patches = torch.randn(4096, 6, generator=generator)
        rna = torch.randn(1, 8, generator=generator)
        output = model([patches], rna)
        self.assertEqual(output['risk'].shape, (1,))
        self.assertTrue(bool((output['risk'] > -4).all()) and bool((output['risk'] < 4).all()))
        # The logits of the base adaptation are the 4-bin hazards; this arm
        # replaces them with the scalar Cox output.
        self.assertEqual(output['logits'].shape, (1, 1))

    def test_metadata_declares_the_contract_and_the_deviations(self):
        model = CAMR2MOfficial(path_dim=6, rna_dim=8, bins=4)
        metadata = model.adaptation_metadata()
        self.assertEqual(metadata['survival_loss_version'], COX_PARTIAL_LIKELIHOOD_VERSION)
        self.assertFalse(metadata['upstream_risk_set_helper_reused'])
        self.assertIn('sigmoid scalar', metadata['survival_head'])
        self.assertFalse(metadata['official_model_reproduction'])
        self.assertTrue(metadata['limits'])

    def test_the_arm_is_registered_in_the_one_fold_comparison(self):
        import comparison_fold0 as compare
        self.assertIn('camr_2m_official', compare.MODELS)
        self.assertIn('camr_2m_official', compare.CAMR_OFFICIAL_MODELS)
        self.assertIn('comparison_camr_official', compare.SOURCE_MODULES)
        # The two CAMR arms differ only in the head and in the survival loss.
        self.assertEqual(compare.CAMR_MODELS, ('camr_2m', 'camr_2m_official'))


if __name__ == '__main__':
    unittest.main()
