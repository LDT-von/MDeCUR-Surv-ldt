"""Independent per-patient likelihood/gradient regression checks, no reference clone."""
import math
import unittest
import torch

from cancer_parity import discrete_nll


def manual_loop(logits, bins, censored, alpha=.5, eps=1e-7):
    values = []
    for row, label, censor in zip(logits, bins, censored):
        hazards = row.sigmoid()
        previous = hazards.new_ones(())
        for k in range(int(label)):
            previous = previous*(1-hazards[k])
        if int(censor):
            term = -(1-alpha)*torch.log((previous*(1-hazards[int(label)])).clamp_min(eps))
        else:
            term = -torch.log(previous.clamp_min(eps))-torch.log(hazards[int(label)].clamp_min(eps))
        values.append(term)
    return torch.stack(values)


class PatientwiseDiscreteNLLTest(unittest.TestCase):
    def test_mixed_censoring_matches_hand_calculation_and_shape(self):
        logits = torch.zeros(2, 2, dtype=torch.float64)
        loss = discrete_nll(logits, [0, 1], [0, 1], alpha=.5, reduction='none')
        self.assertEqual(tuple(loss.shape), (2,))
        torch.testing.assert_close(loss, torch.tensor([math.log(2), math.log(2)], dtype=torch.float64))

    def test_patient_event_and_censor_gradients_have_opposite_correct_signs(self):
        logits = torch.tensor([[1., -1.], [1., -1.]], dtype=torch.float64, requires_grad=True)
        discrete_nll(logits, [0, 0], [0, 1], alpha=0).backward()
        h = torch.sigmoid(torch.tensor(1., dtype=torch.float64))
        torch.testing.assert_close(logits.grad[:, 0], torch.stack(((h-1)/2, h/2)))
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertGreater(float(logits.grad[1, 0]), 0)

    def test_changing_one_censor_label_cannot_change_another_patient_gradient(self):
        x = torch.tensor([[2., -1.], [-2., 1.]], dtype=torch.float64, requires_grad=True)
        g1 = torch.autograd.grad(discrete_nll(x, [0, 1], [0, 1], reduction='sum'), x)[0]
        g2 = torch.autograd.grad(discrete_nll(x, [0, 1], [0, 0], reduction='sum'), x)[0]
        torch.testing.assert_close(g1[0], g2[0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(g1[1], g2[1]))

    def test_single_patient_loop_matches_values_and_all_gradients(self):
        torch.manual_seed(61)
        x = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
        bins = torch.tensor([0, 1, 2, 3, 1, 0, 3, 2, 1])
        c = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1, 0])
        for alpha in (0., .5, 1.):
            actual = discrete_nll(x, bins, c, alpha=alpha, reduction='none')
            expected = manual_loop(x, bins, c, alpha=alpha)
            torch.testing.assert_close(actual, expected)
            ga = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
            ge = torch.autograd.grad(expected.sum(), x)[0]
            torch.testing.assert_close(ga, ge)

    def test_partition_permutation_and_vector_column_invariance(self):
        torch.manual_seed(82)
        x = torch.randn(7, 4, dtype=torch.float64)
        y, c = torch.arange(7) % 4, torch.arange(7) % 2
        whole = discrete_nll(x, y, c, reduction='none')
        singles = torch.cat([discrete_nll(x[i:i+1], y[i:i+1], c[i:i+1], reduction='none') for i in range(7)])
        torch.testing.assert_close(whole, singles)
        torch.testing.assert_close(whole, discrete_nll(x, y[:, None], c, reduction='none'))
        torch.testing.assert_close(whole, discrete_nll(x, y, c[:, None], reduction='none'))
        perm = torch.tensor([4, 1, 6, 0, 3, 5, 2])
        torch.testing.assert_close(whole[perm], discrete_nll(x[perm], y[perm], c[perm], reduction='none'))
        self.assertAlmostEqual(float(whole.mean()), float(discrete_nll(x, y, c)), places=13)
        self.assertAlmostEqual(float(whole.sum()), float(discrete_nll(x, y, c, reduction='sum')), places=13)

    def test_invalid_labels_censoring_and_parameters_are_rejected(self):
        x = torch.zeros(2, 4)
        for y, c in (([.5, 1], [0, 1]), ([0, 4], [0, 1]), ([-1, 1], [0, 1]),
                     ([0, 1], [0, .5]), ([0, 1], [0, float('nan')])):
            with self.assertRaises(ValueError):
                discrete_nll(x, y, c)
        for kwargs in ({'alpha': -1}, {'alpha': 2}, {'eps': 0}, {'eps': float('nan')}):
            with self.assertRaises(ValueError):
                discrete_nll(x, [0, 1], [0, 1], **kwargs)


if __name__ == '__main__':
    unittest.main()
