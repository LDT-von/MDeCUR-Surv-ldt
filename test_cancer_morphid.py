"""Independent generated-data contracts for the frozen MorphID pipeline."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from cancer_morphid_model import (FrozenMorphologyBank, MorphIDSurvival,
                                  _MorphIDHead, composition_clr)
from morphid_probe import CrossFittedRBFProbe


def generated_bags(n, seed):
    generator = torch.Generator().manual_seed(seed)
    rna = torch.randn(n, 2, generator=generator, dtype=torch.float64)
    noise = torch.randn(n, generator=generator, dtype=torch.float64)
    bags = []
    for i in range(n):
        first = torch.cat((torch.full((12, 1), -4.), torch.full((12, 1), 4.)))
        rest = torch.randn(24, 3, generator=generator, dtype=torch.float64)*.12
        rest[:, 0] += .5*(rna[i, 0].square()-1)+noise[i]
        rest[:, 1] += rna[i, 1]
        rest[:, 2] += .4*noise[i]
        bags.append(torch.cat((first.double(), rest), 1))
    return bags, rna, noise


class FrozenMorphIDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        # Independent, unlabelled morphology reference: not the probe/head
        # patients, and not the fresh evaluation patients.
        reference, _, _ = generated_bags(20, 101)
        self.bank = FrozenMorphologyBank.fit(
            reference, ids=[f'reference{i}' for i in range(20)], clusters=2,
            latent_dim=4, max_fit_patches=240, per_patient_cap=12, seed=2)
        self.bags, self.rna, self.noise = generated_bags(40, 202)
        self.ids = [f'train{i}' for i in range(40)]

    def fitted(self, **kwargs):
        model = MorphIDSurvival(copy.deepcopy(self.bank), 2, interaction_rank=3, **kwargs)
        return model.fit_frozen(self.bags, self.rna, ids=self.ids, folds=4, seed=7)

    def test_padding_mask_chunk_and_patch_replication_invariants(self):
        bag = self.bags[0]
        padded = torch.cat((bag, torch.full((17, 4), float('nan'))))
        mask = torch.arange(len(padded)) < len(bag)
        first = self.bank.encode([bag], chunk_size=5)
        for candidate, chunk in (((padded, mask), 7), (bag.flip(0), 11), (bag.repeat(2, 1), 9)):
            result = self.bank.encode([candidate], chunk_size=chunk)
            for key in ('pooled', 'composition', 'clr'):
                torch.testing.assert_close(result[key], first[key], rtol=1e-10, atol=1e-10)
        self.assertEqual(int(self.bank.encode([(padded, mask)])['real_patches'][0]), 24)
        with self.assertRaises(ValueError):
            self.bank.encode([(padded, torch.zeros(len(padded), dtype=torch.bool))])

    def test_clr_zero_replacement_preserves_nonzero_ratios_and_closure(self):
        pi = torch.tensor([[0., .2, .8]], dtype=torch.float64)
        clr = composition_clr(pi)
        self.assertAlmostEqual(float(clr.sum()), 0., places=12)
        self.assertAlmostEqual(float(clr[0, 2]-clr[0, 1]), float(torch.log(torch.tensor(4.)).double()), places=6)
        self.assertTrue(torch.isfinite(clr).all())

    def test_no_survival_label_fit_api_and_explicit_out_of_fold_provenance(self):
        model = self.fitted()
        self.assertFalse(model.audit['decomposition_fit_uses_survival_labels'])
        self.assertFalse(model.audit['r2_used_as_gate'])
        self.assertFalse(model.audit['independence_established'])
        self.assertEqual(model.audit['morphology_basis']['fit_patient_ids'], [f'reference{i}' for i in range(20)])
        for probe in model.probes.values():
            for split in probe.split_audit:
                self.assertFalse(set(split['train_ids']) & set(split['validation_ids']))
        with self.assertRaises(TypeError):
            model.fit_frozen(self.bags, self.rna, ids=self.ids, survival_labels=torch.zeros(40))
        with self.assertRaises(RuntimeError):
            model.fit_frozen(self.bags, self.rna, ids=self.ids)
        with self.assertRaises(ValueError):
            model(self.bags, self.rna, ids=self.ids)

    def test_only_head_trains_and_frozen_state_and_replay_are_exact(self):
        model = self.fitted()
        before = model.frozen_digest()
        old_head = {n: p.detach().clone() for n, p in model.head.named_parameters()}
        labels = torch.arange(40) % 4
        censorship = (torch.arange(40) % 3 == 0).double()
        result = model.fit_survival_head(labels, censorship, ids=self.ids, epochs=8,
                                         lr=.01, batch_size=13)
        self.assertTrue(result['frozen_state_unchanged'])
        self.assertEqual(before, model.frozen_digest())
        self.assertTrue(any(not torch.equal(old_head[n], p) for n, p in model.head.named_parameters()))
        self.assertTrue(all(name.startswith('head.') for name, _ in model.named_parameters()))
        self.assertEqual([r['optimizer_steps'] for r in result['history']], [4]*8)
        self.assertLess(result['history'][-1]['training_nll'], result['initial_training_nll'])
        fresh, rna, _ = generated_bags(7, 303)
        ids = [f'fresh{i}' for i in range(7)]
        expected = model(fresh, rna, ids=ids, chunk_size=5)
        buffer = io.BytesIO()
        torch.save(model.checkpoint(), buffer)
        buffer.seek(0)
        loaded = MorphIDSurvival.from_checkpoint(torch.load(buffer, weights_only=True))
        replay = loaded(fresh, rna, ids=ids, chunk_size=5)
        self.assertTrue(torch.equal(expected['logits'], replay['logits']))
        self.assertTrue(torch.equal(expected['survival'], replay['survival']))
        self.assertEqual(loaded.frozen_digest(), model.frozen_digest())
        with self.assertRaises(RuntimeError):
            loaded.training_oof_logits()
        self.assertTrue(torch.all(expected['survival'][:, 1:] <= expected['survival'][:, :-1]))
        model.train()
        self.assertEqual(before, model.frozen_digest())
        with self.assertRaises(RuntimeError):
            model(fresh, rna, ids=ids)

    def test_absent_and_zero_residual_cannot_create_product_signal(self):
        model = self.fitted()
        model.residual_mean.fill_(3.)
        blocks = {'rna': torch.ones(2, 2, dtype=torch.float64),
                  'predictable': torch.zeros(2, 8, dtype=torch.float64),
                  'composition': torch.zeros(2, 2, dtype=torch.float64),
                  'residual': torch.zeros(2, 8, dtype=torch.float64)}
        available = torch.tensor([[False, True], [True, False]])
        features = model._normalize(blocks, available)
        self.assertTrue(torch.equal(features['interaction_residual'], torch.zeros(2, 2, 4, dtype=torch.float64)))
        self.assertTrue(torch.equal(features['residual'][~available], torch.zeros(2, 4, dtype=torch.float64)))
        _, product = model.head(features)
        self.assertTrue(torch.equal(product, torch.zeros(2, 4, dtype=torch.float64)))

    def test_predictable_rna_basis_can_retain_shared_nonlinear_risk(self):
        head = _MorphIDHead(1, 1, 1, 2, 2, True).double()
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            head.predictable.weight[:, 0] = torch.tensor([1., -1.])
        h = torch.tensor([[-2.], [-1.], [0.], [1.], [2.]], dtype=torch.float64)
        features = {'rna': h, 'predictable': h.square(), 'composition': torch.zeros_like(h),
                    'residual': torch.zeros(5, 1, 1, dtype=torch.float64),
                    'interaction_residual': torch.zeros(5, 1, 1, dtype=torch.float64)}
        logits, interaction = head(features)
        torch.testing.assert_close(logits, torch.cat((h.square(), -h.square()), 1), rtol=0, atol=0)
        self.assertTrue(torch.equal(interaction, torch.zeros_like(logits)))

    def test_unavailable_clusters_are_not_fitted_as_zero_targets(self):
        bank = FrozenMorphologyBank(torch.tensor([[-4., 0., 0., 0.], [50., 0., 0., 0.]]), min_mass=.01)
        bags = [bag[:12].clone() for bag in self.bags]
        model = MorphIDSurvival(bank, 2).fit_frozen(bags, self.rna, ids=self.ids)
        self.assertEqual(model.probe_available.tolist(), [True, False])
        self.assertEqual(model.audit['probe_clusters'][1]['reason'], 'insufficient_valid_patients')
        self.assertEqual(model.audit['probe_clusters'][1]['valid_patients'], 0)
        self.assertTrue(torch.equal(model._training_features['interaction_residual'][:, 1], torch.zeros(40, 4, dtype=torch.float64)))

    def test_bad_identity_and_frozen_representation_contracts(self):
        model = MorphIDSurvival(copy.deepcopy(self.bank), 2)
        for bad_ids in ([None]*40, ['same']*40, 'x'*40, [' '+value for value in self.ids]):
            with self.assertRaises(ValueError):
                model.fit_frozen(self.bags, self.rna, ids=bad_ids)
        with self.assertRaises(ValueError):
            model.fit_frozen(self.bags, self.rna.clone().requires_grad_(), ids=self.ids)
        good = self.fitted()
        with self.assertRaises(ValueError):
            good.fit_survival_head(torch.zeros(40), torch.ones(40), ids=list(reversed(self.ids)))

    def test_failed_frozen_fit_cannot_reuse_partial_cluster_state(self):
        model = MorphIDSurvival(copy.deepcopy(self.bank), 2)
        original = CrossFittedRBFProbe.fit
        calls = []
        def fail_second(probe, *args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise FloatingPointError('deliberate second-cluster failure')
            return original(probe, *args, **kwargs)
        with patch.object(CrossFittedRBFProbe, 'fit', fail_second):
            with self.assertRaisesRegex(FloatingPointError, 'second-cluster'):
                model.fit_frozen(self.bags, self.rna, ids=self.ids)
        self.assertFalse(bool(model.fitted_flag))
        with self.assertRaisesRegex(RuntimeError, 'cannot be retried'):
            model.fit_frozen(self.bags, self.rna, ids=self.ids)
        with self.assertRaises(RuntimeError):
            model.checkpoint()


if __name__ == '__main__':
    unittest.main()
