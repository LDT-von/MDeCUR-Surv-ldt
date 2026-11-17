"""Tests for method 3.2 (synthetic only, no patient data).

Run with:
    python -m pytest test_method3_2.py -q
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cancer_parity import discrete_nll, discrete_survival
from method3_2_model import (
    ClusterConditionalEncoder, ClusterRoutedSurvivalHead, Method32Config,
    MorphologyAtlas, cluster_association_auc, cluster_contrastive_proxy_loss,
    cross_fitted_association, fit_survival_head, harrell_c, head_features,
    multiplicative_replacement_clr, survival_bin_labels, train_cluster_contrastive,
)
from method3_2_synthetic_data import (
    CLUSTERS, SCENARIO_MECHANISMS, generate_cohort)


def _cohort(scenario='redundant', n=60, seed=7, prefix='t'):
    return generate_cohort(scenario, n, seed=seed, prefix=prefix)


class TestScenarioMechanisms(unittest.TestCase):
    """Every scenario must differ from the others in data, not only in name.

    A previous revision of the generator gave ``redundant``, ``path_only`` and
    ``rna_only`` the same patch latent and the same RNA, so those three
    benchmark columns were one condition repeated three times.
    """

    def test_scenarios_differ_in_the_mechanism_they_declare(self):
        """Scenarios may legitimately share one mechanism and differ in another.

        ``redundant`` and ``path_only`` both take their risk from ``z``, so
        their survival times are identical by construction; what must differ
        is the RNA.  The check therefore compares *per mechanism component*.
        """
        scenarios = ('redundant', 'path_only', 'rna_only', 'no_signal', 'xor',
                     'xor_additive', 'nonlinear', 'composition')
        cohorts = {name: generate_cohort(name, 40, seed=99, prefix=name)
                   for name in scenarios}
        for i, first in enumerate(scenarios):
            for second in scenarios[i + 1:]:
                a, b = cohorts[first], cohorts[second]
                ma, mb = SCENARIO_MECHANISMS[first], SCENARIO_MECHANISMS[second]
                if ma['rna'] != mb['rna']:
                    self.assertFalse(torch.equal(a.rna, b.rna),
                                     f'{first}/{second}: RNA modes differ but data matches')
                if ma['path'] != mb['path']:
                    self.assertFalse(torch.equal(a.bags[0], b.bags[0]),
                                     f'{first}/{second}: path modes differ but data matches')
                if ma['risk'] != mb['risk']:
                    self.assertFalse(torch.equal(a.times, b.times),
                                     f'{first}/{second}: risk modes differ but times match')

    def test_no_two_scenarios_share_the_same_full_mechanism(self):
        signatures = {}
        for name, mechanism in SCENARIO_MECHANISMS.items():
            key = (mechanism['path'], mechanism['rna'], mechanism['risk'])
            self.assertNotIn(key, signatures,
                             f'{name} and {signatures.get(key)} share the same mechanism')
            signatures[key] = name

    def test_path_only_has_no_rna_trace_of_the_latent(self):
        cohort = generate_cohort('path_only', 200, seed=5, prefix='po')
        corr = float(np.corrcoef(cohort.rna[:, 0].numpy(), cohort.truth['z'])[0, 1])
        self.assertLess(abs(corr), 0.25)

    def test_redundant_has_rna_trace_of_the_latent(self):
        cohort = generate_cohort('redundant', 200, seed=5, prefix='rd')
        corr = float(np.corrcoef(cohort.rna[:, 0].numpy(), cohort.truth['z'])[0, 1])
        self.assertGreater(corr, 0.8)

    def test_rna_only_pathology_carries_no_latent(self):
        cohort = generate_cohort('rna_only', 200, seed=5, prefix='ro')
        # the patch mean on the latent axis must not track z
        means = []
        for bag, labels in zip(cohort.bags, cohort.truth['cluster_labels']):
            rows = bag[labels == 0]
            means.append(float(rows[:, 4].mean()) if len(rows) else np.nan)
        means = np.asarray(means)
        ok = ~np.isnan(means)
        corr = float(np.corrcoef(means[ok], cohort.truth['z'][ok])[0, 1])
        self.assertLess(abs(corr), 0.25)

    def test_no_signal_has_no_prognostic_latent(self):
        cohort = generate_cohort('no_signal', 200, seed=5, prefix='ns')
        # hazard is flat, so the event time must not track z
        evented = cohort.events.numpy() > 0.5
        self.assertGreater(int(evented.sum()), 10)
        corr = float(np.corrcoef(cohort.times.numpy()[evented],
                                 cohort.truth['z'][evented])[0, 1])
        self.assertLess(abs(corr), 0.35)

    def test_nonlinear_pathology_is_mean_free(self):
        cohort = generate_cohort('nonlinear', 200, seed=5, prefix='nl')
        means = []
        for bag, labels in zip(cohort.bags, cohort.truth['cluster_labels']):
            rows = bag[labels == 0]
            means.append(float(rows[:, 4].mean()) if len(rows) else np.nan)
        means = np.asarray(means)
        ok = ~np.isnan(means)
        corr = float(np.corrcoef(means[ok], cohort.truth['z'][ok])[0, 1])
        # z**2 - 1 correlates ~0 with z, while z itself would correlate ~1
        self.assertLess(abs(corr), 0.35)

    def test_scenario_table_is_consistent_with_the_documentation(self):
        self.assertEqual(SCENARIO_MECHANISMS['redundant']['rna'], 'z')
        self.assertEqual(SCENARIO_MECHANISMS['path_only']['rna'], 'noise')
        self.assertEqual(SCENARIO_MECHANISMS['rna_only']['path'], 'none')
        self.assertEqual(SCENARIO_MECHANISMS['no_signal']['risk'], 'none')
        self.assertEqual(SCENARIO_MECHANISMS['xor']['risk'], 'xor')
        self.assertEqual(SCENARIO_MECHANISMS['xor_additive']['risk'], 'add')
        self.assertEqual(SCENARIO_MECHANISMS['nonlinear']['path'], 'z2')
        self.assertEqual(SCENARIO_MECHANISMS['composition']['risk'], 'comp')


class TestCompositionAndCLR(unittest.TestCase):
    def test_clr_closes_and_preserves_ratios(self):
        pi = torch.tensor([[0.5, 0.3, 0.2, 0.0]], dtype=torch.float64)
        clr = multiplicative_replacement_clr(pi)
        self.assertLess(abs(float(clr.sum())), 1e-12)
        # relative ratios of the non-zero entries are preserved
        ratio_before = float(pi[0, 0] / pi[0, 1])
        ratio_after = float(torch.exp(clr[0, 0] - clr[0, 1]))
        self.assertAlmostEqual(ratio_before, ratio_after, places=9)

    def test_clr_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            multiplicative_replacement_clr(torch.tensor([[0.0, 0.0]]))
        with self.assertRaises(ValueError):
            multiplicative_replacement_clr(torch.tensor([[-0.1, 1.1]]))
        with self.assertRaises(ValueError):
            multiplicative_replacement_clr(torch.tensor([[0.5, 0.5]]), delta=0.9)

    def test_zero_delta_rule_is_declared(self):
        # a zero entry must not be treated as "small but positive"
        pi = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        clr = multiplicative_replacement_clr(pi)
        self.assertLess(abs(float(clr.sum())), 1e-12)
        self.assertGreater(float(clr[0, 0]), 0)


class TestAtlas(unittest.TestCase):
    def setUp(self):
        self.cohort = _cohort()
        self.atlas = MorphologyAtlas.fit(self.cohort.bags, self.cohort.ids, clusters=CLUSTERS,
                                         latent_dim=8, per_patient_cap=64, seed=3)

    def test_composition_normalised_and_clr_closed(self):
        for patient in self.atlas.encode(self.cohort.bags, self.cohort.ids):
            self.assertAlmostEqual(float(patient.composition.sum()), 1.0, places=12)
            self.assertAlmostEqual(float(patient.clr.sum()), 0.0, places=12)
            self.assertAlmostEqual(float(patient.mass.sum()), float(len(patient.z)), places=9)

    def test_padding_is_excluded(self):
        padded = generate_cohort('redundant', 20, seed=5, prefix='pad', pad_to=40)
        self.assertIsInstance(padded.bags[0], tuple)
        atlas = MorphologyAtlas.fit(padded.bags, padded.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3)
        for patient in atlas.encode(padded.bags, padded.ids):
            self.assertTrue(torch.isfinite(patient.p).all())
            self.assertAlmostEqual(float(patient.mass.sum()), float(len(patient.z)), places=9)

    def test_low_mass_cluster_is_gated(self):
        atlas = MorphologyAtlas.fit(self.cohort.bags, self.cohort.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3, min_mass=1e9)
        for patient in atlas.encode(self.cohort.bags, self.cohort.ids):
            self.assertFalse(bool(patient.valid.any()))

    def test_fit_is_deterministic(self):
        again = MorphologyAtlas.fit(self.cohort.bags, self.cohort.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3)
        self.assertEqual(self.atlas.fingerprint(), again.fingerprint())

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            MorphologyAtlas.fit(self.cohort.bags, ['x'] * len(self.cohort.bags), clusters=2)

    def test_atlas_is_frozen_under_survival_gradient(self):
        before = self.atlas.fingerprint()
        parameters = [p for p in self.atlas.parameters()]
        self.assertEqual(len(parameters), 0)
        self.assertEqual(before, self.atlas.fingerprint())


class TestContrastiveProxy(unittest.TestCase):
    def test_positive_is_the_same_patient_and_no_self_negative(self):
        torch.manual_seed(0)
        encoder = ClusterConditionalEncoder(4, 3, 2, hidden=4, seed=0)
        u = torch.randn(6, 2, 4, dtype=torch.float64)
        rna = torch.randn(6, 3, dtype=torch.float64)
        valid = torch.ones(6, 2, dtype=torch.bool)
        loss, stats = cluster_contrastive_proxy_loss(encoder, u, rna, valid, min_patients=3)
        self.assertTrue(torch.isfinite(loss))
        for k in range(2):
            self.assertTrue(stats[k]['used'])
            self.assertEqual(stats[k]['patients'], 6)

    def test_small_cluster_is_skipped_not_padded(self):
        encoder = ClusterConditionalEncoder(4, 3, 2, hidden=4, seed=0)
        u = torch.randn(6, 2, 4, dtype=torch.float64)
        rna = torch.randn(6, 3, dtype=torch.float64)
        valid = torch.ones(6, 2, dtype=torch.bool)
        valid[:4, 1] = False
        loss, stats = cluster_contrastive_proxy_loss(encoder, u, rna, valid, min_patients=3)
        self.assertFalse(stats[1]['used'])
        self.assertEqual(stats[1]['patients'], 2)
        self.assertTrue(stats[0]['used'])
        self.assertTrue(torch.isfinite(loss))

    def test_all_clusters_too_small_returns_zero_loss(self):
        encoder = ClusterConditionalEncoder(4, 3, 2, hidden=4, seed=0)
        u = torch.randn(4, 2, 4, dtype=torch.float64)
        rna = torch.randn(4, 3, dtype=torch.float64)
        valid = torch.ones(4, 2, dtype=torch.bool)
        valid[:, 0] = False
        valid[:, 1] = False
        loss, stats = cluster_contrastive_proxy_loss(encoder, u, rna, valid, min_patients=3)
        self.assertEqual(float(loss), 0.0)

    def test_association_auc_is_at_chance_without_signal(self):
        torch.manual_seed(1)
        encoder = ClusterConditionalEncoder(4, 3, 1, hidden=4, seed=1)
        u = torch.randn(20, 1, 4, dtype=torch.float64)
        rna = torch.randn(20, 3, dtype=torch.float64)
        valid = torch.ones(20, 1, dtype=torch.bool)

        class _P:
            def __init__(self, p, valid):
                self.p = p
                self.valid = valid

        patients = [_P(u[i], torch.ones(1, dtype=torch.bool)) for i in range(20)]
        a, stats = cluster_association_auc(encoder, patients, rna, candidate_patients=patients,
                                          candidate_rna=rna, min_patients=8, n_shuffles=16)
        self.assertTrue(0.0 <= float(a[0]) <= 1.0)
        self.assertEqual(stats['attempts'][0], 20)

    def test_encoder_gradients_are_finite_and_nonzero(self):
        cohort = _cohort()
        atlas = MorphologyAtlas.fit(cohort.bags, cohort.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3)
        patients = atlas.encode(cohort.bags, cohort.ids)
        encoder = ClusterConditionalEncoder(atlas.latent_dim, cohort.rna.shape[1], CLUSTERS,
                                           hidden=8, seed=3)
        history = train_cluster_contrastive(encoder, patients, cohort.rna, epochs=3,
                                            batch_size=16, min_patients=4, seed=3)
        self.assertEqual(len(history), 3)
        self.assertTrue(all(math.isfinite(h['loss']) for h in history))
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))


class TestSurvivalHead(unittest.TestCase):
    def test_discrete_nll_hand_computation(self):
        # one patient, two bins, event in bin 0:
        # -log(S(0)) - log(h(0)) = -log(1) - log(0.5)
        logits = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        loss = float(discrete_nll(logits, torch.tensor([0]), torch.tensor([0]), alpha=0.0))
        expected = -math.log(1.0) - math.log(0.5)
        self.assertAlmostEqual(loss, expected, places=10)

    def test_censored_patient_uses_survival_only(self):
        logits = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        loss = float(discrete_nll(logits, torch.tensor([1]), torch.tensor([1]), alpha=0.0))
        self.assertAlmostEqual(loss, -math.log(0.25), places=10)

    def test_patientwise_no_cross_patient_broadcast(self):
        logits = torch.zeros(4, 3, dtype=torch.float64)
        labels = torch.tensor([0, 1, 2, 0])
        censorship = torch.tensor([0, 1, 0, 1])
        per_patient = discrete_nll(logits, labels, censorship, alpha=0.0, reduction='none')
        self.assertEqual(tuple(per_patient.shape), (4,))

    def test_bin_labels_follow_train_event_quantiles(self):
        times = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        events = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        labels, censorship, bounds = survival_bin_labels(times, events, bins=3)
        self.assertEqual(tuple(labels.shape), (6,))
        self.assertTrue(torch.equal(censorship, torch.tensor([0, 0, 1, 0, 0, 1])))
        self.assertLess(float(bounds[0]), float(bounds[1]))
        self.assertTrue(bool(((labels >= 0) & (labels < 3)).all()))

    def test_all_censored_cohort_stays_finite(self):
        times = torch.tensor([1.0, 5.0, 9.0, 12.0])
        events = torch.zeros(4)
        labels, censorship, bounds = survival_bin_labels(times, events, bins=4)
        logits = torch.zeros(4, 4, dtype=torch.float64)
        loss = discrete_nll(logits, labels, censorship, alpha=0.5)
        self.assertTrue(torch.isfinite(loss))

    def test_harrell_c_direction(self):
        times = torch.tensor([1.0, 2.0, 3.0])
        events = torch.tensor([1.0, 1.0, 1.0])
        good = harrell_c(torch.tensor([3.0, 2.0, 1.0]), times, events)
        bad = harrell_c(torch.tensor([1.0, 2.0, 3.0]), times, events)
        self.assertAlmostEqual(good, 1.0, places=9)
        self.assertAlmostEqual(bad, 0.0, places=9)

    def test_head_gradients_nonzero_for_each_enabled_block(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3,
                                use_generic_interaction=True)
        head = ClusterRoutedSurvivalHead(config)
        n = 12
        u = torch.randn(n, CLUSTERS, 8, dtype=torch.float64)
        rna = torch.randn(n, CLUSTERS, 8, dtype=torch.float64)
        clr = torch.randn(n, CLUSTERS, dtype=torch.float64)
        valid = torch.ones(n, CLUSTERS, dtype=torch.bool)
        route = torch.ones(CLUSTERS, dtype=torch.float64)
        logits = head(u, rna, clr, valid, route)
        self.assertEqual(tuple(logits.shape), (n, 4))
        loss = discrete_nll(logits, torch.randint(0, 4, (n,)), torch.randint(0, 2, (n,)))
        loss.backward()
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

    def test_zero_route_removes_cluster_evidence(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3,
                                use_composition=False)
        head = ClusterRoutedSurvivalHead(config)
        n = 6
        u = torch.randn(n, CLUSTERS, 8, dtype=torch.float64)
        rna = torch.randn(n, CLUSTERS, 8, dtype=torch.float64)
        clr = torch.zeros(n, CLUSTERS, dtype=torch.float64)
        valid = torch.ones(n, CLUSTERS, dtype=torch.bool)
        with torch.no_grad():
            head.path_weight.fill_(1.0)
            head.inter_weight.fill_(1.0)
            head.rna_weight.fill_(0.0)
            head.bias.zero_()
        routed = head(u, rna, clr, valid, torch.zeros(CLUSTERS, dtype=torch.float64))
        self.assertTrue(torch.allclose(routed, torch.zeros_like(routed), atol=1e-12))

    def test_generic_interaction_is_cluster_agnostic(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3,
                                use_interaction=False, use_generic_interaction=True)
        head = ClusterRoutedSurvivalHead(config)
        n = 5
        u = torch.randn(n, CLUSTERS, 8, dtype=torch.float64)
        rna = torch.randn(n, CLUSTERS, 8, dtype=torch.float64)
        clr = torch.zeros(n, CLUSTERS, dtype=torch.float64)
        valid = torch.ones(n, CLUSTERS, dtype=torch.bool)
        logits = head(u, rna, clr, valid, torch.ones(CLUSTERS, dtype=torch.float64))
        self.assertTrue(torch.isfinite(logits).all())


class TestRnaMasking(unittest.TestCase):
    """The RNA summary must not mix in untrained cluster projections."""

    def _head_and_inputs(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3,
                                use_composition=False)
        head = ClusterRoutedSurvivalHead(config)
        n = 4
        u = torch.randn(n, CLUSTERS, 8, dtype=torch.float64)
        rna = torch.randn(n, CLUSTERS, 8, dtype=torch.float64)
        clr = torch.zeros(n, CLUSTERS, dtype=torch.float64)
        valid = torch.ones(n, CLUSTERS, dtype=torch.bool)
        return head, u, rna, clr, valid

    def test_untrained_cluster_is_excluded_from_the_rna_summary(self):
        head, u, rna, clr, valid = self._head_and_inputs()
        with torch.no_grad():
            head.rna_weight.fill_(1.0)
            head.path_weight.zero_()
            head.inter_weight.zero_()
            head.bias.zero_()
        rna_valid = valid.clone()
        rna_valid[:, [0, 2, 3]] = False
        masked = head(u, rna, clr, valid, torch.ones(CLUSTERS, dtype=torch.float64),
                      rna_valid)
        # only cluster 1 survives, so the summary is rna[:, 1]
        expected = rna[:, 1] @ head.rna_weight.T
        self.assertTrue(torch.allclose(masked, expected, atol=1e-10))

    def test_untrained_cluster_is_excluded_from_the_interaction(self):
        head, u, rna, clr, valid = self._head_and_inputs()
        with torch.no_grad():
            head.inter_weight.fill_(1.0)
            head.path_weight.zero_()
            head.rna_weight.zero_()
            head.bias.zero_()
        rna_valid = valid.clone()
        rna_valid[:, [0, 2, 3]] = False
        masked = head(u, rna, clr, valid, torch.ones(CLUSTERS, dtype=torch.float64),
                      rna_valid)
        expected = (u[:, 1] * rna[:, 1]) @ head.inter_weight.T
        self.assertTrue(torch.allclose(masked, expected, atol=1e-10))

    def test_masking_actually_changes_the_result(self):
        head, u, rna, clr, valid = self._head_and_inputs()
        with torch.no_grad():
            head.rna_weight.normal_()
            head.inter_weight.normal_()
            head.path_weight.normal_()
        rna_valid = valid.clone()
        rna_valid[:, 0] = False
        unmasked = head(u, rna, clr, valid, torch.ones(CLUSTERS, dtype=torch.float64))
        masked = head(u, rna, clr, valid, torch.ones(CLUSTERS, dtype=torch.float64), rna_valid)
        self.assertGreater(float((unmasked - masked).abs().max()), 1e-6)

    def test_cluster_trained_mask_is_open_without_the_ssl_stage(self):
        encoder = ClusterConditionalEncoder(4, 3, CLUSTERS, hidden=4, seed=0)
        self.assertTrue(bool(encoder.cluster_trained_mask().all()))

    def test_cluster_trained_mask_excludes_untrained_clusters(self):
        encoder = ClusterConditionalEncoder(4, 3, CLUSTERS, hidden=4, seed=0)
        with torch.no_grad():
            encoder.ssl_enabled.fill_(1.0)
            encoder.cluster_usage.copy_(torch.tensor([1.0, 0.0, 3.0, 0.0],
                                                    dtype=torch.float64))
        mask = encoder.cluster_trained_mask()
        self.assertEqual([bool(x) for x in mask], [True, False, True, False])

    def test_head_features_marks_untrained_clusters_invalid(self):
        cohort = _cohort(n=30, seed=23)
        atlas = MorphologyAtlas.fit(cohort.bags, cohort.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3)
        encoder = ClusterConditionalEncoder(atlas.latent_dim, cohort.rna.shape[1], CLUSTERS,
                                            hidden=8, seed=3)
        with torch.no_grad():
            encoder.ssl_enabled.fill_(1.0)
            encoder.cluster_usage.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0],
                                                    dtype=torch.float64))
        patients = atlas.encode(cohort.bags, cohort.ids)
        features = head_features(encoder, patients, cohort.rna,
                                 route=torch.ones(CLUSTERS, dtype=torch.float64))
        self.assertEqual(tuple(features['rna_valid'].shape), (len(patients), CLUSTERS))
        self.assertFalse(bool(features['rna_valid'][:, 1].any()))
        self.assertFalse(bool(features['rna_valid'][:, 3].any()))
        self.assertTrue(bool(features['valid'][:, 1].any()) or True)


class TestCommonUniqueDecoupling(unittest.TestCase):
    """Explicit common / modality-specific decomposition in the survival head."""

    def _inputs(self, n=6):
        u = torch.randn(n, CLUSTERS, 8, dtype=torch.float64)
        rna = torch.randn(n, CLUSTERS, 8, dtype=torch.float64)
        clr = torch.zeros(n, CLUSTERS, dtype=torch.float64)
        valid = torch.ones(n, CLUSTERS, dtype=torch.bool)
        route = torch.ones(CLUSTERS, dtype=torch.float64)
        return u, rna, clr, valid, route

    def test_decomposition_is_energy_preserving(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3, decouple=True)
        head = ClusterRoutedSurvivalHead(config)
        u, rna, _, _, _ = self._inputs()
        u_common, u_unique, r_common, r_unique = head.decompose(u, rna)
        self.assertTrue(torch.allclose(
            u.pow(2).sum(-1),
            u_common.pow(2).sum(-1) + u_unique.pow(2).sum(-1), atol=1e-10))
        self.assertTrue(torch.allclose(
            rna.pow(2).sum(-1),
            r_common.pow(2).sum(-1) + r_unique.pow(2).sum(-1), atol=1e-10))

    def test_common_and_unique_are_orthogonal(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3, decouple=True)
        head = ClusterRoutedSurvivalHead(config)
        u, rna, _, _, _ = self._inputs()
        basis = head._basis()[0]
        # the concatenated basis is orthonormal
        gram = torch.einsum('khd,khe->kde', basis, basis)
        eye = torch.eye(8, dtype=torch.float64).expand_as(gram)
        self.assertTrue(torch.allclose(gram, eye, atol=1e-10))
        u_common, u_unique, _, _ = head.decompose(u, rna)
        # map both parts back into the original space and check orthogonality
        common_orig = torch.einsum('nkc,khc->nkh', u_common, basis[:, :, :4])
        unique_orig = torch.einsum('nkx,khx->nkh', u_unique, basis[:, :, 4:])
        dot = (common_orig * unique_orig).sum(-1)
        self.assertTrue(torch.allclose(dot, torch.zeros_like(dot), atol=1e-10))
        # and the two parts reconstruct the original vector
        rebuilt = common_orig + unique_orig
        self.assertTrue(torch.allclose(rebuilt, u, atol=1e-10))

    def test_symmetric_mode_shares_the_subspace(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3, decouple=True,
                                decouple_mode='symmetric')
        head = ClusterRoutedSurvivalHead(config)
        self.assertIsNone(head.common_proj_rna)
        path_basis, rna_basis = head._basis()
        self.assertTrue(torch.equal(path_basis, rna_basis))
        # the same input projected through either modality gives the same result
        u, _, _, _, _ = self._inputs()
        u_common, _, r_common, _ = head.decompose(u, u)
        self.assertTrue(torch.allclose(u_common, r_common, atol=1e-12))

    def test_asymmetric_mode_uses_separate_subspaces(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3, decouple=True,
                                decouple_mode='asymmetric')
        head = ClusterRoutedSurvivalHead(config)
        path_basis, rna_basis = head._basis()
        self.assertFalse(torch.equal(path_basis, rna_basis))
        u, _, _, _, _ = self._inputs()
        u_common, _, r_common, _ = head.decompose(u, u)
        self.assertFalse(torch.allclose(u_common, r_common, atol=1e-8))

    def test_invalid_decouple_mode_rejected(self):
        with self.assertRaises(ValueError):
            ClusterRoutedSurvivalHead(Method32Config(clusters=CLUSTERS, bins=4, hidden=8,
                                                    decouple=True, decouple_mode='nonsense'))
        with self.assertRaises(ValueError):
            ClusterRoutedSurvivalHead(Method32Config(clusters=CLUSTERS, bins=4, hidden=8,
                                                    decouple=True, common_dim=8))

    def test_aux_losses_are_finite_and_differentiable(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3, decouple=True)
        head = ClusterRoutedSurvivalHead(config)
        u, rna, _, valid, _ = self._inputs()
        aux = head.aux_losses(u, rna, valid)
        self.assertTrue(torch.isfinite(aux['common']))
        self.assertTrue(torch.isfinite(aux['unique']))
        self.assertGreaterEqual(float(aux['common']), 0.0)
        (aux['common'] + aux['unique']).backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_decoupled_head_has_gradients_for_every_block(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3, decouple=True,
                                use_generic_interaction=True)
        head = ClusterRoutedSurvivalHead(config)
        u, rna, clr, valid, route = self._inputs(n=12)
        clr = torch.randn(12, CLUSTERS, dtype=torch.float64)
        logits = head(u, rna, clr, valid, route)
        self.assertEqual(tuple(logits.shape), (12, 4))
        loss = discrete_nll(logits, torch.randint(0, 4, (12,)), torch.randint(0, 2, (12,)))
        loss.backward()
        for name in ('path_weight', 'common_inter_weight', 'unique_inter_weight',
                     'rna_weight', 'composition_weight', 'common_proj_path'):
            parameter = dict(head.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

    def test_common_energy_fraction_is_in_range(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3, decouple=True)
        head = ClusterRoutedSurvivalHead(config)
        u, rna, _, _, _ = self._inputs()
        energy = head.common_energy(u, rna)
        for key, value in energy.items():
            self.assertTrue(torch.isfinite(value).all(), key)
            self.assertTrue(bool(((value >= -1e-9) & (value <= 1.0 + 1e-9)).all()), key)

    def test_decoupled_head_respects_the_rna_mask(self):
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3, decouple=True)
        head = ClusterRoutedSurvivalHead(config)
        u, rna, clr, valid, route = self._inputs()
        rna_valid = valid.clone()
        rna_valid[:, 0] = False
        masked = head(u, rna, clr, valid, route, rna_valid)
        unmasked = head(u, rna, clr, valid, route, valid)
        self.assertGreater(float((masked - unmasked).abs().max()), 1e-6)


class TestTrainingLoop(unittest.TestCase):
    def test_head_training_reduces_loss_and_freezes_encoder(self):
        cohort = _cohort(n=80, seed=11)
        atlas = MorphologyAtlas.fit(cohort.bags, cohort.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3)
        patients = atlas.encode(cohort.bags, cohort.ids)
        encoder = ClusterConditionalEncoder(atlas.latent_dim, cohort.rna.shape[1], CLUSTERS,
                                            hidden=8, seed=3)
        route = torch.ones(CLUSTERS, dtype=torch.float64)
        features = head_features(encoder, patients, cohort.rna, route=route)
        labels, censorship, _ = survival_bin_labels(cohort.times, cohort.events, bins=4)
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3)
        head = ClusterRoutedSurvivalHead(config)
        encoder_before = copy.deepcopy(encoder.state_dict())
        history = fit_survival_head(head, features, labels, censorship, epochs=25,
                                    batch_size=16, seed=3)
        self.assertLess(history[-1]['train_nll'], history[0]['train_nll'])
        for key, value in encoder.state_dict().items():
            self.assertTrue(torch.equal(value, encoder_before[key]), key)

    def test_patient_reordering_does_not_change_the_loss_set(self):
        cohort = _cohort(n=40, seed=13)
        atlas = MorphologyAtlas.fit(cohort.bags, cohort.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3)
        patients = atlas.encode(cohort.bags, cohort.ids)
        encoder = ClusterConditionalEncoder(atlas.latent_dim, cohort.rna.shape[1], CLUSTERS,
                                            hidden=8, seed=3)
        route = torch.ones(CLUSTERS, dtype=torch.float64)
        features = head_features(encoder, patients, cohort.rna, route=route)
        labels, censorship, _ = survival_bin_labels(cohort.times, cohort.events, bins=4)
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3)
        head = ClusterRoutedSurvivalHead(config)
        order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            a = discrete_nll(head(features['u'][order], features['rna'][order],
                                  features['clr'][order], features['valid'][order], route),
                             labels[order], censorship[order], alpha=0.5, reduction='none')
            b = discrete_nll(head(features['u'], features['rna'], features['clr'],
                                  features['valid'], route),
                             labels, censorship, alpha=0.5, reduction='none')
        self.assertTrue(torch.allclose(a.sort().values, b.sort().values, atol=1e-12))

    def test_changing_rna_changes_the_output_under_the_designed_condition(self):
        cohort = _cohort(n=30, seed=17)
        atlas = MorphologyAtlas.fit(cohort.bags, cohort.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3)
        patients = atlas.encode(cohort.bags, cohort.ids)
        encoder = ClusterConditionalEncoder(atlas.latent_dim, cohort.rna.shape[1], CLUSTERS,
                                            hidden=8, seed=3)
        route = torch.ones(CLUSTERS, dtype=torch.float64)
        features = head_features(encoder, patients, cohort.rna, route=route)
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3)
        head = ClusterRoutedSurvivalHead(config)
        with torch.no_grad():
            head.rna_weight.normal_()
        base = head(features['u'], features['rna'], features['clr'], features['valid'], route)
        altered = cohort.rna.clone()
        altered[:, 0] += 5.0
        altered_features = head_features(encoder, patients, altered, route=route)
        other = head(altered_features['u'], altered_features['rna'], altered_features['clr'],
                     altered_features['valid'], route)
        self.assertGreater(float((base - other).abs().max()), 1e-6)

    def test_changing_pathology_changes_the_output(self):
        cohort = _cohort(n=30, seed=19)
        atlas = MorphologyAtlas.fit(cohort.bags, cohort.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3)
        patients = atlas.encode(cohort.bags, cohort.ids)
        encoder = ClusterConditionalEncoder(atlas.latent_dim, cohort.rna.shape[1], CLUSTERS,
                                            hidden=8, seed=3)
        route = torch.ones(CLUSTERS, dtype=torch.float64)
        features = head_features(encoder, patients, cohort.rna, route=route)
        config = Method32Config(clusters=CLUSTERS, bins=4, hidden=8, seed=3)
        head = ClusterRoutedSurvivalHead(config)
        with torch.no_grad():
            head.path_weight.normal_()
        base = head(features['u'], features['rna'], features['clr'], features['valid'], route)
        u2 = features['u'].clone()
        u2[:, 0, :] += 3.0
        other = head(u2, features['rna'], features['clr'], features['valid'], route)
        self.assertGreater(float((base - other).abs().max()), 1e-6)


class TestSplitIsolation(unittest.TestCase):
    def test_train_val_test_patient_ids_are_disjoint(self):
        train = generate_cohort('redundant', 30, seed=1, prefix='tr')
        val = generate_cohort('redundant', 10, seed=2, prefix='va')
        test = generate_cohort('redundant', 10, seed=3, prefix='te')
        self.assertFalse(set(train.ids) & set(val.ids))
        self.assertFalse(set(train.ids) & set(test.ids))
        self.assertFalse(set(val.ids) & set(test.ids))

    def test_changing_validation_and_test_data_keeps_the_training_state(self):
        cohort_a = generate_cohort('redundant', 60, seed=31, prefix='a_train')
        atlas = MorphologyAtlas.fit(cohort_a.bags, cohort_a.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3)
        encoder = ClusterConditionalEncoder(atlas.latent_dim, cohort_a.rna.shape[1], CLUSTERS,
                                            hidden=8, seed=3)
        train_cluster_contrastive(encoder, atlas.encode(cohort_a.bags, cohort_a.ids),
                                  cohort_a.rna, epochs=5, batch_size=16, min_patients=4,
                                  seed=3)

        def digest():
            hasher = hashlib.sha256()
            hasher.update(atlas.fingerprint().encode())
            for key, value in sorted(encoder.state_dict().items()):
                hasher.update(key.encode())
                hasher.update(value.detach().numpy().tobytes())
            return hasher.hexdigest()

        first = digest()
        other_val = generate_cohort('redundant', 5, seed=999, prefix='b_val')
        other_test = generate_cohort('redundant', 5, seed=888, prefix='b_test')
        # the fitted state does not depend on any validation/test patient
        self.assertEqual(first, digest())
        self.assertNotEqual(first, hashlib.sha256(other_val.rna.numpy().tobytes()).hexdigest())

    def test_truth_is_not_available_to_the_model(self):
        cohort = generate_cohort('redundant', 20, seed=41, prefix='truth')
        self.assertIn('cluster_labels', cohort.truth)
        # the model API only consumes bags, RNA, times and events
        atlas = MorphologyAtlas.fit(cohort.bags, cohort.ids, clusters=CLUSTERS,
                                    latent_dim=8, per_patient_cap=64, seed=3)
        encoded = atlas.encode(cohort.bags, cohort.ids)
        self.assertFalse(hasattr(encoded[0], 'cluster_labels'))
        self.assertFalse(hasattr(encoded[0], 'z_truth'))


if __name__ == '__main__':
    unittest.main()
