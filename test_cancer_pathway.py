"""Known-answer and synthetic end-to-end checks for the full local framework."""
from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from cancer_auxiliary import patient_balanced_local_correlation
from cancer_model import piecewise_nll, survival_at
from cancer_pathway import PathwayLocalSurvival, pathway_correlation_loss, read_pathway_mapping
from cancer_pathway_stream import (PathwayConfig, _feature_scores, effective_weights,
                                   feature_noise_pair, minimum_batch_size, patient_batches,
                                   pathway_objective, pathway_stream_forward, predict_pathway_model,
                                   train_pathway_model, validate_config)
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
from cancer_single_view import PTPatientBag
from model import correlation_loss
from test_uni_views import create_cache, write_csv


class SyntheticBag:
    view_source = 'cached_uni'

    def __init__(self, patient, reference, view_a, view_b):
        self.patient_id = patient
        self.data = dict(features_reference=reference, features_a=view_a, features_b=view_b)
        self.n_patches, self.feature_dim = reference.shape
        self.reads = []

    def blocks(self, size):
        for start in range(0, self.n_patches, size):
            yield (f'{self.patient_id}.pt', start, min(start+size, self.n_patches))

    def read_block(self, token, names):
        self.reads.append((token, tuple(names)))
        return tuple(self.data[name][token[1]:token[2]].clone() for name in names)


def synthetic_mapping(root):
    genes = root/'genes.txt'
    genes.write_text('\n'.join(f'G{i}' for i in range(8))+'\n', encoding='utf-8')
    gmt = root/'pathways.gmt'
    gmt.write_text('Path one\tID1\tG0\tG1\tG2\tGmissing\n'
                   'Path two\tID2\tG2\tG3\tG4\tG5\n', encoding='utf-8')
    return genes, gmt


def synthetic_patient_manifest(root, source='feature_noise'):
    generator = torch.Generator().manual_seed(348)
    rows = []
    for i in range(10):
        split = 'train' if i < 7 else 'val' if i < 9 else 'test'
        if split == 'test':
            feature = 'ABSENT.pt' if source == 'feature_noise' else 'ABSENT.h5'
            rna, time, event = 'ABSENT.npy', 'DO_NOT_PARSE', 'DO_NOT_PARSE'
        else:
            if source == 'feature_noise':
                feature = f'patient{i}.pt'
                torch.save(torch.randn(4+i, 6, generator=generator), root/feature)
            else:
                feature = create_cache(root, f'slide{i}', f'fake{i}')[0].name
            rna = f'rna{i}.npy'
            np.save(root/rna, torch.randn(8, generator=generator).numpy().astype(np.float32))
            time, event = .3+.2*i, i % 2
        rows.append({'patient_id': f'fake{i}', 'split': split,
                     'wsi_pt' if source == 'feature_noise' else 'wsi_h5': feature,
                     'rna_npy': rna, 'time': time, 'event': event})
    manifest = root/f'{source}_patients.csv'
    write_csv(manifest, rows)
    return manifest


def eager_forward(model, bags, rna, rna_a, rna_b):
    """Independent dense softmax calculation of the full fixed-view model."""
    tokens = model.pathways(rna)
    tokens_a, tokens_b = model.pathways(rna_a), model.pathways(rna_b)
    queries = model.match_queries(tokens)
    queries_a, queries_b = model.match_queries(tokens_a), model.match_queries(tokens_b)
    globals_by_view = [[], [], []]
    regions_a, regions_b, matches, local_a, local_b, entropies, maxima = [], [], [], [], [], [], []
    matches_a, matches_b = [], []
    for i, bag in enumerate(bags):
        embedded = [model.base.patch_encoder(bag.data[name])
                    for name in ('features_reference', 'features_a', 'features_b')]
        for index, h in enumerate(embedded):
            score = model.base.attention(h).squeeze(-1)
            globals_by_view[index].append((score.softmax(0)[:, None]*h).sum(0))
        def region(h):
            scores = h@model.pathway_queries.T/math.sqrt(model.hidden)
            return scores.softmax(0).T@h
        regions_a.append(region(embedded[1]))
        regions_b.append(region(embedded[2]))
        match = F.normalize(model.match_path(embedded[0]), dim=-1, eps=1e-6)
        scores = match@queries[i].T/model.match_temperature
        weights = scores.softmax(0)
        matches.append(weights.T@embedded[0])
        for embedded_view, query_view, destination in ((embedded[1], queries_a[i], matches_a),
                                                        (embedded[2], queries_b[i], matches_b)):
            view_keys = F.normalize(model.match_path(embedded_view), dim=-1, eps=1e-6)
            view_scores = view_keys@query_view.T/model.match_temperature
            destination.append(view_scores.softmax(0).T@embedded_view)
        entropy = -torch.special.xlogy(weights, weights).sum(0)
        entropies.append(entropy / math.log(len(weights)) if len(weights) > 1 else entropy)
        maxima.append(weights.max(0).values)
        local_a.append(model.base.local_projector(embedded[1]))
        local_b.append(model.base.local_projector(embedded[2]))
    hp = torch.stack(globals_by_view[0])
    hr = model.base.rna_encoder(rna)
    logits, interaction = model.prediction(hp, hr, torch.stack(matches), tokens)
    p1, r1 = model.local_projection(torch.stack(regions_a), tokens_a)
    p2, r2 = model.local_projection(torch.stack(regions_b), tokens_b)
    match_p1, match_r1 = model.local_projection(torch.stack(matches_a), tokens_a)
    match_p2, match_r2 = model.local_projection(torch.stack(matches_b), tokens_b)
    nmax = max(len(x) for x in local_a)
    padding_a = torch.stack([F.pad(x, (0, 0, 0, nmax-len(x))) for x in local_a])
    padding_b = torch.stack([F.pad(x, (0, 0, 0, nmax-len(x))) for x in local_b])
    mask = torch.stack([torch.arange(nmax) < len(x) for x in local_a])
    patch_loss, _ = patient_balanced_local_correlation(padding_a, padding_b, mask)
    aux = dict(global_p1=model.base.path_projector(torch.stack(globals_by_view[1])),
               global_p2=model.base.path_projector(torch.stack(globals_by_view[2])),
               global_r1=model.base.rna_projector(model.base.rna_encoder(rna_a)),
               global_r2=model.base.rna_projector(model.base.rna_encoder(rna_b)),
               local_p1=p1, local_p2=p2, local_r1=r1, local_r2=r2,
               matched_p1=match_p1, matched_p2=match_p2, matched_r1=match_r1, matched_r2=match_r2)
    return dict(log_rates=logits, interaction_logits=interaction, patch_loss=patch_loss, auxiliary=aux,
                matching_normalized_entropy=torch.stack(entropies), matching_max_weight=torch.stack(maxima))


def fixed_loss(output, times, events, starts, common, local_common):
    aux = output['auxiliary']
    loss = piecewise_nll(output['log_rates'], times, events, starts)+.01*output['patch_loss']
    for x, y, target in ((aux['global_p1'], aux['global_p2'], 1),
                         (aux['global_r1'], aux['global_r2'], 1),
                         (aux['global_p1'][:, :common], aux['global_r1'][:, :common], 1),
                         (aux['global_p1'][:, common:], aux['global_r1'][:, common:], 0)):
        loss = loss+.01*correlation_loss(x, y, target)
    for x, y, target in ((aux['local_p1'], aux['local_p2'], 1),
                         (aux['local_r1'], aux['local_r2'], 1),
                         (aux['local_p1'][..., :local_common], aux['local_r1'][..., :local_common], 1),
                         (aux['local_p1'][..., local_common:], aux['local_r1'][..., local_common:], 0)):
        loss = loss+.01*pathway_correlation_loss(x, y, target)[0]
    for target, selection in ((1, slice(None, local_common)), (0, slice(local_common, None))):
        left = pathway_correlation_loss(aux['matched_p1'][..., selection], aux['matched_r1'][..., selection], target)[0]
        right = pathway_correlation_loss(aux['matched_p2'][..., selection], aux['matched_r2'][..., selection], target)[0]
        loss = loss+.005*(left+right)
    return loss


class CancerPathwayTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_exact_gene_mapping_overlap_and_frozen_reactome_source(self):
        genes, gmt = synthetic_mapping(self.root)
        mapping = read_pathway_mapping(genes, gmt, 8, min_genes=2)
        self.assertEqual([p['indices'] for p in mapping['pathways']], [[0, 1, 2], [2, 3, 4, 5]])
        self.assertEqual(mapping['pathways'][0]['coverage'], .75)
        self.assertIn(2, mapping['pathways'][0]['indices'])
        self.assertIn(2, mapping['pathways'][1]['indices'])
        genes.write_text('G0\nG0\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unique'):
            read_pathway_mapping(genes, gmt, 2)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            read_pathway_mapping(genes, gmt, 8)
        genes, gmt = synthetic_mapping(self.root)
        gmt.write_text('A\tSAME\tG0\tG1\nB\tSAME\tG2\tG3\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_pathway_mapping(genes, gmt, 8, 2)
        root = Path(__file__).parent/'resources'
        metadata = json.loads((root/'reactome_human_roots.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['root_count'], 29)
        pathway_file = root/'reactome_human_roots.gmt'
        self.assertEqual(hashlib.sha256(pathway_file.read_bytes()).hexdigest(), metadata['output_sha256'])
        rows = pathway_file.read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(rows), 29)
        self.assertTrue(all(row.split('\t')[1].startswith('R-HSA-') for row in rows))

    def test_per_pathway_correlations_match_independent_numpy(self):
        rng = np.random.default_rng(37)
        a, b = rng.normal(size=(7, 3, 4)), rng.normal(size=(7, 3, 4))
        an, bn = a-a.mean(0), b-b.mean(0)
        an = an/np.sqrt((an*an).mean(0)+1e-4)
        bn = bn/np.sqrt((bn*bn).mean(0)+1e-4)
        matrix = np.einsum('bkd,bke->kde', an, bn)/len(a)
        for target in (0, 1):
            eye = np.eye(4)
            expected = (((matrix-np.eye(4)*np.diagonal(matrix, axis1=1, axis2=2)[:, :, None])**2).sum((1, 2))*.005
                        +(np.diagonal(matrix, axis1=1, axis2=2)-target).astype(float).__pow__(2).sum(1)).mean()
            actual, info = pathway_correlation_loss(torch.tensor(a), torch.tensor(b), target)
            self.assertAlmostEqual(actual.item(), expected, places=11)
            self.assertEqual((info['patients'], info['pathways']), (7, 3))
        with self.assertRaises(ValueError):
            pathway_correlation_loss(torch.ones(1, 3, 4), torch.ones(1, 3, 4), 1)

    def test_local_shared_objective_detects_shuffled_patient_pairing(self):
        # Orthogonal centered factors, repeated over two explicitly named pathways.
        data = torch.tensor([[-2., 1.], [-1., -2.], [0., 2.], [1., -2.], [2., 1.]], dtype=torch.float64)
        path = torch.stack((data, data[:, [1, 0]]), dim=1)
        matched, _ = pathway_correlation_loss(path, path, 1)
        shuffled, _ = pathway_correlation_loss(path, path[[2, 4, 0, 1, 3]], 1)
        self.assertLess(matched.item(), .02)
        self.assertGreater(shuffled.item(), matched.item()+.1)

    def test_batch_rank_rule_and_tail_merging(self):
        cfg = PathwayConfig(hidden=4, common=2, local_dim=4, local_common=2)
        self.assertEqual(minimum_batch_size(cfg), 5)
        with self.assertRaisesRegex(ValueError, 'patient_batch_size >= 5'):
            validate_config(cfg, 'feature_noise', 4, 2)
        batches = patient_batches(torch.arange(12), 5, 5)
        self.assertEqual([len(x) for x in batches], [5, 7])
        self.assertEqual(torch.cat(batches).tolist(), list(range(12)))
        with self.assertRaisesRegex(ValueError, 'at least 5'):
            patient_batches(torch.arange(4), 5, 5)
        with self.assertRaisesRegex(ValueError, 'positive noise'):
            validate_config(replace(cfg, feature_noise=0), 'feature_noise', 5, 2)
        self.assertEqual(validate_config(replace(cfg, feature_noise=0), 'image_d4', 5, 2), 5)

    def test_feature_noise_is_explicit_reproducible_aligned_and_checkpoint_safe(self):
        reference = torch.arange(1, 31, dtype=torch.float32).reshape(5, 6)
        before = reference.clone()
        block = ('slideA.pt', 0, 5)
        a, b = feature_noise_pair(reference, 'patient', block, 19, .05)
        again_a, again_b = feature_noise_pair(reference, 'patient', block, 19, .05)
        self.assertTrue(torch.equal(a, again_a))
        self.assertTrue(torch.equal(b, again_b))
        self.assertFalse(torch.equal(a, b))
        self.assertFalse(torch.equal(a, reference))
        self.assertTrue(torch.equal(reference, before))
        self.assertEqual(a.shape, reference.shape)
        self.assertFalse(torch.equal(a, feature_noise_pair(reference, 'patient', block, 20, .05)[0]))

    def test_patch_pathway_soft_matching_has_known_behavior(self):
        model = PathwayLocalSurvival(6, 8, [[0, 1], [2, 3]], hidden=4, common=2,
                                     local_dim=4, local_common=2, match_temperature=.2)
        with torch.no_grad():
            model.match_path.weight.copy_(torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0]]))
        features = torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0]])
        query = torch.eye(2)
        scores = _feature_scores(model, features, query)[:, 3:]
        weights = scores.softmax(0)
        self.assertGreater(weights[0, 0].item(), .99)
        self.assertGreater(weights[1, 1].item(), .99)
        matched = weights.T@features
        self.assertGreater(matched[0, 0].item(), matched[0, 1].item())
        self.assertGreater(matched[1, 1].item(), matched[1, 0].item())

    def test_auxiliary_regions_are_independent_of_same_patient_rna_at_forward(self):
        torch.manual_seed(58)
        model = PathwayLocalSurvival(6, 8, [[0, 1, 2], [2, 3, 4]], hidden=4, common=2,
                                     local_dim=4, local_common=2)
        bags = []
        for i in range(2):
            x = torch.randn(5+i, 6)
            bags.append(SyntheticBag(f'p{i}', x, x+.1, x-.2))
        first = torch.randn(2, 8)
        second = torch.randn(2, 8)*7
        a = pathway_stream_forward(model, bags, first, 3, 'image_d4', True, False,
                                   (first+.03, first-.02), checkpoint_chunks=False)
        b = pathway_stream_forward(model, bags, second, 3, 'image_d4', True, False,
                                   (second+.03, second-.02), checkpoint_chunks=False)
        for key in ('pathology_regions1', 'pathology_regions2', 'local_p1', 'local_p2'):
            self.assertTrue(torch.equal(a['auxiliary'][key], b['auxiliary'][key]))
        self.assertFalse(torch.allclose(a['auxiliary']['matched_p1'], b['auxiliary']['matched_p1']))
        self.assertFalse(torch.allclose(a['rna_pathways'], b['rna_pathways']))
        self.assertFalse(torch.allclose(a['matched_path'], b['matched_path']))

    def test_prediction_uses_conditional_matching_not_pathology_auxiliary_queries(self):
        torch.manual_seed(59)
        model = PathwayLocalSurvival(6, 8, [[0, 1, 2], [2, 3, 4]], hidden=4, common=2,
                                     local_dim=4, local_common=2)
        x = torch.randn(7, 6)
        bag = SyntheticBag('p', x, x, x)
        output = pathway_stream_forward(model, [bag], torch.randn(1, 8), 3, 'image_d4')
        output['log_rates'].sum().backward()
        query_grad = model.pathway_queries.grad
        self.assertTrue(query_grad is None or torch.equal(query_grad, torch.zeros_like(query_grad)))
        self.assertGreater(model.match_path.weight.grad.norm().item(), 0.)
        self.assertGreater(model.match_rna.weight.grad.norm().item(), 0.)
        self.assertGreater(model.interaction_head.weight.grad.norm().item(), 0.)

    def test_prediction_reads_reference_only_for_both_view_sources(self):
        torch.manual_seed(60)
        model = PathwayLocalSurvival(6, 8, [[0, 1, 2], [2, 3, 4]], hidden=4, common=2,
                                     local_dim=4, local_common=2)
        reference = torch.randn(7, 6)
        bag = SyntheticBag('p', reference, torch.randn_like(reference), torch.randn_like(reference))
        rna = torch.randn(1, 8)
        first = pathway_stream_forward(model, [bag], rna, 3, 'image_d4', checkpoint_chunks=False)
        self.assertTrue(all(names == ('features_reference',) for _, names in bag.reads))
        bag.data['features_a'].fill_(10000.)
        bag.data['features_b'].fill_(-10000.)
        second = pathway_stream_forward(model, [bag], rna, 3, 'image_d4', checkpoint_chunks=False)
        self.assertTrue(torch.equal(first['log_rates'], second['log_rates']))
        bag.view_source = 'single_view'
        third = pathway_stream_forward(model, [bag], rna, 3, 'feature_noise', checkpoint_chunks=False)
        self.assertTrue(torch.equal(first['log_rates'], third['log_rates']))

    def test_dense_vs_chunked_loss_and_all_gradients_for_image_views(self):
        torch.manual_seed(63)
        arrays = [torch.randn(n, 6, dtype=torch.float64) for n in (3, 6, 5, 4, 7)]
        bags = [SyntheticBag(f'p{i}', x, x+.2*torch.randn_like(x), x+.3*torch.randn_like(x))
                for i, x in enumerate(arrays)]
        rna = torch.randn(5, 8, dtype=torch.float64)
        rna_a, rna_b = rna+.05*torch.randn_like(rna), rna+.05*torch.randn_like(rna)
        times = torch.tensor([.2, 1., 2., .6, 1.7], dtype=torch.float64)
        events = torch.tensor([1., 0., 1., 1., 0.], dtype=torch.float64)
        starts = torch.tensor([0., 1.], dtype=torch.float64)
        base = PathwayLocalSurvival(6, 8, [[0, 1, 2], [2, 3, 4, 5]], hidden=4, common=2,
                                    local_dim=4, local_common=2, bins=2).double()
        eager = deepcopy(base)
        target = eager_forward(eager, bags, rna, rna_a, rna_b)
        target_loss = fixed_loss(target, times, events, starts, 2, 2)
        target_loss.backward()
        gradients = {name: None if p.grad is None else p.grad.clone() for name, p in eager.named_parameters()}
        for chunk, replay in ((1, True), (3, True), (99, False)):
            candidate = deepcopy(base)
            actual = pathway_stream_forward(candidate, bags, rna, chunk, 'image_d4', True, True,
                                             (rna_a, rna_b), checkpoint_chunks=replay)
            torch.testing.assert_close(actual['log_rates'], target['log_rates'], atol=1e-9, rtol=1e-8)
            torch.testing.assert_close(actual['patch_loss'], target['patch_loss'], atol=1e-9, rtol=1e-8)
            torch.testing.assert_close(actual['matching_normalized_entropy'], target['matching_normalized_entropy'], atol=1e-9, rtol=1e-8)
            torch.testing.assert_close(actual['matching_max_weight'], target['matching_max_weight'], atol=1e-9, rtol=1e-8)
            for key in target['auxiliary']:
                torch.testing.assert_close(actual['auxiliary'][key], target['auxiliary'][key], atol=1e-9, rtol=1e-8)
            actual_loss = fixed_loss(actual, times, events, starts, 2, 2)
            torch.testing.assert_close(actual_loss, target_loss, atol=1e-9, rtol=1e-8)
            actual_loss.backward()
            for name, parameter in candidate.named_parameters():
                if gradients[name] is None:
                    self.assertIsNone(parameter.grad, name)
                else:
                    torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-9, rtol=1e-7, msg=name)

    def test_full_objective_gives_gradients_to_matching_local_and_pathway_modules(self):
        torch.manual_seed(66)
        cfg = PathwayConfig(hidden=4, common=2, local_dim=4, local_common=2)
        model = PathwayLocalSurvival(6, 8, [[0, 1, 2], [2, 3, 4]], hidden=4, common=2,
                                     local_dim=4, local_common=2)
        bags = []
        for i in range(5):
            x = torch.randn(4+i, 6)
            bag = SyntheticBag(f'p{i}', x, x, x)
            bag.view_source = 'single_view'
            bags.append(bag)
        batch = dict(bags=bags, rna=torch.randn(5, 8), time=torch.tensor([.2, .5, 1., 1.5, 2.]),
                     event=torch.tensor([1., 0., 1., 0., 1.]), rna_mask=torch.ones(8, dtype=torch.bool))
        loss, stats = pathway_objective(model, batch, torch.tensor([0.]), cfg, 3, 'feature_noise', 22)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for key in effective_weights(cfg):
            self.assertIsNotNone(stats['terms'][key], key)
        for name in ('pathway_queries', 'match_path.weight', 'match_rna.weight',
                     'local_path_projector.weight', 'local_rna_projector.weight', 'interaction_head.weight',
                     'base.patch_encoder.0.weight', 'base.rna_encoder.0.weight', 'pathways.encoders.0.0.weight'):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.norm().item(), 0., name)

    def test_stability_weighted_global_common_is_active_and_records_degeneracy(self):
        torch.manual_seed(471)
        cfg = PathwayConfig(hidden=4, common=2, local_dim=4, local_common=2,
                            common_mode='stability')
        model = PathwayLocalSurvival(6, 8, [[0, 1, 2], [2, 3, 4]], hidden=4, common=2,
                                     local_dim=4, local_common=2)
        bags = []
        for i in range(8):
            x = torch.randn(4+i, 6)
            bag = SyntheticBag(f'p{i}', x, x, x)
            bag.view_source = 'single_view'
            bags.append(bag)
        batch = dict(bags=bags, rna=torch.randn(8, 8), time=torch.linspace(.2, 2., 8),
                     event=torch.tensor([1., 0., 1., 0., 1., 0., 1., 0.]),
                     rna_mask=torch.ones(8, dtype=torch.bool))
        loss, stats = pathway_objective(model, batch, torch.tensor([0.]), cfg, 4, 'feature_noise', 22)
        self.assertFalse(stats['common_skipped'])
        self.assertIsNone(stats['common_skip_reason'])
        self.assertTrue(math.isfinite(stats['terms']['global_common']))
        self.assertGreaterEqual(stats['global_common_ess'], 4.)
        self.assertLessEqual(stats['global_common_ess'], 8.0001)
        loss.backward()
        for name in ('base.path_projector.weight', 'base.rna_projector.weight'):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.norm().item(), 0., name)

        # A constant path projection must be recorded as skipped, never silently
        # replaced by the unweighted objective.
        with torch.no_grad():
            model.base.path_projector.weight.zero_()
            model.base.path_projector.bias.zero_()
        _, degenerate = pathway_objective(model, batch, torch.tensor([0.]), cfg, 4, 'feature_noise', 22)
        self.assertTrue(degenerate['common_skipped'])
        self.assertEqual(degenerate['common_skip_reason'], 'Degenerate weighted common feature variance')
        self.assertIsNone(degenerate['terms']['global_common'])
        self.assertIsNotNone(degenerate['terms']['matched_common'])

    def test_shuffled_common_keeps_subsequent_rna_views_identical(self):
        from cancer_rna import rna_noise_view
        torch.manual_seed(471)
        cfg = PathwayConfig(hidden=4, common=2, local_dim=4, local_common=2,
                            common_mode='stability')
        model = PathwayLocalSurvival(6, 8, [[0, 1, 2], [2, 3, 4]], hidden=4, common=2,
                                     local_dim=4, local_common=2)
        bags = []
        for i in range(8):
            x = torch.randn(4+i, 6)
            bag = SyntheticBag(f'p{i}', x, x, x)
            bag.view_source = 'single_view'
            bags.append(bag)
        batch = dict(bags=bags, rna=torch.randn(8, 8), time=torch.linspace(.2, 2., 8),
                     event=torch.tensor([1., 0., 1., 0., 1., 0., 1., 0.]),
                     rna_mask=torch.ones(8, dtype=torch.bool))
        def collect(mode):
            views = []
            def capture(*args, **kwargs):
                value = rna_noise_view(*args, **kwargs)
                views.append(value.clone())
                return value
            torch.manual_seed(991)
            with torch.no_grad(), patch('cancer_pathway_stream.rna_noise_view', side_effect=capture):
                for seed in (31, 32):
                    pathway_objective(model, batch, torch.tensor([0.]), replace(cfg, common_mode=mode),
                                      4, 'feature_noise', seed)
            return views, torch.get_rng_state()
        reference, reference_state = collect('stability')
        shuffled, shuffled_state = collect('shuffled')
        self.assertEqual(len(reference), 4)
        for a, b in zip(reference, shuffled):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(reference_state, shuffled_state))

    def test_feature_noise_checkpoint_and_noncheckpoint_gradients_match(self):
        torch.manual_seed(67)
        base = PathwayLocalSurvival(6, 8, [[0, 1, 2], [2, 3, 4]], hidden=4, common=2,
                                    local_dim=4, local_common=2)
        bags = []
        for i in range(5):
            x = torch.randn(4+i, 6)
            bag = SyntheticBag(f'p{i}', x, x, x)
            bag.view_source = 'single_view'
            bags.append(bag)
        rna = torch.randn(5, 8)
        rna_views = (rna+.05*torch.randn_like(rna), rna+.05*torch.randn_like(rna))
        time, event, starts = torch.linspace(.1, 1., 5), torch.tensor([1., 0., 1., 0., 1.]), torch.tensor([0.])
        results = []
        for replay in (False, True):
            model = deepcopy(base)
            out = pathway_stream_forward(model, bags, rna, 3, 'feature_noise', True, True,
                                         rna_views, .05, 123, checkpoint_chunks=replay)
            loss = fixed_loss(out, time, event, starts, 2, 2)
            loss.backward()
            results.append((loss, {name: None if p.grad is None else p.grad.clone()
                                   for name, p in model.named_parameters()}))
        torch.testing.assert_close(results[0][0], results[1][0])
        for name, expected in results[0][1].items():
            if expected is None:
                self.assertIsNone(results[1][1][name])
            else:
                torch.testing.assert_close(results[1][1][name], expected, msg=name)

    def test_single_patch_matching_diagnostics_and_source_change_guard(self):
        torch.manual_seed(68)
        model = PathwayLocalSurvival(6, 8, [[0, 1, 2], [2, 3, 4]], hidden=4, common=2,
                                     local_dim=4, local_common=2)
        one = torch.randn(1, 6)
        bag = SyntheticBag('p', one, one, one)
        output = pathway_stream_forward(model, [bag], torch.randn(1, 8), 2, 'image_d4')
        self.assertTrue(torch.equal(output['matching_normalized_entropy'], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(output['matching_max_weight'], torch.ones(1, 2)))
        path = self.root/'patient.pt'
        torch.save(torch.randn(7, 6), path)
        file_bag = PTPatientBag('patient', path.name, self.root)
        try:
            output = pathway_stream_forward(model, [file_bag], torch.randn(1, 8), 3, 'feature_noise')
            before = path.stat()
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns+1_000_000))
            with self.assertRaisesRegex(ValueError, 'changed'):
                output['log_rates'].sum().backward()
        finally:
            file_bag.close()

    def test_training_batches_mask_and_label_free_prediction(self):
        torch.manual_seed(70)
        bags = []
        for i in range(9):
            x = torch.randn(4+i, 6)
            bag = SyntheticBag(f'p{i}', x, x, x)
            bag.view_source = 'single_view'
            bags.append(bag)
        rna = torch.randn(9, 8)
        rna[:7, 0] = 1.
        rna[7:, 0] = 100000.
        def part(indices):
            return dict(ids=[bags[i].patient_id for i in indices], bags=[bags[i] for i in indices],
                        rna=rna[indices], time=torch.arange(1., len(indices)+1),
                        event=torch.tensor(([1., 0.]*len(indices))[:len(indices)]))
        train, val = part(list(range(7))), part([7, 8])
        mapping = {'rna_dim': 8, 'pathways': [{'id': 'a', 'indices': [0, 1, 2]},
                                             {'id': 'b', 'indices': [2, 3, 4]}]}
        cfg = PathwayConfig(hidden=4, common=2, local_dim=4, local_common=2, epochs=2, patience=2)
        fitted = train_pathway_model(train, val, torch.tensor([0.]), cfg, mapping, 5, 3)
        self.assertFalse(fitted['rna_mask'][0])
        self.assertEqual([h['batch_patient_counts'] for h in fitted['history']], [[7], [7]])
        self.assertEqual(fitted['best_epoch'], min(fitted['history'], key=lambda h: h['validation_nll'])['epoch'])
        result = predict_pathway_model(fitted, {k:v for k,v in val.items() if k not in ('time', 'event')})
        self.assertEqual(result['log_rates'].shape, (2, 1))
        self.assertTrue(torch.isfinite(result['log_rates']).all())
        self.assertEqual(fitted['stream_config']['view_source'], 'feature_noise')

    def run_cli(self, manifest, genes, gmt, destination, source, evaluate_test=False):
        command = [sys.executable, '-B', 'train_pathway.py', '--manifest', str(manifest),
                   '--genes', str(genes), '--pathways', str(gmt), '--min-pathway-genes', '2',
                   '--out', str(destination), '--view-source', source, '--device', 'cpu',
                   '--hidden', '4', '--common', '2', '--local-dim', '4', '--local-common', '2',
                   '--patient-batch-size', '5', '--patch-chunk-size', '3', '--epochs', '2', '--patience', '2',
                   '--starts', '0', '1', '--horizon', '1', '--time-unit', 'months', '--endpoint', 'synthetic']
        if evaluate_test:
            command.append('--evaluate-test')
        return subprocess.run(command, cwd=Path(__file__).parent,
                              env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                              capture_output=True, text=True)

    def test_cli_single_feature_views_roundtrip_and_test_isolation(self):
        genes, gmt = synthetic_mapping(self.root)
        manifest = synthetic_patient_manifest(self.root)
        destination = self.root/'result'
        result = self.run_cli(manifest, genes, gmt, destination, 'feature_noise')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((destination/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['architecture'], 'PathwayLocalSurvival')
        self.assertEqual(report['view_source'], 'feature_noise')
        self.assertIn('not image-space', report['view_source_description'])
        self.assertEqual(report['patients'], {'train': 7, 'val': 2})
        self.assertEqual(report['evaluation_risk_numerics']['legacy_probability_risk_c'],
                         report['evaluation_c_index_risk_at_horizon'])
        self.assertIn('legacy_v1', report['metric_conventions']['ipcw_brier'])
        self.assertEqual(report['stream_config']['auxiliary_rng_scheme'],
                         'rna_global__weight_shuffle_isolated_v1')
        self.assertFalse(report['test_evaluated'])
        self.assertFalse((destination/'test_predictions.csv').exists())
        self.assertEqual(report['minimum_patient_batch'], 5)
        self.assertEqual(report['history'][0]['batch_patient_counts'], [7])
        self.assertTrue(all(value > 0 for value in report['effective_loss_weights'].values()))
        self.assertIsNotNone(report['history'][0]['batches'][0]['terms']['pathway_common'])
        self.assertIsNotNone(report['history'][0]['batches'][0]['terms']['pathway_unique'])
        self.assertIsNotNone(report['history'][0]['batches'][0]['terms']['matched_common'])
        self.assertIsNotNone(report['history'][0]['batches'][0]['terms']['matched_unique'])
        self.assertGreaterEqual(report['history'][0]['batches'][0]['matching_normalized_entropy_mean'], 0.)
        self.assertLessEqual(report['history'][0]['batches'][0]['matching_max_weight_mean'], 1.)
        for field in ('patch_encoder', 'rna_encoder', 'global_path_projector', 'global_rna_projector',
                      'patch_local_projector', 'pathway_tokenizer', 'pathway_queries',
                      'local_path_projector', 'local_rna_projector', 'matching_path',
                      'matching_rna', 'interaction_head'):
            self.assertGreater(report['history'][0]['batches'][0]['gradient_norms'][field], 0., field)
        saved = torch.load(destination/'checkpoint.pt', map_location='cpu', weights_only=True)
        model = PathwayLocalSurvival(**saved['model_kwargs'])
        model.load_state_dict(saved['model_state'], strict=True)
        fitted = dict(model=model, starts=saved['starts'], stream_config=saved['stream_config'],
                      **{name:saved[name] for name in ('rna_mean', 'rna_scale', 'rna_mask', 'rna_preprocessing')})
        cohort = load_stream_manifest(manifest, 'single_view', include_test=False)
        try:
            self.assertEqual(report['cohort_input_sha256'], stream_cohort_fingerprint(cohort, 'single_view'))
            prediction = predict_pathway_model(fitted, cohort['val'])
            expected = survival_at(prediction['log_rates'], torch.tensor([1.]), saved['starts'])[:, 0]
            with (destination/'val_predictions.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row['patient_id'] for row in rows], cohort['val']['ids'])
            torch.testing.assert_close(expected, torch.tensor([float(row['survival_at_horizon']) for row in rows]))
            self.assertTrue(all(0 <= float(row['matching_max_weight_mean']) <= 1 for row in rows))
        finally:
            for split in cohort.values():
                for bag in split['bags']:
                    bag.close()
        self.assertNotEqual(self.run_cli(manifest, genes, gmt, destination, 'feature_noise').returncode, 0)
        self.assertNotEqual(self.run_cli(manifest, genes, gmt, self.root/'test', 'feature_noise', True).returncode, 0)

    def test_cli_verified_image_views_roundtrip(self):
        genes, gmt = synthetic_mapping(self.root)
        manifest = synthetic_patient_manifest(self.root, 'image_d4')
        destination = self.root/'image_result'
        result = self.run_cli(manifest, genes, gmt, destination, 'image_d4')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((destination/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['view_source'], 'image_d4')
        self.assertEqual(report['cohort_fingerprint_scheme'], 'verified-cache-stream-v1')
        self.assertFalse(report['test_evaluated'])
        self.assertEqual(report['patches']['train'], 28)
        self.assertIn('verified same-patch', report['view_source_description'])

    def test_frozen_29_pathways_work_end_to_end_with_large_gene_schema(self):
        resource = Path(__file__).parent/'resources'/'reactome_human_roots.gmt'
        rows = [line.split('\t') for line in resource.read_text(encoding='utf-8').splitlines()]
        genes = list(dict.fromkeys(gene for row in rows for gene in row[2:]))
        self.assertGreater(len(genes), 1000)
        genes_path = self.root/'real_pathway_gene_schema.txt'
        genes_path.write_text('\n'.join(genes)+'\n', encoding='utf-8')
        generator = torch.Generator().manual_seed(199)
        patients = []
        for i in range(7):
            torch.save(torch.randn(3+i, 6, generator=generator), self.root/f'resource_slide{i}.pt')
            np.save(self.root/f'resource_rna{i}.npy', torch.randn(len(genes), generator=generator).numpy())
            patients.append({'patient_id':f'resource{i}', 'split':'train' if i < 5 else 'val',
                             'wsi_pt':f'resource_slide{i}.pt', 'rna_npy':f'resource_rna{i}.npy',
                             'time':.5+i*.3, 'event':i % 2})
        manifest = self.root/'resource_patients.csv'
        write_csv(manifest, patients)
        destination = self.root/'real_pathway_synthetic_run'
        command = [sys.executable, '-B', 'train_pathway.py', '--manifest', str(manifest),
                   '--genes', str(genes_path), '--out', str(destination), '--view-source', 'feature_noise',
                   '--hidden', '4', '--common', '2', '--local-dim', '4', '--local-common', '2',
                   '--patient-batch-size', '5', '--patch-chunk-size', '3', '--epochs', '1', '--patience', '1',
                   '--starts', '0', '--horizon', '1', '--time-unit', 'months', '--endpoint', 'synthetic']
        result = subprocess.run(command, cwd=Path(__file__).parent,
                                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((destination/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(len(report['pathway_mapping']['pathways']), 29)
        self.assertEqual(report['pathway_mapping']['resource_metadata']['root_count'], 29)
        self.assertEqual(report['pathway_mapping']['ordered_gene_symbols'], genes)
        self.assertTrue(all(p['active_training_genes'] >= 1 for p in report['pathway_mapping']['pathways']))
        self.assertEqual(report['history'][0]['batches'][0]['pathway_diagnostics']['matched_common']['view_a']['pathways'], 29)
        self.assertFalse(report['test_evaluated'])

    def test_cli_explicit_test_evaluation_is_opt_in_and_preserves_checkpoint(self):
        genes, gmt = synthetic_mapping(self.root)
        manifest = synthetic_patient_manifest(self.root)
        with manifest.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        torch.save(torch.randn(5, 6), self.root/'valid_test.pt')
        np.save(self.root/'valid_test.npy', np.arange(8, dtype=np.float32))
        rows[-1].update(wsi_pt='valid_test.pt', rna_npy='valid_test.npy', time='1.8', event='1')
        write_csv(manifest, rows)
        destination = self.root/'final_test'
        result = self.run_cli(manifest, genes, gmt, destination, 'feature_noise', True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((destination/'report.json').read_text(encoding='utf-8'))
        self.assertTrue(report['test_evaluated'])
        self.assertEqual(report['evaluation_split'], 'test')
        self.assertTrue((destination/'test_predictions.csv').exists())
        self.assertFalse((destination/'val_predictions.csv').exists())
        self.assertEqual(report['patients']['test'], 1)


if __name__ == '__main__':
    unittest.main()
