"""Synthetic integration checks; no patient data or server execution."""
from copy import deepcopy

import pytest
import torch

from cancer_dense_independent import IndependentRegionDenseSurvival
from cancer_dense_model import DensePathwayConfig, DensePathwaySurvival, dense_pathway_objective
from cancer_parity import discrete_nll
from cancer_pathway_stream import LOSS_NAMES, pathway_stream_forward


class GeneratedBag:
    view_source = 'single_view'

    def __init__(self, patient_id, values):
        self.patient_id = patient_id
        self.values = values
        self.n_patches, self.feature_dim = values.shape
        self.padding = 0

    def blocks(self, size):
        for start in range(0, self.n_patches, size):
            yield (self.patient_id, start, min(start + size, self.n_patches))

    def read_block(self, token, names):
        if tuple(names) != ('features_reference',):
            raise ValueError('Generated fixture only provides reference features')
        return (self.values[token[1]:token[2]],)


def pair():
    torch.manual_seed(141)
    args = (6, 8, [[0, 1], [2, 3], [4, 5], [6, 7]])
    original = DensePathwaySurvival(*args, bins=4)
    candidate = IndependentRegionDenseSurvival(*args, bins=4)
    candidate.initialize_from_dense(original)
    return original, candidate


def data(n=20):
    generator = torch.Generator().manual_seed(232)
    bags = [GeneratedBag(str(i), torch.randn(13, 6, generator=generator)) for i in range(n)]
    rna = torch.randn(n, 8, generator=generator)
    return bags, rna


def test_candidate_has_same_parameters_and_matches_original_on_same_inputs():
    original, candidate = pair()
    original.eval()
    candidate.eval()
    assert set(candidate.state_dict()) - set(original.state_dict()) == {'_extra_state'}
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in candidate.parameters())
    with pytest.raises(RuntimeError, match='Missing key'):
        candidate.load_state_dict(original.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        original.load_state_dict(candidate.state_dict(), strict=True)
    torch.manual_seed(313)
    p, g = torch.randn(3, 16), torch.randn(3, 16)
    regions, tokens = torch.randn(3, 4, 16), torch.randn(3, 4, 16)
    old = original.prediction(p, g, regions, tokens)
    new = candidate.prediction_with_pathology_regions(p, g, regions, tokens, regions)
    torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
    torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='pathology-region streaming hook'):
        candidate.prediction(p, g, regions, tokens)


def test_independent_regions_ignore_same_patient_rna_and_original_default_is_unchanged():
    original, candidate = pair()
    original.eval()
    candidate.eval()
    bags, rna = data(3)
    out_a = pathway_stream_forward(candidate, bags, rna, 5, checkpoint_chunks=False)
    out_b = pathway_stream_forward(candidate, bags, -2 * rna + .3, 5, checkpoint_chunks=False)
    torch.testing.assert_close(out_a['independent_pathology_regions'],
                               out_b['independent_pathology_regions'], rtol=0, atol=0)
    assert not torch.allclose(out_a['matched_path'], out_b['matched_path'])
    old = pathway_stream_forward(original, bags, rna, 5, checkpoint_chunks=False)
    assert 'independent_pathology_regions' not in old
    assert original.adaptation_metadata()['synergy_estimator'] is False
    assert candidate.adaptation_metadata()['synergy_estimator'] is False


def test_independent_pathology_regions_exclude_padding_even_in_padding_only_chunks():
    _, candidate = pair()
    candidate.eval()
    bags, rna = data(1)
    real = bags[0].values
    padded = GeneratedBag('padded', torch.cat((real, torch.zeros(14, 6)), dim=0))
    padded.padding = 14
    with torch.no_grad():
        a = pathway_stream_forward(candidate, bags, rna, 4, checkpoint_chunks=False)
        b = pathway_stream_forward(candidate, [padded], rna, 4, checkpoint_chunks=False)
    torch.testing.assert_close(a['independent_pathology_regions'],
                               b['independent_pathology_regions'], rtol=0, atol=0)


def test_survival_gradient_reaches_pathology_query_via_new_interaction_only():
    _, model = pair()
    model.train()
    bags, rna = data(6)
    output = pathway_stream_forward(model, bags, rna, 5, checkpoint_chunks=False)
    labels = torch.arange(6) % 4
    censorship = (torch.arange(6) % 3 == 0).float()
    loss = discrete_nll(output['log_rates'], labels, censorship, alpha=.5)
    grad, = torch.autograd.grad(loss, model.pathway_queries)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    # The same prediction path with cooperation disabled cannot reach the
    # pathology-only query, even though the separate loss branch still can.
    model.interaction_strength = 0.
    output_off = pathway_stream_forward(model, bags, rna, 5, checkpoint_chunks=False)
    loss_off = discrete_nll(output_off['log_rates'], labels, censorship, alpha=.5)
    off_grad, = torch.autograd.grad(loss_off, model.pathway_queries, allow_unused=True)
    assert off_grad is None or torch.count_nonzero(off_grad) == 0


def test_candidate_checkpointed_and_direct_stream_gradients_match():
    _, reference = pair()
    left, right = deepcopy(reference).train(), deepcopy(reference).train()
    bags, rna = data(3)
    results = []
    for model, use_checkpoint in ((left, False), (right, True)):
        output = pathway_stream_forward(model, bags, rna, 4,
                                       checkpoint_chunks=use_checkpoint)['log_rates']
        output.square().sum().backward()
        results.append(output.detach())
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    for (name_a, param_a), (name_b, param_b) in zip(left.named_parameters(), right.named_parameters()):
        assert name_a == name_b
        assert (param_a.grad is None) == (param_b.grad is None)
        if param_a.grad is not None:
            torch.testing.assert_close(param_a.grad, param_b.grad, rtol=0, atol=0)
    torch.testing.assert_close(left.token_cooperation.path_reference,
                               right.token_cooperation.path_reference, rtol=0, atol=0)


def test_matched_original_and_candidate_each_take_a_full_objective_step():
    original, candidate = pair()
    bags, rna = data()
    batch = {'bags': bags, 'rna': rna, 'rna_mask': torch.ones(8, dtype=torch.bool),
             'label': torch.arange(len(bags)) % 4,
             'censorship': (torch.arange(len(bags)) % 3 == 0).float()}
    cfg = DensePathwayConfig()
    def survival(logits, inputs):
        return discrete_nll(logits, inputs['label'], inputs['censorship'], alpha=.5)
    initial = deepcopy(original.state_dict())
    for name, value in candidate.state_dict().items():
        if name != '_extra_state':
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    rng = torch.random.get_rng_state()
    for model in (original, candidate):
        torch.random.set_rng_state(rng)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss, stats, terms = dense_pathway_objective(
            model, batch, cfg, chunk_size=8, view_seed=51, epoch=1,
            survival_loss=survival, return_components=True)
        assert torch.isfinite(loss) and len(terms) == 16
        assert all(stats['auxiliary_enabled'].values())
        optimizer.zero_grad()
        loss.backward()
        assert sum(p.grad.abs().sum() for p in model.token_cooperation.parameters()
                   if p.grad is not None) > 0
        optimizer.step()
        assert not torch.equal(model.token_cooperation.head.weight,
                               initial['token_cooperation.head.weight'])


def test_full_15_term_objective_trainable_and_checkpoint_replays():
    _, model = pair()
    model.train()
    bags, rna = data()
    cfg = DensePathwayConfig()
    batch = {'bags': bags, 'rna': rna, 'rna_mask': torch.ones(8, dtype=torch.bool),
             'label': torch.arange(len(bags)) % 4,
             'censorship': (torch.arange(len(bags)) % 3 == 0).float()}
    def survival(logits, batch):
        return discrete_nll(logits, batch['label'], batch['censorship'], alpha=.5)
    loss, stats, components = dense_pathway_objective(
        model, batch, cfg, chunk_size=8, view_seed=77, epoch=1,
        survival_loss=survival, return_components=True)
    assert torch.isfinite(loss)
    assert all(name in components and stats['auxiliary_enabled'][name] for name in LOSS_NAMES)
    assert all(name in components and stats['auxiliary_enabled'][name] for name in
               ('dense_path_within', 'dense_rna_within', 'dense_common', 'dense_unique'))
    loss.backward()
    assert model.pathway_queries.grad is not None and torch.isfinite(model.pathway_queries.grad).all()
    assert model.pathway_queries.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in model.token_cooperation.parameters() if p.grad is not None) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    before = deepcopy(model.state_dict())
    with torch.no_grad():
        expected = pathway_stream_forward(model, bags[:3], rna[:3], 5, checkpoint_chunks=False)['log_rates']
    _, replay = pair()
    replay.load_state_dict(before, strict=True)
    replay.eval()
    with torch.no_grad():
        actual = pathway_stream_forward(replay, bags[:3], rna[:3], 5, checkpoint_chunks=False)['log_rates']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        if name == '_extra_state':
            assert value == before[name]
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
