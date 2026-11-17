"""Synthetic-only trajectory equivalence for the batched candidate head."""
import io

import pytest
import torch

import method3_1_hierarchical_head as reference_module
from cancer_parity import discrete_nll
from method3_1_batched_head import BatchedHierarchicalSurvivalHead, batched_discrete_nll
from method3_1_hierarchical_head import HierarchicalSurvivalHead
from test_method3_1_hierarchical_head import data


torch.set_num_threads(1)
GRID = ((0., None, None), (0., .1, None), (0., .1, .1), (.1, 1., .01))


@pytest.mark.parametrize('censorship', [torch.tensor([0, 1, 0, 1, 0]),
                                       torch.ones(5), torch.zeros(5)])
def test_batched_nll_matches_patientwise_loss_and_gradient(censorship):
    gen = torch.Generator().manual_seed(35)
    original = torch.randn(4, 5, 3, generator=gen, dtype=torch.float64)*5
    labels = torch.tensor([[0], [1], [2], [2], [1]])
    a, b = original.clone().requires_grad_(), original.clone().requires_grad_()
    actual = batched_discrete_nll(a, labels, censorship[:, None])
    expected = torch.stack([discrete_nll(row, labels, censorship[:, None], alpha=0.) for row in b])
    assert torch.allclose(actual, expected, rtol=0., atol=1e-12)
    actual.sum().backward(); expected.sum().backward()
    assert torch.allclose(a.grad, b.grad, rtol=0., atol=1e-12)


def test_batched_nll_checks_all_candidates_for_invalid_values():
    logits = torch.zeros(2, 3, 2, dtype=torch.float64)
    logits[1, 1, 1] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        batched_discrete_nll(logits, torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError):
        batched_discrete_nll(torch.zeros(2, 3, 2, dtype=torch.float64), torch.zeros(3), torch.full((3,), 2))


@pytest.mark.parametrize('mode', ['per_bin', 'shared'])
def test_four_candidate_full_validation_trajectories_and_logits_match(monkeypatch, mode):
    tr, ty, tc, ti = data(32, 501, 'train')
    va, vy, vc, vi = data(20, 502, 'validation')
    te, _, _, ids = data(23, 503, 'test')
    kwargs = {'epochs': 25, 'lr': .03, 'eval_every': 5}
    batched = BatchedHierarchicalSurvivalHead(bins=3, interaction_mode=mode)
    out = batched.fit(tr, ty, tc, ti, va, vy, vc, vi, candidates=GRID, **kwargs)
    normal_nll = reference_module.discrete_nll
    for index, candidate in enumerate(GRID):
        calls = []
        def logging_nll(logits, labels, censorship, **settings):
            result = normal_nll(logits, labels, censorship, **settings)
            if len(logits) == len(vi) and not torch.is_grad_enabled():
                calls.append(float(result))
            return result
        with monkeypatch.context() as mp:
            mp.setattr(reference_module, 'discrete_nll', logging_nll)
            ref = HierarchicalSurvivalHead(bins=3, interaction_mode=mode)
            single = ref.fit(tr, ty, tc, ti, va, vy, vc, vi, candidates=(candidate,), **kwargs)
        curve = out['candidates'][index]['validation_curve']
        assert [epoch for epoch, _ in curve] == [0, 5, 10, 15, 20, 25]
        assert torch.allclose(torch.tensor([v for _, v in curve], dtype=torch.float64),
                              torch.tensor(calls, dtype=torch.float64), atol=1e-8, rtol=0.)
        assert out['candidates'][index]['epoch'] == single['epoch']
        assert abs(out['candidates'][index]['val_nll']-single['val_nll']) < 1e-8
        x = batched._normalized(te)
        state = batched.candidate_states[index]
        pred = (x['clr']@state['weights']['clr'].T + x['other']@state['weights']['other'].T
                + x['interaction']@state['weights']['interaction'].T + state['bias'])
        assert torch.allclose(pred, ref.predict(te, ids), atol=1e-8, rtol=0.)
        for block, penalty in zip(('clr', 'other', 'interaction'), candidate):
            if penalty is None:
                assert torch.count_nonzero(state['weights'][block]) == 0
    global_ref = HierarchicalSurvivalHead(bins=3, interaction_mode=mode)
    ref_out = global_ref.fit(tr, ty, tc, ti, va, vy, vc, vi, candidates=GRID, **kwargs)
    assert out['penalties'] == ref_out['penalties'] and out['epoch'] == ref_out['epoch']
    assert torch.allclose(batched.predict(te, ids), global_ref.predict(te, ids), atol=1e-8, rtol=0.)
    buffer = io.BytesIO()
    torch.save(batched.checkpoint(), buffer)
    buffer.seek(0)
    restored = HierarchicalSurvivalHead.from_checkpoint(torch.load(buffer, weights_only=True))
    assert torch.equal(restored.predict(te, ids), batched.predict(te, ids))


def test_batched_head_handles_empty_blocks_and_rejects_leakage():
    blocks = {'clr': torch.zeros(4, 1), 'other': torch.zeros(4, 0), 'interaction': torch.zeros(4, 0)}
    y = torch.zeros(4, dtype=torch.long)
    c = torch.tensor([0, 0, 1, 1])
    model = BatchedHierarchicalSurvivalHead(bins=1)
    out = model.fit(blocks, y, c, ['a', 'b', 'c', 'd'], blocks, y, c,
                    ['e', 'f', 'g', 'h'], candidates=((0., None, None), (0., .1, .1)), epochs=5)
    assert out['epoch'] == 0 and out['penalties'] == (0., None, None)
    with pytest.raises(ValueError, match='disjoint'):
        model.predict(blocks, ['a', 'b', 'c', 'd'])
