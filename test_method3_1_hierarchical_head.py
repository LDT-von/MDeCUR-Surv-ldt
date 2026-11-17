"""Tests use only synthetic feature blocks and synthetic survival labels."""
import io

import pytest
import torch

from method3_1_hierarchical_head import HierarchicalSurvivalHead


torch.set_num_threads(1)


def data(n, seed, prefix):
    gen = torch.Generator().manual_seed(seed)
    clr0 = torch.randn(n, 1, generator=gen, dtype=torch.float64)
    other = torch.randn(n, 2, generator=gen, dtype=torch.float64)
    interaction = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    blocks = {'clr': torch.cat((clr0, -clr0), 1), 'other': other, 'interaction': interaction}
    censor = torch.randint(0, 2, (n,), generator=gen)
    bins = torch.randint(0, 3, (n,), generator=gen)
    return blocks, bins, censor, [f'{prefix}-{i}' for i in range(n)]


def fit(mode='per_bin', candidates=((0., None, None),)):
    tr, ty, tc, ti = data(32, 101, 'train')
    va, vy, vc, vi = data(20, 102, 'validation')
    model = HierarchicalSurvivalHead(bins=3, interaction_mode=mode)
    selection = model.fit(tr, ty, tc, ti, va, vy, vc, vi, candidates=candidates, epochs=10, eval_every=5)
    return model, selection, (tr, ty, tc, ti), (va, vy, vc, vi)


def test_empty_optional_blocks_and_epoch_zero():
    blocks = {'clr': torch.zeros(4, 1), 'other': torch.zeros(4, 0), 'interaction': torch.zeros(4, 0)}
    y = torch.zeros(4, dtype=torch.long)
    c = torch.tensor([0, 0, 1, 1])
    model = HierarchicalSurvivalHead(bins=1)
    result = model.fit(blocks, y, c, ['a', 'b', 'c', 'd'], blocks, y, c,
                       ['e', 'f', 'g', 'h'], candidates=((0., None, None),), epochs=10)
    assert result['epoch_zero_included'] and result['epoch'] == 0
    assert result['grad_norms'] is None
    assert torch.isfinite(model.predict(blocks, ['q', 'r', 's', 't'])).all()


def test_closed_blocks_have_exact_zero_weights_and_no_effect():
    model, result, _, _ = fit()
    assert result['penalties'] == (0., None, None)
    assert torch.count_nonzero(model.weights['other']) == 0
    assert torch.count_nonzero(model.weights['interaction']) == 0
    if result['grad_norms'] is not None:
        assert result['grad_norms']['other'] is None and result['grad_norms']['interaction'] is None
    x, _, _, ids = data(15, 103, 'test')
    changed = {k: v.clone() for k, v in x.items()}
    changed['other'] += 1000
    changed['interaction'] -= 3000
    assert torch.equal(model.predict(x, ids), model.predict(changed, ids))


@pytest.mark.parametrize('mode', ['per_bin', 'shared'])
def test_checkpoint_is_exact_isolated_and_rejects_training_and_validation(mode):
    model, _, (tr, _, _, ti), (va, _, _, vi) = fit(mode, ((0., .1, .1),))
    x, _, _, ids = data(15, 104, 'test')
    before = model.predict(x, ids)
    buffer = io.BytesIO()
    snapshot = model.checkpoint()
    torch.save(snapshot, buffer)
    snapshot['weights']['clr'].add_(999.)
    assert torch.equal(before, model.predict(x, ids))
    buffer.seek(0)
    restored = HierarchicalSurvivalHead.from_checkpoint(torch.load(buffer, weights_only=True))
    assert torch.equal(before, restored.predict(x, ids))
    with pytest.raises(ValueError, match='disjoint'):
        restored.predict(tr, ti)
    with pytest.raises(ValueError, match='disjoint'):
        restored.predict(va, vi)
    assert restored.weights['interaction'].shape[0] == (1 if mode == 'shared' else 3)


def test_interaction_weight_receives_nonzero_gradient_on_synthetic_known_signal():
    def make(n, seed, prefix):
        gen = torch.Generator().manual_seed(seed)
        feature = torch.randn(n, 1, generator=gen, dtype=torch.float64)
        observed = (torch.rand(n, generator=gen, dtype=torch.float64) < torch.sigmoid(3*feature[:, 0])).long()
        return ({'clr': torch.zeros(n, 1), 'other': torch.zeros(n, 0), 'interaction': feature},
                torch.zeros(n, dtype=torch.long), 1-observed, [f'{prefix}-{i}' for i in range(n)])
    tr, y, c, ids = make(160, 3, 'tr')
    va, vy, vc, vids = make(160, 4, 'va')
    model = HierarchicalSurvivalHead(bins=1, interaction_mode='shared')
    result = model.fit(tr, y, c, ids, va, vy, vc, vids, candidates=((0., None, .01),), epochs=60)
    assert result['epoch'] > 0 and result['grad_norms']['interaction'] > 0
    assert float(model.weights['interaction'].abs().sum()) > 0


def test_survival_nll_alone_drives_interaction_gradient():
    # At zero coefficients the L2 gradient vanishes, so this explicitly
    # checks the censored likelihood's direct path to the interaction block.
    from cancer_morphid_model import discrete_nll
    from method3_1_hierarchical_head import _Candidate
    block = {'clr': torch.zeros(4, 1, dtype=torch.float64),
             'other': torch.zeros(4, 0, dtype=torch.float64),
             'interaction': torch.tensor([[-2.], [-1.], [1.], [2.]], dtype=torch.float64)}
    candidate = _Candidate({'clr': 1, 'other': 0, 'interaction': 1}, 1, 'shared',
                           (0., None, .1), torch.zeros(1, dtype=torch.float64))
    labels = torch.zeros(4, dtype=torch.long)
    censorship = torch.tensor([1, 1, 0, 0])
    discrete_nll(candidate(block), labels, censorship, alpha=0.).backward()
    assert candidate.interaction.grad is not None
    assert torch.isfinite(candidate.interaction.grad).all()
    assert float(candidate.interaction.grad.abs().sum()) > 0


def test_patient_disjointness_and_only_training_scaling():
    tr, ty, tc, ti = data(24, 201, 'train')
    va, vy, vc, vi = data(20, 202, 'validation')
    with pytest.raises(ValueError, match='overlap'):
        HierarchicalSurvivalHead(bins=3).fit(tr, ty, tc, ti, va, vy, vc, ti[:20],
                                               candidates=((0., None, None),), epochs=5)
    m1, m2 = HierarchicalSurvivalHead(bins=3), HierarchicalSurvivalHead(bins=3)
    m1.fit(tr, ty, tc, ti, va, vy, vc, vi, candidates=((0., None, None),), epochs=5)
    changed = {k: v+100 for k, v in va.items()}
    m2.fit(tr, ty, tc, ti, changed, vy, vc, vi, candidates=((0., None, None),), epochs=5)
    assert all(torch.equal(m1.mean[k], m2.mean[k]) and torch.equal(m1.scale[k], m2.scale[k]) for k in tr)


def test_invalid_grid_and_dimension_mismatch_are_rejected():
    tr, ty, tc, ti = data(24, 301, 'train')
    va, vy, vc, vi = data(20, 302, 'validation')
    with pytest.raises(ValueError, match='cannot be hard-disabled'):
        HierarchicalSurvivalHead(bins=3).fit(tr, ty, tc, ti, va, vy, vc, vi,
                                               candidates=((None, .1, .1),), epochs=5)
    va['other'] = torch.zeros(20, 9)
    with pytest.raises(ValueError, match='dimensions'):
        HierarchicalSurvivalHead(bins=3).fit(tr, ty, tc, ti, va, vy, vc, vi,
                                               candidates=((.1, .1, .1),), epochs=5)
