"""Synthetic equivalence and direct interaction-path checks."""
import io

import pytest
import torch

from method3_1_active_head import ActiveHierarchicalSurvivalHead
from method3_1_batched_head import BatchedHierarchicalSurvivalHead
from method3_1_hierarchical_head import HierarchicalSurvivalHead
from test_method3_1_hierarchical_head import data


torch.set_num_threads(1)
GRID = ((0., None, None), (0., .1, None), (0., .1, .1), (.1, 1., .01))
FORCED = ((0., .1, .1), (.1, 1., .01))


@pytest.mark.parametrize('mode', ['shared', 'per_bin'])
def test_zero_min_epoch_is_identical_to_original_batched_trajectories(mode):
    train, ty, tc, ti = data(35, 7201, 'train')
    val, vy, vc, vi = data(25, 7202, 'validation')
    test, _, _, qi = data(23, 7203, 'test')
    kwargs = dict(candidates=GRID, epochs=25, lr=.03, eval_every=5)
    reference = BatchedHierarchicalSurvivalHead(bins=3, interaction_mode=mode)
    actual = ActiveHierarchicalSurvivalHead(bins=3, interaction_mode=mode)
    ref_out = reference.fit(train, ty, tc, ti, val, vy, vc, vi, **kwargs)
    act_out = actual.fit(train, ty, tc, ti, val, vy, vc, vi, **kwargs)
    assert act_out['epoch'] == ref_out['epoch']
    assert act_out['penalties'] == ref_out['penalties']
    assert act_out['epoch_zero_included'] is True
    for ref, act in zip(ref_out['candidates'], act_out['candidates']):
        assert ref['epoch'] == act['epoch']
        assert torch.equal(torch.tensor(ref['validation_curve'], dtype=torch.float64),
                           torch.tensor(act['validation_curve'], dtype=torch.float64))
    assert torch.equal(reference.predict(test, qi), actual.predict(test, qi))


@pytest.mark.parametrize('mode', ['shared', 'per_bin'])
def test_positive_epoch_requires_enabled_interaction_and_matches_curve(mode):
    train, ty, tc, ti = data(35, 7211, 'train')
    val, vy, vc, vi = data(25, 7212, 'validation')
    head = ActiveHierarchicalSurvivalHead(3, mode, min_select_epoch=5, require_interaction=True)
    result = head.fit(train, ty, tc, ti, val, vy, vc, vi, candidates=FORCED,
                      epochs=25, eval_every=5)
    assert result['epoch'] >= 5 and result['epoch_zero_included'] is False
    assert result['interaction_required'] is True and result['penalties'][2] is not None
    for row in result['candidates']:
        eligible = [(e, v) for e, v in row['validation_curve'] if e >= 5]
        expected = min(eligible, key=lambda x: x[1])
        assert row['epoch'] == expected[0]
        assert row['val_nll'] == expected[1]
    with pytest.raises(ValueError, match='enabled interaction'):
        ActiveHierarchicalSurvivalHead(3, mode, require_interaction=True).fit(
            train, ty, tc, ti, val, vy, vc, vi, candidates=GRID, epochs=5)
    empty = dict(train, interaction=train['interaction'][:, :0])
    empty_val = dict(val, interaction=val['interaction'][:, :0])
    with pytest.raises(ValueError, match='nonempty'):
        ActiveHierarchicalSurvivalHead(3, mode, require_interaction=True).fit(
            empty, ty, tc, ti, empty_val, vy, vc, vi, candidates=FORCED, epochs=5)


def test_require_enabled_does_not_invent_effect_for_constant_interaction():
    train, ty, tc, ti = data(30, 7221, 'train')
    val, vy, vc, vi = data(24, 7222, 'validation')
    train['interaction'].zero_(); val['interaction'].zero_()
    head = ActiveHierarchicalSurvivalHead(3, min_select_epoch=5, require_interaction=True)
    result = head.fit(train, ty, tc, ti, val, vy, vc, vi, candidates=FORCED,
                      epochs=5, eval_every=5)
    assert result['epoch'] == 5 and torch.count_nonzero(head.weights['interaction']) == 0
    assert head.survival_interaction_gradient_norm(train, ty, tc, ti) == 0.


@pytest.mark.parametrize('censorship', [torch.tensor([1, 1, 0, 0]), torch.zeros(4), torch.ones(4)])
def test_direct_survival_gradient_is_finite_and_zero_regularization_independent(censorship):
    features = {'clr': torch.zeros(4, 1, dtype=torch.float64),
                'other': torch.zeros(4, 0, dtype=torch.float64),
                'interaction': torch.tensor([[-2.], [-1.], [1.], [2.]], dtype=torch.float64)}
    labels, ids, vi = torch.zeros(4, dtype=torch.long), ['t0', 't1', 't2', 't3'], ['v0', 'v1', 'v2', 'v3']
    head = ActiveHierarchicalSurvivalHead(1, 'shared', min_select_epoch=5, require_interaction=True)
    head.fit(features, labels, censorship, ids, features, labels, censorship, vi,
             candidates=((0., None, .1),), epochs=5)
    norm = head.survival_interaction_gradient_norm(features, labels, censorship, ids)
    assert torch.isfinite(torch.tensor(norm))
    # Mixed censoring is informative, so NLL alone has a real gradient.
    if torch.equal(censorship, torch.tensor([1, 1, 0, 0])):
        assert norm > 0 and float(head.weights['interaction'].abs().sum()) > 0
    with pytest.raises(ValueError, match='training patient order'):
        head.survival_interaction_gradient_norm(features, labels, censorship, list(reversed(ids)))


def test_checkpoint_restores_inference_and_training_policy():
    train, ty, tc, ti = data(30, 7231, 'train')
    val, vy, vc, vi = data(24, 7232, 'validation')
    test, _, _, qi = data(24, 7233, 'test')
    head = ActiveHierarchicalSurvivalHead(3, 'shared', min_select_epoch=5, require_interaction=True)
    head.fit(train, ty, tc, ti, val, vy, vc, vi, candidates=FORCED, epochs=10)
    expected = head.predict(test, qi)
    stream = io.BytesIO(); torch.save(head.checkpoint(), stream); stream.seek(0)
    snapshot = torch.load(stream, weights_only=True)
    restored = ActiveHierarchicalSurvivalHead.from_checkpoint(snapshot)
    legacy = HierarchicalSurvivalHead.from_checkpoint(snapshot)
    assert restored.min_select_epoch == 5 and restored.require_interaction is True
    assert torch.equal(restored.predict(test, qi), expected)
    assert torch.equal(legacy.predict(test, qi), expected)
    with pytest.raises(ValueError, match='disjoint'):
        restored.predict(val, vi)
