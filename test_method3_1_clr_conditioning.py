"""Small generated-only tests for outcome-free CLR conditioning."""
import io

import pytest
import torch

from method3_1_clr_conditioning import CLRConditioner


def synthetic(n, seed, prefix):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=gen, dtype=torch.float64)
    clr = torch.cat((x, -x), dim=1)
    noise = .01*torch.randn(n, 2, generator=gen, dtype=torch.float64)
    other = torch.cat((torch.sin(x), x.square()), dim=1)+noise
    return clr, other, [f'{prefix}-{i:03}' for i in range(n)]


@pytest.fixture(scope='module')
def fitted():
    torch.set_num_threads(1)
    x, y, ids = synthetic(80, 1, 'train')
    model = CLRConditioner(alpha=.001).fit(x, y, ids, folds=4, seed=9)
    return model, x, y, ids


def test_training_is_patient_oof_and_expanded_clr_only_uses_training_stats(fitted):
    model, x, y, ids = fitted
    feature = model.training_features(ids)
    assert feature['clr'].shape == (len(ids), 6)
    assert torch.equal(feature['clr'][:, :2], x)
    z = (x-model.clr_mean)/model.clr_scale
    assert torch.equal(feature['clr'][:, 2:4], z.tanh())
    assert torch.equal(feature['clr'][:, 4:], z.square())
    assert torch.equal(feature['other'], y-model.probe.oof_predictions)
    for audit in model.metadata['split_audit']:
        assert not set(audit['train_ids']) & set(audit['validation_ids'])
        assert set(audit['train_ids']) | set(audit['validation_ids']) == set(ids)
    assert model.metadata['survival_labels_used'] is False
    assert model.metadata['residual_proves_independence'] is False
    with pytest.raises(ValueError, match='order'):
        model.training_features(ids[::-1])


def test_each_probe_uses_its_own_training_fold_for_preprocessing(fitted):
    model, x, y, _ = fitted
    for probe, audit in zip(model.probe.models, model.metadata['split_audit']):
        train = torch.tensor(audit['train_indices'], dtype=torch.long)
        held = torch.tensor(audit['validation_indices'], dtype=torch.long)
        assert torch.allclose(probe.x_mean, x[train].mean(0), atol=1e-12, rtol=0)
        assert torch.allclose(probe.target_mean, y[train].mean(0), atol=1e-12, rtol=0)
        expected_scale = x[train].std(0, unbiased=False)
        expected_scale = torch.where(expected_scale >= probe.min_scale, expected_scale, torch.ones_like(expected_scale))
        assert torch.allclose(probe.x_scale, expected_scale, atol=1e-12, rtol=0)
        assert torch.equal(model.probe.oof_predictions[held], probe.predict(x[held]))


def test_held_out_targets_cannot_affect_own_fold_prediction():
    x, y, ids = synthetic(32, 20, 'oof-check')
    first = CLRConditioner(alpha=.01).fit(x, y, ids, folds=4, seed=7)
    held = torch.tensor(first.metadata['split_audit'][0]['validation_indices'], dtype=torch.long)
    changed = y.clone()
    changed[held] += 2000
    second = CLRConditioner(alpha=.01).fit(x, changed, ids, folds=4, seed=7)
    # OOF is conditional on the common outer-training feature basis, but the
    # held-out targets themselves cannot be used by their fold estimator.
    assert torch.equal(first.probe.oof_predictions[held], second.probe.oof_predictions[held])


def test_conditional_predictor_learns_nonlinear_known_mean(fitted):
    model, _, y, ids = fitted
    residue = model.training_features(ids)['other']
    target = y-y.mean(0)
    oof_r2 = 1-residue.square().sum(0)/target.square().sum(0)
    assert oof_r2[0] > .85
    # Finite-sample KRR may leave substantial nonlinear tail dependence.
    # Keep this as a counterexample to any independence claim.
    assert 0. < oof_r2[1] < .85


def test_validation_test_disjoint_and_fitted_state_unchanged(fitted):
    model, _, _, ids = fitted
    val_x, val_y, val_ids = synthetic(20, 3, 'val')
    test_x, test_y, test_ids = synthetic(18, 4, 'test')
    mean_before, scale_before = model.clr_mean.clone(), model.clr_scale.clone()
    expected_probe = model.probe.predict(val_x)
    val = model.validation_features(val_x, val_y, val_ids)
    assert torch.equal(val['other'], val_y-expected_probe)
    test = model.test_features(test_x, test_y, test_ids)
    assert torch.isfinite(test['clr']).all() and torch.isfinite(test['other']).all()
    assert torch.equal(model.clr_mean, mean_before) and torch.equal(model.clr_scale, scale_before)
    with pytest.raises(ValueError, match='training'):
        model.test_features(test_x, test_y, ids[:len(test_ids)])
    with pytest.raises(ValueError, match='validation'):
        model.test_features(val_x, val_y, val_ids)
    with pytest.raises(ValueError, match='different order'):
        model.validation_features(val_x, val_y, val_ids[::-1])


def test_roundtrip_replays_fresh_and_training_with_no_aliasing(fitted):
    model, _, _, train_ids = fitted
    val_x, val_y, val_ids = synthetic(20, 3, 'val')
    model.validation_features(val_x, val_y, val_ids)
    test_x, test_y, test_ids = synthetic(10, 11, 'fresh')
    expected = model.test_features(test_x, test_y, test_ids)
    expected_train = model.training_features(train_ids)
    snapshot = model.checkpoint()
    stream = io.BytesIO()
    torch.save(snapshot, stream)
    snapshot['clr_mean'].add_(1000)
    assert torch.equal(model.training_features(train_ids)['clr'], expected_train['clr'])
    stream.seek(0)
    restored = CLRConditioner.from_checkpoint(torch.load(stream, weights_only=True))
    actual = restored.test_features(test_x, test_y, test_ids)
    for key in ('clr', 'other'):
        assert torch.equal(actual[key], expected[key])
        assert torch.equal(restored.training_features(train_ids)[key], expected_train[key])
    with pytest.raises(ValueError, match='validation'):
        restored.test_features(val_x, val_y, val_ids)


def test_constants_and_invalid_shapes_are_explicit():
    ids = [f't-{i}' for i in range(20)]
    x = torch.zeros(20, 2, dtype=torch.float64)
    y = torch.arange(20, dtype=torch.float64).unsqueeze(1)
    model = CLRConditioner().fit(x, y, ids)
    assert torch.equal(model.clr_scale, torch.ones(2, dtype=torch.float64))
    assert torch.isfinite(model.training_features(ids)['clr']).all()
    with pytest.raises(ValueError, match='feature count'):
        model.validation_features(torch.zeros(10, 3), torch.zeros(10, 1), [f'v-{i}' for i in range(10)])
    with pytest.raises(ValueError, match='distinct'):
        CLRConditioner().fit(x, y, ['same']*20)
