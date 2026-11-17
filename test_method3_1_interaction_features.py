"""Synthetic-only checks of patient cross-fitting and interaction maps."""
import io

import pytest
import torch

from method3_1_interaction_features import GlobalResidualInteractions, _stable_moments
from test_method3_1 import generated, make_model


@pytest.fixture(scope='module')
def fitted():
    torch.set_num_threads(1)
    bags, rna, ids = generated()
    backbone = make_model()
    backbone.pretrain(bags, rna, ids, epochs=3)
    backbone.fit_probes(bags, rna, ids, folds=4)
    mappings = GlobalResidualInteractions(backbone).fit(bags, rna, ids)
    validation_bag = (torch.tensor([[0., .3], [float('nan'), float('nan')]], dtype=torch.float64),
                      torch.tensor([True, False]))
    mappings.validation_features([validation_bag], torch.tensor([[.1, -.2]], dtype=torch.float64), ['v-one'])
    return backbone, bags, rna, ids, mappings


def test_exact_outer_maps_and_oof_exclusion(fitted):
    backbone, _, _, ids, mappings = fitted
    features = mappings.training_features()
    u, g = features['global_residual_standardized'], features['rna_standardized']
    assert torch.equal(features['outer'], (u[:, :, None]*g[:, None, :]).flatten(1))
    assert torch.equal(features['tanh_outer'], (torch.tanh(u)[:, :, None]*torch.tanh(g)[:, None, :]).flatten(1))
    assert torch.equal(features['diagonal'], backbone._features(*backbone.cached)[:, -backbone.k*backbone.d:])
    assert features['outer'].shape == (len(ids), backbone.d*backbone.rna_dim)
    assert features['diagonal'].shape == (len(ids), backbone.k*backbone.d)
    assert all(torch.isfinite(t).all() for t in features.values())
    assert not features['outer'].requires_grad
    for split in mappings.probe.split_audit:
        assert not set(split['train_ids']) & set(split['validation_ids'])
    assert sorted(j for x in mappings.probe.split_audit for j in x['validation_ids']) == sorted(ids)


def test_global_mean_keeps_all_real_patches_even_when_clusters_absent(fitted):
    backbone, _, _, _, mappings = fitted
    bag = torch.tensor([[0., .3], [float('nan'), float('nan')]], dtype=torch.float64)
    masked = (bag, torch.tensor([True, False]))
    rna = torch.tensor([[.1, -.2]], dtype=torch.float64)
    features = mappings.validation_features([masked], rna, ['v-one'])
    expected = ((bag[0]-backbone.bank.patch_mean)/backbone.bank.patch_scale) @ backbone.bank.components
    assert torch.allclose(features['global_pooled'][0], expected)
    # A single patch has less than one unit of soft mass in each cluster.
    assert torch.equal(features['diagonal'], torch.zeros_like(features['diagonal']))
    assert torch.isfinite(features['outer']).all()


def test_test_validation_training_patient_isolation_and_replay(fitted):
    _, _, _, train_ids, mappings = fitted
    bags, rna, ids = generated(5, seed=191, prefix='test')
    expected = mappings.test_features(bags, rna, ids)
    with pytest.raises(ValueError, match='training'):
        mappings.test_features(bags, rna, train_ids[:5])
    with pytest.raises(ValueError, match='validation'):
        mappings.test_features([bags[0]], rna[:1], ['v-one'])
    with pytest.raises(ValueError, match='already registered'):
        mappings.validation_features(bags, rna, ids)
    stream = io.BytesIO()
    torch.save(mappings.checkpoint(), stream)
    stream.seek(0)
    restored = GlobalResidualInteractions.from_checkpoint(torch.load(stream, weights_only=True))
    actual = restored.test_features(bags, rna, ids)
    assert all(torch.equal(actual[k], expected[k]) for k in expected)
    assert all(torch.equal(restored.training_features()[k], mappings.training_features()[k]) for k in expected)
    with pytest.raises(ValueError, match='validation'):
        restored.test_features([bags[0]], rna[:1], ['v-one'])


def test_training_cache_alignment_and_constant_scale(fitted):
    backbone, bags, rna, ids, _ = fitted
    changed_rna = rna.clone(); changed_rna[0, 0] += 1
    with pytest.raises(ValueError, match='RNA differs'):
        GlobalResidualInteractions(backbone).fit(bags, changed_rna, ids)
    changed_bags = [x.clone() for x in bags]; changed_bags[0][0, 0] += 1
    with pytest.raises(ValueError, match='Bags differ'):
        GlobalResidualInteractions(backbone).fit(changed_bags, rna, ids)
    _, scale = _stable_moments(torch.tensor([[1., 3.], [1., 4.]], dtype=torch.float64))
    assert scale.tolist() == [1., .5]
