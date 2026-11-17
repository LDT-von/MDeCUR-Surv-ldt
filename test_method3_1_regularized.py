"""Generated-only tests of selection, ablations and inference restoration."""
import io

import pytest
import torch

from method3_1_regularized import RegularizedMethod31, training_hazard_logits
from test_method3_1 import generated, make_model


def test_initial_hazard_uses_training_risk_sets_and_censoring():
    y, c = torch.tensor([0, 1, 2]), torch.tensor([0, 1, 0])
    actual = torch.sigmoid(training_hazard_logits(y, c, 3))
    expected = torch.tensor([1.5/4, .5/3, 1.5/2], dtype=torch.float64)
    assert torch.allclose(actual, expected)
    assert torch.isfinite(training_hazard_logits(torch.tensor([0, 0]), torch.ones(2), 4)).all()


@pytest.fixture(scope='module')
def candidate():
    torch.set_num_threads(1)
    bags, rna, ids = generated()
    val_bags, val_rna, val_ids = generated(16, 71, 'validation')
    b = make_model()
    b.pretrain(bags, rna, ids, epochs=3)
    b.fit_probes(bags, rna, ids, folds=4)
    return b, ids, val_bags, val_rna, val_ids


def test_masks_residual_and_interaction_without_removing_pathology(candidate):
    b, *_ = candidate
    full, no_i, no_u = (RegularizedMethod31(b, arm=a) for a in RegularizedMethod31.ARMS)
    kd = b.k*b.d
    uslice = slice(b.rna_dim+kd, b.rna_dim+2*kd)
    assert full.mask.eq(1).all()
    assert no_i.mask[uslice].eq(1).all() and no_i.mask[-kd:].eq(0).all()
    assert no_u.mask[uslice].eq(0).all() and no_u.mask[-kd:].eq(0).all()
    assert no_u.mask[b.rna_dim+2*kd:-kd].eq(1).all()


def test_validation_selection_and_complete_checkpoint(candidate):
    b, ids, val_bags, val_rna, val_ids = candidate
    arm = RegularizedMethod31(b, arm='no_interaction')
    details = arm.fit(torch.arange(len(ids)) % 3, torch.arange(len(ids)) % 2, ids,
                      val_bags, val_rna, val_ids, torch.arange(len(val_ids)) % 3,
                      torch.arange(len(val_ids)) % 2, penalties=(.1, 1.), epochs=10, eval_every=5)
    assert details['penalty'] in (.1, 1.) and details['epoch'] in (5, 10)
    assert torch.equal(arm.weight[:, -b.k*b.d:], torch.zeros_like(arm.weight[:, -b.k*b.d:]))
    original = arm._train_features()
    modified = original.clone()
    modified[:, -b.k*b.d:] += 1000
    assert torch.equal(arm._standardize(original), arm._standardize(modified))
    fresh_bags, fresh_rna, fresh_ids = generated(8, 72, 'independent-test')
    output = arm.predict(fresh_bags, fresh_rna, fresh_ids)
    assert output.shape == (8, 3) and torch.isfinite(output).all()
    with pytest.raises(ValueError, match='Selection patients'):
        arm.predict(val_bags, val_rna, val_ids)
    with pytest.raises(ValueError, match='disjoint'):
        arm.predict(fresh_bags, fresh_rna, ids[:8])
    stream = io.BytesIO()
    torch.save(arm.checkpoint(), stream)
    stream.seek(0)
    restored = RegularizedMethod31.from_checkpoint(torch.load(stream, weights_only=True))
    assert torch.equal(output, restored.predict(fresh_bags, fresh_rna, fresh_ids))
    with pytest.raises(ValueError, match='Selection patients'):
        restored.predict(val_bags, val_rna, val_ids)
