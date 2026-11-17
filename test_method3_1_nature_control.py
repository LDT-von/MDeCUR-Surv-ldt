"""Pure synthetic tests for the Nature-inspired discrete composition control."""
import io
import math

import pytest
import torch

from cancer_morphid_model import FrozenMorphologyBank, discrete_nll
from method3_1_nature_control import NatureCompositionControl


def _bank(ids=()):
    return FrozenMorphologyBank(torch.tensor([[-3., 0.], [3., 0.]]), temperature=.1,
          metadata={'source': 'synthetic_fixed_centers', 'fit_patient_ids': list(ids)})


def _data(n=20, prefix='train'):
    bags = []
    for i in range(n):
        count = 2 + (i % 7)
        bags.append(torch.cat((torch.tensor([[-3., 0.]]).repeat(count, 1),
                               torch.tensor([[3., 0.]]).repeat(10-count, 1))))
    labels = torch.tensor([0 if i % 7 >= 4 else 1 for i in range(n)])
    censorship = torch.tensor([1 if i % 4 == 0 else 0 for i in range(n)])
    ids = [f'{prefix}-{i}' for i in range(n)]
    return bags, labels, censorship, ids


def test_hand_likelihood_and_only_training_head_updates():
    torch.set_num_threads(1)
    # h0=h1=.5: an event in bin 1 costs -log(.25); a bin-0 censor costs -log(.5).
    logits = torch.zeros(2, 2, dtype=torch.float64)
    expected = torch.tensor([-math.log(.25), -math.log(.5)], dtype=torch.float64)
    got = discrete_nll(logits, torch.tensor([[1], [0]]), torch.tensor([[0], [1]]), alpha=0., reduction='none')
    assert torch.allclose(got, expected)
    bags, labels, censor, ids = _data()
    bank = _bank(ids)
    before_centers = bank.centers.clone()
    model = NatureCompositionControl(bank, bins=2, l2=.01)
    history = model.fit(bags, labels, censor, ids, validation=_data(14, 'val'),
                        l2_grid=(.01,), epochs=100, lr=.03)
    assert min(x['train_nll'] for x in history) < history[0]['train_nll']
    assert torch.equal(before_centers, bank.centers)
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    assert not torch.allclose(model.head.weight, torch.zeros_like(model.head.weight))


def test_prediction_uses_only_composition_and_checkpoint_reloads():
    bags, labels, censor, ids = _data()
    model = NatureCompositionControl(_bank(ids), bins=2)
    model.fit(bags, labels, censor, ids, validation=_data(14, 'val'), l2_grid=(.01,), epochs=60)
    test, _, _, test_ids = _data(7, 'test')
    prediction = model.predict(test, test_ids)
    assert prediction.shape == (7, 2) and torch.isfinite(prediction).all()
    assert not torch.allclose(prediction[0], prediction[-1])
    stream = io.BytesIO()
    torch.save(model.checkpoint(), stream)
    stream.seek(0)
    loaded = NatureCompositionControl.from_checkpoint(torch.load(stream, weights_only=True))
    assert torch.equal(prediction, loaded.predict(test, test_ids))
    with pytest.raises(ValueError, match='overlap'):
        loaded.predict(test, ids[:len(test)])
    with pytest.raises(ValueError, match='overlap'):
        loaded.predict(test, loaded.validation_ids[:len(test)])
    assert loaded.selection == model.selection


def test_training_provenance_and_labels_fail_closed():
    bags, labels, censor, ids = _data()
    leak = NatureCompositionControl(_bank(ids + ['test-patient']), bins=2)
    with pytest.raises(ValueError, match='outside'):
        leak.fit(bags, labels, censor, ids, validation=_data(14, 'val'), epochs=5)
    model = NatureCompositionControl(_bank(ids), bins=2)
    bad = labels.clone(); bad[0] = 9
    before = model.head.weight.detach().clone()
    with pytest.raises(ValueError):
        model.fit(bags, bad, censor, ids, validation=_data(14, 'val'), epochs=5)
    assert torch.equal(before, model.head.weight)
    with pytest.raises(ValueError, match='distinct'):
        model.fit(bags, labels, censor, [ids[0]]*len(ids), validation=_data(14, 'val'), epochs=5)
    with pytest.raises(ValueError, match='overlap'):
        model.fit(bags, labels, censor, ids, validation=(bags, labels, censor, ids), epochs=5)


def test_training_is_deterministic_and_patient_order_invariant():
    bags, labels, censor, ids = _data()
    first = NatureCompositionControl(_bank(ids), bins=2)
    second = NatureCompositionControl(_bank(ids), bins=2)
    first.fit(bags, labels, censor, ids, validation=_data(14, 'val'), l2_grid=(.01,), epochs=25)
    # Full-batch objective is patient-order invariant, up to FP reduction.
    order = torch.arange(len(ids)-1, -1, -1)
    second.fit([bags[i] for i in order], labels[order], censor[order], [ids[i] for i in order],
               validation=_data(14, 'val'), l2_grid=(.01,), epochs=25)
    test, _, _, test_ids = _data(7, 'held-out')
    assert torch.allclose(first.predict(test, test_ids), second.predict(test, test_ids), atol=1e-12, rtol=1e-12)


def test_validation_selection_and_training_risk_set_initialization():
    bags, labels, censor, ids = _data()
    model = NatureCompositionControl(_bank(ids), bins=2)
    history = model.fit(bags, labels, censor, ids, validation=_data(14, 'val'),
                        l2_grid=(0., .1), epochs=10, lr=.03, eval_every=5)
    assert len(history) == 4
    assert model.selection['best']['validation_nll'] == min(x['validation_nll'] for x in history)
    risk0 = len(labels)
    events0 = int(((labels == 0) & (censor == 0)).sum())
    probability = (events0 + .5)/(risk0 + 1.)
    assert math.isclose(model.selection['initial_bias'][0], math.log(probability/(1-probability)))


def test_all_censored_and_empty_later_risk_sets_are_finite():
    bags, labels, _, ids = _data()
    labels = torch.zeros_like(labels)
    censor = torch.ones_like(labels)
    val_bags, val_y, _, val_ids = _data(14, 'val')
    val_y = torch.zeros_like(val_y)
    val_censor = torch.ones_like(val_y)
    model = NatureCompositionControl(_bank(ids), bins=3)
    history = model.fit(bags, labels, censor, ids,
                        validation=(val_bags, val_y, val_censor, val_ids),
                        l2_grid=(.1,), epochs=5, eval_every=5)
    assert all(math.isfinite(z) for z in model.selection['initial_bias'])
    assert model.selection['initial_bias'][1:] == [0., 0.]
    assert math.isfinite(history[0]['validation_nll'])


def test_standardization_fit_only_on_training_and_constant_scale_guard():
    bags, labels, censor, ids = _data()
    vb, vy, vc, vi = _data(14, 'val')
    first = NatureCompositionControl(_bank(ids), bins=2)
    second = NatureCompositionControl(_bank(ids), bins=2)
    first.fit(bags, labels, censor, ids, validation=(vb, vy, vc, vi),
              l2_grid=(.1,), epochs=5, eval_every=5)
    # Massive validation feature change can change model selection, but cannot
    # change moments, which must come from the same training patients only.
    altered_val = [torch.tensor([[-3., 0.]]*9 + [[3., 0.]]) for _ in vb]
    second.fit(bags, labels, censor, ids, validation=(altered_val, vy, vc, vi),
               l2_grid=(.1,), epochs=5, eval_every=5)
    raw = first.bank.encode(bags)['clr']
    assert torch.equal(first.clr_mean, raw.mean(0))
    assert torch.equal(first.clr_mean, second.clr_mean)
    assert torch.equal(first.clr_scale, second.clr_scale)
    constant_bags = [bags[0]]*len(bags)
    constant = NatureCompositionControl(_bank(ids), bins=2)
    constant.fit(constant_bags, labels, censor, ids, validation=(vb, vy, vc, vi),
                 l2_grid=(.1,), epochs=5, eval_every=5)
    assert torch.equal(constant.clr_scale, torch.ones_like(constant.clr_scale))
