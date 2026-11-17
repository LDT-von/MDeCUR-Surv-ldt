"""CPU-only generated-patient checks for method 3.1."""
import copy
import io
import math

import pytest
import torch

from cancer_morphid_model import FrozenMorphologyBank, discrete_nll
from method3_1_model import Method31, patient_contrastive


def generated(n=24, seed=41, prefix='train'):
    rng = torch.Generator().manual_seed(seed)
    rna = torch.randn(n, 2, generator=rng, dtype=torch.float64)
    bags = []
    for i in range(n):
        left = torch.randn(6, 2, generator=rng, dtype=torch.float64)*.1 + torch.tensor([-2., rna[i, 0]])
        right = torch.randn(6, 2, generator=rng, dtype=torch.float64)*.1 + torch.tensor([2., rna[i, 0]])
        bags.append(torch.cat((left, right)))
    return bags, rna, [f'{prefix}-{i}' for i in range(n)]


def make_model():
    bank = FrozenMorphologyBank(torch.tensor([[-2., 0.], [2., 0.]]), temperature=.2,
                                metadata={'source': 'synthetic_fixed_centers', 'fit_patient_ids': []})
    return Method31(bank, 2, projection_dim=4, bins=3, seed=4)


@pytest.fixture(scope='module')
def trained():
    torch.set_num_threads(1)
    bags, rna, ids = generated()
    model = make_model()
    before = {k: v.clone() for k, v in model.path_projection.state_dict().items()}
    history = model.pretrain(bags, rna, ids, epochs=5)
    assert any(not torch.equal(before[k], v) for k, v in model.path_projection.state_dict().items())
    model.fit_probes(bags, rna, ids, folds=4)
    frozen = copy.deepcopy(model.path_projection.state_dict())
    frozen_features = model._features(*model.cached).detach().clone()
    model.fit_survival(torch.arange(len(ids)) % 3, torch.arange(len(ids)) % 2, ids, epochs=6)
    assert all(torch.equal(frozen[k], v) for k, v in model.path_projection.state_dict().items())
    assert torch.equal(frozen_features, model._features(*model.cached).detach())
    assert all(not p.requires_grad for p in model.interaction_rna.parameters())
    assert float(model.head.weight.grad[:, -model.k*model.d:].abs().sum()) > 0
    return model, history, (bags, rna, ids)


def test_patient_contrastive_permutation_and_minimum():
    x = torch.eye(5, dtype=torch.float64, requires_grad=True)
    matched = patient_contrastive(x, x, .2)
    shuffled = patient_contrastive(x, x.roll(1, 0), .2)
    assert matched < shuffled
    matched.backward()
    assert torch.isfinite(x.grad).all() and float(x.grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        patient_contrastive(x[:1], x[:1], .2)


def test_likelihood_hand_calculation():
    logits = torch.zeros(2, 3, dtype=torch.float64)
    # Event in bin 1: S0*h1=.25; censor in bin 0: S0=.5.
    loss = discrete_nll(logits, torch.tensor([1, 0]), torch.tensor([0, 1]), alpha=0, reduction='none')
    assert torch.allclose(loss, torch.tensor([-math.log(.25), -math.log(.5)], dtype=torch.float64))
    assert torch.allclose(loss, discrete_nll(logits, torch.tensor([[1], [0]]), torch.tensor([[0], [1]]), alpha=0, reduction='none'))


def test_composition_masks_and_padding():
    m = make_model()
    full = torch.tensor([[0., .1]], dtype=torch.float64)
    padding = torch.cat((full, torch.tensor([[float('nan'), float('nan')]])))
    _, enc, _ = m._input([(padding, torch.tensor([True, False]))], [[0., 0.]], ['a'])
    reference = m.bank.encode([full])
    assert torch.allclose(enc['composition'], reference['composition'])
    assert torch.allclose(enc['clr'].sum(1), torch.zeros(1, dtype=torch.float64), atol=1e-12)
    assert not enc['available'].any()  # each soft mass is just below the 1.0 cutoff


def test_full_forward_frozen_ssl_and_probe_exclusion(trained):
    model, history, (_, _, ids) = trained
    bags, rna, fresh = generated(6, 42, 'fresh')
    output = model.predict(bags, rna, fresh)
    assert output.shape == (6, 3) and torch.isfinite(output).all()
    assert all(not p.requires_grad for p in model.path_projection.parameters())
    assert len(history) == 5 and all(math.isfinite(x['total']) for x in history)
    for probe in model.probes.values():
        for audit in probe.split_audit:
            assert not set(audit['train_ids']) & set(audit['validation_ids'])
    with pytest.raises(ValueError, match='Training patients'):
        model.predict(bags, rna, ids[:6])


def test_input_modalities_and_checkpoint_roundtrip(trained):
    model, _, _ = trained
    bags, rna, ids = generated(6, 43, 'new')
    expected = model.predict(bags, rna, ids)
    assert not torch.allclose(expected, model.predict(bags, rna + .3, ids))
    changed = [x + torch.tensor([0., .4]) for x in bags]
    assert not torch.allclose(expected, model.predict(changed, rna, ids))
    stream = io.BytesIO()
    torch.save(model.checkpoint(), stream)
    stream.seek(0)
    restored = Method31.from_checkpoint(torch.load(stream, weights_only=True))
    assert torch.equal(expected, restored.predict(bags, rna, ids))
    with pytest.raises(ValueError, match='Training patients'):
        restored.predict(bags, rna, model.train_ids[:6])


def test_bank_patient_leakage_is_rejected():
    bags, rna, ids = generated()
    model = make_model()
    model.bank.metadata['fit_patient_ids'] = [ids[0], 'heldout-patient']
    with pytest.raises(ValueError, match='outside'):
        model.pretrain(bags, rna, ids, epochs=1)
    with pytest.raises(ValueError, match='distinct'):
        make_model().pretrain(bags, rna, [ids[0]]*len(ids), epochs=1)
