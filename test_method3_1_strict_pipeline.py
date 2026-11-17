"""Generated-only tests for strict additive main-effect controls."""
import io

import pytest
import torch

from cancer_morphid_model import discrete_nll
from method3_1_interaction_features import GlobalResidualInteractions
from method3_1_strict_pipeline import Method31StrictPipeline
from test_method3_1 import generated, make_model


@pytest.fixture(scope='module')
def data():
    torch.set_num_threads(1)
    train, rna, ids = generated()
    validation = generated(12, 101, 'strict-val')
    test = generated(9, 102, 'strict-test')
    backbone = make_model()
    backbone.pretrain(train, rna, ids, epochs=3)
    backbone.fit_probes(train, rna, ids, folds=4)
    mapper = GlobalResidualInteractions(backbone).fit(train, rna, ids, folds=4)
    return mapper, ids, validation, test


def _fit(mapper, ids, validation, mode, interaction_mode='per_bin', expand_clr=True):
    bags, rna, val_ids = validation
    model = Method31StrictPipeline(mapper, mode=mode, interaction_mode=interaction_mode, expand_clr=expand_clr)
    model.fit(torch.arange(len(ids)) % 3, torch.arange(len(ids)) % 2, ids,
              bags, rna, val_ids, torch.arange(len(val_ids)) % 3,
              torch.arange(len(val_ids)) % 2, candidates=((.1, .1, .1),),
              epochs=5, eval_every=5)
    return model


def test_default_mode_and_main_blocks_are_modality_separable(data):
    mapper, ids, validation, _ = data
    bags, rna, val_ids = validation
    model = Method31StrictPipeline(mapper)
    assert model.mode == 'tanh_outer'
    model._fit_moments()
    orig = model._fresh_blocks(bags, rna, val_ids, validation=True)
    changed_rna = model._fresh_blocks(bags, rna+2.37, val_ids, validation=True)
    changed_bags = [bag + torch.tensor([.3, .7], dtype=torch.float64) for bag in bags]
    changed_path = model._fresh_blocks(changed_bags, rna, val_ids, validation=True)
    b = mapper.backbone
    path_width = b.k*b.d+b.k+b.k*b.projection_dim+3*b.d
    assert torch.equal(orig['clr'], changed_rna['clr'])
    assert torch.equal(orig['other'][:, :path_width], changed_rna['other'][:, :path_width])
    assert torch.equal(orig['other'][:, path_width:], changed_path['other'][:, path_width:])
    assert not torch.equal(orig['other'][:, :path_width], changed_path['other'][:, :path_width])
    assert not torch.equal(orig['other'][:, path_width:], changed_rna['other'][:, path_width:])


def test_same_main_blocks_none_exact_and_interaction_nll_gradients(data):
    mapper, ids, validation, _ = data
    reference = None
    for mode in ('none', 'diagonal', 'outer', 'tanh_outer', 'raw_outer'):
        model = _fit(mapper, ids, validation, mode)
        blocks = model.training_blocks()
        if reference is None:
            reference = blocks
        else:
            assert torch.equal(blocks['clr'], reference['clr'])
            assert torch.equal(blocks['other'], reference['other'])
        inter = blocks['interaction']
        if mode == 'none':
            assert inter.shape == (len(ids), 0)
            continue
        assert inter.shape[0] == len(ids) and torch.isfinite(inter).all()
        assert inter.std(0, unbiased=False).max() > 1e-6
        col = int(inter.std(0, unbiased=False).argmax())
        censorship = (inter[:, col] < inter[:, col].median()).long()
        weights = torch.zeros(inter.shape[1], 1, dtype=torch.float64, requires_grad=True)
        loss = discrete_nll(inter @ weights, torch.zeros(len(ids), dtype=torch.long), censorship, alpha=0.)
        loss.backward()
        assert weights.grad is not None and torch.isfinite(weights.grad).all()
        assert float(weights.grad.abs().sum()) > 1e-8


def test_same_width_interactions_on_the_protocol_dimensions(data):
    mapper, ids, validation, _ = data
    b = mapper.backbone
    # This is protocol-specific: K=2 and the RNA width is 2. For other
    # dimensions, equal capacity must not be silently assumed.
    assert b.k == 2 and b.rna_dim == 2
    widths = {}
    for mode in ('diagonal', 'outer', 'tanh_outer', 'raw_outer'):
        model = _fit(mapper, ids, validation, mode)
        widths[mode] = model.training_blocks()['interaction'].shape[1]
    assert widths == {name: b.k*b.d for name in widths}


def test_six_dimensional_match_on_three_component_synthetic_bank():
    from cancer_morphid_model import FrozenMorphologyBank
    from method3_1_independent_synthetic import make_independent_splits
    from method3_1_model import Method31
    torch.set_num_threads(1)
    parts, _, _ = make_independent_splits(7261, 'xor', n_train=32, n_val=12, n_test=12)
    tr = parts['train']
    bank = FrozenMorphologyBank.fit(tr['bags'], ids=tr['ids'], clusters=2, latent_dim=3,
                                    max_fit_patches=512, per_patient_cap=16, seed=7261)
    backbone = Method31(bank, 2, projection_dim=4, bins=4, seed=7261)
    backbone.pretrain(tr['bags'], tr['rna'], tr['ids'], epochs=2)
    backbone.fit_probes(tr['bags'], tr['rna'], tr['ids'], folds=4)
    mapper = GlobalResidualInteractions(backbone).fit(tr['bags'], tr['rna'], tr['ids'], folds=4)
    shapes = []
    for mode in ('diagonal', 'outer', 'tanh_outer', 'raw_outer'):
        model = Method31StrictPipeline(mapper, mode=mode)
        shapes.append(model._train_blocks(model._fit_moments())['interaction'].shape)
    assert shapes == [(32, 6)]*4


def test_no_interaction_prediction_is_additive_at_fixed_fitted_model(data):
    mapper, ids, validation, test = data
    model = _fit(mapper, ids, validation, 'none')
    bags, rna, test_ids = test
    # Difference-in-differences vanishes for additive functions even after
    # scaling, trained coefficients, and nonlinear per-modality transforms.
    shifted_bags = [bag + torch.tensor([.3, .7], dtype=torch.float64) for bag in bags]
    shifted_rna = rna + torch.tensor([.5, -.8], dtype=torch.float64)
    base = model.predict(bags, rna, test_ids)
    both = model.predict(shifted_bags, shifted_rna, test_ids)
    only_path = model.predict(shifted_bags, rna, test_ids)
    only_rna = model.predict(bags, shifted_rna, test_ids)
    assert torch.allclose(base+both-only_path-only_rna, torch.zeros_like(base), atol=1e-12, rtol=0.)


def test_train_fitted_nonlinear_clr_has_distinct_linear_control(data):
    mapper, ids, validation, test = data
    full = _fit(mapper, ids, validation, 'none', expand_clr=True)
    linear = _fit(mapper, ids, validation, 'none', expand_clr=False)
    xb, xl = full.training_blocks()['clr'], linear.training_blocks()['clr']
    k = mapper.backbone.k
    assert xb.shape == (len(ids), 3*k) and xl.shape == (len(ids), k)
    assert torch.equal(xb[:, :k], xl)
    z = (xl-full.clr_mean)/full.clr_scale
    assert torch.allclose(xb[:, k:2*k], torch.tanh(z))
    assert torch.allclose(xb[:, 2*k:], z.square())
    bags, rna, test_ids = test
    snapshot = full.checkpoint()
    restored = Method31StrictPipeline.from_checkpoint(snapshot)
    assert restored.expand_clr is True
    assert torch.equal(full.predict(bags, rna, test_ids), restored.predict(bags, rna, test_ids))


@pytest.mark.parametrize('mode', ['none', 'diagonal', 'outer', 'tanh_outer', 'raw_outer'])
def test_all_modes_checkpoint_replay_and_id_boundaries(data, mode):
    mapper, ids, validation, test = data
    model = _fit(mapper, ids, validation, mode)
    bags, rna, test_ids = test
    expected = model.predict(bags, rna, test_ids)
    stream = io.BytesIO()
    torch.save(model.checkpoint(), stream)
    stream.seek(0)
    restored = Method31StrictPipeline.from_checkpoint(torch.load(stream, weights_only=True))
    assert restored.mode == mode
    assert torch.equal(expected, restored.predict(bags, rna, test_ids))
    val_bags, val_rna, val_ids = validation
    with pytest.raises(ValueError, match='overlap|disjoint'):
        restored.predict(val_bags, val_rna, val_ids)
    with pytest.raises(ValueError, match='overlap|disjoint'):
        restored.predict(bags, rna, ids[:len(bags)])
