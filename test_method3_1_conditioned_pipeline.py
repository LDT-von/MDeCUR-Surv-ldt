"""Pure generated-patient checks of the optional CLR-conditioned candidate."""
import io

import pytest
import torch

from method3_1_conditioned_pipeline import CLRConditionedStrictPipeline
from method3_1_interaction_features import GlobalResidualInteractions
from method3_1_strict_pipeline import Method31StrictPipeline
from test_method3_1 import generated, make_model


@pytest.fixture(scope='module')
def fixture():
    torch.set_num_threads(1)
    bags, rna, ids = generated()
    validation = generated(12, 101, 'conditional-val')
    test = generated(9, 102, 'conditional-test')
    backbone = make_model()
    backbone.pretrain(bags, rna, ids, epochs=3)
    backbone.fit_probes(bags, rna, ids, folds=4)
    mapper = GlobalResidualInteractions(backbone).fit(bags, rna, ids, folds=4)
    return mapper, ids, validation, test


def _fit(model, ids, validation):
    bags, rna, val_ids = validation
    model.fit(torch.arange(len(ids)) % 3, torch.arange(len(ids)) % 2, ids,
              bags, rna, val_ids, torch.arange(len(val_ids)) % 3,
              torch.arange(len(val_ids)) % 2, candidates=((.1, .1, .1),),
              epochs=5, eval_every=5)
    return model


def test_conditioning_changes_only_main_block_and_uses_oof(fixture):
    mapper, ids, validation, test = fixture
    base = _fit(Method31StrictPipeline(mapper, mode='tanh_outer'), ids, validation)
    model = _fit(CLRConditionedStrictPipeline(mapper, mode='tanh_outer'), ids, validation)
    original, conditioned = base.training_blocks(), model.training_blocks()
    assert torch.equal(original['clr'], conditioned['clr'])
    assert torch.equal(original['interaction'], conditioned['interaction'])
    assert torch.equal(conditioned['other'], original['other']-model.conditioner.probe.oof_predictions)
    assert not torch.equal(original['other'], conditioned['other'])
    assert not model.conditioner.metadata['survival_labels_used']
    for split in model.conditioner.metadata['split_audit']:
        assert not set(split['train_ids']) & set(split['validation_ids'])
    val_bags, val_rna, val_ids = validation
    orig_val = base._fresh_blocks(val_bags, val_rna, val_ids, validation=True)
    cond_val = model._apply_conditioner(orig_val, val_ids, stage='validation')
    assert torch.equal(cond_val['interaction'], orig_val['interaction'])
    assert torch.equal(cond_val['clr'], orig_val['clr'])
    assert torch.equal(cond_val['other'], orig_val['other']-
                       model.conditioner.probe.predict(orig_val['clr'][:, :mapper.backbone.k]))


@pytest.mark.parametrize('mode', ['none', 'diagonal', 'outer', 'tanh_outer', 'raw_outer'])
def test_checkpoint_predictions_and_split_guards(fixture, mode):
    mapper, ids, validation, test = fixture
    model = _fit(CLRConditionedStrictPipeline(mapper, mode=mode), ids, validation)
    bags, rna, test_ids = test
    expected = model.predict(bags, rna, test_ids)
    assert torch.isfinite(expected).all()
    stream = io.BytesIO()
    torch.save(model.checkpoint(), stream)
    stream.seek(0)
    restored = CLRConditionedStrictPipeline.from_checkpoint(torch.load(stream, weights_only=True))
    assert restored.mode == mode
    assert torch.equal(expected, restored.predict(bags, rna, test_ids))
    val_bags, val_rna, val_ids = validation
    with pytest.raises(ValueError, match='overlap|disjoint'):
        restored.predict(val_bags, val_rna, val_ids)
    with pytest.raises(ValueError, match='overlap|disjoint'):
        restored.predict(bags, rna, ids[:len(bags)])


def test_changed_split_metadata_is_not_silently_accepted(fixture):
    mapper, ids, validation, _ = fixture
    model = _fit(CLRConditionedStrictPipeline(mapper), ids, validation)
    state = model.checkpoint()
    state['conditioner']['val_ids'] = ('different-patient',)
    with pytest.raises(ValueError, match='splits differ'):
        CLRConditionedStrictPipeline.from_checkpoint(state)


def test_active_batched_head_and_restored_inference(fixture):
    from method3_1_active_head import ActiveHierarchicalSurvivalHead
    mapper, ids, validation, test = fixture
    model = CLRConditionedStrictPipeline(mapper, mode='tanh_outer')
    model.head = ActiveHierarchicalSurvivalHead(bins=3, min_select_epoch=5, require_interaction=True)
    _fit(model, ids, validation)
    assert model.head.selection['epoch'] == 5
    assert model.head.selection['penalties'][2] is not None
    labels, censorship = torch.arange(len(ids)) % 3, torch.arange(len(ids)) % 2
    direct_nll_gradient = model.head.survival_interaction_gradient_norm(
        model.training_blocks(), labels, censorship, ids)
    assert direct_nll_gradient > 0 and torch.isfinite(torch.tensor(direct_nll_gradient))
    bags, rna, test_ids = test
    expected = model.predict(bags, rna, test_ids)
    restored = CLRConditionedStrictPipeline.from_checkpoint(model.checkpoint())
    assert torch.equal(expected, restored.predict(bags, rna, test_ids))
