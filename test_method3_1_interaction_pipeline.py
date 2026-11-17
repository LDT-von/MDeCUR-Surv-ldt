"""Generated-data tests for integrated 3.1 interaction pipelines."""
import io

import pytest
import torch

from method3_1_interaction_features import GlobalResidualInteractions
from method3_1_interaction_pipeline import Method31InteractionPipeline
from test_method3_1 import generated, make_model


@pytest.fixture(scope='module')
def fitted_pipelines():
    torch.set_num_threads(1)
    bags, rna, ids = generated(32, 401, 'pipeline-train')
    val_bags, val_rna, val_ids = generated(16, 402, 'pipeline-val')
    model = make_model()
    model.pretrain(bags, rna, ids, epochs=3)
    model.fit_probes(bags, rna, ids, folds=4)
    mapper = GlobalResidualInteractions(model).fit(bags, rna, ids, folds=4)
    y = torch.arange(len(ids)) % 3
    c = (torch.arange(len(ids)) % 2).double()
    vy = torch.arange(len(val_ids)) % 3
    vc = (torch.arange(len(val_ids)) % 2).double()
    outputs = {}
    for mode in Method31InteractionPipeline.MODES:
        pipe = Method31InteractionPipeline(mapper, mode=mode, interaction_mode='shared')
        pipe.fit(y, c, ids, val_bags, val_rna, val_ids, vy, vc,
                 candidates=((.1, .1, .1),), epochs=5, lr=.03, eval_every=5)
        outputs[mode] = pipe
    return outputs, (bags, rna, ids), (val_bags, val_rna, val_ids)


def test_all_arms_have_identical_main_blocks_and_correct_interactions(fitted_pipelines):
    pipelines, _, _ = fitted_pipelines
    reference = pipelines['none'].training_blocks()
    assert reference['interaction'].shape == (32, 0)
    for mode, pipe in pipelines.items():
        blocks = pipe.training_blocks()
        assert torch.equal(blocks['clr'], reference['clr'])
        assert torch.equal(blocks['other'], reference['other'])
        assert all(torch.isfinite(x).all() for x in blocks.values())
    mapped = pipelines['outer'].mapper.training_features()
    assert torch.equal(pipelines['diagonal'].training_blocks()['interaction'], mapped['diagonal'])
    assert torch.equal(pipelines['outer'].training_blocks()['interaction'], mapped['outer'])
    assert torch.equal(pipelines['tanh_outer'].training_blocks()['interaction'], mapped['tanh_outer'])
    pipe = pipelines['raw_outer']
    p = (mapped['global_pooled']-pipe.pooled_mean)/pipe.pooled_scale
    g = mapped['rna_standardized']
    assert torch.equal(pipe.training_blocks()['interaction'], (p.unsqueeze(2)*g.unsqueeze(1)).flatten(1))
    # No interaction inputs or coefficients in the none arm, including after fitting.
    assert pipelines['none'].head.weights['interaction'].numel() == 0


def test_same_mapper_validation_ids_and_test_isolation(fitted_pipelines):
    pipelines, (bags, rna, ids), (val_bags, val_rna, val_ids) = fitted_pipelines
    mapper = pipelines['outer'].mapper
    assert all(pipe.mapper is mapper for pipe in pipelines.values())
    assert mapper.val_ids == tuple(val_ids)
    # Reusing exactly the same registered validation set remains legal.
    mapper.validation_features(val_bags, val_rna, val_ids)
    fresh_bags, fresh_rna, fresh_ids = generated(8, 403, 'pipeline-test')
    for pipe in pipelines.values():
        logits = pipe.predict(fresh_bags, fresh_rna, fresh_ids)
        assert logits.shape == (8, 3) and torch.isfinite(logits).all()
        with pytest.raises(ValueError, match='overlap'):
            pipe.predict(bags[:8], rna[:8], ids[:8])
        with pytest.raises(ValueError, match='overlap'):
            pipe.predict(val_bags, val_rna, val_ids)


def test_checkpoint_restores_full_pipeline_predictions(fitted_pipelines):
    pipelines, _, _ = fitted_pipelines
    bags, rna, ids = generated(8, 404, 'pipeline-checkpoint-test')
    for mode in Method31InteractionPipeline.MODES:
        original = pipelines[mode]
        expected = original.predict(bags, rna, ids)
        file = io.BytesIO()
        torch.save(original.checkpoint(), file)
        file.seek(0)
        restored = Method31InteractionPipeline.from_checkpoint(torch.load(file, weights_only=True))
        assert restored.mode == mode and restored.head.interaction_mode == 'shared'
        assert torch.equal(restored.predict(bags, rna, ids), expected)
        with pytest.raises(RuntimeError, match='OOF'):
            restored.training_blocks()


def test_per_bin_mode_is_supported():
    bags, rna, ids = generated(24, 405, 'per-bin-train')
    val_bags, val_rna, val_ids = generated(12, 406, 'per-bin-val')
    model = make_model()
    model.pretrain(bags, rna, ids, epochs=2)
    model.fit_probes(bags, rna, ids, folds=4)
    mapper = GlobalResidualInteractions(model).fit(bags, rna, ids)
    pipe = Method31InteractionPipeline(mapper, mode='outer', interaction_mode='per_bin')
    pipe.fit(torch.arange(24) % 3, torch.arange(24) % 2, ids, val_bags, val_rna,
             val_ids, torch.arange(12) % 3, torch.arange(12) % 2,
             candidates=((.1, .1, .1),), epochs=3, eval_every=3)
    assert pipe.head.weights['interaction'].shape[0] == 3
