"""Synthetic-only integration tests for the independent 3.1 gate."""
import io

import pytest
import torch

from cancer_morphid_model import FrozenMorphologyBank
from method3_1_guarded_pipeline import GuardedMethod31Pipeline
from method3_1_independent_synthetic import make_independent_splits
from method3_1_interaction_features import GlobalResidualInteractions
from method3_1_model import Method31


def _slice(part, start, end):
    return {k: v[start:end] for k, v in part.items()}


def _tuple(part):
    return part['bags'], part['rna'], part['ids'], part['labels'], part['censorship']


@pytest.fixture(scope='module')
def fixture_data():
    torch.set_num_threads(1)
    d, _, _ = make_independent_splits(9791, 'xor', n_train=48, n_val=48, n_test=25)
    tr, val = d['train'], d['validation']
    bank = FrozenMorphologyBank.fit(tr['bags'], ids=tr['ids'], clusters=2, latent_dim=3,
                                    max_fit_patches=768, per_patient_cap=16, seed=7)
    b = Method31(bank, rna_dim=2, projection_dim=4, bins=4, seed=7)
    b.pretrain(tr['bags'], tr['rna'], tr['ids'], epochs=2)
    b.fit_probes(tr['bags'], tr['rna'], tr['ids'], folds=4)
    mapper = GlobalResidualInteractions(b).fit(tr['bags'], tr['rna'], tr['ids'], folds=4)
    return d, mapper, _slice(val, 0, 24), _slice(val, 24, 48)


def test_gate_split_id_enforced_and_checkpoint_roundtrip(fixture_data):
    d, mapper, selection, gate = fixture_data
    tr, te = d['train'], d['test']
    p = GuardedMethod31Pipeline(mapper)
    result = p.fit(tr['labels'], tr['censorship'], tr['ids'], selection=_tuple(selection),
                   gate=_tuple(gate), test_ids=te['ids'], min_n=2, min_events=1, epochs=5,
                   retain_candidates=True)
    assert result['selected'] in ('on', 'off')
    assert result['gate_outcomes_used_for_refit'] is False
    off, on = p._diagnostic_heads
    assert torch.count_nonzero(off.weights['interaction']) == 0
    assert p.head is (on if result['selected'] == 'on' else off)
    output = p.predict(te['bags'], te['rna'], te['ids'])
    assert output.shape == (25, 4) and torch.isfinite(output).all()
    with pytest.raises(ValueError, match='overlap'):
        p.predict(gate['bags'], gate['rna'], gate['ids'])
    buf = io.BytesIO(); torch.save(p.checkpoint(), buf); buf.seek(0)
    restored = GuardedMethod31Pipeline.from_checkpoint(torch.load(buf, weights_only=True))
    assert restored._diagnostic_heads is None
    assert torch.equal(output, restored.predict(te['bags'], te['rna'], te['ids']))
    with pytest.raises(ValueError, match='overlap'):
        restored.predict(selection['bags'], selection['rna'], selection['ids'])


def test_overlapping_selection_and_gate_fail_before_fit(fixture_data):
    d, mapper, selection, _ = fixture_data
    tr = d['train']
    p = GuardedMethod31Pipeline(mapper)
    with pytest.raises(ValueError, match='disjoint'):
        p.fit(tr['labels'], tr['censorship'], tr['ids'], selection=_tuple(selection),
              gate=_tuple(selection), epochs=5)
