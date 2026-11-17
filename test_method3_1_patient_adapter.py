"""Synthetic-only checks for the explicit-split patient pilot adapter."""
import numpy as np
import pytest
import torch

from cancer_parity import dct_bin_edges
from method3_1_independent_synthetic import make_independent_splits
from method3_1_patient_adapter import (fit_pilot, evaluate_pilot, PilotConfig,
                                        checkpoint_pilot, restore_pilot)


def _convert(data, start, end, *, no_events=False):
    labels = data['labels'][start:end].double()
    count = len(labels)
    event = 1 - data['censorship'][start:end].double()
    if no_events:
        event = torch.zeros_like(event)
    return {'bags': data['bags'][start:end], 'rna': data['rna'][start:end],
            'ids': data['ids'][start:end], 'time': labels + torch.arange(count).double()/(count+1),
            'event': event}


def test_explicit_splits_fit_train_only_and_closed_gate():
    torch.set_num_threads(1)
    data, _, _ = make_independent_splits(9995, 'xor', n_train=160, n_val=240, n_test=120)
    train = _convert(data['train'], 0, 160)
    select = _convert(data['validation'], 0, 120)
    gate = _convert(data['validation'], 120, 240, no_events=True)
    evaluation = _convert(data['test'], 0, 120)
    with pytest.raises(ValueError, match='disjoint'):
        fit_pilot(train, select, gate, train['ids'], PilotConfig())
    fitted = fit_pilot(train, select, gate, evaluation['ids'], PilotConfig())
    assert np.array_equal(fitted['bin_edges'], dct_bin_edges(train['time'], train['event'], 4))
    assert torch.allclose(fitted['rna_scaler']['rna_mean'], train['rna'].double().mean(0))
    assert fitted['fit_result']['gate']['enabled'] is False
    assert fitted['fit_result']['gate']['reason']
    assert fitted['fit_result']['gate_outcomes_used_for_refit'] is False
    out = evaluate_pilot(fitted, evaluation)
    assert out['survival'].shape == (120, 4)
    assert np.isfinite(out['nll_alpha0'])
    restored = restore_pilot(checkpoint_pilot(fitted))
    assert torch.equal(evaluate_pilot(restored, evaluation)['logits'], out['logits'])
    with pytest.raises(ValueError):
        evaluate_pilot(fitted, select)
