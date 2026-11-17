"""CPU synthetic known-answer tests for the patient-paired holdout gate."""
import copy
import json
import math

import pytest
from scipy.stats import t as student_t
import torch

from cancer_parity import discrete_nll
from method3_1_holdout_gate import evaluate_holdout_gate


def inputs(n=40):
    off = torch.zeros(n, 2, dtype=torch.float64)
    on = off.clone()
    on[:, 0] = torch.linspace(.2, 1.2, n, dtype=torch.float64)
    labels = torch.zeros(n, dtype=torch.long)
    censorship = torch.zeros(n, dtype=torch.long)
    ids = [f'gate-{i}' for i in range(n)]
    split = {'train_ids': ['train-0', 'train-1'], 'validation_ids': ['val-0'],
             'test_ids': ['test-0']}
    return off, on, labels, censorship, ids, split


def test_known_answer_and_production_patient_likelihood():
    off, on, y, c, ids, splits = inputs()
    observed = evaluate_holdout_gate(off, on, y, c, ids, **splits)
    diff = discrete_nll(off, y, c, alpha=0., reduction='none')-discrete_nll(on, y, c, alpha=0., reduction='none')
    expected_sd = float(diff.std(unbiased=True))
    critical = float(student_t.ppf(.95, len(ids)-1))
    assert observed['enabled'] and observed['reason'] == 'gain_exceeds_threshold'
    assert observed['selection_kind'] == 'paired_t_lower_bound'
    assert observed['events'] == len(ids) and observed['test_ids_checked']
    assert observed['per_patient_gain'] == diff.tolist()
    assert observed['mean_gain'] == pytest.approx(float(diff.mean()))
    assert observed['std_gain'] == pytest.approx(expected_sd)
    assert observed['standard_error'] == pytest.approx(expected_sd/math.sqrt(len(ids)))
    assert observed['lower_bound'] == pytest.approx(float(diff.mean())-critical*expected_sd/math.sqrt(len(ids)))
    # JSON representation is complete, finite, and safe to reload.
    assert json.loads(json.dumps(observed, allow_nan=False)) == observed


def test_permutation_reverse_and_immutability():
    off, on, y, c, ids, splits = inputs()
    original = [value.clone() for value in (off, on, y, c)]
    result = evaluate_holdout_gate(off, on, y, c, ids, **splits)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(31))
    permuted = evaluate_holdout_gate(off[order], on[order], y[order], c[order],
                                      [ids[i] for i in order], **splits)
    reverse = evaluate_holdout_gate(on, off, y, c, ids, **splits)
    assert permuted['enabled'] == result['enabled']
    assert permuted['mean_gain'] == pytest.approx(result['mean_gain'], abs=1e-14)
    assert permuted['lower_bound'] == pytest.approx(result['lower_bound'], abs=1e-14)
    assert reverse['mean_gain'] == pytest.approx(-result['mean_gain'], abs=1e-14)
    assert not reverse['enabled']
    assert all(torch.equal(value, before) for value, before in zip((off, on, y, c), original))


def test_zero_variance_insufficient_information_and_nonfinite_fail_closed():
    off, on, y, c, ids, splits = inputs()
    assert evaluate_holdout_gate(off, off, y, c, ids, **splits)['reason'] == 'zero_or_negligible_variance'
    # An exactly positive but constant gain has no estimable paired variance.
    constant = off.clone(); constant[:, 0] = 1.
    d = evaluate_holdout_gate(off, constant, y, c, ids, **splits)
    assert not d['enabled'] and d['mean_gain'] > 0 and d['reason'] == 'zero_or_negligible_variance'
    assert evaluate_holdout_gate(off[:12], on[:12], y[:12], c[:12], ids[:12], **splits)['reason'] == 'insufficient_n'
    cens = torch.ones_like(c)
    assert evaluate_holdout_gate(off, on, y, cens, ids, **splits)['reason'] == 'all_censored'
    cens[:4] = 0
    assert evaluate_holdout_gate(off, on, y, cens, ids, **splits)['reason'] == 'insufficient_events'
    bad = on.clone(); bad[0, 0] = float('nan')
    assert evaluate_holdout_gate(off, bad, y, c, ids, **splits)['reason'] == 'nonfinite_logits'


def test_strict_boundary_and_direct_comparator():
    off, on, y, c, ids, splits = inputs()
    result = evaluate_holdout_gate(off, on, y, c, ids, **splits)
    equality = evaluate_holdout_gate(off, on, y, c, ids, min_gain=result['lower_bound'], **splits)
    assert not equality['enabled'] and equality['reason'] == 'gain_not_established'
    direct = evaluate_holdout_gate(off, on, y, c, ids, alpha=.5, min_gain=0., **splits)
    assert direct['enabled'] and direct['critical_value'] == 0.
    assert direct['selection_kind'] == 'direct_mean_comparison'
    assert direct['lower_bound'] == direct['mean_gain']


@pytest.mark.parametrize('problem', ['shape', 'invalid_bin', 'invalid_censorship', 'missing_id', 'duplicate_id',
                                     'gate_train_overlap', 'gate_validation_overlap', 'gate_test_overlap',
                                     'train_validation_overlap', 'invalid_alpha'])
def test_rejects_invalid_input_and_provenance_overlap(problem):
    off, on, y, c, ids, splits = inputs()
    kwargs = copy.deepcopy(splits)
    if problem == 'shape':
        on = on[:, :1]
    elif problem == 'invalid_bin':
        y[0] = 5
    elif problem == 'invalid_censorship':
        c[0] = 2
    elif problem == 'missing_id':
        ids[0] = None
    elif problem == 'duplicate_id':
        ids[0] = ids[1]
    elif problem == 'gate_train_overlap':
        kwargs['train_ids'].append(ids[0])
    elif problem == 'gate_validation_overlap':
        kwargs['validation_ids'].append(ids[0])
    elif problem == 'gate_test_overlap':
        kwargs['test_ids'].append(ids[0])
    elif problem == 'train_validation_overlap':
        kwargs['validation_ids'].append(kwargs['train_ids'][0])
    elif problem == 'invalid_alpha':
        kwargs['alpha'] = .7
    with pytest.raises(ValueError):
        evaluate_holdout_gate(off, on, y, c, ids, **kwargs)


def test_future_test_ids_are_optional_and_empty_gate_fails_closed():
    off, on, y, c, ids, splits = inputs()
    splits.pop('test_ids')
    assert evaluate_holdout_gate(off, on, y, c, ids, **splits)['test_ids_checked'] is False
    empty = evaluate_holdout_gate(off[:0], on[:0], y[:0], c[:0], [], **splits)
    assert not empty['enabled'] and empty['reason'] == 'insufficient_n'
