"""Single-use, patient-paired holdout gate for fixed survival predictors.

This gate compares *already frozen* off/on models on one disjoint gate split.
It does not fit or alter either model. Train and validation identities are
caller-supplied declarations, not proof that upstream provenance is correct.
Patientwise results in the receipt can be sensitive when using real outcomes.
"""
import hashlib
import json
import math

import torch
from scipy.stats import t as student_t

from cancer_parity import discrete_nll


def _patient_ids(values, name, *, expected=None):
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f'{name} must be an explicit sequence of patient IDs')
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f'{name} must be an explicit sequence of patient IDs') from exc
    if any(value is None for value in values):
        raise ValueError(f'{name} cannot contain missing patient IDs')
    result = tuple(str(value) for value in values)
    if (len(set(result)) != len(result) or any(not value or value != value.strip() for value in result)
            or (expected is not None and len(result) != expected)):
        raise ValueError(f'{name} must contain distinct nonempty patient IDs in row order')
    return result


def _result(*, enabled, reason, n, events, gate_ids, min_n, min_events,
            min_gain, alpha, std_floor, gains=None, mean=None, sd=None, se=None,
            critical=None, lower=None, test_ids_checked=False):
    # Encode the sequence unambiguously, without saving patient ID strings.
    material = json.dumps(gate_ids, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return {'version': 'method3_1_single_holdout_gate_v1', 'enabled': bool(enabled),
            'reason': reason, 'n': n, 'events': events,
            'mean_gain': mean, 'std_gain': sd, 'standard_error': se,
            'critical_value': critical, 'lower_bound': lower,
            'min_n': min_n, 'min_events': min_events, 'min_gain': float(min_gain),
            'alpha': float(alpha), 'std_floor': float(std_floor),
            'selection_kind': 'direct_mean_comparison' if alpha == .5 else 'paired_t_lower_bound',
            'per_patient_gain': gains, 'gate_ids_sha256': hashlib.sha256(material).hexdigest(),
            'test_ids_checked': bool(test_ids_checked),
            'gain_definition': 'patientwise discrete NLL(off) minus NLL(on), alpha=0',
            'limitations': ['Caller-supplied patient IDs do not prove data provenance.',
                'A t lower bound is approximate without normally distributed independent patient gains.',
                'The gate split participates in selection and cannot evaluate the selected model.',
                'Prediction gain does not prove statistical independence, causality or PID synergy.']}


def evaluate_holdout_gate(off_logits, on_logits, labels, censorship, gate_ids, *,
                          train_ids, validation_ids, test_ids=None,
                          min_n=30, min_events=5, min_gain=.005, alpha=.05,
                          std_floor=1e-12):
    """Compare two frozen predictors using a disjoint, one-use gate cohort.

    For patient i define ``d_i=NLL_i(off)-NLL_i(on)`` using the production
    censored discrete-time likelihood with alpha=0.  Enable iff
    ``mean(d)-t_(n-1,1-alpha)*std(d,ddof=1)/sqrt(n) > min_gain``.
    ``alpha=.5`` is a direct mean comparison, not a confidence-bound claim.
    Split identities must be reported by the caller; future test IDs need not
    be known.  The caller must reject gate IDs when evaluating later patients.

    Insufficient n/events, nonfinite predictions, numerical overflow and
    zero-variance gain close the gate. Invalid shapes, outcomes, configuration
    or identity overlap raise ValueError rather than silently enabling it.
    """
    if type(min_n) is not int or min_n < 2 or type(min_events) is not int or min_events < 0:
        raise ValueError('min_n must be an integer >=2 and min_events a nonnegative integer')
    if (not math.isfinite(min_gain) or min_gain < 0 or not math.isfinite(alpha)
            or not 0 < alpha <= .5 or not math.isfinite(std_floor) or std_floor < 0):
        raise ValueError('Invalid minimum gain, one-sided alpha, or variance floor')
    off, on = torch.as_tensor(off_logits), torch.as_tensor(on_logits)
    if (off.ndim != 2 or on.shape != off.shape or off.shape[1] < 1
            or off.is_complex() or on.is_complex()):
        raise ValueError('Off and on logits must be matching real [patients,bins] matrices')
    off, on = off.detach().cpu().double(), on.detach().cpu().double()
    n, bins = off.shape
    gate = _patient_ids(gate_ids, 'gate_ids', expected=n)
    groups = [('gate', gate), ('train', _patient_ids(train_ids, 'train_ids')),
              ('validation', _patient_ids(validation_ids, 'validation_ids'))]
    if test_ids is not None:
        groups.append(('test', _patient_ids(test_ids, 'test_ids')))
    for i, (left_name, left) in enumerate(groups):
        for right_name, right in groups[:i]:
            if set(left) & set(right):
                raise ValueError(f'{left_name} and {right_name} patient IDs overlap')
    y, c = torch.as_tensor(labels), torch.as_tensor(censorship)
    if (y.ndim not in (1, 2) or c.ndim not in (1, 2)
            or (y.ndim == 2 and y.shape[1] != 1) or (c.ndim == 2 and c.shape[1] != 1)
            or y.numel() != n or c.numel() != n):
        raise ValueError('One event-bin and censorship value per gate patient is required')
    common = dict(n=n, gate_ids=gate, min_n=min_n, min_events=min_events,
                  min_gain=min_gain, alpha=alpha, std_floor=std_floor,
                  test_ids_checked=test_ids is not None)
    if not n:
        return _result(enabled=False, reason='insufficient_n', events=0, **common)
    # Invoke the production validator before any early-exit branch.
    discrete_nll(torch.zeros(n, bins, dtype=torch.float64), y, c, alpha=0., reduction='none')
    events = int((c.reshape(-1).cpu() == 0).sum())
    common['events'] = events
    if not torch.isfinite(off).all() or not torch.isfinite(on).all():
        return _result(enabled=False, reason='nonfinite_logits', **common)
    if n < min_n:
        return _result(enabled=False, reason='insufficient_n', **common)
    if events == 0:
        return _result(enabled=False, reason='all_censored', **common)
    if events < min_events:
        return _result(enabled=False, reason='insufficient_events', **common)
    with torch.no_grad():
        a = discrete_nll(off, y, c, alpha=0., reduction='none')
        b = discrete_nll(on, y, c, alpha=0., reduction='none')
        differences = a-b
        if not torch.isfinite(differences).all():
            return _result(enabled=False, reason='nonfinite_gains', **common)
        mean, sd = float(differences.mean()), float(differences.std(unbiased=True))
    if not math.isfinite(mean) or not math.isfinite(sd):
        return _result(enabled=False, reason='nonfinite_statistics', **common)
    gains = differences.tolist()
    if sd <= std_floor:
        return _result(enabled=False, reason='zero_or_negligible_variance',
                       gains=gains, mean=mean, sd=sd, **common)
    se = sd/math.sqrt(n)
    critical = 0. if alpha == .5 else float(student_t.ppf(1-alpha, n-1))
    lower = mean-critical*se
    if not all(math.isfinite(x) for x in (se, critical, lower)):
        return _result(enabled=False, reason='nonfinite_statistics', **common)
    enabled = lower > min_gain
    return _result(enabled=enabled, reason='gain_exceeds_threshold' if enabled else 'gain_not_established',
                   gains=gains, mean=mean, sd=sd, se=se, critical=critical, lower=lower, **common)
