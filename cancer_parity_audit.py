"""Independent time-semantics audit of the legacy discrete survival metrics.

This module does not change the legacy scorer or its archived results.  Column
``j`` of cumulative discrete survival is evaluated at the finite training-bin
edge ``edges[j + 1]``.  The last, open-ended bin does not acquire an artificial
endpoint from the evaluation cohort.  Harrell risk and untruncated IPCW C keep
their legacy definitions; only the time-dependent scores use the aligned grid.
"""
import numpy as np

from cancer_metrics import harrell_c
from cancer_parity import dct_risk


def _outcomes(time, event, name):
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    event = np.asarray(event, dtype=np.float64).reshape(-1)
    if time.size == 0 or event.shape != time.shape:
        raise ValueError(f'{name} outcomes must be nonempty and equally sized')
    if not np.isfinite(time).all() or np.any(time < 0):
        raise ValueError(f'{name} times must be finite and nonnegative')
    if not np.isfinite(event).all() or not np.isin(event, (0, 1)).all():
        raise ValueError(f'{name} event indicators must be zero or one')
    return time, event.astype(bool)


def _edges(bin_edges, n_columns):
    edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    if n_columns < 1 or edges.size != n_columns + 1:
        raise ValueError('Bin edges must have exactly one more entry than survival columns')
    if np.isnan(edges).any() or not np.all(edges[1:] > edges[:-1]):
        raise ValueError('Bin edges must be strictly increasing and contain no NaN')
    if not np.isfinite(edges[1:-1]).all():
        raise ValueError('Internal training-bin edges must be finite')
    return edges


def aligned_grid(train_time, train_event, test_time, bin_edges, n_columns):
    """Return finite training-bin edges and their original survival column IDs.

    Outcomes restrict the admissible range but never create an evaluation
    time.  The grid lies in both observed follow-up ranges, is strictly below
    both maxima, and has positive training-estimated censoring survival G.
    This does not silently truncate evaluation patients: full-outcome support
    errors in the downstream metrics are still reported by the main audit.
    """
    from sksurv.nonparametric import CensoringDistributionEstimator
    from sksurv.util import Surv

    train_t, train_e = _outcomes(train_time, train_event, 'Training')
    test_t = np.asarray(test_time, dtype=np.float64).reshape(-1)
    if test_t.size < 2 or not np.isfinite(test_t).all() or np.any(test_t < 0):
        raise ValueError('At least two finite, nonnegative evaluation times are required')
    edges = _edges(bin_edges, n_columns)
    columns = np.arange(n_columns - 1, dtype=np.int64)
    candidates = edges[1:-1]
    lower = max(float(train_t.min()), float(test_t.min()))
    upper = min(float(train_t.max()), float(test_t.max()))
    keep = (candidates >= lower) & (candidates < upper)
    times, columns = candidates[keep], columns[keep]
    if times.size:
        reference = Surv.from_arrays(event=train_e, time=train_t)
        censoring = CensoringDistributionEstimator().fit(reference)
        g = censoring.predict_proba(times)
        keep = np.isfinite(g) & (g > 0)
        times, columns = times[keep], columns[keep]
    if times.size == 0:
        raise ValueError('No finite training-bin edge has evaluation and censoring support')
    return times, columns


def _finite_float(value):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError('Metric returned a nonfinite value')
    return number


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {'count': 0, 'min': None, 'median': None, 'max': None}
    return {'count': int(values.size), 'min': _finite_float(values.min()),
            'median': _finite_float(np.median(values)), 'max': _finite_float(values.max())}


def _censoring_audit(reference, time, event, errors):
    """Summarize event weights and the events that actually contribute pairs.

    An event's contribution to the IPCW denominator is n_i / G(t_i)^2,
    including same-time event/censor comparisons used by sksurv.  Concentration
    summaries are not independent-sample sizes or confidence intervals.
    """
    from sksurv.nonparametric import CensoringDistributionEstimator

    indices = np.flatnonzero(event)
    pair_counts = np.array([np.count_nonzero((time > time[i]) |
                                            ((time == time[i]) & ~event))
                            for i in indices], dtype=np.int64)
    active = pair_counts > 0
    result = {
        'reference': 'training outcomes only', 'tau': None,
        'status': 'unavailable', 'n_evaluation_events': int(indices.size),
        'n_events_with_comparable_pairs': int(active.sum()),
        'n_events_without_comparable_pairs': int((~active).sum()),
        'training_max_time': float(reference['time'].max()),
        'training_g_at_last_observation': None,
        'g_at_event_times': None, 'unsupported_event_times': None,
        'zero_g_event_times': None, 'ipcw_squared_event_weights': None,
        'contributing_event_weights': None, 'weighted_pair_denominator': None,
        'largest_event_pair_weight_fraction': None,
        'contribution_concentration_effective_event_count': None,
        'concentration_caution': 'Descriptive concentration only; comparable pairs share patients.',
    }
    try:
        censoring = CensoringDistributionEstimator().fit(reference)
        result['training_g_at_last_observation'] = _finite_float(
            censoring.predict_proba(np.array([result['training_max_time']]))[0])
        supported_g = []
        failures = 0
        zero = 0
        for i in indices:
            try:
                g = _finite_float(censoring.predict_proba(np.array([time[i]]))[0])
                supported_g.append(g)
                zero += int(g <= 0)
            except (ValueError, ArithmeticError):
                failures += 1
        result['g_at_event_times'] = _summary(supported_g)
        result['unsupported_event_times'] = failures
        result['zero_g_event_times'] = zero
        if failures or zero:
            raise ValueError('IPCW event weights lack positive training censoring support '
                             f'({failures} unsupported, {zero} zero G)')
        g = np.asarray(supported_g, dtype=np.float64)
        weights = np.square(1.0 / g)
        result['ipcw_squared_event_weights'] = _summary(weights)
        result['contributing_event_weights'] = _summary(weights[active])
        contributions = weights[active] * pair_counts[active]
        denominator = _finite_float(contributions.sum())
        result['weighted_pair_denominator'] = denominator
        if denominator > 0:
            shares = contributions / denominator
            result['largest_event_pair_weight_fraction'] = _finite_float(shares.max())
            result['contribution_concentration_effective_event_count'] = _finite_float(
                1.0 / np.square(shares).sum())
        result['status'] = 'defined' if denominator > 0 else 'no_comparable_pairs'
    except Exception as error:
        errors.append(f'censoring_audit: {type(error).__name__}: {error}')
    return result


def audit_parity_metrics(train_time, train_event, test_time, test_event,
                         survival_bins, bin_edges):
    """Return a separate corrected/time_semantic_audit report, never legacy output.

    The signature mirrors cancer_parity.parity_metrics.  Invalid inputs raise;
    unavailable metrics are None with explicit errors.  No tau, weights, time
    edges or model parameters are fitted/selected on evaluation performance.
    AUC uses all admissible aligned columns (it does not drop the first finite
    bin edge).  The finite-grid IBS is not a full-horizon integral.
    """
    from sksurv.metrics import (brier_score, concordance_index_censored,
                                concordance_index_ipcw, cumulative_dynamic_auc,
                                integrated_brier_score)
    from sksurv.util import Surv

    train_t, train_e = _outcomes(train_time, train_event, 'Training')
    time, event = _outcomes(test_time, test_event, 'Evaluation')
    survival = np.asarray(survival_bins, dtype=np.float64)
    if survival.ndim != 2 or survival.shape[0] != time.size or survival.shape[1] < 1:
        raise ValueError('Survival predictions must have shape [evaluation patients, bins]')
    if not np.isfinite(survival).all() or np.any((survival < 0) | (survival > 1)):
        raise ValueError('Survival predictions must be finite probabilities')
    if np.any(np.diff(survival, axis=1) > 1e-12):
        raise ValueError('Cumulative survival must be nonincreasing across columns')
    edges = _edges(bin_edges, survival.shape[1])
    risk = dct_risk(survival)
    reference = Surv.from_arrays(event=train_e, time=train_t)
    observed = Surv.from_arrays(event=event, time=time)
    result = {
        'protocol': 'corrected/time_semantic_audit',
        'legacy_results_overwritten': False,
        'risk_definition': 'minus the summed survival over every original discrete bin',
        'ipcw_tau': None, 'c_index': None, 'c_index_ipcw': None,
        'brier': None, 'ibs': None, 'iauc': None, 'auc_by_time': None,
        'metric_times': [], 'survival_column_indices': [],
        'evaluation_event_ties_at_metric_times': [], 'errors': [],
        'comparable_pairs': None, 'harrell_counts': None,
        'legacy_comparable_pairs': None, 'legacy_harrell_c': None,
        'pair_count_difference_sksurv_minus_legacy': None,
        'time_mapping': 'S[:, j] is evaluated at finite internal training edge bins[j+1]',
        'grid_rule': 'Training edges only, restricted to shared follow-up range and positive training G.',
        'time_dependent_scope': 'Finite internal boundaries only; no open-bin endpoint or interpolation.',
        'boundary_caution': 'Discrete bin-boundary approximation; exact event/edge ties follow sksurv scoring.',
        'selection_caution': 'This rescore does not undo validation-based checkpoint or model selection.',
    }
    errors = result['errors']
    try:
        scores = concordance_index_censored(event, time, risk, tied_tol=1e-8)
        result['c_index'] = _finite_float(scores[0])
        counts = dict(zip(('concordant', 'discordant', 'tied_risk', 'tied_time'),
                          (int(value) for value in scores[1:])))
        counts['comparable_pairs'] = sum(counts[key] for key in
                                         ('concordant', 'discordant', 'tied_risk'))
        result['harrell_counts'] = counts
        result['comparable_pairs'] = counts['comparable_pairs']
    except Exception as error:
        errors.append(f'c_index: {type(error).__name__}: {error}')
    try:
        legacy = harrell_c(time, event.astype(float), risk)
        result['legacy_comparable_pairs'] = int(legacy['comparable_pairs'])
        result['legacy_harrell_c'] = (None if legacy['value'] is None else
                                      _finite_float(legacy['value']))
        if result['comparable_pairs'] is not None:
            result['pair_count_difference_sksurv_minus_legacy'] = (
                result['comparable_pairs'] - result['legacy_comparable_pairs'])
    except Exception as error:
        errors.append(f'legacy_harrell: {type(error).__name__}: {error}')
    try:
        # Deliberately identical to the legacy untruncated IPCW call.
        result['c_index_ipcw'] = _finite_float(
            concordance_index_ipcw(reference, observed, risk)[0])
    except Exception as error:
        errors.append(f'c_index_ipcw: {type(error).__name__}: {error}')
    result['censoring_audit'] = _censoring_audit(reference, time, event, errors)
    try:
        grid, columns = aligned_grid(train_t, train_e, time, edges, survival.shape[1])
        predictions = survival[:, columns]
        result['metric_times'] = grid.tolist()
        result['survival_column_indices'] = columns.tolist()
        result['evaluation_event_ties_at_metric_times'] = [
            int(np.count_nonzero(event & (time == point))) for point in grid]
    except Exception as error:
        errors.append(f'time_grid: {type(error).__name__}: {error}')
        return result
    try:
        _, values = brier_score(reference, observed, predictions, grid)
        result['brier'] = [_finite_float(value) for value in values]
    except Exception as error:
        errors.append(f'brier: {type(error).__name__}: {error}')
    try:
        if grid.size < 2:
            raise ValueError('IBS requires at least two admissible finite training-bin edges')
        result['ibs'] = _finite_float(
            integrated_brier_score(reference, observed, predictions, grid))
    except Exception as error:
        errors.append(f'ibs: {type(error).__name__}: {error}')
    try:
        if any(not np.any(event & (time <= point)) or not np.any(time > point) for point in grid):
            raise ValueError('Every AUC time needs an observed case and a later control')
        values, integrated = cumulative_dynamic_auc(reference, observed, 1 - predictions, grid)
        finite_values = [_finite_float(value) for value in values]
        finite_integrated = _finite_float(integrated)
        result['auc_by_time'], result['iauc'] = finite_values, finite_integrated
    except Exception as error:
        errors.append(f'iauc: {type(error).__name__}: {error}')
    return result
