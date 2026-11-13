"""DCT-comparable protocol pieces: patch budget, discrete bins and metrics.

These helpers reproduce the externally defined conventions of the reference
DCT pipeline so that a comparison isolates the model and the loss, not the
data preparation or the scoring convention.  Nothing here reads or writes
patient features; callers supply already verified bags and labels.

Every convention implemented here was read from the reference source, not
guessed: the quantile binning, the discrete negative log likelihood, the risk
definition, the metric time grid and the censoring reference.
"""
import math

import numpy as np
import pandas as pd
import torch

from cancer_metrics import harrell_c


DCT_ENDPOINT = 'DSS'
DCT_PATCH_BUDGETS = (2048, 4096)


def _json_float(value):
    """Represent an infinite bin edge in a JSON-safe way."""
    number = float(value)
    if math.isinf(number):
        return 'Infinity' if number > 0 else '-Infinity'
    if math.isnan(number):
        return 'NaN'
    return number


# ---------------------------------------------------------------- patch budget

def select_patch_rows(total, num_patches, mode, seed=0, patient_id=''):
    """Reproduce the reference patch-budget convention.

    ``train`` draws ``num_patches`` rows without replacement and keeps them
    sorted; ``eval`` takes evenly spaced rows.  Both are capped by the number
    of available rows and the shortfall becomes explicit zero padding.
    """
    if type(total) is not int or total < 1:
        raise ValueError('A patient bag needs at least one patch row')
    if type(num_patches) is not int or num_patches < 1:
        raise ValueError('The patch budget must be a positive integer')
    if mode not in ('train', 'eval'):
        raise ValueError('Patch budget mode must be train or eval')
    count = min(num_patches, total)
    if mode == 'train':
        rng = np.random.default_rng(abs(hash((int(seed), str(patient_id)))) % (2**63))
        rows = np.sort(rng.choice(total, count, replace=False))
    else:
        rows = np.floor(np.arange(count)*total/count).astype(np.int64)
    return rows.astype(np.int64), int(num_patches-count)


# ------------------------------------------------------------------- bin edges

def dct_bin_edges(train_time, train_event, n_bins=4):
    """Reproduce the reference train-only quantile bin edges.

    Only uncensored training times define the range.  The outer edges become
    infinite so that every patient receives a label.  The edges are the ones
    the reference implementation stores as ``dataset_factory.bins`` and reuses
    for the metric time grid.
    """
    t = np.asarray(train_time, dtype=float).reshape(-1)
    e = np.asarray(train_event, dtype=float).reshape(-1)
    if t.size == 0 or e.shape != t.shape:
        raise ValueError('Invalid training survival data')
    # ``event`` follows the repository convention: 1 marks an observed event.
    values = pd.Series(t[e > .5]).dropna()
    if values.nunique() < n_bins:
        raise ValueError(f'Cannot fit {n_bins} survival bins from {values.nunique()} unique times')
    _, edges = pd.qcut(values, q=n_bins, retbins=True, labels=False, duplicates='drop')
    if len(edges)-1 != n_bins:
        raise ValueError(f'Quantile edges produced {len(edges)-1} bins; expected {n_bins}')
    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def discretize_labels(time, bin_edges):
    """Apply the reference bin edges to every patient, censored or not."""
    values = pd.Series(np.asarray(time, dtype=float).reshape(-1))
    labels = pd.cut(values, bins=np.asarray(bin_edges, dtype=float), labels=False,
                    right=False, include_lowest=True)
    if labels.isna().any():
        raise ValueError('Survival binning produced NaN labels')
    return labels.astype('int64').to_numpy()


# -------------------------------------------------------------- discrete model

def discrete_survival(logits):
    """Cumulative survival of the reference discrete hazard parameterization."""
    hazards = torch.sigmoid(logits)
    return torch.cumprod(1-hazards, dim=1)


def discrete_nll(logits, labels, censorship, alpha=.5, eps=1e-7, reduction='mean'):
    """Reproduce the reference discrete-time negative log likelihood.

    ``censorship`` is 1 for a censored patient and 0 for an observed event,
    matching the reference convention where the uncensored term is weighted
    by ``alpha``.
    """
    if logits.ndim != 2 or logits.shape[0] != len(labels) or logits.shape[1] < 1:
        raise ValueError('Incompatible shapes for the discrete survival likelihood')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite hazard logits')
    y = torch.as_tensor(labels, dtype=torch.long)
    c = torch.as_tensor(censorship, dtype=torch.float32)
    if y.shape != c.shape or y.numel() != logits.shape[0]:
        raise ValueError('Labels and censorship must match the batch')
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1-hazards, dim=1)
    padded = torch.cat([torch.ones_like(c.unsqueeze(1)), survival], dim=1)
    previous = torch.gather(padded, 1, y.unsqueeze(1)).clamp_min(eps)
    this_hazard = torch.gather(hazards, 1, y.unsqueeze(1)).clamp_min(eps)
    this_survival = torch.gather(padded, 1, (y+1).unsqueeze(1)).clamp_min(eps)
    uncensored = -(1-c)*(torch.log(previous)+torch.log(this_hazard))
    censored = -c*torch.log(this_survival)
    negative = censored+uncensored
    loss = (1-alpha)*negative+alpha*uncensored
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    raise ValueError('Invalid reduction')


# ---------------------------------------------------------------------- risks

def dct_risk(survival):
    """Reproduce the reference risk convention: minus the summed survival."""
    values = np.asarray(survival, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError('Expected a [patients, bins] survival matrix')
    return -values.sum(axis=1)


# --------------------------------------------------------------------- metrics

def metric_time_grid(event_times, train_times, bin_edges, n_columns):
    """Reproduce the reference evaluation time grid.

    The candidates mix the observed outcome range with the inner bin edges and
    are restricted to the follow-up range shared by training and evaluation
    rows.  The returned mask selects the matching survival columns.
    """
    times = np.asarray(event_times, dtype=float).reshape(-1)
    train = np.asarray(train_times, dtype=float).reshape(-1)
    edges = np.asarray(bin_edges, dtype=float).reshape(-1)
    columns = int(n_columns)
    if times.size < 2:
        raise ValueError('At least two evaluation outcomes are required')
    if columns < 2 or edges.size < columns+1:
        raise ValueError('Discrete survival bins do not match the prediction columns')
    candidates = np.concatenate(([times.min()+1e-4], edges[1:columns-1], [times.max()-1e-4]))
    lower = max(float(times.min()), float(train.min()))
    upper = min(float(times.max()), float(train.max()))
    valid = np.isfinite(candidates) & (candidates >= lower) & (candidates < upper)
    grid = candidates[valid]
    if grid.size == 0:
        raise ValueError('No evaluation time lies inside both follow-up ranges')
    if np.any(np.diff(grid) <= 0):
        raise ValueError('Evaluation times must be strictly increasing')
    return grid, valid


def parity_metrics(train_time, train_event, test_time, test_event, survival_bins, bin_edges):
    """Harrell C, IPCW C, Brier, IBS and dynamic AUC on the reference grid.

    The censoring reference is fitted on the training rows only.  Failures are
    reported explicitly instead of being replaced by zeros, so a missing number
    can never be read as a poor score.
    """
    from sksurv.metrics import (brier_score, concordance_index_censored,
                                concordance_index_ipcw, cumulative_dynamic_auc,
                                integrated_brier_score)
    from sksurv.util import Surv

    t = np.asarray(test_time, dtype=float).reshape(-1)
    e = np.asarray(test_event, dtype=float).reshape(-1)
    train_t = np.asarray(train_time, dtype=float).reshape(-1)
    train_e = np.asarray(train_event, dtype=float).reshape(-1)
    survival = np.asarray(survival_bins, dtype=np.float64)
    if survival.ndim != 2 or survival.shape[0] != t.size:
        raise ValueError('Survival predictions must have one row per evaluation patient')
    risk = dct_risk(survival)
    result = {'c_index': None, 'c_index_ipcw': None, 'brier': None, 'ibs': None,
              'iauc': None, 'comparable_pairs': None, 'errors': [], 'metric_times': []}
    # The repository convention marks an observed event with 1, so the
    # censoring indicator is its complement.
    event = (e > .5).astype(bool)
    try:
        result['c_index'] = float(concordance_index_censored(event, t, risk, tied_tol=1e-8)[0])
    except Exception as error:
        result['errors'].append(f'c_index: {type(error).__name__}: {error}')
    reference = Surv.from_arrays(event=(train_e > .5).astype(bool), time=train_t)
    observed = Surv.from_arrays(event=event, time=t)
    try:
        result['c_index_ipcw'] = float(concordance_index_ipcw(reference, observed, risk)[0])
    except Exception as error:
        result['errors'].append(f'c_index_ipcw: {type(error).__name__}: {error}')
    try:
        grid, valid = metric_time_grid(t, train_t, bin_edges, survival.shape[1])
        predictions = survival[:, valid]
        result['metric_times'] = [float(x) for x in grid]
        _, brier = brier_score(reference, observed, predictions, grid)
        # The reference implementation returns the per-time Brier scores; the
        # integrated value is a separate number.
        result['brier'] = [float(x) for x in np.asarray(brier).reshape(-1)]
        if grid.size >= 2:
            result['ibs'] = float(integrated_brier_score(reference, observed, predictions, grid))
            _, iauc = cumulative_dynamic_auc(reference, observed, 1-predictions[:, 1:], grid[1:])
            result['iauc'] = float(iauc)
    except Exception as error:
        result['errors'].append(f'time_dependent: {type(error).__name__}: {error}')
    legacy = harrell_c(t, e, risk)
    result['comparable_pairs'] = legacy['comparable_pairs']
    result['harrell_c_local_implementation'] = legacy['value']
    return result


def parity_report_protocol(edges, num_patches, endpoint, split_name, n_bins):
    """Explicit statement of the conventions a parity number is produced under."""
    return {
        'protocol': 'dct_parity_legacy_val',
        'endpoint': endpoint,
        'split': split_name,
        'num_patches': num_patches,
        'n_bins': n_bins,
        'bin_edges': [_json_float(x) for x in edges],
        'risk_definition': 'minus the summed survival over the discrete bins',
        'censoring_reference': 'training rows only',
        'metric_implementation': 'sksurv (concordance_index_censored, concordance_index_ipcw, '
                                 'brier_score, integrated_brier_score, cumulative_dynamic_auc)',
        'selection': 'checkpoint selection maximizes the validation C-index; the earliest epoch wins a tie',
        'scope': 'legacy validation cohort; not an independent outer test set',
        'caution': 'a validation-selected C-index is optimistically biased and is not an '
                   'independent test estimate',
    }
