"""Numerical risk diagnostics with explicit legacy metric conventions."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from cancer_metrics import harrell_c
from cancer_model import piecewise_exposure, survival_at


METRIC_CONVENTIONS = {
    'harrell_c': 'strict observed-time pairs: t_j > t_i and event_i=1; exact risk ties score 0.5',
    'ipcw_brier': 'legacy_v1: training censor KM with all ties in risk set; event G(t-), survivor G(h); divide by all evaluation subjects',
    'selection': 'validation NLL; strict decrease retains earliest tied minimum',
}


def cumulative_hazard_at(log_rates, horizon, starts):
    """Compute H(h) in float64 directly, without recovering it from rounded S."""
    if not math.isfinite(horizon) or horizon < 0:
        raise ValueError('Expected a finite nonnegative horizon')
    if (log_rates.ndim != 2 or starts.ndim != 1 or log_rates.shape[1] != len(starts)
            or not log_rates.is_floating_point() or not torch.isfinite(log_rates).all()):
        raise ValueError('Expected finite floating log-rate logits matching interval starts')
    with torch.no_grad():
        values = log_rates.detach().double()
        exposure = piecewise_exposure(values.new_tensor([horizon]), starts.to(values).detach())[0]
        hazard = (F.softplus(values)+1e-8) @ exposure
        if not torch.isfinite(hazard).all():
            raise ValueError('Cumulative hazard overflow')
        return hazard


def horizon_risk_audit(log_rates, time, event, starts, horizon):
    """Keep legacy probability-risk C alongside numerically stable H-risk C.

    H(h) has the same mathematical ranking as 1-exp(-H(h)). It is a risk score
    for full-follow-up strict-time concordance, not a time-dependent C-index.
    """
    hazard = cumulative_hazard_at(log_rates, horizon, starts).cpu().numpy()
    with torch.no_grad():
        survival = survival_at(log_rates, log_rates.new_tensor([horizon]), starts.to(log_rates))[:, 0].cpu().numpy()
    risk = 1-survival
    t = np.asarray(time.detach().cpu() if isinstance(time, torch.Tensor) else time)
    e = np.asarray(event.detach().cpu() if isinstance(event, torch.Tensor) else event)
    legacy, stable = harrell_c(t, e, risk), harrell_c(t, e, hazard)
    collapsed = 0
    for i in range(len(t)):
        comparable = (t > t[i]) & (e[i] == 1)
        collapsed += int(((risk == risk[i]) & (hazard != hazard[i]) & comparable).sum())
    return {'horizon': float(horizon), 'legacy_probability_risk_c': legacy,
            'cumulative_hazard_risk_c': stable,
            'probability_risk_exactly_one': int((risk == 1).sum()),
            'probability_risk_exactly_zero': int((risk == 0).sum()),
            'comparable_pairs_tied_only_in_probability_risk': collapsed,
            'unique_probability_risk_values': int(len(np.unique(risk))),
            'unique_cumulative_hazard_values': int(len(np.unique(hazard))),
            'risk_definition': 'H(h)=sum_k softplus(logit_k)+1e-8 times interval exposure; computed in float64',
            'scope': 'full follow-up strict-time concordance using horizon risk; not time-dependent C'}
