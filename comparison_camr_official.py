"""CAMR with the upstream survival contract: bounded scalar Cox output.

Why this module exists
----------------------
The existing ``camr_2m`` arm is a *protocol-parity* adaptation: it keeps the
upstream encoder, gates, low-rank fusion, reconstruction heads and
discriminator, but replaces the survival head with unbounded 4-bin discrete
hazard logits so that every arm is trained and scored by the same
``patientwise_clamped_discrete_v1`` likelihood.  Its absolute score therefore
cannot be read as "what CAMR does".

This module restores the **official upstream survival contract**:

1. **Bounded scalar output.**  Upstream builds
   ``classifier = Sequential(Linear(15, 1), Sigmoid())`` and then rescales it
   with ``output_range = 8`` and ``output_shift = -4``, so the model emits one
   risk value in ``(-4, 4)`` per patient.  This module keeps that head exactly
   as upstream writes it.
2. **Continuous Cox partial likelihood.**  Upstream trains that scalar with
   ``CoxLoss1``/``CoxLoss2``.  This module uses the *correct* time-based risk
   set, i.e. the ``CoxLoss1`` formulation, in which the risk set of patient
   ``i`` is ``{j : t_j >= t_i}``.  Upstream's ``CoxLoss2`` instead calls
   ``utils.R_set``, which builds ``torch.tril(torch.ones(n, n))`` - a matrix
   indexed by *row position* rather than by survival time.  That upstream
   helper is not used here; using it would make the risk set depend on the
   arbitrary order of the patients inside a batch.

Scoring the bounded scalar under the shared protocol
----------------------------------------------------
The shared evaluation code needs a ``[patients, bins]`` survival matrix, so
the scalar risk is projected into the protocol's four train-fitted time bins
with the standard proportional-hazards construction: the baseline cumulative
hazard ``H0`` is estimated on the **training fold** with the Breslow
estimator, and

    S_x(t) = exp(-H0(t) * exp(risk_x)).

The implied discrete bin hazards are ``1 - S(t_k) / S(t_{k-1})``, from which
the discrete hazard logits follow.  The projection is monotone in the scalar
risk, so the ranking metrics are unchanged by it; only the calibrated
probabilities are added.  Two conventions are stated explicitly because they
are not identical to the discrete arms:

* the last protocol bin is open-ended in the discrete arms, so its column is
  evaluated here at the largest observed **training** time;
* the baseline is estimated from training-fold outcomes only; no validation
  or test patient contributes to it.

The remaining deviations from upstream are the ones already disclosed for
``camr_2m`` and are listed in :meth:`CAMR2MOfficial.adaptation_metadata`:
BLCA has no CNA input, the 4096-row patch budget must be pooled into upstream's
80-dimensional patient vector, and the protocol-level optimizer and schedule
are shared with every other arm.
"""
import math

import numpy as np
import torch
from torch import nn

from comparison_camr import CAMR2M, INPUT_DIM, UPSTREAM_COMMIT, UPSTREAM_URL

COX_PARTIAL_LIKELIHOOD_VERSION = 'cox_partial_likelihood_brewslow_v1'
METHOD_NAME = 'CAMR-2M-P4096-COX'

# Upstream's own survival output head, restored verbatim.
UPSTREAM_OUTPUT_RANGE = 8.0
UPSTREAM_OUTPUT_SHIFT = -4.0
UPSTREAM_CLASSIFIER_IN = 15


class CAMR2MOfficial(CAMR2M):
    """CAMR with the upstream bounded scalar Cox output.

    Everything except the survival head is inherited from ``CAMR2M``, so the
    two CAMR arms differ **only** in the head and in the survival loss.  That
    makes the pair a controlled comparison of the protocol-parity adaptation
    against the upstream contract.
    """

    method_name = METHOD_NAME
    survival_loss_version = COX_PARTIAL_LIKELIHOOD_VERSION

    def __init__(self, path_dim, rna_dim, bins=4, input_dim=INPUT_DIM, dropout=.1):
        super().__init__(path_dim, rna_dim, bins=bins, input_dim=input_dim, dropout=dropout)
        # Restore the upstream head: one sigmoid unit rescaled to [-4, 4].
        self.upstream_model.classifier = nn.Sequential(
            nn.Linear(UPSTREAM_CLASSIFIER_IN, 1), nn.Sigmoid())
        with torch.no_grad():
            self.upstream_model.output_range.fill_(UPSTREAM_OUTPUT_RANGE)
            self.upstream_model.output_shift.fill_(UPSTREAM_OUTPUT_SHIFT)

    # ------------------------------------------------------------------ output
    def forward(self, patches, rna):
        output = super().forward(patches, rna)
        risk = output['logits'].reshape(-1)
        if risk.shape[0] != rna.shape[0]:
            raise ValueError('The Cox head must emit one risk value per patient')
        if not torch.isfinite(risk).all():
            raise ValueError('Nonfinite Cox risk')
        bounded = ((risk - UPSTREAM_OUTPUT_SHIFT) / UPSTREAM_OUTPUT_RANGE)
        if bool(((bounded < 0.) | (bounded > 1.)).any()):
            raise ValueError('The Cox head escaped its bounded sigmoid range')
        output['risk'] = risk
        return output

    # -------------------------------------------------------------- metadata
    def adaptation_metadata(self):
        metadata = super().adaptation_metadata()
        metadata.update({
            'method': METHOD_NAME,
            'survival_head': 'upstream sigmoid scalar rescaled to [-4, 4]',
            'survival_loss': 'continuous Cox partial likelihood with a time-based risk set',
            'survival_loss_version': COX_PARTIAL_LIKELIHOOD_VERSION,
            'upstream_risk_set_helper_reused': False,
            'upstream_risk_set_helper_note': (
                "Upstream's CoxLoss2 calls utils.R_set, which builds a row-index "
                "lower-triangular matrix instead of a time-based risk set; this arm uses "
                "the time-based risk set of upstream's CoxLoss1 instead."),
            'evaluation_projection': (
                'Breslow baseline cumulative hazard fitted on the training fold; '
                'S(t) = exp(-H0(t) * exp(risk)); the projection is monotone in the risk.'),
            'evaluation_projection_deviations': [
                'The last protocol bin is open-ended in the discrete arms; its column is '
                'evaluated here at the largest observed training time.',
                'The Breslow baseline uses training-fold outcomes only.',
            ],
            'limits': [
                'The absolute score is not comparable with the CAMR paper because the '
                'cohort, the patch budget, the modality set and the protocol differ.',
                'Two modalities only: the third upstream slot carries a constant vector '
                'without patient information.',
                'The 4096-row patch budget is pooled into upstream\'s 80-dimensional '
                'patient vector by attention MIL, which upstream does not do.',
                'The optimizer, schedule and batch schedule are the shared protocol ones, '
                'not upstream\'s Adam 4e-4 with a linear decay.',
                'The discriminator receives its own optimizer here; upstream builds it '
                'but never steps it.',
            ],
        })
        return metadata


# --------------------------------------------------------------- Cox likelihood

def cox_partial_likelihood(risk, time, event, reduction='mean'):
    """Breslow partial likelihood with an explicit time-based risk set.

    ``risk`` is the bounded scalar Cox output, ``time`` the follow-up time and
    ``event`` the observed-event indicator (1 marks an event, matching the
    repository convention).  The risk set of patient ``i`` is
    ``{j : t_j >= t_i}``, so a patient's contribution depends on the *times*
    of the batch and never on another patient's event flag.

    The ``[B, B]`` matrix below is the risk set itself, not a broadcast: each
    entry is a comparison of two follow-up times, and the row-wise log-sum-exp
    is the definition of the Cox partial likelihood.  This is deliberately
    different from the historical discrete-NLL bug, where a per-patient loss
    was multiplied by *another* patient's censoring weight.

    ``reduction='none'`` returns one value per patient, which makes the
    isolation property checkable: a censored patient's row is zero, and
    changing any other patient's event flag leaves this patient's value
    bit-identical.
    """
    if risk.ndim != 1 or risk.shape[0] < 1 or not risk.is_floating_point():
        raise ValueError('Cox risk must be a nonempty one-dimensional float tensor')
    if not torch.isfinite(risk).all():
        raise ValueError('Nonfinite Cox risk')
    t = torch.as_tensor(time, device=risk.device, dtype=risk.dtype).reshape(-1)
    e = torch.as_tensor(event, device=risk.device, dtype=risk.dtype).reshape(-1)
    if t.shape != risk.shape or e.shape != risk.shape:
        raise ValueError('Time and event must provide one value per patient')
    if not torch.isfinite(t).all() or bool((t < 0).any()):
        raise ValueError('Follow-up times must be finite and nonnegative')
    if not torch.isfinite(e).all() or not bool(((e == 0) | (e == 1)).all()):
        raise ValueError('Event indicators must be zero or one')
    mask = t[None, :] >= t[:, None]                       # R[i, j] = t_j >= t_i
    negative_infinity = torch.tensor(float('-inf'), dtype=risk.dtype, device=risk.device)
    masked = torch.where(mask, risk[None, :].expand_as(mask), negative_infinity)
    log_denominator = torch.logsumexp(masked, dim=1)
    loss = -(risk - log_denominator) * e
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    raise ValueError('Invalid reduction')


# ------------------------------------------------------------ Breslow baseline

def fit_brewslow_baseline(time, event, risk, column_times):
    """Cumulative baseline hazard ``H0`` at the requested times.

    The estimator uses the training-fold outcomes and the model's training
    risks only.  ``column_times`` must be strictly increasing and finite.
    """
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    e = np.asarray(event, dtype=np.float64).reshape(-1)
    r = np.asarray(risk, dtype=np.float64).reshape(-1)
    q = np.asarray(column_times, dtype=np.float64).reshape(-1)
    if t.size == 0 or e.shape != t.shape or r.shape != t.shape:
        raise ValueError('Baseline estimation needs one time, event and risk per patient')
    if q.size < 1 or not np.isfinite(q).all() or np.any(np.diff(q) <= 0):
        raise ValueError('Column times must be finite and strictly increasing')
    if not np.isfinite(t).all() or not np.isfinite(r).all():
        raise ValueError('Baseline estimation needs finite times and risks')
    weights = np.exp(np.clip(r, -50., 50.))
    event_times = np.unique(t[e > .5])
    if event_times.size == 0:
        raise ValueError('The training fold has no observed event to anchor the baseline')
    increments = np.empty(event_times.size, dtype=np.float64)
    for index, moment in enumerate(event_times):
        risk_set = weights[t >= moment].sum()
        if not np.isfinite(risk_set) or risk_set <= 0:
            raise ValueError('Degenerate Cox risk set in the Breslow estimator')
        increments[index] = float(np.count_nonzero((t == moment) & (e > .5))) / float(risk_set)
    cumulative = np.cumsum(increments)
    positions = np.searchsorted(event_times, q, side='right') - 1
    values = np.where(positions >= 0, cumulative[np.clip(positions, 0, cumulative.size-1)], 0.)
    return values


def cox_discrete_logits(risk, baseline_cumulative, eps=1e-7):
    """Project Cox risks onto the protocol's discrete bin hazards.

    ``baseline_cumulative`` holds ``H0`` at the four bin column times.  The
    implied proportional-hazards hazard of bin ``k`` is
    ``1 - exp(-(H0(t_k) - H0(t_{k-1})) * exp(risk))``; the returned logits are
    the log-odds of those hazards, so the shared discrete survival and the
    shared discrete NLL apply unchanged.
    """
    values = np.asarray(baseline_cumulative, dtype=np.float64).reshape(-1)
    if values.size < 1 or not np.isfinite(values).all() or np.any(np.diff(values) < -1e-9):
        raise ValueError('The baseline cumulative hazard must be finite and nondecreasing')
    if not isinstance(risk, torch.Tensor):
        raise ValueError('Cox risk must be a tensor')
    logits = torch.empty((risk.shape[0], values.size), dtype=risk.dtype, device=risk.device)
    previous = 0.
    relative = torch.exp(torch.clamp(risk, -50., 50.)).to(torch.float64)
    for index, level in enumerate(values):
        delta = float(level) - previous
        previous = float(level)
        hazard = 1. - torch.exp(-delta * relative)
        hazard = torch.clamp(hazard, eps, 1. - eps).to(risk.dtype)
        logits[:, index] = torch.log(hazard) - torch.log1p(-hazard)
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite projected discrete hazard logits')
    return logits
