"""Selection-free epoch statistics and the pure-noise null of the protocol.

The published comparison table reports one number per arm: the **maximum**
validation Harrell C over the training epochs.  That number is selected on the
same patients that are scored, so it mixes two effects:

* the signal the arm carries, and
* the upward bias of taking the maximum of a noisy trajectory.

This module provides the two quantities that separate them:

``epoch_statistics``
    The mean, spread and maximum of one arm's validation C-index over its
    epochs.  The mean is **selection-free**: it does not look at the
    validation set any harder than the maximum does, but it is not chosen
    by it.  The difference between the maximum and the mean is the
    selection inflation of that arm.

``noise_null``
    The null of the whole protocol: random risk vectors, scored with the
    **same metric call** on the **same cohort**, over the **same number of
    epochs**.  A pure-noise arm does not sit at 0.5 in this table; the
    maximum over 30 noisy draws does.  The null therefore states how large
    a score a protocol with no signal at all can produce, which is the
    reference every published maximum has to be read against.

Nothing here trains, writes into a run directory or opens the outer test
partition.  The functions read a finished epoch history and the finished
validation labels.
"""
from __future__ import annotations

import statistics

import numpy as np


def epoch_statistics(history):
    """Summarize one arm's per-epoch validation C-index.

    ``history`` is the runner's own epoch list, where every entry carries a
    ``c_index``.  The statistics are computed from those recorded values, so
    they describe the trajectory the runner actually produced and not a
    re-evaluation of anything.

    Returns ``None`` when the history does not hold at least two usable
    values, because a mean over fewer than two epochs is not a trajectory
    summary and must not be printed as one.
    """
    values = [entry.get('c_index') for entry in history or []
              if isinstance(entry, dict) and entry.get('c_index') is not None]
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return {'epochs': len(values),
            'mean_c_index': float(mean),
            'sd_c_index': float(statistics.pstdev(values)),
            'min_c_index': float(min(values)),
            'max_c_index': float(max(values)),
            'last_c_index': float(values[-1]),
            'selection_inflation': float(max(values) - mean)}


def noise_null(n_patients, events, times, trials=1000, epochs=30, seed=20261117):
    """The null of the whole protocol: random risk, same metric, same epochs.

    Two statistics are reported, because the published table uses the second
    one and the first one is the honest summary of a run:

    * ``epoch_mean``: the mean Harrell C over the epochs.  An arm that
      carries no signal sits at the chance level here.
    * ``epoch_max``: the maximum over the epochs, which is what the
      validation-selected score is.  Even a pure random risk reaches a high
      maximum, because the maximum of 30 noisy draws is biased upwards.

    ``events`` and ``times`` are the validation outcomes; ``events`` is a
    boolean array.  The metric is the repository's shared concordance call,
    so the null is produced by exactly the code that produces the arm scores.
    """
    from sksurv.metrics import concordance_index_censored

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    events = np.asarray(events).reshape(-1).astype(bool)
    if times.size != n_patients or events.size != n_patients:
        raise ValueError('The null needs one time and one event per patient')
    if epochs < 1 or trials < 1:
        raise ValueError('The null needs a positive trial and epoch count')
    generator = np.random.default_rng(seed)
    means, maxes = [], []
    for _ in range(trials):
        scores = []
        for _ in range(epochs):
            risk = generator.random(n_patients)
            try:
                scores.append(float(concordance_index_censored(events, times, risk,
                                                                tied_tol=1e-8)[0]))
            except Exception:
                continue
        if not scores:
            continue
        means.append(statistics.fmean(scores))
        maxes.append(max(scores))
    if not means:
        return None
    means = np.asarray(means)
    maxes = np.asarray(maxes)
    return {'trials': len(means), 'epochs': epochs,
            'epoch_mean': {'mean': float(means.mean()), 'sd': float(means.std()),
                           'p95': float(np.percentile(means, 95)),
                           'p99': float(np.percentile(means, 99))},
            'epoch_max': {'mean': float(maxes.mean()), 'sd': float(maxes.std()),
                          'p95': float(np.percentile(maxes, 95)),
                          'p99': float(np.percentile(maxes, 99))}}
