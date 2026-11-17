"""Synthetic data generator for method 3.2 (CPU only, no patient data).

All generation mechanisms are explicit and reproducible from a seed.  The
generated ground truth is used for auditing only; it is never fed to the model.

Design of the generative model
------------------------------
There are ``K=4`` morphological clusters.  A latent prognostic factor ``z``
drives both the cluster-0/cluster-1 patch appearance and the RNA value, so the
cross-modal pair is genuinely redundant.  Cluster 2 carries a *different*
latent factor ``w`` that is absent from the RNA, and cluster 3 is pure
nuisance.  Consequently a correct cross-modal retrieval measure must give
cluster 0 and 1 a high rate and clusters 2 and 3 a rate near chance; a measure
that is fooled by everything would return high rates everywhere.

Scenarios
---------
Every scenario declares **three** things explicitly, because a scenario is only
meaningful if all three differ from the others:

* ``path_latent``   what the patch features encode about the patient;
* ``rna_latent``    what the RNA encodes;
* ``eta``           what drives the survival hazard.

``redundant``     path encodes z, RNA encodes z, risk from z.
``path_only``     path encodes z, RNA encodes **noise only**, risk from z.
``rna_only``      path encodes **no prognostic latent**, RNA encodes z, risk from RNA.
``no_signal``     path encodes z, RNA encodes noise, risk from **nothing**
                  (negative control: cross-modal structure exists but is not prognostic).
``xor``           path encodes x_sign (cluster 0 only), RNA encodes y_sign,
                  risk from ``x_sign * y_sign``; each modality alone is useless.
``xor_additive``  same marginals as ``xor`` but risk from ``x_sign + y_sign``.
``nonlinear``     path encodes ``z**2 - 1``, RNA encodes ``z``, risk from ``z**2 - 1``:
                  the linear correlation between the modalities is zero but the
                  dependence is real.
``composition``   path encodes z, RNA encodes noise, risk from the **cluster
                  composition** only.

A previous revision of this file gave all of ``redundant``, ``path_only`` and
``rna_only`` the same RNA (``z``) and the same patch latent (``z``), so those
three scenarios produced **bit-identical cohorts** and their benchmark columns
were not three conditions but one condition three times.  The mechanism table
below replaces that behaviour and is asserted by the unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from method3_2_model import multiplicative_replacement_clr


CLUSTERS = 4
# Coordinate layout: 0..3 cluster identity, 4 shared latent z, 5 cluster-2
# latent w, 6..7 pure noise.  Keeping the number of pure-noise coordinates
# small is required for the *known-answer* atlas check to be well posed: after
# standardisation a cluster separation of ``CLUSTER_OFFSET`` in raw units is
# divided by the raw standard deviation, and with many noise coordinates the
# noise ball grows past that separation and the generative clusters stop being
# separable at all.  This is a property of the synthetic fixture, not of the
# atlas.
PATCH_DIM = 8
RNA_DIM = 6
BIN_EDGES = (0.0, 6.0, 12.0, 24.0, 60.0)
BASE_HAZARD = (0.05, 0.08, 0.12, 0.18)
CLUSTER_OFFSET = 12.0
# Feature-coordinate layout.  Coordinates 0..3 define the cluster identity,
# coordinate 4 carries the shared latent z, coordinate 5 carries the
# cluster-2-only latent w, and the remaining coordinates are noise.  Keeping
# the latent axes disjoint from the cluster axes prevents the latent from
# perturbing the cluster assignment itself.
LATENT_AXIS = 4
CLUSTER2_AXIS = 5
# How strongly each cluster expresses the shared latent z (cluster 2 carries w
# instead, cluster 3 is nuisance).
CLUSTER_Z_STRENGTH = (1.0, 0.7, 0.0, 0.0)
# Which cluster carries the XOR pathology sign (only used by xor scenarios).
XOR_CLUSTER = 0

# ---------------------------------------------------------------------------
# scenario mechanism table
# ---------------------------------------------------------------------------
# ``path``  : which latent the patch features express
#              'z'      -> clusters 0/1 express z (the shared latent)
#              'z2'     -> clusters 0/1 express z**2 - 1 (nonlinear, mean-free)
#              'x'      -> cluster 0 expresses x_sign
#              'none'   -> the patch features carry no prognostic latent
# ``rna``   : which latent the RNA expresses
#              'z'      -> RNA[0] = z
#              'y'      -> RNA[0] = y_sign
#              'noise'  -> RNA[0] = independent noise
# ``risk``  : what drives the hazard
#              'z'      -> eta = 1.3 * z
#              'rna'    -> eta = 1.3 * RNA[0]
#              'xor'    -> eta = 1.6 * (x_sign * y_sign)
#              'add'    -> eta = 0.8 * (x_sign + y_sign)
#              'nonlin' -> eta = 1.3 * (z**2 - 1)
#              'comp'   -> eta = 6.0 * (cluster_probs[3] - 0.25)
#              'none'   -> eta = 0
SCENARIO_MECHANISMS = {
    'redundant':    {'path': 'z',     'rna': 'z',    'risk': 'z'},
    'path_only':    {'path': 'z',     'rna': 'noise', 'risk': 'z'},
    'rna_only':     {'path': 'none',  'rna': 'z',    'risk': 'rna'},
    'no_signal':    {'path': 'z',     'rna': 'noise', 'risk': 'none'},
    'xor':          {'path': 'x',     'rna': 'y',    'risk': 'xor'},
    'xor_additive': {'path': 'x',     'rna': 'y',    'risk': 'add'},
    'nonlinear':    {'path': 'z2',    'rna': 'z',    'risk': 'nonlin'},
    'composition':  {'path': 'z',     'rna': 'noise', 'risk': 'comp'},
}


@dataclass
class SyntheticCohort:
    ids: list
    bags: list                 # list of [n_i, PATCH_DIM] float64 tensors
    rna: torch.Tensor          # [N, RNA_DIM]
    times: torch.Tensor        # [N]
    events: torch.Tensor       # [N] 1 = event
    truth: dict               # audit-only ground truth


def _cluster_direction(k):
    direction = np.zeros(PATCH_DIM, dtype=np.float64)
    direction[k] = 1.0
    direction[k + CLUSTERS] = 0.5
    return direction


def _centers(rng):
    """Cluster centres are separated by a large offset on the cluster axes.

    The offset must dominate the within-cluster spread of the latent axes,
    otherwise an unsupervised atlas splits the generative clusters by latent
    magnitude instead of by cluster identity, and the known-answer check on
    the atlas becomes ill-posed.
    """
    centers = np.zeros((CLUSTERS, PATCH_DIM), dtype=np.float64)
    for k in range(CLUSTERS):
        centers[k] = rng.normal(scale=0.3, size=PATCH_DIM)
        centers[k, k] += CLUSTER_OFFSET
    return centers


def _sample_bag(rng, n_patches, centers, latent_value, w, cluster_probs, noise):
    """Sample one patient's patch bag.

    ``latent_value`` is the value the path latent takes for this patient and
    is already expressed in the units the scenario asks for (raw ``z``,
    ``z**2 - 1``, or ``x_sign``).
    """
    features = np.zeros((n_patches, PATCH_DIM), dtype=np.float64)
    labels = np.zeros(n_patches, dtype=np.int64)
    which = rng.choice(CLUSTERS, size=n_patches, p=cluster_probs)
    for k in range(CLUSTERS):
        rows = np.nonzero(which == k)[0]
        if len(rows) == 0:
            continue
        values = centers[k][None, :].copy()
        values[:, LATENT_AXIS] += CLUSTER_Z_STRENGTH[k] * latent_value
        if k == 2:
            values[:, CLUSTER2_AXIS] += 0.9 * w
        features[rows] = values + noise * rng.normal(size=(len(rows), PATCH_DIM))
        labels[rows] = k
    return features, labels

def _survival_from_eta(rng, eta, censor_rate=0.45, max_time=BIN_EDGES[-1]):
    """Piecewise-constant hazard sampling with independent uniform censoring."""
    base = np.asarray(BASE_HAZARD, dtype=np.float64)
    rate = 1.0 + 1.2 * eta
    hazards = np.clip(base * rate, 1e-4, 0.95)
    survival = np.cumprod(1.0 - hazards)
    draw = rng.uniform()
    bin_index = min(int(np.searchsorted(survival, draw, side='right')), CLUSTERS - 1)
    lower = BIN_EDGES[bin_index]
    upper = BIN_EDGES[bin_index + 1]
    event_time = lower + (upper - lower) * rng.uniform()
    censor_time = rng.uniform(0.0, max_time) if rng.uniform() < censor_rate else np.inf
    if event_time <= censor_time:
        return float(event_time), 1.0
    return float(min(censor_time, max_time)), 0.0


def generate_cohort(scenario, n_patients, seed, *, prefix, patch_range=(6, 30),
                    noise=0.35, pad_to=None, missing_cluster_rate=0.0,
                    censor_rate=0.45, all_censored=False, rare_event_rate=None):
    """Generate one synthetic cohort with disjoint patient IDs."""
    rng = np.random.default_rng(seed)
    if scenario not in SCENARIO_MECHANISMS:
        raise ValueError(f'Unknown scenario {scenario}')
    mechanism = SCENARIO_MECHANISMS[scenario]
    centers = _centers(rng)
    bags, rna_rows, times, events = [], [], [], []
    truth_z, truth_w, truth_x, truth_y, truth_labels, truth_composition = [], [], [], [], [], []
    for _ in range(n_patients):
        patient_id = f'{prefix}_{len(bags):05d}'
        z = rng.normal()
        w = rng.normal()
        x_sign = 1.0 if rng.uniform() < 0.5 else -1.0
        y_sign = 1.0 if rng.uniform() < 0.5 else -1.0
        cluster_probs = rng.dirichlet(np.full(CLUSTERS, 1.5))
        if missing_cluster_rate > 0 and rng.uniform() < missing_cluster_rate:
            cluster_probs[int(rng.integers(CLUSTERS))] = 0.0
            cluster_probs = cluster_probs / cluster_probs.sum()
        n_patches = int(rng.integers(patch_range[0], patch_range[1] + 1))

        # --- what the pathology encodes ---
        path_mode = mechanism['path']
        if path_mode == 'z':
            path_latent = z
        elif path_mode == 'z2':
            path_latent = z ** 2 - 1.0
        elif path_mode == 'x':
            path_latent = x_sign
        elif path_mode == 'none':
            path_latent = 0.0
        else:
            raise ValueError(f'Unknown path mode {path_mode}')

        features, labels = _sample_bag(rng, n_patches, centers, path_latent, w,
                                        cluster_probs, noise)

        # --- what the RNA encodes ---
        rna_mode = mechanism['rna']
        g = rng.normal(size=RNA_DIM)
        rna = np.zeros(RNA_DIM, dtype=np.float64)
        if rna_mode == 'z':
            rna[0] = z + 0.1 * rng.normal()
        elif rna_mode == 'y':
            rna[0] = y_sign + 0.05 * rng.normal()
        elif rna_mode == 'noise':
            rna[0] = rng.normal()
        else:
            raise ValueError(f'Unknown RNA mode {rna_mode}')

        # --- what drives the risk ---
        risk_mode = mechanism['risk']
        if risk_mode == 'z':
            eta = 1.3 * z
        elif risk_mode == 'rna':
            eta = 1.3 * rna[0]
        elif risk_mode == 'xor':
            eta = 1.6 * (x_sign * y_sign)
        elif risk_mode == 'add':
            eta = 0.8 * (x_sign + y_sign)
        elif risk_mode == 'nonlin':
            eta = 1.3 * (z ** 2 - 1.0)
        elif risk_mode == 'comp':
            eta = 6.0 * (cluster_probs[3] - 0.25)
        elif risk_mode == 'none':
            eta = 0.0
        else:
            raise ValueError(f'Unknown risk mode {risk_mode}')

        if rare_event_rate is not None:
            time_value, event = _survival_from_eta(rng, eta, censor_rate=censor_rate)
            if rng.uniform() < rare_event_rate:
                event = 0.0
                time_value = min(time_value, BIN_EDGES[-1])
        else:
            time_value, event = _survival_from_eta(rng, eta, censor_rate=censor_rate)
        if all_censored:
            event = 0.0
        bags.append(torch.from_numpy(features))
        rna_rows.append(rna)
        times.append(time_value)
        events.append(event)
        truth_z.append(z)
        truth_w.append(w)
        truth_x.append(x_sign)
        truth_y.append(y_sign)
        truth_labels.append(labels)
        truth_composition.append(cluster_probs)
    if pad_to is not None:
        padded = []
        for bag in bags:
            missing = pad_to - len(bag)
            if missing <= 0:
                padded.append(bag)
                continue
            junk = torch.zeros(missing, PATCH_DIM, dtype=torch.float64)
            padded.append((torch.cat((bag, junk), 0),
                           torch.cat((torch.ones(len(bag), dtype=torch.bool),
                                      torch.zeros(missing, dtype=torch.bool)))))
        bags = padded
    truth = {
        'cluster_labels': truth_labels,
        'cluster_composition': np.asarray(truth_composition),
        'z': np.asarray(truth_z),
        'w': np.asarray(truth_w),
        'x_sign': np.asarray(truth_x),
        'y_sign': np.asarray(truth_y),
        'cluster_z_strength': CLUSTER_Z_STRENGTH,
        'patient_ids': [f'{prefix}_{i:05d}' for i in range(n_patients)],
    }
    return SyntheticCohort(
        ids=[f'{prefix}_{i:05d}' for i in range(n_patients)],
        bags=bags,
        rna=torch.from_numpy(np.asarray(rna_rows)),
        times=torch.tensor(times, dtype=torch.float64),
        events=torch.tensor(events, dtype=torch.float64),
        truth=truth)


def clr_of_composition(composition):
    """CLR of a composition matrix (audit helper)."""
    return multiplicative_replacement_clr(torch.as_tensor(composition, dtype=torch.float64))
