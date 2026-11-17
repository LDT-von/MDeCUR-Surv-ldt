"""Method 3.2 (CLARA): cluster-conditional contrastive routing on a frozen atlas.

Independent method-three candidate.  It does not reuse the MorphID kernel-ridge
probe, does not use prediction residuals, and is not built on method 3.1.

Pipeline
--------
1. ``MorphologyAtlas``: training-patient-only patch standardisation, PCA and
   deterministic k-means prototypes (a declared engineering substitute for the
   Nature paper's Barlow-Twins + Leiden graph; Leiden is unavailable locally).
   It produces per-patch cluster weights ``q``, patient composition ``pi``, a
   multiplicative-replacement CLR transform, a cluster-mean representation ``p``
   and a quality gate ``V``.
2. ``ClusterConditionalEncoder``: label-free cluster-conditional cross-modal
   contrast (a *contrastive proxy loss*, not a mutual-information estimate) and
   a cross-fitted cluster retrieval rate ``a_k``.
3. ``ClusterRoutedSurvivalHead``: the only outcome-supervised module.  ``a_k``
   routes which cluster evidence reaches the discrete hazard logits.

Every outcome-aware quantity is fitted on training patients only.  The atlas and
the encoder are frozen while the survival head is trained.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cancer_parity import discrete_nll, discrete_survival


METHOD3_2_VERSION = 'method3_2_cluster_routed_contrast_v1'


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _finite_matrix(value, name):
    tensor = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
    if tensor.ndim not in (1, 2) or not torch.isfinite(tensor).all():
        raise ValueError(f'{name} must be a finite tensor')
    return tensor


def _distinct_ids(ids, count):
    if isinstance(ids, (str, bytes)):
        raise ValueError('Patient IDs must be a sequence')
    values = tuple(str(v) for v in ids)
    if len(values) != count or len(set(values)) != count:
        raise ValueError('Patient IDs must be unique and match the row count')
    if any(not v.strip() for v in values):
        raise ValueError('Patient IDs cannot be empty')
    return values


def _bag_rows(bag):
    """Return the real (non padding) patch rows of a bag."""
    if isinstance(bag, torch.Tensor):
        return _finite_matrix(bag, 'Patch features')
    if isinstance(bag, (tuple, list)) and len(bag) == 2:
        rows, mask = bag
        rows = _finite_matrix(rows, 'Patch features')
        mask = torch.as_tensor(mask)
        if mask.dtype != torch.bool or mask.shape != rows.shape[:1]:
            raise ValueError('A masked bag needs [patches, features] and a bool mask')
        return rows[mask]
    raise TypeError('Bags are tensors or (tensor, mask) pairs')


def multiplicative_replacement_clr(composition, delta=1e-6):
    """Multiplicative zero replacement followed by CLR.

    Zero entries mean "this cluster is absent", not "small".  The replacement
    keeps every non-zero ratio intact and the CLR coordinates sum to exactly
    zero (K-1 degrees of freedom).
    """
    pi = _finite_matrix(composition, 'Composition')
    if (pi < 0).any() or (pi.sum(1) <= 0).any():
        raise ValueError('Composition must be nonnegative with positive row sums')
    k = pi.shape[1]
    if not math.isfinite(delta) or not 0 < delta < 1.0 / k:
        raise ValueError('Zero replacement must lie strictly between 0 and 1/K')
    pi = pi / pi.sum(1, keepdim=True)
    zeros = pi == 0
    scale = 1.0 - zeros.sum(1, keepdim=True) * delta
    replaced = torch.where(zeros, torch.full_like(pi, delta), pi * scale)
    logs = replaced.log()
    return logs - logs.mean(1, keepdim=True)


def _kmeans(points, clusters, seed, iterations=25):
    """Deterministic k-means++ with farthest-point empty-cluster repair."""
    generator = torch.Generator().manual_seed(int(seed))
    n = len(points)
    centers = torch.empty(clusters, points.shape[1], dtype=points.dtype)
    first = int(torch.randint(n, (1,), generator=generator))
    centers[0] = points[first]
    closest = ((points - centers[0]) ** 2).sum(1)
    for k in range(1, clusters):
        total = float(closest.sum())
        if total <= 0:
            pick = n - 1
        else:
            probabilities = (closest / closest.sum()).clamp_min(0)
            pick = int(torch.multinomial(probabilities, 1, generator=generator))
        centers[k] = points[pick]
        closest = torch.minimum(closest, ((points - centers[k]) ** 2).sum(1))
    repairs = 0
    assignment = torch.cdist(points, centers).argmin(1)
    for _ in range(iterations):
        for k in range(clusters):
            members = points[assignment == k]
            if len(members) == 0:
                repairs += 1
                distances = torch.cdist(points, centers)
                far = int(distances.min(1).values.argmax())
                centers[k] = points[far]
                assignment = torch.cdist(points, centers).argmin(1)
                continue
            centers[k] = members.mean(0)
        assignment = torch.cdist(points, centers).argmin(1)
    return centers, repairs


# --------------------------------------------------------------------------
# morphology atlas
# --------------------------------------------------------------------------

@dataclass
class PatientMorphology:
    """Frozen morphology description of one patient."""
    patient_id: str
    z: torch.Tensor              # [n, L] PCA coordinates
    q: torch.Tensor              # [n, K] cluster weights
    p: torch.Tensor              # [K, L] cluster-mean representation
    mass: torch.Tensor          # [K]
    composition: torch.Tensor   # [K]
    clr: torch.Tensor            # [K]
    valid: torch.Tensor         # [K] bool


class MorphologyAtlas(nn.Module):
    """Training-only morphology atlas: standardise -> PCA -> k-means prototypes.

    Not the Nature paper's Leiden community detection and not its 47 HPC.  The
    cluster count is an engineering choice for mechanism work, not a clinical
    selection, and no survival label is accepted anywhere in this class.
    """

    def __init__(self, mean, scale, components, centers, *, temperature=1.0,
                 assignment='hard', min_mass=1.0, metadata=None):
        super().__init__()
        mean = _finite_matrix(mean, 'Patch mean').reshape(-1)
        scale = _finite_matrix(scale, 'Patch scale').reshape(-1)
        components = _finite_matrix(components, 'PCA components')
        centers = _finite_matrix(centers, 'Cluster centers')
        if (scale <= 0).any():
            raise ValueError('Patch scale must be positive')
        if components.shape[1] != centers.shape[1]:
            raise ValueError('Component and prototype dimensions differ')
        if mean.shape != (components.shape[0],) or scale.shape != mean.shape:
            raise ValueError('Patch preprocessing shapes disagree')
        if assignment not in ('hard', 'soft'):
            raise ValueError('Assignment must be hard or soft')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive')
        if not math.isfinite(min_mass) or min_mass < 0:
            raise ValueError('Minimum mass must be finite and nonnegative')
        self.register_buffer('patch_mean', mean)
        self.register_buffer('patch_scale', scale)
        self.register_buffer('components', components)
        self.register_buffer('centers', centers)
        self.temperature = float(temperature)
        self.assignment = assignment
        self.min_mass = float(min_mass)
        self.metadata = copy.deepcopy(metadata or {'source': 'caller_supplied', 'fit_patient_ids': []})

    @property
    def clusters(self):
        return int(self.centers.shape[0])

    @property
    def latent_dim(self):
        return int(self.centers.shape[1])

    @classmethod
    @torch.no_grad()
    def fit(cls, bags, ids, *, clusters=4, latent_dim=8, per_patient_cap=64,
            seed=3, assignment='hard', temperature=1.0, min_mass=1.0):
        bags = list(bags)
        ids = _distinct_ids(ids, len(bags))
        if clusters < 1 or latent_dim < 1 or per_patient_cap < 1:
            raise ValueError('Atlas budget must be positive')
        generator = torch.Generator().manual_seed(int(seed))
        sampled = []
        for bag in bags:
            rows = _bag_rows(bag)
            if len(rows) == 0:
                continue
            if len(rows) > per_patient_cap:
                index = torch.randperm(len(rows), generator=generator)[:per_patient_cap]
                rows = rows[index]
            sampled.append(rows)
        if not sampled:
            raise ValueError('No usable patch features for the atlas')
        pool = torch.cat(sampled, 0)
        mean = pool.mean(0)
        scale = pool.std(0).clamp_min(1e-8)
        standardized = (pool - mean) / scale
        latent_dim = int(min(latent_dim, standardized.shape[1]))
        _, _, vh = torch.linalg.svd(standardized, full_matrices=False)
        components = vh[:latent_dim].T
        projected = standardized @ components
        centers, repairs = _kmeans(projected, clusters, seed)
        return cls(mean, scale, components, centers, temperature=temperature,
                   assignment=assignment, min_mass=min_mass,
                   metadata={'source': 'method3_2_pca_kmeans', 'fit_patient_ids': list(ids),
                             'seed': int(seed), 'empty_cluster_repairs': int(repairs),
                             'assignment': assignment})

    @torch.no_grad()
    def encode(self, bags, ids=None):
        """Assign clusters and summarise composition for arbitrary patients."""
        bags = list(bags)
        if ids is None:
            ids = [f'patient_{i}' for i in range(len(bags))]
        ids = _distinct_ids(ids, len(bags))
        out = []
        for identity, bag in zip(ids, bags):
            rows = _bag_rows(bag)
            if len(rows) == 0:
                raise ValueError(f'Patient {identity} has no real patch')
            z = ((rows - self.patch_mean) / self.patch_scale) @ self.components
            distances = torch.cdist(z, self.centers)
            if self.assignment == 'hard':
                q = torch.zeros(len(z), self.clusters, dtype=z.dtype)
                q[torch.arange(len(z)), distances.argmin(1)] = 1.0
            else:
                q = torch.softmax(-(distances ** 2) / self.temperature, dim=1)
            mass = q.sum(0)
            composition = mass / mass.sum()
            p = torch.zeros(self.clusters, self.latent_dim, dtype=z.dtype)
            for k in range(self.clusters):
                weight = mass[k]
                if float(weight) > 0:
                    p[k] = (q[:, k:k + 1] * z).sum(0) / weight
            clr = multiplicative_replacement_clr(composition.unsqueeze(0))[0]
            valid = mass >= self.min_mass
            out.append(PatientMorphology(identity, z, q, p, mass, composition, clr, valid))
        return out

    def fingerprint(self):
        digest = hashlib.sha256()
        for name in ('patch_mean', 'patch_scale', 'components', 'centers'):
            digest.update(self.state_dict()[name].numpy().tobytes())
        digest.update(json.dumps({'temperature': self.temperature, 'assignment': self.assignment,
                                  'min_mass': self.min_mass}, sort_keys=True).encode())
        return digest.hexdigest()


# --------------------------------------------------------------------------
# cluster-conditional contrastive encoder
# --------------------------------------------------------------------------

class ClusterConditionalEncoder(nn.Module):
    """Label-free encoder used by the cluster-conditional contrastive proxy loss.

    ``cluster_conditioned=False`` removes the cluster conditioning (one shared
    pathology projection, one shared RNA projection and one shared score for
    every cluster); it is the explicit control for "cluster condition
    cancelled".

    When the cluster condition is active every cluster owns its own pathology
    projection *and* its own RNA projection.  A cluster that carries no
    cross-modal signal then simply fails to learn, instead of corrupting a
    shared RNA encoder that other clusters still need.
    """

    def __init__(self, latent_dim, rna_dim, clusters, hidden=8, *, cluster_conditioned=True,
                 seed=3):
        super().__init__()
        if hidden < 1 or clusters < 1 or latent_dim < 1 or rna_dim < 1:
            raise ValueError('Dimensions must be positive')
        self.k = clusters
        self.hidden = hidden
        self.cluster_conditioned = bool(cluster_conditioned)
        # How often each cluster could actually be trained.  A cluster whose
        # eligible-patient count never reaches ``min_patients`` keeps its
        # random initialisation; that projection must not leak into any
        # RNA-dependent term.
        self.register_buffer('cluster_usage', torch.zeros(clusters, dtype=torch.float64))
        # 1.0 once the label-free stage has run, 0.0 when it was skipped.
        # It is a buffer so that a reloaded checkpoint keeps the same mask
        # behaviour as the model it was saved from.
        self.register_buffer('ssl_enabled', torch.zeros(1, dtype=torch.float64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.path_weight = nn.Parameter(torch.randn(clusters, latent_dim, hidden, dtype=torch.float64) * 0.1)
            self.path_bias = nn.Parameter(torch.zeros(clusters, hidden, dtype=torch.float64))
            if self.cluster_conditioned:
                self.rna_weight1 = nn.Parameter(torch.randn(clusters, rna_dim, hidden, dtype=torch.float64) * 0.1)
                self.rna_bias1 = nn.Parameter(torch.zeros(clusters, hidden, dtype=torch.float64))
                self.rna_weight2 = nn.Parameter(torch.randn(clusters, hidden, hidden, dtype=torch.float64) * 0.1)
                self.rna_bias2 = nn.Parameter(torch.zeros(clusters, hidden, dtype=torch.float64))
            else:
                self.rna_weight1 = nn.Parameter(torch.randn(rna_dim, hidden, dtype=torch.float64) * 0.1)
                self.rna_bias1 = nn.Parameter(torch.zeros(hidden, dtype=torch.float64))
                self.rna_weight2 = nn.Parameter(torch.randn(hidden, hidden, dtype=torch.float64) * 0.1)
                self.rna_bias2 = nn.Parameter(torch.zeros(hidden, dtype=torch.float64))

    def cluster_trained_mask(self):
        """Which clusters received gradient from the contrastive stage.

        When the label-free stage ran, a cluster that never reached
        ``min_patients`` eligible patients in any batch kept its random
        initialisation, so it is excluded.  When the label-free stage did not
        run at all, every cluster is consistently untrained and the mask is
        open; that is the deliberate "no_ssl" control, where the noise is
        uniform across clusters instead of mixed with trained ones.
        """
        if not bool(self.ssl_enabled.item()):
            return torch.ones(self.k, dtype=torch.bool)
        return self.cluster_usage > 0


    def encode_path(self, p):
        """p: [N, K, L] -> [N, K, h]"""
        u = torch.tanh(torch.einsum('nkl,klh->nkh', p, self.path_weight) + self.path_bias)
        if not self.cluster_conditioned:
            u = u.mean(1, keepdim=True).expand_as(u)
        return u

    def encode_rna(self, rna, cluster=None):
        """rna: [N, d_g] -> [N, h] (or [N, K, h] when clusters are requested)"""
        if self.cluster_conditioned:
            if cluster is not None:
                index = int(cluster)
                h = torch.tanh(rna @ self.rna_weight1[index] + self.rna_bias1[index])
                return torch.tanh(h @ self.rna_weight2[index] + self.rna_bias2[index])
            h = torch.tanh(torch.einsum('nd,kdh->nkh', rna, self.rna_weight1)
                           + self.rna_bias1.unsqueeze(0))
            return torch.tanh(torch.einsum('nkh,khg->nkg', h, self.rna_weight2)
                               + self.rna_bias2.unsqueeze(0))
        h = torch.tanh(rna @ self.rna_weight1 + self.rna_bias1)
        return torch.tanh(h @ self.rna_weight2 + self.rna_bias2)

    def score(self, u, r):
        """Scorer s(u, r) used by both the contrastive loss and the association audit.

        The scorer is the cosine similarity between the cluster-specific
        pathology projection and the RNA embedding, i.e. exactly the quantity
        the contrastive proxy optimises.  Using a different score at audit time
        would measure something the objective never trained.  The cluster
        specificity enters through the projections; when the cluster condition
        is cancelled the projections are shared and so is the score.
        """
        return F.cosine_similarity(u.unsqueeze(0), r.unsqueeze(0)).squeeze(0)


def cluster_contrastive_proxy_loss(encoder, u_left, rna_raw, valid, *, temperature=0.2,
                                    min_patients=8):
    """Cluster-conditional contrastive proxy loss.

    For every cluster ``k`` the patients with ``valid[:, k]`` form the candidate
    set.  The positive pair is the same patient (its pathology cluster mean
    against its own RNA embedding); negatives are *different* patients of the
    same cluster.  A patient is never its own negative and patients are never
    duplicated.

    The returned value is an optimisation target, **not** a mutual information
    estimate.  ``log M - E[L]`` is only an information lower bound under the
    usual independent-sampling assumptions, which the finite, cluster-filtered
    batches used here do not guarantee.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be positive and finite')
    if u_left.ndim != 3 or rna_raw.ndim != 2:
        raise ValueError('Expected u [N,K,h] and rna [N,d_g]')
    if u_left.shape[0] != rna_raw.shape[0] or u_left.shape[2] != encoder.hidden:
        raise ValueError('Batch sizes or hidden sizes disagree')
    total = torch.zeros((), dtype=u_left.dtype)
    weight = 0.0
    per_cluster = {}
    for k in range(u_left.shape[1]):
        members = torch.nonzero(valid[:, k], as_tuple=False).flatten()
        m = len(members)
        if m < min_patients:
            per_cluster[k] = {'patients': int(m), 'loss': None, 'used': False}
            continue
        u = u_left[members, k]
        r = encoder.encode_rna(rna_raw[members], cluster=k)
        logits = F.normalize(u, dim=-1) @ F.normalize(r, dim=-1).T / temperature
        target = torch.arange(m)
        loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
        total = total + loss * m
        weight += m
        per_cluster[k] = {'patients': int(m), 'loss': float(loss.detach()), 'used': True}
    if weight == 0:
        return torch.zeros((), dtype=u_left.dtype), per_cluster
    return total / weight, per_cluster


def _batch_view(batch, fraction, generator, min_mass):
    """Feature-level pathology view: resample each patient's patches."""
    p2, comps, clrs, masses, valids = [], [], [], [], []
    for patient in batch:
        n = len(patient.z)
        keep = max(1, int(round(fraction * n)))
        index = torch.randint(n, (keep,), generator=generator)
        z = patient.z[index]
        q = patient.q[index]
        mass = q.sum(0)
        composition = mass / mass.sum()
        p = torch.zeros_like(patient.p)
        for k in range(q.shape[1]):
            if float(mass[k]) > 0:
                p[k] = (q[:, k:k + 1] * z).sum(0) / mass[k]
        clr = multiplicative_replacement_clr(composition.unsqueeze(0))[0]
        p2.append(p)
        comps.append(composition)
        clrs.append(clr)
        masses.append(mass)
        valids.append(mass >= min_mass)
    return (torch.stack(p2), torch.stack(comps), torch.stack(clrs),
            torch.stack(masses), torch.stack(valids))


def train_cluster_contrastive(encoder, patients, rna, *, epochs=60, batch_size=32,
                               temperature=0.2, min_patients=8, lr=1e-2, seed=3,
                               shuffle_pairs=False, patch_fraction=0.8,
                               rna_noise=0.05, min_mass=1.0):
    """Label-free training of the cluster-conditional contrastive proxy.

    ``shuffle_pairs`` permutes which RNA vector is paired with which patient
    inside every batch; it is the negative control for the retrieval rate.
    """
    n = len(patients)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(int(seed))
    history = []
    usage = torch.zeros(encoder.k, dtype=torch.float64)
    with torch.no_grad():
        encoder.ssl_enabled.fill_(1.0)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator)
        epoch_loss, epoch_batches = 0.0, 0
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            if len(index) < 2:
                continue
            batch = [patients[int(i)] for i in index]
            p1 = torch.stack([b.p for b in batch])
            p2, _, _, _, valid2 = _batch_view(batch, patch_fraction, generator, min_mass)
            valid1 = torch.stack([b.valid for b in batch])
            if shuffle_pairs:
                perm = torch.randperm(len(index), generator=generator)
                rna_batch = rna[index][perm]
            else:
                rna_batch = rna[index]
            rna_batch = rna_batch + rna_noise * torch.randn(
                len(index), rna.shape[1], generator=generator, dtype=rna.dtype)
            u1 = encoder.encode_path(p1)
            u2 = encoder.encode_path(p2)
            joint_valid = valid1 & valid2
            loss1, stats1 = cluster_contrastive_proxy_loss(
                encoder, u1, rna_batch, joint_valid, temperature=temperature,
                min_patients=min_patients)
            loss2, stats2 = cluster_contrastive_proxy_loss(
                encoder, u2, rna_batch, joint_valid, temperature=temperature,
                min_patients=min_patients)
            for stats in (stats1, stats2):
                for key, entry in stats.items():
                    if entry['used']:
                        usage[key] += 1.0
            loss = 0.5 * (loss1 + loss2)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite contrastive proxy loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_batches += 1
        history.append({'epoch': epoch, 'loss': epoch_loss / max(1, epoch_batches)})
    with torch.no_grad():
        encoder.cluster_usage.copy_(usage)
    return history


@torch.no_grad()
def cluster_association_auc(encoder, patients, rna, *, candidate_patients, candidate_rna,
                             min_patients=8, n_shuffles=32, seed=3):
    """Cluster-conditional cross-modal association strength ``a_k``.

    For every evaluated patient and cluster, the score of the *true* pair (its
    own RNA) is compared with the scores of the same patient's pathology
    representation against ``n_shuffles`` different patients' RNA.  The
    reported quantity is the fraction of shuffled pairs that score *below* the
    true pair, i.e. the ROC AUC of "true pair versus shuffled pair".

    This replaces a top-1 retrieval rate, which has almost no statistical power
    with more than a handful of candidates.  The AUC is a *ranking* statistic:
    it says the learned score prefers the true pairing, and it says nothing
    about an absolute amount of shared information.  A cluster with too few
    eligible patients is unusable and reported as 0 with ``usable=False``.
    """
    encoder.eval()
    generator = torch.Generator().manual_seed(int(seed))
    u_eval = encoder.encode_path(torch.stack([x.p for x in patients]))
    wins = np.zeros(encoder.k, dtype=np.float64)
    attempts = np.zeros(encoder.k, dtype=np.float64)
    for k in range(encoder.k):
        eligible = [i for i, patient in enumerate(patients) if bool(patient.valid[k])]
        if len(eligible) < min_patients:
            continue
        r_eval = encoder.encode_rna(rna, cluster=k)
        r_pool = encoder.encode_rna(candidate_rna, cluster=k)
        for i in eligible:
            others = [j for j, patient in enumerate(candidate_patients)
                      if bool(patient.valid[k])]
            if len(others) < min_patients:
                continue
            own = encoder.score(u_eval[i, k], r_eval[i])
            permuted = [others[int(j)] for j in
                        torch.randperm(len(others), generator=generator)[:n_shuffles]]
            scores = torch.stack([encoder.score(u_eval[i, k], r_pool[j]) for j in permuted])
            attempts[k] += 1
            wins[k] += float((scores < own).to(torch.float64).mean())
    a = torch.zeros(encoder.k, dtype=torch.float64)
    for k in range(encoder.k):
        if attempts[k] > 0:
            a[k] = wins[k] / attempts[k]
    stats = {'attempts': attempts.tolist(), 'wins': wins.tolist(),
             'a': a.tolist(), 'usable': [bool(x > 0) for x in attempts.tolist()],
             'n_shuffles': int(n_shuffles)}
    return a, stats


def cross_fitted_association(build_encoder, patients, rna, *, folds=4, seed=3, epochs=60,
                             batch_size=32, temperature=0.2, min_patients=8, lr=1e-2,
                             shuffle_pairs=False, patch_fraction=0.8, rna_noise=0.05,
                             min_mass=1.0, n_shuffles=32):
    """Patient-axis cross-fitted cluster association strength.

    The encoder is refitted on ``folds-1`` folds of patients and evaluated on
    the held-out fold, so no patient contributes to the encoder that scores it.
    The held-out patient is compared against the RNA of the *fit* patients,
    which keeps the negatives strictly different patients.
    """
    n = len(patients)
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(n, generator=generator)
    fold_of = torch.empty(n, dtype=torch.long)
    for position, patient_index in enumerate(order.tolist()):
        fold_of[patient_index] = position % folds
    k = build_encoder().k
    attempts_total = np.zeros(k, dtype=np.float64)
    wins_total = np.zeros(k, dtype=np.float64)
    for fold in range(folds):
        held = torch.nonzero(fold_of == fold, as_tuple=False).flatten().tolist()
        fit = torch.nonzero(fold_of != fold, as_tuple=False).flatten().tolist()
        if len(held) < 2 or len(fit) < 2:
            continue
        encoder = build_encoder()
        train_cluster_contrastive(
            encoder, [patients[i] for i in fit], rna[fit], epochs=epochs,
            batch_size=batch_size, temperature=temperature, min_patients=min_patients,
            lr=lr, seed=seed + 17 * fold, shuffle_pairs=shuffle_pairs,
            patch_fraction=patch_fraction, rna_noise=rna_noise, min_mass=min_mass)
        _, stats = cluster_association_auc(
            encoder, [patients[i] for i in held], rna[held],
            candidate_patients=[patients[i] for i in fit], candidate_rna=rna[fit],
            min_patients=min_patients, n_shuffles=n_shuffles, seed=seed + fold)
        for c in range(k):
            attempts_total[c] += stats['attempts'][c]
            wins_total[c] += stats['wins'][c]
    a = torch.zeros(k, dtype=torch.float64)
    for c in range(k):
        if attempts_total[c] > 0:
            a[c] = wins_total[c] / attempts_total[c]
    stats = {'attempts': attempts_total.tolist(), 'wins': wins_total.tolist(),
             'a': a.tolist(), 'folds': folds, 'cross_fitted': True}
    return a, stats


# --------------------------------------------------------------------------
# survival head
# --------------------------------------------------------------------------

@dataclass
class Method32Config:
    clusters: int = 4
    bins: int = 4
    hidden: int = 8
    route: str = 'info'                 # 'info' | 'uniform'
    use_composition: bool = True
    use_interaction: bool = True
    use_generic_interaction: bool = False
    block_pathology: bool = False
    block_rna: bool = False
    alpha: float = 0.5
    temperature: float = 0.2
    min_patients: int = 8
    folds: int = 4
    epochs_ssl: int = 60
    epochs_head: int = 150
    batch_size: int = 32
    lr_ssl: float = 1e-2
    lr_head: float = 5e-3
    seed: int = 3
    cluster_conditioned: bool = True
    shuffle_pairs: bool = False
    patch_fraction: float = 0.8
    rna_noise: float = 0.05
    min_mass: float = 1.0
    # --- explicit common / unique (shared / modality-specific) decomposition ---
    # ``decouple`` splits the shared interaction space into a *common* part and
    # a *modality-specific* part instead of letting one dense interaction
    # vector absorb both.  ``decouple_mode`` selects how the split is made.
    decouple: bool = False
    decouple_mode: str = 'symmetric'     # 'symmetric' | 'asymmetric'
    lambda_common: float = 0.0            # weight of the common-alignment term
    lambda_unique: float = 0.0            # weight of the unique-decorrelation term
    common_dim: int = 0                   # 0 -> hidden // 2
    extra: dict = field(default_factory=dict)


class ClusterRoutedSurvivalHead(nn.Module):
    """Additive discrete-hazard head whose cluster weights come from ``a_k``.

    The head is linear in its inputs, but every cluster contributes through the
    *same* pathology and interaction directions.  The relative weighting
    implied by ``a_k`` therefore changes the mixed evidence vector and cannot be
    undone by rescaling one parameter, and a cluster with ``a_k = 0``
    contributes exactly nothing to the logits.

    ``route='uniform'`` sets every cluster weight to one, which is the explicit
    control for "information routing".

    Two corrections and one extension live here.

    *Correction 1 -- masked RNA summary.*  The patient-level RNA term used to
    average the RNA embedding over *all* clusters, including clusters whose
    contrastive projection never received gradient.  Those clusters keep their
    random initialisation, so the average mixed trained and untrained
    projections.  ``rna_valid`` now masks the RNA summary (and the RNA side of
    the interaction) to clusters that are both present in the patient and
    actually trained.

    *Extension -- explicit common / unique decomposition.*  ``decouple=True``
    splits every modality representation into a **common** part (the projection
    onto a learned orthonormal subspace) and a **modality-specific** part (the
    exact orthogonal residual, so ``||x||^2 = ||x_common||^2 + ||x_unique||^2``).
    The common part carries the cross-modal interaction, the unique parts carry
    the modality-specific signals, and the two parts of each modality are
    orthogonal by construction.  ``decouple_mode='symmetric'`` forces both
    modalities to use the *same* subspace, which is what makes the common
    part comparable across modalities; ``'asymmetric'`` lets each modality pick
    its own subspace and is the control for "the shared subspace is what does
    the work".

    The decomposition is *outcome-supervised*: it is fitted in the survival
    stage on frozen label-free features, not in the encoder.  It is therefore
    NOT the DeCUR / PIBD encoder-level disentanglement.
    """

    def __init__(self, config: Method32Config):
        super().__init__()
        self.config = config
        b, h = config.bins, config.hidden
        k = config.clusters
        self.path_weight = nn.Parameter(torch.zeros(b, h, dtype=torch.float64))
        self.generic_weight = nn.Parameter(torch.zeros(b, h, dtype=torch.float64))
        self.composition_weight = nn.Parameter(torch.zeros(b, k, dtype=torch.float64))
        self.rna_weight = nn.Parameter(torch.zeros(b, h, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(b, dtype=torch.float64))
        if config.decouple:
            c = int(config.common_dim) or max(1, h // 2)
            if not 1 <= c < h:
                raise ValueError('common_dim must lie strictly between 0 and hidden')
            self.common_dim = c
            # The unique parts live in an (h - c)-dimensional basis, so the
            # weights that read them must have that width.
            self.path_weight = nn.Parameter(torch.zeros(b, h - c, dtype=torch.float64))
            self.rna_weight = nn.Parameter(torch.zeros(b, h - c, dtype=torch.float64))
            # Learned raw subspaces; QR turns them into orthonormal bases whose
            # first ``c`` columns span the common subspace and whose remaining
            # columns span its orthogonal complement.
            self.common_proj_path = nn.Parameter(torch.zeros(k, h, h, dtype=torch.float64))
            if config.decouple_mode == 'symmetric':
                self.register_parameter('common_proj_rna', None)
            elif config.decouple_mode == 'asymmetric':
                self.common_proj_rna = nn.Parameter(torch.zeros(k, h, h, dtype=torch.float64))
            else:
                raise ValueError('decouple_mode must be "symmetric" or "asymmetric"')
            self.common_inter_weight = nn.Parameter(torch.zeros(b, c, dtype=torch.float64))
            self.unique_inter_weight = nn.Parameter(torch.zeros(b, h - c, dtype=torch.float64))
            # The dense interaction is replaced by the common/unique pair.
            self.register_parameter('inter_weight', None)
        else:
            self.inter_weight = nn.Parameter(torch.zeros(b, h, dtype=torch.float64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            for parameter in self.parameters():
                parameter.data = parameter.data + 0.01 * torch.randn_like(parameter)
            if config.decouple:
                # A random raw matrix makes QR start from a generic basis; a
                # zero matrix would make the QR gradient degenerate.
                self.common_proj_path.data = self.common_proj_path.data + torch.randn(
                    k, h, h, dtype=torch.float64)
                if self.common_proj_rna is not None:
                    self.common_proj_rna.data = self.common_proj_rna.data + torch.randn(
                        k, h, h, dtype=torch.float64)

    # -- common / unique helpers ------------------------------------------------

    def _basis(self):
        """Full orthonormal bases [K, h, h] per cluster.

        The first ``c`` columns span the common subspace, the remaining
        ``h - c`` columns span its orthogonal complement (the modality-specific
        subspace).  Both parts are expressed in their own basis, so they have
        the right widths and are exactly orthogonal.
        """
        def orthonormal(matrix):
            q, r = torch.linalg.qr(matrix)
            # Fix the sign so the basis is a deterministic function of the
            # parameter; QR alone leaves column signs ambiguous.
            signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            return q * signs.unsqueeze(-2)
        path = orthonormal(self.common_proj_path)
        if self.common_proj_rna is None:
            return path, path
        return path, orthonormal(self.common_proj_rna)

    def decompose(self, u, rna):
        """Split both modalities into common and modality-specific parts.

        ``u`` [N,K,h], ``rna`` [N,K,h] -> (u_common, u_unique, r_common, r_unique),
        each [N,K,c] / [N,K,h-c].  The common part lies in the learned common
        subspace and the unique part in its orthogonal complement, so the two
        are exactly orthogonal and the energies add up to the original energy.
        """
        basis_path, basis_rna = self._basis()
        c = self.common_dim
        u_common = torch.einsum('nkh,khc->nkc', u, basis_path[:, :, :c])
        u_unique = torch.einsum('nkh,khx->nkx', u, basis_path[:, :, c:])
        r_common = torch.einsum('nkh,khc->nkc', rna, basis_rna[:, :, :c])
        r_unique = torch.einsum('nkh,khx->nkx', rna, basis_rna[:, :, c:])
        return u_common, u_unique, r_common, r_unique

    def common_energy(self, u, rna):
        """Fraction of each representation's energy that lies in the common subspace.

        This is an interpretability diagnostic, not a performance number.
        """
        u_common, _, r_common, _ = self.decompose(u, rna)
        u_energy = u.pow(2).sum(-1).clamp_min(1e-12)
        r_energy = rna.pow(2).sum(-1).clamp_min(1e-12)
        return {'path_common_fraction': (u_common.pow(2).sum(-1) / u_energy),
                'rna_common_fraction': (r_common.pow(2).sum(-1) / r_energy)}

    def aux_losses(self, u, rna, valid):
        """Common-alignment and unique-decorrelation regularisers.

        ``common`` pulls the two common parts together (they are supposed to
        describe the same thing), ``unique`` pushes the two modality-specific
        parts apart (they are supposed to describe different things).

        ``valid`` must be the *RNA* validity mask, not the plain cluster
        presence mask: a cluster whose RNA projection was never trained has a
        random embedding, and aligning it to the pathology embedding would
        pull the common subspace towards noise.
        """
        u_common, u_unique, r_common, r_unique = self.decompose(u, rna)
        mask = valid.to(u.dtype)
        denom = mask.sum().clamp_min(1.0)
        u_cn = F.normalize(u_common, dim=-1, eps=1e-8)
        r_cn = F.normalize(r_common, dim=-1, eps=1e-8)
        common = (1.0 - (u_cn * r_cn).sum(-1) * mask).sum() / denom
        u_un = F.normalize(u_unique, dim=-1, eps=1e-8)
        r_un = F.normalize(r_unique, dim=-1, eps=1e-8)
        unique = ((u_un * r_un).sum(-1) ** 2 * mask).sum() / denom
        return {'common': common, 'unique': unique}

    # -- prediction --------------------------------------------------------------

    def forward(self, u, rna, clr, valid, route, rna_valid=None):
        """u: [N,K,h], rna: [N,K,h] or [N,h], clr: [N,K], valid: [N,K], route: [K] -> [N,B]

        ``rna_valid`` [N,K] masks the RNA-dependent terms.  It defaults to
        ``valid`` when the caller has no extra information.
        """
        n = u.shape[0]
        logits = torch.zeros(n, self.config.bins, dtype=u.dtype)
        route = route.to(u.dtype).view(1, -1)
        mask = valid.to(u.dtype) * route                       # [N,K]
        if rna.ndim == 2:
            rna = rna.unsqueeze(1).expand(-1, self.config.clusters, -1)
        if rna_valid is None:
            rna_valid = valid
        rna_mask = rna_valid.to(u.dtype) * route
        if self.config.decouple:
            u_common, u_unique, r_common, r_unique = self.decompose(u, rna)
            if not self.config.block_pathology:
                # pathology-specific signal lives in the modality-specific part
                path = torch.einsum('nkh,bh->nkb', u_unique, self.path_weight)
                logits = logits + (path * mask.unsqueeze(-1)).sum(1)
                if self.config.use_interaction:
                    common_inter = torch.einsum('nkc,nkc,bc->nkb', u_common, r_common,
                                                 self.common_inter_weight)
                    logits = logits + (common_inter * rna_mask.unsqueeze(-1)).sum(1)
                    unique_inter = torch.einsum('nkx,nkx,bx->nkb', u_unique, r_unique,
                                                 self.unique_inter_weight)
                    logits = logits + (unique_inter * rna_mask.unsqueeze(-1)).sum(1)
                if self.config.use_generic_interaction:
                    pooled = (u * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1e-8)
                    rna_pooled = (rna * rna_mask.unsqueeze(-1)).sum(1) / rna_mask.sum(1, keepdim=True).clamp_min(1e-8)
                    generic = torch.einsum('nh,nh,bh->nb', pooled, rna_pooled, self.generic_weight)
                    logits = logits + generic
            if self.config.use_composition and not self.config.block_pathology:
                composition = clr * valid.to(u.dtype)
                logits = logits + composition @ self.composition_weight.T
            if not self.config.block_rna:
                rna_unique_mean = (r_unique * rna_mask.unsqueeze(-1)).sum(1) / rna_mask.sum(1, keepdim=True).clamp_min(1e-8)
                logits = logits + rna_unique_mean @ self.rna_weight.T
            return logits + self.bias
        if not self.config.block_pathology:
            path = torch.einsum('nkh,bh->nkb', u, self.path_weight)
            logits = logits + (path * mask.unsqueeze(-1)).sum(1)
            if self.config.use_interaction:
                inter = torch.einsum('nkh,nkh,bh->nkb', u, rna, self.inter_weight)
                logits = logits + (inter * rna_mask.unsqueeze(-1)).sum(1)
            if self.config.use_generic_interaction:
                pooled = (u * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1e-8)
                rna_pooled = (rna * rna_mask.unsqueeze(-1)).sum(1) / rna_mask.sum(1, keepdim=True).clamp_min(1e-8)
                generic = torch.einsum('nh,nh,bh->nb', pooled, rna_pooled, self.generic_weight)
                logits = logits + generic
        if self.config.use_composition and not self.config.block_pathology:
            composition = clr * valid.to(u.dtype)
            logits = logits + composition @ self.composition_weight.T
        if not self.config.block_rna:
            rna_mean = (rna * rna_mask.unsqueeze(-1)).sum(1) / rna_mask.sum(1, keepdim=True).clamp_min(1e-8)
            logits = logits + rna_mean @ self.rna_weight.T
        return logits + self.bias

    def cluster_contributions(self, u, rna, clr, valid, route, rna_valid=None):
        """Per-cluster, per-bin logit contributions (model-internal, not causal)."""
        route = route.to(u.dtype).view(1, -1)
        mask = valid.to(u.dtype) * route
        if rna.ndim == 2:
            rna = rna.unsqueeze(1).expand(-1, self.config.clusters, -1)
        if rna_valid is None:
            rna_valid = valid
        rna_mask = rna_valid.to(u.dtype) * route
        if self.config.decouple:
            u_common, u_unique, r_common, r_unique = self.decompose(u, rna)
            path = torch.einsum('nkh,bh->nkb', u_unique, self.path_weight) * mask.unsqueeze(-1)
            inter = torch.zeros_like(path)
            if self.config.use_interaction:
                common = torch.einsum('nkc,nkc,bc->nkb', u_common, r_common,
                                       self.common_inter_weight) * rna_mask.unsqueeze(-1)
                unique = torch.einsum('nkx,nkx,bx->nkb', u_unique, r_unique,
                                       self.unique_inter_weight) * rna_mask.unsqueeze(-1)
                inter = common + unique
            return {'path': path, 'interaction': inter,
                    'composition': (clr * valid.to(u.dtype)) @ self.composition_weight.T}
        path = torch.einsum('nkh,bh->nkb', u, self.path_weight) * mask.unsqueeze(-1)
        inter = torch.zeros_like(path)
        if self.config.use_interaction:
            inter = torch.einsum('nkh,nkh,bh->nkb', u, rna, self.inter_weight) * rna_mask.unsqueeze(-1)
        return {'path': path, 'interaction': inter,
                'composition': (clr * valid.to(u.dtype)) @ self.composition_weight.T}


def survival_bin_labels(times, events, *, bins=4):
    """Training-only discrete binning.

    Internal boundaries are the quantiles of the *training event* times.  An
    event falls in the bin that contains its time; a censored patient falls in
    the bin that contains its censoring time (the last bin when it is beyond
    the last boundary).  ``censorship = 1 - event``.
    """
    times = torch.as_tensor(times, dtype=torch.float64).reshape(-1)
    events = torch.as_tensor(events, dtype=torch.float64).reshape(-1)
    if len(times) != len(events) or len(times) == 0:
        raise ValueError('Times and events must align and be non-empty')
    if (times < 0).any() or not torch.isfinite(times).all():
        raise ValueError('Times must be finite and nonnegative')
    event_times = times[events > 0.5]
    quantiles = torch.linspace(0, 1, bins + 1, dtype=torch.float64)[1:-1]
    if len(event_times) < bins - 1:
        bounds = torch.linspace(float(times.min()), float(times.max()) + 1e-6, bins + 1,
                                dtype=torch.float64)[1:-1]
    else:
        bounds = torch.quantile(event_times, quantiles)
    bounds = torch.sort(bounds).values
    labels = torch.bucketize(times, bounds, right=True).clamp(0, bins - 1)
    return labels.long(), (1.0 - events).long(), bounds


def harrell_c(risk, times, events):
    """Harrell's C on comparable pairs (higher is better)."""
    risk = torch.as_tensor(risk, dtype=torch.float64).reshape(-1)
    times = torch.as_tensor(times, dtype=torch.float64).reshape(-1)
    events = torch.as_tensor(events, dtype=torch.float64).reshape(-1)
    concordant, comparable = 0.0, 0.0
    for i in range(len(times)):
        if events[i] < 0.5:
            continue
        for j in range(len(times)):
            if i == j:
                continue
            # j must be observed at least as long as i's event time
            if times[j] < times[i] or (times[j] == times[i] and events[j] > 0.5):
                continue
            comparable += 1
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] == risk[j]:
                concordant += 0.5
    if comparable == 0:
        return None
    return concordant / comparable


def fit_survival_head(model, features, labels, censorship, *, epochs=150, batch_size=32,
                      lr=5e-3, alpha=0.5, seed=3, val_features=None, val_labels=None,
                      val_censorship=None):
    """Train the discrete-hazard head on training patients only.

    ``features`` is a dict of tensors: ``u`` [N,K,h], ``rna`` [N,h], ``clr`` [N,K],
    ``valid`` [N,K], ``rna_valid`` [N,K], ``route`` [K].

    When the head is configured with ``decouple=True``, the optional
    ``lambda_common`` / ``lambda_unique`` terms add the common-alignment and
    unique-decorrelation regularisers.  They are *auxiliary* terms on the
    survival stage and do not touch the label-free encoder.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(int(seed))
    n = len(labels)
    history = []
    lambda_common = float(getattr(model.config, 'lambda_common', 0.0))
    lambda_unique = float(getattr(model.config, 'lambda_unique', 0.0))
    use_aux = bool(getattr(model.config, 'decouple', False)) and (lambda_common > 0 or lambda_unique > 0)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator)
        total, batches = 0.0, 0
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            if len(index) < 2:
                continue
            logits = model(features['u'][index], features['rna'][index],
                           features['clr'][index], features['valid'][index],
                           features['route'], features.get('rna_valid')[index]
                           if features.get('rna_valid') is not None else None)
            loss = discrete_nll(logits, labels[index], censorship[index], alpha=alpha)
            if use_aux:
                aux = model.aux_losses(features['u'][index], features['rna'][index],
                                       features['rna_valid'][index]
                                       if features.get('rna_valid') is not None
                                       else features['valid'][index])
                loss = loss + lambda_common * aux['common'] + lambda_unique * aux['unique']
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite survival loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        entry = {'epoch': epoch, 'train_nll': total / max(1, batches)}
        if val_features is not None:
            with torch.no_grad():
                logits = model(val_features['u'], val_features['rna'], val_features['clr'],
                               val_features['valid'], val_features['route'],
                               val_features.get('rna_valid'))
                entry['val_nll'] = float(discrete_nll(logits, val_labels, val_censorship, alpha=0.0))
        history.append(entry)
    return history


@torch.no_grad()
def evaluate_head(model, features, labels, censorship, *, times=None, events=None, alpha=0.0):
    model.eval()
    logits = model(features['u'], features['rna'], features['clr'], features['valid'],
                   features['route'], features.get('rna_valid'))
    nll = discrete_nll(logits, labels, censorship, alpha=alpha, reduction='none')
    survival = discrete_survival(logits)
    risk = 1.0 - survival[:, -1]
    out = {'nll': float(nll.mean()), 'per_patient_nll': nll.tolist(),
           'survival_last': survival[:, -1].tolist(), 'risk': risk.tolist()}
    if times is not None and events is not None:
        out['harrell_c'] = harrell_c(risk, times, events)
    return out


def head_features(encoder, patients, rna, *, route):
    """Frozen features for the survival head.

    ``u`` and ``rna`` are both [N, K, h] when the cluster condition is active
    (one RNA embedding per cluster) and [N, h] when it is cancelled.

    ``rna_valid`` [N, K] is the mask that keeps untrained cluster projections
    out of the RNA-dependent terms.  A cluster is usable for a patient when
    the atlas says the cluster is present (``valid``) *and* the label-free
    stage actually trained that cluster's RNA projection.
    """
    with torch.no_grad():
        u = encoder.encode_path(torch.stack([p.p for p in patients]))
        v = encoder.encode_rna(rna)
        clr = torch.stack([p.clr for p in patients])
        valid = torch.stack([p.valid for p in patients])
        trained = encoder.cluster_trained_mask().to(valid.dtype)
        rna_valid = valid & trained.view(1, -1).bool()
    return {'u': u, 'rna': v, 'clr': clr, 'valid': valid, 'rna_valid': rna_valid,
            'route': route}


def fit_method3_2(atlas, train_bags, train_rna, train_ids, *, config: Method32Config,
                  times=None, events=None, val_bags=None, val_rna=None, val_ids=None,
                  val_times=None, val_events=None):
    """Full two-stage fit on training patients.

    Stage 1 (label free): cluster-conditional contrastive encoder plus a
    cross-fitted cluster retrieval rate.  Stage 2: survival head on frozen
    features.
    """
    patients = atlas.encode(train_bags, train_ids)
    rna = _finite_matrix(train_rna, 'Train RNA')
    if len(patients) != len(rna):
        raise ValueError('RNA rows must match the patient count')

    def build_encoder():
        return ClusterConditionalEncoder(atlas.latent_dim, rna.shape[1], atlas.clusters,
                                         hidden=config.hidden,
                                         cluster_conditioned=config.cluster_conditioned,
                                         seed=config.seed)

    ssl_history = None
    if config.epochs_ssl > 0:
        encoder = build_encoder()
        ssl_history = train_cluster_contrastive(
            encoder, patients, rna, epochs=config.epochs_ssl,
            batch_size=config.batch_size, temperature=config.temperature,
            min_patients=config.min_patients, lr=config.lr_ssl, seed=config.seed,
            shuffle_pairs=config.shuffle_pairs, patch_fraction=config.patch_fraction,
            rna_noise=config.rna_noise, min_mass=config.min_mass)
        a, retrieval_stats = cross_fitted_association(
            build_encoder, patients, rna, folds=config.folds, seed=config.seed,
            epochs=config.epochs_ssl, batch_size=config.batch_size,
            temperature=config.temperature, min_patients=config.min_patients,
            lr=config.lr_ssl, shuffle_pairs=config.shuffle_pairs,
            patch_fraction=config.patch_fraction, rna_noise=config.rna_noise,
            min_mass=config.min_mass)
        encoder = build_encoder()
        train_cluster_contrastive(
            encoder, patients, rna, epochs=config.epochs_ssl,
            batch_size=config.batch_size, temperature=config.temperature,
            min_patients=config.min_patients, lr=config.lr_ssl, seed=config.seed + 101,
            shuffle_pairs=config.shuffle_pairs, patch_fraction=config.patch_fraction,
            rna_noise=config.rna_noise, min_mass=config.min_mass)
    else:
        encoder = build_encoder()
        a = torch.zeros(atlas.clusters, dtype=torch.float64)
        retrieval_stats = {'attempts': [0] * atlas.clusters, 'wins': [0] * atlas.clusters,
                           'a': a.tolist(), 'cross_fitted': False}

    if config.route == 'info':
        route = a.clone()
    elif config.route == 'uniform':
        route = torch.ones(atlas.clusters, dtype=torch.float64)
    else:
        raise ValueError('route must be "info" or "uniform"')

    head = ClusterRoutedSurvivalHead(config)
    train_features = head_features(encoder, patients, rna, route=route)
    labels, censorship, bounds = survival_bin_labels(times, events, bins=config.bins)
    val_features = None
    if val_bags is not None:
        val_patients = atlas.encode(val_bags, val_ids)
        val_features = head_features(encoder, val_patients, _finite_matrix(val_rna, 'Val RNA'),
                                     route=route)
        val_labels, val_censorship, _ = survival_bin_labels(val_times, val_events,
                                                            bins=config.bins)
    history = fit_survival_head(
        head, train_features, labels, censorship, epochs=config.epochs_head,
        batch_size=config.batch_size, lr=config.lr_head, alpha=config.alpha,
        seed=config.seed, val_features=val_features, val_labels=val_labels if val_features is not None else None,
        val_censorship=val_censorship if val_features is not None else None)
    return {'atlas': atlas, 'encoder': encoder, 'head': head, 'route': route,
            'retrieval_a': a, 'retrieval_stats': retrieval_stats, 'ssl_history': ssl_history,
            'head_history': history, 'bin_bounds': bounds, 'train_ids': list(train_ids),
            'cluster_usage': encoder.cluster_usage.tolist(),
            'cluster_trained': encoder.cluster_trained_mask().tolist()}


@torch.no_grad()
def predict_method3_2(fitted, bags, rna, ids):
    """Predict hazard / survival / risk for unseen patients with a frozen model."""
    patients = fitted['atlas'].encode(bags, ids)
    rna = _finite_matrix(rna, 'RNA')
    features = head_features(fitted['encoder'], patients, rna, route=fitted['route'])
    fitted['head'].eval()
    logits = fitted['head'](features['u'], features['rna'], features['clr'],
                            features['valid'], features['route'], features['rna_valid'])
    survival = discrete_survival(logits)
    return {'logits': logits, 'hazard': torch.sigmoid(logits), 'survival': survival,
            'risk': 1.0 - survival[:, -1], 'features': features}
