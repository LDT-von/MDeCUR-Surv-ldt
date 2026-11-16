"""MorphID frozen-representation mechanism model, independent of DensePathway.

Unlabelled morphology and patient-axis cross-fitted RNA probes are frozen.
Only the discrete-survival head receives outcome gradients. Prediction
residuals are not identified independent/private biological information.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import torch
from torch import nn

from cancer_parity import DISCRETE_NLL_VERSION, discrete_nll, discrete_survival
from morphid_probe import CrossFittedRBFProbe, patient_fold_ids, r2_score


MORPHID_VERSION = 'morphid_frozen_conditional_residual_v1'


def _matrix(value, name):
    tensor = torch.as_tensor(value)
    if tensor.requires_grad:
        raise ValueError(f'{name} must be an already frozen representation')
    tensor = tensor.detach().cpu().double()
    if tensor.ndim != 2 or min(tensor.shape) < 1 or not torch.isfinite(tensor).all():
        raise ValueError(f'{name} requires a finite nonempty matrix')
    return tensor


def _identities(ids, bags):
    if isinstance(ids, (str, bytes)):
        raise ValueError('Patient IDs must be a sequence')
    supplied = tuple(ids)
    if any(value is None for value in supplied):
        raise ValueError('Patient IDs cannot be missing')
    ids = tuple(str(value) for value in supplied)
    if len(ids) != len(bags) or len(set(ids)) != len(ids) or any(not value or value != value.strip() for value in ids):
        raise ValueError('Unique patient IDs must match the supplied bags')
    for identity, bag in zip(ids, bags):
        if hasattr(bag, 'patient_id') and str(bag.patient_id) != identity:
            raise ValueError('Patient ID differs from the bag identity')
    return ids


def _row_blocks(bag, chunk_size):
    """Exclude padding before finite checks, including padding-only chunks."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Positive integer chunk size required')
    if isinstance(bag, torch.Tensor):
        rows = _matrix(bag, 'Frozen patch features')
        yield from rows.split(chunk_size)
        return
    if isinstance(bag, tuple) and len(bag) == 2:
        rows, mask = map(torch.as_tensor, bag)
        if rows.ndim != 2 or mask.dtype != torch.bool or mask.shape != rows.shape[:1]:
            raise ValueError('A masked bag requires [patches,features] and a boolean patch mask')
        rows = _matrix(rows[mask], 'Real masked patch features')
        yield from rows.split(chunk_size)
        return
    if not hasattr(bag, 'blocks') or not hasattr(bag, 'read_block'):
        raise TypeError('Use a tensor, (tensor,mask), or a reference-feature streaming bag')
    total = getattr(bag, 'n_patches', None)
    padding = getattr(bag, 'padding', 0)
    if type(total) is not int or type(padding) is not int or not 0 <= padding < total:
        raise ValueError('A streaming bag needs at least one declared real patch')
    real, cursor = total-padding, 0
    for token in bag.blocks(chunk_size):
        length = int(token[2]-token[1])
        if length < 1 or cursor+length > total:
            raise ValueError('Invalid streaming block coverage')
        count = max(0, min(length, real-cursor))
        if count:
            rows, = bag.read_block(token, ('features_reference',))
            if rows.ndim != 2 or len(rows) != length:
                raise ValueError('Streaming block row count differs from its token')
            yield _matrix(rows[:count], 'Real streaming patch features')
        cursor += length
    if cursor != total:
        raise ValueError('Streaming blocks do not cover the declared budget')


def composition_clr(composition, zero_delta=1e-6):
    """Multiplicative zero replacement then CLR, preserving nonzero ratios."""
    pi = _matrix(composition, 'Cluster composition')
    if (pi < 0).any() or (pi.sum(1) <= 0).any():
        raise ValueError('Cluster masses must be nonnegative with positive row totals')
    if not math.isfinite(zero_delta) or not 0 < zero_delta < 1/pi.shape[1]:
        raise ValueError('Zero replacement must lie strictly between zero and 1/K')
    pi = pi/pi.sum(1, keepdim=True)
    zeros = pi == 0
    remaining = 1-zeros.sum(1, keepdim=True)*zero_delta
    replaced = torch.where(zeros, torch.full_like(pi, zero_delta), pi*remaining)
    logs = replaced.log()
    return logs-logs.mean(1, keepdim=True)


class FrozenMorphologyBank(nn.Module):
    """Fixed feature-space prototypes; not HPL's Leiden/KNN implementation.

    A supplied bank may come from a separately fitted unlabelled reference.
    ``fit`` offers bounded, patient-balanced PCA/k-means for mechanism work.
    No survival label is accepted. This supplies feature clusters, not a
    pathologist-validated phenotype or an automatic choice of clinical K.
    """
    def __init__(self, centers, *, mean=None, scale=None, components=None,
                 temperature=1., min_mass=1e-4, zero_delta=1e-6, metadata=None):
        super().__init__()
        centers = _matrix(centers, 'Morphology centers')
        dimension = centers.shape[1]
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Positive prototype temperature required')
        if not math.isfinite(min_mass) or not 0 <= min_mass < 1:
            raise ValueError('Cluster mass cutoff must lie in [0,1)')
        components = torch.eye(dimension, dtype=torch.float64) if components is None else _matrix(components, 'Frozen components')
        if components.shape[1] != dimension:
            raise ValueError('Projection and prototype dimensions differ')
        mean = torch.zeros(components.shape[0], dtype=torch.float64) if mean is None else torch.as_tensor(mean).double().cpu()
        scale = torch.ones_like(mean) if scale is None else torch.as_tensor(scale).double().cpu()
        if (mean.shape != (components.shape[0],) or scale.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('Invalid frozen patch preprocessing')
        for name, value in (('centers', centers), ('patch_mean', mean), ('patch_scale', scale), ('components', components)):
            self.register_buffer(name, value.detach().clone())
        self.temperature, self.min_mass, self.zero_delta = float(temperature), float(min_mass), float(zero_delta)
        composition_clr(torch.ones(1, len(centers)), zero_delta)
        self.metadata = copy.deepcopy(metadata or {'source': 'caller_supplied_frozen_centers', 'fit_patient_ids': []})

    @property
    def clusters(self):
        return len(self.centers)

    @property
    def latent_dim(self):
        return self.centers.shape[1]

    @classmethod
    @torch.no_grad()
    def fit(cls, bags, *, ids, clusters=4, latent_dim=8, max_fit_patches=4096,
            per_patient_cap=128, iterations=25, seed=3, **kwargs):
        bags = list(bags)
        ids = _identities(ids, bags)
        if (not bags or any(type(v) is not int or v < 1 for v in
                            (clusters, latent_dim, max_fit_patches, per_patient_cap, iterations))
                or max_fit_patches < len(bags)):
            raise ValueError('Invalid unlabelled bank fitting budget')
        cap = min(per_patient_cap, max_fit_patches//len(bags))
        generator = torch.Generator().manual_seed(seed)
        samples = []
        for bag in bags:
            retained, priorities = None, torch.empty(0, dtype=torch.float64)
            for rows in _row_blocks(bag, 4096):
                keys = torch.rand(len(rows), generator=generator, dtype=torch.float64)
                retained = rows if retained is None else torch.cat((retained, rows))
                priorities = torch.cat((priorities, keys))
                chosen = priorities.argsort()[:cap]
                retained, priorities = retained[chosen], priorities[chosen]
            samples.append(retained)
        sample = torch.cat(samples)
        if len(sample) < max(clusters, 2):
            raise ValueError('Too few unlabelled patches for the requested bank')
        mean, scale = sample.mean(0), sample.std(0, unbiased=False)
        scale = torch.where(scale >= 1e-8, scale, torch.ones_like(scale))
        normalized = (sample-mean)/scale
        _, _, right = torch.linalg.svd(normalized, full_matrices=False)
        dimension = min(latent_dim, sample.shape[1], len(sample)-1)
        components = right[:dimension].T.contiguous()
        anchors = components.abs().argmax(0)
        signs = components[anchors, torch.arange(dimension)].sign()
        components *= torch.where(signs == 0, torch.ones_like(signs), signs)
        z = normalized @ components
        centers = [z[int(torch.randint(len(z), (), generator=generator))].clone()]
        distance = (z-centers[0]).square().sum(1)
        for _ in range(1, clusters):
            index = int(distance.argmax())
            if float(distance[index]) <= 1e-12:
                raise ValueError('Too few distinct representations; reduce cluster count')
            centers.append(z[index].clone())
            distance = torch.minimum(distance, (z-centers[-1]).square().sum(1))
        centers = torch.stack(centers)
        for _ in range(iterations):
            distances = torch.cdist(z, centers).square()
            labels = distances.argmin(1)
            updated = []
            for k in range(clusters):
                selected = z[labels == k]
                updated.append(selected.mean(0) if len(selected) else z[int(distances.min(1).values.argmax())])
            updated = torch.stack(updated)
            if torch.allclose(updated, centers, rtol=0, atol=1e-10):
                centers = updated
                break
            centers = updated
        counts = torch.bincount(torch.cdist(z, centers).argmin(1), minlength=clusters)
        if (counts == 0).any():
            raise ValueError('Empty fitted prototype; reduce cluster count')
        return cls(centers, mean=mean, scale=scale, components=components,
                   metadata={'source': 'frozen_train_only_pca_kmeans', 'fit_patient_ids': list(ids),
                             'sampled_patches': len(sample), 'hard_fit_counts': counts.tolist(),
                             'seed': seed, 'outcomes_used': False, 'hpl_replication': False}, **kwargs)

    @torch.no_grad()
    def encode(self, bags, *, chunk_size=4096):
        if self.centers.device.type != 'cpu':
            raise ValueError('Frozen mechanism implementation currently requires CPU buffers')
        pooled, compositions, available, counts = [], [], [], []
        for bag in bags:
            numerator = torch.zeros_like(self.centers)
            mass = torch.zeros(self.clusters, dtype=torch.float64)
            count = 0
            for rows in _row_blocks(bag, chunk_size):
                if rows.shape[1] != len(self.patch_mean):
                    raise ValueError('Patch feature dimension differs from the frozen bank')
                z = ((rows-self.patch_mean)/self.patch_scale) @ self.components
                q = torch.softmax(-torch.cdist(z, self.centers).square()/self.temperature, dim=1)
                mass += q.sum(0)
                numerator += q.T @ z
                count += len(rows)
            if not count:
                raise ValueError('At least one real patch is required')
            pi = mass/count
            valid = (pi >= self.min_mass) & (mass > 0)
            p = numerator/mass.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
            p[~valid] = 0.
            pooled.append(p); compositions.append(pi); available.append(valid); counts.append(count)
        if not pooled:
            raise ValueError('At least one patient bag is required')
        pi = torch.stack(compositions)
        return {'pooled': torch.stack(pooled), 'composition': pi,
                'clr': composition_clr(pi, self.zero_delta),
                'available': torch.stack(available), 'real_patches': torch.tensor(counts)}

    def get_extra_state(self):
        return {'temperature': self.temperature, 'min_mass': self.min_mass,
                'zero_delta': self.zero_delta, 'metadata': self.metadata}

    def set_extra_state(self, state):
        self.temperature, self.min_mass, self.zero_delta = state['temperature'], state['min_mass'], state['zero_delta']
        self.metadata = copy.deepcopy(state['metadata'])


class _MorphIDHead(nn.Module):
    """Additive blocks plus an explicit residual-by-RNA product path."""
    def __init__(self, rna_dim, clusters, dimension, bins, rank, interaction):
        super().__init__()
        self.clusters, self.dimension, self.use_interaction = clusters, dimension, bool(interaction)
        self.bias = nn.Parameter(torch.zeros(bins))
        self.rna = nn.Linear(rna_dim, bins, bias=False)
        # Predicted morphology is an RNA-derived nonlinear basis. Keeping it
        # prevents a limited RNA head from losing predictable prognostic signal.
        self.predictable = nn.Linear(clusters*dimension, bins, bias=False)
        self.composition = nn.Linear(clusters, bins, bias=False)
        self.residual = nn.Linear(clusters*dimension, bins, bias=False)
        self.residual_factor = nn.Linear(dimension, rank, bias=False)
        self.rna_factor = nn.Linear(rna_dim, rank, bias=False)
        self.interaction_head = nn.Linear(rank, bins, bias=False)

    def forward(self, features):
        rna, residual = features['rna'], features['residual']
        logits = (self.bias+self.rna(rna)+self.predictable(features['predictable'])
                  +self.composition(features['composition'])+self.residual(residual.flatten(1)))
        interaction = torch.zeros_like(logits)
        if self.use_interaction:
            product = torch.tanh(self.residual_factor(features['interaction_residual']))*torch.tanh(self.rna_factor(rna))[:, None]
            interaction = self.interaction_head(product.mean(1))
        return logits+interaction, interaction


class MorphIDSurvival(nn.Module):
    """One frozen-bank/cross-fitted-probe model with a trainable survival head.

    Probe OOF scores are conditional on the supplied fixed morphology basis;
    fitting that basis on all outer-training patients is not fully nested
    morphology validation. An independent unlabelled bank is used in the
    synthetic mechanism tests. There is no R² gate asserting independence.
    """
    def __init__(self, morphology, rna_dim, *, bins=4, interaction_rank=8,
                 interaction=True, probe_config=None, min_probe_patients=8, seed=3):
        super().__init__()
        if not isinstance(morphology, FrozenMorphologyBank):
            raise TypeError('A frozen morphology bank is required')
        if any(type(v) is not int or v < 1 for v in (rna_dim, bins, interaction_rank, min_probe_patients)):
            raise ValueError('Positive integer dimensions and patient minimum required')
        self.morphology, self.rna_dim, self.bins = morphology, rna_dim, bins
        self.probe_config = copy.deepcopy(probe_config or {'alpha': 1e-3})
        self.min_probe_patients, self.seed = min_probe_patients, seed
        self.probes = nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = _MorphIDHead(rna_dim, morphology.clusters, morphology.latent_dim,
                                     bins, interaction_rank, interaction).double()
        self.interaction_rank, self.use_interaction = interaction_rank, bool(interaction)
        self.register_buffer('fitted_flag', torch.tensor(False))
        self.register_buffer('probe_available', torch.zeros(morphology.clusters, dtype=torch.bool))
        dimensions = {'rna': rna_dim, 'predictable': morphology.clusters*morphology.latent_dim,
                      'composition': morphology.clusters, 'residual': morphology.clusters*morphology.latent_dim}
        for name, dimension in dimensions.items():
            self.register_buffer(name+'_mean', torch.zeros(dimension, dtype=torch.float64))
            self.register_buffer(name+'_scale', torch.ones(dimension, dtype=torch.float64))
            self.register_buffer(name+'_mask', torch.ones(dimension, dtype=torch.bool))
        self._training_features, self._train_ids, self.audit = None, (), {}
        self._fit_attempted = False

    def frozen_digest(self):
        digest = hashlib.sha256()
        for name, value in self.state_dict().items():
            if name.startswith('head.') or not isinstance(value, torch.Tensor):
                continue
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _raw_blocks(self, encoded, rna, predictable):
        valid = encoded['available'] & self.probe_available[None]
        residual = (encoded['pooled']-predictable)*valid[..., None]
        return {'rna': rna, 'predictable': predictable.flatten(1),
                'composition': encoded['clr'],
                'residual': (residual*encoded['composition'].sqrt()[..., None]).flatten(1)}, residual, valid

    def _normalize(self, blocks, available):
        features = {}
        for name, values in blocks.items():
            centered = torch.where(getattr(self, name+'_mask'), values-getattr(self, name+'_mean'), torch.zeros_like(values))
            features[name] = centered/getattr(self, name+'_scale')
        features['residual'] = features['residual'].reshape(-1, self.morphology.clusters, self.morphology.latent_dim)
        features['residual'] *= available[..., None]
        # No centering in the product path: a zero/absent raw residual must
        # contribute exactly zero, rather than a -training_mean artifact.
        product = torch.where(self.residual_mask, blocks['residual']/self.residual_scale,
                              torch.zeros_like(blocks['residual']))
        features['interaction_residual'] = product.reshape_as(features['residual'])*available[..., None]
        return features

    @torch.no_grad()
    def fit_frozen(self, bags, rna, *, ids, folds=4, seed=3, chunk_size=4096):
        """Fit only unlabelled reconstruction components; accepts no outcomes."""
        if bool(self.fitted_flag) or self._fit_attempted:
            raise RuntimeError('Frozen fitting cannot be retried/refitted; create a new model')
        bags = list(bags); identities = _identities(ids, bags)
        rna = _matrix(rna, 'Frozen RNA representations')
        if rna.shape != (len(bags), self.rna_dim):
            raise ValueError('One fixed RNA representation per patient is required')
        assignments = patient_fold_ids(identities, folds, seed)
        encoded = self.morphology.encode(bags, chunk_size=chunk_size)
        # After input validation, a failed attempt may contain partial probe
        # state. Refuse reuse rather than carry stale clusters into a retry.
        self._fit_attempted = True
        predictable = torch.zeros_like(encoded['pooled'])
        audits = []
        for k in range(self.morphology.clusters):
            selected = torch.where(encoded['available'][:, k])[0]
            counts = torch.bincount(assignments[selected], minlength=folds)
            reason = None
            if len(selected) < self.min_probe_patients:
                reason = 'insufficient_valid_patients'
            elif (counts == 0).any() or (len(selected)-counts < 2).any():
                reason = 'insufficient_fold_coverage'
            if reason:
                audits.append({'cluster': k, 'fitted': False, 'reason': reason,
                               'valid_patients': len(selected), 'independence_established': False})
                continue
            probe = CrossFittedRBFProbe(**self.probe_config).fit(
                rna[selected], encoded['pooled'][selected, k], [identities[i] for i in selected],
                folds=folds, seed=seed, fold_ids=assignments[selected])
            prediction = probe.predict(rna)
            prediction[selected] = probe.oof_predictions
            predictable[:, k] = prediction
            self.probes[str(k)] = probe
            self.probe_available[k] = True
            audits.append({'cluster': k, 'fitted': True, 'valid_patients': len(selected),
                           'oof_r2': r2_score(encoded['pooled'][selected, k], probe.oof_predictions),
                           'splits': copy.deepcopy(probe.split_audit), 'independence_established': False})
        if not bool(self.probe_available.any()):
            raise ValueError('No cluster supports patient-level cross-fitted probes')
        blocks, _, available = self._raw_blocks(encoded, rna, predictable)
        for name, values in blocks.items():
            mean, scale = values.mean(0), values.std(0, unbiased=False)
            mask = scale >= 1e-8
            getattr(self, name+'_mean').copy_(mean)
            getattr(self, name+'_scale').copy_(torch.where(mask, scale, torch.ones_like(scale)))
            getattr(self, name+'_mask').copy_(mask)
        self._training_features = {k: v.clone() for k, v in self._normalize(blocks, available).items()}
        self._train_ids = identities
        self.fitted_flag.fill_(True)
        self.audit = {'version': MORPHID_VERSION, 'loss_version': DISCRETE_NLL_VERSION,
                      'fit_patients': len(bags), 'probe_clusters': audits,
                      'decomposition_fit_uses_survival_labels': False,
                      'input_representation_provenance': 'Caller supplied; prior survival supervision is not inferred from frozen tensors.',
                      'morphology_basis': copy.deepcopy(self.morphology.metadata),
                      'training_residual_source': 'patient-out-of-fold probe predictions',
                      'inference_probe': 'mean of fixed fold models; finite-sample shift from OOF is possible',
                      'r2_used_as_gate': False, 'independence_established': False,
                      'scope': 'Frozen CPU mechanism implementation; no patient benefit or HPL interpretation claim'}
        self.eval()
        return self

    @torch.no_grad()
    def encode_for_inference(self, bags, rna, *, ids, chunk_size=4096):
        if not bool(self.fitted_flag):
            raise RuntimeError('Frozen MorphID components have not been fitted')
        bags = list(bags); identities = _identities(ids, bags)
        fitted = set(self._train_ids) | set(self.morphology.metadata.get('fit_patient_ids', []))
        if fitted.intersection(identities):
            raise ValueError('Inference IDs overlap fitting patients; use cached training OOF logits for training diagnostics')
        rna = _matrix(rna, 'Frozen RNA representations')
        if rna.shape != (len(bags), self.rna_dim):
            raise ValueError('One fixed RNA representation per patient is required')
        encoded = self.morphology.encode(bags, chunk_size=chunk_size)
        predicted = torch.zeros_like(encoded['pooled'])
        for name, probe in self.probes.items():
            predicted[:, int(name)] = probe.predict(rna)
        blocks, residual, available = self._raw_blocks(encoded, rna, predicted)
        return self._normalize(blocks, available), {**encoded, 'predictable': predicted, 'residual': residual,
                                                   'residual_available': available}

    def forward(self, bags, rna, *, ids, chunk_size=4096):
        if self.training:
            raise RuntimeError('Train the head with cached OOF features via fit_survival_head')
        features, evidence = self.encode_for_inference(bags, rna, ids=ids, chunk_size=chunk_size)
        logits, interaction = self.head(features)
        survival = discrete_survival(logits)
        return {'logits': logits, 'survival': survival, 'risk': -survival.sum(1),
                'interaction_logits': interaction, 'evidence': evidence}

    @torch.no_grad()
    def training_oof_logits(self):
        if self._training_features is None:
            raise RuntimeError('OOF training features are not present in this deployment checkpoint')
        return self.head(self._training_features)[0]

    def fit_survival_head(self, labels, censorship, *, ids, epochs=30, lr=5e-4,
                          weight_decay=5e-4, alpha=.5, batch_size=20, seed=3):
        if self._training_features is None or tuple(map(str, ids)) != self._train_ids:
            raise ValueError('Labels must follow the cached OOF patient ID order')
        if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1
                or not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0):
            raise ValueError('Invalid survival-head training configuration')
        labels, censorship = torch.as_tensor(labels), torch.as_tensor(censorship)
        initial = float(discrete_nll(self.training_oof_logits(), labels, censorship, alpha=alpha))
        before = self.frozen_digest()
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=weight_decay)
        generator = torch.Generator().manual_seed(seed)
        history = []
        self.head.train()
        for epoch in range(1, epochs+1):
            order = torch.randperm(len(self._train_ids), generator=generator)
            steps = 0
            for selected in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                features = {k: v[selected] for k, v in self._training_features.items()}
                logits, _ = self.head(features)
                loss = discrete_nll(logits, labels[selected], censorship[selected], alpha=alpha)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.head.parameters()):
                    raise FloatingPointError('Nonfinite MorphID head gradient')
                optimizer.step(); steps += 1
            history.append({'epoch': epoch, 'optimizer_steps': steps,
                            'training_nll': float(discrete_nll(self.training_oof_logits(), labels, censorship, alpha=alpha))})
        self.eval()
        if self.frozen_digest() != before:
            raise RuntimeError('Frozen morphology/probe/normalization state changed during head training')
        return {'initial_training_nll': initial, 'history': history, 'frozen_state_unchanged': True,
                'optimizer_scope': 'survival_head_only', 'source': 'cached_patient_OOF_features'}

    def checkpoint(self):
        if not bool(self.fitted_flag):
            raise RuntimeError('Cannot checkpoint an unfitted MorphID model')
        return {'architecture': MORPHID_VERSION, 'loss_version': DISCRETE_NLL_VERSION,
                'config': {'rna_dim': self.rna_dim, 'bins': self.bins,
                           'interaction_rank': self.interaction_rank, 'interaction': self.use_interaction,
                           'probe_config': self.probe_config, 'min_probe_patients': self.min_probe_patients,
                           'seed': self.seed}, 'state': copy.deepcopy(self.state_dict()),
                'probe_keys': list(self.probes), 'train_ids': list(self._train_ids),
                'audit': copy.deepcopy(self.audit)}

    @classmethod
    def from_checkpoint(cls, payload):
        if payload.get('architecture') != MORPHID_VERSION or payload.get('loss_version') != DISCRETE_NLL_VERSION:
            raise ValueError('Unsupported MorphID checkpoint/loss version')
        state = payload['state']
        bank = FrozenMorphologyBank(state['morphology.centers'], mean=state['morphology.patch_mean'],
                                    scale=state['morphology.patch_scale'], components=state['morphology.components'])
        model = cls(bank, **payload['config'])
        for key in payload['probe_keys']:
            model.probes[key] = CrossFittedRBFProbe(**model.probe_config)
        model.load_state_dict(state, strict=True)
        model._train_ids, model.audit = tuple(payload['train_ids']), copy.deepcopy(payload['audit'])
        model.eval()
        return model
