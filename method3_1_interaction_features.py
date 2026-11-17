"""Outcome-free, patient-cross-fitted global residual interaction features.

The morphology bank and SSL representation are fitted first on the outer
training patients. OOF below is conditional on that shared outer-training
basis, not a fully nested assessment of morphology or SSL. Global residuals
mean prediction errors under an RBF KRR family, not independence or PID.
All arrays, including inputs retained by the kernel probes, may be sensitive.
"""
import copy

import torch

from cancer_morphid_model import FrozenMorphologyBank, _row_blocks
from method3_1_model import Method31, _ids, _matrix
from morphid_probe import CrossFittedRBFProbe


def _stable_moments(values):
    values = torch.as_tensor(values).detach().cpu().double()
    if values.ndim != 2 or not len(values) or not torch.isfinite(values).all():
        raise ValueError('Finite nonempty training feature matrix required')
    mean, std = values.mean(0), values.std(0, unbiased=False)
    return mean, torch.where(std >= 1e-3, std, torch.ones_like(std))


def _clone_features(features):
    return {k: v.detach().clone() for k, v in features.items()}


class GlobalResidualInteractions:
    """Three parameter-free interaction maps over one fixed 3.1 backbone.

    ``diagonal`` is the existing cluster residual × fixed RNA projection;
    ``outer`` and ``tanh_outer`` are the full global residual × RNA products
    after training-only component normalization. All use identical patients.
    Supervised survival fitting and model selection belong to the caller.
    """
    def __init__(self, backbone):
        if not isinstance(backbone, Method31) or backbone.phase != 'probes_frozen':
            raise ValueError('A training-only SSL and probe-fitted backbone is required')
        self.backbone = backbone
        self.train_ids = tuple(backbone.train_ids)
        self.val_ids = ()
        self.probe = None
        self.u_mean = self.u_scale = self.rna_mean = self.rna_scale = None
        self._train_features = None
        self.metadata = None

    def _validate(self, bags, rna, ids):
        bags = list(bags)
        identities = _ids(ids, len(bags))
        for identity, bag in zip(identities, bags):
            if hasattr(bag, 'patient_id') and str(bag.patient_id) != identity:
                raise ValueError('Patient ID differs from the supplied bag identity')
        rna = _matrix(rna, len(bags), self.backbone.rna_dim, 'RNA')
        return bags, rna, identities

    @torch.no_grad()
    def _global_mean(self, bags, chunk_size=4096):
        """Mean every real patch in frozen standardized/PCA coordinates.

        It is computed directly, not by summing cluster features after
        availability masking, and therefore preserves small/absent clusters.
        """
        bank = self.backbone.bank
        output = []
        for bag in bags:
            total = torch.zeros(bank.latent_dim, dtype=torch.float64)
            count = 0
            for block in _row_blocks(bag, chunk_size):
                if block.shape[1] != len(bank.patch_mean):
                    raise ValueError('Patch dimension differs from morphology bank')
                z = ((block-bank.patch_mean)/bank.patch_scale) @ bank.components
                total += z.sum(0)
                count += len(block)
            if not count:
                raise ValueError('Every patient requires at least one real patch')
            output.append(total/count)
        return torch.stack(output)

    def _construct(self, pooled, rna, prediction, encoded, cluster_c):
        u = (pooled-prediction-self.u_mean)/self.u_scale
        g = (rna-self.rna_mean)/self.rna_scale
        raw_outer = (u.unsqueeze(2)*g.unsqueeze(1)).flatten(1)
        tanh_outer = (torch.tanh(u).unsqueeze(2)*torch.tanh(g).unsqueeze(1)).flatten(1)
        # Existing feature code applies the cluster availability mask.
        diagonal = self.backbone._features(encoded, rna, cluster_c)[:, -self.backbone.k*self.backbone.d:]
        return {'global_pooled': pooled, 'global_prediction': prediction,
                'global_residual_standardized': u, 'rna_standardized': g,
                'outer': raw_outer, 'tanh_outer': tanh_outer, 'diagonal': diagonal}

    def fit(self, bags, rna, ids, *, folds=4, seed=None):
        """Fit KRR and normalization using outer-training patients, no labels."""
        if self.probe is not None:
            raise RuntimeError('A fitted feature map cannot be refitted')
        bags, rna, identities = self._validate(bags, rna, ids)
        if identities != self.train_ids:
            raise ValueError('Feature fitting must match backbone training patients in order')
        if self.backbone.cached is None:
            raise ValueError('Backbone training OOF cache is required to verify matching training inputs')
        cached_enc, cached_rna, cluster_c = self.backbone.cached
        if not torch.equal(rna, cached_rna):
            raise ValueError('RNA differs from the backbone training cache')
        _, encoded, _ = self.backbone._input(bags, rna, identities)
        for key in ('pooled', 'composition', 'available', 'real_patches'):
            if not torch.equal(encoded[key], cached_enc[key]):
                raise ValueError('Bags differ from the backbone training cache')
        pooled = self._global_mean(bags)
        seed = self.backbone.seed if seed is None else seed
        probe = CrossFittedRBFProbe(alpha=self.backbone.alpha).fit(
            rna, pooled, identities, folds=folds, seed=seed)
        u = pooled-probe.oof_predictions
        self.u_mean, self.u_scale = _stable_moments(u)
        self.rna_mean, self.rna_scale = _stable_moments(rna)
        self._train_features = _clone_features(
            self._construct(pooled, rna, probe.oof_predictions, encoded, cluster_c))
        self.probe = probe
        self.metadata = {'folds': folds, 'seed': seed, 'survival_labels_used': False,
                         'training': 'OOF KRR conditional on shared outer-training bank and SSL',
                         'inference': 'mean of fold-model predictions',
                         'normalization': 'training OOF residual and RNA; std below 1e-3 -> 1',
                         'residual_is_independence_proof': False,
                         'split_audit': copy.deepcopy(probe.split_audit)}
        return self

    def training_features(self):
        if self.probe is None:
            raise RuntimeError('Fit global cross-fitted probe first')
        return _clone_features(self._train_features)

    @torch.no_grad()
    def _fresh(self, bags, rna, ids):
        if self.probe is None:
            raise RuntimeError('Fit global cross-fitted probe first')
        bags, rna, identities = self._validate(bags, rna, ids)
        if set(identities) & set(self.train_ids):
            raise ValueError('Fresh patients overlap training patients')
        bank_ids = set(map(str, self.backbone.bank.metadata.get('fit_patient_ids', [])))
        if set(identities) & bank_ids:
            raise ValueError('Fresh patients appeared in the morphology bank fit')
        _, encoded, _ = self.backbone._input(bags, rna, identities)
        cluster_c = torch.stack([self.backbone.probes[str(j)].predict(rna)
                                 for j in range(self.backbone.k)], dim=1)
        pooled, prediction = self._global_mean(bags), self.probe.predict(rna)
        return identities, _clone_features(self._construct(pooled, rna, prediction, encoded, cluster_c))

    def validation_features(self, bags, rna, ids):
        identities, features = self._fresh(bags, rna, ids)
        if self.val_ids and identities != self.val_ids:
            raise ValueError('Validation patient identities are already registered')
        self.val_ids = identities
        return features

    def test_features(self, bags, rna, ids):
        if not self.val_ids:
            raise ValueError('Register validation patients before evaluating a test set')
        identities, features = self._fresh(bags, rna, ids)
        if set(identities) & set(self.val_ids):
            raise ValueError('Test patients overlap validation patients')
        return features

    def checkpoint(self):
        """Complete feature-map snapshot; kernel models retain training inputs."""
        if self.probe is None:
            raise RuntimeError('Fit global cross-fitted probe first')
        b = self.backbone
        return {'version': 'method3_1_global_interactions_v1',
                'model_config': {'rna_dim': b.rna_dim, 'projection_dim': b.projection_dim,
                                 'bins': b.bins, 'temperature': b.temperature, 'alpha': b.alpha,
                                 'seed': b.seed, 'min_effective_mass': b.min_effective_mass},
                'bank': {'centers': b.bank.centers.clone(), 'mean': b.bank.patch_mean.clone(),
                         'scale': b.bank.patch_scale.clone(), 'components': b.bank.components.clone(),
                         'temperature': b.bank.temperature, 'min_mass': b.bank.min_mass,
                         'zero_delta': b.bank.zero_delta, 'metadata': copy.deepcopy(b.bank.metadata)},
                'backbone_state': copy.deepcopy(b.state_dict()), 'train_ids': self.train_ids,
                'val_ids': self.val_ids, 'probe_state': copy.deepcopy(self.probe.state_dict()),
                'u_mean': self.u_mean.clone(), 'u_scale': self.u_scale.clone(),
                'rna_mean': self.rna_mean.clone(), 'rna_scale': self.rna_scale.clone(),
                'training_features': _clone_features(self._train_features), 'metadata': copy.deepcopy(self.metadata)}

    @classmethod
    def from_checkpoint(cls, snapshot):
        if snapshot.get('version') != 'method3_1_global_interactions_v1':
            raise ValueError('Unsupported interaction feature checkpoint')
        b = Method31(FrozenMorphologyBank(**snapshot['bank']), **snapshot['model_config'])
        for j in range(b.k):
            b.probes[str(j)] = CrossFittedRBFProbe(alpha=b.alpha)
        b.load_state_dict(snapshot['backbone_state'], strict=True)
        b.train_ids, b.phase = tuple(snapshot['train_ids']), 'probes_frozen'
        for p in list(b.path_projection.parameters())+list(b.rna_projection.parameters()):
            p.requires_grad_(False)
        obj = cls(b)
        obj.val_ids = tuple(snapshot['val_ids'])
        obj.probe = CrossFittedRBFProbe(alpha=b.alpha)
        obj.probe.load_state_dict(snapshot['probe_state'], strict=True)
        obj.u_mean, obj.u_scale = snapshot['u_mean'].clone(), snapshot['u_scale'].clone()
        obj.rna_mean, obj.rna_scale = snapshot['rna_mean'].clone(), snapshot['rna_scale'].clone()
        obj._train_features = _clone_features(snapshot['training_features'])
        obj.metadata = copy.deepcopy(snapshot['metadata'])
        return obj
