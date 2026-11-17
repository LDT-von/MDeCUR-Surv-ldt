"""Leakage-aware, validation-selected survival heads for method 3.1.

Graph, SSL and OOF probes are fitted beforehand on training patients only.
The validation set selects an explicit L2 penalty and epoch; it is not used
to fit the bank, SSL or probes. This module never accesses data on disk.
"""
import copy
import math

import torch
from torch import nn

from cancer_morphid_model import FrozenMorphologyBank, discrete_nll
from method3_1_model import Method31
from morphid_probe import CrossFittedRBFProbe


def training_hazard_logits(labels, censorship, bins, *, smooth=.5):
    """Train-only smoothed per-bin event/risk initialization (censor at end)."""
    if type(bins) is not int or bins < 1 or not math.isfinite(smooth) or smooth <= 0:
        raise ValueError('Positive bins and smoothing required')
    y, c = torch.as_tensor(labels), torch.as_tensor(censorship)
    # Reuse the production validator, including shape, bin and censor checks.
    discrete_nll(torch.zeros(y.numel(), bins, dtype=torch.float64), y, c, alpha=0.)
    y, c = y.reshape(-1).long(), c.reshape(-1)
    result = []
    for t in range(bins):
        risk = (y >= t).sum().double()
        events = ((y == t) & (c == 0)).sum().double()
        hazard = (events+smooth)/(risk+2*smooth)
        result.append(torch.logit(hazard))
    return torch.stack(result)


class RegularizedMethod31:
    """One fixed-backbone survival arm with explicit, block-level ablations.

    ``no_residual`` removes both residual main effects and residual products;
    ``no_interaction`` removes only residual products. All retain exactly the
    same bank, RNA, predictable, CLR and projected features. SSL pathology
    projections remain in all arms, so no_residual is not pathology removal.
    """
    ARMS = ('full', 'no_interaction', 'no_residual')

    def __init__(self, backbone, *, arm='full'):
        if not isinstance(backbone, Method31) or backbone.phase != 'probes_frozen':
            raise ValueError('An SSL- and probe-fitted training backbone is required')
        if arm not in self.ARMS:
            raise ValueError('Unknown ablation arm')
        self.backbone, self.arm = backbone, arm
        self.train_ids = tuple(backbone.train_ids)
        self.width = int(backbone.feature_mean.numel())
        self.mask = torch.ones(self.width, dtype=torch.float64)
        kd = backbone.k*backbone.d
        if arm == 'no_residual':
            self.mask[backbone.rna_dim+kd:backbone.rna_dim+2*kd] = 0
        if arm in ('no_residual', 'no_interaction'):
            self.mask[-kd:] = 0
        self.mean = self.scale = self.weight = self.bias = None
        self.selection = None
        self.val_ids = ()

    @torch.no_grad()
    def _train_features(self):
        return self.backbone._features(*self.backbone.cached).detach().clone()

    @torch.no_grad()
    def _fresh_features(self, bags, rna, ids):
        identities, enc, rna = self.backbone._input(bags, rna, ids)
        if set(identities) & set(self.train_ids):
            raise ValueError('Fresh patients must be disjoint from training patients')
        bank_ids = self.backbone.bank.metadata.get('fit_patient_ids', [])
        if set(identities) & set(map(str, bank_ids)):
            raise ValueError('Fresh patients appeared in the morphology bank fit')
        c = torch.stack([self.backbone.probes[str(j)].predict(rna)
                         for j in range(self.backbone.k)], dim=1)
        return self.backbone._features(enc, rna, c).detach()

    def _standardize(self, values):
        return (values-self.mean)/self.scale*self.mask

    def fit(self, labels, censorship, train_ids, val_bags, val_rna, val_ids,
            val_labels, val_censorship, *, penalties=(0., .01, .1, 1., 10.),
            epochs=160, lr=.03, eval_every=5):
        if self.selection is not None:
            raise RuntimeError('Fit once; use a fresh arm for another experiment')
        if tuple(map(str, train_ids)) != self.train_ids:
            raise ValueError('Training labels must follow backbone patient order')
        if (type(epochs) is not int or epochs < 1 or type(eval_every) is not int
                or eval_every < 1 or not math.isfinite(lr) or lr <= 0 or not penalties
                or any(not math.isfinite(p) or p < 0 for p in penalties)):
            raise ValueError('Invalid training grid')
        tr, va = self._train_features(), self._fresh_features(val_bags, val_rna, val_ids)
        val_ids = tuple(map(str, val_ids))
        self.mean = tr.mean(0)
        std = tr.std(0, unbiased=False)
        # Constant training features receive no arbitrary large multiplier on
        # novel held-out values; mean and scale are always train-only.
        self.scale = torch.where(std >= 1e-3, std, torch.ones_like(std))
        tr, va = self._standardize(tr), self._standardize(va)
        train_y, train_c = torch.as_tensor(labels), torch.as_tensor(censorship)
        val_y, val_c = torch.as_tensor(val_labels), torch.as_tensor(val_censorship)
        initial_bias = training_hazard_logits(train_y, train_c, self.backbone.bins)
        best, candidates = None, []
        for penalty in penalties:
            head = nn.Linear(self.width, self.backbone.bins).double()
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.copy_(initial_bias)
            optimizer = torch.optim.Adam(head.parameters(), lr=lr)
            local_best = None
            for epoch in range(1, epochs+1):
                optimizer.zero_grad(set_to_none=True)
                nll = discrete_nll(head(tr), train_y, train_c, alpha=0.)
                # Explicit objective L2 (bias excluded), not AdamW decay.
                objective = nll + .5*float(penalty)*head.weight.square().sum()
                objective.backward()
                optimizer.step()
                if epoch % eval_every == 0 or epoch == epochs:
                    with torch.no_grad():
                        val_nll = float(discrete_nll(head(va), val_y, val_c, alpha=0.))
                    if not math.isfinite(val_nll):
                        raise FloatingPointError('Nonfinite validation NLL')
                    if local_best is None or val_nll < local_best['val_nll']:
                        local_best = {'penalty': float(penalty), 'epoch': epoch,
                                      'val_nll': val_nll,
                                      'weight': head.weight.detach().clone(),
                                      'bias': head.bias.detach().clone()}
            candidates.append({k: local_best[k] for k in ('penalty', 'epoch', 'val_nll')})
            if best is None or local_best['val_nll'] < best['val_nll']:
                best = local_best
        self.weight, self.bias = best['weight'], best['bias']
        self.val_ids = val_ids
        self.selection = {'penalty': best['penalty'], 'epoch': best['epoch'],
                          'validation_nll': best['val_nll'], 'candidates': candidates,
                          'selection': 'validation NLL; fixed grid and epochs',
                          'training_objective': 'mean discrete NLL + penalty/2 * sum(weight^2)',
                          'bias_penalized': False, 'arm': self.arm,
                          'initialization': 'train risk-set hazards with 0.5 event and nonevent pseudo-counts'}
        return copy.deepcopy(self.selection)

    @torch.no_grad()
    def predict(self, bags, rna, ids):
        if self.selection is None:
            raise RuntimeError('Fit the survival arm before prediction')
        ids = tuple(map(str, ids))
        if set(ids) & set(self.val_ids):
            raise ValueError('Selection patients are not an independent evaluation set')
        x = self._standardize(self._fresh_features(bags, rna, ids))
        return x @ self.weight.T + self.bias

    @torch.no_grad()
    def train_logits(self):
        if self.selection is None:
            raise RuntimeError('Fit the survival arm first')
        return self._standardize(self._train_features()) @ self.weight.T + self.bias

    def checkpoint(self):
        if self.selection is None:
            raise RuntimeError('Fit the survival arm first')
        b = self.backbone
        return {'version': 'method3_1_regularized_v1', 'arm': self.arm,
                'model_config': {'rna_dim': b.rna_dim, 'projection_dim': b.projection_dim,
                                 'bins': b.bins, 'temperature': b.temperature, 'alpha': b.alpha,
                                 'seed': b.seed, 'min_effective_mass': b.min_effective_mass},
                'bank': {'centers': b.bank.centers.clone(), 'mean': b.bank.patch_mean.clone(),
                         'scale': b.bank.patch_scale.clone(), 'components': b.bank.components.clone(),
                         'temperature': b.bank.temperature, 'min_mass': b.bank.min_mass,
                         'zero_delta': b.bank.zero_delta, 'metadata': copy.deepcopy(b.bank.metadata)},
                'backbone_state': copy.deepcopy(b.state_dict()), 'train_ids': self.train_ids,
                'val_ids': self.val_ids,
                'mean': self.mean.clone(), 'scale': self.scale.clone(), 'weight': self.weight.clone(),
                'bias': self.bias.clone(), 'selection': copy.deepcopy(self.selection)}

    @classmethod
    def from_checkpoint(cls, snapshot):
        if snapshot.get('version') != 'method3_1_regularized_v1':
            raise ValueError('Unsupported regularized method 3.1 checkpoint')
        b = Method31(FrozenMorphologyBank(**snapshot['bank']), **snapshot['model_config'])
        for j in range(b.k):
            b.probes[str(j)] = CrossFittedRBFProbe(alpha=b.alpha)
        b.load_state_dict(snapshot['backbone_state'], strict=True)
        b.train_ids, b.phase = tuple(snapshot['train_ids']), 'probes_frozen'
        for p in list(b.path_projection.parameters())+list(b.rna_projection.parameters()):
            p.requires_grad_(False)
        obj = cls(b, arm=snapshot['arm'])
        obj.mean, obj.scale = snapshot['mean'].clone(), snapshot['scale'].clone()
        obj.weight, obj.bias = snapshot['weight'].clone(), snapshot['bias'].clone()
        obj.selection = copy.deepcopy(snapshot['selection'])
        obj.val_ids = tuple(snapshot['val_ids'])
        return obj
