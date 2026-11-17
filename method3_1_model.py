"""Method 3.1: frozen morphology, patient-level contrastive SSL, and survival.

Research prototype for CPU feature bags.  The supplied morphology bank must be
fitted on training patients only; its k-means option is not Leiden.  No outcome
is accepted until after self-supervision and cross-fitted probes are frozen.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from cancer_morphid_model import FrozenMorphologyBank, discrete_nll
from morphid_probe import CrossFittedRBFProbe, patient_fold_ids


def _ids(ids, n):
    if isinstance(ids, (str, bytes)):
        raise ValueError('Provide one distinct patient ID per row')
    values = tuple(ids)
    if any(x is None for x in values):
        raise ValueError('Patient IDs cannot be missing')
    values = tuple(str(x) for x in values)
    if len(values) != n or len(set(values)) != n or any(not x.strip() for x in values):
        raise ValueError('Provide one distinct patient ID per row')
    return values


def _matrix(x, n, d, name):
    x = torch.as_tensor(x, dtype=torch.float64).detach().cpu()
    if x.shape != (n, d) or not torch.isfinite(x).all():
        raise ValueError(f'{name} must be a finite [{n}, {d}] matrix')
    return x


def patient_contrastive(left, right, temperature):
    """Symmetric patient-level contrast; exactly one positive per patient.

    Callers select a cluster's available patients before calling.  Rows must
    be unique patients and in matching order; no patch is treated as a patient.
    """
    if left.ndim != 2 or right.shape != left.shape or len(left) < 2:
        raise ValueError('At least two paired patient representations required')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be positive')
    scores = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).T / temperature
    target = torch.arange(len(left), device=left.device)
    return (F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target)) / 2


class Method31(nn.Module):
    """Independent method-three candidate, not a method-two extension.

    The common outer-training bank and SSL basis are held fixed while probes
    cross-fit. Consequently probe OOF predictions are conditional on that
    basis, not a fully nested evaluation of representation learning.
    """
    def __init__(self, morphology, rna_dim, *, projection_dim=8, bins=4,
                 temperature=.2, alpha=1e-3, seed=3, min_effective_mass=1.):
        super().__init__()
        if not isinstance(morphology, FrozenMorphologyBank):
            raise TypeError('A fixed training-only morphology bank is required')
        if any(type(v) is not int or v < 1 for v in (rna_dim, projection_dim, bins)):
            raise ValueError('Dimensions must be positive integers')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive')
        if not math.isfinite(min_effective_mass) or min_effective_mass < 0:
            raise ValueError('Minimum soft cluster mass must be finite and nonnegative')
        self.bank = morphology
        self.rna_dim, self.projection_dim, self.bins = rna_dim, projection_dim, bins
        self.temperature, self.alpha, self.seed = float(temperature), float(alpha), int(seed)
        self.min_effective_mass = float(min_effective_mass)
        self.k, self.d = morphology.clusters, morphology.latent_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.path_projection = nn.Sequential(nn.Linear(self.d, projection_dim), nn.Tanh()).double()
            self.rna_projection = nn.Sequential(nn.Linear(rna_dim, projection_dim), nn.Tanh()).double()
            # RNA, predictable, residual, CLR, availability, projected path,
            # projected RNA, and explicit residual-by-RNA interaction.
            width = rna_dim + 2*self.k*self.d + 2*self.k + self.k*projection_dim + projection_dim + self.k*self.d
            self.head = nn.Linear(width, bins).double()
            self.interaction_rna = nn.Linear(rna_dim, self.d, bias=False).double()
        # Fix this random projection before computing feature normalization.
        # Otherwise training changes the very feature being normalized by a
        # scale estimated under an obsolete projection. Only the subsequent
        # linear survival head learns interaction coefficients.
        for parameter in self.interaction_rna.parameters():
            parameter.requires_grad_(False)
        self.probes = nn.ModuleDict()
        self.register_buffer('feature_mean', torch.zeros(width, dtype=torch.float64))
        self.register_buffer('feature_scale', torch.ones(width, dtype=torch.float64))
        self.train_ids = ()
        self.cached = None
        self.phase = 'new'
        self.ssl_history = []

    def _input(self, bags, rna, ids):
        bags = list(bags)
        identities = _ids(ids, len(bags))
        for identity, bag in zip(identities, bags):
            if hasattr(bag, 'patient_id') and str(bag.patient_id) != identity:
                raise ValueError('Patient ID differs from the supplied bag identity')
        rna = _matrix(rna, len(bags), self.rna_dim, 'RNA')
        encoded = self.bank.encode(bags)
        mass = encoded['composition']*encoded['real_patches'].unsqueeze(1)
        encoded['available'] &= mass >= self.min_effective_mass
        return identities, encoded, rna

    def pretrain(self, bags, rna, ids, *, epochs=100, lr=.005, cross_weight=1.,
                 path_noise=.03, rna_noise=.03, mismatch=False):
        """Fit SSL projections using only unlabeled outer-training patients.

        Two views are additive feature perturbations after frozen cluster
        pooling; these are not augmented original WSI image tiles.
        """
        if self.phase != 'new':
            raise RuntimeError('Use a fresh model for each SSL fit')
        if type(epochs) is not int or epochs < 1 or lr <= 0 or cross_weight < 0:
            raise ValueError('Invalid SSL configuration')
        if min(path_noise, rna_noise) < 0 or not all(math.isfinite(x) for x in (lr, cross_weight, path_noise, rna_noise)):
            raise ValueError('Invalid SSL parameters')
        identities, enc, rna = self._input(bags, rna, ids)
        bank_ids = self.bank.metadata.get('fit_patient_ids', [])
        if bank_ids and not set(map(str, bank_ids)).issubset(identities):
            raise ValueError('Morphology bank was fitted using patients outside the SSL training set')
        p, valid = enc['pooled'], enc['available']
        if len(identities) < 2 or not any(int(valid[:, j].sum()) >= 2 for j in range(self.k)):
            raise ValueError('SSL requires at least two patients in a valid cluster')
        generator = torch.Generator().manual_seed(self.seed)
        optimizer = torch.optim.Adam(list(self.path_projection.parameters()) + list(self.rna_projection.parameters()), lr=lr)
        # A fixed permutation preserves every individual RNA observation.
        permutation = torch.roll(torch.arange(len(rna)), 1) if mismatch else torch.arange(len(rna))
        history = []
        for _ in range(epochs):
            p1 = p + path_noise*torch.randn(p.shape, generator=generator, dtype=p.dtype)
            p2 = p + path_noise*torch.randn(p.shape, generator=generator, dtype=p.dtype)
            g1 = rna + rna_noise*torch.randn(rna.shape, generator=generator, dtype=rna.dtype)
            g2 = rna + rna_noise*torch.randn(rna.shape, generator=generator, dtype=rna.dtype)
            a, b = self.path_projection(p1), self.path_projection(p2)
            c, d = self.rna_projection(g1), self.rna_projection(g2)
            p_losses, cross_losses = [], []
            for j in range(self.k):
                selected = torch.where(valid[:, j])[0]
                if len(selected) < 2:
                    continue
                p_losses.append(patient_contrastive(a[selected, j], b[selected, j], self.temperature))
                # A permutation is used solely as a deliberate mismatched-pair
                # negative control; it is not a conditional MI estimator.
                cross_losses.append(patient_contrastive(a[selected, j], c[permutation[selected]], self.temperature))
            pp = torch.stack(p_losses).mean()
            gg = patient_contrastive(c, d, self.temperature)
            pg = torch.stack(cross_losses).mean()
            loss = pp + gg + cross_weight*pg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(q.grad is None or not torch.isfinite(q.grad).all() for q in list(self.path_projection.parameters())+list(self.rna_projection.parameters())):
                raise FloatingPointError('Missing or nonfinite SSL gradient')
            optimizer.step()
            history.append({'total': float(loss.detach()), 'path': float(pp.detach()),
                            'rna': float(gg.detach()), 'cross': float(pg.detach())})
        for q in list(self.path_projection.parameters())+list(self.rna_projection.parameters()):
            q.requires_grad_(False)
        self.train_ids = identities
        self.ssl_history = history
        self.phase = 'ssl_frozen'
        return copy.deepcopy(history)

    def _features(self, enc, rna, c):
        valid = enc['available']
        mask = valid.unsqueeze(-1).to(rna.dtype)
        u = (enc['pooled']-c)*mask
        c = c*mask
        zp = self.path_projection(enc['pooled'])*mask
        zg = self.rna_projection(rna)
        # Product is the only explicit joint input to the otherwise linear head.
        product = u*self.interaction_rna(rna).unsqueeze(1)
        return torch.cat([rna, c.flatten(1), u.flatten(1), enc['clr'], valid.to(rna.dtype),
                          zp.flatten(1), zg, product.flatten(1)], dim=1)

    def fit_probes(self, bags, rna, ids, *, folds=4, min_patients=8):
        if self.phase != 'ssl_frozen':
            raise RuntimeError('Fit SSL before cross-fitted probes')
        identities, enc, rna = self._input(bags, rna, ids)
        if identities != self.train_ids:
            raise ValueError('Probe rows must match the SSL training patients in order')
        assignment = patient_fold_ids(identities, folds, self.seed)
        predictions = torch.zeros_like(enc['pooled'])
        for j in range(self.k):
            selected = torch.where(enc['available'][:, j])[0]
            counts = torch.bincount(assignment[selected], minlength=folds)
            if len(selected) < min_patients or (counts == 0).any() or (len(selected)-counts < 2).any():
                raise ValueError(f'Cluster {j} lacks cross-fit patients; merge/drop it explicitly before fitting')
            model = CrossFittedRBFProbe(alpha=self.alpha).fit(
                rna[selected], enc['pooled'][selected, j], [identities[i] for i in selected],
                folds=folds, seed=self.seed, fold_ids=assignment[selected])
            predictions[selected, j] = model.oof_predictions
            self.probes[str(j)] = model
        # Cache inputs, not a computation graph. All feature generators are fixed.
        self.cached = (enc, rna, predictions)
        with torch.no_grad():
            x = self._features(enc, rna, predictions)
            self.feature_mean.copy_(x.mean(0))
            self.feature_scale.copy_(x.std(0, unbiased=False).clamp_min(1e-5))
        self.phase = 'probes_frozen'
        return self

    def _logits(self, enc, rna, c):
        return self.head((self._features(enc, rna, c)-self.feature_mean)/self.feature_scale)

    def fit_survival(self, labels, censorship, ids, *, epochs=100, lr=.01, weight_decay=1e-3):
        if self.phase != 'probes_frozen' or tuple(map(str, ids)) != self.train_ids:
            raise ValueError('Fit probes first and preserve training patient order')
        if type(epochs) is not int or epochs < 1 or lr <= 0:
            raise ValueError('Invalid survival training parameters')
        labels, censorship = torch.as_tensor(labels), torch.as_tensor(censorship)
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=weight_decay)
        history = []
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = discrete_nll(self._logits(*self.cached), labels, censorship, alpha=0.)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.head.parameters()):
                raise FloatingPointError('Missing or nonfinite survival gradient')
            optimizer.step()
            history.append(float(loss.detach()))
        self.phase = 'ready'
        return history

    @torch.no_grad()
    def predict(self, bags, rna, ids):
        if self.phase != 'ready':
            raise RuntimeError('Train survival head first')
        identities, enc, rna = self._input(bags, rna, ids)
        if set(identities) & set(self.train_ids):
            raise ValueError('Training patients require cached OOF predictions, not inference probes')
        if set(identities) & set(map(str, self.bank.metadata.get('fit_patient_ids', []))):
            raise ValueError('Inference patients appeared in the morphology bank fit')
        c = torch.stack([self.probes[str(j)].predict(rna) for j in range(self.k)], dim=1)
        return self._logits(enc, rna, c)

    def checkpoint(self):
        """Return a complete inference snapshot; never load untrusted pickle files."""
        if self.phase != 'ready':
            raise RuntimeError('Only a trained model has an inference snapshot')
        return {'version': 'method3_1_v1',
                'config': {'rna_dim': self.rna_dim, 'projection_dim': self.projection_dim,
                           'bins': self.bins, 'temperature': self.temperature,
                           'alpha': self.alpha, 'seed': self.seed,
                           'min_effective_mass': self.min_effective_mass},
                'bank': {'centers': self.bank.centers.clone(), 'mean': self.bank.patch_mean.clone(),
                         'scale': self.bank.patch_scale.clone(), 'components': self.bank.components.clone(),
                         'temperature': self.bank.temperature, 'min_mass': self.bank.min_mass,
                         'zero_delta': self.bank.zero_delta, 'metadata': copy.deepcopy(self.bank.metadata)},
                'train_ids': self.train_ids, 'state': copy.deepcopy(self.state_dict())}

    @classmethod
    def from_checkpoint(cls, snapshot):
        if snapshot.get('version') != 'method3_1_v1':
            raise ValueError('Unsupported method 3.1 checkpoint')
        model = cls(FrozenMorphologyBank(**snapshot['bank']), **snapshot['config'])
        for j in range(model.k):
            model.probes[str(j)] = CrossFittedRBFProbe(alpha=model.alpha)
        model.load_state_dict(snapshot['state'], strict=True)
        model.train_ids = tuple(snapshot['train_ids'])
        for p in list(model.path_projection.parameters())+list(model.rna_projection.parameters()):
            p.requires_grad_(False)
        model.phase = 'ready'
        model.eval()
        return model
