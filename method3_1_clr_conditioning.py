"""Training-only CLR conditioning for method 3.1 composition comparisons.

This is an outcome-free mechanism candidate, not a proof of statistical
independence, unique information, or partial information decomposition.
The OOF probes are conditional on the caller's shared outer-training feature
basis, which itself must never be fitted using validation or test patients.
Checkpoint snapshots contain training patient IDs and kernel support inputs.
"""
import copy
import math

import torch

from method3_1_model import _ids
from morphid_probe import CrossFittedRBFProbe


def _finite_matrix(values, name, n=None, width=None):
    x = torch.as_tensor(values).detach().cpu().double()
    if x.ndim != 2 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError(f'{name} requires a finite, nonempty patient-by-feature matrix')
    if n is not None and len(x) != n:
        raise ValueError(f'{name} patient count does not match')
    if width is not None and x.shape[1] != width:
        raise ValueError(f'{name} feature count differs from the training fit')
    return x


def _moments(x):
    std = x.std(0, unbiased=False)
    return x.mean(0), torch.where(std >= 1e-3, std, torch.ones_like(std))


class CLRConditioner:
    """Nonlinear CLR-only basis and patient-cross-fitted feature residuals.

    The CLR basis is `[CLR, tanh(z), z^2]`, with `z` standardized using only
    the outer training CLR.  For a fixed alpha RBF KRR, other features are
    reconstructed from CLR; training residuals use OOF models and fresh rows
    use their ensemble average. No survival outcomes are accepted.  Explicit
    interaction features must be passed separately by the caller, unchanged.

    Feature IDs only enforce the identities supplied by the caller; they do
    not verify that the caller's upstream graph/SSL basis was fitted safely.
    """
    def __init__(self, *, alpha=.01, bandwidth=None, pca_dim=None):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('KRR regularization alpha must be finite and positive')
        self.config = {'alpha': float(alpha), 'bandwidth': bandwidth, 'pca_dim': pca_dim}
        self.probe = None
        self.clr_mean = self.clr_scale = None
        self.clr_dim = self.other_dim = None
        self.train_ids, self.val_ids = (), ()
        self._train_clr = self._train_residual = None
        self.metadata = None

    def _inputs(self, clr, other, ids):
        clr = _finite_matrix(clr, 'CLR', width=self.clr_dim)
        other = _finite_matrix(other, 'Other features', n=len(clr), width=self.other_dim)
        identities = _ids(ids, len(clr))
        return clr, other, identities

    def _expand_clr(self, clr):
        z = (clr-self.clr_mean)/self.clr_scale
        return torch.cat((clr, torch.tanh(z), z.square()), dim=1)

    def fit(self, train_clr, train_other, train_ids, *, folds=4, seed=3):
        if self.probe is not None:
            raise RuntimeError('Use a fresh conditioner for each training set')
        clr, other, ids = self._inputs(train_clr, train_other, train_ids)
        if type(folds) is not int or folds < 2 or type(seed) is not int:
            raise ValueError('At least two patient folds and an integer seed are required')
        clr_mean, clr_scale = _moments(clr)
        probe = CrossFittedRBFProbe(**self.config).fit(clr, other, ids, folds=folds, seed=seed)
        self.clr_dim, self.other_dim = clr.shape[1], other.shape[1]
        self.clr_mean, self.clr_scale = clr_mean, clr_scale
        self._train_clr = clr.clone()
        self._train_residual = (other-probe.oof_predictions).clone()
        self.train_ids = ids
        self.probe = probe
        self.metadata = {
            'folds': folds, 'seed': seed, 'survival_labels_used': False,
            'train_prediction': 'patient-level OOF RBF KRR conditional on shared outer-training feature basis',
            'fresh_prediction': 'mean of the training-fold model predictions',
            'normalization': 'CLR train mean/std; std below 1e-3 replaced by one',
            'clr_basis': 'original CLR, tanh(train-standardized CLR), square(train-standardized CLR)',
            'residual_proves_independence': False, 'pid_identified': False,
            'split_audit': copy.deepcopy(probe.split_audit),
        }
        return self

    def training_features(self, ids):
        if self.probe is None:
            raise RuntimeError('Fit conditioner first')
        if _ids(ids, len(self.train_ids)) != self.train_ids:
            raise ValueError('Training IDs must match the fitted order')
        return {'clr': self._expand_clr(self._train_clr).clone(),
                'other': self._train_residual.clone()}

    @torch.no_grad()
    def _fresh(self, clr, other, ids):
        if self.probe is None:
            raise RuntimeError('Fit conditioner first')
        clr, other, ids = self._inputs(clr, other, ids)
        if set(ids) & set(self.train_ids):
            raise ValueError('Fresh patients must be disjoint from training patients')
        prediction = self.probe.predict(clr)
        return ids, {'clr': self._expand_clr(clr).clone(), 'other': (other-prediction).clone()}

    def validation_features(self, clr, other, ids):
        ids, result = self._fresh(clr, other, ids)
        if self.val_ids and self.val_ids != ids:
            raise ValueError('Validation identities are already registered in a different order')
        self.val_ids = ids
        return result

    def test_features(self, clr, other, ids):
        if not self.val_ids:
            raise RuntimeError('Register validation patients before evaluating test patients')
        ids, result = self._fresh(clr, other, ids)
        if set(ids) & set(self.val_ids):
            raise ValueError('Test patients overlap validation patients')
        return result

    def checkpoint(self):
        """Return complete inference and OOF state; may contain sensitive inputs."""
        if self.probe is None:
            raise RuntimeError('Fit conditioner before saving')
        return {'version': 'method3_1_clr_conditioning_v1',
                'config': copy.deepcopy(self.config), 'probe_state': copy.deepcopy(self.probe.state_dict()),
                'clr_dim': self.clr_dim, 'other_dim': self.other_dim,
                'clr_mean': self.clr_mean.clone(), 'clr_scale': self.clr_scale.clone(),
                'train_ids': self.train_ids, 'val_ids': self.val_ids,
                'train_clr': self._train_clr.clone(), 'train_residual': self._train_residual.clone(),
                'metadata': copy.deepcopy(self.metadata)}

    @classmethod
    def from_checkpoint(cls, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get('version') != 'method3_1_clr_conditioning_v1':
            raise ValueError('Unsupported CLR conditioner checkpoint')
        obj = cls(**snapshot['config'])
        obj.probe = CrossFittedRBFProbe(**obj.config)
        obj.probe.load_state_dict(snapshot['probe_state'], strict=True)
        obj.clr_dim, obj.other_dim = snapshot['clr_dim'], snapshot['other_dim']
        obj.clr_mean, obj.clr_scale = snapshot['clr_mean'].clone(), snapshot['clr_scale'].clone()
        obj.train_ids, obj.val_ids = tuple(snapshot['train_ids']), tuple(snapshot['val_ids'])
        obj._train_clr, obj._train_residual = snapshot['train_clr'].clone(), snapshot['train_residual'].clone()
        obj.metadata = copy.deepcopy(snapshot['metadata'])
        if tuple(obj.probe.ids) != obj.train_ids:
            raise ValueError('Probe IDs differ from the conditioner training IDs')
        return obj
