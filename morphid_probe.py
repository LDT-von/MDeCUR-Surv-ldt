"""Frozen nonlinear conditional-mean probes for MorphID mechanism checks.

The reconstruction target is a frozen pathology representation, never a
survival label.  Low held-out R² is a statement about this predictor family;
it is not evidence of independence or of biologically unique information.
All numerical fitting uses float64 and closed-form kernel ridge regression.
"""
from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch import nn


PROBE_VERSION = 'morphid_frozen_rbf_crossfit_v1'


def _matrix(value, name, *, rows=None):
    result = torch.as_tensor(value).detach().to(device='cpu', dtype=torch.float64)
    if result.ndim != 2 or result.shape[1] < 1 or (rows is not None and len(result) != rows):
        raise ValueError(f'{name} must be a [patients,features] matrix with matching rows')
    if not torch.isfinite(result).all():
        raise ValueError(f'{name} must contain finite values')
    return result


def _targets(value, rows):
    result = torch.as_tensor(value).detach().to(device='cpu', dtype=torch.float64)
    if result.ndim < 1 or len(result) != rows or result[0:1].numel() < 1:
        raise ValueError('Targets must contain one nonempty representation per patient')
    if not torch.isfinite(result).all():
        raise ValueError('Targets must contain finite values')
    return result


def _validate_config(alpha, bandwidth, pca_dim, min_scale):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    if bandwidth is not None and (not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('bandwidth must be finite and positive when supplied')
    if pca_dim is not None and (type(pca_dim) is not int or pca_dim < 1):
        raise ValueError('pca_dim must be a positive integer or None')
    if not math.isfinite(min_scale) or min_scale <= 0:
        raise ValueError('min_scale must be finite and positive')


def _rbf(left, right, bandwidth):
    # torch.cdist avoids cancellation from subtracting large squared norms.
    distance = torch.cdist(left, right, compute_mode='donot_use_mm_for_euclid_dist').square()
    if not torch.isfinite(distance).all():
        raise FloatingPointError('Nonfinite kernel distances')
    return torch.exp(-distance/(2*bandwidth.square()))


class _ResizableBuffers(nn.Module):
    """Allow a fresh module to load fitted tensors of data-dependent shape."""

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name, current in self._buffers.items():
            saved = state_dict.get(prefix+name)
            if isinstance(saved, torch.Tensor) and current is not None:
                # Cholesky solves can return column-major coefficients. Keep
                # their layout so replay uses the same GEMM reduction order.
                self._buffers[name] = torch.empty_like(
                    saved, dtype=current.dtype, device=current.device,
                    memory_format=torch.preserve_format)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class FrozenRBFProbe(_ResizableBuffers):
    """Train-only preprocessing and closed-form RBF kernel ridge.

    The objective is mean squared reconstruction error plus ``alpha`` times
    squared RKHS norm; consequently the dual solve uses ``K+n*alpha*I``.
    No hyperparameter is selected using targets or held-out observations.
    An optional PCA cap is fitted separately on each probe's training rows.
    """

    def __init__(self, alpha=1e-3, bandwidth=None, pca_dim=None, min_scale=1e-8):
        super().__init__()
        _validate_config(alpha, bandwidth, pca_dim, min_scale)
        self.alpha, self.bandwidth, self.pca_dim, self.min_scale = (
            float(alpha), bandwidth, pca_dim, float(min_scale))
        for name in ('x_mean', 'x_scale', 'target_mean'):
            self.register_buffer(name, torch.empty(0, dtype=torch.float64))
        for name in ('pca_components', 'train_x', 'dual_coef'):
            self.register_buffer(name, torch.empty((0, 0), dtype=torch.float64))
        self.register_buffer('bandwidth_value', torch.tensor(1., dtype=torch.float64))
        self.register_buffer('fitted_flag', torch.tensor(False))
        self.target_shape = ()
        self._metadata = {}

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    @torch.no_grad()
    def fit(self, x, y):
        if bool(self.fitted_flag):
            raise RuntimeError('A frozen probe cannot be refitted; create a fresh instance')
        x = _matrix(x, 'RNA representations')
        if len(x) < 2:
            raise ValueError('At least two training patients are required')
        y = _targets(y, len(x))
        self.target_shape = tuple(y.shape[1:])
        flat_y = y.reshape(len(y), -1)
        mean, scale = x.mean(0), x.std(0, unbiased=False)
        scale = torch.where(scale >= self.min_scale, scale, torch.ones_like(scale))
        standardized = (x-mean)/scale
        if not torch.isfinite(standardized).all():
            raise FloatingPointError('RNA standardization produced nonfinite values')
        components = torch.empty((x.shape[1], 0), dtype=torch.float64)
        if self.pca_dim is not None:
            rank_cap = min(self.pca_dim, len(x)-1, x.shape[1])
            _, _, right = torch.linalg.svd(standardized, full_matrices=False)
            components = right[:rank_cap].T.contiguous()
            # Fix the arbitrary SVD column signs without changing its space.
            anchors = components.abs().argmax(0)
            signs = components[anchors, torch.arange(rank_cap)].sign()
            components *= torch.where(signs == 0, torch.ones_like(signs), signs)
            transformed = standardized @ components
        else:
            transformed = standardized
        if self.bandwidth is None:
            distances = torch.pdist(transformed)
            positive = distances[distances > self.min_scale]
            bandwidth = positive.median() if len(positive) else torch.tensor(1., dtype=torch.float64)
        else:
            bandwidth = torch.tensor(float(self.bandwidth), dtype=torch.float64)
        kernel = _rbf(transformed, transformed, bandwidth)
        regularizer = len(x)*self.alpha
        target_mean = flat_y.mean(0)
        system = kernel + regularizer*torch.eye(len(x), dtype=torch.float64)
        factor = torch.linalg.cholesky(system)
        coefficient = torch.cholesky_solve(flat_y-target_mean, factor)
        if not torch.isfinite(coefficient).all():
            raise FloatingPointError('Kernel ridge solve produced nonfinite coefficients')
        eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min(0.)
        self.x_mean, self.x_scale = mean, scale
        self.pca_components, self.train_x = components, transformed
        self.target_mean, self.dual_coef = target_mean, coefficient
        self.bandwidth_value = bandwidth
        self.fitted_flag.fill_(True)
        self._metadata = {
            'version': PROBE_VERSION, 'n_train': len(x), 'input_features': x.shape[1],
            'transformed_features': transformed.shape[1], 'target_shape': list(self.target_shape),
            'bandwidth': float(bandwidth), 'alpha': self.alpha, 'diagonal_regularizer': regularizer,
            'input_rank': int(torch.linalg.matrix_rank(transformed)),
            'kernel_rank': int(torch.linalg.matrix_rank(kernel)),
            'effective_degrees_of_freedom': float((eigenvalues/(eigenvalues+regularizer)).sum()),
            'preprocessing_fit': 'this probe training patients only',
            'survival_labels_used': False, 'independence_established': False,
        }
        self.eval()
        return self

    @torch.no_grad()
    def predict(self, x):
        if not bool(self.fitted_flag):
            raise RuntimeError('Probe has not been fitted')
        x = _matrix(x, 'RNA representations')
        if x.shape[1] != len(self.x_mean):
            raise ValueError('RNA feature dimensions differ from the fitted probe')
        device = self.train_x.device
        x = x.to(device)
        transformed = (x-self.x_mean)/self.x_scale
        if self.pca_components.shape[1]:
            transformed = transformed @ self.pca_components
        if not torch.isfinite(transformed).all():
            raise FloatingPointError('RNA transformation produced nonfinite values')
        result = _rbf(transformed, self.train_x, self.bandwidth_value) @ self.dual_coef + self.target_mean
        return result.reshape((len(x),)+self.target_shape)

    def forward(self, x):
        return self.predict(x)

    def get_extra_state(self):
        return {'version': PROBE_VERSION, 'alpha': self.alpha, 'bandwidth': self.bandwidth,
                'pca_dim': self.pca_dim, 'min_scale': self.min_scale,
                'target_shape': list(self.target_shape), 'metadata': self._metadata}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get('version') != PROBE_VERSION:
            raise ValueError('Unsupported frozen probe state version')
        _validate_config(state['alpha'], state['bandwidth'], state['pca_dim'], state['min_scale'])
        self.alpha, self.bandwidth, self.pca_dim, self.min_scale = (
            state['alpha'], state['bandwidth'], state['pca_dim'], state['min_scale'])
        self.target_shape = tuple(state['target_shape'])
        self._metadata = copy.deepcopy(state['metadata'])


def _patient_ids(ids, n):
    if isinstance(ids, (str, bytes)):
        raise ValueError('Patient IDs must be a sequence, not one string')
    try:
        supplied = tuple(ids)
        if any(value is None for value in supplied):
            raise ValueError('Patient IDs cannot be missing')
        values = tuple(str(value) for value in supplied)
    except TypeError as error:
        raise ValueError('One unique patient ID per row is required') from error
    if len(values) != n or len(set(values)) != n or any(not v or v != v.strip() for v in values):
        raise ValueError('One unique nonempty patient ID per row is required')
    return values


def patient_fold_ids(ids, folds=4, seed=3):
    """Balanced patient folds stable under row permutation and Python hashing."""
    if type(folds) is not int or folds < 2 or type(seed) is not int:
        raise ValueError('At least two folds and an integer seed are required')
    values = _patient_ids(ids, len(ids))
    if len(values) < folds:
        raise ValueError('There must be at least one patient per fold')
    ordered = sorted(range(len(values)), key=lambda i: hashlib.sha256(
        f'{seed}\x00{values[i]}'.encode('utf-8')).digest())
    result = torch.empty(len(values), dtype=torch.long)
    for number, index in enumerate(ordered):
        result[index] = number % folds
    return result


class CrossFittedRBFProbe(_ResizableBuffers):
    """Patient cross-fitting with no own-target use in training residuals.

    ``oof_predictions`` uses one held-out model per training patient.
    ``predict`` averages all fold models for new observations. These are
    deliberately different estimators: their finite-sample distribution shift
    must not be described as exact train/inference equivalence.
    """

    def __init__(self, alpha=1e-3, bandwidth=None, pca_dim=None, min_scale=1e-8):
        super().__init__()
        _validate_config(alpha, bandwidth, pca_dim, min_scale)
        self.config = dict(alpha=float(alpha), bandwidth=bandwidth, pca_dim=pca_dim,
                           min_scale=float(min_scale))
        self.models = nn.ModuleList()
        self.register_buffer('oof_predictions', torch.empty(0, dtype=torch.float64))
        self.register_buffer('fold_ids', torch.empty(0, dtype=torch.long))
        self.register_buffer('fitted_flag', torch.tensor(False))
        self.ids = ()
        self.split_audit = []
        self.seed = 3

    @property
    def metadata(self):
        return {'version': PROBE_VERSION, 'n_patients': len(self.ids), 'folds': len(self.models),
                'seed': self.seed, 'survival_labels_used': False, 'independence_established': False,
                'training_estimator': 'held-out fold predictions',
                'inference_estimator': 'mean fold-model ensemble',
                'oof_ensemble_shift': 'Finite-sample distributions can differ; no exact parity claim.',
                'probe_config': copy.deepcopy(self.config),
                'models': [model.metadata for model in self.models]}

    @torch.no_grad()
    def fit(self, x, y, ids, folds=4, seed=3, fold_ids=None):
        if bool(self.fitted_flag):
            raise RuntimeError('A frozen cross-fitted probe cannot be refitted')
        x = _matrix(x, 'RNA representations')
        y = _targets(y, len(x))
        values = _patient_ids(ids, len(x))
        if type(folds) is not int or folds < 2 or type(seed) is not int:
            raise ValueError('At least two folds and an integer seed are required')
        if fold_ids is None:
            assignments = patient_fold_ids(values, folds, seed)
        else:
            raw = torch.as_tensor(fold_ids).detach().cpu()
            if (raw.ndim != 1 or len(raw) != len(x) or raw.dtype == torch.bool
                    or raw.is_complex() or not torch.isfinite(raw).all()
                    or not torch.equal(raw, raw.long().to(raw.dtype))):
                raise ValueError('fold_ids must contain one integer per patient')
            assignments = raw.long()
        if set(assignments.tolist()) != set(range(folds)):
            raise ValueError('fold_ids must cover every configured fold exactly')
        if any(int((assignments != f).sum()) < 2 for f in range(folds)):
            raise ValueError('Every fold must retain at least two training patients')
        models, audits = [], []
        predictions = torch.empty_like(y)
        for fold in range(folds):
            training = torch.where(assignments != fold)[0]
            held_out = torch.where(assignments == fold)[0]
            model = FrozenRBFProbe(**self.config).fit(x[training], y[training])
            predictions[held_out] = model.predict(x[held_out])
            models.append(model)
            audits.append({'fold': fold, 'train_indices': training.tolist(),
                           'validation_indices': held_out.tolist(),
                           'train_ids': [values[i] for i in training.tolist()],
                           'validation_ids': [values[i] for i in held_out.tolist()]})
        self.models = nn.ModuleList(models)
        self.oof_predictions, self.fold_ids = predictions, assignments
        self.ids, self.split_audit, self.seed = values, audits, seed
        self.fitted_flag.fill_(True)
        self.eval()
        return self

    @torch.no_grad()
    def predict(self, x):
        if not bool(self.fitted_flag):
            raise RuntimeError('Cross-fitted probe has not been fitted')
        return torch.stack([model.predict(x) for model in self.models]).mean(0)

    def forward(self, x):
        return self.predict(x)

    def get_extra_state(self):
        return {'version': PROBE_VERSION, 'config': self.config, 'ids': list(self.ids),
                'split_audit': self.split_audit, 'seed': self.seed, 'n_models': len(self.models)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get('version') != PROBE_VERSION:
            raise ValueError('Unsupported cross-fitted probe state version')
        _validate_config(**state['config'])
        self.config = copy.deepcopy(state['config'])
        self.ids, self.seed = tuple(state['ids']), state['seed']
        self.split_audit = copy.deepcopy(state['split_audit'])

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        extra = state_dict.get(prefix+'_extra_state', {})
        count = extra.get('n_models', 0) if isinstance(extra, dict) else 0
        if type(count) is not int or count < 0:
            raise ValueError('Invalid saved cross-fit model count')
        self.models = nn.ModuleList([FrozenRBFProbe() for _ in range(count)])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


def r2_score(y, prediction, reference_mean=None):
    """Patient-axis R² audit with negative scores and undefined outputs exposed.

    Constant evaluation targets are marked undefined (None), even if a supplied
    training mean differs from that constant. This prevents a degenerate target
    from appearing to establish predictive information.
    """
    y = torch.as_tensor(y).detach().cpu().double()
    prediction = torch.as_tensor(prediction).detach().cpu().double()
    if (y.ndim < 1 or not len(y) or y.shape != prediction.shape or y[0:1].numel() < 1
            or not torch.isfinite(y).all() or not torch.isfinite(prediction).all()):
        raise ValueError('Finite matching patient-axis targets and predictions required')
    flat, estimates = y.reshape(len(y), -1), prediction.reshape(len(y), -1)
    evaluation_mean = flat.mean(0)
    if reference_mean is None:
        baseline, kind = evaluation_mean, 'evaluation_mean'
    else:
        baseline = torch.as_tensor(reference_mean).detach().cpu().double().reshape(-1)
        if baseline.numel() not in (1, flat.shape[1]) or not torch.isfinite(baseline).all():
            raise ValueError('reference_mean must match the target representation')
        kind = 'supplied_reference_mean'
    variability = (flat-evaluation_mean).square().sum(0)
    denominator = (flat-baseline).square().sum(0)
    if not torch.isfinite(variability).all() or not torch.isfinite(denominator).all():
        raise FloatingPointError('R² sums overflowed float64')
    valid = (variability > 0) & (denominator > 0)
    scores = 1-(flat-estimates).square().sum(0)/denominator.clamp_min(torch.finfo(torch.float64).tiny)
    per_output = [float(value) if bool(ok) else None for value, ok in zip(scores, valid)]
    return {'per_output': per_output,
            'uniform_average': float(scores[valid].mean()) if bool(valid.any()) else None,
            'valid_outputs': valid.tolist(), 'n_patients': len(y),
            'target_shape': list(y.shape[1:]), 'baseline': kind,
            'interpretation': 'Given-probe-family mean-prediction audit; not an independence test.'}
