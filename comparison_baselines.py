"""Unimodal and linear baselines for the five-fold BLCA comparison.

Two baselines are provided, both under the same protocol as the multimodal
arms: DSS in months, four train-fitted quantile bins, patientwise discrete NLL
or an equivalent closed-form fit, risk = -sum(S) and the shared
``parity_metrics`` call.

``RNAOnlyRidge``
    RNA-only ridge regression.  The historical diagnostic
    (``diagnose_rna_only_baseline.py``) fitted a ridge on the *observed
    follow-up bin* one-hot targets, which is not a censoring-aware survival
    model.  This baseline instead fits a linear hazard score under the shared
    patientwise discrete NLL, so the RNA-only arm is compared under exactly
    the same likelihood as the deep arms.  The ridge alpha is chosen on the
    training rows only.

``WSIOnlyMIL``
    WSI-only attention-MIL over the same fixed patch budget, with the same
    discrete-hazard head, trained under the same objective.  The RNA branch is
    removed rather than zero-filled.

Both baselines are fitted and selected strictly on the training rows of the
current fold.

``load_dct_reference``
    The archived DCT reference is **loaded, never re-run**.  Its per-fold
    numbers come from the reference implementation's own legacy validation
    folds and are aggregated with the same mean/std rule as every other arm.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cancer_parity import discrete_nll, discrete_survival

DCT_REFERENCE_ARCHIVE = Path(__file__).resolve().parent / 'reference' / 'dct_5fold_archive.json'


RIDGE_ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def _dual_ridge(train_x, train_y, target_x, alpha):
    """Closed-form ridge without forming a genes-by-genes matrix.

    Returns the fitted weights of shape ``[d, targets]`` and the predictions
    for ``target_x``.  The dual form is used because ``d`` (genes) is far
    larger than the number of training patients.
    """
    x, y, v = (np.asarray(a, dtype=np.float64) for a in (train_x, train_y, target_x))
    if (x.ndim != 2 or y.ndim != 2 or v.ndim != 2 or len(x) != len(y)
            or x.shape[1] != v.shape[1] or not len(x) or not len(v)
            or not all(np.isfinite(a).all() for a in (x, y, v))
            or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError('Finite paired matrices and a positive ridge alpha are required')
    solver = np.linalg.solve(x @ x.T + alpha*np.eye(len(x)), y)
    weights = x.T @ solver
    return weights, v @ weights


def _one_hot(labels, bins):
    return np.eye(bins, dtype=np.float64)[np.asarray(labels, dtype=np.int64)]


class RNAOnlyRidge:
    """RNA-only ridge baseline fitted under the shared discrete NLL.

    The model is a linear hazard score ``w^T x + b`` per bin, fitted with an
    L2 penalty, using only the training rows and labels.  The alpha sweep
    minimises the same patientwise discrete NLL that the deep arms optimise.
    """

    method_name = 'RNA-only-Ridge-DNLL'

    def __init__(self, bins=4, alphas=RIDGE_ALPHAS):
        self.bins = int(bins)
        self.alphas = tuple(alphas)
        self.coefficients = None
        self.intercept = None
        self.selected_alpha = None
        self.train_mean = None
        self.train_scale = None
        self.train_loss = None

    def fit(self, train_rna, train_labels, train_event):
        """Select the ridge alpha and fit the weights on the training rows."""
        x = np.asarray(train_rna, dtype=np.float64)
        y = np.asarray(train_labels, dtype=np.int64)
        e = np.asarray(train_event, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(y) or len(x) != len(e) or not len(x):
            raise ValueError('Training RNA, labels and events must align')
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite training RNA')
        mean, scale = x.mean(0), x.std(0)
        scale[scale < 1e-8] = 1.0
        x = (x-mean)/scale
        one_hot = _one_hot(y, self.bins)
        best = None
        for alpha in self.alphas:
            weights, _ = _dual_ridge(x, one_hot, x, alpha)
            intercept = (one_hot - x @ weights).mean(0)
            loss = self._nll(x @ weights + intercept, y, e)
            if best is None or loss < best[0]:
                best = (loss, alpha, weights, intercept)
        loss, alpha, weights, intercept = best
        self.coefficients, self.intercept = weights, intercept
        self.selected_alpha = alpha
        self.train_mean, self.train_scale = mean, scale
        self.train_loss = loss
        return self

    def _nll(self, logits, labels, event):
        logits = torch.as_tensor(np.asarray(logits, dtype=np.float64))
        censorship = torch.as_tensor(1-np.asarray(event, dtype=np.float64), dtype=torch.float64)
        return float(discrete_nll(logits, torch.as_tensor(np.asarray(labels, dtype=np.int64)),
                                  censorship, alpha=.5))

    def predict_logits(self, rna):
        """Discrete-hazard logits for the given RNA rows."""
        if self.coefficients is None:
            raise ValueError('The ridge baseline must be fitted before prediction')
        x = np.asarray(rna, dtype=np.float64)
        x = (x-self.train_mean)/self.train_scale
        return x @ self.coefficients + self.intercept

    def predict_survival(self, rna):
        logits = torch.as_tensor(self.predict_logits(rna))
        return discrete_survival(logits).numpy()

    def adaptation_metadata(self):
        return {'method': self.method_name, 'official_model_reproduction': False,
                'modalities': ['RNA'],
                'training_objective': 'patientwise discrete NLL (alpha=0.5) over a linear hazard score',
                'selection': 'ridge alpha chosen on the training rows by the same discrete NLL',
                'alphas': list(self.alphas),
                'selected_alpha': self.selected_alpha,
                'notes': ['RNA standardization uses training-row mean/std only.',
                          'The historical observed-follow-up-bin ridge is a different estimator and is not reused.']}


class WSIOnlyMIL(nn.Module):
    """WSI-only attention-MIL baseline under the shared discrete-hazard head.

    The RNA branch is removed rather than zero-filled, so the arm cannot leak
    RNA information through a constant input.
    """

    method_name = 'WSI-only-MIL-DNLL'

    def __init__(self, path_dim, bins=4, hidden=128, dropout=.1):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (path_dim, bins, hidden)):
            raise ValueError('Positive integer dimensions required')
        self.path_dim, self.bins = path_dim, bins
        self.patch_encoder = nn.Sequential(nn.Linear(path_dim, hidden), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(hidden, max(1, hidden//2)), nn.Tanh(),
                                       nn.Linear(max(1, hidden//2), 1))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, bins))

    def forward(self, patches):
        if isinstance(patches, torch.Tensor):
            if patches.ndim != 3 or patches.shape[2] != self.path_dim:
                raise ValueError('Patch tensor must be [B,N,path_dim]')
            bags = list(patches.unbind(0))
        elif isinstance(patches, (list, tuple)):
            bags = list(patches)
        else:
            raise ValueError('Provide a patch tensor or a list of bags')
        parameter = next(self.parameters())
        pooled = []
        for bag in bags:
            if (not isinstance(bag, torch.Tensor) or bag.ndim != 2 or len(bag) < 1
                    or bag.shape[1] != self.path_dim or not torch.isfinite(bag).all()):
                raise ValueError('Each patch bag must be a finite nonempty [N,path_dim] tensor')
            features = self.patch_encoder(bag.to(device=parameter.device, dtype=parameter.dtype))
            weights = self.attention(features).softmax(dim=0)
            pooled.append((weights*features).sum(0))
        return {'logits': self.head(torch.stack(pooled))}

    def auxiliary_losses(self, output):
        return {}

    def adaptation_metadata(self):
        return {'method': self.method_name, 'official_model_reproduction': False,
                'modalities': ['pathology'],
                'training_objective': 'patientwise discrete NLL (alpha=0.5)',
                'notes': ['The RNA branch is removed, not zero-filled.',
                          'Padded zero feature rows remain visible to MIL to match the fixed-budget legacy protocol.']}


def load_dct_reference(path=None):
    """Load the archived DCT five-fold reference numbers.

    The reference implementation is not re-run here.  Its numbers are read
    from a committed archive so that the summary can place every arm on the
    same table, and they are aggregated with the same mean/std rule.

    Returns a dict with the per-fold records and a provenance block.  The
    loader refuses an archive that does not cover exactly the five protocol
    folds, so a partial archive can never masquerade as a complete one.
    """
    from comparison_protocol import FOLDS

    target = Path(path) if path else DCT_REFERENCE_ARCHIVE
    if not target.is_file():
        raise FileNotFoundError(f'Missing DCT reference archive: {target}')
    payload = json.loads(target.read_text(encoding='utf-8'))
    folds = payload.get('folds')
    if not isinstance(folds, list) or not folds:
        raise ValueError('DCT reference archive has no folds')
    by_fold = {}
    for record in folds:
        fold = int(record['fold'])
        if fold in by_fold:
            raise ValueError(f'Duplicate fold {fold} in the DCT reference archive')
        by_fold[fold] = record
    if set(by_fold) != set(FOLDS):
        raise ValueError(f'DCT reference archive must cover folds {FOLDS}, got {sorted(by_fold)}')
    ordered = [by_fold[fold] for fold in FOLDS]
    return {'archive_id': payload.get('archive_id'),
            'archive_path': str(target.resolve()),
            'protocol': payload.get('protocol'),
            'source': payload.get('source'),
            'limits': payload.get('limits', []),
            'official_model_reproduction': True,
            're_run': False,
            'folds': ordered}
