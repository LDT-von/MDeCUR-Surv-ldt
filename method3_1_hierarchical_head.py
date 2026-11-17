"""Hierarchical, validation-selected survival head for method 3.1 research.

There are three explicit precomputed feature blocks: CLR composition, other
main effects, and cross-modal interaction features. This component consumes
no images, patient files or model-specific directories. It is deliberately
separate from feature fitting: callers must fit morphology, RNA preprocessing,
and interaction construction using training patients only.
"""
import copy
import math

import torch
from torch import nn

from cancer_morphid_model import discrete_nll
from method3_1_regularized import training_hazard_logits


_BLOCKS = ('clr', 'other', 'interaction')


def _identities(ids, n):
    if isinstance(ids, (str, bytes)):
        raise ValueError('Provide a distinct patient ID for every row')
    values = tuple(ids)
    if any(x is None for x in values):
        raise ValueError('Patient IDs cannot be missing')
    values = tuple(map(str, values))
    if len(values) != n or len(set(values)) != n or any(not x or x != x.strip() for x in values):
        raise ValueError('Provide a distinct nonempty patient ID for every row')
    return values


def _blocks(values, *, dimensions=None):
    if not isinstance(values, dict) or set(values) != set(_BLOCKS):
        raise ValueError('Supply exactly clr, other, and interaction blocks')
    out, n = {}, None
    for name in _BLOCKS:
        x = torch.as_tensor(values[name], dtype=torch.float64).detach().cpu()
        if x.ndim != 2 or x.shape[0] < 1 or not torch.isfinite(x).all():
            raise ValueError(f'{name} must be a finite nonempty-row matrix')
        if n is None:
            n = len(x)
        if len(x) != n or (dimensions is not None and x.shape[1] != dimensions[name]):
            raise ValueError('Block row counts and fitted feature dimensions must agree')
        out[name] = x.clone()
    if out['clr'].shape[1] < 1:
        raise ValueError('At least one CLR coordinate is required')
    return out


def _grid(candidates):
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError('Provide at least one candidate of three block penalties')
    result = []
    for entry in candidates:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise ValueError('Each candidate is (CLR, other, interaction) penalties')
        if entry[0] is None:
            raise ValueError('The CLR anchor cannot be hard-disabled')
        row = []
        for val in entry:
            if val is None:
                row.append(None)
            elif isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError('Penalties must be finite nonnegative floats or None')
            else:
                row.append(float(val))
        result.append(tuple(row))
    return tuple(result)


class _Candidate(nn.Module):
    def __init__(self, dimensions, bins, mode, penalties, bias):
        super().__init__()
        self.mode, self.penalties = mode, penalties
        self.clr = nn.Parameter(torch.zeros(bins, dimensions['clr'], dtype=torch.float64))
        self.other = nn.Parameter(torch.zeros(bins, dimensions['other'], dtype=torch.float64),
                                  requires_grad=penalties[1] is not None and dimensions['other'] > 0)
        self.interaction = nn.Parameter(torch.zeros(1 if mode == 'shared' else bins,
                                                     dimensions['interaction'], dtype=torch.float64),
                                        requires_grad=penalties[2] is not None and dimensions['interaction'] > 0)
        self.bias = nn.Parameter(bias.detach().clone())

    def forward(self, blocks):
        return (blocks['clr'] @ self.clr.T + blocks['other'] @ self.other.T
                + blocks['interaction'] @ self.interaction.T + self.bias)

    def penalty(self):
        loss = self.bias.new_zeros(())
        for name, coefficient in zip(_BLOCKS, self.penalties):
            if coefficient is not None:
                loss = loss + .5*coefficient*getattr(self, name).square().sum()
        return loss


class HierarchicalSurvivalHead:
    """Train-only scaling, risk-set intercept, block L2 and validation choice.

    ``interaction_mode='shared'`` uses one interaction coefficient vector
    shared across bins; ``'per_bin'`` gives each bin independent coefficients.
    A candidate is (lambda_clr, lambda_other, lambda_interaction), with None
    hard-disabling that block. All coefficients start at zero and each bias is
    initialized from training risk sets. Epoch zero is evaluated per candidate.

    Inputs are detached tensors: the interaction coefficients receive gradient
    but this head does not train the upstream feature construction. This is an
    inference-complete head checkpoint, not a full upstream model checkpoint.
    """
    def __init__(self, bins=4, interaction_mode='per_bin'):
        if type(bins) is not int or bins < 1 or interaction_mode not in ('per_bin', 'shared'):
            raise ValueError('Positive bins and interaction_mode per_bin/shared required')
        self.bins, self.interaction_mode = bins, interaction_mode
        self.dimensions, self.mean, self.scale = None, None, None
        self.train_ids, self.val_ids = (), ()
        self.weights, self.bias, self.selection = None, None, None

    def _normalized(self, blocks):
        return {name: (blocks[name]-self.mean[name])/self.scale[name] for name in _BLOCKS}

    def fit(self, train_blocks, train_labels, train_censorship, train_ids,
            val_blocks, val_labels, val_censorship, val_ids, *, candidates,
            epochs=160, lr=.03, eval_every=5):
        if self.selection is not None:
            raise RuntimeError('Use a fresh head for each independent fit')
        grid = _grid(candidates)
        if (type(epochs) is not int or epochs < 1 or type(eval_every) is not int
                or eval_every < 1 or not math.isfinite(lr) or lr <= 0):
            raise ValueError('Invalid training budget')
        train = _blocks(train_blocks)
        dimensions = {name: train[name].shape[1] for name in _BLOCKS}
        validation = _blocks(val_blocks, dimensions=dimensions)
        train_ids = _identities(train_ids, len(train['clr']))
        val_ids = _identities(val_ids, len(validation['clr']))
        if set(train_ids) & set(val_ids):
            raise ValueError('Training and validation patient IDs overlap')
        train_y, train_c = torch.as_tensor(train_labels), torch.as_tensor(train_censorship)
        val_y, val_c = torch.as_tensor(val_labels), torch.as_tensor(val_censorship)
        # Production NLL validates label shapes, valid bins and censorship.
        initial_bias = training_hazard_logits(train_y, train_c, self.bins)
        discrete_nll(torch.zeros(len(train_ids), self.bins, dtype=torch.float64), train_y, train_c, alpha=0.)
        discrete_nll(torch.zeros(len(val_ids), self.bins, dtype=torch.float64), val_y, val_c, alpha=0.)
        self.mean = {name: train[name].mean(0) for name in _BLOCKS}
        self.scale = {}
        for name in _BLOCKS:
            std = (train[name].std(0, unbiased=False) if dimensions[name]
                   else torch.empty(0, dtype=torch.float64))
            self.scale[name] = torch.where(std >= 1e-3, std, torch.ones_like(std))
        train = self._normalized(train)
        validation = self._normalized(validation)
        best, summaries = None, []
        for coefficients in grid:
            model = _Candidate(dimensions, self.bins, self.interaction_mode, coefficients, initial_bias)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
            local_best = None
            last_gradients = None
            # Epoch zero is a genuine option and protects the initial intercept
            # estimate from being forced to fit unsupported associations.
            for epoch in range(epochs+1):
                if epoch:
                    optimizer.zero_grad(set_to_none=True)
                    loss = discrete_nll(model(train), train_y, train_c, alpha=0.) + model.penalty()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training objective')
                    loss.backward()
                    for p in model.parameters():
                        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                            raise FloatingPointError('Missing or nonfinite head gradient')
                    last_gradients = {name: (None if getattr(model, name).grad is None else
                                             float(getattr(model, name).grad.norm())) for name in _BLOCKS}
                    optimizer.step()
                if epoch == 0 or epoch % eval_every == 0 or epoch == epochs:
                    with torch.no_grad():
                        value = float(discrete_nll(model(validation), val_y, val_c, alpha=0.))
                        training_value = float(discrete_nll(model(train), train_y, train_c, alpha=0.))
                    if not math.isfinite(value):
                        raise FloatingPointError('Nonfinite validation NLL')
                    if local_best is None or value < local_best['val_nll']:
                        local_best = {'val_nll': value, 'train_nll': training_value, 'epoch': epoch,
                                      'penalties': coefficients, 'grad_norms': copy.deepcopy(last_gradients),
                                      'weights': {name: getattr(model, name).detach().clone() for name in _BLOCKS},
                                      'bias': model.bias.detach().clone()}
            summaries.append({k: copy.deepcopy(local_best[k]) for k in
                              ('val_nll', 'train_nll', 'epoch', 'penalties', 'grad_norms')})
            if best is None or local_best['val_nll'] < best['val_nll']:
                best = local_best
        self.dimensions = dimensions
        self.train_ids, self.val_ids = train_ids, val_ids
        self.weights = {k: v.clone() for k, v in best['weights'].items()}
        self.bias = best['bias'].clone()
        self.selection = {'val_nll': best['val_nll'], 'train_nll': best['train_nll'],
                          'epoch': best['epoch'], 'penalties': tuple(best['penalties']),
                          'grad_norms': copy.deepcopy(best['grad_norms']), 'candidates': summaries,
                          'selection': 'minimum validation NLL; candidate order then earlier epoch breaks ties',
                          'train_objective': 'mean discrete NLL + sum(block penalty/2 * squared weights)',
                          'bias_penalized': False, 'epoch_zero_included': True,
                          'interaction_mode': self.interaction_mode}
        return copy.deepcopy(self.selection)

    @torch.no_grad()
    def predict(self, blocks, ids):
        if self.selection is None:
            raise RuntimeError('Fit the head before predicting')
        blocks = _blocks(blocks, dimensions=self.dimensions)
        ids = _identities(ids, len(blocks['clr']))
        if set(ids) & (set(self.train_ids) | set(self.val_ids)):
            raise ValueError('Evaluation IDs must be disjoint from training and validation IDs')
        x = self._normalized(blocks)
        return (x['clr'] @ self.weights['clr'].T + x['other'] @ self.weights['other'].T
                + x['interaction'] @ self.weights['interaction'].T + self.bias)

    def checkpoint(self):
        if self.selection is None:
            raise RuntimeError('Fit the head before saving')
        return {'version': 'method3_1_hierarchical_head_v1', 'bins': self.bins,
                'interaction_mode': self.interaction_mode, 'dimensions': copy.deepcopy(self.dimensions),
                'mean': {k: v.clone() for k, v in self.mean.items()},
                'scale': {k: v.clone() for k, v in self.scale.items()},
                'train_ids': self.train_ids, 'val_ids': self.val_ids,
                'weights': {k: v.clone() for k, v in self.weights.items()},
                'bias': self.bias.clone(), 'selection': copy.deepcopy(self.selection)}

    @classmethod
    def from_checkpoint(cls, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get('version') != 'method3_1_hierarchical_head_v1':
            raise ValueError('Unsupported hierarchical head checkpoint')
        model = cls(snapshot['bins'], snapshot['interaction_mode'])
        model.dimensions = copy.deepcopy(snapshot['dimensions'])
        model.mean = {k: v.clone() for k, v in snapshot['mean'].items()}
        model.scale = {k: v.clone() for k, v in snapshot['scale'].items()}
        model.weights = {k: v.clone() for k, v in snapshot['weights'].items()}
        model.bias = snapshot['bias'].clone()
        model.train_ids, model.val_ids = tuple(snapshot['train_ids']), tuple(snapshot['val_ids'])
        model.selection = copy.deepcopy(snapshot['selection'])
        return model
