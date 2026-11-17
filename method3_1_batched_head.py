"""Batch independent survival-head candidates without changing their objective.

This CPU implementation is mathematically equivalent to running the candidate
loop in HierarchicalSurvivalHead: candidate losses are SUMMED, not averaged,
so Adam sees the same gradient for each candidate. No real data are loaded.
"""
import copy
import math

import torch
from torch import nn

from cancer_morphid_model import discrete_nll
from method3_1_regularized import training_hazard_logits
from method3_1_hierarchical_head import HierarchicalSurvivalHead, _BLOCKS, _blocks, _grid, _identities


def batched_discrete_nll(logits, labels, censorship, eps=1e-7):
    """Patientwise alpha=0 NLL for logits [candidates, patients, bins]."""
    if logits.ndim != 3 or min(logits.shape) < 1 or not logits.is_floating_point():
        raise ValueError('Batched logits must be [candidates,patients,bins]')
    if not math.isfinite(eps) or not 0 < eps < 1:
        raise ValueError('eps must be in (0,1)')
    # Reuse the production validator (also checks finite logits and outcomes).
    discrete_nll(logits[0], labels, censorship, alpha=0., eps=eps)
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite hazard logits')
    y = torch.as_tensor(labels, device=logits.device).reshape(-1).long()
    c = torch.as_tensor(censorship, device=logits.device).reshape(-1).to(logits.dtype)
    hazard = torch.sigmoid(logits)
    survival = torch.cumprod(1-hazard, dim=2)
    padded = torch.cat((torch.ones_like(survival[:, :, :1]), survival), dim=2)
    index = y[None, :, None].expand(logits.shape[0], -1, -1)
    previous = torch.gather(padded, 2, index).squeeze(2).clamp_min(eps)
    event_hazard = torch.gather(hazard, 2, index).squeeze(2).clamp_min(eps)
    current = torch.gather(padded, 2, index+1).squeeze(2).clamp_min(eps)
    loss = -(1-c)[None, :]*(previous.log()+event_hazard.log())-c[None, :]*current.log()
    return loss.mean(dim=1)


class _CandidateBatch(nn.Module):
    def __init__(self, dimensions, bins, mode, grid, bias):
        super().__init__()
        n = len(grid)
        self.clr = nn.Parameter(torch.zeros(n, bins, dimensions['clr'], dtype=torch.float64))
        self.other = nn.Parameter(torch.zeros(n, bins, dimensions['other'], dtype=torch.float64))
        self.interaction = nn.Parameter(torch.zeros(n, 1 if mode == 'shared' else bins,
                                                     dimensions['interaction'], dtype=torch.float64))
        self.bias = nn.Parameter(bias[None, :].expand(n, -1).clone())
        self.masks = {name: torch.tensor([row[i] is not None for row in grid], dtype=torch.float64)
                      for i, name in enumerate(_BLOCKS)}
        self.coefficients = {name: torch.tensor([0. if row[i] is None else row[i] for row in grid],
                                                dtype=torch.float64)
                             for i, name in enumerate(_BLOCKS)}

    def forward(self, blocks):
        total = self.bias[:, None, :]
        for name in _BLOCKS:
            w = getattr(self, name)*self.masks[name][:, None, None]
            total = total + torch.einsum('nd,ctd->cnt', blocks[name], w)
        return total

    def penalty(self):
        total = self.bias.new_zeros(len(self.bias))
        for name in _BLOCKS:
            w = getattr(self, name)
            total = total + .5*self.coefficients[name]*w.square().sum(dim=(1, 2))
        return total


class BatchedHierarchicalSurvivalHead(HierarchicalSurvivalHead):
    """Same fit/predict/checkpoint interface, with candidate axes batched."""
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
        initial_bias = training_hazard_logits(train_y, train_c, self.bins)
        discrete_nll(torch.zeros(len(train_ids), self.bins, dtype=torch.float64), train_y, train_c, alpha=0.)
        discrete_nll(torch.zeros(len(val_ids), self.bins, dtype=torch.float64), val_y, val_c, alpha=0.)
        self.mean = {name: train[name].mean(0) for name in _BLOCKS}
        self.scale = {}
        for name in _BLOCKS:
            std = (train[name].std(0, unbiased=False) if dimensions[name]
                   else torch.empty(0, dtype=torch.float64))
            self.scale[name] = torch.where(std >= 1e-3, std, torch.ones_like(std))
        train, validation = self._normalized(train), self._normalized(validation)
        model = _CandidateBatch(dimensions, self.bins, self.interaction_mode, grid, initial_bias)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best = [None]*len(grid)
        curves = [[] for _ in grid]
        last_gradients = [None]*len(grid)
        for epoch in range(epochs+1):
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                objectives = batched_discrete_nll(model(train), train_y, train_c) + model.penalty()
                if not torch.isfinite(objectives).all():
                    raise FloatingPointError('Nonfinite training objective')
                objectives.sum().backward()
                for p in model.parameters():
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Missing or nonfinite head gradient')
                last_gradients = []
                for i, coefficients in enumerate(grid):
                    last_gradients.append({name: (float(getattr(model, name).grad[i].norm())
                        if coefficient is not None and dimensions[name] > 0 else None)
                        for name, coefficient in zip(_BLOCKS, coefficients)})
                optimizer.step()
            if epoch == 0 or epoch % eval_every == 0 or epoch == epochs:
                with torch.no_grad():
                    val_losses = batched_discrete_nll(model(validation), val_y, val_c)
                    train_losses = batched_discrete_nll(model(train), train_y, train_c)
                for i, coefficients in enumerate(grid):
                    value, train_value = float(val_losses[i]), float(train_losses[i])
                    if not math.isfinite(value):
                        raise FloatingPointError('Nonfinite validation NLL')
                    curves[i].append((epoch, value))
                    if best[i] is None or value < best[i]['val_nll']:
                        best[i] = {'val_nll': value, 'train_nll': train_value, 'epoch': epoch,
                                   'penalties': coefficients, 'grad_norms': copy.deepcopy(last_gradients[i]),
                                   'weights': {name: getattr(model, name)[i].detach().clone()
                                               for name in _BLOCKS},
                                   'bias': model.bias[i].detach().clone()}
        winner = 0
        for i in range(1, len(best)):
            if best[i]['val_nll'] < best[winner]['val_nll']:
                winner = i
        selected = best[winner]
        # Audit snapshots are independent of the selected inference checkpoint.
        self.candidate_states = [{'weights': {k: v.clone() for k, v in row['weights'].items()},
                                  'bias': row['bias'].clone()} for row in best]
        self.dimensions = dimensions
        self.train_ids, self.val_ids = train_ids, val_ids
        self.weights = {k: v.clone() for k, v in selected['weights'].items()}
        self.bias = selected['bias'].clone()
        summaries = []
        for i, candidate in enumerate(best):
            summary = {k: copy.deepcopy(candidate[k]) for k in
                       ('val_nll', 'train_nll', 'epoch', 'penalties', 'grad_norms')}
            summary['validation_curve'] = curves[i]
            summaries.append(summary)
        self.selection = {'val_nll': selected['val_nll'], 'train_nll': selected['train_nll'],
                          'epoch': selected['epoch'], 'penalties': tuple(selected['penalties']),
                          'grad_norms': copy.deepcopy(selected['grad_norms']), 'candidates': summaries,
                          'selection': 'minimum validation NLL; candidate order then earlier epoch breaks ties',
                          'train_objective': 'mean discrete NLL + sum(block penalty/2 * squared weights)',
                          'bias_penalized': False, 'epoch_zero_included': True,
                          'interaction_mode': self.interaction_mode}
        return copy.deepcopy(self.selection)
