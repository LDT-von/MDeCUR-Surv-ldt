"""Batch-preserving explicit-interaction evaluation for Method 3.1.

The optional minimum selection epoch makes an enabled-path diagnostic possible
without silently choosing an untrained epoch-zero candidate. A positive epoch
is not a guarantee of a nonzero effect on degenerate inputs. This module never
loads data and accepts only arrays supplied by its caller.
"""
import copy
import math

import torch

from cancer_morphid_model import discrete_nll
from method3_1_batched_head import BatchedHierarchicalSurvivalHead, _CandidateBatch, batched_discrete_nll
from method3_1_hierarchical_head import _BLOCKS, _blocks, _grid, _identities
from method3_1_regularized import training_hazard_logits


class ActiveHierarchicalSurvivalHead(BatchedHierarchicalSurvivalHead):
    """Identical candidate-batched optimizer with configurable epoch selection.

    ``require_interaction`` rejects an empty block or candidates that hard-mask
    it; it never fabricates a nonzero weight. The default settings are the
    original batched implementation, including epoch zero as a candidate.
    """
    def __init__(self, bins=4, interaction_mode='per_bin', *, min_select_epoch=0,
                 require_interaction=False):
        super().__init__(bins, interaction_mode=interaction_mode)
        if type(min_select_epoch) is not int or min_select_epoch < 0:
            raise ValueError('min_select_epoch must be a nonnegative integer')
        if type(require_interaction) is not bool:
            raise ValueError('require_interaction must be boolean')
        self.min_select_epoch = min_select_epoch
        self.require_interaction = require_interaction

    def fit(self, train_blocks, train_labels, train_censorship, train_ids,
            val_blocks, val_labels, val_censorship, val_ids, *, candidates,
            epochs=160, lr=.03, eval_every=5):
        if self.selection is not None:
            raise RuntimeError('Use a fresh head for each independent fit')
        grid = _grid(candidates)
        if (type(epochs) is not int or epochs < 1 or type(eval_every) is not int
                or eval_every < 1 or not math.isfinite(lr) or lr <= 0
                or self.min_select_epoch > epochs):
            raise ValueError('Invalid training budget or selection epoch')
        train = _blocks(train_blocks)
        dimensions = {name: train[name].shape[1] for name in _BLOCKS}
        validation = _blocks(val_blocks, dimensions=dimensions)
        if self.require_interaction and (not dimensions['interaction'] or any(row[2] is None for row in grid)):
            raise ValueError('All candidates require a nonempty, enabled interaction block')
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
        best, curves, last_gradients = [None]*len(grid), [[] for _ in grid], [None]*len(grid)
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
                last_gradients = [{name: (float(getattr(model, name).grad[i].norm())
                    if coefficient is not None and dimensions[name] > 0 else None)
                    for name, coefficient in zip(_BLOCKS, coefficients)}
                    for i, coefficients in enumerate(grid)]
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
                    if epoch >= self.min_select_epoch and (best[i] is None or value < best[i]['val_nll']):
                        best[i] = {'val_nll': value, 'train_nll': train_value, 'epoch': epoch,
                                   'penalties': coefficients, 'grad_norms': copy.deepcopy(last_gradients[i]),
                                   'weights': {name: getattr(model, name)[i].detach().clone() for name in _BLOCKS},
                                   'bias': model.bias[i].detach().clone()}
        if any(row is None for row in best):
            raise RuntimeError('No evaluation checkpoint satisfies min_select_epoch')
        winner = 0
        for i in range(1, len(best)):
            if best[i]['val_nll'] < best[winner]['val_nll']:
                winner = i
        selected = best[winner]
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
                          'bias_penalized': False, 'epoch_zero_included': self.min_select_epoch == 0,
                          'min_select_epoch': self.min_select_epoch,
                          'interaction_required': self.require_interaction,
                          'interaction_mode': self.interaction_mode}
        return copy.deepcopy(self.selection)

    def survival_interaction_gradient_norm(self, train_blocks, labels, censorship, train_ids):
        """Audit direct survival-NLL gradient, without any regularizer term."""
        if self.selection is None:
            raise RuntimeError('Fit the head before auditing its gradient')
        blocks = _blocks(train_blocks, dimensions=self.dimensions)
        ids = _identities(train_ids, len(blocks['clr']))
        if ids != self.train_ids:
            raise ValueError('Gradient audit rows must match training patient order')
        x = self._normalized(blocks)
        interaction = self.weights['interaction'].detach().clone().requires_grad_(True)
        logits = (x['clr'] @ self.weights['clr'].T + x['other'] @ self.weights['other'].T
                  + x['interaction'] @ interaction.T + self.bias)
        objective = discrete_nll(logits, labels, censorship, alpha=0.)
        gradient, = torch.autograd.grad(objective, interaction)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite survival-only interaction gradient')
        return float(gradient.norm())

    def checkpoint(self):
        state = super().checkpoint()
        state['active_head_config'] = {'min_select_epoch': self.min_select_epoch,
                                       'require_interaction': self.require_interaction}
        return state

    @classmethod
    def from_checkpoint(cls, snapshot):
        # Base inference format remains compatible with HierarchicalSurvivalHead.
        result = super().from_checkpoint(snapshot)
        config = snapshot.get('active_head_config', {})
        result.min_select_epoch = config.get('min_select_epoch', 0)
        result.require_interaction = config.get('require_interaction', False)
        return result
