"""Condition the 3.1 main-effect block on training-only morphology CLR.

The strict unconditioned comparator has separate pathology/RNA main effects;
conditioning RNA features on pathology CLR can mix modalities, so this
candidate is *not* itself claimed to be a strict additive baseline or an
identification of independent/unique information. Upstream bank/SSL use a
shared outer-training basis: subsequent OOF probes are conditional on it.
"""
import copy

import torch

from method3_1_clr_conditioning import CLRConditioner
from method3_1_strict_pipeline import Method31StrictPipeline


class CLRConditionedStrictPipeline(Method31StrictPipeline):
    """Same 3.1 interaction block with CLR-predicted main effects removed.

    The outcome-free conditioner is fitted before the survival head. Its
    training residuals are patient-level OOF; fresh patients use a mean of
    fold predictors. The CLR block is the same training-standardized nonlinear
    basis as ``Method31StrictPipeline(expand_clr=True)``.
    """
    def __init__(self, mapper, *, mode='tanh_outer', interaction_mode='per_bin',
                 conditioner_alpha=.01):
        super().__init__(mapper, mode=mode, interaction_mode=interaction_mode, expand_clr=True)
        self.conditioner_alpha = conditioner_alpha
        self.conditioner = None

    def _apply_conditioner(self, blocks, ids, *, stage):
        if self.conditioner is None:
            raise RuntimeError('Fit the CLR conditioner before constructing residual features')
        base_clr = blocks['clr'][:, :self.mapper.backbone.k]
        if stage == 'train':
            conditional = self.conditioner.training_features(ids)
        elif stage == 'validation':
            conditional = self.conditioner.validation_features(base_clr, blocks['other'], ids)
        elif stage == 'test':
            conditional = self.conditioner.test_features(base_clr, blocks['other'], ids)
        else:
            raise ValueError('Unknown feature split')
        # A separate CLR transformation in the strict comparator must match
        # exactly: all arms use the same training means and nonlinear basis.
        if not torch.equal(conditional['clr'], blocks['clr']):
            raise RuntimeError('Conditioner CLR basis differs from strict training-only basis')
        return {'clr': conditional['clr'], 'other': conditional['other'],
                'interaction': blocks['interaction'].clone()}

    def fit(self, train_labels, train_censorship, train_ids,
            val_bags, val_rna, val_ids, val_labels, val_censorship, *,
            candidates, epochs=160, lr=.03, eval_every=5,
            folds=4, conditioner_seed=None):
        if self.head.selection is not None or self.conditioner is not None:
            raise RuntimeError('Use a fresh pipeline for each independent fit')
        train_ids, val_ids = tuple(train_ids), tuple(val_ids)
        if tuple(map(str, train_ids)) != self.train_ids:
            raise ValueError('Training labels must follow mapper training IDs')
        mapped = self._fit_moments()
        train = self._train_blocks(mapped)
        validation = self._fresh_blocks(val_bags, val_rna, val_ids, validation=True)
        k = self.mapper.backbone.k
        seed = self.mapper.backbone.seed if conditioner_seed is None else conditioner_seed
        self.conditioner = CLRConditioner(alpha=self.conditioner_alpha).fit(
            train['clr'][:, :k], train['other'], train_ids, folds=folds, seed=seed)
        train = self._apply_conditioner(train, train_ids, stage='train')
        validation = self._apply_conditioner(validation, val_ids, stage='validation')
        result = self.head.fit(train, train_labels, train_censorship, train_ids,
                               validation, val_labels, val_censorship, val_ids,
                               candidates=candidates, epochs=epochs, lr=lr, eval_every=eval_every)
        self._training_blocks = {k: v.clone() for k, v in train.items()}
        return copy.deepcopy(result)

    @torch.no_grad()
    def predict(self, bags, rna, ids):
        if self.head.selection is None:
            raise RuntimeError('Fit the survival head before prediction')
        ids = tuple(ids)
        blocks = self._fresh_blocks(bags, rna, ids, validation=False)
        blocks = self._apply_conditioner(blocks, ids, stage='test')
        return self.head.predict(blocks, ids)

    def checkpoint(self):
        if self.conditioner is None:
            raise RuntimeError('Fit the CLR conditioner before saving')
        snapshot = super().checkpoint()
        snapshot['version'] = 'method3_1_conditioned_pipeline_v1'
        snapshot['conditioner'] = self.conditioner.checkpoint()
        snapshot['conditioner_alpha'] = self.conditioner_alpha
        return snapshot

    @classmethod
    def from_checkpoint(cls, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get('version') != 'method3_1_conditioned_pipeline_v1':
            raise ValueError('Unsupported CLR-conditioned pipeline snapshot')
        strict_snapshot = dict(snapshot)
        strict_snapshot['version'] = 'method3_1_strict_pipeline_v1'
        parent = Method31StrictPipeline.from_checkpoint(strict_snapshot)
        conditioner = CLRConditioner.from_checkpoint(snapshot['conditioner'])
        if (conditioner.train_ids != parent.mapper.train_ids or
                conditioner.val_ids != parent.mapper.val_ids or
                conditioner.train_ids != parent.head.train_ids or
                conditioner.val_ids != parent.head.val_ids):
            raise ValueError('Conditioner, mapper and head patient splits differ')
        model = cls(parent.mapper, mode=parent.mode, interaction_mode=parent.head.interaction_mode,
                    conditioner_alpha=snapshot['conditioner_alpha'])
        model.head = parent.head
        model.pooled_mean, model.pooled_scale = parent.pooled_mean, parent.pooled_scale
        model.clr_mean, model.clr_scale = parent.clr_mean, parent.clr_scale
        model.conditioner = conditioner
        return model
