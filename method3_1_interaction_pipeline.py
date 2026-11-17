"""End-to-end research wrapper for the method 3.1 interaction comparisons.

Upstream bank, SSL encoders and KRR probes are frozen before survival labels
are supplied.  Every arm receives identical CLR and main-effect blocks, plus
an optional, explicitly named interaction block. All feature transformations
and head normalization are fitted on training patients only.

Checkpoint snapshots retain KRR support inputs and OOF training features; they
can therefore contain sensitive data when used with non-synthetic inputs.
"""
import copy

import torch

from method3_1_hierarchical_head import HierarchicalSurvivalHead
from method3_1_interaction_features import GlobalResidualInteractions


def _stable_moments(x):
    x = torch.as_tensor(x).detach().cpu().double()
    if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
        raise ValueError('A finite, nonempty training matrix is required')
    mean, std = x.mean(0), x.std(0, unbiased=False)
    return mean, torch.where(std >= 1e-3, std, torch.ones_like(std))


class Method31InteractionPipeline:
    """Shared frozen 3.1 representation plus one fitted survival head.

    Feature modes:
      none       - no explicit interaction terms;
      diagonal   - prior masked cluster residual x fixed RNA projection;
      outer      - global RNA-prediction residual x RNA outer product;
      tanh_outer - outer product after separate tanh transformations;
      raw_outer  - global pathology mean x RNA (no KRR residualization).

    Main effects are identical in all modes: the legacy main features except
    CLR and its old diagonal interaction, plus z, tanh(z) and z^2 separately
    for global pathology, global residual and RNA. Their training-only moments
    are fixed before survival fitting. The nonlinear main effects are essential
    controls but do not ensure exact matching of interaction arm capacity.
    """
    MODES = ('none', 'diagonal', 'outer', 'tanh_outer', 'raw_outer')

    def __init__(self, mapper, *, mode='none', interaction_mode='per_bin'):
        if not isinstance(mapper, GlobalResidualInteractions) or mapper.probe is None:
            raise ValueError('A training-fitted global interaction mapper is required')
        if mode not in self.MODES:
            raise ValueError('Unsupported interaction feature mode')
        self.mapper, self.mode = mapper, mode
        self.head = HierarchicalSurvivalHead(mapper.backbone.bins, interaction_mode=interaction_mode)
        self.train_ids = tuple(mapper.train_ids)
        self.pooled_mean = self.pooled_scale = None
        self._training_blocks = None

    def _legacy(self, enc, rna, c):
        """Extract existing CLR and main blocks without the old interaction."""
        b = self.mapper.backbone
        old = b._features(enc, rna, c).detach()
        kd = b.k*b.d
        clr_start = b.rna_dim+2*kd
        clr_end = clr_start+b.k
        clr = old[:, clr_start:clr_end]
        main = torch.cat((old[:, :clr_start], old[:, clr_end:-kd]), dim=1)
        return clr, main

    def _blocks(self, mapped, enc, rna, c):
        if self.pooled_mean is None or self.pooled_scale is None:
            raise RuntimeError('Training-only pooled moments have not been fitted')
        clr, legacy_main = self._legacy(enc, rna, c)
        pooled_z = (mapped['global_pooled']-self.pooled_mean)/self.pooled_scale
        residual_z, rna_z = mapped['global_residual_standardized'], mapped['rna_standardized']
        nonlinear = []
        for z in (pooled_z, residual_z, rna_z):
            nonlinear.extend((z, torch.tanh(z), z.square()))
        other = torch.cat((legacy_main, *nonlinear), dim=1)
        if self.mode == 'none':
            interaction = other.new_empty((len(other), 0))
        elif self.mode == 'raw_outer':
            interaction = (pooled_z.unsqueeze(2)*rna_z.unsqueeze(1)).flatten(1)
        else:
            interaction = mapped[self.mode]
        return {'clr': clr.detach().clone(), 'other': other.detach().clone(),
                'interaction': interaction.detach().clone()}

    def _fit_moments(self):
        mapped = self.mapper.training_features()
        self.pooled_mean, self.pooled_scale = _stable_moments(mapped['global_pooled'])
        return mapped

    def _train_blocks(self, mapped):
        cached = self.mapper.backbone.cached
        if cached is None:
            raise RuntimeError('Training OOF backbone cache is unavailable')
        return self._blocks(mapped, *cached)

    @torch.no_grad()
    def _fresh_blocks(self, bags, rna, ids, *, validation):
        # The mapper validates IDs and train/validation/test exclusion first.
        if isinstance(ids, (str, bytes)):
            raise ValueError('Provide a sequence of patient IDs')
        bags, ids = list(bags), tuple(ids)
        mapped = (self.mapper.validation_features(bags, rna, ids) if validation
                  else self.mapper.test_features(bags, rna, ids))
        b = self.mapper.backbone
        _, enc, rna = b._input(bags, rna, ids)
        c = torch.stack([b.probes[str(j)].predict(rna) for j in range(b.k)], dim=1)
        return self._blocks(mapped, enc, rna, c)

    def fit(self, train_labels, train_censorship, train_ids,
            val_bags, val_rna, val_ids, val_labels, val_censorship, *,
            candidates, epochs=160, lr=.03, eval_every=5):
        """Fit and select a head; validation outcomes never update upstream."""
        if self.head.selection is not None:
            raise RuntimeError('Use a fresh pipeline for each independent fit')
        if isinstance(train_ids, (str, bytes)) or isinstance(val_ids, (str, bytes)):
            raise ValueError('Provide sequences of patient IDs')
        if tuple(map(str, train_ids)) != self.train_ids:
            raise ValueError('Training labels must follow mapper training IDs')
        val_ids = tuple(val_ids)
        mapped = self._fit_moments()
        train = self._train_blocks(mapped)
        validation = self._fresh_blocks(val_bags, val_rna, val_ids, validation=True)
        result = self.head.fit(train, train_labels, train_censorship, self.train_ids,
                               validation, val_labels, val_censorship, val_ids,
                               candidates=candidates, epochs=epochs, lr=lr, eval_every=eval_every)
        self._training_blocks = {k: v.clone() for k, v in train.items()}
        return copy.deepcopy(result)

    def training_blocks(self):
        """Return cached OOF training blocks for audits, if still in memory."""
        if self._training_blocks is None:
            raise RuntimeError('No in-memory OOF training blocks; fit the pipeline first')
        return {k: v.clone() for k, v in self._training_blocks.items()}

    @torch.no_grad()
    def predict(self, bags, rna, ids):
        if self.head.selection is None:
            raise RuntimeError('Fit the survival head before prediction')
        if isinstance(ids, (str, bytes)):
            raise ValueError('Provide a sequence of patient IDs')
        ids = tuple(ids)
        blocks = self._fresh_blocks(bags, rna, ids, validation=False)
        return self.head.predict(blocks, ids)

    def checkpoint(self):
        """Capture the upstream mapper, fitted head, modes and input moments."""
        if self.head.selection is None:
            raise RuntimeError('Fit the survival head before saving')
        return {'version': 'method3_1_interaction_pipeline_v1', 'mode': self.mode,
                'mapper': self.mapper.checkpoint(), 'head': self.head.checkpoint(),
                'pooled_mean': self.pooled_mean.clone(), 'pooled_scale': self.pooled_scale.clone()}

    @classmethod
    def from_checkpoint(cls, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get('version') != 'method3_1_interaction_pipeline_v1':
            raise ValueError('Unsupported method 3.1 interaction pipeline snapshot')
        mapper = GlobalResidualInteractions.from_checkpoint(snapshot['mapper'])
        head = HierarchicalSurvivalHead.from_checkpoint(snapshot['head'])
        if mapper.train_ids != head.train_ids or mapper.val_ids != head.val_ids:
            raise ValueError('Mapper and head patient splits differ')
        model = cls(mapper, mode=snapshot['mode'], interaction_mode=head.interaction_mode)
        model.head = head
        model.pooled_mean = snapshot['pooled_mean'].clone()
        model.pooled_scale = snapshot['pooled_scale'].clone()
        return model
