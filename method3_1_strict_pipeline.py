"""Strict additive main effects for the method 3.1 interaction controls.

No RNA-conditioned prediction or residual enters the non-interaction block.
The frozen upstream encoders may have been trained using paired modalities;
the distinction here is a property of their fixed *inference functions*, not
a claim of statistically independent features or unsupervised identification.
"""
import torch

from method3_1_interaction_pipeline import Method31InteractionPipeline


class Method31StrictPipeline(Method31InteractionPipeline):
    """Patient-level fixed pathology-only/RNA-only main paths plus interaction.

    The CLR block, pathology pooled patches, availability, pathology SSL head
    and global pooled representation are functions of the pathology bag alone.
    The RNA block and its SSL head are functions of the RNA vector alone.
    Univariate tanh/squares are computed separately on standardized pathology
    and RNA, never on the RNA-conditional residual. Interaction modes retain
    their original 3.1 meanings and are the only mixed *input feature* block.
    """
    def __init__(self, mapper, *, mode='tanh_outer', interaction_mode='per_bin', expand_clr=True):
        # Enabled by default; validation may still select epoch 0 or a
        # coefficient grid entry that hard-disables the interaction.
        super().__init__(mapper, mode=mode, interaction_mode=interaction_mode)
        if not isinstance(expand_clr, bool):
            raise ValueError('expand_clr must be boolean')
        self.expand_clr = expand_clr
        self.clr_mean = self.clr_scale = None

    def _fit_moments(self):
        mapped = super()._fit_moments()
        if self.mapper.backbone.cached is None:
            raise RuntimeError('Training OOF cache is required')
        clr = self.mapper.backbone.cached[0]['clr'].detach().cpu().double()
        self.clr_mean = clr.mean(0)
        std = clr.std(0, unbiased=False)
        self.clr_scale = torch.where(std >= 1e-3, std, torch.ones_like(std))
        return mapped

    def _main_parts(self, mapped, enc, rna):
        b = self.mapper.backbone
        valid = enc['available'].to(rna.dtype)
        mask = valid.unsqueeze(-1)
        path = enc['pooled']*mask
        path_ssl = b.path_projection(enc['pooled'])*mask
        global_z = (mapped['global_pooled']-self.pooled_mean)/self.pooled_scale
        path_main = torch.cat((path.flatten(1), valid, path_ssl.flatten(1),
                               global_z, torch.tanh(global_z), global_z.square()), dim=1)
        rna_z = mapped['rna_standardized']
        rna_ssl = b.rna_projection(rna)
        rna_main = torch.cat((rna, rna_ssl, rna_z, torch.tanh(rna_z), rna_z.square()), dim=1)
        return path_main, rna_main

    def _blocks(self, mapped, enc, rna, c):
        if self.pooled_mean is None or self.pooled_scale is None:
            raise RuntimeError('Training-only pooled moments have not been fitted')
        path, gene = self._main_parts(mapped, enc, rna)
        other = torch.cat((path, gene), dim=1)
        if self.mode == 'none':
            interaction = other.new_empty((len(other), 0))
        elif self.mode == 'raw_outer':
            pooled_z = (mapped['global_pooled']-self.pooled_mean)/self.pooled_scale
            interaction = (pooled_z.unsqueeze(2)*mapped['rna_standardized'].unsqueeze(1)).flatten(1)
        else:
            interaction = mapped[self.mode]
        clr = enc['clr']
        if self.expand_clr:
            if self.clr_mean is None or self.clr_scale is None:
                raise RuntimeError('Training-only CLR moments have not been fitted')
            clr_z = (clr-self.clr_mean)/self.clr_scale
            clr = torch.cat((clr, torch.tanh(clr_z), clr_z.square()), dim=1)
        return {'clr': clr.detach().clone(), 'other': other.detach().clone(),
                'interaction': interaction.detach().clone()}

    def checkpoint(self):
        snapshot = super().checkpoint()
        snapshot['version'] = 'method3_1_strict_pipeline_v1'
        snapshot['expand_clr'] = self.expand_clr
        snapshot['clr_mean'] = self.clr_mean.clone()
        snapshot['clr_scale'] = self.clr_scale.clone()
        return snapshot

    @classmethod
    def from_checkpoint(cls, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get('version') != 'method3_1_strict_pipeline_v1':
            raise ValueError('Unsupported strict interaction pipeline snapshot')
        payload = dict(snapshot)
        payload['version'] = 'method3_1_interaction_pipeline_v1'
        parent = Method31InteractionPipeline.from_checkpoint(payload)
        model = cls(parent.mapper, mode=parent.mode, interaction_mode=parent.head.interaction_mode,
                    expand_clr=snapshot['expand_clr'])
        model.head = parent.head
        model.pooled_mean, model.pooled_scale = parent.pooled_mean, parent.pooled_scale
        model.clr_mean = snapshot['clr_mean'].clone()
        model.clr_scale = snapshot['clr_scale'].clone()
        return model
