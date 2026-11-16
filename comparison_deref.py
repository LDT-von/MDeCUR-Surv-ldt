"""DeReF adaptation to the unified BLCA DSS parity protocol.

Upstream source: https://github.com/ZJUMAI/DeReF at
6bdbf81e76d1c7fa1d49b05f1a8fec2fbfc7b6d3.

The upstream innovation modules are executed, not reimplemented:

* ``Transformer_P`` / ``Transformer_G`` (Nyström attention encoders),
* ``Private_encoder`` (molecular and pathology private features),
* ``Share_encoder`` (common and explored/red shared features),
* ``Gate`` and ``expert`` (dynamic mixture of experts),
* the upstream ``nll_loss`` and the upstream feature-distance MSE objective.

Runtime adaptations, none of which modifies the checkout on disk:

1. ``Transformer_P``/``Transformer_G`` expand the class token with an
   unconditional ``.cuda()``; the class token is moved to the input device
   instead.
2. The upstream genomics branch stacks the six omic groups and then inserts
   an extra leading axis, which yields a rank-4 tensor that the upstream
   transformer cannot consume.  The adapter feeds the upstream encoders the
   ``[1, 6, D]`` tensor the upstream code intends (groups on the token axis,
   one patient per call).
3. The upstream network is written for batch size 1 (its own default is
   ``--batch_size 1``); the adapter therefore calls the upstream network once
   per patient and stacks the results.
4. The upstream path branch expects the raw 1024-d UNI v1 feature vector, so
   the adapter requires ``path_dim == 1024`` and refuses any other width
   instead of silently projecting it.

This is ``DeReF-2M-P4096-DNLL``: an adaptation, not an original-paper
reproduction.
"""
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

import comparison_sources as sources


METHOD_NAME = 'DeReF-2M-P4096-DNLL'
UPSTREAM_URL = 'https://github.com/ZJUMAI/DeReF'
UPSTREAM_COMMIT = '6bdbf81e76d1c7fa1d49b05f1a8fec2fbfc7b6d3'
NUM_OMIC_GROUPS = 6
UNI_FEATURE_DIM = 1024

# Upstream default objective: survival NLL + alpha * feature-distance MSE.
DEFAULT_AUXILIARY_WEIGHTS = {'feature_mse': 1.0}


def _device_neutral_transformer_p(self, features):
    """Device-neutral copy of the upstream ``Transformer_P.forward``."""
    import numpy as np
    H = features.shape[1]
    _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    add_length = _H * _W - H
    h = torch.cat([features, features[:, :add_length, :]], dim=1)
    B = h.shape[0]
    cls_tokens = self.cls_token.expand(B, -1, -1).to(device=features.device)
    h = torch.cat((cls_tokens, h), dim=1)
    h = self.layer1(h)
    h = self.pos_layer(h, _H, _W)
    h = self.layer2(h)
    h = self.norm(h)
    return h[:, 0], h[:, 1:]


def _device_neutral_transformer_g(self, features):
    """Device-neutral copy of the upstream ``Transformer_G.forward``."""
    cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).to(device=features.device)
    h = torch.cat((cls_tokens, features), dim=1)
    h = self.layer1(h)
    h = self.layer2(h)
    h = self.norm(h)
    return h[:, 0], h[:, 1:]


class DeReF2M(nn.Module):
    """Two-modality DeReF adapter that executes the upstream network.

    The upstream ``DeReF`` module is instantiated with the six upstream
    omic-group sizes and the upstream encoders, gates and experts are used
    unchanged.
    """

    method_name = METHOD_NAME
    auxiliary_weights = DEFAULT_AUXILIARY_WEIGHTS

    def __init__(self, path_dim, rna_dim, omic_indices, bins=4, model_size='small',
                 fusion='concat', seed=3):
        super().__init__()
        if not omic_indices or len(omic_indices) != NUM_OMIC_GROUPS:
            raise ValueError(f'DeReF upstream requires exactly {NUM_OMIC_GROUPS} omic groups')
        if any(not group for group in omic_indices):
            raise ValueError('Every DeReF omic group must contain matched genes')
        if max(max(group) for group in omic_indices) >= rna_dim:
            raise ValueError('Omic group gene index outside the RNA dimension')
        if any(type(x) is not int or x < 1 for x in (rna_dim, bins)):
            raise ValueError('Positive integer dimensions required')
        self.path_dim, self.rna_dim, self.bins = path_dim, rna_dim, bins
        self.omic_indices = [list(map(int, group)) for group in omic_indices]
        self.omic_sizes = [len(group) for group in self.omic_indices]
        self.seed = seed

        modules = sources.load_deref_modules()
        self.upstream = modules['models.deref.network']
        # Runtime device patches of the two upstream transformers.
        self.upstream.Transformer_P.forward = _device_neutral_transformer_p
        self.upstream.Transformer_G.forward = _device_neutral_transformer_g

        self.model = self.upstream.DeReF(omic_sizes=self.omic_sizes, n_classes=bins,
                                         fusion=fusion, model_size=model_size)
        if path_dim != UNI_FEATURE_DIM:
            # The upstream pathomics input width is the raw UNI v1 width.  Only
            # the input width of the first linear layer is changed; every
            # upstream layer after it is untouched.
            self.model.pathomics_fc[0] = nn.Linear(path_dim, 256)

    # ------------------------------------------------------------------ input
    def _omic_inputs(self, rna):
        if (not isinstance(rna, torch.Tensor) or rna.ndim != 2 or len(rna) < 1
                or rna.shape[1] != self.rna_dim or not rna.is_floating_point()
                or not torch.isfinite(rna).all()):
            raise ValueError('RNA must be a finite nonempty [B,rna_dim] tensor')
        return [rna[:, group] for group in self.omic_indices]

    def forward(self, patches, rna):
        """Return discrete-hazard logits plus the upstream feature list.

        The upstream network is executed once per patient because the upstream
        implementation is written for batch size one.
        """
        if isinstance(patches, torch.Tensor):
            if patches.ndim != 3 or patches.shape[0] != len(rna) or patches.shape[2] != self.path_dim:
                raise ValueError('Patch tensor must be [B,N,path_dim]')
            bags = list(patches.unbind(0))
        elif isinstance(patches, (list, tuple)) and len(patches) == len(rna):
            bags = list(patches)
        else:
            raise ValueError('Provide one patch bag per RNA row')
        for bag in bags:
            if (not isinstance(bag, torch.Tensor) or bag.ndim != 2 or len(bag) < 1
                    or bag.shape[1] != self.path_dim or not bag.is_floating_point()
                    or not torch.isfinite(bag).all()):
                raise ValueError('Each patch bag must be a finite nonempty [N,path_dim] tensor')
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        rna = rna.to(device=device, dtype=dtype)
        # Per patient, the six upstream omic groups as 1-D gene vectors.
        omic_by_patient = [[rna[patient][group] for group in self.omic_indices]
                           for patient in range(len(rna))]

        logits_list, features_list, scores_list = [], [], []
        for index, bag in enumerate(bags):
            # The upstream pathomics branch treats the patch bag as a token
            # matrix: pathomics_fc maps [N, 1024] -> [N, 256] and the upstream
            # unsqueeze(0) restores the leading patient axis.
            x_path = bag.to(device=device, dtype=dtype)  # [N, 1024]
            genomics_features = [
                self.model.genomics_fc[group].forward(omic_by_patient[index][group])
                for group in range(NUM_OMIC_GROUPS)]
            # Upstream intent: one token per omic group, one patient per call.
            genomics_features = torch.stack(genomics_features).unsqueeze(0)  # [1, 6, D]
            cls_path, patch_path = self.model.pathomics_encoder(
                self.model.pathomics_fc(x_path).unsqueeze(0))  # [1, N, 256]
            cls_omic, patch_omic = self.model.genomics_encoder(genomics_features)

            feature_g = self.model.m_private(cls_omic)
            feature_p = self.model.p_private(cls_path)
            feature_comm = self.model.com_encoder(cls_omic, cls_path)
            feature_red = self.model.red_encoder(cls_omic, cls_path)

            cat_total = torch.concat([feature_g, feature_p, feature_comm, feature_red], dim=0)
            seg_l = random.choice(self.model.seg_length)
            cat_new = cat_total.reshape(4, seg_l, 128 // seg_l).transpose(1, 0).reshape(1, -1)

            out1 = self.model.exp1(cat_new)
            out2 = self.model.exp2(cat_new)
            out3 = self.model.exp3(cat_new)
            out4 = self.model.exp4(cat_new)
            out_total = torch.concat([out1, out2, out3, out4], dim=0)
            score = self.model.gate(cat_total)
            out_total = out_total * score
            logits = self.model.classifier(out_total.view(1, -1))
            logits_list.append(logits)
            features_list.append([feature_g, feature_p, feature_comm, feature_red])
            scores_list.append(score)

        logits = torch.cat(logits_list, dim=0)
        return {'logits': logits, 'features': features_list, 'scores': scores_list}

    def auxiliary_losses(self, output):
        """Upstream feature-distance MSE between the four disentangled views."""
        if 'features' not in output:
            return {}
        total = 0.0
        count = 0
        for feature_g, feature_p, feature_comm, feature_red in output['features']:
            total = total + (F.mse_loss(feature_g, feature_comm) + F.mse_loss(feature_p, feature_comm)
                             + F.mse_loss(feature_g, feature_red) + F.mse_loss(feature_p, feature_red)
                             - F.mse_loss(feature_g, feature_p))
            count += 1
        return {'feature_mse': total / max(1, count)}

    def adaptation_metadata(self):
        return {'method': METHOD_NAME, 'official_model_reproduction': False,
                'upstream_url': UPSTREAM_URL, 'upstream_commit': UPSTREAM_COMMIT,
                'upstream_modules_executed': ['Transformer_P', 'Transformer_G',
                                              'Private_encoder', 'Share_encoder',
                                              'Gate', 'expert', 'NystromAttention',
                                              'BilinearFusion', 'SNN_Block'],
                'modalities': ['pathology', 'RNA'],
                'omic_groups': NUM_OMIC_GROUPS,
                'auxiliary_weights': dict(self.auxiliary_weights),
                'runtime_diffs': [
                    'Transformer_P/Transformer_G class-token .cuda() -> .to(device=features.device).',
                    'The upstream genomics stack is passed as the [1, 6, D] token tensor the upstream code intends.',
                    'The upstream network is executed once per patient because upstream assumes batch size 1.',
                ],
                'adaptations': [
                    'The six omic groups come from the upstream csv/signatures.csv functional families mapped onto the protocol RNA vector by exact gene-symbol matching.',
                    'The upstream Cox-free nll_loss is replaced by the shared patientwise discrete NLL; the upstream feature-distance MSE is retained as an auxiliary objective.',
                    'The upstream random feature reorganisation (seg_length choice) is kept and remains stochastic per forward pass.',
                ]}
