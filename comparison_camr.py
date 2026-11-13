"""Explicit two-modality CAMR adaptation for the fixed-input comparison.

Source: https://github.com/wxq-ustc/CAMR at
8fbae0a0bc4e9da31547b5eab5d773e6b2049e93, networks.py/fusion.py/utils.py.

This is CAMR-2M-P4096-DNLL, not an execution of the original three-modality
CAMR. It retains shared/private encoders, cross-modal gates, low-rank fusion,
reconstruction and orthogonality constraints. It adds attention MIL and RNA
input projections, removes the unavailable CNA branch, replaces the Cox head
with discrete-hazard logits, and supplies an explicit alternating adversarial
objective. No fake CNA input is constructed.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


METHOD_NAME = 'CAMR-2M-P4096-DNLL'
UPSTREAM_URL = 'https://github.com/wxq-ustc/CAMR'
UPSTREAM_COMMIT = '8fbae0a0bc4e9da31547b5eab5d773e6b2049e93'
DEFAULT_AUXILIARY_WEIGHTS = {'adversarial': .6, 'reconstruction': .8, 'orthogonality': .05}


def difference_loss(shared, private):
    """Upstream DiffLoss: centered, row-normalized cross-correlation energy.

    Norms are detached exactly as in the published utility. No rank or
    independence guarantee is implied by a small value.
    """
    if shared.ndim != 2 or private.shape != shared.shape or not len(shared):
        raise ValueError('Shared/private representations must be matching [B,D] matrices')
    a, b = shared-shared.mean(0, keepdim=True), private-private.mean(0, keepdim=True)
    a = a/(a.norm(dim=1, keepdim=True).detach()+1e-6)
    b = b/(b.norm(dim=1, keepdim=True).detach()+1e-6)
    return (a.T @ b).square().mean()


class TwoWayLowRankFusion(nn.Module):
    """Two-input counterpart of the original three-factor LMF, no dummy input."""
    def __init__(self, dimension, rank=16):
        super().__init__()
        if type(dimension) is not int or dimension < 1 or type(rank) is not int or rank < 1:
            raise ValueError('Positive integer fusion dimension and rank required')
        self.dimension = dimension
        self.rna_factor = nn.Parameter(torch.empty(rank, dimension+1, dimension))
        self.path_factor = nn.Parameter(torch.empty(rank, dimension+1, dimension))
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, dimension))
        for parameter in (self.rna_factor, self.path_factor, self.fusion_weights):
            nn.init.xavier_normal_(parameter)

    def forward(self, rna, path):
        if rna.ndim != 2 or path.shape != rna.shape or rna.shape[1] != self.dimension:
            raise ValueError('Fusion expects two matching [B,D] inputs')
        one = rna.new_ones((len(rna), 1))
        rna = torch.einsum('bi,rio->bro', torch.cat((one, rna), 1), self.rna_factor)
        path = torch.einsum('bi,rio->bro', torch.cat((one, path), 1), self.path_factor)
        return torch.einsum('r,bro->bo', self.fusion_weights[0], rna*path)+self.fusion_bias


class CrossModalGate(nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.hidden = nn.Linear(dimension, dimension)
        self.gate = nn.Bilinear(dimension, dimension, dimension)
        self.output = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, own, other):
        return self.output(torch.sigmoid(self.gate(own, other))*F.relu(self.hidden(own)))


class ModalityDiscriminator(nn.Module):
    """Logit version of upstream discriminator; stable BCEWithLogits loss."""
    def __init__(self, dimension):
        super().__init__()
        self.first = nn.Linear(dimension, 128)
        self.second = nn.Linear(128, 32)
        self.final = nn.Linear(32, 1)

    def forward(self, features, *, frozen_parameters=False):
        def linear(layer, x):
            weight = layer.weight.detach() if frozen_parameters else layer.weight
            bias = layer.bias.detach() if frozen_parameters else layer.bias
            return F.linear(x, weight, bias)
        x = F.leaky_relu(linear(self.first, features), negative_slope=.2)
        x = torch.tanh(linear(self.second, x))
        return linear(self.final, x)


class CAMR2M(nn.Module):
    """Attention-MIL WSI plus RNA CAMR adaptation, with explicit D/G losses.

    Padded zero feature rows deliberately remain visible to MIL, matching the
    fixed-budget legacy protocol. There is no padding mask or CNA argument.
    Model-specific adversarial/reconstruction/difference losses stay enabled.
    The caller owns two disjoint optimizers and the shared survival objective.
    """
    method_name = METHOD_NAME
    auxiliary_weights = DEFAULT_AUXILIARY_WEIGHTS

    def __init__(self, path_dim, rna_dim, bins=4, input_dim=80, latent_dim=60,
                 rank=16, dropout=.1):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (path_dim, rna_dim, bins, input_dim, latent_dim, rank)):
            raise ValueError('Positive integer model dimensions required')
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('dropout must be finite and lie in [0,1)')
        self.path_dim, self.rna_dim = path_dim, rna_dim
        self.bins, self.input_dim, self.latent_dim = bins, input_dim, latent_dim
        self.rank, self.dropout = rank, float(dropout)
        self.patch_encoder = nn.Sequential(nn.Linear(path_dim, input_dim), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(input_dim, max(1, input_dim//2)), nn.Tanh(),
                                       nn.Linear(max(1, input_dim//2), 1))
        self.rna_adapter = nn.Linear(rna_dim, input_dim)
        self.common = nn.Sequential(nn.Linear(input_dim, 320), nn.ReLU(),
                                    nn.Linear(320, 128), nn.ReLU(), nn.Linear(128, latent_dim), nn.ReLU())
        self.private_rna = nn.Sequential(nn.Linear(input_dim, 320), nn.ReLU(),
                                         nn.Linear(320, latent_dim), nn.ReLU())
        self.private_path = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(),
                                          nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, latent_dim), nn.ReLU())
        # The upstream executable reuses one gate for all modality orderings.
        self.common_gate = CrossModalGate(latent_dim, dropout)
        self.low_rank_fusion = TwoWayLowRankFusion(latent_dim, rank)
        self.survival_encoder = nn.Sequential(
            nn.Linear(3*latent_dim, 900), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(900, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 15), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(15, bins)
        self.reconstruct_rna = nn.Sequential(nn.Linear(2*latent_dim, 256), nn.ReLU(),
                                             nn.Linear(256, input_dim), nn.ReLU())
        self.reconstruct_path = nn.Sequential(nn.Linear(2*latent_dim, 256), nn.ReLU(),
                                              nn.Linear(256, input_dim), nn.ReLU())
        self.discriminator = ModalityDiscriminator(latent_dim)

    def _pool(self, patches, batch_size):
        if isinstance(patches, torch.Tensor):
            if patches.ndim != 3 or patches.shape[0] != batch_size:
                raise ValueError('Patch tensor must have shape [B,N,D]')
            bags = list(patches.unbind(0))
        elif isinstance(patches, (list, tuple)) and len(patches) == batch_size:
            bags = patches
        else:
            raise ValueError('Provide one patch tensor per RNA row')
        pooled = []
        parameter = next(self.parameters())
        for bag in bags:
            if (not isinstance(bag, torch.Tensor) or bag.ndim != 2 or not len(bag)
                    or bag.shape[1] != self.path_dim or not bag.is_floating_point()
                    or not torch.isfinite(bag).all()):
                raise ValueError('Each patch bag must be a finite nonempty [N,path_dim] tensor')
            features = self.patch_encoder(bag.to(device=parameter.device, dtype=parameter.dtype))
            weights = self.attention(features).softmax(dim=0)
            pooled.append((weights*features).sum(0))
        return torch.stack(pooled)

    def forward(self, patches, rna):
        if (not isinstance(rna, torch.Tensor) or rna.ndim != 2 or not len(rna)
                or rna.shape[1] != self.rna_dim or not rna.is_floating_point()
                or not torch.isfinite(rna).all()):
            raise ValueError('RNA must be a finite nonempty [B,rna_dim] tensor')
        parameter = next(self.parameters())
        path_input = self._pool(patches, len(rna))
        rna_input = self.rna_adapter(rna.to(device=parameter.device, dtype=parameter.dtype))
        path_common, rna_common = self.common(path_input), self.common(rna_input)
        path_private, rna_private = self.private_path(path_input), self.private_rna(rna_input)
        common = self.low_rank_fusion(self.common_gate(rna_common, path_common),
                                      self.common_gate(path_common, rna_common))
        logits = self.head(self.survival_encoder(torch.cat((common, rna_private, path_private), 1)))
        return {'logits': logits, 'path_input': path_input, 'rna_input': rna_input,
                'path_common': path_common, 'rna_common': rna_common,
                'path_private': path_private, 'rna_private': rna_private, 'fused_common': common,
                'path_reconstruction': self.reconstruct_path(torch.cat((common, path_private), 1)),
                'rna_reconstruction': self.reconstruct_rna(torch.cat((common, rna_private), 1))}

    def discriminator_loss(self, output):
        """D-only loss: feature detach prevents all encoder gradients."""
        rna = self.discriminator(output['rna_common'].detach())
        path = self.discriminator(output['path_common'].detach())
        return .5*(F.binary_cross_entropy_with_logits(rna, torch.ones_like(rna))
                   +F.binary_cross_entropy_with_logits(path, torch.zeros_like(path)))

    def auxiliary_losses(self, output):
        """Encoder-side confusion, reconstruction and original DiffLoss.

        D weights are detached for confusion, while feature gradients remain.
        This enforces independent optimizer ownership even if the caller does
        not temporarily toggle requires_grad on the discriminator.
        """
        rna = self.discriminator(output['rna_common'], frozen_parameters=True)
        path = self.discriminator(output['path_common'], frozen_parameters=True)
        adversarial = .5*(F.binary_cross_entropy_with_logits(rna, torch.full_like(rna, .5))
                          +F.binary_cross_entropy_with_logits(path, torch.full_like(path, .5)))
        reconstruction = F.mse_loss(output['rna_reconstruction'], output['rna_input']) \
                         +F.mse_loss(output['path_reconstruction'], output['path_input'])
        orthogonality = .5*(difference_loss(output['rna_common'], output['rna_private'])
                             +difference_loss(output['path_common'], output['path_private']))
        return {'adversarial': adversarial, 'reconstruction': reconstruction, 'orthogonality': orthogonality}

    def main_objective_components(self, output):
        """Alias for runners; raw auxiliary terms, survival supplied by caller."""
        return self.auxiliary_losses(output)

    def main_parameters(self):
        return (parameter for name, parameter in self.named_parameters()
                if not name.startswith('discriminator.'))

    def discriminator_parameters(self):
        return self.discriminator.parameters()

    def adaptation_metadata(self):
        return {'method': METHOD_NAME, 'official_model_reproduction': False,
                'upstream_url': UPSTREAM_URL, 'upstream_commit': UPSTREAM_COMMIT,
                'modalities': ['pathology', 'RNA'], 'fake_CNA_used': False,
                'input_dim': self.input_dim, 'latent_dim': self.latent_dim, 'rank': self.rank,
                'auxiliary_weights': dict(self.auxiliary_weights),
                'adaptations': [
                    'Three modalities reduced to pathology/RNA; no CNA input fabricated.',
                    'Added attention MIL over fixed-budget patch features and a learned RNA input projection.',
                    'Reconstruction targets are the learned 80D input embeddings, not the original full patch/RNA arrays.',
                    'Removed CNA private/reconstruction branch and changed three-way low-rank fusion to two-way.',
                    'Replaced scalar bounded Cox output with unbounded discrete-hazard logits.',
                    'Uses an explicit D-only classification step and encoder confusion-to-0.5 step.',
                    'Padded zero feature rows remain visible to MIL to match legacy fixed-budget inputs.',
                ]}
