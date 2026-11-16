"""Two-modality CAMR adaptation built on the **unmodified upstream modules**.

Source: https://github.com/wxq-ustc/CAMR at
8fbae0a0bc4e9da31547b5eab5d773e6b2049e93.

Unlike the earlier reimplementation, this adapter *executes the upstream
classes*:

* ``networks.MisaLMFGatedRec`` - shared encoder, three private encoders,
  bilinear cross-modal gates, low-rank fusion (``fusion.LMF``), the survival
  encoder and the three reconstruction heads.
* ``fusion.Discriminator`` - the modality discriminator.
* ``utils.DiffLoss`` - the shared/private orthogonality energy.
* ``utils.MSE`` and ``utils.regularize_weights`` - reconstruction and the
  upstream L1 weight regularization.

Nothing in the upstream checkout is edited: the classes are imported from the
read-only tree by ``comparison_sources.load_camr_modules`` and only *called*.

Adaptations (all of them protocol level, none of them structural)
-----------------------------------------------------------------
1. **Three modalities -> two.**  BLCA provides UNI pathology features and bulk
   RNA only; there is no CNA input in this protocol.  The third upstream slot
   is therefore filled with a **constant vector that carries no patient
   information** (the upstream low-rank fusion multiplies three factors, so the
   slot cannot simply be zeroed).  No expression, CNA or survival value is
   fabricated for that slot.
2. **Input contract.**  Upstream consumes one 80-dimensional vector per
   modality per patient.  Pathology is pooled from the fixed 4096-row patch
   budget with attention MIL, and RNA is projected to the same 80 dimensions;
   both projections live *outside* the upstream module.
3. **Survival head.**  The upstream bounded Cox output (sigmoid, scaled to
   [-4, 4]) is replaced by unbounded discrete-hazard logits, because the
   shared protocol scores every arm with the same discrete NLL.  The upstream
   encoder stack, the fusion and the reconstruction heads are untouched.
4. **Discriminator update.**  Upstream builds the discriminator but omits it
   from the optimizer, so its adversarial term never updates a single weight.
   Here the discriminator has its own optimizer and is updated explicitly.
   The adversarial objective itself is upstream's own formulation: the shared
   RNA representation is the "real" class and the shared pathology
   representation is the "fake" class.

Everything else - the gate structure, the low-rank fusion, the reconstruction
targets, the orthogonality energy and the auxiliary weights (0.6 / 0.8 / 0.05)
- is upstream's.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import comparison_sources as sources


METHOD_NAME = 'CAMR-2M-P4096-DNLL'
UPSTREAM_URL = 'https://github.com/wxq-ustc/CAMR'
UPSTREAM_COMMIT = '8fbae0a0bc4e9da31547b5eab5d773e6b2049e93'
DEFAULT_AUXILIARY_WEIGHTS = {'adversarial': .6, 'reconstruction': .8, 'orthogonality': .05,
                              'weight_regularization': 2e-6}
LATENT_DIM = 60          # upstream hard-codes 60 shared/private dimensions
INPUT_DIM = 80           # upstream hard-codes the 80-dimensional patient vectors


class CAMR2M(nn.Module):
    """Attention-MIL pathology plus RNA, on top of the upstream CAMR model.

    The public interface is unchanged with respect to the other arms: a
    ``forward(patches, rna)`` that returns at least ``logits``, an
    ``auxiliary_losses(output)`` mapping, an ``auxiliary_weights`` mapping and
    the disjoint ``main_parameters`` / ``discriminator_parameters`` sets.
    """

    method_name = METHOD_NAME
    auxiliary_weights = DEFAULT_AUXILIARY_WEIGHTS

    def __init__(self, path_dim, rna_dim, bins=4, input_dim=INPUT_DIM,
                 dropout=.1):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (path_dim, rna_dim, bins, input_dim)):
            raise ValueError('Positive integer model dimensions required')
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('dropout must be finite and lie in [0,1)')
        self.path_dim, self.rna_dim = path_dim, rna_dim
        self.bins, self.input_dim = bins, input_dim

        upstream = sources.load_camr_modules()
        self._upstream_names = {'networks': 'networks', 'fusion': 'fusion', 'utils': 'utils'}
        networks, fusion, utils = (upstream['networks'], upstream['fusion'], upstream['utils'])
        self._upstream_classes = {
            'MisaLMFGatedRec': networks.MisaLMFGatedRec,
            'LMF': fusion.LMF,
            'Discriminator': fusion.Discriminator,
            'DiffLoss': utils.DiffLoss,
            'MSE': utils.MSE,
        }

        # The upstream model itself: shared encoder, private encoders, bilinear
        # cross-modal gates, low-rank fusion, survival encoder, reconstructions.
        self.upstream_model = networks.MisaLMFGatedRec(
            input_dim, bins, hidden_size=20, hidden_size1=input_dim, dropout=dropout)
        # Protocol alignment: discrete-hazard logits instead of the bounded Cox
        # output.  The upstream forward applies ``classifier`` then rescales it
        # with ``output_range`` / ``output_shift``; neutralising the rescaling
        # leaves the upstream computation untouched.
        self.upstream_model.classifier = nn.Linear(15, bins)
        with torch.no_grad():
            self.upstream_model.output_range.fill_(1.)
            self.upstream_model.output_shift.fill_(0.)

        # Upstream losses and discriminator, used exactly as upstream uses them.
        self.diff_loss = utils.DiffLoss()
        self.mse = utils.MSE()
        self.discriminator = fusion.Discriminator(LATENT_DIM)
        self.regularize_weights = utils.regularize_weights

        # Input adapters (outside the upstream module).
        self.patch_encoder = nn.Sequential(nn.Linear(path_dim, input_dim), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(input_dim, max(1, input_dim//2)), nn.Tanh(),
                                       nn.Linear(max(1, input_dim//2), 1))
        self.rna_adapter = nn.Linear(rna_dim, input_dim)
        # Constant third slot: no patient information whatsoever.
        self.register_buffer('placeholder', torch.ones(1, input_dim), persistent=False)

    # ------------------------------------------------------------------ input
    def _pool(self, patches, batch_size):
        """Attention MIL over the fixed patch budget; padded rows stay visible."""
        if isinstance(patches, torch.Tensor):
            if patches.ndim != 3 or patches.shape[0] != batch_size:
                raise ValueError('Patch tensor must have shape [B,N,D]')
            bags = list(patches.unbind(0))
        elif isinstance(patches, (list, tuple)) and len(patches) == batch_size:
            bags = patches
        else:
            raise ValueError('Provide one patch bag per RNA row')
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

    # ---------------------------------------------------------------- forward
    def forward(self, patches, rna):
        """Run the upstream CAMR model on RNA (gene slot) and pathology."""
        if (not isinstance(rna, torch.Tensor) or rna.ndim != 2 or not len(rna)
                or rna.shape[1] != self.rna_dim or not rna.is_floating_point()
                or not torch.isfinite(rna).all()):
            raise ValueError('RNA must be a finite nonempty [B,rna_dim] tensor')
        parameter = next(self.parameters())
        path_input = self._pool(patches, len(rna))
        rna_input = self.rna_adapter(rna.to(device=parameter.device, dtype=parameter.dtype))
        placeholder = self.placeholder.expand(len(rna), -1).to(rna_input.dtype)
        outputs = self.upstream_model(rna_input, path_input, placeholder)
        (logits, rna_target, rna_reconstruction, path_target, path_reconstruction,
         placeholder_target, placeholder_reconstruction, rna_common, path_common,
         placeholder_common, rna_private, path_private,
         placeholder_private) = outputs
        return {'logits': logits,
                'rna_input': rna_target, 'path_input': path_target,
                'placeholder_input': placeholder_target,
                'rna_common': rna_common, 'path_common': path_common,
                'placeholder_common': placeholder_common,
                'rna_private': rna_private, 'path_private': path_private,
                'placeholder_private': placeholder_private,
                'rna_reconstruction': rna_reconstruction,
                'path_reconstruction': path_reconstruction,
                'placeholder_reconstruction': placeholder_reconstruction}

    # ----------------------------------------------------------------- losses
    def _adversarial_terms(self, output, detach_features, freeze_discriminator):
        """Upstream's discriminator formulation, reduced to the two real modalities.

        Upstream labels the shared *genomic* representation as "real" and the
        shared *pathology* and *CNA* representations as "fake".  Without a CNA
        input the third fake term is dropped; the placebo slot carries no
        patient information and would only teach the discriminator a constant.
        """
        layers = self.discriminator.model
        if freeze_discriminator:
            def linear(index, x):
                layer = layers[index]
                return F.linear(x, layer.weight.detach(), layer.bias.detach())

            def evaluate(features):
                x = F.leaky_relu(linear(0, features), negative_slope=.2)
                x = torch.tanh(linear(2, x))
                return torch.sigmoid(linear(4, x))
        else:
            evaluate = self.discriminator

        rna = output['rna_common'].detach() if detach_features else output['rna_common']
        path = output['path_common'].detach() if detach_features else output['path_common']
        real = evaluate(rna)
        fake = evaluate(path)
        bce = nn.BCELoss()
        return .5*(bce(real, torch.ones_like(real))+bce(fake, torch.zeros_like(fake)))

    def discriminator_loss(self, output):
        """D-only loss: the features are detached, so no encoder gets a gradient."""
        return self._adversarial_terms(output, detach_features=True, freeze_discriminator=False)

    def auxiliary_losses(self, output):
        """Encoder-side terms: upstream adversarial, reconstruction and DiffLoss."""
        adversarial = self._adversarial_terms(output, detach_features=False,
                                              freeze_discriminator=True)
        reconstruction = (self.mse(output['rna_input'], output['rna_reconstruction'])
                          + self.mse(output['path_input'], output['path_reconstruction'])
                          + self.mse(output['placeholder_input'],
                                     output['placeholder_reconstruction']))
        orthogonality = (self.diff_loss(output['rna_common'], output['rna_private'])
                         + self.diff_loss(output['path_common'], output['path_private'])
                         + self.diff_loss(output['placeholder_common'],
                                          output['placeholder_private']))/3
        regularization = self.regularize_weights(self.upstream_model)
        return {'adversarial': adversarial, 'reconstruction': reconstruction,
                'orthogonality': orthogonality, 'weight_regularization': regularization}

    # ------------------------------------------------------------- parameters
    def main_parameters(self):
        return (parameter for name, parameter in self.named_parameters()
                if not name.startswith('discriminator.'))

    def discriminator_parameters(self):
        return self.discriminator.parameters()

    def adaptation_metadata(self):
        return {'method': METHOD_NAME, 'official_model_reproduction': False,
                'upstream_url': UPSTREAM_URL, 'upstream_commit': UPSTREAM_COMMIT,
                'upstream_modules_executed': sorted(self._upstream_classes),
                'upstream_model_class': 'networks.MisaLMFGatedRec',
                'modalities': ['pathology', 'RNA'],
                'fake_CNA_used': False,
                'placeholder_slot': 'constant vector without patient information',
                'input_dim': self.input_dim, 'latent_dim': LATENT_DIM,
                'auxiliary_weights': dict(self.auxiliary_weights),
                'adaptations': [
                    'Three upstream slots reduced to pathology/RNA; the third slot is a '
                    'constant placeholder, no CNA-like data is fabricated.',
                    'Attention MIL pooling of the 4096-row patch budget and an RNA input '
                    'projection are applied outside the upstream module.',
                    'The bounded Cox output is replaced by discrete-hazard logits; the '
                    'upstream encoder, gates, low-rank fusion and reconstructions are used as is.',
                    'The discriminator receives its own optimizer; upstream omits it from '
                    'the optimizer and therefore never updates it.',
                    'The upstream third "fake" discriminator term is dropped because the '
                    'placeholder carries no patient information.',
                    'Padded zero feature rows remain visible to MIL to match the legacy '
                    'fixed-budget inputs.',
                ]}
