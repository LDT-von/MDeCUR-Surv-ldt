"""PIBD adaptation to the unified BLCA DSS parity protocol.

Upstream source: https://github.com/zylbuaa/PIBD at
bd5bd94e6f8d48e7679c6e68209a3a65c9e56a78.

The upstream innovation modules are executed, not reimplemented:

* ``PIB`` (prototypical information bottleneck with risk-level proxies),
* ``PoE`` (product of experts),
* ``MITransformerLayer``/``MIAttention`` (prototypical information
  disentanglement),
* ``MIEstimator``/``CLUBMean`` (mutual-information upper bound),
* ``SNN_Block`` (pathway encoders),
* ``NLLSurvLoss`` (upstream survival likelihood, used for the proxy term).

Three runtime adaptations are applied, none of which modifies the checkout on
disk:

1. ``PIB.forward`` training branch hard-codes ``.cuda()`` on the proxy index
   and the attention mask.  A device-neutral copy of that method is installed
   at runtime; the diff is three device calls and nothing else.
2. ``PoE.gaussian_noise``/``PIB.gaussian_noise`` hard-code ``.cuda()``; a
   device-neutral sampler with identical semantics is installed at runtime.
3. The upstream training loop is replaced by the shared patientwise discrete
   NLL; the upstream proxy, KL and CLUB terms are retained as explicit
   auxiliary objectives.

This is ``PIBD-2M-P4096-DNLL``: an adaptation, not an original-paper
reproduction.
"""
import math
import types

import torch
from torch import nn
from torch.nn import functional as F

import comparison_sources as sources


METHOD_NAME = 'PIBD-2M-P4096-DNLL'
UPSTREAM_URL = 'https://github.com/zylbuaa/PIBD'
UPSTREAM_COMMIT = 'bd5bd94e6f8d48e7679c6e68209a3a65c9e56a78'

# Upstream loss composition: proxy NLL (alpha) + KL (beta) + CLUB MI.
DEFAULT_AUXILIARY_WEIGHTS = {'proxy': 1.0, 'kl': 0.1, 'club': 0.1}

# Runtime diff ledger for the upstream PIB.forward device patch.
PIB_FORWARD_DIFF = [
    'torch.tensor(proxy_indices).long().cuda() -> .to(device=x.device)',
    'torch.zeros_like(att, dtype=torch.bool).cuda() -> .to(device=x.device)',
    'torch.arange(att.size(0)) -> torch.arange(att.size(0), device=x.device)',
]


def _device_neutral_pib_forward(self, x, y=None, c=None):
    """Device-neutral copy of the upstream ``PIB.forward``.

    The body below is the upstream implementation with three ``.cuda()`` calls
    replaced by explicit device transfers.  No formula, index, threshold,
    ordering or return value changed.
    """
    feature_num = x.shape[1]
    device = x.device
    z = self.encoder_result(x)
    mu_proxy, sigma_proxy = self.encoder_proxies()
    eps_proxy = self.gaussian_noise(samples=([self.num_classes * 2, self.sample_num]),
                                     K=self.z_dim, seed=self.seed)
    z_proxy_sample = mu_proxy.unsqueeze(dim=1) + sigma_proxy.unsqueeze(dim=1) * eps_proxy
    z_proxy = torch.mean(z_proxy_sample, dim=1)
    z_norm = F.normalize(z, dim=2)
    z_proxy_norm = torch.unsqueeze(F.normalize(z_proxy), dim=0)
    att = torch.matmul(z_norm, torch.transpose(z_proxy_norm, 1, 2))

    if y is None and c is None:
        att_unbind_proxy = torch.cat(torch.unbind(att, dim=1), dim=1)
        _, att_topk_proxy_idx = torch.topk(att_unbind_proxy, self.topk, dim=1)
        att_topk_proxy_idx = att_topk_proxy_idx % (self.num_classes * 2)
        positive_proxy_idx, _ = torch.mode(att_topk_proxy_idx, dim=1)
        positive_proxy_idx = positive_proxy_idx.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(dim=1)
        proxy_loss = None
    else:
        proxy_indices = [self.proxies_dict[str(int(c_item)) + "," + str(int(y_item))]
                         for c_item, y_item in zip(c, y)]
        proxy_indices = torch.tensor(proxy_indices).long().to(device=device)
        mask = torch.zeros_like(att, dtype=torch.bool).to(device=device)
        mask[torch.arange(att.size(0), device=device), :, proxy_indices] = True
        att_positive = torch.masked_select(att, mask).view(att.size(0), att.size(1), 1)
        att_negative = torch.masked_select(att, ~mask).view(att.size(0), att.size(1), -1)
        att_topk_positive, att_topk_idx_positive = torch.topk(att_positive.squeeze(dim=2),
                                                             self.topk, dim=1)
        att_topk_negative, _ = torch.topk(att_negative, self.topk, dim=1)
        att_positive_mean = torch.mean(att_topk_positive, dim=1)
        att_negative_mean = torch.mean(torch.mean(att_topk_negative, dim=1), dim=1)
        proxy_loss = -(att_positive_mean - att_negative_mean).mean()
        positive_proxy_idx = proxy_indices.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(dim=1)

    mu_proxy_repeat = mu_proxy.repeat(x.shape[0], 1, 1)
    sigma_proxy_repeat = sigma_proxy.repeat(x.shape[0], 1, 1)
    mu_topk = torch.gather(mu_proxy_repeat, 1, positive_proxy_idx).squeeze(dim=1)
    sigma_topk = torch.gather(sigma_proxy_repeat, 1, positive_proxy_idx).squeeze(dim=1)
    att_unbind = torch.cat(torch.unbind(att, dim=2), dim=1)
    att_topk, att_topk_idx = torch.topk(att_unbind, self.topk, dim=1)
    att_topk_idx = att_topk_idx % feature_num
    z_topk = torch.gather(z, 1, att_topk_idx.unsqueeze(dim=2).repeat(1, 1, self.z_dim))
    decoder_logits_proxy = torch.mean(self.decoder_logits(z_proxy_sample), dim=1)
    return (decoder_logits_proxy, mu_proxy, sigma_proxy, z_topk, mu_topk, sigma_topk,
            proxy_loss)


class _PibdArgs:
    """Minimal argparse-shaped namespace expected by the upstream ``PIBD``."""

    def __init__(self, omic_sizes, num_classes, encoding_dim, wsi_projection_dim,
                 ratio_wsi=.5, ratio_omics=.5, sample_num=16, seed=3, num_patches=4096,
                 alpha=.5, beta=.1, omics_format='pathways'):
        self.omic_sizes = list(omic_sizes)
        self.n_classes = int(num_classes)
        self.encoding_dim = int(encoding_dim)
        self.wsi_projection_dim = int(wsi_projection_dim)
        self.ratio_wsi = float(ratio_wsi)
        self.ratio_omics = float(ratio_omics)
        self.sample_num = int(sample_num)
        self.seed = int(seed)
        self.num_patches = int(num_patches)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.omics_format = str(omics_format)


class PIBD2M(nn.Module):
    """Two-modality PIBD adapter that reuses the upstream model verbatim.

    The upstream ``PIBD`` module is instantiated with the protocol dimensions
    and its submodules are called directly, so the innovation modules remain
    the ones published upstream.
    """

    method_name = METHOD_NAME
    auxiliary_weights = DEFAULT_AUXILIARY_WEIGHTS

    def __init__(self, path_dim, rna_dim, omic_indices, bins=4, projection_dim=128,
                 sample_num=16, seed=3, num_patches=4096, ratio_wsi=.5, ratio_omics=.5,
                 bag_size=512, alpha=.5, beta=.1):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (path_dim, rna_dim, bins, projection_dim,
                                                      sample_num, num_patches, bag_size)):
            raise ValueError('Positive integer dimensions required')
        if not omic_indices or not all(len(group) > 0 for group in omic_indices):
            raise ValueError('At least one pathway with matched genes is required')
        if max(max(group) for group in omic_indices) >= rna_dim:
            raise ValueError('Pathway gene index outside the RNA dimension')
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must lie in [0,1]')
        self.path_dim, self.rna_dim, self.bins = path_dim, rna_dim, bins
        self.omic_indices = [list(map(int, group)) for group in omic_indices]
        self.omic_sizes = [len(group) for group in self.omic_indices]
        self.num_patches, self.bag_size = num_patches, bag_size
        self.seed = seed
        self.projection_dim = projection_dim

        modules = sources.load_pibd_modules()
        self.upstream = modules['models.model_PIBD']
        sources.device_agnostic(self.upstream)
        # Runtime device patch of the upstream PIB.forward.  The function is
        # installed on the class so it binds as a normal method.
        self.upstream.PIB.forward = _device_neutral_pib_forward

        args = _PibdArgs(self.omic_sizes, bins, path_dim, projection_dim,
                         ratio_wsi=ratio_wsi, ratio_omics=ratio_omics,
                         sample_num=sample_num, seed=seed, num_patches=num_patches,
                         alpha=alpha, beta=beta)
        self.args = args
        # The upstream class owns the bag size constant; align it with the
        # parity patch budget without touching the file.
        self.upstream.BAG_SIZE = bag_size
        self.model = self.upstream.PIBD(args, omic_names=[], omics_input_dim=rna_dim)
        # The upstream path projection expects the raw UNI feature dimension;
        # PIB_omics, PID, to_logits and CLUB are built by the upstream code.

    # ------------------------------------------------------------------ input
    def _pathway_inputs(self, rna):
        """Split the patient RNA vector into the upstream per-pathway inputs."""
        if (not isinstance(rna, torch.Tensor) or rna.ndim != 2 or len(rna) < 1
                or rna.shape[1] != self.rna_dim or not rna.is_floating_point()
                or not torch.isfinite(rna).all()):
            raise ValueError('RNA must be a finite nonempty [B,rna_dim] tensor')
        return [rna[:, group] for group in self.omic_indices]

    def _device(self):
        return next(self.parameters()).device

    def forward(self, patches, rna, labels=None, censorship=None, generator=None):
        """Return discrete-hazard logits and the upstream auxiliary tensors.

        ``patches`` is either a ``[B,N,D]`` tensor or a list of ``[N,D]``
        tensors, one per patient.  ``labels``/``censorship`` are only used for
        the upstream proxy objective; they are never used at evaluation time.
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
        device = self._device()
        rna = rna.to(device=device, dtype=next(self.parameters()).dtype)
        pathway_inputs = self._pathway_inputs(rna)

        # Upstream per-pathway encoders (SNN_Block stacks).
        h_omic = [self.model.sig_networks[index].forward(values)
                  for index, values in enumerate(pathway_inputs)]
        h_omic_bag = torch.stack(h_omic).permute(1, 0, 2)  # [B, pathways, D]

        training_labels = labels
        if training_labels is not None:
            training_labels = training_labels.reshape(-1).long().to(device)
        training_censorship = censorship
        if training_censorship is not None:
            training_censorship = training_censorship.reshape(-1).long().to(device)

        iterations = max(1, self.num_patches // self.bag_size)
        logits_list = []
        proxy_total, kl_total = 0.0, 0.0
        for _ in range(iterations):
            idx = self._bag_indices(bags[0].shape[0], generator)
            h_wsi_bag = torch.stack([bag.to(device=device, dtype=rna.dtype)[idx] for bag in bags])

            decoder_logits_proxy_wsi, mu_proxy_wsi, sigma_proxy_wsi, h_wsi_bag, \
                mu_topk_wsi, sigma_topk_wsi, proxy_loss_wsi = self.model.PIB_wsi(
                    h_wsi_bag, training_labels, training_censorship)
            decoder_logits_proxy_omic, mu_proxy_omic, sigma_proxy_omic, h_omic_bag, \
                mu_topk_omic, sigma_topk_omic, proxy_loss_omic = self.model.PIB_omics(
                    h_omic_bag, training_labels, training_censorship)

            mu_list = [mu_topk_wsi, mu_topk_omic]
            var_list = [sigma_topk_wsi, sigma_topk_omic]
            poe_features = self.model.PoE(mu_list, var_list)
            poe_features = torch.mean(poe_features, dim=1).unsqueeze(dim=1)
            poe_embed = poe_features.expand(h_wsi_bag.shape[0], 1, -1)

            histology, pathways, global_embed = self.model.PID(
                h_wsi_bag, h_omic_bag, poe_embed, False)
            histology = torch.mean(histology, dim=1)
            pathways = torch.mean(pathways, dim=1)
            global_embed = torch.mean(global_embed, dim=1)

            logits_list.append(self.model.to_logits(
                torch.cat([histology, pathways, global_embed], dim=-1)))
            if self.training and training_labels is not None:
                proxy_total += float(self._proxy_loss(decoder_logits_proxy_wsi)
                                     + self._proxy_loss(decoder_logits_proxy_omic))
                kl_total += float(self._kl_loss(mu_proxy_wsi, sigma_proxy_wsi)
                                  + self._kl_loss(mu_proxy_omic, sigma_proxy_omic))

        logits = torch.mean(torch.stack(logits_list), dim=0)
        output = {'logits': logits, 'histology': histology, 'pathways': pathways,
                  'global_embed': global_embed}
        if self.training and training_labels is not None:
            output['proxy_loss'] = proxy_total / iterations
            output['kl_loss'] = kl_total / iterations
            output['club_loss'] = float(self.model.CLUB.learning_loss(
                histology, pathways, global_embed))
            output['club_estimate'] = float(self.model.CLUB(histology, pathways, global_embed))
        return output

    def _bag_indices(self, num_patches, generator=None):
        """Select one upstream-style random bag of patches."""
        count = min(self.bag_size, num_patches)
        if generator is None:
            idx = torch.randperm(num_patches)[:count]
        else:
            idx = torch.randperm(num_patches, generator=generator)[:count]
        return idx.to(self._device())

    def _proxy_loss(self, proxy_logits):
        """Upstream proxy objective over the (censor, bin) proxy classes.

        The upstream ``get_loss_proxy`` builds its censor vector and label
        vector with an unconditional ``.cuda()``; the runtime patch below
        transfers them to the logits device instead.  The objective itself is
        the upstream ``NLLSurvLoss``.
        """
        bins = self.bins
        device = proxy_logits.device
        censor = torch.cat([torch.zeros(bins), torch.ones(bins)]).to(device=device)
        y = torch.arange(0, bins).repeat(2).to(device=device)
        return self.model.loss_surv(h=proxy_logits, y=y, t=None, c=censor)

    def _kl_loss(self, mu, sigma):
        """Upstream KL between the prior and the proxy posterior."""
        prior = torch.zeros_like(mu), torch.ones_like(sigma)
        value = self.upstream.KL_between_normals((mu, sigma), prior)
        return torch.mean(value)

    def auxiliary_losses(self, output):
        """Upstream proxy/KL/CLUB terms that accompany the shared survival loss."""
        if 'proxy_loss' not in output:
            return {}
        return {'proxy': output['proxy_loss'], 'kl': output['kl_loss'],
                'club': output['club_loss']}

    def adaptation_metadata(self):
        return {'method': METHOD_NAME, 'official_model_reproduction': False,
                'upstream_url': UPSTREAM_URL, 'upstream_commit': UPSTREAM_COMMIT,
                'upstream_modules_executed': ['PIB', 'PoE', 'MITransformerLayer',
                                              'MIAttention', 'MIEstimator', 'CLUBMean',
                                              'SNN_Block', 'NLLSurvLoss'],
                'modalities': ['pathology', 'RNA'],
                'auxiliary_weights': dict(self.auxiliary_weights),
                'runtime_diffs': list(PIB_FORWARD_DIFF),
                'adaptations': [
                    'Upstream Gaussian samplers and the upstream PIB.forward device calls are replaced at runtime; the checkout is unchanged.',
                    'The upstream per-epoch random WSI bag is drawn from the fixed-budget patch rows.',
                    'The upstream training loop is replaced by the shared patientwise discrete NLL; upstream proxy, KL and CLUB terms are retained as auxiliary objectives.',
                    'Pathway RNA groups come from the protocol train-fitted GMT mapping, never from the held-out split.',
                ]}
