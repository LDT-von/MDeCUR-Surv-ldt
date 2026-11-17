"""DIMAF adaptation to the unified BLCA DSS parity protocol.

Upstream source: https://github.com/Trustworthy-AI-UU-NKI/DIMAF at
3fe81236aafac88977b730745314b8c9906cb4d3.

The upstream innovation modules are executed, not reimplemented:

* ``MultiSNN`` / ``SNN_Block`` (per-pathway RNA encoders),
* ``CrossAttentionLayer`` (the four disentangled attention blocks),
* ``DIMAF`` (prototype type embeddings, fusion and survival head),
* ``PANTHERBase`` / ``DirNIWNet`` / ``mog_eval`` (prototype slide embedding),
* ``DistanceCorrelationLoss`` / ``OrthogonalLoss`` / ``HSICLoss`` (the
  disentanglement objectives),
* the upstream ``kmeans_clustering`` prototype fitting.

Adaptations, none of which modifies the checkout on disk:

1. The upstream prototype file is replaced by prototypes clustered on the
   **training rows of the current fold only**, so the validation partition
   never influences the prototypes.
2. The upstream slide-embedding step (``create_slide_embeddings``) is driven
   over the protocol patch bags instead of the upstream H5 dataloader; the
   upstream ``PANTHERBase``/``DirNIWNet`` math is unchanged.
3. The survival objective is the shared patientwise discrete NLL
   (``cancer_parity.discrete_nll``) so every arm of the comparison optimises
   the same likelihood; the upstream disentanglement loss is retained as an
   auxiliary objective with its own weight.

This is ``DIMAF-2M-P4096-DNLL``: an adaptation, not an original-paper
reproduction.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import comparison_sources as sources


METHOD_NAME = 'DIMAF-2M-P4096-DNLL'
UPSTREAM_URL = 'https://github.com/Trustworthy-AI-UU-NKI/DIMAF'
UPSTREAM_COMMIT = '3fe81236aafac88977b730745314b8c9906cb4d3'
UNI_FEATURE_DIM = 1024

DEFAULT_AUXILIARY_WEIGHTS = {'disentanglement': 7.0}
DEFAULT_DISENTANGLEMENT = 'distcor'

# The upstream prototype step samples ``n_proto * n_proto_patches`` patch rows
# from the training slides (upstream default: 16 * 100000 = 1.6e6 rows) and
# clusters them with the upstream ``kmeans_clustering``.  That sample needs
# 1.6e6 * 1024 * 4 B = 6.4 GB, and the scikit-learn KMeans working copy is
# float64, so the upstream default does not fit next to the other arms on this
# host.  The quota is therefore reduced to 16 * 10000 = 1.6e5 rows, which is
# still 1e4 rows per prototype and keeps the step inside a few gigabytes.
# This is a documented adaptation of the upstream sampling size; the sampling
# rule, the upstream clustering implementation and the prototype count are
# unchanged.
DIMAF_PROTOTYPE_PATCHES_PER_PROTO = 10000


def _load_prototype_module():
    """Import the upstream prototype clustering helper."""
    root = sources.UPSTREAM['DIMAF']['root']
    return sources._load_module_from_path(
        'embeddings.prototype', root / 'src' / 'embeddings' / 'prototype.py',
        [str(root / 'src')])


def sample_prototype_rows(bags, n_proto, n_patches_per_proto, feature_dim, seed=3):
    """Draw the prototype clustering sample from the training bags.

    This reproduces the upstream ``embeddings.prototype.get_patches`` rule: the
    sample holds at most ``n_proto * n_patches_per_proto`` rows, a fixed quota
    is drawn from every training slide with ``torch.randperm``, and the sample
    is what the upstream ``kmeans_clustering`` then clusters.  Unlike the
    upstream code the rows are read from the protocol bags in bounded blocks
    instead of one dense in-memory copy of every slide, which is what makes
    the step fit in memory on this host.  The upstream ``get_patches`` samples
    with ``torch.randperm`` per slide; the same generator is used here so the
    draw is reproducible from the recorded seed.
    """
    if type(n_proto) is not int or n_proto < 1:
        raise ValueError('n_proto must be a positive integer')
    if type(n_patches_per_proto) is not int or n_patches_per_proto < 1:
        raise ValueError('n_patches_per_proto must be a positive integer')
    if not isinstance(bags, (list, tuple)) or not bags:
        raise ValueError('A nonempty list of training bags is required')
    # The upstream rule draws a quota of rows from every training slide with
    # ``torch.randperm``.  The rows are collected per patient bag and read
    # through the bag's batched ``read_rows`` call, which copies only the
    # requested rows; reading whole slides would pull the complete training
    # bags (tens of gigabytes) through the page cache.
    generator = torch.Generator().manual_seed(seed)
    offsets_per_bag = []
    total_slides = 0
    for bag in bags:
        offsets, cursor = [], 0
        for block in bag.blocks(bag.n_patches):
            offsets.append((cursor, cursor + (block[2] - block[1])))
            cursor += block[2] - block[1]
            total_slides += 1
        if cursor != bag.n_patches:
            raise ValueError('The source bag rows do not match its declared patch count')
        offsets_per_bag.append((bag, offsets))
    if total_slides < 1:
        raise ValueError('The training bags hold no slide rows')
    total_target = n_proto * n_patches_per_proto
    per_slide = max(1, -(-total_target // total_slides))
    # The sample is allocated once: repeated concatenation would hold a second
    # full copy of a multi-gigabyte tensor while it is being built.
    sample = torch.zeros((total_target, feature_dim), dtype=torch.float32)
    collected = 0
    for bag, offsets in offsets_per_bag:
        if collected >= total_target:
            break
        wanted = []
        for base, stop in offsets:
            rows = stop - base
            if rows < 1:
                continue
            take = min(per_slide, rows, total_target - collected + len(wanted))
            indices = torch.randperm(rows, generator=generator)[:take]
            indices, _ = torch.sort(indices)
            wanted.extend((base + int(index)) for index in indices)
        if not wanted:
            continue
        room = total_target - collected
        if len(wanted) > room:
            wanted = wanted[:room]
        selected = bag.read_rows(wanted)
        if selected.shape[0] != len(wanted):
            raise ValueError('The bag did not return the requested rows')
        sample[collected:collected+selected.shape[0]] = selected
        collected += selected.shape[0]
    if collected < n_proto:
        raise ValueError('Not enough training patch features to fit prototypes')
    return sample[:collected].contiguous()

def fit_prototypes(patch_features, n_proto=16, n_iter=50, feature_dim=None, seed=3):
    """Cluster pathology prototypes on training-fold patch features only.

    ``patch_features`` must already come from the training rows.  The upstream
    ``kmeans_clustering`` is used, so the clustering itself is the upstream
    implementation.
    """
    if type(n_proto) is not int or n_proto < 1:
        raise ValueError('n_proto must be a positive integer')
    if not isinstance(patch_features, torch.Tensor) or patch_features.ndim != 2:
        raise ValueError('Prototype features must be a [N, D] tensor')
    if feature_dim is None:
        feature_dim = patch_features.shape[1]
    if patch_features.shape[1] != feature_dim:
        raise ValueError('Prototype features must have the expected width')
    if len(patch_features) < n_proto:
        raise ValueError('Not enough training patch features to fit prototypes')
    if not torch.isfinite(patch_features).all():
        raise ValueError('Nonfinite patch features in prototype fitting')
    module = _load_prototype_module()
    torch.manual_seed(seed)
    # The upstream ``kmeans_clustering`` calls scikit-learn's ``KMeans``
    # without a ``random_state``, so its initialisation is drawn from NumPy's
    # **global** random state.  That makes the prototypes depend on whatever
    # ran before in the process, and two identical runs in two processes
    # produced different prototypes (measured: the DIMAF arm was not
    # bit-reproducible across two identical-seed runs, while the two arms that
    # do not cluster were).  Seeding NumPy's global state from the recorded
    # seed immediately before the call makes the upstream clustering
    # reproducible without changing the upstream file or the upstream algorithm.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        weights = module.kmeans_clustering(patch_features.cpu(), n_proto, n_iter)
    finally:
        np.random.set_state(state)
    array = np.asarray(weights, dtype=np.float64)
    if array.shape != (1, n_proto, feature_dim):
        raise ValueError('Unexpected prototype array shape')
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite prototypes')
    return array[0]


class PantherSlideEmbedding(nn.Module):
    """Upstream PANTHER slide embedding driven over protocol patch bags.

    The upstream ``PANTHERBase`` performs the EM update over the patch set of
    one slide.  The upstream wrapper additionally reshapes the mixture output
    into the ``[B, p, d+1]`` prob/mean tensor that the upstream ``DIMAF``
    model consumes; that reshape is reproduced here unchanged.
    """

    def __init__(self, feature_dim, prototypes, n_proto=16, em_iter=1, tau=.001,
                 ot_eps=.1, fix_proto=True):
        super().__init__()
        modules = sources.load_dimaf_modules()
        # The upstream DirNIWNet stores the prototypes as a float32 parameter.
        prototypes = np.ascontiguousarray(np.asarray(prototypes, dtype=np.float32))
        self.panther_base = modules['models.PANTHER.layers'].PANTHERBase(
            feature_dim, prototypes, p=n_proto, L=em_iter, tau=tau, ot_eps=ot_eps,
            fix_proto=fix_proto)
        self.n_proto = n_proto
        self.feature_dim = feature_dim

    def forward(self, bags):
        """Return the ``[B, n_proto, d+1]`` embedding for a list of bags."""
        if not isinstance(bags, (list, tuple)) or not bags:
            raise ValueError('A nonempty list of patch bags is required')
        outputs = []
        for bag in bags:
            if (not isinstance(bag, torch.Tensor) or bag.ndim != 2 or len(bag) < 1
                    or bag.shape[1] != self.feature_dim or not torch.isfinite(bag).all()):
                raise ValueError('Each patch bag must be a finite nonempty [N, D] tensor')
            data = bag.unsqueeze(0).to(device=next(self.parameters()).device,
                                       dtype=next(self.parameters()).dtype)
            out, _ = self.panther_base(data)
            outputs.append(out[0])
        stacked = torch.stack(outputs)  # [B, p + 2 p d]
        probability = stacked[:, :self.n_proto]
        mean = stacked[:, self.n_proto:self.n_proto*(1+self.feature_dim)]
        mean = mean.reshape(-1, self.n_proto, self.feature_dim)
        return torch.cat([probability.unsqueeze(-1), mean], dim=-1)

class DIMAF2M(nn.Module):
    """Two-modality DIMAF adapter reusing the upstream model and losses."""

    method_name = METHOD_NAME
    auxiliary_weights = DEFAULT_AUXILIARY_WEIGHTS

    def __init__(self, path_dim, rna_dim, omic_indices, bins=4, single_out_dim=256,
                 n_proto=16, em_iter=1, tau=.001, ot_eps=.1, fix_proto=True,
                 disentanglement=DEFAULT_DISENTANGLEMENT, prototypes=None, seed=3):
        super().__init__()
        if not omic_indices or not all(len(group) > 0 for group in omic_indices):
            raise ValueError('At least one pathway with matched genes is required')
        if max(max(group) for group in omic_indices) >= rna_dim:
            raise ValueError('Pathway gene index outside the RNA dimension')
        if disentanglement not in ('distcor', 'orthogonal', 'hsic'):
            raise ValueError('Unknown upstream disentanglement loss')
        self.path_dim, self.rna_dim, self.bins = path_dim, rna_dim, bins
        self.omic_indices = [list(map(int, group)) for group in omic_indices]
        self.omic_sizes = [len(group) for group in self.omic_indices]
        self.n_proto = n_proto
        self.seed = seed

        modules = sources.load_dimaf_modules()
        self.upstream = modules
        self.loss_module = modules['survival.losses']
        # The upstream DIMAF consumes the prob/mean slide embedding of width
        # d+1, where d is the pathology feature width (upstream in_dim).
        self.model = modules['models.DIMAF.main_model'].DIMAF(
            rna_dims=self.omic_sizes,
            histo_dim=path_dim+1,
            device=torch.device('cpu'),
            num_classes=bins,
            single_out_dim=single_out_dim,
            loss_fn=None,
            num_proto_wsi=n_proto)
        if disentanglement == 'distcor':
            self.disentanglement_loss = self.loss_module.DistanceCorrelationLoss()
        elif disentanglement == 'orthogonal':
            self.disentanglement_loss = self.loss_module.OrthogonalLoss()
        else:
            self.disentanglement_loss = self.loss_module.HSICLoss()
        self.disentanglement_name = disentanglement
        if prototypes is None:
            self.embedding = None
        else:
            self.embedding = PantherSlideEmbedding(
                self.path_dim, prototypes, n_proto=n_proto, em_iter=em_iter,
                tau=tau, ot_eps=ot_eps, fix_proto=fix_proto)

    def attach_prototypes(self, prototypes):
        """Install training-fold prototypes for the slide embedding."""
        if prototypes.shape != (self.n_proto, self.path_dim):
            raise ValueError('Prototype array shape does not match n_proto and the feature width')
        self.embedding = PantherSlideEmbedding(
            self.path_dim, prototypes, n_proto=self.n_proto)
        return self

    def _pathway_inputs(self, rna):
        if (not isinstance(rna, torch.Tensor) or rna.ndim != 2 or len(rna) < 1
                or rna.shape[1] != self.rna_dim or not rna.is_floating_point()
                or not torch.isfinite(rna).all()):
            raise ValueError('RNA must be a finite nonempty [B,rna_dim] tensor')
        return [rna[:, group] for group in self.omic_indices]

    def forward(self, patches, rna, return_attn=False):
        """Return discrete-hazard logits and the upstream disentangled views."""
        if isinstance(patches, torch.Tensor):
            if patches.ndim != 3 or patches.shape[0] != len(rna) or patches.shape[2] != self.path_dim:
                raise ValueError('Patch tensor must be [B,N,path_dim]')
            bags = list(patches.unbind(0))
        elif isinstance(patches, (list, tuple)) and len(patches) == len(rna):
            bags = list(patches)
        else:
            raise ValueError('Provide one patch bag per RNA row')
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        rna = rna.to(device=device, dtype=dtype)
        pathway_inputs = self._pathway_inputs(rna)
        if self.embedding is None:
            raise ValueError('Training-fold prototypes must be attached before the forward pass')
        with torch.no_grad():
            wsi_embedding = self.embedding([bag.to(device=device, dtype=dtype)
                                            for bag in bags])
        wsi_embedding = wsi_embedding.detach()
        output = self.model.forward_mm_no_loss(wsi_embedding, pathway_inputs, return_attn)
        return {'logits': output['logits'], 'wsi_repr': output['wsi_repr'],
                'rna_repr': output['rna_repr'], 'wsi_rna_repr': output['wsi_rna_repr'],
                'rna_wsi_repr': output['rna_wsi_repr']}

    def auxiliary_losses(self, output):
        """Upstream disentanglement loss over the four representations."""
        value, _ = self.disentanglement_loss(
            uni_repr_rna=output['rna_repr'], uni_repr_wsi=output['wsi_repr'],
            uni_repr_rna_wsi=output['rna_wsi_repr'], uni_repr_wsi_rna=output['wsi_rna_repr'])
        return {'disentanglement': float(value)}

    def adaptation_metadata(self):
        return {'method': METHOD_NAME, 'official_model_reproduction': False,
                'upstream_url': UPSTREAM_URL, 'upstream_commit': UPSTREAM_COMMIT,
                'upstream_modules_executed': ['DIMAF', 'CrossAttentionLayer', 'MultiSNN',
                                              'SNN_Block', 'PANTHERBase', 'DirNIWNet',
                                              'mog_eval', 'DistanceCorrelationLoss',
                                              'OrthogonalLoss', 'HSICLoss',
                                              'kmeans_clustering'],
                'modalities': ['pathology', 'RNA'],
                'auxiliary_weights': dict(self.auxiliary_weights),
                'disentanglement_loss': self.disentanglement_name,
                'adaptations': [
                    'Prototypes are clustered on the training rows of the current fold only; the upstream prototype pickle is not used.',
                    'The upstream slide-embedding step runs over the protocol patch bags instead of the upstream H5 dataloader.',
                    'The survival objective is the shared patientwise discrete NLL; the upstream disentanglement loss is retained as an auxiliary objective.',
                    'The prototype clustering sample is drawn with the upstream '
                    'torch.randperm rule but with '
                    f'{DIMAF_PROTOTYPE_PATCHES_PER_PROTO} rows per prototype instead of '
                    'the upstream default of 100000, and the rows are read from the '
                    'bags in bounded batches instead of one dense copy of every slide.',
                ]}
