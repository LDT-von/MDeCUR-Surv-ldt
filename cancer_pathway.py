"""Pathway-aware pathology/RNA survival model and local correlation objectives.

Pathway-indexed pathology regions are constructed from pathology alone for the
local auxiliary target.  A separate RNA-conditioned soft correspondence is used
for prediction.  Neither is a spatial-transcriptomic ground-truth measurement.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from cancer_model import CancerDeCUR


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_pathway_mapping(genes_path, gmt_path, rna_dim, min_genes=5):
    """Map GMT gene symbols exactly to the explicit ordered RNA vector schema.

    Filtering depends only on named gene sets and the input schema, not on any
    expression values, patient labels, validation metrics, or model output.
    """
    if type(rna_dim) is not int or rna_dim < 1 or type(min_genes) is not int or min_genes < 1:
        raise ValueError('Positive RNA dimension and minimum pathway gene count are required')
    genes = Path(genes_path).read_text(encoding='utf-8-sig').splitlines()
    genes = [gene.strip() for gene in genes]
    if len(genes) != rna_dim or any(not gene for gene in genes) or len(set(genes)) != len(genes):
        raise ValueError('Ordered genes must have exactly RNA-dimension unique nonempty names')
    index = {gene: i for i, gene in enumerate(genes)}
    records, seen, skipped = [], set(), []
    for number, line in enumerate(Path(gmt_path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split('\t')]
        if len(parts) < 3 or not parts[0] or not parts[1] or any(not gene for gene in parts[2:]):
            raise ValueError(f'Malformed GMT row {number}; expected name, ID/description and genes')
        name, ident = parts[:2]
        if ident in seen:
            raise ValueError(f'Duplicate pathway ID: {ident}')
        seen.add(ident)
        members = list(dict.fromkeys(parts[2:]))
        matched = sorted(index[gene] for gene in members if gene in index)
        item = {'name': name, 'id': ident, 'source_genes': len(members),
                'matched_genes': len(matched), 'coverage': len(matched)/len(members)}
        if len(matched) < min_genes:
            skipped.append(item)
        else:
            item['indices'] = matched
            records.append(item)
    if not records:
        raise ValueError('No pathway has enough exact gene-symbol matches')
    return {'gene_sha256': _file_sha256(genes_path), 'pathway_sha256': _file_sha256(gmt_path),
            'ordered_gene_symbols': genes,
            'rna_dim': rna_dim, 'minimum_matched_genes': min_genes,
            'gene_identifier_matching': 'exact', 'overlap_between_pathways': 'allowed',
            'pathways': records, 'skipped_pathways': skipped}


class GenePathwayTokenizer(nn.Module):
    """A separate, trainable encoder for the predefined genes of each pathway."""
    def __init__(self, rna_dim, pathway_indices, hidden):
        super().__init__()
        if not pathway_indices:
            raise ValueError('At least one pathway is required')
        self.rna_dim = rna_dim
        self.count = len(pathway_indices)
        self.encoders = nn.ModuleList()
        for i, members in enumerate(pathway_indices):
            if (not members or len(set(members)) != len(members)
                    or any(type(n) is not int or n < 0 or n >= rna_dim for n in members)):
                raise ValueError('Pathway indices must be unique and within RNA vector bounds')
            self.register_buffer(f'indices_{i}', torch.tensor(members, dtype=torch.long))
            self.encoders.append(nn.Sequential(nn.Linear(len(members), hidden), nn.LayerNorm(hidden), nn.GELU()))

    def forward(self, rna):
        if rna.ndim != 2 or rna.shape[1] != self.rna_dim:
            raise ValueError('Pathway tokenizer requires [patients, ordered RNA genes]')
        return torch.stack([encoder(rna.index_select(1, getattr(self, f'indices_{i}')))
                            for i, encoder in enumerate(self.encoders)], dim=1)


class PathwayLocalSurvival(nn.Module):
    """Patient/global DeCUR plus pathway-indexed local decoupling and fusion."""
    def __init__(self, path_dim, rna_dim, pathway_indices, hidden=16, common=8,
                 local_dim=8, local_common=4, bins=1, interaction_strength=.2,
                 match_temperature=.2):
        super().__init__()
        if (type(hidden) is not int or hidden < 2 or type(common) is not int or not 0 < common < hidden
                or type(local_dim) is not int or type(local_common) is not int
                or not 0 < local_common < local_dim or type(bins) is not int or bins < 1
                or not math.isfinite(interaction_strength) or interaction_strength < 0
                or not math.isfinite(match_temperature) or match_temperature <= 0):
            raise ValueError('Invalid global/local dimensions or interaction/matching configuration')
        self.hidden, self.common = hidden, common
        self.local_dim, self.local_common = local_dim, local_common
        self.pathway_count = len(pathway_indices)
        self.interaction_strength, self.match_temperature = float(interaction_strength), float(match_temperature)
        self.base = CancerDeCUR(path_dim, rna_dim, hidden=hidden, common=common,
                                bins=bins, prediction_space='encoder', local_dim=local_dim)
        self.pathways = GenePathwayTokenizer(rna_dim, pathway_indices, hidden)
        # These queries have one shared parameter per pathway across all
        # patients; they do not receive patient RNA in the auxiliary branch.
        self.pathway_queries = nn.Parameter(torch.randn(self.pathway_count, hidden)*.02)
        # Prediction matching has separate parameters from auxiliary projectors.
        self.match_path = nn.Linear(hidden, local_common, bias=False)
        self.match_rna = nn.Linear(hidden, local_common, bias=False)
        self.local_path_projector = nn.Linear(hidden, local_dim)
        self.local_rna_projector = nn.Linear(hidden, local_dim)
        self.interaction_path = nn.Linear(hidden, hidden)
        self.interaction_rna = nn.Linear(hidden, hidden)
        self.interaction_head = nn.Linear(hidden, bins, bias=False)

    def match_queries(self, rna_tokens):
        """Prediction query for RNA-conditioned soft patch matching."""
        return F.normalize(self.match_rna(rna_tokens), dim=-1, eps=1e-6)

    def prediction(self, pooled_path, global_rna, matched_pathways, rna_tokens):
        if matched_pathways.shape != rna_tokens.shape:
            raise ValueError('Matched pathology and RNA pathway tokens must agree in shape')
        base = self.base.head(torch.cat((pooled_path, global_rna), dim=-1))
        interaction = (F.gelu(self.interaction_path(matched_pathways))
                       * F.gelu(self.interaction_rna(rna_tokens))).mean(dim=1)
        extra = self.interaction_head(interaction)
        return base+self.interaction_strength*extra, extra

    def local_projection(self, pathology_regions, rna_tokens):
        if pathology_regions.shape != rna_tokens.shape:
            raise ValueError('One pathology region per RNA pathway is required')
        return self.local_path_projector(pathology_regions), self.local_rna_projector(rna_tokens)


def pathway_correlation_loss(path, rna, target, eps=1e-4, off_weight=.005):
    """Per-pathway patient correlation, averaged over pathway identities.

    The patient axis B is the only independent sampling axis; pathways and
    patches are never flattened into extra independent patient observations.
    """
    if path.ndim != 3 or path.shape != rna.shape or path.shape[0] < 2 or path.shape[-1] < 1:
        raise ValueError('Expected matching [B,K,D] pathway features with B>=2')
    if target not in (0, 1) or not torch.isfinite(path).all() or not torch.isfinite(rna).all():
        raise ValueError('Correlation target or features are invalid')
    pc, rc = path-path.mean(0, keepdim=True), rna-rna.mean(0, keepdim=True)
    pv, rv = pc.square().mean(0), rc.square().mean(0)
    pn, rn = pc/(pv+eps).sqrt(), rc/(rv+eps).sqrt()
    corr = torch.einsum('bkd,bke->kde', pn, rn)/len(path)
    diag = corr.diagonal(dim1=-2, dim2=-1)
    off = corr-torch.diag_embed(diag)
    loss = ((diag-target).square().sum(-1)+off_weight*off.square().sum((-2, -1))).mean()
    info = {'pathways': path.shape[1], 'patients': path.shape[0],
            'mean_diagonal': diag.detach().mean().item(),
            'low_variance_coordinates': int(((pv < eps) | (rv < eps)).sum().item())}
    return loss, info
