"""Adapt externally supplied TRAINING-only cluster labels to Method 3.1.

This does not run Leiden and centroid/softmax assignment is not HPL's scanpy
ingest.  It lets the research prototype use a fixed externally computed map
instead of its optional PCA/k-means surrogate. Inputs are supplied in memory.
"""
import torch

from cancer_morphid_model import FrozenMorphologyBank


def bank_from_training_cluster_labels(bags, labels, ids, *, latent_dim=8,
                                      label_provenance, temperature=1., min_mass=1e-4):
    bags, labels, ids = list(bags), list(labels), tuple(map(str, ids))
    if not bags or len(labels) != len(bags) or len(ids) != len(bags) or len(set(ids)) != len(ids):
        raise ValueError('Each training patient needs a unique ID, bag and label vector')
    if not isinstance(label_provenance, str) or not label_provenance.strip():
        raise ValueError('Explicit source of the external training labels is required')
    if type(latent_dim) is not int or latent_dim < 1:
        raise ValueError('latent_dim must be a positive integer')
    rows, all_labels = [], []
    for bag, lab in zip(bags, labels):
        bag = torch.as_tensor(bag, dtype=torch.float64).detach().cpu()
        lab = torch.as_tensor(lab).detach().cpu()
        if bag.ndim != 2 or not len(bag) or not torch.isfinite(bag).all():
            raise ValueError('Each bag must be a finite nonempty matrix without padding')
        if lab.ndim != 1 or len(lab) != len(bag) or lab.dtype == torch.bool or lab.is_complex():
            raise ValueError('One integer cluster label is required per patch')
        if not torch.isfinite(lab).all() or not torch.equal(lab, lab.long().to(lab.dtype)):
            raise ValueError('Cluster labels must be finite integers')
        rows.append(bag); all_labels.append(lab.long())
    x, y = torch.cat(rows), torch.cat(all_labels)
    keys = sorted(set(y.tolist()))
    if len(keys) < 2 or keys != list(range(len(keys))):
        raise ValueError('Cluster labels must be contiguous integers from zero with at least two clusters')
    mean = x.mean(0)
    scale = x.std(0, unbiased=False)
    scale = torch.where(scale >= 1e-8, scale, torch.ones_like(scale))
    normalized = (x-mean)/scale
    _, _, vh = torch.linalg.svd(normalized, full_matrices=False)
    d = min(latent_dim, x.shape[1], len(x)-1)
    components = vh[:d].T.contiguous()
    anchors = components.abs().argmax(0)
    signs = components[anchors, torch.arange(d)].sign()
    components *= torch.where(signs == 0, torch.ones_like(signs), signs)
    z = normalized @ components
    centers = torch.stack([z[y == j].mean(0) for j in keys])
    return FrozenMorphologyBank(centers, mean=mean, scale=scale, components=components,
        temperature=temperature, min_mass=min_mass,
        metadata={'source': 'external_training_cluster_labels_nearest_centroid',
                  'fit_patient_ids': list(ids), 'label_provenance_as_supplied': label_provenance,
                  'assignment': 'softmax over squared centroid distance, not HPL ingest',
                  'training_patch_counts': torch.bincount(y).tolist(),
                  'outcomes_used': False, 'hpl_replication': False})
