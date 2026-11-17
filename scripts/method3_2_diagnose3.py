"""Debug the atlas k-means behaviour against the generator clusters."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from method3_2_model import MorphologyAtlas, _kmeans
from method3_2_synthetic_data import CLUSTERS, generate_cohort

train = generate_cohort('redundant', 160, seed=1011, prefix='diag_train')
pool = torch.cat([b for b in train.bags], 0)
mean = pool.mean(0)
scale = pool.std(0).clamp_min(1e-8)
std = (pool - mean) / scale
_, sv, vh = torch.linalg.svd(std, full_matrices=False)
print('singular value head:', np.round(sv[:10].numpy(), 2))
components = vh[:8].T
projected = std @ components
centers, repairs = _kmeans(projected, 4, 11)
print('empty cluster repairs:', repairs)
d = torch.cdist(projected, centers)
assign = d.argmin(1)
print('kmeans cluster sizes:', np.bincount(assign.numpy(), minlength=4))
labels = np.concatenate(train.truth['cluster_labels'])
print('generator cluster sizes:', np.bincount(labels, minlength=4))
conf = np.zeros((4, 4), dtype=int)
for a, b in zip(assign.numpy(), labels):
    conf[a, b] += 1
print(conf)

# generator cluster means in PCA space
for k in range(4):
    sel = labels == k
    print('gen', k, 'pca-space mean norm', float(projected[sel].mean(0).norm()),
          'spread', float(projected[sel].std(0).mean()))
print('gen cluster pairwise distance in raw space:')
raw_means = np.stack([pool.numpy()[labels == k].mean(0) for k in range(4)])
print(np.round(np.linalg.norm(raw_means[:, None] - raw_means[None], axis=-1), 2))
