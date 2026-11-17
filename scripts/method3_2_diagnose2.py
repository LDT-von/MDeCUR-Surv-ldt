"""Audit: does the atlas cluster assignment align with the generator clusters?"""
import os
import sys

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from method3_2_model import MorphologyAtlas
from method3_2_synthetic_data import CLUSTERS, generate_cohort

train = generate_cohort('redundant', 160, seed=1011, prefix='diag_train')
atlas = MorphologyAtlas.fit(train.bags, train.ids, clusters=CLUSTERS, latent_dim=8,
                             per_patient_cap=64, seed=11)
rna = train.rna
z = train.truth['z']

pred, true = [], []
for bag, labels in zip(train.bags, train.truth['cluster_labels']):
    raw = bag if isinstance(bag, torch.Tensor) else bag[0]
    zz = ((raw - atlas.patch_mean) / atlas.patch_scale) @ atlas.components
    pred.append(torch.cdist(zz, atlas.centers).argmin(1).numpy())
    true.append(np.asarray(labels))
print('ARI:', round(float(adjusted_rand_score(np.concatenate(true), np.concatenate(pred))), 3))
confusion = np.zeros((CLUSTERS, CLUSTERS), dtype=int)
for p, t in zip(pred, true):
    for a, b in zip(p, t):
        confusion[a, b] += 1
print('confusion (atlas row, generator col):')
print(confusion)

patients = atlas.encode(train.bags, train.ids)
P = torch.stack([p.p for p in patients]).numpy()
for k in range(CLUSTERS):
    X = P[:, k, :]
    coef, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], rna[:, 0].numpy(), rcond=None)
    pred_r = np.c_[X, np.ones(len(X))] @ coef
    ss = 1 - ((pred_r - rna[:, 0].numpy()) ** 2).sum() / ((rna[:, 0].numpy() - rna[:, 0].numpy().mean()) ** 2).sum()
    print(f'atlas cluster {k}: linear p_k -> rna0 in-sample R2 = {ss:.3f}')
print('patch feature variance per coordinate:',
      np.round(np.concatenate([b.numpy() for b in train.bags]).var(0), 2))
