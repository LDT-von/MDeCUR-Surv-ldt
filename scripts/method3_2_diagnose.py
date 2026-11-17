"""Diagnostic for the method 3.2 contrastive stage (no patient data)."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from method3_2_model import (ClusterConditionalEncoder, MorphologyAtlas,
                              cluster_association_auc, cross_fitted_association,
                              train_cluster_contrastive)
from method3_2_synthetic_data import CLUSTERS, generate_cohort

train = generate_cohort('redundant', 160, seed=1011, prefix='diag_train')
atlas = MorphologyAtlas.fit(train.bags, train.ids, clusters=CLUSTERS, latent_dim=8,
                             per_patient_cap=64, seed=11)
patients = atlas.encode(train.bags, train.ids)
rna = train.rna
z = train.truth['z']
P = torch.stack([p.p for p in patients]).numpy()
print('raw mean coord corr with z:',
      [round(float(np.corrcoef(np.stack([b.mean(0).numpy() for b in train.bags])[:, j], z)[0, 1]), 3)
       for j in range(6)])
for k in range(CLUSTERS):
    X = P[:, k, :]
    coef, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], rna[:, 0].numpy(), rcond=None)
    pred = np.c_[X, np.ones(len(X))] @ coef
    ss = 1 - ((pred - rna[:, 0].numpy()) ** 2).sum() / ((rna[:, 0].numpy() - rna[:, 0].numpy().mean()) ** 2).sum()
    print(f'cluster {k}: linear p_k -> rna0 in-sample R2 = {ss:.3f}')

for epochs, lr, batch, hidden, temp in [(60, 1e-2, 32, 8, 0.2),
                                         (200, 1e-2, 32, 8, 0.2),
                                         (200, 1e-2, 32, 8, 0.5),
                                         (400, 3e-2, 32, 16, 0.2)]:
    enc = ClusterConditionalEncoder(atlas.latent_dim, rna.shape[1], atlas.clusters,
                                    hidden=hidden, seed=11)
    hist = train_cluster_contrastive(enc, patients, rna, epochs=epochs, batch_size=batch,
                                     temperature=temp, min_patients=8, lr=lr, seed=11)
    a, stats = cluster_association_auc(enc, patients, rna, candidate_patients=patients,
                                       candidate_rna=rna, min_patients=8, n_shuffles=32)
    ac, statc = cross_fitted_association(
        lambda: ClusterConditionalEncoder(atlas.latent_dim, rna.shape[1], atlas.clusters,
                                          hidden=hidden, seed=11),
        patients, rna, folds=4, seed=11, epochs=epochs, batch_size=batch, temperature=temp,
        min_patients=8, lr=lr)
    print(f'epochs={epochs} lr={lr} h={hidden} T={temp} loss {hist[0]["loss"]:.4f}->{hist[-1]["loss"]:.4f} '
          f'in-sample a={[round(x,3) for x in a.tolist()]} '
          f'cross-fitted a={[round(x,3) for x in ac.tolist()]}')
