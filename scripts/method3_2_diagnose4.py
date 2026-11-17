"""Check per-generator-cluster mean signal against the latent z."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from method3_2_synthetic_data import CLUSTERS, LATENT_AXIS, generate_cohort

train = generate_cohort('redundant', 160, seed=1011, prefix='diag_train')
z = train.truth['z']
for k in range(CLUSTERS):
    means = []
    for bag, labels in zip(train.bags, train.truth['cluster_labels']):
        rows = bag[labels == k]
        means.append(rows[:, LATENT_AXIS].mean() if len(rows) else np.nan)
    means = np.asarray(means)
    ok = ~np.isnan(means)
    print(f'gen cluster {k}: n={ok.sum()} corr(mean axis4, z) = '
          f'{np.corrcoef(means[ok], z[ok])[0, 1]:+.3f}')
print('rna0 vs z:', round(float(np.corrcoef(train.rna[:, 0].numpy(), z)[0, 1]), 3))
