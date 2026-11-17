"""Read-only, synthetic diagnosis of failed cluster cross-fit groups."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cancer_morphid_model import FrozenMorphologyBank
from method3_1_independent_synthetic import make_independent_splits
from method3_1_model import Method31
from morphid_probe import patient_fold_ids


def diagnose(seed):
    data, _, _ = make_independent_splits(seed, 'heteroscedastic_null', n_train=160, n_val=600, n_test=360)
    train = data['train']
    bank = FrozenMorphologyBank.fit(train['bags'], ids=train['ids'], clusters=2, latent_dim=3,
                                    max_fit_patches=2560, per_patient_cap=16, seed=seed)
    model = Method31(bank, rna_dim=2, projection_dim=4, bins=4, seed=seed)
    ids, encoded, _ = model._input(train['bags'], train['rna'], train['ids'])
    assignments = patient_fold_ids(ids, 4, seed)
    counts = []
    for k in range(model.k):
        selected = torch.where(encoded['available'][:, k])[0]
        counts.append({'cluster': k, 'available_patients': len(selected),
                       'fold_counts': torch.bincount(assignments[selected], minlength=4).tolist()})
    return {'seed': seed, 'min_effective_mass': model.min_effective_mass, 'clusters': counts}


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(json.dumps([diagnose(seed) for seed in (1815, 1820)], indent=2))
