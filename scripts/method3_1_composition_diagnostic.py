"""Post-hoc generated-only diagnostic of CLR information redundancy.

Replays previously inspected controlled seeds. The secondary Ridge probe has
patient-wise held-out folds and fits its scaler within each fold. The upstream
bank, SSL and OOF nonlinear probes are shared outer-training fits, and are NOT
refit inside the secondary Ridge folds. Thus this is conditional OOF diagnostic
evidence, not strict fully nested evaluation or an independent test.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cancer_morphid_model import FrozenMorphologyBank, discrete_nll  # noqa: E402
from method3_1_model import Method31  # noqa: E402
from method3_1_nature_control import NatureCompositionControl  # noqa: E402
from method3_1_regularized import RegularizedMethod31  # noqa: E402
from method3_1_synthetic_data import make_composition_splits  # noqa: E402

SEEDS = (211, 223, 227)
FORMAL_RESULT = ROOT/'results'/'method3_1_controlled_20261116.json'
OUTPUT = ROOT/'results'/'method3_1_composition_diagnostic.json'
WATCHED = ('method3_1_model.py', 'method3_1_regularized.py', 'method3_1_nature_control.py',
           'method3_1_synthetic_data.py', 'cancer_morphid_model.py', 'morphid_probe.py',
           'cancer_parity.py', 'scripts/method3_1_composition_diagnostic.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nll(logits, data):
    return float(discrete_nll(logits, data['labels'], data['censorship'], alpha=0.))


def model_fit(seed):
    parts, truth, _ = make_composition_splits(seed, n_train=80, n_val=80, n_test=240)
    tr, va, te = (parts[x] for x in ('train', 'validation', 'test'))
    assert not set(tr['ids']) & (set(va['ids']) | set(te['ids']))
    assert not set(va['ids']) & set(te['ids'])
    bank = FrozenMorphologyBank.fit(tr['bags'], ids=tr['ids'], clusters=2, latent_dim=3,
                                    max_fit_patches=1280, per_patient_cap=16, seed=seed)
    backbone = Method31(bank, rna_dim=2, projection_dim=4, bins=4, seed=seed)
    backbone.pretrain(tr['bags'], tr['rna'], tr['ids'], epochs=35)
    backbone.fit_probes(tr['bags'], tr['rna'], tr['ids'], folds=4)
    fit_args = (tr['labels'], tr['censorship'], tr['ids'], va['bags'], va['rna'],
                va['ids'], va['labels'], va['censorship'])
    full = RegularizedMethod31(backbone, arm='full')
    full.fit(*fit_args)
    clr_only = RegularizedMethod31(backbone, arm='full')
    # Feature order is RNA(2), predictable(6), residual(6), CLR(2), ...
    # Use explicit dimensions to avoid assuming the numerical slice 14:16.
    begin = backbone.rna_dim + 2*backbone.k*backbone.d
    end = begin+backbone.k
    clr_only.mask.zero_()
    clr_only.mask[begin:end] = 1
    clr_only.fit(*fit_args)
    nature = NatureCompositionControl(copy.deepcopy(bank), bins=4)
    nature.fit(tr['bags'], tr['labels'], tr['censorship'], tr['ids'],
               validation=(va['bags'], va['labels'], va['censorship'], va['ids']))

    raw = full._train_features().numpy()
    target = raw[:, begin]
    other = np.concatenate((raw[:, :begin], raw[:, end:]), axis=1)
    folds = KFold(4, shuffle=True, random_state=seed)
    probe = make_pipeline(StandardScaler(), Ridge(alpha=1.))
    estimated = cross_val_predict(probe, other, target, cv=folds)
    denominator = np.sum((target-target.mean())**2)
    if denominator == 0:
        raise ValueError('Generated CLR is constant; R² would be undefined')
    oof_r2 = float(1-np.sum((estimated-target)**2)/denominator)
    rna0 = tr['rna'][:, 0].numpy()
    corr = float(np.corrcoef(target, rna0)[0, 1])
    with torch.no_grad():
        full_logits = full.predict(te['bags'], te['rna'], te['ids'])
        clr_logits = clr_only.predict(te['bags'], te['rna'], te['ids'])
        nature_logits = nature.predict(te['bags'], te['ids'])
    true_risk = truth['test']['generating_risk_audit_only']
    full_corr = float(np.corrcoef(full_logits.mean(1).numpy(), true_risk)[0, 1])
    nature_corr = float(np.corrcoef(nature_logits.mean(1).numpy(), true_risk)[0, 1])
    difference = float((clr_logits-nature_logits).abs().max())
    return {'seed': seed, 'n': {'train': len(tr['ids']), 'validation': len(va['ids']),
                                'test': len(te['ids'])},
            'non_clr_to_clr': {'patientwise_4fold_r2': oof_r2, 'ridge_alpha': 1.,
                'rna0_clr_pearson': corr, 'probe_input_width': other.shape[1],
                'outer_train_shared_representation': True,
                'secondary_scaler_fit_within_each_probe_fold': True},
            'full': {'selection': copy.deepcopy(full.selection), 'test_nll': nll(full_logits, te),
                     'test_logit_mean_correlation_to_generating_risk': full_corr},
            'clr_masked_regularized': {'selection': copy.deepcopy(clr_only.selection),
                                       'test_nll': nll(clr_logits, te)},
            'nature_clr': {'selection': copy.deepcopy(nature.selection), 'test_nll': nll(nature_logits, te),
                           'test_logit_mean_correlation_to_generating_risk': nature_corr},
            'matched_clr_test_max_abs_logit_difference': difference,
            'parity_within_1e_minus_10': difference < 1e-10}


def main():
    if OUTPUT.exists():
        raise FileExistsError('Refusing to overwrite the existing composition diagnostic')
    torch.set_num_threads(1)
    formal_before = sha(FORMAL_RESULT)
    hashes = {name: sha(ROOT/name) for name in WATCHED}
    result = {'synthetic_only': True, 'case': 'composition_signal', 'seeds': list(SEEDS),
              'status': 'post-hoc diagnosis of previously inspected test seeds',
              'source_hashes': hashes, 'formal_result_sha256': formal_before,
              'limitations': ['The generated truth is only an audit target and is not a feature or training label.',
                'The secondary Ridge patient folds reuse an outer-training bank, SSL, and cross-fitted probes; '
                'the feature generation is not strictly nested in the secondary probe folds.',
                'RNA shares the composition latent factor and thus can proxy risk even without a direct outcome coefficient.',
                'CLR direction can flip with unlabeled cluster index permutation.',
                'These inspected seeds and comparisons do not provide independent confirmation, clinical benefit, or causality.'],
              'runs': [model_fit(seed) for seed in SEEDS]}
    if sha(FORMAL_RESULT) != formal_before or any(sha(ROOT/name) != value for name, value in hashes.items()):
        raise RuntimeError('Source or formal result changed while the diagnostic was running')
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    print(f'Wrote {OUTPUT}')
    for row in result['runs']:
        print('seed', row['seed'], 'conditional_oof_r2', round(row['non_clr_to_clr']['patientwise_4fold_r2'], 4),
              'rna0_corr', round(row['non_clr_to_clr']['rna0_clr_pearson'], 4),
              'clr_logits_max_difference', row['matched_clr_test_max_abs_logit_difference'])


if __name__ == '__main__':
    main()
