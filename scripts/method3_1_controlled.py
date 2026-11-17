"""Pre-registered CPU synthetic controls; no real datasets are opened."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cancer_morphid_model import FrozenMorphologyBank, discrete_nll  # noqa: E402
from method3_1_model import Method31  # noqa: E402
from method3_1_nature_control import NatureCompositionControl  # noqa: E402
from method3_1_regularized import RegularizedMethod31, training_hazard_logits  # noqa: E402
from method3_1_synthetic_data import make_composition_splits  # noqa: E402
from scripts.method3_1_synthetic import sample, retrieval  # noqa: E402

SEEDS = (211, 223, 227, 229, 233, 239)
CASES = ('shared', 'independent_no_signal', 'rna_only', 'pathology_only', 'xor',
         'composition_signal', 'composition_null')
ARMS = ('original_60_steps', 'full', 'no_interaction', 'no_residual',
        'cross_off', 'nature_clr', 'intercept')
TRAIN, VALIDATION, TEST = 80, 80, 240


def nll(logits, data):
    return float(discrete_nll(logits, data['labels'], data['censorship'], alpha=0.))


def generate(case, seed):
    if case.startswith('composition_'):
        parts, truth, metadata = make_composition_splits(seed, n_train=TRAIN, n_val=VALIDATION,
            n_test=TEST, composition_effect=2.5 if case == 'composition_signal' else 0.)
        return parts, truth, metadata
    rng = np.random.default_rng(seed)
    parts = {name: sample(rng, count, case, f'{seed}-{name}') for name, count in
             (('train', TRAIN), ('validation', VALIDATION), ('test', TEST))}
    return parts, None, {'source': 'previous synthetic generator', 'case': case}


def composition_audit(bank, parts, truth):
    # Infer the cluster-label permutation exclusively for audit using training
    # truth, after the bank is fitted. Never use it to train any model.
    correct = [0, 0]
    for bag, true in zip(parts['train']['bags'], truth['train']['true_cluster_labels_audit_only']):
        z = ((bag-bank.patch_mean)/bank.patch_scale) @ bank.components
        pred = torch.cdist(z, bank.centers).argmin(1).numpy()
        correct[0] += int(np.sum(pred == true))
        correct[1] += int(np.sum(1-pred == true))
    invert = correct[1] > correct[0]
    with torch.no_grad():
        pi = bank.encode(parts['test']['bags'])['composition'].numpy()
    estimate = pi[:, 0 if invert else 1]
    actual = truth['test']['true_cluster1_fraction_audit_only']
    return {'training_patch_mapping_accuracy': max(correct)/sum(correct),
            'test_soft_fraction_mae': float(np.mean(np.abs(estimate-actual))),
            'test_soft_fraction_pearson': float(np.corrcoef(estimate, actual)[0, 1]),
            'truth_used_only_after_fitting': True}


def arm_record(arm, train_nll, val_nll, test_nll, selection=None, parameters=None):
    return {'arm': arm, 'train_nll': train_nll, 'validation_nll': val_nll,
            'test_nll': test_nll, 'selection': selection, 'parameter_count': parameters}


def run_one(case, seed):
    parts, truth, metadata = generate(case, seed)
    tr, va, te = (parts[x] for x in ('train', 'validation', 'test'))
    sets = [set(x['ids']) for x in (tr, va, te)]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    bank = FrozenMorphologyBank.fit(tr['bags'], ids=tr['ids'], clusters=2, latent_dim=3,
                                    max_fit_patches=1280, per_patient_cap=16, seed=seed)
    backbone = Method31(bank, rna_dim=2, projection_dim=4, bins=4, seed=seed)
    backbone.pretrain(tr['bags'], tr['rna'], tr['ids'], epochs=35)
    backbone.fit_probes(tr['bags'], tr['rna'], tr['ids'], folds=4)
    outcomes = []
    old = copy.deepcopy(backbone)
    old.fit_survival(tr['labels'], tr['censorship'], tr['ids'], epochs=60)
    with torch.no_grad():
        outcomes.append(arm_record('original_60_steps', nll(old._logits(*old.cached), tr),
            nll(old.predict(va['bags'], va['rna'], va['ids']), va),
            nll(old.predict(te['bags'], te['rna'], te['ids']), te),
            parameters=sum(p.numel() for p in old.head.parameters())))
    for name in ('full', 'no_interaction', 'no_residual'):
        model = RegularizedMethod31(backbone, arm=name)
        selection = model.fit(tr['labels'], tr['censorship'], tr['ids'],
            va['bags'], va['rna'], va['ids'], va['labels'], va['censorship'])
        outcomes.append(arm_record(name, nll(model.train_logits(), tr), selection['validation_nll'],
            nll(model.predict(te['bags'], te['rna'], te['ids']), te), selection,
            int(model.mask.sum())*backbone.bins+backbone.bins))
    off = Method31(copy.deepcopy(bank), rna_dim=2, projection_dim=4, bins=4, seed=seed)
    off.pretrain(tr['bags'], tr['rna'], tr['ids'], epochs=35, cross_weight=0.)
    off.fit_probes(tr['bags'], tr['rna'], tr['ids'], folds=4)
    off_head = RegularizedMethod31(off, arm='full')
    selection = off_head.fit(tr['labels'], tr['censorship'], tr['ids'],
        va['bags'], va['rna'], va['ids'], va['labels'], va['censorship'])
    outcomes.append(arm_record('cross_off', nll(off_head.train_logits(), tr), selection['validation_nll'],
        nll(off_head.predict(te['bags'], te['rna'], te['ids']), te), selection,
        int(off_head.mask.sum())*off.bins+off.bins))
    nature = NatureCompositionControl(copy.deepcopy(bank), bins=4)
    nature.fit(tr['bags'], tr['labels'], tr['censorship'], tr['ids'],
        validation=(va['bags'], va['labels'], va['censorship'], va['ids']))
    with torch.no_grad():
        clr, _ = nature._encode(tr['bags'], tr['ids'])
        x = (clr-nature.clr_mean)/nature.clr_scale
        train_nll = nll(nature.head(x), tr)
    outcomes.append(arm_record('nature_clr', train_nll, nature.selection['best']['validation_nll'],
        nll(nature.predict(te['bags'], te['ids']), te), nature.selection,
        sum(p.numel() for p in nature.head.parameters())))
    intercept = training_hazard_logits(tr['labels'], tr['censorship'], 4)
    outcomes.append(arm_record('intercept', *(nll(intercept.expand(len(d['ids']), -1), d)
        for d in (tr, va, te)), selection={'rule': 'training smoothed empirical bin hazards'}, parameters=4))
    return {'case': case, 'seed': seed, 'split_events': {
            name: int((1-parts[name]['censorship']).sum()) for name in parts},
            'data_metadata': metadata, 'outcomes': outcomes,
            'pair_retrieval_test': retrieval(backbone, te),
            'cross_off_retrieval_test': retrieval(off, te),
            'composition_audit': composition_audit(bank, parts, truth) if truth is not None else None}


def main():
    torch.set_num_threads(1)
    watched = ('method3_1_model.py', 'method3_1_regularized.py', 'method3_1_nature_control.py',
               'method3_1_synthetic_data.py', 'scripts/method3_1_synthetic.py',
               'scripts/method3_1_controlled.py', 'METHOD3_1_CONTROLS_PREREG_20261116.md')
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in watched}
    records = []
    for case in CASES:
        for seed in SEEDS:
            try:
                row = run_one(case, seed)
                outcomes = {r['arm']: r['test_nll'] for r in row['outcomes']}
                print(f"{case} {seed} full={outcomes['full']:.3f} clr={outcomes['nature_clr']:.3f} "
                      f"null={outcomes['intercept']:.3f}", flush=True)
            except Exception:
                row = {'case': case, 'seed': seed, 'failure': traceback.format_exc()}
                print(f'{case} {seed} FAILED: {row["failure"]}', flush=True)
            records.append(row)
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError('Source changed while experiment was running; output is invalid')
    summary = {}
    for case in CASES:
        summary[case] = {}
        rows = [r for r in records if r['case'] == case and 'failure' not in r]
        for arm in ARMS:
            differences, values = [], []
            for row in rows:
                vals = {v['arm']: v['test_nll'] for v in row['outcomes']}
                values.append(vals[arm]); differences.append(vals[arm]-vals['intercept'])
            ci = None
            if len(differences) > 1:
                center = float(np.mean(differences))
                radius = float(stats.t.ppf(.975, len(differences)-1)*stats.sem(differences))
                ci = [center-radius, center+radius]
            summary[case][arm] = {'n_seeds': len(values),
                'test_nll_mean': float(np.mean(values)) if values else None,
                'paired_minus_intercept_mean': float(np.mean(differences)) if differences else None,
                'paired_minus_intercept_t95_interval': ci,
                'wins_vs_intercept': sum(x < 0 for x in differences)}
    result = {'synthetic_only': True, 'seeds': list(SEEDS), 'cases': list(CASES),
              'arms': list(ARMS), 'split_sizes': [TRAIN, VALIDATION, TEST], 'hashes': hashes,
              'protocol': 'METHOD3_1_CONTROLS_PREREG_20261116.md',
              'failures': sum('failure' in r for r in records), 'summary': summary, 'runs': records}
    path = ROOT/'results'/'method3_1_controlled_20261116.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    print(f'Wrote {path}; failures={result["failures"]}', flush=True)


if __name__ == '__main__':
    main()
