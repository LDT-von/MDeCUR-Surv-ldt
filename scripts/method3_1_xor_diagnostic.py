"""Post-hoc XOR representational audit on synthetic generated patients only.

Generator risk and latent p0 are supervised targets of diagnostic probes, not
features supplied to Method31. This is NOT a survival training comparison.
Replays the six controlled-experiment seeds without modifying earlier outputs.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cancer_morphid_model import FrozenMorphologyBank  # noqa: E402
from method3_1_model import Method31  # noqa: E402
from scripts.method3_1_synthetic import sample  # noqa: E402

SEEDS = (211, 223, 227, 229, 233, 239)
SPLIT_SIZES = (('train', 80), ('validation', 80), ('test', 240))


def probe(train_x, test_x, train_truth, test_truth):
    """Standardized least-squares truth probe; inputs/targets synthetic only."""
    mean, scale = train_x.mean(0), train_x.std(0)
    scale = np.where(scale < 1e-8, 1., scale)
    x = np.column_stack([np.ones(len(train_x)), (train_x-mean)/scale])
    t = np.column_stack([np.ones(len(test_x)), (test_x-mean)/scale])
    prediction = t @ (np.linalg.pinv(x) @ train_truth)
    denominator = float(np.sum((test_truth-test_truth.mean())**2))
    r2 = 1-float(np.sum((test_truth-prediction)**2))/denominator
    return {'test_r2': r2,
            'test_sign_accuracy': float(np.mean(np.sign(prediction) == np.sign(test_truth))),
            'test_pearson': float(np.corrcoef(prediction, test_truth)[0, 1]),
            'train_probe_dimension': int(train_x.shape[1])}


def outer(path, rna):
    return (path[:, :, None]*rna[:, None, :]).reshape(len(path), -1)


def run(seed, prior):
    rng, parts, truth = np.random.default_rng(seed), {}, {}
    for name, n in SPLIT_SIZES:
        # Copy the RNG state to retrieve the exact latent variables generated
        # by sample(), without modifying the data generator or input streams.
        shadow = np.random.default_rng()
        shadow.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        p, g = shadow.normal(size=(n, 2)), shadow.normal(size=(n, 2))
        parts[name] = sample(rng, n, 'xor', f'{seed}-{name}')
        if not np.array_equal(g, parts[name]['rna'].numpy()):
            raise RuntimeError('Cannot prove that replayed RNA matches generated split')
        truth[name] = {'risk': 1.4*np.sign(p[:, 0]*g[:, 0]), 'latent_p0': p[:, 0]}
    train, test = parts['train'], parts['test']
    bank = FrozenMorphologyBank.fit(train['bags'], ids=train['ids'], clusters=2, latent_dim=3,
                                    max_fit_patches=1280, per_patient_cap=16, seed=seed)
    model = Method31(bank, rna_dim=2, projection_dim=4, bins=4, seed=seed)
    model.pretrain(train['bags'], train['rna'], train['ids'], epochs=35)
    model.fit_probes(train['bags'], train['rna'], train['ids'], folds=4)
    enc, g, c = model.cached
    x = model._features(enc, g, c).detach().numpy()
    _, enc_test, g_test = model._input(test['bags'], test['rna'], test['ids'])
    c_test = torch.stack([model.probes[str(j)].predict(g_test) for j in range(model.k)], dim=1)
    x_test = model._features(enc_test, g_test, c_test).detach().numpy()
    kd = model.k*model.d
    # Order is RNA, predictable, residual, CLR, mask, SSL outputs, product.
    u, u_test = x[:, model.rna_dim+kd:model.rna_dim+2*kd], x_test[:, model.rna_dim+kd:model.rna_dim+2*kd]
    p = (enc['pooled']*enc['available'].unsqueeze(-1)).numpy().reshape(-1, kd)
    p_test = (enc_test['pooled']*enc_test['available'].unsqueeze(-1)).numpy().reshape(-1, kd)
    rna, rna_test = train['rna'].numpy(), test['rna'].numpy()
    observed_product = np.asarray([float(b[:, 0].mean())*float(r[0])
                                   for b, r in zip(train['bags'], train['rna'])])[:, None]
    observed_product_test = np.asarray([float(b[:, 0].mean())*float(r[0])
                                        for b, r in zip(test['bags'], test['rna'])])[:, None]
    y, y_test = truth['train']['risk'], truth['test']['risk']
    previous = next(r for r in prior['runs'] if r.get('case') == 'xor' and r.get('seed') == seed)
    previous_arms = {r['arm']: {'test_nll': r['test_nll'], 'selection': r['selection']}
                     for r in previous['outcomes']
                     if r['arm'] in ('full', 'no_interaction', 'intercept')}
    return {'seed': seed,
            'truth_probes': {
                'current_diagonal_residual_product': probe(x[:, -kd:], x_test[:, -kd:], y, y_test),
                'observed_patch_mean_times_rna0': probe(observed_product, observed_product_test, y, y_test),
                'sign_of_observed_product': probe(np.sign(observed_product), np.sign(observed_product_test), y, y_test),
                'residual_full_outer_product': probe(outer(u, rna), outer(u_test, rna_test), y, y_test),
                'pathology_full_outer_product': probe(outer(p, rna), outer(p_test, rna_test), y, y_test),
                'pathology_to_latent_p0': probe(p, p_test, truth['train']['latent_p0'], truth['test']['latent_p0'])},
            'train_raw_current_product_sd': x[:, -kd:].std(0).tolist(),
            'test_raw_current_product_sd': x_test[:, -kd:].std(0).tolist(),
            'test_cluster_available_fraction': enc_test['available'].double().mean(0).tolist(),
            'formal_survival_results_as_recorded': previous_arms}


def main():
    torch.set_num_threads(1)
    formal_path = ROOT/'results'/'method3_1_controlled_20261116.json'
    formal = json.loads(formal_path.read_text(encoding='utf8'))
    watched = ['method3_1_model.py', 'morphid_probe.py', 'cancer_morphid_model.py',
               'scripts/method3_1_synthetic.py', 'scripts/method3_1_xor_diagnostic.py']
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in watched}
    result = {'synthetic_only': True, 'scenario': 'xor', 'seeds': list(SEEDS),
              'split_sizes': dict(SPLIT_SIZES), 'generator_risk': '1.4*sign(latent_p0*RNA0)',
              'diagnostic_fit': 'Unpenalized least squares on TRAIN generator truth; train-only standardization; TEST truth audited once.',
              'limitations': ['Generator truth is used as the diagnostic supervised target only; not available to a survival model.',
                  'None of these truth probes is an outcome-supervised survival arm or a fair formal baseline.',
                  'Post-hoc exploratory diagnostic on already inspected test distributions; do not claim independent confirmation.',
                  'Visible signed product deliberately uses the generator-known first patch and RNA coordinates; not a transferable discovery.',
                  'Unpenalized diagnostic probes have different input capacities; full outer products use more features.',
                  'Outer-trained morphology and SSL are shared when the KRR probe cross-fits; no patient data are used.'],
              'source_hashes': hashes,
              'formal_result_sha256': hashlib.sha256(formal_path.read_bytes()).hexdigest(),
              'runs': [run(seed, formal) for seed in SEEDS]}
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != h for name, h in hashes.items()):
        raise RuntimeError('Source changed while the diagnostic was running')
    output = ROOT/'results'/'method3_1_xor_diagnostic.json'
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing diagnostic result')
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    print(f'Wrote {output}')
    for row in result['runs']:
        values = row['truth_probes']
        print(row['seed'], 'current_r2', round(values['current_diagonal_residual_product']['test_r2'], 3),
              'visible_r2', round(values['observed_patch_mean_times_rna0']['test_r2'], 3),
              'visible_sign_r2', round(values['sign_of_observed_product']['test_r2'], 3))


if __name__ == '__main__':
    main()
