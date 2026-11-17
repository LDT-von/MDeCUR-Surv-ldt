"""Small independent-patient synthetic audit of the actual Method31 model.

No real data, outcomes, external paths, or network are accessed. Run with
``python Conditional_DeCUR_Surv/scripts/method3_1_synthetic.py``. This is a
mechanism diagnostic, not clinical validation or proof of mutual information.
"""
import json
import hashlib
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cancer_morphid_model import FrozenMorphologyBank, discrete_nll  # noqa: E402
from method3_1_model import Method31, patient_contrastive  # noqa: E402

ORIGINAL_SEEDS = (23, 47, 73)
NEW_SEEDS = (111, 137, 173)
SEEDS = ORIGINAL_SEEDS + NEW_SEEDS
CASES = ('shared', 'independent_no_signal', 'rna_only', 'pathology_only', 'xor')
ARMS = {'paired': (1.0, False), 'shuffled': (1.0, True), 'cross_off': (0.0, False)}
N_TRAIN, N_TEST = 80, 120


def sample(rng, n, case, prefix):
    p = rng.normal(size=(n, 2))
    g = rng.normal(size=(n, 2))
    if case == 'shared':
        g = p + 0.25 * g
        risk = 1.2 * p[:, 0]
    elif case == 'rna_only':
        risk = 1.2 * g[:, 0]
    elif case == 'pathology_only':
        risk = 1.2 * p[:, 0]
    elif case == 'xor':
        risk = 1.4 * np.sign(p[:, 0] * g[:, 0])
    else:
        risk = np.zeros(n)
    # Each patient has a bag of 16 synthetic 4-dimensional patch features.
    patch = rng.normal(scale=.4, size=(n, 16, 4))
    patch[:, :, :2] += p[:, None, :]
    bags = [torch.tensor(x, dtype=torch.float64) for x in patch]
    ids = [f'{prefix}-{i:04d}' for i in range(n)]
    # Discrete event time, independent administrative/random censoring.
    logits = np.array([-1.6, -1.4, -1.2, -1.0])[None, :] + risk[:, None]
    hazard = 1 / (1 + np.exp(-logits))
    event_draw = rng.uniform(size=(n, 4)) < hazard
    first = np.where(event_draw.any(axis=1), event_draw.argmax(axis=1), 4)
    censor = rng.integers(0, 4, size=n)
    observed = first <= censor
    labels = np.where(observed, first, censor)
    return {'bags': bags, 'rna': torch.tensor(g, dtype=torch.float64), 'ids': ids,
            'labels': torch.tensor(labels, dtype=torch.long),
            'censorship': torch.tensor(~observed, dtype=torch.float64)}


def baseline(train, test):
    # Intercept-only hazard baseline, fitted solely on training outcomes.
    logits = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    opt = torch.optim.Adam([logits], lr=.05)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = discrete_nll(logits.expand(len(train['ids']), -1), train['labels'], train['censorship'], alpha=0.)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(discrete_nll(logits.expand(len(test['ids']), -1), test['labels'], test['censorship'], alpha=0.))


@torch.no_grad()
def retrieval(model, data):
    _, enc, _ = model._input(data['bags'], data['rna'], data['ids'])
    pp = model.path_projection(enc['pooled'])
    gg = model.rna_projection(data['rna'])
    clusters = []
    for k in range(model.k):
        mask = enc['available'][:, k]
        p, g = pp[mask, k], gg[mask]
        if len(p) < 2:
            continue
        score = F.normalize(p, dim=1) @ F.normalize(g, dim=1).T
        target = torch.arange(len(p))
        clusters.append({'cluster': k, 'n': len(p), 'chance_top1': 1 / len(p),
                         'top1_path_to_rna': float((score.argmax(1) == target).double().mean()),
                         'top1_rna_to_path': float((score.argmax(0) == target).double().mean()),
                         'contrastive_loss': float(patient_contrastive(p, g, model.temperature))})
    return clusters


def main():
    torch.set_num_threads(1)
    model_sha256 = hashlib.sha256((ROOT / 'method3_1_model.py').read_bytes()).hexdigest()
    records = []
    for case in CASES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            train = sample(rng, N_TRAIN, case, 'train')
            test = sample(rng, N_TEST, case, 'test')
            assert not set(train['ids']) & set(test['ids'])
            try:
                bank = FrozenMorphologyBank.fit(train['bags'], ids=train['ids'], clusters=2,
                                                latent_dim=3, max_fit_patches=1280,
                                                per_patient_cap=16, seed=seed)
                constant_nll = baseline(train, test)
            except Exception:
                records.append({'case': case, 'seed': seed, 'failure': traceback.format_exc()})
                print(f'{case} seed={seed}: bank/baseline FAILED', flush=True)
                continue
            for arm, (cross_weight, mismatch) in ARMS.items():
                record = {'case': case, 'seed': seed, 'arm': arm,
                          'cross_weight': cross_weight, 'mismatch': mismatch,
                          'test_intercept_only_nll_alpha0': constant_nll,
                          'train_events': int((1-train['censorship']).sum()),
                          'test_events': int((1-test['censorship']).sum())}
                try:
                    model = Method31(bank, rna_dim=2, projection_dim=4, bins=4, seed=seed)
                    hist = model.pretrain(train['bags'], train['rna'], train['ids'], epochs=35,
                                          cross_weight=cross_weight, mismatch=mismatch)
                    record['heldout_retrieval'] = retrieval(model, test)
                    model.fit_probes(train['bags'], train['rna'], train['ids'], folds=4)
                    survival_history = model.fit_survival(train['labels'], train['censorship'], train['ids'], epochs=60)
                    logits = model.predict(test['bags'], test['rna'], test['ids'])
                    record.update(test_nll_alpha0=float(discrete_nll(logits, test['labels'], test['censorship'], alpha=0.)),
                                  test_nll_alpha05=float(discrete_nll(logits, test['labels'], test['censorship'], alpha=.5)),
                                  final_ssl_cross_loss=hist[-1]['cross'],
                                  final_train_survival_objective=survival_history[-1])
                    print(f"{case} seed={seed} {arm}: NLL0={record['test_nll_alpha0']:.4f} baseline={constant_nll:.4f}", flush=True)
                except Exception:
                    record['failure'] = traceback.format_exc()
                    print(f'{case} seed={seed} {arm}: FAILED', flush=True)
                records.append(record)
    def summarize(seed_group):
        summary = {}
        for case in CASES:
            summary[case] = {}
            for arm in ARMS:
                selected = [r for r in records if r.get('case') == case and r.get('arm') == arm
                            and r.get('seed') in seed_group and 'failure' not in r]
                if selected:
                    summary[case][arm] = {'successful_runs': len(selected),
                        'expected_runs': len(seed_group),
                        'mean_test_nll_alpha0': float(np.mean([r['test_nll_alpha0'] for r in selected])),
                        'mean_intercept_nll_alpha0': float(np.mean([r['test_intercept_only_nll_alpha0'] for r in selected])),
                        'mean_path_to_rna_top1': float(np.mean([np.mean([c['top1_path_to_rna'] for c in r['heldout_retrieval']]) for r in selected])),
                        'mean_retrieval_chance': float(np.mean([np.mean([c['chance_top1'] for c in r['heldout_retrieval']]) for r in selected]))}
                else:
                    summary[case][arm] = {'successful_runs': 0, 'expected_runs': len(seed_group)}
        return summary
    if model_sha256 != hashlib.sha256((ROOT / 'method3_1_model.py').read_bytes()).hexdigest():
        raise RuntimeError('Model changed during this run; discard output and rerun the experiment')
    result = {'synthetic_only': True, 'seeds': list(SEEDS), 'n_train': N_TRAIN, 'n_test': N_TEST,
              'folds': 4, 'ssl_epochs': 35, 'survival_epochs': 60, 'clusters': 2,
              'patient_split': 'independent generated patients, disjoint synthetic IDs',
              'model_source_sha256': model_sha256,
              'training': 'Method31 fit_survival uses alpha=0; test evaluation reports alpha=0 and alpha=0.5',
              'limitations': ['Small fixed synthetic setups and three seeds do not establish clinical performance.',
                  'Contrastive retrieval is a diagnostic, not an estimate or proof of MI, independence, or synergy.',
                  'The KRR OOF basis is conditional on outer-train morphology and SSL; ensemble inference differs from OOF.',
                  'PCA/kmeans bank is not Nature Leiden or pathologist-validated HPL.',
                  'Arms retain trainable within-modality SSL; shuffled uses a fixed cyclic RNA permutation during cross-modal SSL.',
                  'No patient data were read; all survival outcomes and censoring are simulated.'],
              'seed_groups': {'original': list(ORIGINAL_SEEDS), 'new': list(NEW_SEEDS)},
              'summary': summarize(SEEDS),
              'summary_by_seed_group': {'original': summarize(ORIGINAL_SEEDS), 'new': summarize(NEW_SEEDS)},
              'runs': records}
    path = ROOT / 'results' / 'method3_1_synthetic.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Wrote {path}; failures={sum("failure" in r for r in records)}', flush=True)


if __name__ == '__main__':
    main()
