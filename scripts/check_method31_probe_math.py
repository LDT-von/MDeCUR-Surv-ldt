"""Synthetic-only audit of a frozen nonlinear probe's mathematical limits.

Run from any working directory: python Conditional_DeCUR_Surv/scripts/check_method31_probe_math.py
No patient inputs, survival outcomes, files, or network connections are used.
"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morphid_probe import CrossFittedRBFProbe  # noqa: E402


SEED = 20261116
N_TRAIN = 320
N_TEST = 4000
FOLDS = 4


def r2(y, prediction, baseline):
    return float(1.0 - np.sum((y - prediction) ** 2) / np.sum((y - baseline) ** 2))


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def run_probe(g_train, y_train, g_test, y_test):
    ids = [f'synthetic-{i:04d}' for i in range(N_TRAIN)]
    probe = CrossFittedRBFProbe(alpha=1e-3, bandwidth=None)
    probe.fit(g_train[:, None], y_train[:, None], ids, folds=FOLDS, seed=SEED)
    oof = probe.oof_predictions.numpy().reshape(-1)
    test = probe.predict(g_test[:, None]).numpy().reshape(-1)
    for audit in probe.split_audit:
        assert not set(audit['train_ids']).intersection(audit['validation_ids'])
    return probe, oof, test


def main():
    # Limiting CPU threads makes the small synthetic kernel solves predictable.
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    g_train, g_test = rng.normal(size=N_TRAIN), rng.normal(size=N_TEST)
    print(f'Seed={SEED}; synthetic train={N_TRAIN}, independent test={N_TEST}, OOF folds={FOLDS}')
    print('Probe: existing CrossFittedRBFProbe, alpha=0.001, train-only median bandwidth')

    # E[P|G]=G². Symmetry makes Pearson correlation near zero even though P
    # is highly predictable from G by an appropriate nonlinear predictor.
    p_train = g_train ** 2 + 0.1 * rng.normal(size=N_TRAIN)
    p_test = g_test ** 2 + 0.1 * rng.normal(size=N_TEST)
    probe, oof, test = run_probe(g_train, p_train, g_test, p_test)
    slope, intercept = np.polyfit(g_train, p_train, 1)
    linear_test = slope * g_test + intercept
    print('\nS1: P = G^2 + independent Gaussian noise (sd=0.1)')
    print('Fold train/held-out sizes:', [
        (len(a['train_indices']), len(a['validation_indices'])) for a in probe.split_audit
    ])
    print(f'Test Pearson corr(G,P)={corr(g_test, p_test):.5f}')
    print(f'Test linear R^2={r2(p_test, linear_test, p_train.mean()):.5f}')
    print(f'OOF KRR R^2={r2(p_train, oof, p_train.mean()):.5f}')
    print(f'Independent-test KRR ensemble R^2={r2(p_test, test, p_train.mean()):.5f}')

    # E[P|G]=0, but Var(P|G)=G². Conditional-mean regression cannot
    # remove this dependence, even with a perfect population predictor.
    p_train = g_train * rng.normal(size=N_TRAIN)
    p_test = g_test * rng.normal(size=N_TEST)
    _, oof, test = run_probe(g_train, p_train, g_test, p_test)
    residual = p_test - test
    low = np.abs(g_test) <= 0.5
    high = np.abs(g_test) >= 1.5
    print('\nS4: P = G * epsilon, G and epsilon independent standard normal')
    print(f'OOF KRR R^2={r2(p_train, oof, p_train.mean()):.5f}')
    print(f'Independent-test KRR ensemble R^2={r2(p_test, test, p_train.mean()):.5f}')
    print(f'Test corr(G,residual)={corr(g_test, residual):.5f}')
    print(f'Test corr(G^2,residual^2)={corr(g_test ** 2, residual ** 2):.5f}')
    print(f'Low |G|<=0.5: n={int(low.sum())}, residual mean={residual[low].mean():.5f}, variance={residual[low].var():.5f}')
    print(f'High |G|>=1.5: n={int(high.sum())}, residual mean={residual[high].mean():.5f}, variance={residual[high].var():.5f}')
    print(f'High/low residual-variance ratio={residual[high].var() / residual[low].var():.3f}')
    print('\nInterpretation: OOF predictions and fold-ensemble test predictions are different estimators.')
    print('G^2 is nonlinear predictable despite near-zero correlation. A poor conditional-mean')
    print('prediction does not establish independence: the residual can retain G-dependent variance.')
    print('These single-seed synthetic diagnostics do not establish model or clinical performance.')


if __name__ == '__main__':
    main()
