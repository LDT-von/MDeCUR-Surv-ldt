"""Summarise the method 3.2 synthetic benchmark into a compact table."""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, 'results', 'method3_2_synthetic_20261117.json')
data = json.load(open(PATH, encoding='utf-8'))

runs = data['runs']
arms = []
for key in runs:
    scenario, seed, arm = key.split('|')
    if arm not in arms:
        arms.append(arm)

print('=== scale ===')
print(data['scale'])
print('elapsed_seconds', data['elapsed_seconds'])
print('atlas ARI', data['atlas_audit']['cluster_ari'])
print('composition MAE vs truth', data['atlas_audit']['composition_mae_vs_truth'])
print('CLR max |sum|', data['atlas_audit']['clr_sum_max_deviation'])
print('leakage', data['leakage_check']['test_change_keeps_state'],
      data['leakage_check']['val_change_keeps_state'])

scenarios = data['scale']['scenarios']
seeds = data['scale']['seeds']

print('\n=== per scenario: mean over seeds (NLL lower better / C higher better) ===')
for scenario in scenarios:
    print(f'\n--- {scenario} ---')
    header = f'{"arm":<20}'
    for arm in arms:
        vals_nll, vals_c = [], []
        for seed in seeds:
            entry = runs.get(f'{scenario}|{seed}|{arm}')
            if not entry or entry.get('status') != 'ok':
                continue
            vals_nll.append(entry['nll'])
            if entry.get('harrell_c') is not None:
                vals_c.append(entry['harrell_c'])
        mean_nll = sum(vals_nll) / len(vals_nll) if vals_nll else float('nan')
        mean_c = sum(vals_c) / len(vals_c) if vals_c else float('nan')
        header += f'{mean_nll:8.3f}/{mean_c:6.3f} '
    print(header)

print('\n=== overall mean over all scenarios and seeds ===')
for arm in arms:
    vals_nll, vals_c = [], []
    for scenario in scenarios:
        for seed in seeds:
            entry = runs.get(f'{scenario}|{seed}|{arm}')
            if not entry or entry.get('status') != 'ok':
                continue
            vals_nll.append(entry['nll'])
            if entry.get('harrell_c') is not None:
                vals_c.append(entry['harrell_c'])
    mean_nll = sum(vals_nll) / len(vals_nll)
    mean_c = sum(vals_c) / len(vals_c) if vals_c else float('nan')
    print(f'{arm:<20} nll {mean_nll:.4f}  c {mean_c:.4f}  runs {len(vals_nll)}')

print('\n=== retrieval a_k by scenario (seed 11) ===')
for scenario in scenarios:
    entry = runs.get(f'{scenario}|11|full')
    if entry and entry.get('status') == 'ok':
        a = entry.get('retrieval_a')
        trained = entry.get('cluster_trained')
        print(f'{scenario:<16} a={[round(x, 3) for x in a]} trained={trained} '
              f'rna_valid_frac={entry.get("rna_valid_fraction")}')

print('\n=== decouple diagnostics ===')
for scenario in scenarios[:4]:
    for arm in ('decouple', 'decouple_noregu', 'decouple_asym'):
        entry = runs.get(f'{scenario}|11|{arm}')
        if entry and entry.get('status') == 'ok':
            print(f'{scenario:<14} {arm:<16} path_common_frac='
                  f'{entry.get("path_common_fraction"):.3f} rna_common_frac='
                  f'{entry.get("rna_common_fraction"):.3f} nll={entry["nll"]:.4f} '
                  f'c={entry.get("harrell_c")}')

failed = [k for k, v in runs.items() if v.get('status') != 'ok']
print('\nfailed runs:', failed if failed else 'none')
