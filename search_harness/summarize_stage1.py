"""Summarise the Stage 1 screening grid and apply the pre-registered filter.

Reads every run directory under the Stage 1 output root, reports the full grid
including failures and configurations worse than the baseline, and applies the
family dominance rule written in HYPERPARAM_SEARCH_PROTOCOL_20261112.md.

Nothing here selects on a single-fold C-index ranking.  Stage 1 only decides
which factor families deserve the 5-fold budget in Stage 2.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from search_config import (REPO as _REPO, RUN as _RUN, FOLD as _FOLD,
                           FOLD_DIR as _FOLD_DIR, GENES as _GENES, PY as _PY,
                           MEASURE as _MEASURE, LD_LIBRARY as _LD_LIBRARY,
                           PATCH_CHUNK_SIZE as _PATCH_CHUNK_SIZE)

import json
from pathlib import Path

import numpy as np

RUN = _RUN
OUT = RUN / 'stage1_summary.json'

# The Stage 1 reference is the baseline configuration of THIS search, trained
# with the same patch_chunk_size (8192).  It is filled in from C00_baseline
# once that run has finished; until then no comparison is reported.
HISTORICAL_REFERENCE = {
    'source': '/data1/decur_runs/DEV_FIVE_FOLD_CINDEX_20261112',
    'patch_chunk_size': 2048,
    'fold_0_validation_nll': 2.333065986633301,
    'fold_0_c_index': 0.6370967741935484,
    'note': 'historical arm at patch_chunk_size 2048; reported for reference only, '
            'not the paired comparison target of this search',
}

# Family of every grid entry, and the family quota for Stage 2 (max 1 each).
FAMILY = {
    'C00_baseline': 'baseline',
    'C01_interaction_0.0': 'interaction', 'C02_interaction_0.05': 'interaction',
    'C03_interaction_0.5': 'interaction', 'C04_interaction_1.0': 'interaction',
    'C05_interaction_2.0': 'interaction',
    'C06_match_temp_0.05': 'match_temp', 'C07_match_temp_0.1': 'match_temp',
    'C08_match_temp_0.5': 'match_temp', 'C09_match_temp_1.0': 'match_temp',
    'C10_baseline_batch34': 'capacity', 'C11_hidden_32_batch34': 'capacity',
    'C12_local_16': 'capacity', 'C13_hidden_32_local_16_batch34': 'capacity',
    'C14_baseline_batch65': 'capacity', 'C15_hidden_64_batch65': 'capacity',
    'C16_lr_0.001': 'optimisation', 'C17_lr_0.006': 'optimisation',
    'C18_wd_0.01': 'optimisation', 'C19_wd_0.0001': 'optimisation',
    'C20_aux_x10': 'aux_weight', 'C21_aux_x100': 'aux_weight', 'C22_aux_off': 'aux_weight',
    'C23_feature_noise_0.15': 'noise', 'C24_rna_noise_0.15': 'noise', 'C25_rna_noise_0.0': 'noise',
    'C26_starts_0_12_36': 'time_bins', 'C27_starts_0_24': 'time_bins',
    'C28_starts_0_6_18_36': 'time_bins',
    'C29_inter_0.5_temp_0.1': 'combo', 'C30_inter_1.0_temp_0.1': 'combo',
}

C_TOLERANCE = 0.03


def read_run(path):
    report = path / 'report.json'
    if not report.exists():
        return None
    try:
        data = json.loads(report.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return None
    config = data.get('config', {})
    starts = data.get('interval_starts', [0])
    return {
        'run': path.name,
        'family': FAMILY.get(path.name, 'unknown'),
        'status': 'completed',
        'c_index': data['evaluation_c_index_risk_at_horizon']['value'],
        'comparable_pairs': data['evaluation_c_index_risk_at_horizon']['comparable_pairs'],
        'ipcw_brier': data['evaluation_ipcw_brier_at_horizon']['value'],
        'validation_nll': data['validation_nll'],
        'best_epoch': data['best_epoch'],
        'epochs_run': len(data['history']),
        'interval_starts': starts,
        'interaction_strength': config.get('interaction_strength'),
        'match_temperature': config.get('match_temperature'),
        'hidden': config.get('hidden'), 'common': config.get('common'),
        'local_dim': config.get('local_dim'), 'local_common': config.get('local_common'),
        'lr': config.get('lr'), 'weight_decay': config.get('weight_decay'),
        'patient_batch_size': data.get('stream_config', {}).get('patient_batch_size'),
        'feature_noise': config.get('feature_noise'), 'rna_noise': config.get('rna_noise'),
        'wall_seconds': data.get('runtime', {}).get('wall_seconds'),
        'gpu_peak_allocated_mb': (data.get('runtime', {}).get('cuda') or {}).get('max_allocated_bytes', 0) / 2**20,
    }


def main():
    rows = []
    for path in sorted(RUN.iterdir()):
        if not path.is_dir() or path.name.startswith('.'):
            continue
        row = read_run(path)
        if row is None:
            log = RUN / f'train_{path.name}.log'
            status = 'failed_or_running'
            detail = ''
            if log.exists():
                text = log.read_text(encoding='utf-8', errors='replace')
                detail = text.strip().splitlines()[-1][:300] if text.strip() else ''
            rows.append({'run': path.name, 'family': FAMILY.get(path.name, 'unknown'),
                         'status': status, 'detail': detail})
        else:
            rows.append(row)

    done = [r for r in rows if r.get('status') == 'completed']
    base = next((r for r in done if r['run'] == 'C00_baseline'), None)
    if base is None:
        print('C00_baseline has not finished yet; no comparison is reported.')
        return
    BASELINE = {'run': 'C00_baseline', 'c_index': base['c_index'],
                'ipcw_brier': base['ipcw_brier'], 'validation_nll': base['validation_nll']}

    # Same discretisation => validation NLL is comparable across runs.
    base_starts = base['interval_starts']

    for r in done:
        r['delta_c_index'] = r['c_index'] - BASELINE['c_index']
        r['delta_brier'] = r['ipcw_brier'] - BASELINE['ipcw_brier']
        r['nll_comparable'] = r['interval_starts'] == base_starts
        r['delta_validation_nll'] = (r['validation_nll'] - BASELINE['validation_nll']
                                     if r['nll_comparable'] else None)
        r['c_within_tolerance'] = r['delta_c_index'] >= -C_TOLERANCE
        r['beats_baseline_on'] = (
            ['c_index'] if r['delta_c_index'] > 0 else []) + \
            (['ipcw_brier'] if r['delta_brier'] < 0 else []) + \
            (['validation_nll'] if r['nll_comparable'] and r['delta_validation_nll'] < 0 else [])
        r['admissible'] = (r['c_within_tolerance'] and len(r['beats_baseline_on']) >= 1)

    # Family quota: at most one admissible configuration per family.
    selected, rejected = [], []
    by_family = {}
    for r in sorted([x for x in done if x['admissible'] and x['run'] != 'C00_baseline'],
                    key=lambda x: (-x['delta_c_index'], x['ipcw_brier'])):
        by_family.setdefault(r['family'], []).append(r)
    for family, members in by_family.items():
        selected.append(members[0])
        rejected.extend(members[1:])
    selected.sort(key=lambda x: -x['delta_c_index'])

    summary = {
        'protocol': 'HYPERPARAM_SEARCH_PROTOCOL_20261112.md',
        'baseline_anchor': BASELINE,
        'baseline_source': 'C00_baseline of this search (patch_chunk_size 8192)',
        'historical_reference_arm': HISTORICAL_REFERENCE,
        'runs_total': len(rows),
        'runs_completed': len(done),
        'c_tolerance': C_TOLERANCE,
        'single_fold_c_index_standard_error_estimate': 0.03,
        'rows': sorted(rows, key=lambda x: x['run']),
        'stage2_selection': {
            'rule': 'per-family quota of 1, ordered by delta C-index among admissible runs',
            'selected': [r['run'] for r in selected],
            'selected_families': sorted({r['family'] for r in selected}),
            'rejected_admissible_by_quota': [r['run'] for r in rejected],
        },
        'limits': [
            'Stage 1 is a single 61-patient fold; its C-index standard error is about 0.03, '
            'so it cannot rank configurations and is used only to decide which families '
            'deserve the 5-fold budget.',
            'Validation NLL is only comparable between runs with identical interval starts.',
            'The outer test set remains sealed and was not loaded.',
            'Feature noise operates on encoded patch features, not on WSI images.',
            'All Stage 1 runs use patch_chunk_size 8192; the historical chunk 2048 five-fold '
            'result is a reference arm, not a paired comparison target.',
        ],
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'{"run":32s} {"family":14s} {"C":>7s} {"dC":>7s} {"Brier":>7s} {"dBrier":>8s} {"NLL":>9s} ep')
    for r in sorted(done, key=lambda x: -x.get('c_index', 0)):
        print(f'{r["run"]:32s} {r["family"]:14s} {r["c_index"]:7.4f} {r["delta_c_index"]:+7.4f} '
              f'{r["ipcw_brier"]:7.4f} {r["delta_brier"]:+8.4f} {r["validation_nll"]:9.4f} '
              f'{r["best_epoch"]:3d}  {"ADMIT" if r["admissible"] else "-"}')
    for r in rows:
        if r.get('status') != 'completed':
            print(f'{r["run"]:32s} {r["family"]:14s} {r["status"]}')
    print()
    print('baseline (this search, chunk 8192):',
          f'C={BASELINE["c_index"]:.4f} Brier={BASELINE["ipcw_brier"]:.4f} '
          f'NLL={BASELINE["validation_nll"]:.4f}')
    print('historical reference arm (chunk 2048, fold 0):',
          f'C={HISTORICAL_REFERENCE["fold_0_c_index"]:.4f} '
          f'NLL={HISTORICAL_REFERENCE["fold_0_validation_nll"]:.4f}')
    print('completed:', len(done), '/', len(rows))
    print('Stage 2 selected (family quota 1):', summary['stage2_selection']['selected'])
    print('rejected by quota:', summary['stage2_selection']['rejected_admissible_by_quota'])


if __name__ == '__main__':
    main()
