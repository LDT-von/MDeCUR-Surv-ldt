"""Stage 1 hyperparameter screening for the new pathway-local framework.

Protocol: HYPERPARAM_SEARCH_PROTOCOL_20261112.md (pre-registered before any run).

Single internal development fold (fold_0, 243 train / 61 val), full training,
all patches, chunk 2048, checkpoint selected by validation NLL.  Every
configuration in the grid is run and every result is reported, including
failures and configurations that are worse than the baseline.  The selection
metric for Stage 1 is validation NLL; the C-index is recorded but must not
drive selection.

Two configuration families need an explicit larger patient batch, because the
active identity-correlation terms impose the necessary rank bound
patient_batch_size >= dim+1 on the correlated dimension:

  * hidden/common above 19  -> patient_batch_size >= 33 (hidden 32) or 65 (hidden 64)
  * local_dim above 19      -> patient_batch_size >= local_dim+1

Capacity runs therefore carry their own matched control at the same patient
batch size, so the capacity effect is not confounded with the batch size.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from search_config import (REPO as _REPO, RUN as _RUN, FOLD as _FOLD,
                           FOLD_DIR as _FOLD_DIR, GENES as _GENES, PY as _PY,
                           MEASURE as _MEASURE, LD_LIBRARY as _LD_LIBRARY,
                           PATCH_CHUNK_SIZE as _PATCH_CHUNK_SIZE)

import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN, FOLD, GENES = _RUN, _FOLD, _GENES
REPO, PY, MEASURE = str(_REPO), _PY, _MEASURE
# Protocol revision (HYPERPARAM_SEARCH_PROTOCOL_20261112.md section 3.0): the
# screening uses patch_chunk_size 8192 for every run, including the baseline.
# The historical chunk 2048 five-fold result is reported as a reference arm but
# is not the paired comparison target.

AUX_WEIGHTS = ('global_path_within', 'global_rna_within', 'global_common',
               'global_unique', 'patch_within', 'pathway_path_within',
               'pathway_rna_within', 'pathway_common', 'pathway_unique',
               'matched_common', 'matched_unique')

# Pre-registered Stage 1 grid.  Every entry is one full training run.
GRID = [
    # --- anchor: must reproduce the recorded fold_0 baseline --------------
    ('C00_baseline',           {}),

    # --- interaction strength: the interaction branch carries the signal --
    ('C01_interaction_0.0',    {'interaction_strength': 0.0}),
    ('C02_interaction_0.05',   {'interaction_strength': 0.05}),
    ('C03_interaction_0.5',    {'interaction_strength': 0.5}),
    ('C04_interaction_1.0',    {'interaction_strength': 1.0}),
    ('C05_interaction_2.0',    {'interaction_strength': 2.0}),

    # --- matching sharpness of the RNA-conditioned soft correspondence -----
    ('C06_match_temp_0.05',    {'match_temperature': 0.05}),
    ('C07_match_temp_0.1',     {'match_temperature': 0.1}),
    ('C08_match_temp_0.5',     {'match_temperature': 0.5}),
    ('C09_match_temp_1.0',     {'match_temperature': 1.0}),

    # --- capacity, each with a matched patient-batch control --------------
    ('C10_baseline_batch34',   {'patient_batch_size': 34}),
    ('C11_hidden_32_batch34',  {'patient_batch_size': 34, 'hidden': 32, 'common': 16}),
    ('C12_local_16',           {'local_dim': 16, 'local_common': 8}),
    ('C13_hidden_32_local_16_batch34', {'patient_batch_size': 34, 'hidden': 32, 'common': 16,
                                        'local_dim': 16, 'local_common': 8}),
    ('C14_baseline_batch65',   {'patient_batch_size': 65}),
    ('C15_hidden_64_batch65',  {'patient_batch_size': 65, 'hidden': 64, 'common': 32}),
    # --- optimisation ------------------------------------------------------
    ('C16_lr_0.001',           {'lr': 0.001}),
    ('C17_lr_0.006',           {'lr': 0.006}),
    ('C18_wd_0.01',            {'weight_decay': 0.01}),
    ('C19_wd_0.0001',          {'weight_decay': 0.0001}),

    # --- auxiliary weighting ----------------------------------------------
    ('C20_aux_x10',            {'global_path_within': 0.005, 'global_rna_within': 0.005,
                                'global_common': 0.01, 'global_unique': 0.01,
                                'patch_within': 0.01, 'pathway_path_within': 0.005,
                                'pathway_rna_within': 0.005, 'pathway_common': 0.01,
                                'pathway_unique': 0.01, 'matched_common': 0.005,
                                'matched_unique': 0.005}),
    ('C21_aux_x100',           {'global_path_within': 0.05, 'global_rna_within': 0.05,
                                'global_common': 0.1, 'global_unique': 0.1,
                                'patch_within': 0.1, 'pathway_path_within': 0.05,
                                'pathway_rna_within': 0.05, 'pathway_common': 0.1,
                                'pathway_unique': 0.1, 'matched_common': 0.05,
                                'matched_unique': 0.05}),
    ('C22_aux_off',            {name: 0.0 for name in AUX_WEIGHTS}),

    # --- feature-noise level of the single feature view --------------------
    ('C23_feature_noise_0.15', {'feature_noise': 0.15}),
    ('C24_rna_noise_0.15',     {'rna_noise': 0.15}),
    ('C25_rna_noise_0.0',      {'rna_noise': 0.0}),

    # --- time discretisation: piecewise-constant hazard intervals ----------
    ('C26_starts_0_12_36',     {'starts': [0.0, 12.0, 36.0]}),
    ('C27_starts_0_24',        {'starts': [0.0, 24.0]}),
    ('C28_starts_0_6_18_36',   {'starts': [0.0, 6.0, 18.0, 36.0]}),

    # --- pre-registered combinations of the two strongest levers -----------
    ('C29_inter_0.5_temp_0.1', {'interaction_strength': 0.5, 'match_temperature': 0.1}),
    ('C30_inter_1.0_temp_0.1', {'interaction_strength': 1.0, 'match_temperature': 0.1}),
]

BASE = {
    'view_source': 'feature_noise',
    'feature_noise': '0.05', 'rna_noise': '0.05',
    'common_mode': 'stability', 'strength': '2', 'min_weight': '0.1',
    'interaction_strength': '0.2', 'match_temperature': '0.2',
    'hidden': '16', 'common': '8', 'local_dim': '8', 'local_common': '4',
    'global_path_within': '0.0005', 'global_rna_within': '0.0005',
    'global_common': '0.001', 'global_unique': '0.001', 'patch_within': '0.001',
    'pathway_path_within': '0.0005', 'pathway_rna_within': '0.0005',
    'pathway_common': '0.001', 'pathway_unique': '0.001',
    'matched_common': '0.0005', 'matched_unique': '0.0005',
    'patient_batch_size': '20', 'patch_chunk_size': '8192',
    'epochs': '80', 'patience': '15', 'seed': '17',
    'lr': '0.003', 'weight_decay': '0.001',
    'horizon': '12', 'time_unit': 'months',
    'endpoint': 'Dataset-provided OS in months; time origin unverified',
}

FLAG = {
    'view_source': '--view-source',
    'interaction_strength': '--interaction-strength',
    'match_temperature': '--match-temperature',
    'hidden': '--hidden', 'common': '--common',
    'local_dim': '--local-dim', 'local_common': '--local-common',
    'lr': '--lr', 'weight_decay': '--weight-decay',
    'feature_noise': '--feature-noise', 'rna_noise': '--rna-noise',
    'global_path_within': '--global-path-within', 'global_rna_within': '--global-rna-within',
    'global_common': '--global-common', 'global_unique': '--global-unique',
    'patch_within': '--patch-within',
    'pathway_path_within': '--pathway-path-within',
    'pathway_rna_within': '--pathway-rna-within',
    'pathway_common': '--pathway-common', 'pathway_unique': '--pathway-unique',
    'matched_common': '--matched-common', 'matched_unique': '--matched-unique',
    'common_mode': '--common-mode', 'strength': '--strength', 'min_weight': '--min-weight',
    'patient_batch_size': '--patient-batch-size', 'patch_chunk_size': '--patch-chunk-size',
    'epochs': '--epochs', 'patience': '--patience', 'seed': '--seed',
    'horizon': '--horizon', 'time_unit': '--time-unit', 'endpoint': '--endpoint',
}


def build_command(run_id, overrides, out_dir, device='cuda:0'):
    args = dict(BASE)
    args.update({k: str(v) for k, v in overrides.items()})
    starts = args.pop('starts', ['0'])
    cmd = [PY, '-u', MEASURE,
           '--manifest', str(FOLD), '--genes', GENES,
           '--out', str(out_dir), '--device', device]
    for key, value in args.items():
        cmd.extend([FLAG[key], value])
    cmd.extend(['--starts'] + [str(s) for s in starts])
    return cmd


def main():
    selected = sys.argv[1:]
    RUN.mkdir(parents=True, exist_ok=True)
    grid = GRID if not selected else [g for g in GRID if g[0] in selected]
    # Interleaved worker split so that a slow configuration cannot leave one
    # worker idle while the other still has a queue.
    worker = os.environ.get('WORKER_ID', '0')
    total = int(os.environ.get('TOTAL_WORKERS', '1'))
    mine = [(i, g) for i, g in enumerate(grid) if i % total == int(worker)]
    print(f'worker {worker}/{total}: {len(mine)} configurations', flush=True)
    for _, (run_id, overrides) in mine:
        out = RUN / run_id
        if out.exists():
            print(f'{run_id}: already exists, skipped', flush=True)
            continue
        cmd = build_command(run_id, overrides, out)
        env = dict(os.environ)
        env['LD_LIBRARY_PATH'] = _LD_LIBRARY
        env['CUDA_VISIBLE_DEVICES'] = os.environ.get('PHYSICAL_GPU', '0')
        (RUN / f'command_{run_id}.txt').write_text(' '.join(cmd) + '\n', encoding='utf-8')
        print(f'{run_id}: start', flush=True)
        started = time.time()
        with (RUN / f'train_{run_id}.log').open('w', encoding='utf-8') as handle:
            proc = subprocess.run(cmd, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT, cwd=REPO)
        print(f'{run_id}: exit={proc.returncode} wall={time.time()-started:.1f}s', flush=True)

if __name__ == '__main__':
    main()
