"""Stage 2: run the selected family configurations over all five internal folds.

Each fold is an independent training run; the RNA scaler and the constant-gene
mask are fitted on that fold's training patients only.  The 76 outer-test
patients never appear in a manifest and are never loaded.

The baseline five-fold result already exists under
/data1/decur_runs/DEV_FIVE_FOLD_CINDEX_20261112 and is reused as the paired
reference; only the new configurations are trained here.

Usage:
    WORKER_ID=0 TOTAL_WORKERS=2 PHYSICAL_GPU=0 python stage2_five_folds.py C03_interaction_0.5
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from search_config import (REPO as _REPO, RUN as _RUN, FOLD as _FOLD,
                           FOLD_DIR as _FOLD_DIR, GENES as _GENES, PY as _PY,
                           MEASURE as _MEASURE, LD_LIBRARY as _LD_LIBRARY,
                           PATCH_CHUNK_SIZE as _PATCH_CHUNK_SIZE)

import os
import subprocess
import sys
from pathlib import Path

from stage1_screen import GRID, build_command  # noqa: E402

RUN, FOLDS, GENES = _RUN, _FOLD_DIR, _GENES
REPO, PY, MEASURE = str(_REPO), _PY, _MEASURE
OVERRIDES = {name: values for name, values in GRID}
# Stage 1 entries whose grid row differs from the baseline only by parameters
# that are already expressed in the command builder.
# Stage 3 runs use a separate output root so they never mix with Stage 2.
STAGE = os.environ.get('STAGE_ROOT', 'stage2')


def build_fold_command(run_id, overrides, fold, out_dir, seed=17, device='cuda:0'):
    args = dict(overrides)
    args['seed'] = seed
    base = build_command(run_id, args, out_dir, device=device)
    cmd = []
    i = 0
    while i < len(base):
        if base[i] == '--manifest':
            cmd.extend(['--manifest', str(FOLDS / f'fold_{fold}.csv')])
            i += 2
            continue
        if base[i] == '--out':
            cmd.extend(['--out', str(out_dir)])
            i += 2
            continue
        cmd.append(base[i])
        i += 1
    return cmd


def main():
    configs = sys.argv[1:]
    if not configs:
        raise SystemExit('give at least one configuration id')
    worker = int(os.environ.get('WORKER_ID', '0'))
    total = int(os.environ.get('TOTAL_WORKERS', '1'))
    seed = int(os.environ.get('SEED', '17'))
    jobs = []
    for run_id in configs:
        for fold in range(5):
            jobs.append((run_id, fold))
    mine = [j for i, j in enumerate(jobs) if i % total == worker]
    print(f'worker {worker}/{total}: {len(mine)} jobs (seed {seed})', flush=True)
    for run_id, fold in mine:
        tag = run_id if seed == 17 else f'{run_id}_seed{seed}'
        out = RUN / STAGE / tag / f'FOLD_{fold}'
        if out.exists():
            print(f'{tag} fold {fold}: exists, skipped', flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_fold_command(run_id, OVERRIDES[run_id], fold, out, seed=seed)
        env = dict(os.environ)
        env['LD_LIBRARY_PATH'] = '/home/ubuntu/.conda/envs/trisurv/lib'
        env['CUDA_VISIBLE_DEVICES'] = os.environ.get('PHYSICAL_GPU', '0')
        (out.parent / f'command_{tag}_fold{fold}.txt').write_text(' '.join(cmd) + '\n', encoding='utf-8')
        print(f'{tag} fold {fold}: start', flush=True)
        with (out.parent / f'train_{tag}_fold{fold}.log').open('w', encoding='utf-8') as handle:
            proc = subprocess.run(cmd, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT, cwd=REPO)
        print(f'{tag} fold {fold}: exit={proc.returncode}', flush=True)

if __name__ == '__main__':
    main()
