"""Unattended orchestrator for the whole pre-registered search.

The protocol's selection rules are deterministic, so the entire Stage 1 ->
Stage 2 -> Stage 3 -> report pipeline can run without human intervention:

  Stage 1  31 configurations on one internal fold.
  Stage 2  the configurations selected by the family-quota rule, plus the
           baseline, each over all five internal folds.
  Stage 3  the Stage 2 winner and the baseline at seeds 29 and 43, five folds
           each, to separate fold variance from seed variance.

Selection follows HYPERPARAM_SEARCH_PROTOCOL_20261112.md exactly.  The
orchestrator never changes the grid, the filter or the comparison target; it
only executes them in order and records what happened.

Usage:
    python orchestrate.py
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

RUN = _RUN
REPO, PY = str(_REPO), _PY
LOG = RUN / 'logs'

STAGE1_WORKERS = int(os.environ.get('STAGE1_WORKERS', '6'))
STAGE2_WORKERS = int(os.environ.get('STAGE2_WORKERS', '6'))
STAGE3_WORKERS = int(os.environ.get('STAGE3_WORKERS', '6'))
# Protocol section 3: Stage 2 runs at most five configurations, the baseline
# included.
MAX_STAGE2_CONFIGS = int(os.environ.get('MAX_STAGE2_CONFIGS', '5'))
# Family of every Stage 1 grid entry, used for the family quota.
FAMILY_OF = {
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
    'C23_feature_noise_0.15': 'noise', 'C24_rna_noise_0.15': 'noise',
    'C25_rna_noise_0.0': 'noise',
    'C26_starts_0_12_36': 'time_bins', 'C27_starts_0_24': 'time_bins',
    'C28_starts_0_6_18_36': 'time_bins',
    'C29_inter_0.5_temp_0.1': 'combo', 'C30_inter_1.0_temp_0.1': 'combo',
}


def log(message):
    stamp = time.strftime('%H:%M:%S')
    print(f'[{stamp}] {message}', flush=True)
    with (LOG / 'orchestrator.log').open('a', encoding='utf-8') as handle:
        handle.write(f'[{stamp}] {message}\n')


def run(command, **kwargs):
    log('run: ' + ' '.join(str(c) for c in command))
    return subprocess.run(command, cwd=REPO, **kwargs)


def stage1_status():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stage1_queue import QUEUE, read_queue
    items = read_queue()
    done = [i for i in items if i.get('status') == 'completed']
    failed = [i for i in items if i.get('status') == 'failed']
    pending = [i for i in items if i.get('status') is None]
    return done, failed, pending


def stage1_workers_running():
    result = subprocess.run(['pgrep', '-f', 'stage1_queue.py'],
                            capture_output=True, text=True)
    return len([x for x in result.stdout.split() if x.strip()])


def reconcile():
    """Repair queue entries left behind by a killed worker."""
    run([PY, str(RUN / 'reconcile_queue.py')])


def start_stage1():
    """Launch Stage 1 workers, but only if none are already running.

    The queue is file-locked and claim-based, so workers started by an earlier
    invocation and workers started here never duplicate work: an already
    finished run directory is skipped and an already claimed entry stays
    claimed by its original owner.
    """
    existing = stage1_workers_running()
    if existing:
        log(f'Stage 1: {existing} workers already running, not launching more')
        while stage1_workers_running():
            time.sleep(30)
        return
    log('Stage 1: launching workers')
    processes = []
    for index in range(STAGE1_WORKERS):
        env = dict(os.environ)
        env.update({'WORKER_ID': f'auto{index}', 'PHYSICAL_GPU': str(index % 2),
                    'IDLE_LIMIT': '3', 'LD_LIBRARY_PATH': '/home/ubuntu/.conda/envs/trisurv/lib'})
        handle = (LOG / f'stage1_w{index}.log').open('w', encoding='utf-8')
        processes.append(subprocess.Popen(
            [PY, '-u', str(RUN / 'stage1_queue.py')], env=env, stdout=handle,
            stderr=subprocess.STDOUT, cwd=REPO))
    for process in processes:
        process.wait()


def release_all_claims():
    """Clear stale queue claims once the workers have exited."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fcntl
    from stage1_queue import LOCK, QUEUE, read_queue, write_queue
    with LOCK.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            items = read_queue()
            for item in items:
                if item.get('status') is None and item.get('claimed_by') is not None:
                    item['claimed_by'] = None
            write_queue(items)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def select_stage2():
    """Apply the family quota and the Stage 2 budget cap from the protocol.

    Protocol section 3: at most one configuration per factor family, and at
    most five configurations in total including the baseline.  The order of
    the families follows the pre-registered priority in section 1
    (interaction strength, matching sharpness, time discretisation, capacity,
    auxiliary weighting, optimisation, noise, combinations), so that the
    budget is spent on the factors the mechanism diagnostic prioritised.
    """
    run([PY, str(RUN / 'summarize_stage1.py')])
    summary = json.loads((RUN / 'stage1_summary.json').read_text(encoding='utf-8'))
    selected = summary['stage2_selection']['selected']
    priority = ['interaction', 'match_temp', 'time_bins', 'capacity',
                'aux_weight', 'optimisation', 'noise', 'combo']
    families = {}
    for name in selected:
        family = FAMILY_OF.get(name, 'unknown')
        families.setdefault(family, []).append(name)
    ordered = []
    for family in priority:
        ordered.extend(families.get(family, []))
    for family in sorted(families):
        if family not in priority:
            ordered.extend(families[family])
    # The baseline always enters Stage 2 as the paired comparison target and
    # occupies one of the five slots.
    budget = MAX_STAGE2_CONFIGS - 1
    chosen = ordered[:budget]
    configs = ['C00_baseline'] + chosen
    log(f'Stage 2 configurations (cap {MAX_STAGE2_CONFIGS}): {configs}')
    log(f'families represented: {sorted({FAMILY_OF.get(c, "unknown") for c in configs})}')
    if len(ordered) > budget:
        log(f'NOT selected because of the Stage 2 budget cap: {ordered[budget:]}')
    (RUN / 'stage2_configs.json').write_text(json.dumps(configs, indent=2), encoding='utf-8')
    return configs


def start_stage2(configs):
    log('Stage 2: launching workers')
    processes = []
    for index in range(STAGE2_WORKERS):
        env = dict(os.environ)
        env.update({'WORKER_ID': str(index), 'TOTAL_WORKERS': str(STAGE2_WORKERS),
                    'PHYSICAL_GPU': str(index % 2), 'SEED': '17',
                    'LD_LIBRARY_PATH': '/home/ubuntu/.conda/envs/trisurv/lib'})
        handle = (LOG / f'stage2_w{index}.log').open('w', encoding='utf-8')
        processes.append(subprocess.Popen(
            [PY, '-u', str(RUN / 'stage2_five_folds.py')] + configs, env=env,
            stdout=handle, stderr=subprocess.STDOUT, cwd=REPO))
    for process in processes:
        process.wait()


def pick_stage3():
    """The Stage 2 winner and the baseline; selection is by pooled OOF C-index."""
    run([PY, str(RUN / 'summarize_stage2.py')])
    summary = json.loads((RUN / 'stage2_summary.json').read_text(encoding='utf-8'))
    scored = [(name, entry['pooled']['c_index']) for name, entry in summary.items()
              if 'pooled' in entry and entry.get('folds_complete')]
    if not scored:
        log('Stage 3 skipped: no complete Stage 2 configuration')
        return []
    scored.sort(key=lambda x: -x[1])
    winner = scored[0][0]
    log(f'Stage 3: winner by pooled OOF C = {winner} ({scored[0][1]:.4f}); '
        f'ranking = {[(n, round(c, 4)) for n, c in scored]}')
    targets = ['C00_baseline']
    if winner != 'C00_baseline':
        targets.append(winner)
    (RUN / 'stage3_configs.json').write_text(json.dumps(targets, indent=2), encoding='utf-8')
    return targets


def start_stage3(targets):
    log('Stage 3: launching workers')
    processes = []
    for seed in (29, 43):
        for index in range(STAGE3_WORKERS):
            env = dict(os.environ)
            env.update({'WORKER_ID': str(index), 'TOTAL_WORKERS': str(STAGE3_WORKERS),
                        'PHYSICAL_GPU': str(index % 2), 'SEED': str(seed),
                        'STAGE_ROOT': 'stage3',
                        'LD_LIBRARY_PATH': '/home/ubuntu/.conda/envs/trisurv/lib'})
            handle = (LOG / f'stage3_s{seed}_w{index}.log').open('w', encoding='utf-8')
            processes.append(subprocess.Popen(
                [PY, '-u', str(RUN / 'stage2_five_folds.py')] + targets, env=env,
                stdout=handle, stderr=subprocess.STDOUT, cwd=REPO))
    for process in processes:
        process.wait()


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    log('orchestrator start')
    if (RUN / 'queue.json').exists():
        done, failed, pending = stage1_status()
        log(f'Stage 1 resuming: {len(done)} done, {len(failed)} failed, {len(pending)} pending')
    else:
        done, failed, pending = [], [], []
    if pending:
        start_stage1()
    reconcile()
    done, failed, pending = stage1_status()
    log(f'Stage 1 finished: {len(done)} completed, {len(failed)} failed')
    if failed:
        log(f'failed configurations (reported, not hidden): {[i["run"] for i in failed]}')
    if pending:
        log(f'WARNING: {len(pending)} configurations still pending; '
            'Stage 2 proceeds on the completed ones and the report marks the rest')
    # Release any queue claims held by the workers so the state file is clean.
    release_all_claims()

    configs = select_stage2()
    start_stage2(configs)
    log('Stage 2 finished')

    targets = pick_stage3()
    if targets:
        start_stage3(targets)
        log('Stage 3 finished')

    run([PY, str(RUN / 'summarize_stage1.py')])
    run([PY, str(RUN / 'summarize_stage2.py')])
    run([PY, str(RUN / 'summarize_stage3.py')])
    run([PY, str(RUN / 'build_final_report.py')])
    log('orchestrator complete; report at FINAL_REPORT.md')


if __name__ == '__main__':
    main()
