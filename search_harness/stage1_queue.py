"""Dynamic Stage 1 scheduler: workers claim the next grid entry from a shared queue.

Each worker takes one configuration at a time from a lock-protected queue file,
so the number of concurrent workers can be changed at any moment without losing
or duplicating work.  Completed runs are skipped on restart.

Usage:
    WORKER_ID=0 PHYSICAL_GPU=0 python stage1_queue.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from search_config import (REPO as _REPO, RUN as _RUN, FOLD as _FOLD,
                           FOLD_DIR as _FOLD_DIR, GENES as _GENES, PY as _PY,
                           MEASURE as _MEASURE, LD_LIBRARY as _LD_LIBRARY,
                           PATCH_CHUNK_SIZE as _PATCH_CHUNK_SIZE)

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

RUN = _RUN
QUEUE = RUN / 'queue.json'
LOCK = RUN / 'queue.lock'
from stage1_screen import GRID, build_command  # noqa: E402

REPO, PY, MEASURE = str(_REPO), _PY, _MEASURE


def read_queue():
    """Read the queue.  Callers must hold the lock when they intend to write."""
    if not QUEUE.exists():
        items = [{'run': name, 'overrides': values} for name, values in GRID]
        write_queue(items)
        return items
    try:
        return json.loads(QUEUE.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        # A truncated file means the last writer was interrupted.  Rebuild the
        # queue from the grid so no configuration is silently lost.
        items = [{'run': name, 'overrides': values} for name, values in GRID]
        for item in items:
            if (RUN / item['run'] / 'report.json').exists():
                item['status'] = 'completed'
        write_queue(items)
        return items


def write_queue(items):
    """Atomically replace the queue file so readers never see a partial write."""
    temporary = QUEUE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(items, indent=2), encoding='utf-8')
    os.replace(temporary, QUEUE)


def claim(worker):
    """Claim the next pending entry.

    An entry whose run directory already exists is completed and is therefore
    not claimable; an entry whose training process is still alive (for example
    one left behind by a killed worker) is also not claimable.  This makes the
    queue safe to repair at any time.

    Returns the claimed item, or None when nothing is claimable right now.
    The second return value distinguishes "the queue is finished" from
    "everything that is left is still in flight".
    """
    alive = running_configurations()
    in_flight = False
    with LOCK.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            items = read_queue()
            for item in items:
                if item.get('status') is not None:
                    continue
                out = RUN / item['run']
                if out.exists():
                    item['status'] = 'completed'
                    item['reconciled'] = 'output_already_present'
                    write_queue(items)
                    continue
                if str(out) in alive or item.get('claimed_by') == 'in_flight_external':
                    # A training process for this configuration is still
                    # running; leave it to finish and do not duplicate it.
                    item['claimed_by'] = 'in_flight_external'
                    item['claimed_at'] = time.time()
                    in_flight = True
                    continue
                if item.get('claimed_by') is not None:
                    continue
                item['claimed_by'] = worker
                item['claimed_at'] = time.time()
                write_queue(items)
                return item, True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    return None, in_flight


def release(run_id, worker, status):
    with LOCK.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            items = read_queue()
            for item in items:
                if item['run'] == run_id:
                    item['status'] = status
                    item['claimed_by'] = None
                    item['finished_at'] = time.time()
            write_queue(items)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def running_configurations():
    """Map output directories to the training processes currently alive."""
    result = subprocess.run(['ps', '-eo', 'pid,cmd'], capture_output=True, text=True)
    alive = set()
    for line in result.stdout.splitlines():
        if 'run_pathway_with_measure.py' not in line:
            continue
        parts = line.split('--out')
        if len(parts) < 2:
            continue
        alive.add(parts[1].strip().split()[0])
    return alive


def main():
    worker = os.environ.get('WORKER_ID', '0')
    env_gpu = os.environ.get('PHYSICAL_GPU', '0')
    idle_limit = int(os.environ.get('IDLE_LIMIT', '0'))
    idle = 0
    while True:
        item, in_flight = claim(worker)
        if item is None:
            if in_flight:
                # Work remains but it is already being trained elsewhere.
                time.sleep(30)
                continue
            idle += 1
            if idle_limit and idle >= idle_limit:
                print(f'[{worker}] queue empty, exiting', flush=True)
                return
            time.sleep(10)
            continue
        idle = 0
        run_id = item['run']
        out = RUN / run_id
        if out.exists():
            print(f'[{worker}] {run_id}: exists, skipped', flush=True)
            release(run_id, worker, 'skipped_existing')
            continue
        cmd = build_command(run_id, item['overrides'], out)
        env = dict(os.environ)
        env['LD_LIBRARY_PATH'] = _LD_LIBRARY
        env['CUDA_VISIBLE_DEVICES'] = env_gpu
        (RUN / f'command_{run_id}.txt').write_text(' '.join(cmd) + '\n', encoding='utf-8')
        print(f'[{worker}] {run_id}: start on physical GPU{env_gpu}', flush=True)
        started = time.time()
        with (RUN / f'train_{run_id}.log').open('w', encoding='utf-8') as handle:
            proc = subprocess.run(cmd, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT, cwd=REPO)
        elapsed = time.time()-started
        release(run_id, worker, 'completed' if proc.returncode == 0 else 'failed')
        print(f'[{worker}] {run_id}: exit={proc.returncode} wall={elapsed:.1f}s', flush=True)
        if proc.returncode != 0:
            print(f'[{worker}] {run_id}: FAILED, see train_{run_id}.log', flush=True)


if __name__ == '__main__':
    main()
