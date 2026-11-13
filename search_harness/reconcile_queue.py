"""Reconcile the Stage 1 queue with the runs that actually finished.

A worker that is killed while a training run is in flight leaves an orphaned
training process and a queue entry that still says "claimed".  This script
marks such entries as completed when their output directory exists, and
releases entries whose run directory is missing and whose training process is
gone.  It never invents a result: it only reads what is on disk.
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
from pathlib import Path

from stage1_queue import QUEUE, write_queue  # noqa: E402

RUN = _RUN


def running_configurations():
    """Map patient-visible run directories to the training processes alive."""
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
    from stage1_queue import read_queue
    items = read_queue()
    alive = running_configurations()
    changed = 0
    for item in items:
        out = RUN / item['run']
        has_report = (out / 'report.json').exists()
        training = str(out) in alive
        if item.get('status') is None:
            if has_report:
                item['status'] = 'completed'
                item['reconciled'] = True
                changed += 1
            elif item.get('claimed_by') is not None and not training:
                # The worker died before the run produced anything.
                item['claimed_by'] = None
                item['reconciled'] = 'released_without_result'
                changed += 1
    write_queue(items)
    done = sum(1 for i in items if i.get('status') == 'completed')
    failed = sum(1 for i in items if i.get('status') == 'failed')
    pending = sum(1 for i in items if i.get('status') is None)
    print(f'reconciled {changed} entries: {done} completed, {failed} failed, {pending} pending')


if __name__ == '__main__':
    main()
