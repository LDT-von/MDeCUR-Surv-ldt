"""Compact progress view for the hyperparameter search.

Reports, per Stage 1 configuration: whether it finished, the validation NLL
and C-index, and the elapsed wall time, plus queue depth.  Intended to be run
repeatedly while the workers are training.
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

RUN = _RUN


def main():
    queue = RUN / 'queue.json'
    if queue.exists():
        items = json.loads(queue.read_text(encoding='utf-8'))
        pending = [i for i in items if i.get('status') is None]
        done = [i for i in items if i.get('status') == 'completed']
        failed = [i for i in items if i.get('status') == 'failed']
        print(f'queue: {len(done)} completed, {len(failed)} failed, {len(pending)} pending')
        if failed:
            print('failed:', ', '.join(i['run'] for i in failed))
    print()

    rows = []
    for path in sorted(RUN.iterdir()):
        if not path.is_dir() or not (path / 'report.json').exists():
            continue
        data = json.loads((path / 'report.json').read_text(encoding='utf-8'))
        rows.append({
            'run': path.name,
            'c': data['evaluation_c_index_risk_at_horizon']['value'],
            'brier': data['evaluation_ipcw_brier_at_horizon']['value'],
            'nll': data['validation_nll'],
            'best_ep': data['best_epoch'],
            'epochs': len(data['history']),
            'minutes': data['runtime']['wall_seconds']/60,
            'starts': data['interval_starts'],
        })
    rows.sort(key=lambda x: x['nll'])
    print(f'{"run":32s} {"NLL":>9s} {"C":>7s} {"Brier":>7s} {"best_ep":>7s} {"min":>6s}  starts')
    for r in rows:
        print(f'{r["run"]:32s} {r["nll"]:9.4f} {r["c"]:7.4f} {r["brier"]:7.4f} '
              f'{r["best_ep"]:7d} {r["minutes"]:6.1f}  {r["starts"]}')


if __name__ == '__main__':
    main()
