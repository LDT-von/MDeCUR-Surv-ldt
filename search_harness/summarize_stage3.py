"""Stage 3: separate fold-to-fold variance from seed-to-seed variance.

The same configuration is trained on all five internal folds under several
seeds.  The spread across folds at a fixed seed is the fold component; the
spread across seeds at a fixed fold is the seed component.  No seed is selected
on performance: every seed is reported.
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
STAGE3 = RUN / 'stage3'


def read_run(path):
    report = path / 'report.json'
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding='utf-8'))
    return {
        'fold': int(path.name.split('_')[-1]),
        'c_index': data['evaluation_c_index_risk_at_horizon']['value'],
        'ipcw_brier': data['evaluation_ipcw_brier_at_horizon']['value'],
        'validation_nll': data['validation_nll'],
        'best_epoch': data['best_epoch'],
    }


def main():
    result = {}
    # Layout written by stage2_five_folds.py:
    #   stage2/<config>/FOLD_<n>            (seed 17)
    #   stage3/<config>_seed<N>/FOLD_<n>    (seeds 29 and 43)
    # The seed-17 result of a configuration is reused for the baseline so the
    # seed spread is computed over three seeds rather than two.
    for root, seed_default in ((RUN / 'stage2', '17'), (STAGE3, None)):
        if not root.exists():
            continue
        for tag_dir in sorted(root.iterdir()):
            if not tag_dir.is_dir():
                continue
            folds = [read_run(tag_dir / f'FOLD_{f}') for f in range(5)]
            present = [f for f in folds if f is not None]
            if not present:
                continue
            name = tag_dir.name
            config, _, seed = name.partition('_seed')
            seed = seed or seed_default
            entry = result.setdefault(config, {'seeds': {}})
            entry['seeds'][seed] = {
                'folds': present,
                'complete': len(present) == 5,
                'mean_c_index': float(np.mean([f['c_index'] for f in present])) if present else None,
                'std_c_index': float(np.std([f['c_index'] for f in present], ddof=1)) if len(present) > 1 else None,
                'mean_validation_nll': float(np.mean([f['validation_nll'] for f in present])) if present else None,
                'source': str(tag_dir),
            }
    for config, entry in result.items():
        means = [s['mean_c_index'] for s in entry['seeds'].values() if s['mean_c_index'] is not None]
        entry['seed_spread_of_mean_c_index'] = (float(max(means)-min(means)) if len(means) > 1 else None)
        entry['std_of_mean_c_index_across_seeds'] = (float(np.std(means, ddof=1)) if len(means) > 1 else None)
    out = RUN / 'stage3_summary.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, entry in result.items():
        print(name)
        for seed, data in entry['seeds'].items():
            if data['folds']:
                folds = ' '.join(f"{f['fold']}:{f['c_index']:.4f}" for f in data['folds'])
                print(f'   seed {seed:6s} {folds}  mean={data["mean_c_index"]:.4f} '
                      f'sd={data["std_c_index"]:.4f} complete={data["complete"]}')
        print(f'   seed spread of mean C: {entry["seed_spread_of_mean_c_index"]}')


if __name__ == '__main__':
    main()
