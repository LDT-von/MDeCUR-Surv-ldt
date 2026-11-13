"""Throughput probe: how does patch_chunk_size change epoch wall time?

This is a cost/throughput measurement only.  It runs three epochs on the same
fold with the same seed under whatever machine load exists, so the comparison
between chunk sizes is paired even if the absolute numbers are inflated by
contention from other projects.

The probe also records whether the validation trajectory changes with chunk
size.  It is expected to change, because the feature-noise RNG is keyed on the
chunk row bounds; that is a documented property of the streaming trainer, not a
bug.  This probe therefore measures cost only and is NOT a protocol change.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from search_config import (REPO as _REPO, RUN as _RUN, FOLD as _FOLD,
                           FOLD_DIR as _FOLD_DIR, GENES as _GENES, PY as _PY,
                           MEASURE as _MEASURE, LD_LIBRARY as _LD_LIBRARY,
                           PATCH_CHUNK_SIZE as _PATCH_CHUNK_SIZE)

import json
import subprocess
import sys
import time
from pathlib import Path

REPO, PY, MEASURE = str(_REPO), _PY, _MEASURE
FOLD, GENES = str(_FOLD), _GENES
OUT = _RUN / 'throughput_probe'
BASE = [
    '--manifest', FOLD, '--genes', GENES, '--device', 'cuda:0',
    '--view-source', 'feature_noise', '--feature-noise', '0.05', '--rna-noise', '0.05',
    '--common-mode', 'stability', '--strength', '2', '--min-weight', '0.1',
    '--interaction-strength', '0.2', '--hidden', '16', '--common', '8',
    '--local-dim', '8', '--local-common', '4',
    '--global-path-within', '0.0005', '--global-rna-within', '0.0005',
    '--global-common', '0.001', '--global-unique', '0.001', '--patch-within', '0.001',
    '--pathway-path-within', '0.0005', '--pathway-rna-within', '0.0005',
    '--pathway-common', '0.001', '--pathway-unique', '0.001',
    '--matched-common', '0.0005', '--matched-unique', '0.0005',
    '--patient-batch-size', '20',
    '--epochs', '3', '--patience', '15', '--seed', '17',
    '--starts', '0', '--horizon', '12', '--time-unit', 'months',
    '--endpoint', 'Dataset-provided OS in months; time origin unverified',
]


def probe(chunk, gpu):
    out = OUT / f'chunk_{chunk}'
    if out.exists():
        return json.loads((out / 'epochs.json').read_text())
    cmd = [PY, '-u', MEASURE, '--out', str(out)] + BASE + ['--patch-chunk-size', str(chunk)]
    env = {'LD_LIBRARY_PATH': '/home/ubuntu/.conda/envs/trisurv/lib',
           'CUDA_VISIBLE_DEVICES': str(gpu), 'PATH': '/usr/bin:/bin'}
    started = time.time()
    with (OUT / f'chunk_{chunk}.log').open('w') as handle:
        subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO)
    epochs = []
    for line in (OUT / f'chunk_{chunk}.log').read_text().splitlines():
        if line.startswith('{'):
            try:
                epochs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    record = {'chunk_size': chunk, 'wall_seconds': time.time()-started,
              'epochs': [{'epoch': e['epoch'], 'seconds': e['seconds'],
                          'validation_nll': e['validation_nll']} for e in epochs]}
    (OUT / 'epochs.json').write_text(json.dumps(record, indent=2))
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = sys.argv[1] if len(sys.argv) > 1 else '1'
    chunks = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [2048, 8192]
    results = []
    for chunk in chunks:
        record = probe(chunk, gpu)
        results.append(record)
        print(f'chunk={chunk:6d} wall={record["wall_seconds"]:8.1f}s '
              f'per_epoch={[round(e["seconds"],1) for e in record["epochs"]]} '
              f'val_nll={[round(e["validation_nll"],4) for e in record["epochs"]]}', flush=True)
    if len(results) == 2:
        a, b = results
        print(f'\nspeedup {a["chunk_size"]}->{b["chunk_size"]}: '
              f'{a["wall_seconds"]/b["wall_seconds"]:.2f}x')
        print(f'val_nll epoch3: {a["epochs"][-1]["validation_nll"]:.6f} vs '
              f'{b["epochs"][-1]["validation_nll"]:.6f} '
              f'(differs because the feature-noise RNG is keyed on chunk row bounds)')


if __name__ == '__main__':
    main()
