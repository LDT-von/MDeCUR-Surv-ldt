"""Warm the OS page cache for the single-view patch feature files.

The first epoch of every training run is cold-read bound: the 29 GB of
single-view .pt features are mmapped and the pages are faulted in from disk.
With several concurrent workers each run pays that cost separately, and the
measured first-epoch time roughly doubled under four-way concurrency.

This script reads every feature file once so the pages stay resident in the
page cache and are shared by all workers.  It only reads; it writes nothing and
changes no data.
"""
import os
import sys
from pathlib import Path

import torch

FEATURE_DIR = Path('/data/CPathPatchFeature/blca/uni/pt_files')


def main():
    files = sorted(FEATURE_DIR.glob('*.pt'))
    print(f'{len(files)} feature files under {FEATURE_DIR}', flush=True)
    total = 0
    for index, path in enumerate(files, 1):
        try:
            tensor = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        except Exception as error:  # pragma: no cover - diagnostic script
            print(f'{path.name}: skipped ({error})', flush=True)
            continue
        # Touch the pages so they are resident, without copying them.
        total += int(tensor.shape[0]) * int(tensor.shape[-1]) * 4
        if index % 50 == 0:
            print(f'  {index}/{len(files)} files, {total/2**30:.1f} GiB of feature data touched',
                  flush=True)
    print(f'done: {total/2**30:.1f} GiB of feature data resident', flush=True)
    with open('/proc/meminfo') as handle:
        for line in handle:
            if line.startswith(('MemFree', 'Cached', 'MemAvailable')):
                print(line.strip(), flush=True)


if __name__ == '__main__':
    main()
