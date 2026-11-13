"""Keep the single-view feature pages resident in the OS page cache.

The 29 GB of single-view .pt features are mmapped by every training run.  When
several workers run at once, or when another project on the same host takes
memory, the pages get evicted and every epoch starts paying disk reads again
(measured: epoch time degrading from 178 s to 355 s with 27.9% iowait).

This daemon re-reads the feature files periodically so the pages stay cached.
It only reads; it writes nothing and changes no data.  It yields when memory
is already tight, so it never causes swapping.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

FEATURE_DIR = Path('/data/CPathPatchFeature/blca/uni/pt_files')
INTERVAL = int(os.environ.get('WARM_INTERVAL', '600'))
# Do not read when less than this much memory is available, to avoid
# contributing to swap pressure.
MIN_AVAILABLE_MB = int(os.environ.get('MIN_AVAILABLE_MB', '12000'))


def available_mb():
    with open('/proc/meminfo') as handle:
        for line in handle:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    return 0


def touch_all():
    import torch
    files = sorted(FEATURE_DIR.glob('*.pt'))
    touched = 0
    for path in files:
        try:
            tensor = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        except Exception:
            continue
        touched += int(tensor.shape[0])
    return len(files), touched


def main():
    print(f'page-cache warmer: interval {INTERVAL}s, min available {MIN_AVAILABLE_MB} MB',
          flush=True)
    while True:
        free = available_mb()
        if free < MIN_AVAILABLE_MB:
            print(f'skipping: only {free} MB available', flush=True)
        else:
            started = time.time()
            count, touched = touch_all()
            print(f'touched {count} files / {touched} patches in {time.time()-started:.0f}s '
                  f'({available_mb()} MB available now)', flush=True)
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
