"""Run train_pathway.py in-process and report GPU/RSS/time peak statistics."""
import os
import resource
import runpy
import sys
import time
from pathlib import Path

import torch

REPO = os.environ.get('SEARCH_REPO', str(Path(__file__).resolve().parent))
sys.path.insert(0, REPO)
sys.argv = ['train_pathway.py'] + sys.argv[1:]

# torch 2.10 raises "Invalid device argument" if reset_peak_memory_stats is the
# first CUDA call; initialise the context first so the unmodified trainer runs.
if torch.cuda.is_available():
    torch.zeros(1, device='cuda:0')
    torch.cuda.synchronize()

start = time.time()
try:
    runpy.run_path(f'{REPO}/train_pathway.py', run_name='__main__')
finally:
    elapsed = time.time() - start
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'MEASURE wall_seconds={elapsed:.2f} cpu_peak_rss_mb={rss_mb:.1f}')
    if torch.cuda.is_available():
        print(f'MEASURE gpu_peak_allocated_mb={torch.cuda.max_memory_allocated(0)/2**20:.1f} '
              f'gpu_peak_reserved_mb={torch.cuda.max_memory_reserved(0)/2**20:.1f} '
              f'gpu_device={torch.cuda.get_device_name(0)}')
