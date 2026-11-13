"""Wait for the auxiliary-loss comparison to finish, then summarize it."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path('/data1/decur_runs/PARITY_DCT_FOLD0')
DEADLINE = time.time() + 21600


def epoch_count(path):
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('{') and '"epoch"' in line:
                count += 1
    return count


while time.time() < DEADLINE:
    on = epoch_count(BASE/'aux_on.log')
    off = epoch_count(BASE/'aux_off.log')
    if on >= 30 and off >= 30:
        break
    time.sleep(120)

subprocess.run([sys.executable, '/data1/MDeCUR-Surv-ldt/scripts/summarize_parity_controls.py'],
               check=False)
for tag in ('aux_on', 'aux_off'):
    path = BASE/f'{tag}/report.json'
    if path.exists():
        report = json.loads(path.read_text(encoding='utf-8'))
        print(tag, 'best_epoch', report.get('best_epoch'),
              'best_c_index', report.get('best_validation_c_index'))
