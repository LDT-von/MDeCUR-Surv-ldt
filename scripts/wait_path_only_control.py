"""Wait for the ``path_only`` control run and summarize its validation C-index."""
import json
import os
import time

import numpy as np

BASE = '/data1/decur_runs/PARITY_DCT_FOLD0'
LOG = f'{BASE}/ctrl_path_only.log'
SUMMARY = f'{BASE}/ctrl_path_only.summary.json'

deadline = time.time() + 7200
rows = []
while time.time() < deadline:
    if os.path.exists(LOG):
        rows = []
        with open(LOG, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                line = line.strip()
                if line.startswith('{') and '"epoch"' in line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        if len(rows) >= 30:
            break
    time.sleep(60)

scores = [row['c_index'] for row in rows if row.get('c_index') is not None]
summary = {
    'epochs': len(rows),
    'mean': float(np.mean(scores)) if scores else None,
    'median': float(np.median(scores)) if scores else None,
    'max': float(np.max(scores)) if scores else None,
    'last': scores[-1] if scores else None,
    'per_epoch': [{'epoch': row.get('epoch'), 'c_index': row.get('c_index')} for row in rows],
}
with open(SUMMARY, 'w', encoding='utf-8') as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps({key: value for key, value in summary.items() if key != 'per_epoch'}, indent=2))
