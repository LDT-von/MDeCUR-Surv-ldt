"""Summarize paired, same-seed synthetic comparisons after a completed run."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (('full', 'no_interaction'), ('no_interaction', 'no_residual'),
         ('full', 'cross_off'), ('full', 'nature_clr'), ('full', 'original_60_steps'),
         ('full', 'intercept'))


def main():
    source = ROOT/'results'/'method3_1_controlled_20261116.json'
    data = json.loads(source.read_text(encoding='utf8'))
    result = {}
    for case in data['cases']:
        result[case] = {}
        rows = [x for x in data['runs'] if x['case'] == case and 'failure' not in x]
        for left, right in PAIRS:
            diffs = []
            for row in rows:
                arms = {a['arm']: a['test_nll'] for a in row['outcomes']}
                diffs.append(arms[left] - arms[right])
            interval = None
            if len(diffs) >= 2:
                m = float(np.mean(diffs))
                spread = float(stats.t.ppf(.975, len(diffs)-1)*stats.sem(diffs))
                interval = [m-spread, m+spread]
            result[case][f'{left}_minus_{right}'] = {'mean': float(np.mean(diffs)) if diffs else None,
                'per_seed': diffs, 'paired_t95_interval': interval,
                'interpretation': 'Negative values favor the first arm; six-seed interval is descriptive.'}
    out = {'source': source.name, 'summary': result,
           'caveat': 'Differences are paired by generated seed. These are synthetic diagnostics, not clinical inference.'}
    target = ROOT/'results'/'method3_1_pairwise_20261116.json'
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    print(target)


if __name__ == '__main__':
    main()
