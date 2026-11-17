"""Create shareable synthetic summaries; retain raw receipts locally unchanged."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('method3_1_gate_study_20261117', 'method3_1_gate_size_study_20261117',
         'method3_1_stability_20261117', 'method3_1_condition_negative_20261117')


def main():
    for name in NAMES:
        source = ROOT / 'results' / (name + '.json')
        raw = source.read_bytes()
        data = json.loads(raw)
        if data.get('synthetic_only') is not True:
            raise ValueError(f'Not marked synthetic-only: {source}')
        groups = []
        for item in data.get('runs', []):
            group = {key: item[key] for key in ('seed', 'case', 'decisions', 'decision',
                     'events', 'split_ids_sha256') if key in item}
            if 'failure' in item:
                group['failure_final_line'] = item['failure'].strip().splitlines()[-1]
            group['outcomes'] = [
                {key: arm[key] for key in ('arm', 'test_nll', 'interaction_active',
                    'interaction_weight_norm', 'interaction_patient_axis_std_by_bin') if key in arm}
                for arm in item.get('outcomes', [])]
            groups.append(group)
        out = {'synthetic_only': True, 'source_raw_sha256': hashlib.sha256(raw).hexdigest(),
               'source_raw_local_only': source.name,
               'omitted': 'Raw patientwise synthetic predictions and traceback paths; original retained locally.',
               'source_sha256': data.get('source_sha256'), 'sizes': data.get('sizes'),
               'failures': data.get('failures'), 'summary': data.get('summary'), 'groups': groups}
        target = ROOT / 'results' / (name + '_public_summary.json')
        with target.open('x', encoding='utf8') as stream:
            json.dump(out, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        print(target.name, target.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
