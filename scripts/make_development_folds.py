#!/usr/bin/env python3
"""Create event-stratified inner folds solely from an existing development manifest.

The original split is consulted for patient IDs only. No RNA or WSI feature is
opened. Output directories must be new; every fold has a train/val-only CSV.
"""
import argparse
import csv
import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'Invalid or duplicate CSV headers: {path}')
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f'CSV row has excess fields: {path}')
        return reader.fieldnames, rows


def split_ids(rows, name):
    values = [row[name].strip() for row in rows if row[name] and row[name].strip()]
    if len(values) != len(set(values)):
        raise ValueError(f'Duplicate patient ID in original {name}')
    return set(values)


def resolve_paths(row, source_dir):
    """Keep every field; make feature paths independent of the new CSV location."""
    row = dict(row)
    for key in ('rna_npy', 'wsi_npy', 'wsi_h5', 'wsi_pt'):
        if key not in row or not row[key]:
            continue
        text = row[key]
        def absolute(value):
            path = Path(value)
            return str(path if path.is_absolute() else (source_dir / path).resolve())
        if text.lstrip().startswith('['):
            values = json.loads(text)
            if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
                raise ValueError(f'Invalid path list in {key}')
            if any(not Path(v).is_absolute() for v in values):
                row[key] = json.dumps([absolute(v) for v in values], ensure_ascii=False)
        elif not Path(text).is_absolute():
            row[key] = absolute(text)
    return row


def make_folds(manifest, original_split, output, n_folds, seed):
    manifest, original_split, output = map(Path, (manifest, original_split, output))
    headers, rows = read_csv(manifest)
    split_headers, split_rows = read_csv(original_split)
    if not {'patient_id', 'split', 'event'} <= set(headers):
        raise ValueError('Manifest must contain patient_id, split and event')
    if not {'inner_train', 'inner_val', 'outer_test'} <= set(split_headers):
        raise ValueError('Original split must contain inner_train, inner_val and outer_test')
    original = {key: split_ids(split_rows, key) for key in ('inner_train', 'inner_val', 'outer_test')}
    if any(not group for group in original.values()):
        raise ValueError('Original patient groups must be nonempty')
    if (original['inner_train'] & original['inner_val'] or
        original['inner_train'] & original['outer_test'] or
        original['inner_val'] & original['outer_test']):
        raise ValueError('Original split groups overlap')
    by_id = {}
    for row in rows:
        pid = (row['patient_id'] or '').strip()
        group = (row['split'] or '').strip()
        if not pid or pid in by_id or group not in ('train', 'val'):
            raise ValueError('Manifest requires unique patient IDs and train/val groups')
        if pid not in original['inner_' + group]:
            raise ValueError('Manifest patient does not match original fold0 development split')
        if row['event'] not in ('0', '1'):
            raise ValueError('Manifest event must be 0 or 1')
        by_id[pid] = resolve_paths(row, manifest.resolve().parent)
    if set(by_id) != original['inner_train'] | original['inner_val']:
        raise ValueError('Manifest must exactly cover original fold0 development patients')
    if n_folds < 2:
        raise ValueError('Need at least two folds')
    strata = {event: sorted(pid for pid, row in by_id.items() if row['event'] == event)
              for event in ('0', '1')}
    if any(len(group) < n_folds for group in strata.values()):
        raise ValueError('Each event stratum must contain at least n_folds patients')
    rng = random.Random(seed)
    validation = [set() for _ in range(n_folds)]
    offset = 0
    for group in strata.values():
        rng.shuffle(group)
        for i, pid in enumerate(group):
            validation[(offset + i) % n_folds].add(pid)
        offset = (offset + len(group)) % n_folds
    all_ids = set(by_id)
    if Counter(pid for fold in validation for pid in fold) != Counter(all_ids):
        raise AssertionError('Validation folds must cover each development patient exactly once')
    for val in validation:
        train = all_ids - val
        if train & val or (train | val) & original['outer_test'] or train | val != all_ids:
            raise AssertionError('Fold isolation failure')

    output.mkdir(parents=True, exist_ok=False)
    report = {'manifest': str(manifest.resolve()), 'manifest_sha256': digest(manifest),
              'original_split': str(original_split.resolve()), 'original_split_sha256': digest(original_split),
              'script_sha256': digest(Path(__file__)), 'python_version': platform.python_version(),
              'seed': seed, 'n_folds': n_folds, 'development_patients': len(all_ids),
              'outer_test_ids_checked': len(original['outer_test']),
              'feature_files_opened': False, 'rna_preprocessing': 'fit independently by each training run',
              'path_handling': 'relative feature paths resolved against original manifest directory',
              'folds': []}
    for index, val in enumerate(validation):
        target = output / f'fold_{index}.csv'
        counts = {'train': {'patients': 0, 'events': 0}, 'val': {'patients': 0, 'events': 0}}
        with target.open('x', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            for pid in sorted(all_ids):
                row = dict(by_id[pid])
                row['split'] = 'val' if pid in val else 'train'
                writer.writerow(row)
                counts[row['split']]['patients'] += 1
                counts[row['split']]['events'] += int(row['event'])
        report['folds'].append({'file': target.name, 'sha256': digest(target), **counts})
    with (output / 'metadata.json').open('x', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--original-split', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(make_folds(args.manifest, args.original_split, args.out, args.folds, args.seed), indent=2))


if __name__ == '__main__':
    main()
