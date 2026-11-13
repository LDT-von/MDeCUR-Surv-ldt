"""Build the parity inputs: reference splits, reference endpoint, ordered RNA.

The script reads the reference split table, the reference clinical table and
the reference RNA matrix, then writes the explicit patient manifest that the
streaming trainer consumes.  It reproduces the reference rules:

* the split membership is taken verbatim from the reference split CSV;
* the endpoint is the reference disease-specific survival column and the
  event indicator is ``1 - censorship``;
* the RNA vector keeps the row order of the RNA matrix, because the pathway
  tokenizer indexes that order;
* no validation or test outcome influences any fitted quantity here.

A metadata file records the hash of every source file and the exact counts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build(clinical_csv, split_csv, rna_csv, feature_dir, out_dir, endpoint='dss',
          split_columns=('train', 'val'), feature_suffix='.pt'):
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} already exists and is not empty')
    (out / 'rna').mkdir(parents=True, exist_ok=True)
    clinical = pd.read_csv(clinical_csv)
    rna = pd.read_csv(rna_csv, index_col=0)
    split = pd.read_csv(split_csv)
    missing_columns = set(split_columns) - set(split.columns)
    if missing_columns:
        raise ValueError(f'Split file is missing columns {sorted(missing_columns)}')
    label_column = f'survival_months_{endpoint}'
    censor_column = f'censorship_{endpoint}'
    for column in (label_column, censor_column):
        if column not in clinical.columns:
            raise ValueError(f'Clinical table is missing {column}')
    genes = list(rna.index.astype(str))
    patients = rna.columns.astype(str)

    groups = {name: [p for p in split[name].dropna().astype(str)] for name in split_columns}
    overlap = set()
    names = list(groups)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            overlap |= set(groups[first]) & set(groups[second])
    if overlap:
        raise ValueError(f'Split columns overlap on {len(overlap)} patients')
    # The reference clinical table is reduced by dropping rows with a missing
    # label or censor flag, so a patient without an outcome never enters a
    # cohort.  The same reduction is applied here and recorded below.
    clinical = clinical.dropna(subset=[label_column, censor_column, 'wsi'])
    clinical = clinical.drop_duplicates(subset=['case id'], keep='first')
    known = set(clinical['patient_id'].astype(str))
    dropped = {name: [p for p in members if p not in known]
               for name, members in groups.items()}
    groups = {name: [p for p in members if p in known] for name, members in groups.items()}
    for name, members in groups.items():
        if not members:
            raise ValueError(f'The {name} split is empty after the label reduction')

    rows = []
    stats = {name: {'patients': 0, 'slides': 0, 'events': 0} for name in split_columns}
    for name, members in groups.items():
        for patient in members:
            if patient not in clinical['patient_id'].astype(str).values:
                raise ValueError(f'{patient} is not in the clinical table')
            if patient not in patients:
                raise ValueError(f'{patient} has no RNA vector')
            time = float(clinical.loc[clinical['patient_id'].astype(str) == patient, label_column].iloc[0])
            censored = float(clinical.loc[clinical['patient_id'].astype(str) == patient, censor_column].iloc[0])
            if not np.isfinite(time) or time < 0:
                raise ValueError(f'{patient} has an invalid follow-up time')
            event = 1 - int(censored)
            slides = [s.strip().removesuffix('.svs')
                      for s in str(clinical.loc[clinical['patient_id'].astype(str) == patient, 'wsi'].iloc[0]).split(',')
                      if s.strip()]
            if not slides:
                raise ValueError(f'{patient} has no slides')
            paths = []
            for slide in slides:
                source = Path(feature_dir) / f'{slide}{feature_suffix}'
                if not source.is_file():
                    raise FileNotFoundError(source)
                paths.append(str(source))
            vector = rna[patient].to_numpy(dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f'{patient} RNA is not finite')
            np.save(out / 'rna' / f'{patient}.npy', vector)
            rows.append({'patient_id': patient, 'split': name,
                         'wsi_pt': json.dumps(paths), 'rna_npy': f'rna/{patient}.npy',
                         'time': f'{time:.6f}', 'event': event,
                         'time_exact': repr(float(time))})
            stats[name]['patients'] += 1
            stats[name]['slides'] += len(slides)
            stats[name]['events'] += event

    manifest = out / 'manifest.csv'
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['patient_id', 'split', 'wsi_pt',
                                                    'rna_npy', 'time', 'event'],
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    # The reference pipeline fits the survival bins and the censoring
    # distribution on the clinical values at double precision, while the
    # model input is a float32 tensor.  Keeping the double-precision values
    # in a separate file preserves that distinction exactly.
    reference = out / 'survival_reference.csv'
    with reference.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient_id', 'split', 'time', 'event'])
        for row in rows:
            writer.writerow([row['patient_id'], row['split'],
                             row['time_exact'], int(row['event'])])
    genes_file = out / 'genes.txt'
    genes_file.write_text('\n'.join(genes)+'\n', encoding='utf-8')

    metadata = {
        'purpose': 'reference-protocol parity inputs for a like-for-like comparison',
        'endpoint': f'{endpoint.upper()} (disease-specific survival)',
        'event_definition': f'event = 1 - censorship_{endpoint}',
        'clinical_source': str(Path(clinical_csv).resolve()),
        'clinical_sha256': sha256_file(clinical_csv),
        'split_source': str(Path(split_csv).resolve()),
        'split_sha256': sha256_file(split_csv),
        'split_columns': list(split_columns),
        'patients_dropped_for_missing_labels': {name: len(values)
                                                 for name, values in dropped.items() if values},
        'rna_source': str(Path(rna_csv).resolve()),
        'rna_sha256': sha256_file(rna_csv),
        'gene_count': len(genes),
        'gene_order_file': 'genes.txt',
        'gene_order_sha256': sha256_file(genes_file),
        'feature_source': str(Path(feature_dir).resolve()),
        'feature_note': 'external patch features; encoder provenance is unverified',
        'manifest': str(manifest.resolve()),
        'manifest_sha256': sha256_file(manifest),
        'survival_reference': str(reference.resolve()),
        'survival_reference_sha256': sha256_file(reference),
        'survival_reference_note': ('double-precision clinical follow-up times and event '
                                   'indicators, used for the bin edges, the censoring '
                                   'reference and the metric time grid'),
        'counts': stats,
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                        encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clinical', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--rna', required=True)
    parser.add_argument('--feature-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--endpoint', default='dss', choices=['dss', 'os', 'pfs', 'dfs'])
    parser.add_argument('--split-columns', nargs='+', default=['train', 'val'])
    parser.add_argument('--feature-suffix', default='.pt')
    args = parser.parse_args()
    metadata = build(args.clinical, args.split, args.rna, args.feature_dir, args.out,
                     args.endpoint, tuple(args.split_columns), args.feature_suffix)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
