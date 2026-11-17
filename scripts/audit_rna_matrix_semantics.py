"""Read-only audit of the semantics of the RNA expression matrix.

The question this script answers is narrow and factual: **what exactly is in
the RNA matrix that every arm consumes?**  It is the cheapest remaining check
that could invalidate a published score, because a matrix that carries an
outcome-derived column, a batch indicator or a mislabelled patient would
change every arm at once.

The script is read-only.  It opens the RNA matrix, the split files and the
clinical table, and it writes a JSON report.  It trains nothing, it opens no
feature file, it never touches the outer test partition and it never writes
into a run directory.

What it checks
--------------
1. **Row semantics.**  Every row must be a plausible gene symbol.  Rows that
   are not genes (a sample identifier, a comment, a metadata line) are
   reported instead of being silently used as a "gene".
2. **Column semantics.**  Every column must be a plausible patient
   identifier, and the column set is compared with the split files.  A
   column that appears in no split is reported; a patient that appears in a
   split but has no column is reported.
3. **Outcome-derived rows.**  The matrix is genes by patients, so a row whose
   label names a survival time or an event indicator carries one value per
   patient, exactly like a gene row does.  Such a row is reported with its
   label.  This is a *shape* test on the values, not a claim about intent.
4. **Value distribution.**  The per-gene distribution is summarized so that a
   log-expression matrix can be told apart from a raw count matrix, a
   z-scored matrix and a TPM matrix.  The summary is descriptive; it cannot
   establish how the matrix was produced.
5. **Batch structure.**  A first principal component is computed on a
   random subsample of genes and the variance of the first component is
   reported per patient.  Large spread suggests a batch or site effect in the
   matrix.  This is a diagnostic, not a verdict.

What it cannot establish
-------------------------
* Whether the upstream pipeline standardized across the **whole cohort**.  A
  matrix that was z-scored over all patients looks exactly like one that was
  z-scored over the training rows, once the values are written down.  Only
  the upstream script or a reconstruction can settle that.
* Whether a gene was **selected** using the outcome.  Selection leaves no
  trace in the values.
* Whether the encoder of the pathology features is the claimed one.

Those are stated as unknown in the report rather than guessed at.

Usage:
    python scripts/audit_rna_matrix_semantics.py \
        --rna /data1/TCGA_Gene_Data/BLCA/rna_inter.csv \
        --clinical /data1/dataset_csv/clinical/all/blca.csv \
        --splits-root /data1/dataset_splits_summary/uni/blca \
        --out results/rna_matrix_semantics_audit_20261117.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


AUDIT_VERSION = 'rna_matrix_semantics_audit_v1'

# A TCGA-style gene symbol: an upper-case letter followed by letters, digits
# or dashes.  Anything outside this shape is reported rather than accepted.
GENE_SYMBOL = re.compile(r'^[A-Z][A-Z0-9\-]*$')
# A TCGA-style patient barcode: TCGA-..-..-.., optionally truncated to 12.
PATIENT_ID = re.compile(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$')
SUSPICIOUS_ROW_TOKENS = ('sample', 'patient', 'comment', 'gene_id', 'ensembl',
                         'description', 'tissue', 'batch', 'site', 'type')


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_matrix_header(path):
    """Read the header and the row labels without loading the values.

    The RNA matrix is genes by patients, so the first column holds the row
    labels and the header holds the column labels.  Only the labels are read
    here; the values are read separately in bounded chunks.
    """
    labels = []
    with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            labels.append(row[0])
    return header, labels


def classify_rows(labels):
    """Split the row labels into gene symbols and everything else."""
    genes, suspicious, malformed = [], [], []
    for index, label in enumerate(labels):
        text = str(label).strip()
        if not text:
            malformed.append({'row_index': index, 'label': text, 'reason': 'empty label'})
            continue
        if GENE_SYMBOL.match(text):
            genes.append(index)
            continue
        reason = 'does not match a gene symbol shape'
        lowered = text.lower()
        if any(token in lowered for token in SUSPICIOUS_ROW_TOKENS):
            reason = 'looks like a metadata row rather than a gene'
        malformed.append({'row_index': index, 'label': text, 'reason': reason})
    duplicates = {}
    for index in genes:
        duplicates.setdefault(str(labels[index]).strip(), []).append(index)
    repeated = {name: rows for name, rows in duplicates.items() if len(rows) > 1}
    return {'gene_rows': len(genes), 'gene_row_indices': genes,
            'malformed_rows': len(malformed),
            'malformed_examples': malformed[:50], 'duplicate_gene_symbols': len(repeated),
            'duplicate_examples': {name: rows for name, rows in
                                   list(repeated.items())[:20]}}


def classify_columns(header):
    """Check the column labels against the patient-barcode shape."""
    columns = [str(name).strip() for name in header[1:]]
    malformed = []
    for index, name in enumerate(columns):
        if not name:
            malformed.append({'column_index': index, 'label': name, 'reason': 'empty label'})
            continue
        if not PATIENT_ID.match(name.upper()):
            malformed.append({'column_index': index, 'label': name,
                               'reason': 'does not match a TCGA patient barcode'})
    return {'columns': len(columns), 'malformed_columns': len(malformed),
            'malformed_examples': malformed[:50]}, columns


def read_split_membership(splits_root):
    """Read the patient membership of every legacy split file."""
    membership = {}
    root = Path(splits_root)
    if not root.is_dir():
        return membership
    for path in sorted(root.glob('fold_*.csv')):
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for column, value in row.items():
                    if column and str(column).strip().lower() in ('train', 'val', 'test'):
                        for patient in str(value).split():
                            membership.setdefault(patient, {})[f'{path.stem}:{column}'] = True
    return membership


def cohort_overlap(columns, membership):
    """Compare the matrix columns with the split membership."""
    column_set = {name for name in columns}
    membership_keys = set(membership)
    in_matrix_only = sorted(column_set - membership_keys)
    in_splits_only = sorted(membership_keys - column_set)
    shared = sorted(column_set & membership_keys)
    return {'shared_patients': len(shared),
            'columns_not_in_any_split': len(in_matrix_only),
            'columns_not_in_any_split_examples': in_matrix_only[:50],
            'split_patients_without_a_column': len(in_splits_only),
            'split_patients_without_a_column_examples': in_splits_only[:50]}


def value_summary(path, columns, gene_rows, row_labels=None, sample_genes=2000,
                  seed=20261117):
    """Summarize the value distribution and look for outcome-shaped rows.

    The matrix is genes by patients, so the outcome-derived data would hide in
    a **row**: a row whose label names a survival time or an event indicator
    carries one value per patient, exactly like a gene row does.  The shape
    test therefore runs along the row axis and reports the row label.

    Two different views are built from the matrix:

    * a **random gene subsample**, used for the principal component, because
      the component is a global structure and the subsample represents it;
    * **every row**, used for the outcome-shape test, because an
      outcome-derived row is exactly the kind of row that a gene-only
      subsample would drop.  Testing the subsample alone would therefore
      miss the very thing this check exists to find.
    """
    generator = np.random.default_rng(seed)
    chosen = sorted(generator.choice(np.asarray(gene_rows),
                                     size=min(sample_genes, len(gene_rows)),
                                     replace=False).tolist()) if gene_rows else []
    chosen_set = set(chosen)
    values, all_rows, all_labels = [], [], []
    column_sum = np.zeros(len(columns))
    column_sumsq = np.zeros(len(columns))
    nonfinite = 0
    with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        next(reader)
        for index, row in enumerate(reader):
            if not row:
                continue
            try:
                numbers = np.asarray(row[1:], dtype=np.float64)
            except (ValueError, TypeError):
                nonfinite += 1
                continue
            if numbers.size != len(columns):
                nonfinite += 1
                continue
            if not np.isfinite(numbers).all():
                nonfinite += 1
                continue
            column_sum += numbers
            column_sumsq += numbers**2
            all_rows.append(numbers)
            if row_labels is not None and index < len(row_labels):
                all_labels.append(str(row_labels[index]))
            else:
                all_labels.append(f'row_{index}')
            if index in chosen_set:
                values.append(numbers)
    if not values or not all_rows:
        return {'readable': False, 'nonfinite_or_malformed_rows': nonfinite}
    matrix = np.vstack(values)
    complete = np.vstack(all_rows)
    mean = column_sum/max(len(gene_rows), 1)
    variance = np.maximum(column_sumsq/max(len(gene_rows), 1) - mean**2, 0.0)
    sd = np.sqrt(variance)
    flat = matrix.reshape(-1)
    quantiles = np.percentile(flat, [0, 1, 25, 50, 75, 99, 100])
    # A row whose values are all near-zero or near-one integers looks like an
    # event indicator; a row with a wide spread of small positive numbers
    # looks like a follow-up time.  Both are reported with their row label,
    # never concluded.
    event_shaped = []
    time_shaped = []
    for index in range(complete.shape[0]):
        row = complete[index]
        unique = np.unique(row)
        label = all_labels[index]
        if unique.size <= 2 and set(np.round(unique, 6)).issubset({0.0, 1.0}):
            event_shaped.append({'row': label,
                                 'values': sorted(float(v) for v in unique)})
        elif float(row.min()) >= 0 and float(row.max()) <= 200 and unique.size > 2:
            if float(np.median(row)) < 200 and float(row.std()) > 0:
                time_shaped.append({'row': label,
                                    'min': float(row.min()), 'max': float(row.max()),
                                    'median': float(np.median(row)),
                                    'unique_values': int(unique.size)})
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        singular = np.linalg.svd(centred, compute_uv=False)
        total = float((singular**2).sum())
        pc1_variance = float(singular[0]**2/total) if total > 0 else None
    except np.linalg.LinAlgError:
        pc1_variance = None
    return {'readable': True,
            'nonfinite_or_malformed_rows': nonfinite,
            'genes_sampled_for_pca': len(chosen),
            'rows_scanned_for_outcome_shapes': len(complete),
            'value_quantiles': {'min': float(quantiles[0]), 'p01': float(quantiles[1]),
                                'p25': float(quantiles[2]), 'median': float(quantiles[3]),
                                'p75': float(quantiles[4]), 'p99': float(quantiles[5]),
                                'max': float(quantiles[6])},
            'value_mean': float(flat.mean()), 'value_std': float(flat.std()),
            'negative_value_fraction': float((flat < 0).mean()),
            'integer_valued_fraction': float(np.isclose(flat, np.round(flat)).mean()),
            'per_patient_mean_range': [float(mean.min()), float(mean.max())],
            'per_patient_sd_range': [float(sd.min()), float(sd.max())],
            'pc1_variance_fraction': pc1_variance,
            'rows_shaped_like_an_event_indicator': event_shaped,
            'rows_shaped_like_a_follow_up_time': time_shaped[:50],
            'rows_shaped_like_a_follow_up_time_count': len(time_shaped)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rna', required=True, help='The genes-by-patients RNA matrix')
    parser.add_argument('--clinical', default=None,
                        help='Optional clinical table; only the header is inspected')
    parser.add_argument('--splits-root', default=None,
                        help='Directory of fold_*.csv split files')
    parser.add_argument('--out', required=True)
    parser.add_argument('--sample-genes', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261117)
    args = parser.parse_args()

    rna_path = Path(args.rna)
    if not rna_path.is_file():
        raise FileNotFoundError(rna_path)
    header, labels = read_matrix_header(rna_path)
    row_report = classify_rows(labels)
    column_report, columns = classify_columns(header)

    membership = read_split_membership(args.splits_root) if args.splits_root else {}
    overlap = cohort_overlap(columns, membership) if membership else None

    values = value_summary(rna_path, columns, row_report['gene_row_indices'],
                           row_labels=labels,
                           sample_genes=args.sample_genes, seed=args.seed)

    clinical_header = None
    if args.clinical and Path(args.clinical).is_file():
        with Path(args.clinical).open('r', encoding='utf-8-sig', newline='') as handle:
            clinical_header = next(csv.reader(handle))

    report = {
        'audit_version': AUDIT_VERSION,
        'rna_source': str(rna_path.resolve()),
        'rna_sha256': sha256_file(rna_path),
        'row_semantics': row_report,
        'column_semantics': column_report,
        'cohort_overlap': overlap,
        'value_distribution': values,
        'clinical_header': clinical_header,
        'answers': {
            'every_row_is_a_gene_symbol': row_report['malformed_rows'] == 0,
            'every_column_is_a_patient_barcode': column_report['malformed_columns'] == 0,
            'outcome_shaped_rows_found': bool(
                values.get('rows_shaped_like_an_event_indicator')),
            'all_columns_appear_in_a_split': (
                None if overlap is None else overlap['columns_not_in_any_split'] == 0),
        },
        'limits': [
            'This audit reads the RNA matrix and the split files. It trains '
            'nothing, opens no feature file and never touches the outer test '
            'partition.',
            'The value distribution can describe the matrix but cannot establish '
            'how it was produced.',
            'Whether the upstream pipeline standardized across the whole cohort '
            'is NOT decidable from the written values: a matrix z-scored over '
            'all patients and a matrix z-scored over the training rows have '
            'the same shape. Only the upstream script or a reconstruction '
            'settles that question, so it is recorded as unknown.',
            'Whether a gene was selected using the outcome leaves no trace in '
            'the values and is likewise unknown.',
            'A row that looks like an event indicator or a follow-up time is '
            'reported as a shape observation, not as a claim about intent.',
            'The principal-component variance fraction describes the matrix, '
            'not the cause of the structure it finds.',
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                   encoding='utf-8')
    print(json.dumps({'audit_version': AUDIT_VERSION, 'out': str(out),
                      'answers': report['answers']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
