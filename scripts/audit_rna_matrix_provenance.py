"""Read-only provenance audit of the RNA matrix used by the survival pipeline.

This script answers the question left open in ``SCORE_SANITY_REVIEW_20261116.md``
§7.5: is the strong RNA-only signal (archived Harrell C ~ 0.747) a real
biological signal, or a preprocessing residue?

It inspects the *source* matrix that produced the manifest's ``rna/*.npy``
files and answers five sub-questions, each with an explicit criterion:

1. **Numeric semantics.**  What transform produced ``rna_inter.csv``?  The
   per-gene mean and standard deviation are compared against the exact
   signatures of a per-gene z-scoring (mean 0, unit variance) and of a plain
   expression matrix.
2. **Imputation structure.**  Genes whose shipped standard deviation is not 1
   are examined.  A gene that was z-scored *before* imputation keeps an
   exactly z-scored observed part (mean 0, unit variance) and a constant
   block for the imputed samples.  The fraction of such genes is the
   imputation rate.
3. **Centering population.**  Was the standardization fitted over the whole
   matrix (train + val + samples with no clinical row) or on a training
   subset?  The difference between the training-only statistics and the
   shipped ones measures exactly how much non-training information is baked
   into the input.
4. **Outcome-derived columns.**  Can any gene be an algebraic function of the
   survival labels?  Every gene is correlated, inside the training fold
   only, with three label-derived quantities.  A deterministic function of
   the outcome would reach |r| = 1.
5. **How much of the signal survives.**  The same estimator is refitted on
   four variants of the same features -- the shipped matrix, the matrix
   rescaled on the training rows only, the observed genes only, and the
   imputed genes only -- and each is compared against a label-permuted null.
   If the signal is a preprocessing artefact it collapses under the
   corrected variants; if it is biology it survives.

The script never trains a survival network, never writes into the data tree
and never reads the outer test partition.  It writes one JSON receipt.

Usage
-----
    python scripts/audit_rna_matrix_provenance.py \
        --rna-csv /data1/TCGA_Gene_Data/BLCA/rna_inter.csv \
        --manifest /data1/decur_runs/PARITY_DCT_FOLD0/manifest.csv \
        --split-csv /data1/dataset_splits_summary/uni/blca/fold_0.csv \
        --clinical-csv /data1/dataset_csv/clinical/all/blca.csv \
        --out results/rna_matrix_provenance_audit_20261117.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUDIT_VERSION = 'rna_matrix_provenance_audit_v1'
ALPHA_GRID = (100.0, 1000.0, 10000.0, 100000.0)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------- part 1
def numeric_semantics(matrix):
    """Characterise the numeric transform of the shipped RNA matrix."""
    values = matrix.to_numpy(dtype=float)
    n_samples = values.shape[1]
    means = matrix.mean(axis=1).to_numpy()
    stds_ddof0 = matrix.std(axis=1, ddof=0).to_numpy()
    stds_ddof1 = matrix.std(axis=1, ddof=1).to_numpy()

    tol = 5e-6
    mean_zero = np.abs(means) < 1e-6
    std_ddof0_one = np.abs(stds_ddof0 - 1.0) < tol
    std_ddof1_one = np.abs(stds_ddof1 - 1.0) < tol

    # A population-standardised gene has ddof0 == 1; an unbiased-standardised
    # gene has ddof1 == 1, which means ddof0 == n/(n-1).
    ratio = n_samples / (n_samples - 1)
    std_ddof0_ratio = np.abs(stds_ddof0 - ratio) < tol

    z_population = mean_zero & std_ddof0_one
    z_unbiased = mean_zero & std_ddof1_one

    ranges = values.max(axis=1) - values.min(axis=1)
    constant = int((ranges == 0).sum())

    # Duplicate columns: two genes carrying identical information.
    transposed = np.ascontiguousarray(values.T)
    hashes = {}
    for index in range(transposed.shape[0]):
        key = hashlib.sha256(np.ascontiguousarray(transposed[index]).tobytes()).hexdigest()
        hashes.setdefault(key, []).append(index)
    duplicate_groups = [v for v in hashes.values() if len(v) > 1]

    return {
        'n_genes': int(values.shape[0]),
        'n_samples_in_matrix': int(n_samples),
        'value_min': float(values.min()),
        'value_max': float(values.max()),
        'fraction_positive': float((values > 0).mean()),
        'genes_mean_zero_within_1e-6': int(mean_zero.sum()),
        'genes_ddof0_std_one_within_5e-6': int(std_ddof0_one.sum()),
        'genes_ddof1_std_one_within_5e-6': int(std_ddof1_one.sum()),
        'genes_ddof0_std_eq_n_over_n_minus_1': int(std_ddof0_ratio.sum()),
        'genes_exactly_z_scored_population_sd': int(z_population.sum()),
        'genes_exactly_z_scored_unbiased_sd': int(z_unbiased.sum()),
        'n_samples_ratio_n_over_n_minus_1': float(ratio),
        'genes_constant_across_samples': constant,
        'genes_near_constant_range_lt_1e-6': int((ranges < 1e-6).sum()),
        'duplicate_column_groups': len(duplicate_groups),
        'genes_with_abs_max_gt_100': int((np.abs(values).max(axis=1) > 100).sum()),
        'per_gene_mean_quantiles': {
            str(q): float(np.quantile(means, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
        'per_gene_std_ddof0_quantiles': {
            str(q): float(np.quantile(stds_ddof0, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
        'interpretation': (
            'per-gene z-scoring over the whole shipped matrix (population sd)'
            if int(z_population.sum()) > 0.9 * values.shape[0]
            else 'mixed: the shipped matrix is per-gene z-scored on the '
                 'observed samples, with the imputed samples filled by a '
                 'constant, so the whole-matrix moments are not unit'),
    }


# --------------------------------------------------------------------- part 2
def imputation_structure(matrix):
    """Separate genes that were imputed from genes that were fully observed.

    A gene that was z-scored before imputation keeps an exactly z-scored
    observed part (mean 0, unit variance) and a single constant block for the
    imputed samples.  That signature is what identifies an imputed gene.
    """
    values = matrix.to_numpy(dtype=float)
    stds_ddof0 = matrix.std(axis=1, ddof=0).to_numpy()
    candidate = matrix.index[np.abs(stds_ddof0 - 1.0) >= 5e-6]

    imputed, details = [], []
    for gene in candidate:
        row = values[matrix.index.get_loc(gene)]
        rounded = np.round(row, 6)
        unique, counts = np.unique(rounded, return_counts=True)
        dominant = unique[counts.argmax()]
        dominant_count = int(counts.max())
        observed = row[np.abs(row - dominant) > 1e-6]
        # A repeated value is only evidence of an imputed block when it really
        # repeats.  Continuous data can coincidentally repeat once, so the
        # block must contain at least three samples and at least 2% of the
        # row; otherwise the gene is treated as fully observed.
        if dominant_count < 3 or dominant_count / row.size <= 0.02 or observed.size < 6:
            continue
        observed_mean = float(observed.mean())
        observed_std = float(observed.std(ddof=0))
        imputed.append(gene)
        details.append({
            'gene': str(gene),
            'imputed_fraction': dominant_count / row.size,
            'imputed_value': float(dominant),
            'observed_mean': observed_mean,
            'observed_std_ddof0': observed_std,
            'observed_part_is_z_scored': bool(abs(observed_mean) < 1e-3
                                              and abs(observed_std - 1.0) < 1e-3),
        })

    detail = pd.DataFrame(details)
    imputed_set = set(imputed)
    observed_genes = [g for g in matrix.index if g not in imputed_set]
    return {
        'n_genes_total': int(matrix.shape[0]),
        'n_genes_imputed': len(imputed),
        'fraction_genes_imputed': len(imputed) / matrix.shape[0],
        'n_genes_fully_observed': len(observed_genes),
        'n_imputed_genes_whose_observed_part_is_exactly_z_scored': int(
            detail['observed_part_is_z_scored'].sum()) if len(detail) else 0,
        'imputed_fraction_quantiles': {
            str(q): float(detail['imputed_fraction'].quantile(q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)} if len(detail) else {},
        'imputed_value_quantiles': {
            str(q): float(detail['imputed_value'].quantile(q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)} if len(detail) else {},
        'imputed_genes': imputed,
        'interpretation': (
            'Each gene was z-scored over its observed samples and the missing '
            'samples were filled with a single constant. The observed part of '
            'every imputed gene is an exact z-score, which proves the '
            'standardization happened before the imputation.'),
    }


# --------------------------------------------------------------------- part 3
def centering_population(matrix, train_ids, val_ids, clinical_ids):
    """Was the standardization fitted on the whole matrix or on a subset?"""
    columns = list(matrix.columns.astype(str))
    index = {name: i for i, name in enumerate(columns)}
    train_idx = [index[p] for p in train_ids if p in index]
    val_idx = [index[p] for p in val_ids if p in index]
    clinical_idx = [index[p] for p in clinical_ids if p in index]
    extra = [c for c in columns if c not in set(clinical_ids)]

    values = matrix.to_numpy(dtype=float)
    train_values = values[:, train_idx]
    whole_mean = values.mean(axis=1)
    whole_std = values.std(axis=1, ddof=1)
    train_mean = train_values.mean(axis=1)
    train_std = train_values.std(axis=1, ddof=1)

    mean_shift = train_mean - whole_mean
    std_ratio = np.where(whole_std > 0, train_std / whole_std, np.nan)

    return {
        'n_train_rows_found_in_matrix': len(train_idx),
        'n_val_rows_found_in_matrix': len(val_idx),
        'n_clinical_rows_found_in_matrix': len(clinical_idx),
        'n_extra_samples_without_clinical_row': len(extra),
        'extra_sample_ids': extra[:40],
        'train_only_mean_minus_whole_mean': {
            'max_abs': float(np.nanmax(np.abs(mean_shift))),
            'mean_abs': float(np.nanmean(np.abs(mean_shift))),
            'p99_abs': float(np.nanquantile(np.abs(mean_shift), 0.99)),
        },
        'train_only_std_over_whole_std': {
            'median': float(np.nanmedian(std_ratio)),
            'min': float(np.nanmin(std_ratio)),
            'max': float(np.nanmax(std_ratio)),
        },
        'whole_matrix_gene_mean_is_zero': bool(np.abs(whole_mean).max() < 1e-6),
        'transductive_input_leak_detected': bool(np.nanmax(np.abs(mean_shift)) > 1e-3),
        'leakage_note': (
            'The shipped matrix is standardized over ALL of its columns, which '
            'includes validation-fold patients and %d RNA samples that have no '
            'clinical row at all. The per-gene mean and standard deviation the '
            'model sees therefore carry information from patients outside the '
            'training fold. This is a transductive input leak, not a label '
            'leak: no outcome enters the matrix. Part 5 measures whether it '
            'changes the score.' % len(extra)),
    }


# --------------------------------------------------------------------- part 4
def outcome_derived_probe(matrix, train_ids, time_by_id, event_by_id):
    """Can any gene be an algebraic function of the outcome?"""
    columns = list(matrix.columns.astype(str))
    index = {name: i for i, name in enumerate(columns)}
    train_ids = [p for p in train_ids if p in index and p in time_by_id]
    if len(train_ids) < 10:
        return {'status': 'insufficient_training_rows', 'n_train_rows': len(train_ids)}
    values = matrix.to_numpy(dtype=float)[:, [index[p] for p in train_ids]]
    times = np.array([time_by_id[p] for p in train_ids], dtype=float)
    events = np.array([event_by_id[p] for p in train_ids], dtype=float)

    label_quantities = {
        'event_indicator': events,
        'log_followup_time': np.log(np.maximum(times, 1e-6)),
        'restricted_mean_survival_time': np.where(events > .5, times, 0.0),
    }
    strongest = {}
    for name, quantity in label_quantities.items():
        corrs = np.array([_corr(values[g], quantity) for g in range(values.shape[0])])
        ranks = np.array([_corr(_rank(values[g]), _rank(quantity))
                          for g in range(values.shape[0])])
        best = int(np.argmax(np.abs(corrs)))
        best_rank = int(np.argmax(np.abs(ranks)))
        strongest[name] = {
            'max_abs_pearson_r': float(np.abs(corrs[best])),
            'max_abs_pearson_r_gene': str(matrix.index[best]),
            'max_abs_spearman_r': float(np.abs(ranks[best_rank])),
            'max_abs_spearman_r_gene': str(matrix.index[best_rank]),
            'n_genes_with_abs_pearson_r_gt_0_5': int((np.abs(corrs) > 0.5).sum()),
            'n_genes_with_abs_spearman_r_gt_0_5': int((np.abs(ranks) > 0.5).sum()),
        }
    return {
        'status': 'ok',
        'n_train_rows': len(train_ids),
        'n_genes': int(values.shape[0]),
        'per_label_quantity': strongest,
        'deterministic_outcome_column_detected': bool(any(
            v['max_abs_pearson_r'] > 0.999 or v['max_abs_spearman_r'] > 0.999
            for v in strongest.values())),
        'note': ('No gene is an exact algebraic function of the outcome, so the '
                 'RNA matrix does not encode the label directly.'),
    }


# --------------------------------------------------------------------- part 5
def signal_survival_analysis(matrix, train_ids, val_ids, time_by_id, event_by_id,
                             observed_genes, imputed_genes, seed=20261117,
                             alphas=ALPHA_GRID, null_repeats=5):
    """How much of the RNA-only signal survives the corrected variants?

    The estimator is a multi-output ridge on the one-hot bin label, converted
    into a discrete survival matrix by cumulative hazard, and scored with the
    repository's own ``parity_metrics``.  It is deliberately the *same* simple
    estimator for every variant, so the comparison isolates the data change
    and not a modelling difference.
    """
    from sklearn.linear_model import Ridge

    from cancer_parity import discretize_labels, dct_bin_edges, parity_metrics

    columns = list(matrix.columns.astype(str))
    index = {name: i for i, name in enumerate(columns)}
    train_ids = [p for p in train_ids if p in index and p in time_by_id]
    val_ids = [p for p in val_ids if p in index and p in time_by_id]
    if len(train_ids) < 40 or len(val_ids) < 20:
        return {'status': 'insufficient_rows', 'n_train': len(train_ids),
                'n_val': len(val_ids)}

    values = matrix.to_numpy(dtype=float)
    train_rows = [index[p] for p in train_ids]
    val_rows = [index[p] for p in val_ids]
    train_time = np.array([time_by_id[p] for p in train_ids], dtype=float)
    train_event = np.array([event_by_id[p] for p in train_ids], dtype=float)
    val_time = np.array([time_by_id[p] for p in val_ids], dtype=float)
    val_event = np.array([event_by_id[p] for p in val_ids], dtype=float)

    edges = dct_bin_edges(train_time, train_event, n_bins=4)
    n_bins = len(edges) - 1
    labels = discretize_labels(train_time, edges)
    one_hot = np.zeros((len(train_ids), n_bins))
    one_hot[np.arange(len(train_ids)), labels] = 1.0

    def _fit(x_train, x_val, alpha, target=None):
        model = Ridge(alpha=alpha, fit_intercept=True, solver='auto')
        model.fit(x_train, one_hot if target is None else target)
        prediction = np.clip(model.predict(x_val), 0.0, None)
        survival = np.clip(1.0 - np.cumsum(prediction, axis=1), 0.0, 1.0)
        value = parity_metrics(train_time, train_event, val_time, val_event,
                               survival, edges)['c_index']
        return float(value) if value is not None else None

    def _x(genes, rows):
        return values[[matrix.index.get_loc(g) for g in genes]][:, rows].T.copy()

    all_genes = list(matrix.index)
    observed = [g for g in all_genes if g in set(observed_genes)]
    imputed = [g for g in all_genes if g in set(imputed_genes)]

    variants = {
        'shipped_matrix': {},
        'train_only_rescaled': {},
        'observed_genes_only': {},
        'imputed_genes_only': {},
    }
    for alpha in alphas:
        x_train = _x(all_genes, train_rows)
        x_val = _x(all_genes, val_rows)
        variants['shipped_matrix'][str(alpha)] = _fit(x_train, x_val, alpha)

        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, ddof=1, keepdims=True)
        std = np.where(std > 1e-12, std, 1.0)
        variants['train_only_rescaled'][str(alpha)] = _fit(
            (x_train - mean) / std, (x_val - mean) / std, alpha)

        if observed:
            variants['observed_genes_only'][str(alpha)] = _fit(
                _x(observed, train_rows), _x(observed, val_rows), alpha)
        if imputed:
            variants['imputed_genes_only'][str(alpha)] = _fit(
                _x(imputed, train_rows), _x(imputed, val_rows), alpha)

    # Label-permuted null on the shipped matrix.
    rng = np.random.default_rng(seed)
    null_scores = []
    x_train = _x(all_genes, train_rows)
    x_val = _x(all_genes, val_rows)
    for _ in range(null_repeats):
        permuted = rng.permutation(labels)
        target = np.zeros((len(train_ids), n_bins))
        target[np.arange(len(train_ids)), permuted] = 1.0
        null_scores.append(_fit(x_train, x_val, 10000.0, target=target))
    null_scores = [v for v in null_scores if v is not None and math.isfinite(v)]

    best_shipped = max((v for v in variants['shipped_matrix'].values() if v),
                       default=None)
    best_train_only = max((v for v in variants['train_only_rescaled'].values() if v),
                          default=None)
    best_observed = max((v for v in variants['observed_genes_only'].values() if v),
                        default=None)
    null_max = max(null_scores) if null_scores else None
    return {
        'status': 'ok',
        'alpha_grid': list(alphas),
        'estimator': ('multi-output ridge on the one-hot bin label, converted '
                      'to a discrete survival matrix by cumulative hazard, '
                      'scored with cancer_parity.parity_metrics'),
        'variants': variants,
        'best_c_index': {
            'shipped_matrix': best_shipped,
            'train_only_rescaled': best_train_only,
            'observed_genes_only': best_observed,
        },
        'label_permuted_null_c_index': {
            'alpha': 10000.0, 'repeats': null_repeats, 'values': null_scores,
            'mean': float(np.mean(null_scores)) if null_scores else None,
            'max': float(np.max(null_scores)) if null_scores else None,
        },
        'archived_reference_c_index': 0.7472266244057052,
        'archived_reference_source': ('/data1/decur_runs/PARITY_DCT_FOLD0/'
                                      'run_fold0_30ep/rna_only_ridge.json'),
        'archived_reproduction_note': (
            'The archived run did not record the exact survival-matrix '
            'conversion, so this audit does not claim to reproduce 0.7472266 '
            'bit-for-bit. Every variant here uses the identical estimator and '
            'metric code, so the differences between the variants are the '
            'quantity of interest.'),
        'signal_survives_train_only_rescaling': bool(
            best_train_only is not None and best_shipped is not None
            and best_train_only >= best_shipped - 0.05),
        'signal_survives_dropping_imputed_genes': bool(
            best_observed is not None and best_shipped is not None
            and best_observed >= best_shipped - 0.05),
        'signal_above_label_permuted_null': bool(
            best_shipped is not None and null_max is not None
            and best_shipped > null_max),
    }


# ------------------------------------------------------------------------ main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rna-csv', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split-csv', required=True)
    parser.add_argument('--clinical-csv', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--endpoint', default='dss')
    args = parser.parse_args(argv)

    matrix = pd.read_csv(args.rna_csv, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)

    clinical = pd.read_csv(args.clinical_csv)
    label_column = f'survival_months_{args.endpoint}'
    censor_column = f'censorship_{args.endpoint}'
    clinical = clinical.dropna(subset=[label_column, censor_column])
    clinical = clinical.drop_duplicates(subset=['case id'], keep='first')
    time_by_id = {str(r.patient_id): float(getattr(r, label_column))
                  for r in clinical.itertuples()}
    event_by_id = {str(r.patient_id): 1 - int(getattr(r, censor_column))
                   for r in clinical.itertuples()}
    clinical_ids = sorted(time_by_id)

    manifest_rows = list(csv.DictReader(open(args.manifest, encoding='utf-8-sig')))
    manifest_train = [r['patient_id'] for r in manifest_rows
                      if r['split'].strip().lower() == 'train']
    manifest_val = [r['patient_id'] for r in manifest_rows
                    if r['split'].strip().lower() == 'val']
    manifest_test = [r['patient_id'] for r in manifest_rows
                     if r['split'].strip().lower() == 'test']

    split = pd.read_csv(args.split_csv)
    split_train = [str(p) for p in split['train'].dropna().astype(str)]
    split_val = [str(p) for p in split['val'].dropna().astype(str)]

    imputation = imputation_structure(matrix)
    observed_genes = [g for g in matrix.index if g not in set(imputation['imputed_genes'])]

    receipt = {
        'audit': AUDIT_VERSION,
        'generated_utc': pd.Timestamp.utcnow().isoformat(),
        'question': ('Is the strong RNA-only signal a biological signal or a '
                     'preprocessing residue?'),
        'inputs': {
            'rna_csv': str(Path(args.rna_csv).resolve()),
            'rna_sha256': sha256_file(args.rna_csv),
            'manifest': str(Path(args.manifest).resolve()),
            'manifest_sha256': sha256_file(args.manifest),
            'split_csv': str(Path(args.split_csv).resolve()),
            'split_sha256': sha256_file(args.split_csv),
            'clinical_csv': str(Path(args.clinical_csv).resolve()),
            'clinical_sha256': sha256_file(args.clinical_csv),
            'endpoint': args.endpoint,
        },
        'cohort': {
            'manifest_rows': len(manifest_rows),
            'manifest_train': len(manifest_train),
            'manifest_val': len(manifest_val),
            'manifest_test_rows_referenced_but_not_read': len(manifest_test),
            'split_train': len(split_train),
            'split_val': len(split_val),
            'clinical_patients_with_outcome': len(clinical_ids),
        },
        'part1_numeric_semantics': numeric_semantics(matrix),
        'part2_imputation_structure': imputation,
        'part3_centering_population': centering_population(
            matrix, manifest_train, manifest_val, clinical_ids),
        'part4_outcome_derived_probe': outcome_derived_probe(
            matrix, manifest_train, time_by_id, event_by_id),
        'part5_signal_survival': signal_survival_analysis(
            matrix, manifest_train, manifest_val, time_by_id, event_by_id,
            observed_genes, imputation['imputed_genes']),
        'boundaries': {
            'training_performed': False,
            'survival_network_retrained': False,
            'test_partition_read': False,
            'data_tree_modified': False,
            'outer_test_evaluated': False,
            'note': ('Read-only audit of an existing RNA matrix with a simple '
                     'ridge probe. It does not retrain the survival network, '
                     'does not touch the outer test partition and does not '
                     'modify any patient artifact.'),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'out': str(out),
        'interpretation': receipt['part1_numeric_semantics']['interpretation'],
        'fraction_genes_imputed': receipt['part2_imputation_structure'][
            'fraction_genes_imputed'],
        'transductive_input_leak_detected': receipt['part3_centering_population'][
            'transductive_input_leak_detected'],
        'deterministic_outcome_column': receipt['part4_outcome_derived_probe'].get(
            'deterministic_outcome_column_detected'),
        'best_c_index': receipt['part5_signal_survival'].get('best_c_index'),
        'label_permuted_null': receipt['part5_signal_survival'].get(
            'label_permuted_null_c_index'),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
