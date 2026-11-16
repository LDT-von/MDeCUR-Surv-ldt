"""Shared BLCA protocol layer for the decoupling comparators.

One cohort loader, one binning rule, one metric call and one explicit
train-only fitting boundary are shared by every arm (PIBD, CAMR, DeReF, DIMAF,
DCT reference, RNA-only ridge, WSI-only).  The point of this module is that a
comparison cannot silently differ in data preparation: every arm consumes the
same per-fold tensors and the same labels.

Fold scope
----------
The user reduced the schedule from five folds to **one fold** to fit the
available time budget.  ``FOLDS`` is therefore ``(0,)``.  Two consequences
are explicit rather than silent:

* the protocol identifier and version carry the fold count, so a plan built
  under the old five-fold setting can never be mistaken for a one-fold plan;
* the DCT reference archive still holds five folds.  The summary therefore
  reads **only the folds the protocol declares** from that archive and states
  that the reference row is a single-fold value, not a five-fold mean.

Leakage boundary
----------------
Everything that is estimated from data is fitted on the **training rows of the
current fold only**:

* survival quantile bin edges      -> uncensored training times only
* RNA standardization statistics  -> training rows only
* DIMAF pathology prototypes      -> training-fold patch features only
* RNA-only ridge alpha             -> inner cross-validation on training rows only

Validation rows are only ever scored.  The outer test partition is rejected
before any file is opened.

Patient splits are read from the manifest itself; the manifest is authoritative
and is never rewritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from cancer_parity import (dct_bin_edges, discretize_labels, discrete_nll,
                           discrete_survival, dct_risk, parity_metrics,
                           DISCRETE_NLL_VERSION)
from cancer_rna import fit_rna_scaler, transform_rna
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint

PROTOCOL = 'blca_1fold_dss_uni_v1_p4096_dnll_v1'
ENDPOINT = 'DSS'
TIME_UNIT = 'months'
N_BINS = 4
ALPHA_SURV = .5
NUM_PATCHES = 4096
# The schedule was reduced from five folds to one by explicit user decision
# (2026-11-16, time budget).  Only fold 0 is in scope now.
FOLDS = (0,)

# Every arm must agree on these, otherwise the comparison is not a comparison.
SHARED_SETTINGS = {
    'protocol': PROTOCOL,
    'endpoint': ENDPOINT,
    'time_unit': TIME_UNIT,
    'n_bins': N_BINS,
    'alpha_surv': ALPHA_SURV,
    'num_patches': NUM_PATCHES,
    'folds': list(FOLDS),
    'fold_count': len(FOLDS),
    'fold_scope_note': ('The user reduced the schedule from five folds to one. '
                         'Every mean/std over folds is therefore a single-fold '
                         'value with a zero-width spread, not a five-fold '
                         'estimate.'),
    'selection': 'maximum validation Harrell C, earliest exact tie',
    'loss_version': DISCRETE_NLL_VERSION,
    'risk_definition': 'minus the summed survival over the discrete bins',
    'metric_implementation': 'sksurv parity_metrics with train-only censoring reference',
    'outer_test_used': False,
}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


@dataclass
class FoldCohort:
    """One fold, with every fitted quantity traceable to the training rows."""
    fold: int
    manifest: str
    survival_reference: str
    train_ids: list
    val_ids: list
    train_time: np.ndarray
    train_event: np.ndarray
    val_time: np.ndarray
    val_event: np.ndarray
    bin_edges: np.ndarray
    train_labels: np.ndarray
    val_labels: np.ndarray
    rna_scaler: dict
    train_rna: torch.Tensor
    val_rna: torch.Tensor
    train_bags: list
    val_bags: list
    input_sha256: dict = field(default_factory=dict)

    @property
    def patients(self) -> dict:
        return {'train': len(self.train_ids), 'val': len(self.val_ids)}

    @property
    def events(self) -> dict:
        return {'train': int(self.train_event.sum()), 'val': int(self.val_event.sum())}

    def metadata(self) -> dict:
        return {
            'fold': self.fold,
            'manifest': self.manifest,
            'survival_reference': self.survival_reference,
            'patients': self.patients,
            'events': self.events,
            'bin_edges': [None if math.isinf(x) else float(x) for x in self.bin_edges],
            'bin_edges_note': 'outer edges are +/-inf by construction',
            'order_sha256': {'train': json_hash(self.train_ids), 'val': json_hash(self.val_ids)},
            'labels_sha256': {
                'train': json_hash([self.train_time.tolist(), self.train_event.tolist()]),
                'val': json_hash([self.val_time.tolist(), self.val_event.tolist()])},
            'rna_preprocessing': self.rna_scaler.get('rna_preprocessing'),
            'input_sha256': self.input_sha256,
            'fit_boundary': {
                'bin_edges': 'train uncensored times only',
                'rna_scaler': 'train rows only',
                'validation_rows': 'scored only; never fitted',
                'outer_test': 'not loaded',
            },
        }


def _reject_test_rows(manifest):
    """Refuse a manifest that mentions the outer test partition.

    The check runs on the manifest text alone, before any feature file, RNA
    file or label of a test row is opened.
    """
    with Path(manifest).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get('split', '').strip().lower() not in ('train', 'val'):
                raise ValueError('Expected an explicit train/val-only manifest; '
                                 'the outer test partition is rejected BEFORE input I/O')
    return True


def read_survival_reference(path) -> dict:
    """Double-precision follow-up times and event indicators keyed by patient."""
    reference = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            identity = row['patient_id'].strip()
            split = row.get('split', 'train').strip().lower()
            if not identity or identity in reference:
                raise ValueError('Survival reference requires unique nonempty patient IDs')
            if split not in ('train', 'val'):
                raise ValueError('Survival reference must contain train/val rows only')
            time, event = float(row['time']), float(row['event'])
            if not math.isfinite(time) or time < 0 or event not in (0., 1.):
                raise ValueError('Invalid double-precision survival reference')
            reference[identity] = (split, time, event)
    return reference


def load_fold(manifest, survival_reference, fold, *, path_views='single_view',
              include_test=False, rna_scale_policy='none'):
    """Load one fold and fit every train-only quantity.

    ``include_test`` must stay false: the outer test partition is never opened
    by this comparison.
    """
    if include_test:
        raise ValueError('The outer test partition is not used by this protocol')
    if fold not in FOLDS:
        raise ValueError(f'Unknown fold {fold}; expected one of {FOLDS}')
    _reject_test_rows(manifest)
    cohort = load_stream_manifest(manifest, path_views=path_views, include_test=False)
    reference = read_survival_reference(survival_reference)
    expected = {p for split in ('train', 'val') for p in cohort[split]['ids']}
    if set(reference) != expected:
        raise ValueError('Survival reference IDs differ from the manifest cohort')
    for split in ('train', 'val'):
        for index, identity in enumerate(cohort[split]['ids']):
            row = reference[identity]
            if row[0] != split:
                raise ValueError('Survival reference split differs from the manifest')
            if (row[2] != float(cohort[split]['event'][index])
                    or not np.isclose(row[1], float(cohort[split]['time'][index]),
                                     rtol=2e-7, atol=2e-6)):
                raise ValueError('Survival reference labels differ from the manifest')

    train_time = np.array([reference[p][1] for p in cohort['train']['ids']])
    train_event = np.array([reference[p][2] for p in cohort['train']['ids']])
    val_time = np.array([reference[p][1] for p in cohort['val']['ids']])
    val_event = np.array([reference[p][2] for p in cohort['val']['ids']])

    edges = dct_bin_edges(train_time, train_event, N_BINS)
    scaler = fit_rna_scaler(cohort['train']['rna'], rna_scale_policy)
    normalized = {split: transform_rna(data['rna'], scaler)
                  for split, data in cohort.items()}

    result = FoldCohort(
        fold=fold,
        manifest=str(Path(manifest).resolve()),
        survival_reference=str(Path(survival_reference).resolve()),
        train_ids=list(cohort['train']['ids']),
        val_ids=list(cohort['val']['ids']),
        train_time=train_time, train_event=train_event,
        val_time=val_time, val_event=val_event,
        bin_edges=edges,
        train_labels=discretize_labels(train_time, edges),
        val_labels=discretize_labels(val_time, edges),
        rna_scaler=scaler,
        train_rna=normalized['train'], val_rna=normalized['val'],
        train_bags=list(cohort['train']['bags']),
        val_bags=list(cohort['val']['bags']),
        input_sha256={
            'manifest': sha256_file(manifest),
            'survival_reference': sha256_file(survival_reference),
            'cohort_input': stream_cohort_fingerprint(cohort, path_views),
        },
    )
    # The bags stay open: the returned cohort owns them and the caller closes
    # them when the fold is finished.
    return result


def close_bags(cohort) -> None:
    for data in cohort.values():
        for bag in data['bags']:
            if hasattr(bag, 'close'):
                bag.close()


def materialize_bag(bag, num_patches=NUM_PATCHES, chunk_size=NUM_PATCHES):
    """Read the complete fixed patch budget, including explicit zero padding."""
    arrays = [bag.read_block(block, ('features_reference',))[0] for block in bag.blocks(chunk_size)]
    patches = torch.cat(arrays) if arrays else torch.zeros((0, bag.feature_dim))
    if patches.shape[0] != num_patches:
        raise ValueError(f'Expected a {num_patches}-row patch budget, got {patches.shape[0]}')
    return patches


def survival_loss(logits, labels, event, alpha=ALPHA_SURV):
    """Patientwise clamped discrete NLL; censorship is the complement of event."""
    censorship = torch.as_tensor(1-np.asarray(event, dtype=np.float64), dtype=torch.float64)
    return discrete_nll(logits.double(),
                        torch.as_tensor(labels, dtype=torch.long),
                        censorship.to(logits.device), alpha=alpha)


def evaluate_predictions(fold_cohort, survival_bins) -> dict:
    """Score one fold with the shared parity metric call."""
    survival = np.asarray(survival_bins, dtype=np.float64)
    result = parity_metrics(fold_cohort.train_time, fold_cohort.train_event,
                             fold_cohort.val_time, fold_cohort.val_event,
                             survival, fold_cohort.bin_edges)
    result['validation_nll'] = float(survival_loss(
        torch.as_tensor(np.log(np.clip(1-survival, 1e-12, 1-1e-12))),
        fold_cohort.val_labels, fold_cohort.val_event).mean())
    if result['c_index'] is None:
        raise ValueError('No defined validation Harrell C; keep the error, do not substitute')
    return result


def logits_to_survival(logits) -> np.ndarray:
    """Discrete hazard logits -> cumulative survival matrix."""
    return discrete_survival(torch.as_tensor(logits)).detach().cpu().numpy()


def aggregate_folds(records) -> dict:
    """Mean +/- standard deviation of C-index, IBS and iAUC over folds."""
    if not records:
        raise ValueError('No fold records to aggregate')
    keys = ('c_index', 'ibs', 'iauc', 'c_index_ipcw')
    summary = {}
    for key in keys:
        values = [float(r[key]) for r in records if r.get(key) is not None]
        if len(values) != len(records):
            summary[key] = None
            summary[f'{key}_missing'] = len(records)-len(values)
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[key] = float(array.mean())
        summary[f'{key}_std'] = float(array.std(ddof=0))
        summary[f'{key}_values'] = values
    summary['folds'] = [int(r['fold']) for r in records]
    summary['n_folds'] = len(records)
    return summary


def fold_selection_guard(metrics) -> bool:
    """A fold is usable for comparison only when every headline metric exists."""
    return all(metrics.get(key) is not None for key in ('c_index', 'ibs', 'iauc'))
