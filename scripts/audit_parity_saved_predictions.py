"""Audit archived parity predictions without loading features or model weights.

Only the supplied CSV files and report.json are parsed.  checkpoint.pt is read
as opaque bytes for hashing, never unpickled.  This verifies supplied table
consistency and archived Harrell-score reproduction, not absence of every
possible form of leakage or independent-test generalization.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancer_parity import dct_risk, parity_metrics
from cancer_parity_audit import audit_parity_metrics


TIME_ATOL, TIME_RTOL = 1e-5, 1e-6
RISK_ATOL, RISK_RTOL = 1e-7, 1e-6
C_ATOL = 1e-10


class AuditInputError(ValueError):
    """A safe error message containing no patient identifiers."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path, required, label):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required).issubset(fields):
            raise AuditInputError(f'{label}: missing required or duplicate CSV columns')
        rows = list(reader)
    if not rows:
        raise AuditInputError(f'{label}: empty CSV')
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise AuditInputError(f'{label}: malformed CSV row width')
    return rows, fields


def _identity(row, label):
    identity = row['patient_id'].strip()
    if not identity:
        raise AuditInputError(f'{label}: empty patient identifier')
    return identity


def _case_key(identity):
    upper = identity.upper()
    if upper.startswith('TCGA-'):
        if len(upper) < 12:
            raise AuditInputError('Truncated TCGA patient identifier')
        return upper[:12]
    return identity


def _label(row, label):
    try:
        time, event = float(row['time']), float(row['event'])
    except (TypeError, ValueError):
        raise AuditInputError(f'{label}: nonnumeric time or event') from None
    if not np.isfinite(time) or time < 0 or not np.isfinite(event) or event not in (0, 1):
        raise AuditInputError(f'{label}: time must be finite/nonnegative and event binary')
    return time, int(event)


def _same_label(first, second):
    return first[1] == second[1] and bool(np.isclose(first[0], second[0],
                                                  atol=TIME_ATOL, rtol=TIME_RTOL))


def _read_manifest(path, counts):
    rows, _ = _csv_rows(path, ('patient_id', 'split', 'time', 'event'), 'manifest')
    patients, case_owners = {}, {}
    counts['manifest_rows'] = len(rows)
    for row in rows:
        identity = _identity(row, 'manifest')
        split = row['split'].strip()
        if split not in ('train', 'val'):
            raise AuditInputError('manifest: only explicit train/val rows are accepted')
        label = _label(row, 'manifest')
        if identity in patients:
            previous = patients[identity]
            if previous['split'] != split:
                raise AuditInputError('manifest: patient appears in both train and validation')
            if not _same_label(previous['label'], label):
                raise AuditInputError('manifest: repeated patient rows have conflicting labels')
            continue
        canonical = _case_key(identity)
        if canonical in case_owners:
            previous_identity = case_owners[canonical]
            if patients[previous_identity]['split'] != split:
                raise AuditInputError('manifest: TCGA case overlaps train/validation through different IDs')
            raise AuditInputError('manifest: multiple distinct patient IDs map to one TCGA case within a split')
        case_owners[canonical] = identity
        patients[identity] = {'split': split, 'label': label}
    train = [identity for identity, row in patients.items() if row['split'] == 'train']
    val = [identity for identity, row in patients.items() if row['split'] == 'val']
    if not train or len(val) < 2:
        raise AuditInputError('manifest: nonempty training and at least two validation patients are required')
    counts.update(train_patients=len(train), val_patients=len(val),
                  duplicate_manifest_rows=len(rows) - len(patients))
    return patients, train, val


def _read_reference(path, patients, counts):
    rows, _ = _csv_rows(path, ('patient_id', 'time', 'event'), 'survival_reference')
    reference, cases = {}, set()
    for row in rows:
        identity = _identity(row, 'survival_reference')
        canonical = _case_key(identity)
        if identity in reference or canonical in cases:
            raise AuditInputError('survival_reference: duplicate patient or TCGA case')
        reference[identity] = _label(row, 'survival_reference')
        cases.add(canonical)
    if not set(patients).issubset(reference):
        raise AuditInputError('survival_reference: missing manifest patients')
    if any(not _same_label(row['label'], reference[identity]) for identity, row in patients.items()):
        raise AuditInputError('survival_reference: labels disagree with the manifest')
    counts.update(reference_patients=len(reference),
                  unused_reference_patients=len(set(reference) - set(patients)))
    return reference


def _read_report(path):
    try:
        with Path(path).open(encoding='utf-8-sig') as handle:
            report = json.load(handle)
        edges = np.array([float(value) for value in report['bin_edges']], dtype=np.float64)
    except (ValueError, TypeError, KeyError):
        raise AuditInputError('report: invalid JSON object or missing/invalid bin edges') from None
    if edges.ndim != 1 or edges.size < 2 or np.isnan(edges).any() or not np.all(edges[1:] > edges[:-1]):
        raise AuditInputError('report: bin edges must be strictly increasing without NaN')
    if not np.isfinite(edges[1:-1]).all():
        raise AuditInputError('report: internal bin edges must be finite')
    if report.get('n_bins', edges.size - 1) != edges.size - 1:
        raise AuditInputError('report: n_bins disagrees with bin edges')
    return report, edges


def _read_predictions(path, validation_ids, reference, edges, counts):
    rows, fields = _csv_rows(path, ('patient_id', 'risk'), 'predictions')
    expected_columns = [f'survival_bin_{index}' for index in range(len(edges) - 1)]
    actual_columns = {name for name in fields if name.startswith('survival_bin_')}
    if actual_columns != set(expected_columns):
        raise AuditInputError('predictions: survival columns disagree with the report bin count')
    stored = {}
    for row in rows:
        identity = _identity(row, 'predictions')
        if identity in stored:
            raise AuditInputError('predictions: duplicate patient identifier')
        stored[identity] = row
    if set(stored) != set(validation_ids):
        raise AuditInputError('predictions: patient IDs do not exactly equal the manifest validation IDs')
    try:
        survival = np.array([[float(stored[i][column]) for column in expected_columns]
                             for i in validation_ids], dtype=np.float64)
        stored_risk = np.array([float(stored[i]['risk']) for i in validation_ids], dtype=np.float64)
    except (TypeError, ValueError):
        raise AuditInputError('predictions: nonnumeric survival or risk') from None
    if not np.isfinite(survival).all() or np.any((survival < 0) | (survival > 1)):
        raise AuditInputError('predictions: invalid survival probabilities')
    if np.any(np.diff(survival, axis=1) > 1e-12) or not np.isfinite(stored_risk).all():
        raise AuditInputError('predictions: increasing cumulative survival or nonfinite risk')
    risk = dct_risk(survival)
    maximum_error = float(np.max(np.abs(stored_risk - risk)))
    if not np.allclose(stored_risk, risk, atol=RISK_ATOL, rtol=RISK_RTOL):
        raise AuditInputError('predictions: stored risk disagrees with minus summed survival')
    if 'bin_label' in fields:
        try:
            labels = np.array([int(stored[i]['bin_label']) for i in validation_ids])
        except (ValueError, TypeError):
            raise AuditInputError('predictions: invalid stored bin labels') from None
        expected = np.searchsorted(edges, [reference[i][0] for i in validation_ids], side='right') - 1
        if not np.array_equal(labels, expected):
            raise AuditInputError('predictions: stored bin labels disagree with reference times and edges')
    counts['prediction_rows'] = len(rows)
    return survival, stored_risk, maximum_error


def _score(value, description):
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise AuditInputError(f'report: missing or invalid {description}') from None
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise AuditInputError(f'report: invalid {description}')
    return score


def _json_safe_metrics(metrics):
    """Legacy libraries can return NaN without raising; never emit JSON NaN."""
    nonfinite = []

    def convert(value, path):
        if isinstance(value, dict):
            return {key: convert(item, f'{path}.{key}') for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item, f'{path}[{index}]') for index, item in enumerate(value)]
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                nonfinite.append(path)
                return None
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        return value

    safe = convert(metrics, 'legacy_metrics')
    safe.setdefault('errors', []).extend(f'{path}: nonfinite metric replaced by None' for path in nonfinite)
    return safe


def _check_selection(report, recomputed, result):
    try:
        best_score = _score(report['best_validation_c_index'], 'best validation C')
        final_score = _score(report['final_validation']['c_index'], 'final validation C')
        best_epoch, history = report['best_epoch'], report['history']
    except (KeyError, TypeError):
        raise AuditInputError('report: missing checkpoint-selection fields') from None
    if type(best_epoch) is not int or best_epoch < 1 or not isinstance(history, list) or not history:
        raise AuditInputError('report: invalid best epoch or epoch history')
    epochs, finite_history = [], []
    for row in history:
        if not isinstance(row, dict) or type(row.get('epoch')) is not int or row['epoch'] < 1 or 'c_index' not in row:
            raise AuditInputError('report: malformed epoch history')
        epochs.append(row['epoch'])
        if row['c_index'] is not None:
            finite_history.append((row['epoch'], _score(row['c_index'], 'history validation C')))
    if len(set(epochs)) != len(epochs) or epochs != sorted(epochs) or not finite_history:
        raise AuditInputError('report: duplicate/unordered epochs or no valid history C')
    peak = max(score for _, score in finite_history)
    first_peak = min(epoch for epoch, score in finite_history if score == peak)
    selected = dict(finite_history).get(best_epoch)
    result['selection_summary'] = {'history_epochs': len(history), 'reported_best_epoch': best_epoch,
                                   'earliest_history_peak_epoch': first_peak,
                                   'reported_best_validation_c_index': best_score,
                                   'reported_final_validation_c_index': final_score,
                                   'history_peak_c_index': peak,
                                   'history_selected_epoch_c_index': selected}
    if best_epoch != first_peak or selected is None:
        raise AuditInputError('report: best_epoch is not the earliest history maximum')
    if recomputed is None or not all(abs(value - recomputed) <= C_ATOL
                                     for value in (best_score, final_score, peak, selected)):
        raise AuditInputError('report: archived best/final/history C does not reproduce from saved predictions')


def audit_saved_predictions(manifest, survival_reference, reference_run):
    """Return an ID-free audit summary; mismatches have status='failed'."""
    run = Path(reference_run)
    paths = {'manifest': Path(manifest), 'survival_reference': Path(survival_reference),
             'report': run / 'report.json', 'checkpoint_opaque_bytes': run / 'checkpoint.pt',
             'predictions': run / 'val_predictions.csv'}
    result = {
        'protocol': 'saved_parity_predictions_read_only_audit', 'status': 'failed',
        'status_scope': 'supplied table consistency and archived Harrell-score reproduction',
        'counts': {}, 'input_sha256': {}, 'input_sha256_after': {}, 'errors': [],
        'comparison_tolerances': {'time_atol': TIME_ATOL, 'time_rtol': TIME_RTOL,
                                  'risk_atol': RISK_ATOL, 'risk_rtol': RISK_RTOL,
                                  'c_index_atol': C_ATOL},
        'limitations': ['No features, RNA arrays or checkpoint objects are loaded.',
                        'Checkpoint bytes are hashed, not unpickled or used for inference.',
                        'Archived bin edges are used; this rescore does not establish their original fitting provenance.',
                        'This does not establish absence of preprocessing or pretraining leakage.',
                        'Validation-selected scores remain validation-selected after rescoring.',
                        'No permutation significance test or retraining is performed.'],
    }
    try:
        for name, path in paths.items():
            result['input_sha256'][name] = _sha256(path)
        patients, train_ids, val_ids = _read_manifest(paths['manifest'], result['counts'])
        reference = _read_reference(paths['survival_reference'], patients, result['counts'])
        report, edges = _read_report(paths['report'])
        if ('manifest_sha256' in report and
                report['manifest_sha256'] != result['input_sha256']['manifest']):
            raise AuditInputError('report: manifest SHA256 differs from the supplied manifest')
        survival, stored_risk, risk_error = _read_predictions(
            paths['predictions'], val_ids, reference, edges, result['counts'])
        train_time, train_event = (np.array([reference[i][column] for i in train_ids]) for column in (0, 1))
        val_time, val_event = (np.array([reference[i][column] for i in val_ids]) for column in (0, 1))
        result['counts'].update(train_events=int(train_event.sum()), val_events=int(val_event.sum()))
        for key, actual in (('train', len(train_ids)), ('val', len(val_ids))):
            if 'patients' in report and report['patients'].get(key) != actual:
                raise AuditInputError('report: stored patient counts disagree with the supplied tables')
        for key, actual in (('train', int(train_event.sum())), ('val', int(val_event.sum()))):
            if 'events' in report and report['events'].get(key) != actual:
                raise AuditInputError('report: stored event counts disagree with the supplied tables')
        result['risk_max_abs_reproduction_error'] = risk_error
        arguments = (train_time, train_event, val_time, val_event, survival, edges)
        result['legacy_metrics'] = _json_safe_metrics(parity_metrics(*arguments))
        result['finite_edge_audit'] = audit_parity_metrics(*arguments)
        from sksurv.metrics import concordance_index_censored
        result['c_index_from_stored_risk'] = float(concordance_index_censored(
            val_event.astype(bool), val_time, stored_risk, tied_tol=1e-8)[0])
        recomputed = result['legacy_metrics']['c_index']
        if recomputed is None or not np.isfinite(recomputed):
            raise AuditInputError('predictions: no finite validation Harrell C can be reconstructed')
        if (abs(result['c_index_from_stored_risk'] - recomputed) > C_ATOL or
                abs(result['finite_edge_audit']['c_index'] - recomputed) > C_ATOL):
            raise AuditInputError('predictions: stored-risk or finite-edge Harrell C differs from legacy C')
        _check_selection(report, recomputed, result)
        result['status'] = 'passed'
    except AuditInputError as error:
        result['errors'].append(str(error))
    except Exception as error:
        # Do not serialize arbitrary exception text: filenames/CSV values may
        # contain patient identifiers.  Specific validation errors above are safe.
        result['errors'].append(f'Audit could not complete ({type(error).__name__}); no score validated')
    finally:
        for name, before in result['input_sha256'].items():
            try:
                after = _sha256(paths[name])
                result['input_sha256_after'][name] = after
                if before != after:
                    result['status'] = 'failed'
                    result['errors'].append(f'Input changed during audit: {name}')
            except Exception:
                result['status'] = 'failed'
                result['errors'].append(f'Input could not be rehashed after audit: {name}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--survival-reference', required=True)
    parser.add_argument('--reference-run', required=True,
                        help='Directory containing report.json, checkpoint.pt and val_predictions.csv')
    parser.add_argument('--out', required=True, help='New JSON file; existing files are never overwritten')
    args = parser.parse_args(argv)
    output = Path(args.out)
    if output.exists():
        print('Audit output already exists; nothing was overwritten.', file=sys.stderr)
        return 2
    result = audit_saved_predictions(args.manifest, args.survival_reference, args.reference_run)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open('x', encoding='utf-8') as handle:
            handle.write(serialized)
    except FileExistsError:
        print('Audit output was created concurrently; nothing was overwritten.', file=sys.stderr)
        return 2
    print(f"Saved-prediction audit status: {result['status']}; patient IDs are omitted.")
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
