"""Read-only summary of an existing fixed-R gradient audit JSON.

This program never loads a checkpoint, a patient file, a manifest or a model.
It reads one already-produced audit JSON and reports the fields that the
initial summary table omitted or mislabelled:

  * ``sum_weighted_auxiliary.survival_graph_space`` -- the actual
    auxiliary-sum-versus-survival-gradient relation.  The earlier label
    "auxiliary sum versus total descent direction" was wrong: the code
    computes ``relation(survival, reconstructed)`` for the descent direction,
    whose ratio is ~1 by construction, and ``relation(survival,
    summed_auxiliary)`` for the aggregate.
  * ``total_descent_direction`` -- survival gradient versus the total
    gradient, reported with the correct name.
  * Absolute (not only relative) per-module norms, because a ratio whose
    denominator is small can be large while both absolute norms are tiny.
  * The number of genuinely distinct batches, since the two view seeds of a
    batch produce near-identical gradients and are not independent repeats.

No model, no data, no optimization, no overwrite of any existing artifact.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True  # strings and other JSON scalars are labels, not measurements


def check_finite(value, where):
    if isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, f'{where}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_finite(item, f'{where}[{index}]')
    elif not finite(value):
        raise ValueError(f'Nonfinite value in audit output at {where}')


def aggregate_view(space):
    """Keep only the relation numbers; drop verbose identifier fields."""
    keys = ('status', 'direction', 'cosine', 'survival_norm',
            'raw_auxiliary_norm_in_survival_space',
            'weighted_auxiliary_norm_in_survival_space',
            'weighted_norm_ratio_to_survival', 'raw_dot', 'weighted_dot',
            'outside_survival_graph_raw_auxiliary_norm',
            'survival_connected_tensors', 'both_connected_tensors')
    return {key: space.get(key) for key in keys if key in space}


def module_rows(space, modules):
    rows = []
    for name in sorted(modules):
        module = modules[name]
        rows.append({'module': name, **aggregate_view(module)})
    return rows


def summarize(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return {'count': 0}
    clean = sorted(clean)
    middle = len(clean)//2
    median = clean[middle] if len(clean) % 2 else 0.5*(clean[middle-1]+clean[middle])
    return {'count': len(clean), 'min': clean[0], 'max': clean[-1],
            'median': median, 'mean': sum(clean)/len(clean)}


def extract(payload):
    records = payload.get('records')
    if not records:
        raise ValueError('Audit JSON has no records')
    terms = {}
    names = sorted(payload['summary'].keys())
    for name in names:
        if name == 'sum_weighted_auxiliary':
            continue
        term = {}
        for record in records:
            item = record['audit']['terms'].get(name)
            if item is None:
                continue
            term.setdefault('status', []).append(item['status'])
            if item['status'] != 'computed':
                continue
            term.setdefault('survival_space', []).append(aggregate_view(item['survival_graph_space']))
            for module, space in item['modules'].items():
                term.setdefault('modules', {}).setdefault(module, []).append(aggregate_view(space))
        if not term:
            continue
        terms[name] = {
            'status_counts': {status: term['status'].count(status) for status in sorted(set(term['status']))},
            'survival_space': {key: summarize([space.get(key) for space in term['survival_space']])
                               for key in ('cosine', 'weighted_norm_ratio_to_survival',
                                           'survival_norm', 'weighted_auxiliary_norm_in_survival_space')},
            'modules': {module: {key: summarize([space.get(key) for space in spaces])
                                 for key in ('cosine', 'weighted_norm_ratio_to_survival',
                                             'survival_norm', 'weighted_auxiliary_norm_in_survival_space')}
                        for module, spaces in term.get('modules', {}).items()},
        }
    totals = []
    for record in records:
        totals.append(record['audit']['sum_weighted_auxiliary'])
    descent = [aggregate_view(record['audit']['total_descent_direction']) for record in records]
    aggregate = [aggregate_view(item['survival_graph_space']) for item in totals]
    reconstruction = [record['audit']['gradient_reconstruction'] for record in records]
    return {
        'records': len(records),
        'batches': len({record['batch_index'] for record in records}),
        'patients_total': len({p for record in records for p in record['patient_ids']}),
        'auxiliary_sum_versus_survival_gradient': {
            'note': ('relation(grad survival, summed weighted auxiliary). This is the quantity the '
                     'earlier summary mislabelled as "auxiliary sum versus total descent direction".'),
            'per_record': aggregate,
            'cosine': summarize([space.get('cosine') for space in aggregate]),
            'direction_counts': {direction: sum(1 for space in aggregate if space.get('direction') == direction)
                                 for direction in sorted({space.get('direction') for space in aggregate})},
            'weighted_norm_ratio_to_survival': summarize([space.get('weighted_norm_ratio_to_survival')
                                                          for space in aggregate]),
            'modules': {module: {key: summarize([space.get(key) for space in spaces])
                                 for key in ('cosine', 'survival_norm',
                                             'weighted_auxiliary_norm_in_survival_space',
                                             'weighted_norm_ratio_to_survival')}
                        for module, spaces in _module_spaces(totals).items()},
        },
        'survival_gradient_versus_total_gradient': {
            'note': ('relation(grad survival, grad total). Its ratio is ~1 by construction and says '
                     'only that the auxiliary part barely moves the total direction.'),
            'per_record': descent,
            'cosine': summarize([space.get('cosine') for space in descent]),
            'weighted_norm_ratio_to_survival': summarize([space.get('weighted_norm_ratio_to_survival')
                                                          for space in descent]),
        },
        'norm_of_sum_over_sum_of_norms': summarize([record['audit']['sum_weighted_auxiliary']
                                                     ['norm_of_sum_over_sum_of_norms'] for record in records]),
        'gradient_reconstruction': {
            'absolute_l2_error': summarize([item['absolute_l2_error'] for item in reconstruction]),
            'relative_l2_error': summarize([item['relative_l2_error'] for item in reconstruction]),
            'direct_total_norm': summarize([item['direct_total_norm'] for item in reconstruction]),
        },
        'terms': terms,
        'limits': ['Read-only re-report of an existing audit JSON; no new model forward or backward.',
                   'Cosines are local Euclidean geometry at the audited checkpoint, not AdamW updates.',
                   'Two view repeats of one batch are near-duplicates; the distinct-batch count is reported.',
                   'No validation or test data is involved and nothing is trained.'],
    }


def _module_spaces(totals):
    modules = {}
    for item in totals:
        for module, space in item['modules'].items():
            modules.setdefault(module, []).append(aggregate_view(space))
    return modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, help='Existing fixed_r_gradients.json produced earlier')
    parser.add_argument('--out', help='New JSON path; existing files are never overwritten')
    args = parser.parse_args()
    source = Path(args.audit)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding='utf-8'))
    check_finite(payload, 'audit')
    result = {'source_audit': str(source), 'source_audit_sha256': sha256_file(source),
              'protocol': payload.get('protocol'),
              'training_performed': payload.get('training_performed'),
              'optimizer_steps': payload.get('optimizer_steps'),
              'validation_loaded': payload.get('validation_loaded'),
              'test_loaded': payload.get('test_loaded'),
              'summary': extract(payload)}
    if args.out:
        destination = Path(args.out)
        if destination.exists():
            raise FileExistsError(destination)
        with destination.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        print(json.dumps({'out': str(destination), 'sha256': sha256_file(destination)}, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
