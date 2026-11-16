"""Run the three-arm V3 pipeline on generated mixed-censoring CPU inputs.

This engineering receipt is not a patient experiment or an independent test of
prognostic benefit. It performs 30 optimizer epochs per arm, then reconstructs
each saved best/last model and checks its predictions and complete metric dict.
Generated feature arrays/checkpoints remain in a new local directory, including
on failure. A compact JSON receipt and concise progress log use new paths.
"""
import argparse
import csv
import io
import json
import hashlib
import inspect
from contextlib import redirect_stdout
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import comparison_fold0 as compare
from cancer_dense_model import DensePathwaySurvival
from cancer_parity import discretize_labels, dct_bin_edges, _json_float
from cancer_parity_bag import BudgetedPatientBag
from cancer_rna import fit_rna_scaler, transform_rna
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
from comparison_camr import CAMR2M


def write_inputs(root):
    """Generate the mixed-event fixture without any temporary-file cleanup."""
    root.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(18)
    rows = []
    for i in range(24):
        patches = torch.randn(4096+i % 3, 6, generator=generator)+3*torch.randn(1, 6, generator=generator)
        torch.save(patches, root/f'p{i}.pt')
        np.save(root/f'g{i}.npy', torch.randn(8, generator=generator).numpy())
        split = 'train' if i < 20 else 'val'
        time = i+1 if i < 20 else [2.5, 7.5, 12.5, 18.5][i-20]
        event = int(i % 3 != 0) if split == 'train' else [1, 0, 1, 1][i-20]
        rows.append([f'p{i}', split, f'p{i}.pt', f'g{i}.npy', time, event])
    manifest, reference = root/'manifest.csv', root/'survival_reference.csv'
    with manifest.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient_id', 'split', 'wsi_pt', 'rna_npy', 'time', 'event'])
        writer.writerows(rows)
    with reference.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient_id', 'split', 'time', 'event'])
        writer.writerows([r[0], r[1], r[4], r[5]] for r in rows)
    cohort = load_stream_manifest(manifest, 'single_view', include_test=False)
    try:
        fingerprint = stream_cohort_fingerprint(cohort, 'single_view')
    finally:
        compare.close(cohort)
    ref_run = root/'generated_identity_reference'
    ref_run.mkdir()
    # This placeholder is hashed by make_plan, never unpickled or reused.
    (ref_run/'checkpoint.pt').write_bytes(b'synthetic-reference-file')
    original = {'endpoint': 'DSS', 'time_unit': 'months', 'n_bins': 4, 'alpha_surv': .5,
                'num_patches': 4096, 'config': {'seed': 3, 'epochs': 30, 'rna_scale_policy': 'none'},
                'history': [{}]*30, 'effective_loss_weights': {'any': 0.},
                'manifest_sha256': compare.digest(manifest), 'cohort_input_sha256': fingerprint,
                'patients': {'train': 20, 'val': 4}, 'events': {'train': 13, 'val': 3},
                'bin_edges': [_json_float(x) for x in dct_bin_edges(
                    np.arange(1, 21), np.array([int(i % 3 != 0) for i in range(20)]))],
                'model_kwargs': dict(path_dim=6, rna_dim=8, pathway_indices=[
                    [j, (j+1) % 8, (j+3) % 8] for j in range(8)]+[[0, 2, 4], [1, 3, 5]],
                                     hidden=16, common=8, local_dim=8, local_common=4, bins=4,
                                     interaction_strength=.2, match_temperature=.2)}
    compare.write_new(ref_run/'report.json', original)
    return SimpleNamespace(root=root, manifest=manifest, reference=reference, ref_run=ref_run)


class ProgressLog(io.TextIOBase):
    """Reduce the runner's large epoch JSON to a flushed engineering log."""
    def __init__(self, handle):
        self.handle, self.pending = handle, ''

    def write(self, value):
        self.pending += value
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if line:
                record = json.loads(line)
                self.handle.write(json.dumps({key: record.get(key) for key in
                    ('model', 'epoch', 'c_index', 'validation_nll', 'optimizer_steps', 'patient_batches')})+'\n')
                self.handle.flush()
        return len(value)

    def flush(self):
        self.handle.flush()


def validate(output, generated_root=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    generated_root = Path(generated_root) if generated_root else output.with_name(output.stem+'_generated')
    log_path = output.with_suffix('.log')
    if log_path.exists() or generated_root.exists():
        raise FileExistsError('Validation requires new generated root and progress log')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        case = write_inputs(generated_root)
        plan_path = case.root/'plan.json'
        compare.write_new(plan_path, compare.make_plan(case.manifest, case.reference, case.ref_run))
        plan_sha = compare.digest(plan_path)
        frozen_sources = compare.sources()
        reports = []
        replays = []
        state_checks = []
        arms = ('mdecur_full_dense', 'mdecur_dense_aux_off', 'camr_2m')
        for arm in arms:
            run_path = case.root/arm
            print(f'Starting {arm}; progress log: {log_path}', flush=True)
            with log_path.open('a' if log_path.exists() else 'x', encoding='utf-8') as log:
                log.write(json.dumps({'arm_started': arm})+'\n')
                log.flush()
                with redirect_stdout(ProgressLog(log)):
                    report = compare.run_model(plan_path, plan_sha, arm, run_path)
            reports.append(report)
            cohort, reference, edges, original, _ = compare.load_inputs(
                case.manifest, case.reference, case.ref_run)
            try:
                scaler = fit_rna_scaler(cohort['train']['rna'], 'none')
                normalized = transform_rna(cohort['val']['rna'], scaler)
                bags = [BudgetedPatientBag(b, 4096, 'eval', 3) for b in cohort['val']['bags']]
                labels = torch.from_numpy(discretize_labels(reference['val'][0], edges))
                for selection in ('best', 'last'):
                    checkpoint = torch.load(run_path/f'checkpoint_{selection}.pt',
                                            map_location='cpu', weights_only=True)
                    model = (DensePathwaySurvival(**original['model_kwargs']) if arm in compare.DENSE_MODELS
                             else CAMR2M(cohort['train']['bags'][0].feature_dim,
                                         cohort['train']['rna'].shape[1], bins=4))
                    model.load_state_dict(checkpoint['model_state'])
                    metrics, survival = compare.evaluate(arm, model, cohort['val'], bags,
                                                          normalized, labels, reference, edges, 'cpu')
                    with (run_path/f'val_predictions_{selection}.csv').open(newline='', encoding='utf-8') as handle:
                        saved = list(csv.DictReader(handle))
                    expected = np.array([[float(row[f'survival_bin_{i}']) for i in range(4)]
                                         for row in saved])
                    if not np.array_equal(survival, expected):
                        raise AssertionError(f'{arm}/{selection} checkpoint prediction replay differs')
                    if metrics != report[f'{selection}_validation']:
                        raise AssertionError(f'{arm}/{selection} metric replay differs')
                    replays.append({'arm': arm, 'selection': selection,
                                    'epoch': checkpoint['epoch'],
                                    'prediction_max_abs_error': float(np.max(np.abs(survival-expected))),
                                    'complete_metric_dict_equal': True})
                best_state = torch.load(run_path/'checkpoint_best.pt', map_location='cpu', weights_only=True)['model_state']
                last_state = torch.load(run_path/'checkpoint_last.pt', map_location='cpu', weights_only=True)['model_state']
                changed = [name for name in best_state if not torch.equal(best_state[name], last_state[name])]
                state_checks.append({'arm': arm, 'best_epoch': report['best_epoch'], 'last_epoch': report['last_epoch'],
                                     'different_tensor_count': len(changed), 'different_state': bool(changed)})
                if report['best_epoch'] != report['last_epoch'] and not changed:
                    raise AssertionError('Different selected epochs unexpectedly saved identical state')
            finally:
                compare.close(cohort)
            print(f'Completed {arm}: 30 epochs and both checkpoint replays', flush=True)
        if len({r['training_sampling_sha256'] for r in reports}) != 1:
            raise AssertionError('Patch sampling differs between arms')
        if len({r['training_schedule_sha256'] for r in reports}) != 1:
            raise AssertionError('Patient batching differs between arms')
        summary = compare.summarize([case.root/arm for arm in arms], case.root/'summary.json')
        if compare.sources() != frozen_sources:
            raise AssertionError('Sources changed during final validation')
        receipt = {'scope': 'CPU generated data only; not patient/CUDA/prognostic evidence',
                   'protocol': compare.PROTOCOL, 'loss_version': compare.SETTINGS['loss_version'],
                   'source_sha256': frozen_sources,
                   'run_model_function_sha256': hashlib.sha256(inspect.getsource(compare.run_model).encode()).hexdigest(),
                   'plan_sha256': plan_sha, 'all_arms_same_plan': len({r['plan_sha256'] for r in reports}) == 1,
                   'generated_root': str(case.root.resolve()), 'progress_log': str(log_path.resolve()),
                   'generated_patients': {'train': 20, 'val': 4},
                   'generated_events': {'train': 13, 'val': 3},
                   'epochs_per_arm': 30, 'arms': list(arms),
                   'training_sampling_sha256': reports[0]['training_sampling_sha256'],
                   'training_schedule_sha256': reports[0]['training_schedule_sha256'],
                   'optimizer_steps_per_arm': {r['model_id']: r['optimizer_steps'] for r in reports},
                   'discriminator_steps_per_arm': {r['model_id']: r['discriminator_optimizer_steps'] for r in reports},
                   'best_last_replay': replays,
                   'best_last_state_checks': state_checks,
                   'synthetic_validation_rows': summary['rows'],
                   'main_full_auxiliary_terms': list(reports[0]['history'][-1]['train_terms']),
                   'test_evaluated': False}
        compare.write_new(output, receipt)
        return receipt
    finally:
        torch.set_num_threads(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--generated-root')
    args = parser.parse_args()
    result = validate(args.out, args.generated_root)
    print(json.dumps({'protocol': result['protocol'], 'arms': result['arms'],
                      'epochs_per_arm': 30, 'checkpoint_replays': len(result['best_last_replay']),
                      'out': str(Path(args.out).resolve())}))
