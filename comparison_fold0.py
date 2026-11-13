"""Locked one-fold BLCA parity comparison; best and LAST EPOCH, never five-fold.

Uses the published legacy parity scoring unchanged, including its short metric
grid conventions. New architectures are explicitly labelled adaptations.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from cancer_parity import (_json_float, dct_bin_edges, discretize_labels, discrete_nll,
                           discrete_survival, dct_risk, parity_metrics)
from cancer_parity_bag import BudgetedPatientBag
from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import PathwayConfig, pathway_stream_forward
from cancer_rna import fit_rna_scaler, transform_rna
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint


MODELS = ('mdecur_aux_off', 'camr_2m')
UNAVAILABLE = {'murrenet': 'Verified author implementation not available; no placeholder model.',
               'mohd': 'Verified author implementation and hierarchical input contract not available.'}
PROTOCOL = 'blca_legacy_fold0_matched_inputs_v1'
SETTINGS = dict(fold=0, endpoint='DSS', time_unit='months', seed=3, epochs=30,
                patient_batch_size=8, num_patches=4096, patch_chunk_size=4096,
                n_bins=4, alpha_surv=.5, lr=5e-4, weight_decay=5e-4,
                scheduler='CosineAnnealingLR(T_max=30)', rna_scale_policy='none',
                selection='maximum validation Harrell C, earliest exact tie',
                metric_convention='unmodified cancer_parity.py legacy scoring')
SOURCE_MODULES = ('comparison_fold0', 'comparison_camr', 'cancer_parity', 'cancer_parity_bag',
                  'cancer_pathway', 'cancer_pathway_stream', 'cancer_model', 'cancer_rna',
                  'cancer_metrics', 'cancer_stream_data', 'cancer_single_view', 'cancer_stream', 'model')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def sources():
    return {name: digest(importlib.import_module(name).__file__) for name in SOURCE_MODULES}


def read_reference(path, cohort):
    values = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            identity, split = row['patient_id'].strip(), row['split'].strip()
            if not identity or identity in values or split not in ('train', 'val'):
                raise ValueError('Survival reference requires unique train/val identities, no test')
            t, e = float(row['time']), float(row['event'])
            if not math.isfinite(t) or t < 0 or e not in (0., 1.):
                raise ValueError('Invalid double-precision survival reference')
            values[identity] = (split, t, e)
    expected = {p for data in cohort.values() for p in data['ids']}
    if set(values) != expected:
        raise ValueError('Survival reference IDs differ from the complete train/val cohort')
    result = {}
    for split, data in cohort.items():
        rows = [values[p] for p in data['ids']]
        if any(r[0] != split for r in rows):
            raise ValueError('Survival reference split differs from manifest')
        t, e = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
        if (not np.array_equal(e, data['event'].numpy())
                or not np.allclose(t, data['time'].numpy(), atol=2e-6, rtol=2e-7)):
            raise ValueError('Double survival reference labels differ from manifest')
        result[split] = (t, e)
    return result


def load_inputs(manifest, survival_reference, reference_run):
    with Path(manifest).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if any(r['split'].strip().lower() not in ('train', 'val') for r in rows):
        raise ValueError('Expected explicit train/val-only manifest; test rejected BEFORE input I/O')
    report_path = Path(reference_run)/'report.json'
    ckpt_path = Path(reference_run)/'checkpoint.pt'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('endpoint') != 'DSS' or report.get('time_unit') != 'months'
            or report.get('n_bins') != 4 or report.get('alpha_surv') != .5
            or report.get('num_patches') != 4096 or report['config']['seed'] != 3
            or report['config']['epochs'] != 30 or len(report['history']) != 30
            or any(report['effective_loss_weights'].values())
            or report['config']['rna_scale_policy'] != 'none'):
        raise ValueError('Reference must be the completed 30-epoch seed3 DSS parity aux-off run')
    if report['manifest_sha256'] != digest(manifest):
        raise ValueError('Manifest differs from completed reference run')
    rna_files = sorted({str((Path(manifest).resolve().parent/r['rna_npy']).resolve()) for r in rows})
    file_hashes = {str(Path(p).resolve()): digest(p) for p in
                   [manifest, survival_reference, report_path, ckpt_path]+rna_files}
    cohort = load_stream_manifest(manifest, 'single_view', include_test=False)
    try:
        fingerprint = stream_cohort_fingerprint(cohort, 'single_view')
        if fingerprint != report['cohort_input_sha256']:
            raise ValueError('Patient inputs differ from completed parity reference')
        for split, data in cohort.items():
            if (len(data['ids']) != report['patients'][split]
                    or int(data['event'].sum()) != report['events'][split]):
                raise ValueError('Patient/event counts differ from parity reference')
        reference = read_reference(survival_reference, cohort)
        edges = dct_bin_edges(*reference['train'], 4)
        if [_json_float(x) for x in edges] != report['bin_edges']:
            raise ValueError('Train-fitted bin edges differ from parity reference')
        return cohort, reference, edges, report, file_hashes
    except BaseException:
        close(cohort)
        raise


def close(cohort):
    for data in cohort.values():
        for bag in data['bags']:
            if hasattr(bag, 'close'):
                bag.close()


def input_metadata(cohort, reference, edges, report, file_hashes):
    return {'file_sha256': file_hashes, 'cohort_input_sha256': report['cohort_input_sha256'],
            'bin_edges': [_json_float(x) for x in edges],
            'patients': {s: len(d['ids']) for s, d in cohort.items()},
            'events': {s: int(d['event'].sum()) for s, d in cohort.items()},
            'order_sha256': {s: json_hash(d['ids']) for s, d in cohort.items()},
            'double_labels_sha256': {s: json_hash([t.tolist(), e.tolist()]) for s, (t, e) in reference.items()},
            'path_dim': cohort['train']['bags'][0].feature_dim, 'rna_dim': cohort['train']['rna'].shape[1],
            'native_model_kwargs': report['model_kwargs'],
            'prior_sealed_patient_overlap': 'Not established by manifest without historical ID intersection; do not infer sealed status.'}


def make_plan(manifest, survival_reference, reference_run):
    cohort, reference, edges, report, file_hashes = load_inputs(manifest, survival_reference, reference_run)
    try:
        if os.environ.get('PYTHONHASHSEED') != '0':
            raise ValueError('Start process with PYTHONHASHSEED=0; sampler uses Python hash')
        return {'protocol': PROTOCOL, 'settings': SETTINGS, 'available_models': list(MODELS),
                'unavailable_models': UNAVAILABLE, 'inputs': input_metadata(cohort, reference, edges, report, file_hashes),
                'source_sha256': sources(), 'python_hash_seed': '0',
                'sampler_runtime_signature': hash((3001, 'comparison_sampler_signature')),
                'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                             'numpy': np.__version__, 'sksurv': importlib.import_module('sksurv').__version__},
                'scope': 'one supplied legacy fold0 only; best/last EPOCH, not an independent test',
                'reference_run': str(Path(reference_run).resolve()),
                'manifest': str(Path(manifest).resolve()),
                'survival_reference': str(Path(survival_reference).resolve())}
    finally:
        close(cohort)


def materialize(bags, indices, device):
    result = []
    for i in indices:
        bag = bags[i]
        arrays = [bag.read_block(block, ('features_reference',))[0] for block in bag.blocks(4096)]
        patches = torch.cat(arrays)
        if patches.shape != (4096, bag.feature_dim):
            raise ValueError('Expected complete 4096-row budget including explicit zero padding')
        result.append(patches.to(device))
    return result


def forward(model_id, model, bags, rna, indices, device):
    if model_id == 'mdecur_aux_off':
        output = pathway_stream_forward(model, [bags[i] for i in indices], rna.to(device),
                                        chunk_size=4096, view_source='feature_noise', checkpoint_chunks=True)
        return {'logits': output['log_rates']}
    return model(materialize(bags, indices, device), rna.to(device))


def evaluate(model_id, model, data, bags, normalized, labels, reference, edges, device):
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(len(bags)):
            output = forward(model_id, model, bags, normalized[i:i+1], [i], device)
            logits.append(output['logits'].detach().cpu())
    logits = torch.cat(logits)
    survival = discrete_survival(logits).numpy()
    result = parity_metrics(*reference['train'], *reference['val'], survival, edges)
    result['validation_nll'] = float(discrete_nll(logits, labels, torch.tensor(1-reference['val'][1]), alpha=.5))
    if result['c_index'] is None:
        raise ValueError('No defined validation Harrell C; keep errors, do not select a substitute metric')
    return result, survival


def save_predictions(path, ids, survival):
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient_id', 'risk']+[f'survival_bin_{k}' for k in range(4)])
        writer.writerows([p, float(r), *map(float, s)] for p, r, s in zip(ids, dct_risk(survival), survival))


def run_model(plan_path, plan_sha256, model_id, out, device='cpu', smoke=False):
    if model_id not in MODELS:
        raise ValueError(UNAVAILABLE.get(model_id, 'Unknown model; no implicit substitute'))
    destination = Path(out)
    if destination.exists():
        raise FileExistsError(destination)
    if digest(plan_path) != plan_sha256:
        raise ValueError('Plan SHA mismatch')
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    # Rebuild and compare ALL inputs, code, settings and RNG signature before training.
    actual = make_plan(plan['manifest'], plan['survival_reference'], plan['reference_run'])
    if actual != plan:
        raise ValueError('Plan no longer matches inputs/source/runtime/sampling settings')
    cohort, reference, edges, original_report, hashes = load_inputs(
        plan['manifest'], plan['survival_reference'], plan['reference_run'])
    started = time.perf_counter()
    try:
        if input_metadata(cohort, reference, edges, original_report, hashes) != plan['inputs']:
            raise ValueError('Inputs changed between plan verification and training load')
        destination.mkdir(parents=True, exist_ok=False)  # Atomically reserve this run, including failures.
        device = torch.device(device)
        if device.type == 'cuda':
            torch.zeros(1, device=device)
            torch.cuda.synchronize(device)
        torch.manual_seed(3)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        runtime = {'device': str(device), 'cuda_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                   'deterministic_algorithms': 'warn_only', 'tf32': False}
        scaler = fit_rna_scaler(cohort['train']['rna'], 'none')
        normalized = {s: transform_rna(d['rna'], scaler) for s, d in cohort.items()}
        bags = {s: [BudgetedPatientBag(b, 4096, 'train' if s == 'train' else 'eval', 3)
                    for b in d['bags']] for s, d in cohort.items()}
        labels = {s: torch.from_numpy(discretize_labels(t, edges)) for s, (t, _) in reference.items()}
        if model_id == 'mdecur_aux_off':
            model = PathwayLocalSurvival(**original_report['model_kwargs']).to(device)
            metadata = {'label': 'MDeCUR-aux-off-common-plan', 'auxiliary_weights': {},
                        'adaptations': ['New run under fixed common PYTHONHASHSEED; historical score is not silently reused.']}
            main_parameters = list(model.parameters())
            d_optimizer = None
        else:
            from comparison_camr import CAMR2M
            model = CAMR2M(plan['inputs']['path_dim'], plan['inputs']['rna_dim'], bins=4).to(device)
            metadata = model.adaptation_metadata()
            main_parameters = list(model.main_parameters())
            d_optimizer = torch.optim.AdamW(model.discriminator_parameters(), lr=5e-4, weight_decay=5e-4)
        optimizer = torch.optim.AdamW(main_parameters, lr=5e-4, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=30) if d_optimizer else None
        generator = torch.Generator().manual_seed(3)
        history, best, best_survival, best_metrics, best_epoch = [], None, None, None, None
        sample_hash = hashlib.sha256()
        count_epochs = 1 if smoke else 30
        for epoch in range(1, count_epochs+1):
            model.train()
            for bag in bags['train']:
                bag.resample(3000+epoch)
                sample_hash.update(json_hash(bag._rows).encode())
            order = torch.randperm(len(bags['train']), generator=generator)
            losses, n = {}, 0
            for selected in order.split(8):
                indices = selected.tolist()
                output = forward(model_id, model, bags['train'], normalized['train'][selected], indices, device)
                survival_loss = discrete_nll(output['logits'], labels['train'][selected].to(device),
                                             torch.tensor(1-reference['train'][1][indices], device=device), alpha=.5)
                components = {'survival': survival_loss}
                if d_optimizer:
                    d_optimizer.zero_grad(set_to_none=True)
                    d_loss = model.discriminator_loss(output)
                    if not torch.isfinite(d_loss):
                        raise FloatingPointError('Nonfinite discriminator objective')
                    d_loss.backward()  # Encoders detached inside the discriminator loss.
                    if any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for p in model.discriminator_parameters()):
                        raise FloatingPointError('Nonfinite discriminator gradient')
                    d_optimizer.step()
                    d_optimizer.zero_grad(set_to_none=True)
                    auxiliary = model.auxiliary_losses(output)
                    components.update(auxiliary)
                    loss = survival_loss+.6*auxiliary['adversarial']+.8*auxiliary['reconstruction']+.05*auxiliary['orthogonality']
                else:
                    loss = survival_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training objective')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in main_parameters):
                    raise FloatingPointError('Nonfinite parameter gradient')
                optimizer.step()
                for name, value in components.items():
                    losses[name] = losses.get(name, 0.)+len(indices)*float(value.detach())
                n += len(indices)
            scheduler.step()
            if d_scheduler:
                d_scheduler.step()
            metrics, survival = evaluate(model_id, model, cohort['val'], bags['val'], normalized['val'],
                                          labels['val'], reference, edges, device)
            history.append({'epoch': epoch, 'train_terms': {k: v/n for k, v in losses.items()},
                            'learning_rate_after_step': scheduler.get_last_lr()[0], **metrics})
            print(json.dumps({'model': model_id, **history[-1]}, allow_nan=False), flush=True)
            if best_metrics is None or metrics['c_index'] > best_metrics['c_index']:
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_metrics, best_survival, best_epoch = metrics, survival.copy(), epoch
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for data in cohort.values():
            for bag in data['bags']:
                bag.assert_unchanged()
        if any(digest(p) != h for p, h in plan['inputs']['file_sha256'].items()) or sources() != plan['source_sha256'] or digest(plan_path) != plan_sha256:
            raise RuntimeError('Input, code or plan changed during run; refuse to publish comparison output')
        common = {'model_id': model_id, 'metadata': metadata, 'plan_sha256': plan_sha256,
                  'runtime': runtime,
                  'settings': plan['settings'], 'bin_edges': plan['inputs']['bin_edges'],
                  'native_model_kwargs': original_report['model_kwargs'] if model_id == 'mdecur_aux_off' else None,
                  'rna_dim': plan['inputs']['rna_dim'], 'path_dim': plan['inputs']['path_dim']}
        for label, state, epoch in (('best', best, best_epoch), ('last', last_state, count_epochs)):
            torch.save({**common, 'model_state': state, 'epoch': epoch}, destination/f'checkpoint_{label}.pt')
        save_predictions(destination/'val_predictions_best.csv', cohort['val']['ids'], best_survival)
        save_predictions(destination/'val_predictions_last.csv', cohort['val']['ids'], survival)
        result = {**common, 'protocol': PROTOCOL, 'smoke_only': smoke, 'test_evaluated': False,
                  'patients': plan['inputs']['patients'], 'events': plan['inputs']['events'],
                  'source_sha256': plan['source_sha256'], 'epochs_completed': count_epochs,
                  'training_sampling_sha256': sample_hash.hexdigest(), 'best_epoch': best_epoch,
                  'last_epoch': count_epochs, 'best_validation': best_metrics, 'last_validation': metrics,
                  'parameters': sum(p.numel() for p in model.parameters()), 'history': history,
                  'wall_seconds': time.perf_counter()-started,
                  'limits': ['Legacy validation selects epoch; no independent test or claimed significance.',
                             'Best and last refer to epochs of ONE fold, not additional folds.',
                             'Legacy metric-column/time mapping is intentionally preserved; IAUC scope must be labelled.',
                             'Adapted methods are not original-paper result reproductions.']}
        write_new(destination/'report.json', result)
        return result
    finally:
        close(cohort)


def summarize(run_dirs, out):
    records = [json.loads((Path(d)/'report.json').read_text(encoding='utf-8')) for d in run_dirs]
    if not records or len({r['plan_sha256'] for r in records}) != 1:
        raise ValueError('Every compared model must use exactly the same input/protocol plan')
    if (any(r['smoke_only'] or r['epochs_completed'] != 30 or r['test_evaluated'] for r in records)
            or any(r['model_id'] not in MODELS or r['last_epoch'] != 30
                   or not 1 <= r['best_epoch'] <= 30 for r in records)
            or len({r['model_id'] for r in records}) != len(records)
            or len({r['training_sampling_sha256'] for r in records}) != 1
            or len({json_hash(r.get('runtime')) for r in records}) != 1):
        raise ValueError('Refuse incomplete/smoke/test/duplicate/different-sampling runs')
    rows = []
    for record in records:
        for mode in ('best', 'last'):
            m = record[f'{mode}_validation']
            rows.append({'model': record['model_id'], 'selection': mode,
                         'epoch': record[f'{mode}_epoch'], 'Harrell_C': m['c_index'],
                         'IPCW_C': m['c_index_ipcw'], 'NLL': m['validation_nll'],
                         'Brier': m['brier'], 'IBS_legacy_grid': m['ibs'],
                         'AUC_legacy_times': m['iauc'], 'metric_times': m['metric_times'], 'errors': m['errors']})
    result = {'protocol': PROTOCOL, 'plan_sha256': records[0]['plan_sha256'], 'rows': rows,
              'Harrell_C_rankings': {mode: [r['model'] for r in sorted(
                  [x for x in rows if x['selection'] == mode], key=lambda x: -x['Harrell_C'])]
                  for mode in ('best', 'last')},
              'unavailable_models': UNAVAILABLE,
              'metric_directions': {'Harrell_C': 'higher', 'IPCW_C': 'higher', 'AUC_legacy_times': 'higher',
                                    'NLL': 'lower', 'Brier': 'lower', 'IBS_legacy_grid': 'lower'},
              'limits': 'Point estimates on one validation-selected fold, no significance or overall superiority claim.'}
    write_new(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--manifest', required=True)
    p.add_argument('--survival-reference', required=True)
    p.add_argument('--reference-run', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('run')
    p.add_argument('--plan', required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--model', choices=MODELS+tuple(UNAVAILABLE), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke', action='store_true', help='One-epoch engineering check; excluded from comparison')
    p = sub.add_parser('summary')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = parser.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    if a.action == 'plan':
        write_new(a.out, make_plan(a.manifest, a.survival_reference, a.reference_run))
    elif a.action == 'run':
        run_model(a.plan, a.plan_sha256, a.model, a.out, a.device, a.smoke)
    else:
        summarize(a.runs, a.out)
    print(json.dumps({'action': a.action, 'out': a.out}), flush=True)


if __name__ == '__main__':
    main()
