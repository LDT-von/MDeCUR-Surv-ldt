"""Plan then audit auxiliary gradients on fixed TRAIN batches; never optimize.

Validation/test feature, RNA and label values are not loaded. The old checkpoint
has only a combined train/val content hash, so its historical content hash cannot
be recomputed here. A separate train-only hash is pinned by the pre-audit plan.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, fields
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import (PathwayConfig, LOSS_NAMES, effective_weights,
                                   patient_batches, validate_config)
from cancer_rna import fit_rna_scaler, transform_rna
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
from diagnose_fixed_r_decomposition import state_digest
from diagnose_pathway_query_permutation import check_artifacts, sha256_file
from pathway_gradient_audit import audit_batch


PROTOCOL = 'fixed_r_train_gradients_v1'
ORDER_SEED = 20261112
BATCH_COUNT = 3
REPEATS = (11, 22)
SOURCE_MODULES = ('pathway_gradient_audit', 'cancer_pathway_stream', 'cancer_pathway',
                  'cancer_model', 'cancer_rna', 'cancer_stream', 'cancer_stream_data',
                  'cancer_single_view', 'cancer_auxiliary', 'model', 'uni_views.core',
                  'diagnose_fixed_r_decomposition', 'diagnose_pathway_query_permutation')


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def source_hashes():
    return {'entrypoint': sha256_file(__file__), **{
        name: sha256_file(importlib.import_module(name).__file__) for name in SOURCE_MODULES}}


def fixed_batches(ids, size, minimum):
    """Label-blind CPU permutation; no retries or selection by diagnostic values."""
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate training identities')
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(ORDER_SEED))
    batches = patient_batches(order, size, minimum)
    if len(batches) < BATCH_COUNT:
        raise ValueError(f'Protocol requires at least {BATCH_COUNT} full training batches')
    return [{'batch_index': index, 'indices_in_training_order': chosen.tolist(),
             'patient_ids': [ids[i] for i in chosen.tolist()],
             'views': [{'repeat': repeat, 'view_seed': ORDER_SEED+repeat*100003+index,
                        'rna_noise_seed': ORDER_SEED+repeat*100019+index} for repeat in REPEATS]}
            for index, chosen in enumerate(batches[:BATCH_COUNT])]


def training_rna_paths(manifest):
    path = Path(manifest).resolve()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    # No resolution/opening/parsing of held-out paths or labels.
    return sorted({str((path.parent/row['rna_npy']).resolve()) for row in rows
                   if row['split'].strip().lower() == 'train'})


@contextmanager
def loaded_training_reference(manifest, run_dir, expected_sha256, device):
    paths = {'manifest': Path(manifest), 'checkpoint': Path(run_dir)/'checkpoint.pt',
             'report': Path(run_dir)/'report.json'}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes['checkpoint'].lower() != expected_sha256.lower():
        raise ValueError('Checkpoint SHA256 differs from expected R checkpoint')
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
    report = json.loads(paths['report'].read_text(encoding='utf-8'))
    check_artifacts(checkpoint, report, hashes['manifest'])
    if set(checkpoint['config']) != {field.name for field in fields(PathwayConfig)}:
        raise ValueError('Expected complete saved PathwayConfig, no inferred defaults')
    cfg = PathwayConfig(**checkpoint['config'])
    stream = checkpoint['stream_config']
    minimum = validate_config(cfg, stream['view_source'], stream['patient_batch_size'], stream['patch_chunk_size'])
    if (not cfg.interaction_strength > 0 or not all(w > 0 for w in effective_weights(cfg).values())
            or effective_weights(cfg) != checkpoint['effective_loss_weights']):
        raise ValueError('Expected full R with positive interaction and all 11 auxiliary weights')
    if (stream['auxiliary_rng_scheme'] != 'rna_global__weight_shuffle_isolated_v1'
            or stream['checkpoint_chunks'] is not True or stream['minimum_patient_batch'] != minimum
            or stream['rna_noise'] != cfg.rna_noise
            or stream['feature_noise'] != (cfg.feature_noise if stream['view_source'] == 'feature_noise' else None)):
        raise ValueError('Saved streaming/RNG protocol disagrees with objective')
    if (checkpoint['rna_preprocessing']['policy'] != cfg.rna_scale_policy
            or checkpoint['rna_preprocessing']['min_std'] != cfg.rna_min_std):
        raise ValueError('Saved RNA preprocessing disagrees with config')
    source = 'single_view' if stream['view_source'] == 'feature_noise' else 'cached_uni'
    rna_hashes = {path: sha256_file(path) for path in training_rna_paths(manifest)}
    cohort = load_stream_manifest(manifest, source, include_test=False, train_only=True)
    try:
        train = cohort['train']
        for name, actual in (('patients', len(train['ids'])), ('events', int(train['event'].sum())),
                             ('patches', sum(bag.n_patches for bag in train['bags']))):
            if report[name]['train'] != actual:
                raise ValueError(f'Training {name} disagree with saved report')
        mapping = checkpoint['pathway_mapping']
        kwargs = checkpoint['model_kwargs']
        if (mapping['rna_dim'] != train['rna'].shape[1] or kwargs['rna_dim'] != mapping['rna_dim']
                or kwargs['path_dim'] != train['bags'][0].feature_dim
                or kwargs['pathway_indices'] != [p['indices'] for p in mapping['pathways']]
                or kwargs['bins'] != len(checkpoint['starts'])):
            raise ValueError('Saved model/pathway mapping/input dimensions disagree')
        for name in ('hidden', 'common', 'local_dim', 'local_common', 'interaction_strength', 'match_temperature'):
            if kwargs[name] != getattr(cfg, name):
                raise ValueError(f'Model kwargs and config disagree on {name}')
        fitted = {key: checkpoint[key] for key in ('rna_mean', 'rna_scale', 'rna_mask', 'rna_preprocessing')}
        refit = fit_rna_scaler(train['rna'], cfg.rna_scale_policy, cfg.rna_min_std)
        differences = {}
        for key in ('rna_mean', 'rna_scale', 'rna_mask', 'rna_train_std'):
            a, b = checkpoint[key].cpu(), refit[key].cpu()
            same = torch.equal(a, b) if key == 'rna_mask' else torch.allclose(a, b, atol=1e-8, rtol=1e-5)
            if a.shape != b.shape or not same:
                raise ValueError(f'Saved {key} differs from train-only refit')
            if key != 'rna_mask':
                differences[key] = float((a-b).abs().max())
        normalized = transform_rna(train['rna'], fitted)  # Saved scaler, never replace it.
        with torch.random.fork_rng(devices=[]):
            model = PathwayLocalSurvival(**kwargs).to(device)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        model.train().requires_grad_(True)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError('Nonfinite checkpoint model state')
        # Mutation of buffers in a training-mode forward would invalidate a fixed
        # checkpoint audit. Current model uses LayerNorm, no BatchNorm/dropout.
        if any(isinstance(module, (torch.nn.modules.batchnorm._BatchNorm,
                                   torch.nn.modules.dropout._DropoutNd)) for module in model.modules()):
            raise ValueError('Stateful/stochastic train-mode layers require a reviewed protocol')
        train_hash = stream_cohort_fingerprint(cohort, source)
        metadata = {'artifact_sha256': hashes, 'training_input_sha256': train_hash,
                    'training_rna_files_sha256': rna_hashes, 'training_order_sha256': json_digest(train['ids']),
                    'saved_combined_cohort_sha256': checkpoint['cohort_input_sha256'],
                    'historical_combined_cohort_recomputed': False,
                    'historical_content_limit': 'Checkpoint stores only combined train/val hash; training content is newly pinned, not independently matched to historical per-train hash.',
                    'train_scaler_max_difference': differences, 'model_state_sha256': state_digest(model),
                    'training_patients': len(train['ids']), 'training_events': int(train['event'].sum()),
                    'training_patches': sum(bag.n_patches for bag in train['bags'])}
        yield model, train, normalized, cfg, checkpoint, metadata
        if state_digest(model) != metadata['model_state_sha256'] or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Model state or parameter .grad changed during diagnosis')
        if stream_cohort_fingerprint(cohort, source) != train_hash:
            raise RuntimeError('Training inputs changed during diagnosis')
        if any(sha256_file(paths[key]) != value for key, value in hashes.items()):
            raise RuntimeError('Input artifacts changed during diagnosis')
        if any(sha256_file(path) != value for path, value in rna_hashes.items()):
            raise RuntimeError('Training RNA files changed during diagnosis')
    finally:
        for bag in cohort['train']['bags']:
            if hasattr(bag, 'close'):
                bag.close()


def plan_payload(model, train, cfg, checkpoint, metadata):
    stream = checkpoint['stream_config']
    return {'protocol': PROTOCOL, 'order_seed': ORDER_SEED, 'repeats': list(REPEATS),
            'batch_count': BATCH_COUNT, 'patient_batch_size': stream['patient_batch_size'],
            'patch_chunk_size': stream['patch_chunk_size'], 'view_source': stream['view_source'],
            'config': asdict(cfg), 'effective_loss_weights': effective_weights(cfg),
            'model_kwargs': checkpoint['model_kwargs'], 'pathway_mapping': checkpoint['pathway_mapping'],
            'inputs': metadata, 'source_sha256': source_hashes(),
            'runtime_versions': {'python': platform.python_version(), 'torch': str(torch.__version__)},
            'batches': fixed_batches(train['ids'], stream['patient_batch_size'], stream['minimum_patient_batch']),
            'validation_loaded': False, 'test_loaded': False,
            'selection': 'first 3 batches of label-blind CPU randperm, production tail merge; all patches; no result-based retries',
            'mode': 'train for objective semantics; no optimizer, no updates, no validation prediction'}


def make_plan(manifest, run_dir, expected_sha256):
    with loaded_training_reference(manifest, run_dir, expected_sha256, torch.device('cpu')) as loaded:
        model, train, _, cfg, checkpoint, metadata = loaded
        result = plan_payload(model, train, cfg, checkpoint, metadata)
    return result


def numeric_summary(values):
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(x):
        return None
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite audit summary')
    return {'count': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'min': float(x.min()), 'max': float(x.max())}


def summarize_relations(items):
    return {'status_counts': dict(Counter(x['status'] for x in items)),
            'direction_counts': dict(Counter(x['direction'] for x in items if x['direction'] is not None)),
            **{key: numeric_summary([x[key] for x in items]) for key in (
                'cosine', 'survival_norm', 'weighted_auxiliary_norm_in_survival_space',
                'weighted_norm_ratio_to_survival', 'weighted_dot',
                'outside_survival_graph_raw_auxiliary_norm')}}


def summarize_records(records):
    result = {}
    for name in LOSS_NAMES:
        terms = [r['audit']['terms'][name] for r in records]
        computed = [t for t in terms if t['status'] == 'computed']
        result[name] = {'term_status_counts': dict(Counter(t['status'] for t in terms)),
                        'survival_graph_space': summarize_relations([t['survival_graph_space'] for t in computed]),
                        'modules': {module: summarize_relations([t['modules'][module] for t in computed])
                                    for module in (computed[0]['modules'] if computed else [])}}
    totals = [r['audit']['sum_weighted_auxiliary'] for r in records]
    result['sum_weighted_auxiliary'] = {
        'survival_graph_space': summarize_relations([t['survival_graph_space'] for t in totals]),
        'modules': {module: summarize_relations([t['modules'][module] for t in totals])
                    for module in totals[0]['modules']}}
    return result


def rng_snapshot(device):
    return {'cpu': torch.get_rng_state().clone(),
            'cuda': torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None}


def run(manifest, run_dir, expected_sha256, plan_path, expected_plan_sha256, device):
    device = torch.device(device)
    if sha256_file(plan_path).lower() != expected_plan_sha256.lower():
        raise ValueError('Plan SHA256 differs from pre-audit record')
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if device.type not in ('cpu', 'cuda') or torch.is_inference_mode_enabled():
        raise ValueError('Audit requires CPU/CUDA and autograd, not inference_mode')
    if device.type == 'cuda':
        torch.zeros(1, device=device)  # Explicit context init; no training wrapper.
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    records = []
    with loaded_training_reference(manifest, run_dir, expected_sha256, device) as loaded:
        model, train, normalized, cfg, checkpoint, metadata = loaded
        actual_plan = plan_payload(model, train, cfg, checkpoint, metadata)
        if actual_plan != plan:
            raise ValueError('Plan no longer matches inputs, source, config, versions or fixed batch schedule')
        state_before = state_digest(model)
        rng_before = rng_snapshot(device)
        with torch.enable_grad():
            for entry in plan['batches']:
                indices = torch.tensor(entry['indices_in_training_order'], dtype=torch.long)
                batch = {'bags': [train['bags'][i] for i in indices.tolist()],
                         'rna': normalized[indices].to(device), 'time': train['time'][indices].to(device),
                         'event': train['event'][indices].to(device), 'rna_mask': checkpoint['rna_mask']}
                for seeds in entry['views']:
                    batch_start = time.perf_counter()
                    audit = audit_batch(model, batch, checkpoint['starts'].to(device), cfg,
                                        plan['patch_chunk_size'], plan['view_source'],
                                        seeds['view_seed'], seeds['rna_noise_seed'])
                    if state_digest(model) != state_before or any(p.grad is not None for p in model.parameters()):
                        raise RuntimeError('Fixed model state or .grad changed in a gradient audit')
                    record = {'batch_index': entry['batch_index'], 'repeat': seeds['repeat'],
                              'patient_ids': entry['patient_ids'], 'patients': len(indices),
                              'events': int(batch['event'].sum()),
                              'patches': sum(bag.n_patches for bag in batch['bags']),
                              'seconds': time.perf_counter()-batch_start, 'audit': audit}
                    records.append(record)
                    print(json.dumps({'batch_index': entry['batch_index'], 'repeat': seeds['repeat'],
                                      'patients': len(indices), 'seconds': record['seconds'],
                                      'gradient_reconstruction': audit['gradient_reconstruction']}), flush=True)
        rng_after = rng_snapshot(device)
        if any(not torch.equal(value, rng_after[key]) for key, value in rng_before.items() if value is not None):
            raise RuntimeError('Diagnostic changed caller CPU/CUDA RNG state')
        state_after = state_digest(model)
    # This is after context-exit input/model guards, never emit success before them.
    if sha256_file(plan_path).lower() != expected_plan_sha256.lower() or source_hashes() != plan['source_sha256']:
        raise RuntimeError('Plan or source changed during diagnosis')
    return {'protocol': PROTOCOL, 'plan_sha256': expected_plan_sha256.lower(), 'plan': plan,
            'training_performed': False, 'optimizer_steps': 0, 'validation_loaded': False, 'test_loaded': False,
            'model_state_before_sha256': state_before, 'model_state_after_sha256': state_after,
            'parameter_grads_remain_none': True, 'rng_unchanged': True, 'inputs_unchanged': True,
            'unique_diagnosed_patients': len({p for r in records for p in r['patient_ids']}),
            'records': records, 'summary': summarize_records(records),
            'runtime': {'device': str(device), 'seconds': time.perf_counter()-begin,
                        'cuda_max_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                        'cuda_max_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None},
            'limits': ['Current checkpoint local Euclidean gradients, not historical gradients or AdamW update directions.',
                       'Three training batches with two views each; repeated views are not independent patients.',
                       'Negative auxiliary inner product does not establish worse validation or test performance.',
                       'No loss selection, coefficient changes, model updates, validation inference or cross-validation.',
                       plan['inputs']['historical_content_limit']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--plan-out', help='Create a new CPU-only plan; no forward/backward')
    choice.add_argument('--plan', help='Previously saved plan; never reselect batches')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--device', choices=('cpu', 'cuda:0', 'cuda:1'), default='cpu')
    parser.add_argument('--out', help='New audit JSON path, no overwrite')
    args = parser.parse_args()
    if args.plan_out:
        if args.out or args.plan_sha256 or args.device != 'cpu':
            parser.error('Plan stage requires CPU and only --plan-out')
        destination = Path(args.plan_out)
    else:
        if not args.out or not args.plan_sha256:
            parser.error('Audit requires --out and independently recorded --plan-sha256')
        destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    if args.plan_out:
        result = make_plan(args.manifest, args.run_dir, args.checkpoint_sha256)
    else:
        result = run(args.manifest, args.run_dir, args.checkpoint_sha256,
                     args.plan, args.plan_sha256, args.device)
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'out': str(destination), 'sha256': sha256_file(destination), 'protocol': PROTOCOL}), flush=True)


if __name__ == '__main__':
    main()
