"""Frozen-checkpoint validation decomposition; no fitting, test data or training.

Query interventions preserve target pathology, global RNA and RNA value tokens.
The training-mean query uses training RNA only. Attention differences are exact
two-pass streaming statistics, never a full patient-by-patch attention table.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import _feature_scores
from cancer_rna import fit_rna_scaler, transform_rna
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
from cancer_survival_diagnostics import cumulative_hazard_at, horizon_risk_audit
from diagnose_pathway_query_permutation import (check_artifacts, compare_original,
    fixed_derangements, forward_patient, sha256_file, validate_manifest_splits, validation_metrics)


def summary(values):
    x = np.asarray(values, dtype=float)
    if not x.size or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite diagnostic values')
    return {'mean': float(x.mean()), 'median': float(np.median(x)),
            'p90': float(np.percentile(x, 90)), 'max': float(x.max())}


def paired_change(before, after, relative_floor=1e-8):
    """Each row is a patient; relative norms are undefined for near-zero rows."""
    a, b = before.detach().double().cpu(), after.detach().double().cpu()
    if a.shape != b.shape or a.ndim < 1 or not len(a):
        raise ValueError('Expected paired patient arrays')
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    delta, norm = b-a, a.norm(dim=1)
    valid = norm > relative_floor
    return {'mean_signed_element_change': float(delta.mean()),
            'patient_mean_absolute': summary(delta.abs().mean(1).numpy()),
            'patient_l2_change': summary(delta.norm(dim=1).numpy()),
            'relative_l2': summary((delta.norm(dim=1)[valid]/norm[valid]).numpy()) if valid.any() else None,
            'relative_l2_defined_patients': int(valid.sum()),
            'relative_l2_near_zero_baseline_patients': int((~valid).sum()),
            'relative_l2_denominator_floor': relative_floor}


def training_mean_query(model, normalized_training_rna, floor=1e-6):
    """Mean unit query per pathway, then renormalize; near-zero means become zero."""
    if not len(normalized_training_rna):
        raise ValueError('Training RNA is required for the fixed query')
    total = None
    with torch.inference_mode():
        for row in normalized_training_rna.split(1):
            q = model.match_queries(model.pathways(row))
            total = q.double() if total is None else total+q.double()
        mean = total/len(normalized_training_rna)
        norms = mean.norm(dim=-1, keepdim=True)
        valid = norms > floor
        fixed = torch.where(valid, mean/norms.clamp_min(floor), torch.zeros_like(mean)).to(q.dtype)
    return fixed, {'source': 'training RNA only, checkpoint scaler/tokenizer/match map; no validation fitting',
                   'training_patients': len(normalized_training_rna), 'mean_unit_query_norms': norms[0, :, 0].cpu().tolist(),
                   'near_zero_mean_pathway_indices': (~valid)[0, :, 0].nonzero().flatten().cpu().tolist(),
                   'mean_norm_floor': floor, 'fallback': 'zero query gives uniform patch scores for that pathway',
                   'fixed_query_sha256': hashlib.sha256(fixed.cpu().contiguous().numpy().tobytes()).hexdigest()}


def streaming_attention_l1(model, bag, queries, chunk_size):
    """Return [query-case,pathway] L1 to queries[0], normalized over ALL patches."""
    if not queries or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Queries and positive chunk size required')
    parameter = next(model.parameters())
    k = model.pathway_count
    def scores(block):
        raw, = bag.read_block(block, ('features_reference',))
        features = model.base.patch_encoder(raw.to(parameter))
        return torch.stack([_feature_scores(model, features, q[0])[:, -k:]
                            for q in queries], dim=1).double()
    with torch.inference_mode():
        normalizers, seen = None, 0
        for block in bag.blocks(chunk_size):
            current = scores(block).logsumexp(0)
            normalizers = current if normalizers is None else torch.logaddexp(normalizers, current)
            seen += block[2]-block[1]
        if seen != bag.n_patches or normalizers is None:
            raise ValueError('Attention pass did not cover the complete patient bag')
        distance = torch.zeros_like(normalizers)
        mass = torch.zeros_like(normalizers)
        seen = 0
        for block in bag.blocks(chunk_size):
            weights = (scores(block)-normalizers).exp()
            distance += (weights-weights[:, :1, :]).abs().sum(0)
            mass += weights.sum(0)
            seen += block[2]-block[1]
        if seen != bag.n_patches or not torch.allclose(mass, torch.ones_like(mass), atol=1e-7, rtol=1e-7):
            raise ValueError('Attention weights do not sum to one over the complete bag')
        return distance.cpu()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def loaded_reference(manifest, run_dir, expected_sha256, device):
    validate_manifest_splits(manifest)  # Reject test rows BEFORE opening any feature/RNA.
    directory = Path(run_dir)
    paths = {'manifest': Path(manifest), 'checkpoint': directory/'checkpoint.pt',
             'report': directory/'report.json', 'predictions': directory/'val_predictions.csv'}
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    if hashes['checkpoint'].lower() != expected_sha256.lower():
        raise ValueError('Checkpoint SHA256 differs from the explicitly supplied expected value')
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
    report = json.loads(paths['report'].read_text(encoding='utf-8'))
    check_artifacts(checkpoint, report, hashes['manifest'])
    strength = checkpoint['model_kwargs']['interaction_strength']
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError('Expected an R checkpoint with positive interaction strength')
    if 'interaction_strength' in checkpoint['config'] and checkpoint['config']['interaction_strength'] != strength:
        raise ValueError('Config and model interaction strengths disagree')
    source = checkpoint['stream_config']['view_source']
    if source not in ('feature_noise', 'image_d4'):
        raise ValueError('Unsupported checkpoint view source')
    path_views = 'single_view' if source == 'feature_noise' else 'cached_uni'
    cohort = load_stream_manifest(manifest, path_views=path_views, include_test=False)
    try:
        if stream_cohort_fingerprint(cohort, path_views) != checkpoint['cohort_input_sha256']:
            raise ValueError('Loaded cohort fingerprint differs from checkpoint')
        preprocessing = checkpoint['rna_preprocessing']
        refit = fit_rna_scaler(cohort['train']['rna'], preprocessing['policy'], preprocessing['min_std'])
        differences = {}
        for key in ('rna_mean', 'rna_scale', 'rna_mask'):
            a, b = checkpoint[key].cpu(), refit[key].cpu()
            same = torch.equal(a, b) if key == 'rna_mask' else torch.allclose(a, b, atol=1e-8, rtol=1e-5)
            if not same:
                raise ValueError(f'Saved {key} differs from train-only refit')
            if key != 'rna_mask':
                differences[key] = float((a-b).abs().max())
        with torch.random.fork_rng(devices=[]):
            model = PathwayLocalSurvival(**checkpoint['model_kwargs']).to(device)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        model.eval().requires_grad_(False)
        fitted = {key: checkpoint[key] for key in ('rna_mean', 'rna_scale', 'rna_mask', 'rna_preprocessing')}
        yield model, cohort, fitted, checkpoint, report, paths, hashes, differences
        if any(sha256_file(paths[key]) != value for key, value in hashes.items()):
            raise RuntimeError('Input artifacts changed during diagnosis')
    finally:
        for split in cohort.values():
            for bag in split['bags']:
                bag.close()


def run(manifest, run_dir, expected_sha256, device, count=5, seed=20261110):
    with loaded_reference(manifest, run_dir, expected_sha256, device) as loaded:
        model, cohort, fitted, checkpoint, report, paths, hashes, refit_info = loaded
        before_state = state_digest(model)
        train = transform_rna(cohort['train']['rna'], fitted).to(device)
        val = transform_rna(cohort['val']['rna'], fitted).to(device)
        starts, horizon = checkpoint['starts'].cpu(), float(report['prediction_horizon'])
        cfg, strength = checkpoint['stream_config'], model.interaction_strength
        permutations = fixed_derangements(len(val), count, seed)
        names = ['original']+[f'permuted_{i+1}' for i in range(count)]+['training_fixed', 'uniform', 'interaction_off']
        collected = {name: [] for name in names}
        attention = []
        with torch.inference_mode():
            fixed, fixed_info = training_mean_query(model, train)
            queries = [model.match_queries(model.pathways(row)) for row in val.split(1)]
            # First reproduce the entire original cohort; no intervention runs
            # until the stored predictions and metrics pass the original guard.
            originals = [forward_patient(model, bag, val[i:i+1], cfg) for i, bag in enumerate(cohort['val']['bags'])]
            original_logits = torch.cat([o['log_rates'].cpu() for o in originals])
            original_metrics, original_survival = validation_metrics(original_logits, cohort, starts, horizon)
            original_error = compare_original(report, paths['predictions'], cohort['val']['ids'],
                                              original_metrics, original_survival)
            for i, (bag, original) in enumerate(zip(cohort['val']['bags'], originals)):
                base = model.base.head(torch.cat((original['path_hidden'], original['rna_hidden']), dim=-1))
                qcases = [queries[i]]+[queries[p[i]] for p in permutations]+[fixed, torch.zeros_like(queries[i])]
                attention.append(streaming_attention_l1(model, bag, qcases, cfg['patch_chunk_size']))
                for index, name in enumerate(names):
                    if name in ('original', 'interaction_off'):
                        output = original
                    else:
                        output = forward_patient(model, bag, val[i:i+1], cfg, qcases[index], original['rna_pathways'])
                    for key in ('path_hidden', 'rna_hidden', 'rna_pathways'):
                        if not torch.allclose(output[key], original[key], atol=1e-6, rtol=1e-6):
                            raise RuntimeError(f'Intervention unexpectedly changed target {key}')
                    weighted = strength*output['interaction_logits']
                    if not torch.allclose(output['log_rates'], base+weighted, atol=1e-6, rtol=1e-6):
                        raise RuntimeError('Base plus interaction does not reproduce prediction')
                    disabled = name == 'interaction_off'
                    record = {'matched_path': output['matched_path'], 'base_logits': base,
                              'interaction_raw': output['interaction_logits'],
                              'interaction_applied': torch.zeros_like(weighted) if disabled else weighted,
                              'logits': base if disabled else output['log_rates']}
                    collected[name].append({key: value.cpu() for key, value in record.items()})
                print(json.dumps({'validation_patient': i+1, 'of': len(val)}), flush=True)
        arrays = {name: {key: torch.cat([row[key] for row in records]) for key in records[0]}
                  for name, records in collected.items()}
        reference = arrays['original']
        attention = torch.stack(attention)  # [patient, query-case, pathway], never [patch,pathway].
        results = {}
        for index, name in enumerate(names):
            values = arrays[name]
            metrics, survival = validation_metrics(values['logits'], cohort, starts, horizon)
            h = cumulative_hazard_at(values['logits'], horizon, starts)
            original_h = cumulative_hazard_at(reference['logits'], horizon, starts)
            att = attention[:, 0 if name == 'interaction_off' else index, :]
            results[name] = {
                'metrics': metrics, 'difference_from_original': {key: metrics[key]-original_metrics[key]
                                                              for key in ('nll', 'harrell_c', 'ipcw_brier')},
                'risk_numerics': horizon_risk_audit(values['logits'], cohort['val']['time'], cohort['val']['event'], starts, horizon),
                'changes': {key: paired_change(reference[key], values[key]) for key in reference},
                'cumulative_hazard_change': paired_change(original_h, h),
                'survival_change': paired_change(torch.from_numpy(original_survival), torch.from_numpy(survival)),
                'attention_l1_patient_mean': summary(att.mean(1).numpy()),
                'attention_l1_mean_by_pathway': att.mean(0).tolist(),
                'matched_change_mean_l2_by_pathway': (values['matched_path']-reference['matched_path']).double().norm(dim=-1).mean(0).tolist(),
                'absolute_levels': {key: summary(value.double().reshape(len(val), -1).norm(dim=1).numpy())
                                    for key, value in values.items()}}
        after_state = state_digest(model)
        if after_state != before_state:
            raise RuntimeError('Frozen model state changed during diagnosis')
        source_names = ('diagnose_fixed_r_decomposition', 'diagnose_pathway_query_permutation', 'cancer_pathway',
                        'cancer_pathway_stream', 'cancer_model', 'cancer_metrics', 'cancer_survival_diagnostics',
                        'cancer_rna', 'cancer_stream_data', 'cancer_single_view', 'cancer_stream',
                        'cancer_auxiliary', 'model', 'uni_views.core')
        return {'diagnostic': 'fixed_R_validation_decomposition_v1', 'test_loaded': False, 'training_performed': False,
                'seed': seed, 'permutations': permutations, 'validation_patients': len(val),
                'validation_order_sha256': hashlib.sha256(json.dumps(cohort['val']['ids'], separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
                'horizon': horizon, 'interaction_strength': strength, 'fixed_query': fixed_info,
                'pathway_axis': [{'index': i, 'id': item.get('id'), 'name': item.get('name')}
                                 for i, item in enumerate(checkpoint['pathway_mapping']['pathways'])],
                'reference_metric_conventions': report.get('metric_conventions'),
                'original_reproduction_max_survival_difference': original_error,
                'train_scaler_refit_max_differences': refit_info, 'cases': results,
                'model_state_before_sha256': before_state, 'model_state_after_sha256': after_state,
                'inputs': {**{key+'_sha256': value for key, value in hashes.items()},
                           'script_sha256': sha256_file(__file__),
                           'cohort_input_sha256': checkpoint['cohort_input_sha256'],
                           'source_sha256': {name: sha256_file(importlib.util.find_spec(name).origin) for name in source_names}},
                'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__, 'device': str(device),
                            'rng': 'numpy default_rng PCG64 for validation derangements'},
                'limits': ['All cases reuse one validation-selected frozen checkpoint and the same patients; no independent test evidence.',
                           'Query replacement is a distribution perturbation, not a causal or spatial-expression estimate.',
                           'interaction_off retains raw branch diagnostics but applies zero contribution; this is not retraining the I arm.',
                           'Fixed-query calibration uses training RNA only, no pathology or validation labels.',
                           'Patient summaries weight patients equally; pathways/patches are not independent patients.',
                           'Attention L1 is computed in float64 using reference keys and two complete streaming passes; range [0,2].']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--run-dir', required=True, help='New-version R checkpoint/report/val_predictions directory')
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--out', required=True, help='New JSON path; never overwritten')
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    data = run(args.manifest, args.run_dir, args.checkpoint_sha256, torch.device(args.device))
    with Path(args.out).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Frozen R decomposition saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
