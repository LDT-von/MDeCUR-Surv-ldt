"""Validation-only model-dependence diagnosis: permute pathology-matching queries.

Only the RNA-derived query used for patch scores is replaced by another
validation patient's query. The patient's pathology, global RNA and original
RNA pathway value tokens remain unchanged. This distribution perturbation is
not a causal effect or a measurement of patch-level gene expression.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cancer_metrics import harrell_c, ipcw_brier
from cancer_model import piecewise_nll, survival_at
from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import pathway_stream_forward
from cancer_rna import fit_rna_scaler, transform_rna
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest_splits(manifest):
    with Path(manifest).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if 'split' not in (reader.fieldnames or []):
            raise ValueError('Manifest has no split field')
        splits = {row['split'].strip().lower() for row in reader}
    if splits != {'train', 'val'}:
        raise ValueError('Manifest must contain train and val only, with no test rows')


def fixed_derangements(n, count, seed):
    """Uniformly sample permutations conditioned on no fixed point."""
    if n < 2 or count < 1:
        raise ValueError('At least two validation patients and one permutation are required')
    rng = np.random.default_rng(seed)
    target = np.arange(n)
    permutations = []
    for _ in range(count):
        for attempt in range(10000):
            perm = rng.permutation(n)
            if np.all(perm != target):
                permutations.append(perm.tolist())
                break
        else:
            raise RuntimeError('Failed to sample a derangement')
    return permutations


def forward_patient(model, bag, normalized_rna, stream_config, donor_query=None, expected_tokens=None):
    """Run the original model; optionally replace exactly one call to its query method."""
    kwargs = dict(chunk_size=stream_config['patch_chunk_size'], view_source=stream_config['view_source'],
                  checkpoint_chunks=False)
    if donor_query is None:
        return pathway_stream_forward(model, [bag], normalized_rna, **kwargs)
    if expected_tokens is None:
        raise ValueError('Original target-patient RNA tokens are required for query replacement')

    def replacement(tokens):
        if tokens.shape != expected_tokens.shape or not torch.allclose(tokens, expected_tokens, atol=1e-6, rtol=1e-6):
            raise ValueError('Query received unexpected target-patient RNA tokens')
        if donor_query.shape[:-1] != tokens.shape[:-1] or donor_query.device != tokens.device:
            raise ValueError('Donor query shape or device is incompatible')
        return donor_query

    with patch.object(model, 'match_queries', side_effect=replacement) as hooked:
        output = pathway_stream_forward(model, [bag], normalized_rna, **kwargs)
        if hooked.call_count != 1:
            raise RuntimeError('Expected exactly one target query call in inference')
    return output


def validation_metrics(logits, cohort, starts, horizon):
    val, train = cohort['val'], cohort['train']
    with torch.no_grad():
        logits = logits.cpu()
        nll = float(piecewise_nll(logits, val['time'], val['event'], starts).item())
        survival = survival_at(logits, torch.tensor([horizon]), starts)[:, 0].numpy()
    c = harrell_c(val['time'].numpy(), val['event'].numpy(), 1-survival)
    brier = ipcw_brier(train['time'].numpy(), train['event'].numpy(), val['time'].numpy(),
                       val['event'].numpy(), survival, horizon)
    if c['value'] is None or brier['value'] is None:
        raise ValueError('Metrics are not estimable under the project metric definitions')
    return {'nll': nll, 'harrell_c': c['value'], 'ipcw_brier': brier['value'],
            'comparable_pairs': c['comparable_pairs'], 'brier_info': brier}, survival


def patient_survival_changes(original, perturbed):
    """Summarize paired horizon survival changes without exporting patient IDs."""
    before = np.asarray(original, dtype=float)
    after = np.asarray(perturbed, dtype=float)
    if (before.ndim != 1 or before.size == 0 or after.shape != before.shape
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or np.any((before < 0) | (before > 1) | (after < 0) | (after > 1))):
        raise ValueError('Expected paired finite survival probabilities in [0, 1]')
    delta = after - before
    absolute = np.abs(delta)
    return {'patients': int(before.size), 'mean_signed': float(delta.mean()),
            'mean_absolute': float(absolute.mean()), 'median_absolute': float(np.median(absolute)),
            'p90_absolute': float(np.percentile(absolute, 90)), 'max_absolute': float(absolute.max())}


def check_artifacts(checkpoint, report, manifest_hash):
    if (checkpoint['architecture'] != 'PathwayLocalSurvival' or report['architecture'] != checkpoint['architecture']
            or checkpoint['evaluation_split'] != 'val' or report['evaluation_split'] != 'val'
            or checkpoint['test_evaluated'] is not False or report['test_evaluated'] is not False):
        raise ValueError('Expected a validation-only PathwayLocalSurvival checkpoint and report')
    if checkpoint['manifest_sha256'] != manifest_hash or report['manifest_sha256'] != manifest_hash:
        raise ValueError('Manifest fingerprint differs from checkpoint or report')
    for key in ('cohort_input_sha256', 'model_kwargs', 'pathway_mapping', 'rna_preprocessing',
                'config', 'stream_config', 'effective_loss_weights'):
        if checkpoint[key] != report[key]:
            raise ValueError(f'Checkpoint and report disagree on {key}')
    if checkpoint['stream_config']['view_source'] != report['view_source']:
        raise ValueError('View source differs between checkpoint and report')
    starts = checkpoint['starts'].cpu()
    if not torch.equal(starts, torch.tensor(report['interval_starts'], dtype=starts.dtype)):
        raise ValueError('Checkpoint and report differ in interval starts')
    if not math.isfinite(report['prediction_horizon']) or report['prediction_horizon'] <= 0:
        raise ValueError('Expected a finite positive prediction horizon')


def compare_original(report, csv_path, identities, metrics, survival):
    rows = {}
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            ident = row['patient_id']
            if not ident or ident in rows:
                raise ValueError('Validation prediction IDs must be unique and nonempty')
            rows[ident] = float(row['survival_at_horizon'])
    if set(rows) != set(identities):
        raise ValueError('Stored validation predictions do not match validation IDs')
    saved = np.array([rows[identity] for identity in identities])
    if not np.isfinite(saved).all() or not np.allclose(saved, survival, atol=1e-6, rtol=1e-6):
        raise ValueError('Original checkpoint predictions disagree with stored predictions')
    for key, original in (('nll', report['evaluation_nll']),
                          ('harrell_c', report['evaluation_c_index_risk_at_horizon']['value']),
                          ('ipcw_brier', report['evaluation_ipcw_brier_at_horizon']['value'])):
        if (original is None or not math.isfinite(original) or not math.isfinite(metrics[key])
                or abs(metrics[key]-original) > 2e-5):
            raise ValueError(f'Recomputed original {key} differs from saved report')
    return float(np.max(np.abs(saved-survival)))


def run(manifest, run_dir, expected_sha256, device, count, seed):
    validate_manifest_splits(manifest)
    directory = Path(run_dir)
    ckpt_path, report_path = directory/'checkpoint.pt', directory/'report.json'
    prediction_path = directory/'val_predictions.csv'
    actual_sha = sha256_file(ckpt_path)
    if actual_sha.lower() != expected_sha256.lower():
        raise ValueError('Checkpoint SHA256 differs from the explicitly supplied expected value')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    check_artifacts(checkpoint, report, sha256_file(manifest))
    source = checkpoint['stream_config']['view_source']
    if source not in ('feature_noise', 'image_d4'):
        raise ValueError('Unsupported view source')
    path_views = 'single_view' if source == 'feature_noise' else 'cached_uni'
    cohort = load_stream_manifest(manifest, path_views=path_views, include_test=False)
    try:
        if stream_cohort_fingerprint(cohort, path_views=path_views) != checkpoint['cohort_input_sha256']:
            raise ValueError('Loaded train/validation cohort fingerprint differs from checkpoint')
        preprocessing = checkpoint['rna_preprocessing']
        refit = fit_rna_scaler(cohort['train']['rna'], preprocessing['policy'], preprocessing['min_std'])
        scaler_max_abs_differences = {}
        for key in ('rna_mean', 'rna_scale', 'rna_mask'):
            saved, recalculated = checkpoint[key].cpu(), refit[key].cpu()
            same = (torch.equal(saved, recalculated) if key == 'rna_mask' else
                    torch.allclose(saved, recalculated, atol=1e-8, rtol=1e-5))
            if not same:
                raise ValueError(f'Saved {key} differs from recomputation on training patients')
            if key != 'rna_mask':
                scaler_max_abs_differences[key] = (saved-recalculated).abs().max().item()
        model = PathwayLocalSurvival(**checkpoint['model_kwargs']).to(device)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        model.eval()
        fitted = {key: checkpoint[key] for key in ('rna_mean', 'rna_scale', 'rna_mask', 'rna_preprocessing')}
        val = cohort['val']
        normalized = transform_rna(val['rna'], fitted).to(device)
        starts = checkpoint['starts'].cpu()
        horizon = float(report['prediction_horizon'])
        perms = fixed_derangements(len(val['ids']), count, seed)
        saved_config = checkpoint['stream_config']
        with torch.inference_mode():
            # Compute each donor query separately, preserving original single-patient inference semantics.
            queries = [model.match_queries(model.pathways(normalized[i:i+1])) for i in range(len(val['ids']))]
            original = [forward_patient(model, bag, normalized[i:i+1], saved_config)
                        for i, bag in enumerate(val['bags'])]
            original_logits = torch.cat([item['log_rates'].cpu() for item in original])
            metrics, original_survival = validation_metrics(original_logits, cohort, starts, horizon)
            max_abs = compare_original(report, prediction_path, val['ids'], metrics, original_survival)
            results = []
            for number, perm in enumerate(perms, 1):
                logits = []
                for i, bag in enumerate(val['bags']):
                    changed = forward_patient(model, bag, normalized[i:i+1], saved_config,
                                              queries[perm[i]], original[i]['rna_pathways'])
                    for key in ('path_hidden', 'rna_hidden', 'rna_pathways'):
                        if not torch.allclose(changed[key], original[i][key], atol=1e-6, rtol=1e-6):
                            raise RuntimeError(f'Query permutation unexpectedly changed {key}')
                    logits.append(changed['log_rates'].cpu())
                perturbed, perturbed_survival = validation_metrics(torch.cat(logits), cohort, starts, horizon)
                difference = {key: perturbed[key]-metrics[key] for key in ('nll', 'harrell_c', 'ipcw_brier')}
                results.append({'number': number, 'donor_indices_in_validation_order': perm,
                                'metrics': perturbed, 'difference_permuted_minus_original': difference,
                                'paired_survival_change': patient_survival_changes(original_survival,
                                                                                  perturbed_survival)})
                print(json.dumps({'permutation': number, 'of': count, 'difference': difference}), flush=True)
        order_hash = hashlib.sha256(json.dumps(val['ids'], ensure_ascii=False, separators=(',', ':'))
                                    .encode('utf-8')).hexdigest()
        return {'diagnostic': 'validation_query_permutation_only', 'test_loaded': False,
                'seed': seed, 'permutations': count, 'validation_patients': len(val['ids']),
                'validation_order_sha256': order_hash, 'prediction_horizon': horizon,
                'original': metrics, 'original_max_absolute_survival_difference': max_abs,
                'train_refit_scaler_max_absolute_differences': scaler_max_abs_differences,
                'results': results, 'inputs': {'manifest_sha256': sha256_file(manifest),
                'cohort_input_sha256': checkpoint['cohort_input_sha256'], 'checkpoint_sha256': actual_sha,
                'report_sha256': sha256_file(report_path), 'predictions_sha256': sha256_file(prediction_path),
                'script_sha256': sha256_file(__file__),
                'source_sha256': {name: sha256_file(importlib.util.find_spec(name).origin) for name in
                                  ('cancer_metrics', 'cancer_model', 'cancer_pathway',
                                   'cancer_pathway_stream', 'cancer_rna', 'cancer_stream_data',
                                   'cancer_single_view', 'cancer_stream', 'cancer_auxiliary',
                                   'model', 'uni_views.core')}},
                'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__,
                            'permutation_rng': 'numpy default_rng (PCG64)', 'device': str(device)},
                'limits': ['Only query scores change; original patient pathology, global RNA, and RNA pathway value tokens are retained.',
                           'This is a distribution perturbation/model-dependence diagnostic, not a causal effect or spatial expression measurement.',
                           'Patient-level survival changes are descriptive, at the fixed prediction horizon; they do not measure prediction accuracy.',
                           'All permutations reuse the same validation patients and trained checkpoint; they are not independent test samples.',
                           'The validation cohort was used for checkpoint selection; do not infer independent performance or clinical utility.']}
    finally:
        for split in cohort.values():
            for bag in split['bags']:
                bag.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='explicit train/val-only input CSV')
    parser.add_argument('--run-dir', required=True, help='completed original checkpoint/report/predictions directory')
    parser.add_argument('--checkpoint-sha256', required=True, help='independently recorded expected checkpoint SHA256')
    parser.add_argument('--device', required=True, help='explicit torch device, e.g. cpu or cuda:0')
    parser.add_argument('--permutations', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--out', required=True, help='new JSON output path, never overwritten')
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    result = run(args.manifest, args.run_dir, args.checkpoint_sha256, torch.device(args.device),
                 args.permutations, args.seed)
    with Path(args.out).open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(f'Validation-only diagnostic saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
