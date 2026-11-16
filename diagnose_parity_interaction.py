"""Frozen legacy parity interaction interventions, not retrained ablations.

Requires PathwayLocalSurvival (the dense architecture is explicitly rejected).
Reports scaled contribution and patient-wise logit spans separately from raw
interaction. Missing historical input contracts are exposed, never described as
successful baseline reproduction. No parameter updates or epoch selection.
"""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cancer_parity import discrete_survival, parity_metrics
from cancer_parity_bag import BudgetedPatientBag
from cancer_pathway import PathwayLocalSurvival, read_pathway_mapping
from cancer_pathway_stream import pathway_stream_forward
from cancer_rna import transform_rna
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
from diagnose_rna_only_baseline import metric_text, write_result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def legacy_model(checkpoint, device):
    architecture = checkpoint.get('architecture')
    state = checkpoint['model_state']
    if (architecture not in (None, 'PathwayLocalSurvival') or
            any(k.startswith(('decomposition_head.', 'token_cooperation.')) for k in state)):
        raise ValueError('This diagnostic supports legacy PathwayLocalSurvival only; dense checkpoints are rejected')
    with torch.random.fork_rng(devices=[]):
        model = PathwayLocalSurvival(**checkpoint['model_kwargs'])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_reference(path):
    reference = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            identity = row['patient_id'].strip()
            t, e = float(row['time']), float(row['event'])
            if (not identity or identity in reference or not np.isfinite(t) or t < 0
                    or e not in (0., 1.) or row.get('split', 'train').strip() not in ('train', 'val')):
                raise ValueError('Unique finite train/val survival-reference labels required')
            reference[identity] = (t, e)
    return reference


def common_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--survival-reference', required=True)
    parser.add_argument('--num-patches', type=int, required=True)
    parser.add_argument('--n-bins', type=int, default=None, help='Optional assertion against the checkpoint')
    parser.add_argument('--patch-chunk-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--device', default='cpu', help='Explicit device; no automatic GPU selection')
    parser.add_argument('--out', required=True)
    return parser


@contextmanager
def loaded_legacy_inputs(args):
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    if args.num_patches < 1 or args.patch_chunk_size < 1:
        raise ValueError('Positive patch budget and chunk size required')
    paths = {'checkpoint': Path(args.checkpoint), 'manifest': Path(args.manifest),
             'survival_reference': Path(args.survival_reference)}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
    missing = []
    adjacent_report = paths['checkpoint'].parent/'report.json'
    if adjacent_report.exists():
        paths['adjacent_report'] = adjacent_report
        hashes['adjacent_report'] = file_sha256(adjacent_report)
        reported = json.loads(adjacent_report.read_text(encoding='utf-8-sig')).get('controls', {})
        if (not checkpoint.get('controls') and reported.get('frozen_modality', 'none') != 'none'):
            raise ValueError('Historical checkpoint omits controls but its report records a frozen modality; replay is unverified')
    for key, value in (('manifest_sha256', hashes['manifest']),
                       ('survival_reference_sha256', hashes['survival_reference'])):
        if key not in checkpoint:
            missing.append(key)
        elif checkpoint[key] != value:
            raise ValueError(f'Checkpoint {key} does not match the diagnostic input')
    if checkpoint.get('architecture') is None:
        missing.append('architecture')
    if 'patch_chunk_size' not in checkpoint:
        missing.append('patch_chunk_size')
    elif checkpoint['patch_chunk_size'] != args.patch_chunk_size:
        raise ValueError('Requested patch chunk size differs from checkpoint input contract')
    model = legacy_model(checkpoint, torch.device(args.device))
    bins = checkpoint['model_kwargs']['bins']
    if args.n_bins is not None and args.n_bins != bins:
        raise ValueError('Requested bin count differs from checkpoint')
    edges = np.asarray(checkpoint['bin_edges'], dtype=float)
    if (edges.shape != (bins+1,) or not np.isneginf(edges[0]) or not np.isposinf(edges[-1])
            or not np.isfinite(edges[1:-1]).all() or not np.all(np.diff(edges) > 0)):
        raise ValueError('Invalid checkpoint bin-edge contract')
    for key in ('rna_mean', 'rna_scale', 'rna_mask', 'rna_preprocessing'):
        if key not in checkpoint:
            raise ValueError(f'Saved RNA transform required: {key}')
    genes, pathways = getattr(args, 'genes', None), getattr(args, 'pathways', None)
    if bool(genes) != bool(pathways):
        raise ValueError('Supply both optional genes and pathways assertions, or neither')
    if genes:
        expected = checkpoint['pathway_mapping']
        mapping = read_pathway_mapping(genes, pathways, checkpoint['model_kwargs']['rna_dim'],
                                       expected.get('minimum_matched_genes', 1))
        for key in ('gene_sha256', 'pathway_sha256'):
            if mapping[key] != expected.get(key):
                raise ValueError(f'Supplied pathway schema differs from checkpoint: {key}')
    cohort = load_stream_manifest(args.manifest, path_views='single_view', include_test=False)
    try:
        current_cohort = stream_cohort_fingerprint(cohort, path_views='single_view')
        if 'cohort_input_sha256' not in checkpoint:
            missing.append('cohort_input_sha256')
        elif current_cohort != checkpoint['cohort_input_sha256']:
            raise ValueError('Cohort content differs from checkpoint')
        reference = load_reference(args.survival_reference)
        if set(reference) != {i for split in ('train', 'val') for i in cohort[split]['ids']}:
            raise ValueError('Survival reference IDs must exactly cover train and validation')
        observed = {}
        for split in ('train', 'val'):
            values = np.array([reference[i] for i in cohort[split]['ids']])
            if (not np.array_equal(values[:, 1], cohort[split]['event'].numpy()) or
                    not np.allclose(values[:, 0], cohort[split]['time'].numpy(), rtol=2e-7, atol=2e-6)):
                raise ValueError('Survival reference disagrees with manifest outcomes')
            observed[split] = values
        normalized = transform_rna(cohort['val']['rna'], checkpoint)
        bags = [BudgetedPatientBag(b, args.num_patches, 'eval', args.seed) for b in cohort['val']['bags']]
        from parity_checkpoint_controls import apply_checkpoint_controls
        bags, normalized, controls = apply_checkpoint_controls(checkpoint, bags, normalized)
        if not controls.get('input_contract_verified', False):
            missing.append('input_controls_or_patch_budget')
        before = state_digest(model)
        evidence = {'input_sha256': hashes, 'current_cohort_input_sha256': current_cohort,
                    'missing_historical_contracts': missing,
                    'input_verification': 'historical_input_unverified' if missing else 'saved_input_contracts_verified',
                    'baseline_prediction_reproduction_checked': False,
                    'baseline_note': 'No archived prediction comparison is performed; this is recomputation, not claimed exact reproduction.',
                    'controls_replay': controls, 'rna_transform': 'saved checkpoint values; no refitting',
                    'checkpoint_discrete_nll_version': checkpoint.get('loss_version',
                        checkpoint.get('discrete_nll_version', 'historical_unversioned')),
                    'selection': checkpoint.get('selection', 'historical_unspecified'),
                    'parameters_updated': False, 'model_state_sha256_before': before,
                    'architecture': 'PathwayLocalSurvival', 'bins': bins,
                    'num_patches': args.num_patches, 'patch_chunk_size': args.patch_chunk_size}
        yield model, bags, normalized.to(args.device), observed, edges, evidence
        if state_digest(model) != before:
            raise RuntimeError('Diagnostic changed model parameters or buffers')
        if any(file_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError('Diagnostic inputs changed while running')
        for data in cohort.values():
            for bag in data['bags']:
                bag.assert_unchanged()
        evidence['model_state_sha256_after'] = before
        evidence['model_and_input_guards_passed'] = True
    finally:
        for data in cohort.values():
            for bag in data['bags']:
                bag.close()


def predict_at_strength(model, bags, rna, chunk_size, strength):
    if not np.isfinite(strength) or strength < 0:
        raise ValueError('Finite nonnegative interaction strength required')
    original = model.interaction_strength
    try:
        model.interaction_strength = float(strength)
        with torch.no_grad():
            return pathway_stream_forward(model, bags, rna, chunk_size, 'feature_noise',
                                          include_views=False, include_patch=False,
                                          checkpoint_chunks=False)
    finally:
        model.interaction_strength = original


def decompose_output(output, strength):
    logits = output['log_rates'].detach().double().cpu()
    raw = output['interaction_logits'].detach().double().cpu()
    if logits.shape != raw.shape or logits.ndim != 2 or not torch.isfinite(logits).all() or not torch.isfinite(raw).all():
        raise ValueError('Finite paired patient-by-bin logits required')
    weighted = float(strength)*raw
    base = logits-weighted
    def span(x):
        return x.amax(0)-x.amin(0)
    base_span, weighted_span = span(base), span(weighted)
    base_mean = float(base.abs().mean())
    ratios = [float(a/b) if b > 1e-12 else None for a, b in zip(weighted_span, base_span)]
    return {'interaction_strength': float(strength),
            'raw_interaction_abs_mean': float(raw.abs().mean()),
            'weighted_interaction_abs_mean': float(weighted.abs().mean()),
            'base_abs_mean': base_mean,
            'weighted_to_base_abs_ratio': float(weighted.abs().mean())/base_mean if base_mean > 1e-12 else None,
            'raw_interaction_span_by_bin': span(raw).tolist(),
            'weighted_interaction_span_by_bin': weighted_span.tolist(),
            'base_span_by_bin': base_span.tolist(), 'total_span_by_bin': span(logits).tolist(),
            'weighted_to_base_span_ratio_by_bin': ratios,
            'span_definition': 'max minus min across patients within each time bin; undefined ratios are null',
            'ratio_is_attribution_fraction': False}


def score_output(output, observed, edges):
    survival = discrete_survival(output['log_rates'].detach().cpu()).numpy()
    return parity_metrics(observed['train'][:, 0], observed['train'][:, 1],
                          observed['val'][:, 0], observed['val'][:, 1], survival, edges)


def main():
    args = common_parser().parse_args()
    with loaded_legacy_inputs(args) as (model, bags, rna, observed, edges, evidence):
        original = float(model.interaction_strength)
        cases = []
        for tag, strength in [('checkpoint_original', original), ('interaction_off', 0.), ('interaction_strength_1', 1.)]:
            output = predict_at_strength(model, bags, rna, args.patch_chunk_size, strength)
            metrics = score_output(output, observed, edges)
            cases.append({'case': tag, **metrics, **decompose_output(output, strength),
                          'matching_normalized_entropy_mean': float(output['matching_normalized_entropy'].mean()),
                          'matching_max_weight_mean': float(output['matching_max_weight'].mean())})
        result = {'checkpoint': args.checkpoint, 'evidence': evidence, 'cases': cases,
                  'scope': 'Frozen legacy model inference intervention; not retrained model comparison or generalization proof.',
                  'original_interaction_strength': original,
                  'caution': 'An unchanged C-index does not prove unchanged predictions or an unused interaction branch.'}
    write_result(args.out, result)
    for item in cases:
        print(f"{item['case']:28} C={metric_text(item['c_index'])} IPCW={metric_text(item['c_index_ipcw'])} "
              f"IBS={metric_text(item['ibs'])} IAUC={metric_text(item['iauc'])} "
              f"weighted/base={metric_text(item['weighted_to_base_abs_ratio'])}")


if __name__ == '__main__':
    main()
