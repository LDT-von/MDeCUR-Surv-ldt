"""Train the full local pathway/pathology survival framework on explicit inputs.

Default execution fits train, selects the checkpoint on validation, and never
loads test inputs.  The two pathology view sources are deliberately distinct:
feature_noise is a feature-level experiment, image_d4 requires verified UNI H5.
"""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from cancer_metrics import harrell_c, ipcw_brier
from cancer_model import piecewise_exposure, piecewise_nll, survival_at
from cancer_pathway import read_pathway_mapping
from cancer_pathway_stream import (PathwayConfig, effective_weights, minimum_batch_size,
                                   predict_pathway_model, train_pathway_model, validate_config)
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
from cancer_survival_diagnostics import METRIC_CONVENTIONS, cumulative_hazard_at, horizon_risk_audit


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Explicit train/val CSV; patient ordering and splits are authoritative')
    parser.add_argument('--genes', required=True, help='Ordered gene names, one per line, matching every RNA vector')
    parser.add_argument('--pathways', default=str(Path(__file__).parent/'resources'/'reactome_human_roots.gmt'),
                        help='Explicit GMT with gene symbols; default is the frozen Reactome human root sets')
    parser.add_argument('--min-pathway-genes', type=int, default=5,
                        help='Drop a pathway only when it has fewer exact gene-schema matches')
    parser.add_argument('--out', required=True, help='New output directory; existing directories are never overwritten')
    parser.add_argument('--view-source', choices=['feature_noise', 'image_d4'], required=True,
                        help='feature_noise uses single_view PT; image_d4 uses verified cached_uni H5')
    parser.add_argument('--evaluate-test', action='store_true',
                        help='Explicitly load/evaluate test after validation selects the checkpoint')
    parser.add_argument('--starts', required=True, nargs='+', type=float, help='Fixed interval starts, beginning at zero')
    parser.add_argument('--horizon', required=True, type=float, help='Evaluation time in the same units as labels')
    parser.add_argument('--time-unit', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--patient-batch-size', type=int, default=20)
    parser.add_argument('--patch-chunk-size', type=int, default=2048)
    parser.add_argument('--device', default='cpu', help='Explicit device; no automatic GPU selection')
    defaults = PathwayConfig()
    for key in ('hidden', 'common', 'local_dim', 'local_common', 'epochs', 'patience', 'seed'):
        parser.add_argument('--'+key.replace('_', '-'), type=int, default=getattr(defaults, key))
    for key in ('interaction_strength', 'match_temperature', 'lr', 'weight_decay', 'rna_noise',
                'feature_noise', 'rna_min_std', 'strength', 'min_weight', 'global_path_within',
                'global_rna_within', 'global_common', 'global_unique', 'patch_within',
                'pathway_path_within', 'pathway_rna_within', 'pathway_common', 'pathway_unique',
                'matched_common', 'matched_unique'):
        parser.add_argument('--'+key.replace('_', '-'), type=float, default=getattr(defaults, key))
    parser.add_argument('--rna-scale-policy', choices=['mask', 'clamp'], default=defaults.rna_scale_policy)
    parser.add_argument('--common-mode', choices=['off', 'uniform', 'stability', 'shuffled'], default=defaults.common_mode)
    return parser


def main():
    args = make_parser().parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    if not torch.isfinite(torch.tensor(args.horizon)) or args.horizon < 0:
        raise ValueError('Prediction horizon must be finite and nonnegative')
    starts = torch.tensor(args.starts, dtype=torch.float32)
    piecewise_exposure(torch.tensor([0.]), starts)
    requested_device = torch.device(args.device)
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable in the selected PyTorch environment')
    fields = PathwayConfig.__dataclass_fields__
    cfg = PathwayConfig(**{name: getattr(args, name) for name in fields})
    required_batch = validate_config(cfg, args.view_source, args.patient_batch_size, args.patch_chunk_size)
    source = 'single_view' if args.view_source == 'feature_noise' else 'cached_uni'
    start = time.perf_counter()
    manifest_sha256 = sha256_file(args.manifest)
    # Validate the public gene-set and user gene-schema files before hashing
    # or opening potentially tens of gigabytes of patient feature tensors.
    gene_count = len(Path(args.genes).read_text(encoding='utf-8-sig').splitlines())
    pathway_mapping = read_pathway_mapping(args.genes, args.pathways, gene_count,
                                           min_genes=args.min_pathway_genes)
    default_gmt = (Path(__file__).parent/'resources'/'reactome_human_roots.gmt').resolve()
    if Path(args.pathways).resolve() == default_gmt:
        meta = json.loads(default_gmt.with_suffix('.json').read_text(encoding='utf-8'))
        if meta['output_sha256'] != pathway_mapping['pathway_sha256']:
            raise ValueError('Frozen Reactome resource hash does not match its recorded source metadata')
        pathway_mapping['resource_metadata'] = meta
    cohort = load_stream_manifest(args.manifest, path_views=source, include_test=args.evaluate_test)
    cohort_hash = stream_cohort_fingerprint(cohort, path_views=source)
    if str(args.device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    fitted = train_pathway_model(cohort['train'], cohort['val'], starts, cfg, pathway_mapping,
                                 patient_batch_size=args.patient_batch_size,
                                 patch_chunk_size=args.patch_chunk_size, view_source=args.view_source,
                                 device=args.device)
    split = 'test' if args.evaluate_test else 'val'
    evaluation = cohort[split]
    outputs = predict_pathway_model(fitted, {k: v for k, v in evaluation.items() if k not in ('time', 'event')})
    with torch.no_grad():
        nll = piecewise_nll(outputs['log_rates'], evaluation['time'], evaluation['event'], starts).item()
        survival = survival_at(outputs['log_rates'], torch.tensor([args.horizon]), starts)[:, 0]
    c_index = harrell_c(evaluation['time'].numpy(), evaluation['event'].numpy(), 1-survival.numpy())
    risk_audit = horizon_risk_audit(outputs['log_rates'], evaluation['time'], evaluation['event'], starts, args.horizon)
    hazard = cumulative_hazard_at(outputs['log_rates'], args.horizon, starts)
    brier = ipcw_brier(cohort['train']['time'].numpy(), cohort['train']['event'].numpy(),
                       evaluation['time'].numpy(), evaluation['event'].numpy(), survival.numpy(), args.horizon)
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = {'architecture': 'PathwayLocalSurvival', 'model_state': fitted['model'].state_dict(),
                  'model_kwargs': fitted['model_kwargs'], 'pathway_mapping': fitted['pathway_mapping'],
                  'rna_mean': fitted['rna_mean'], 'rna_scale': fitted['rna_scale'],
                  'rna_mask': fitted['rna_mask'], 'rna_train_std': fitted['rna_train_std'],
                  'rna_preprocessing': fitted['rna_preprocessing'], 'starts': fitted['starts'],
                  'stream_config': fitted['stream_config'], 'config': asdict(cfg),
                  'effective_loss_weights': fitted['effective_loss_weights'],
                  'manifest_sha256': manifest_sha256, 'cohort_input_sha256': cohort_hash,
                  'evaluation_split': split, 'test_evaluated': bool(args.evaluate_test)}
    torch.save(checkpoint, destination/'checkpoint.pt')
    with (destination/f'{split}_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient_id', 'survival_at_horizon', 'interaction_logit_mean',
                         'matching_normalized_entropy_mean', 'matching_max_weight_mean', 'cumulative_hazard_at_horizon'])
        writer.writerows((identity, float(probability), float(interaction.mean()),
                          float(entropy.mean()), float(max_weight.mean()), float(cumulative))
                         for identity, probability, interaction, entropy, max_weight, cumulative in
                         zip(evaluation['ids'], survival, outputs['interaction_logits'],
                             outputs['matching_normalized_entropy'], outputs['matching_max_weight'], hazard))
    cuda = None
    if str(args.device).startswith('cuda'):
        gpu = torch.device(args.device)
        cuda = {'max_allocated_bytes': torch.cuda.max_memory_allocated(gpu),
                'max_reserved_bytes': torch.cuda.max_memory_reserved(gpu)}
    report = {
        'architecture': 'PathwayLocalSurvival', 'status': 'research_only',
        'prediction_path': 'full pathology and RNA encoders plus RNA-conditioned soft patch/pathway interaction',
        'local_auxiliary_path': ('pathology-only shared pathway queries plus RNA-conditioned soft matched regions; '
                                 'both compared with RNA tokens by pathway identity'),
        'view_source': args.view_source,
        'view_source_description': ('RMS-scaled Gaussian perturbations of encoded patch features; not image-space augmentation'
                                    if source == 'single_view' else 'verified same-patch UNI image-D4 A/B caches'),
        'source_encoder': 'external feature encoder unverified' if source == 'single_view' else 'verified UNI v1 cache contract',
        'manifest_sha256': manifest_sha256, 'cohort_input_sha256': cohort_hash,
        'cohort_fingerprint_scheme': 'single-view-pt-stream-v1' if source == 'single_view' else 'verified-cache-stream-v1',
        'pathway_mapping': fitted['pathway_mapping'], 'rna_preprocessing': fitted['rna_preprocessing'],
        'model_kwargs': fitted['model_kwargs'], 'config': asdict(cfg),
        'effective_loss_weights': effective_weights(cfg), 'minimum_patient_batch': required_batch,
        'stream_config': fitted['stream_config'], 'endpoint': args.endpoint, 'time_unit': args.time_unit,
        'interval_starts': args.starts, 'prediction_horizon': args.horizon,
        'best_epoch': fitted['best_epoch'], 'validation_nll': fitted['validation_nll'],
        'evaluation_split': split, 'evaluation_nll': nll,
        'evaluation_c_index_risk_at_horizon': c_index, 'evaluation_ipcw_brier_at_horizon': brier,
        'evaluation_risk_numerics': risk_audit, 'metric_conventions': METRIC_CONVENTIONS,
        'patients': {name: len(data['ids']) for name, data in cohort.items()},
        'events': {name: int(data['event'].sum()) for name, data in cohort.items()},
        'patches': {name: sum(bag.n_patches for bag in data['bags']) for name, data in cohort.items()},
        'test_evaluated': bool(args.evaluate_test), 'history': fitted['history'],
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'device': args.device, 'wall_seconds': time.perf_counter()-start, 'cuda': cuda},
        'limitations': [
            'Local pathology/pathway associations are learned, weakly supervised, and have no patch-level RNA ground truth.',
            'Zero correlation does not prove independent or biologically identifiable unique factors.',
            'Pathway sets and matched genes may overlap; patient count, not patch or pathway count, controls independent samples.',
            'Input gene names/order must be audited externally; a schema hash cannot prove each RNA file was generated in that order.',
        ],
    }
    with (destination/'report.json').open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f'Pathway survival research output written to {destination}; test_evaluated={args.evaluate_test}')


if __name__ == '__main__':
    main()
