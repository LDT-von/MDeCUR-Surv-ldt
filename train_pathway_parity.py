"""Train the pathway model under the reference protocol so numbers compare.

This runner uses the full dense-token extension by default. The historical
prediction architecture requires --legacy-architecture; disabling auxiliaries
is an explicit ablation. Both use the streaming pathway inputs and the
following reference survival conventions:

* the survival likelihood becomes the reference discrete-time likelihood over
  train-only quantile bins instead of the continuous piecewise likelihood;
* the patch bag is reduced to the reference patch budget with the reference
  sampling rule;
* the risk score becomes the reference "minus the summed survival";
* the checkpoint is selected by the reference validation C-index peak;
* the reported metrics come from the reference metric implementations.

All auxiliary terms are enabled by default. Disabling them is an explicit
survival-only training ablation that retains the selected prediction
architecture. This entry point records its own configuration and report.
"""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from cancer_parity import (_json_float, discrete_nll, discrete_survival, dct_bin_edges,
                           discretize_labels, dct_risk, parity_metrics,
                           parity_report_protocol)
from cancer_parity_bag import BudgetedPatientBag
from cancer_pathway import PathwayLocalSurvival, read_pathway_mapping
from cancer_pathway_stream import (PathwayConfig, effective_weights, minimum_batch_size,
                                   pathway_objective, validate_config, patient_batches)
from cancer_dense_model import (DensePathwayConfig, DensePathwaySurvival, DENSE_LOSS_NAMES,
                                dense_pathway_objective, dense_weights, disable_all_auxiliary,
                                require_full_auxiliary)
from cancer_rna import fit_rna_scaler, transform_rna
from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True,
                        help='Explicit train/val CSV; the split membership is authoritative')
    parser.add_argument('--survival-reference', required=True,
                        help='Double-precision follow-up times and event indicators')
    parser.add_argument('--genes', required=True, help='Ordered gene names, one per line')
    parser.add_argument('--pathways', required=True, help='Explicit GMT with gene symbols')
    parser.add_argument('--out', required=True, help='New output directory; never overwritten')
    parser.add_argument('--num-patches', type=int, required=True,
                        help='Reference patch budget per patient')
    parser.add_argument('--n-bins', type=int, default=4, help='Number of discrete survival bins')
    parser.add_argument('--alpha-surv', type=float, default=.5,
                        help='Censored likelihood weight is 1-alpha; event likelihood retains unit weight')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patient-batch-size', type=int, default=20)
    parser.add_argument('--patch-chunk-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--rna-scale-policy', choices=['mask', 'clamp', 'none'], default='none',
                        help='The reference pipeline feeds the raw expression values')
    parser.add_argument('--disable-auxiliary', action='store_true',
                        help='Explicit ablation only: turn every legacy+dense auxiliary term off')
    parser.add_argument('--legacy-architecture', action='store_true',
                        help='Explicit historical architecture; default is full dense-token model')
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--time-unit', required=True)
    parser.add_argument('--split-name', required=True, help='Name of the split, for the report')
    return parser


def _budget(data, num_patches, mode, seed):
    """Wrap every patient bag with the reference patch budget."""
    return dict(data, bags=[BudgetedPatientBag(bag, num_patches, mode, seed)
                            for bag in data['bags']])


def load_survival_reference(path):
    """Read the double-precision follow-up times and event indicators.

    The reference pipeline fits the survival bins, the censoring distribution
    and the metric time grid on the clinical values at double precision.  The
    model input stays a float32 tensor; only the reference quantities use the
    unrounded values.
    """
    rows = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            identity = row['patient_id'].strip()
            time_value, event = float(row['time']), float(row['event'])
            if (not identity or identity in rows or not np.isfinite(time_value) or time_value < 0
                    or event not in (0., 1.) or row.get('split', 'train').strip() not in ('train', 'val')):
                raise ValueError('Survival reference requires unique valid train/val labels')
            rows[identity] = (time_value, event)
    if not rows:
        raise ValueError('The survival reference file is empty')
    return rows


def _logits(model, bags, rna, chunk_size, view_source='feature_noise', include_views=False,
            rna_views=None, cfg=None, rna_mask=None):
    from cancer_pathway_stream import pathway_stream_forward
    output = pathway_stream_forward(model, bags, rna, chunk_size, view_source,
                                    include_views=include_views, include_patch=False,
                                    rna_views=rna_views, feature_noise=cfg.feature_noise if cfg else 0.,
                                    view_seed=0)
    return output['log_rates']


def main():
    args = make_parser().parse_args()
    started = time.perf_counter()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    if args.num_patches < 1 or args.n_bins < 2:
        raise ValueError('A positive patch budget and at least two survival bins are required')
    if not 0 <= args.alpha_surv <= 1:
        raise ValueError('alpha_surv must lie in [0, 1]')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('The requested CUDA device is unavailable in this environment')

    cohort = load_stream_manifest(args.manifest, path_views='single_view', include_test=False)
    cohort_hash = stream_cohort_fingerprint(cohort, path_views='single_view')
    manifest_sha256 = sha256_file(args.manifest)

    # The bins and the censoring reference use the double-precision clinical
    # values, exactly like the reference pipeline.
    reference = load_survival_reference(args.survival_reference)
    if set(reference) != {identity for split in ('train', 'val') for identity in cohort[split]['ids']}:
        raise ValueError('Survival reference IDs must exactly match train/val manifest')
    for split in ('train', 'val'):
        for index, identity in enumerate(cohort[split]['ids']):
            if identity not in reference:
                raise ValueError(f'{identity} has no survival reference row')
            t, e = reference[identity]
            if e != float(cohort[split]['event'][index]) or not np.isclose(
                    t, float(cohort[split]['time'][index]), rtol=2e-7, atol=2e-6):
                raise ValueError('Survival reference labels differ from manifest')
    train_time = np.array([reference[i][0] for i in cohort['train']['ids']])
    train_event = np.array([reference[i][1] for i in cohort['train']['ids']])
    val_time = np.array([reference[i][0] for i in cohort['val']['ids']])
    val_event = np.array([reference[i][1] for i in cohort['val']['ids']])

    # The bins are fitted on the training rows only.
    edges = dct_bin_edges(train_time, train_event, args.n_bins)
    labels = {split: torch.from_numpy(discretize_labels(values, edges))
              for split, values in (('train', train_time), ('val', val_time))}
    censorship = {split: 1-values for split, values in (('train', train_event),
                                                        ('val', val_event))}

    gene_count = len(Path(args.genes).read_text(encoding='utf-8-sig').splitlines())
    pathway_mapping = read_pathway_mapping(args.genes, args.pathways, gene_count, min_genes=1)

    # The reference patch budget: training draws a random subset, evaluation
    # takes evenly spaced rows.
    train = _budget(cohort['train'], args.num_patches, 'train', args.seed)
    validation = _budget(cohort['val'], args.num_patches, 'eval', args.seed)

    config_class = PathwayConfig if args.legacy_architecture else DensePathwayConfig
    cfg = config_class(epochs=args.epochs, patience=args.epochs+1, lr=args.lr,
                        weight_decay=args.weight_decay, seed=args.seed,
                        rna_scale_policy=args.rna_scale_policy)
    if args.disable_auxiliary:
        from cancer_pathway_stream import LOSS_NAMES
        cfg = config_class(**{**asdict(cfg),
                               **{name: 0. for name in LOSS_NAMES},
                               **({name: 0. for name in DENSE_LOSS_NAMES} if not args.legacy_architecture else {}),
                               'common_mode': 'off'})
    elif not args.legacy_architecture:
        require_full_auxiliary(cfg)
    minimum = validate_config(cfg, 'feature_noise', args.patient_batch_size, args.patch_chunk_size)
    scaler = fit_rna_scaler(train['rna'], cfg.rna_scale_policy, cfg.rna_min_std)
    normalized = transform_rna(train['rna'], scaler)
    validation_normalized = transform_rna(validation['rna'], scaler)

    dimension = train['bags'][0].feature_dim
    model_kwargs = dict(path_dim=dimension, rna_dim=train['rna'].shape[1],
                        pathway_indices=[item['indices'] for item in pathway_mapping['pathways']],
                        hidden=cfg.hidden, common=cfg.common, local_dim=cfg.local_dim,
                        local_common=cfg.local_common, bins=args.n_bins,
                        interaction_strength=cfg.interaction_strength,
                        match_temperature=cfg.match_temperature)
    torch.manual_seed(args.seed)
    model_class = PathwayLocalSurvival if args.legacy_architecture else DensePathwaySurvival
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def survival_loss(logits, batch):
        return discrete_nll(logits, batch['label'].to(device),
                            batch['censorship'].to(device), alpha=args.alpha_surv)

    order_generator = torch.Generator().manual_seed(args.seed)
    history = []
    best, best_cindex, best_epoch = None, -np.inf, 0
    for epoch in range(1, args.epochs+1):
        epoch_start = time.perf_counter()
        model.train()
        # The reference redraws the training patch subset at every epoch.
        for bag in train['bags']:
            bag.resample(args.seed*1000+epoch)
        order = torch.randperm(len(train['ids']), generator=order_generator)
        total, patients, skipped, auxiliary_batches = 0.0, 0, 0, []
        for indices in patient_batches(order, args.patient_batch_size, minimum):
            if len(indices) < 1:
                continue
            optimizer.zero_grad(set_to_none=True)
            batch = {'bags': [train['bags'][i] for i in indices.tolist()],
                     'rna': normalized[indices].to(device),
                     'time': train['time'][indices].to(device),
                     'event': train['event'][indices].to(device),
                     'rna_mask': scaler['rna_mask'],
                     'label': labels['train'][indices],
                     'censorship': torch.from_numpy(censorship['train'][indices]).float()}
            if args.legacy_architecture:
                loss, stats = pathway_objective(model, batch, None, cfg, args.patch_chunk_size,
                                                'feature_noise', args.seed*1000+epoch,
                                                survival_loss=survival_loss)
            else:
                loss, stats = dense_pathway_objective(model, batch, cfg, args.patch_chunk_size,
                                                      args.seed*1000+epoch, epoch,
                                                      survival_loss=survival_loss)
                if not args.disable_auxiliary and stats['common_skipped']:
                    raise FloatingPointError('Full common auxiliary degenerated; refuse silent loss disabling')
                auxiliary_batches.append(stats)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite parity parameter gradient')
            optimizer.step()
            total += float(stats['train_nll'])*len(indices)
            patients += len(indices)
            skipped += int(bool(stats['common_skipped']))
        scheduler.step()
        validation_metrics = _evaluate(model, validation, validation_normalized, labels['val'],
                                       censorship['val'], edges, args.patch_chunk_size,
                                       args.n_bins, train_time, train_event, val_time,
                                       val_event, args.alpha_surv)
        record = {'epoch': epoch, 'train_nll': total/max(1, patients),
                  'auxiliary_batches': auxiliary_batches,
                  'common_skipped_batches': skipped,
                  'seconds': time.perf_counter()-epoch_start, **validation_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
        if validation_metrics['c_index'] is not None and validation_metrics['c_index'] > best_cindex:
            best_cindex = validation_metrics['c_index']
            best_epoch = epoch
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best is None:
        raise RuntimeError('No validation C-index could be computed for any epoch')
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    last_survival = discrete_survival(_predict(model, validation, validation_normalized,
                                               args.patch_chunk_size)).numpy()
    model.load_state_dict(best, strict=True)
    final = _evaluate(model, validation, validation_normalized, labels['val'],
                      censorship['val'], edges, args.patch_chunk_size, args.n_bins,
                      train_time, train_event, val_time, val_event, args.alpha_surv)

    destination.mkdir(parents=True, exist_ok=False)
    architecture = 'PathwayLocalSurvival' if args.legacy_architecture else 'DensePathwaySurvival'
    checkpoint = {'architecture': architecture, 'model_state': model.state_dict(),
                'survival_reference_sha256': sha256_file(args.survival_reference),
                'model_kwargs': model_kwargs, 'pathway_mapping': pathway_mapping,
                'rna_mean': scaler['rna_mean'], 'rna_scale': scaler['rna_scale'],
                'rna_mask': scaler['rna_mask'], 'rna_train_std': scaler['rna_train_std'],
                'rna_preprocessing': scaler['rna_preprocessing'],
                'bin_edges': [_json_float(x) for x in edges], 'config': asdict(cfg),
                'manifest_sha256': manifest_sha256, 'cohort_input_sha256': cohort_hash,
                'epoch': best_epoch, 'selection': 'best'}
    torch.save(checkpoint, destination/'checkpoint.pt')  # Historical alias remains the selected checkpoint.
    torch.save(checkpoint, destination/'checkpoint_best.pt')
    torch.save({**checkpoint, 'model_state': last_state, 'epoch': args.epochs, 'selection': 'last'},
               destination/'checkpoint_last.pt')
    with (destination/'val_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient_id', 'risk', 'bin_label']+[f'survival_bin_{k}'
                                                             for k in range(args.n_bins)])
        logits = _predict(model, validation, validation_normalized, args.patch_chunk_size)
        survival = discrete_survival(logits).numpy()
        writer.writerows([identity, float(risk), int(label)]+[float(x) for x in row]
                         for identity, risk, label, row in zip(validation['ids'],
                                                              dct_risk(survival),
                                                              labels['val'].numpy(), survival))
    (destination/'val_predictions_best.csv').write_bytes((destination/'val_predictions.csv').read_bytes())
    with (destination/'val_predictions_last.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient_id', 'risk', 'bin_label']+[f'survival_bin_{k}' for k in range(args.n_bins)])
        writer.writerows([identity, float(risk), int(label), *map(float, row)]
                         for identity, risk, label, row in zip(validation['ids'], dct_risk(last_survival),
                                                              labels['val'].numpy(), last_survival))
    report = {
        'status': 'research_only',
        'protocol': parity_report_protocol(edges, args.num_patches, args.endpoint,
                                            args.split_name, args.n_bins),
        'architecture': architecture,
        'conventions_changed_against_the_main_pipeline': [
            'discrete-time likelihood over train-only quantile bins instead of the continuous likelihood',
            'a fixed patch budget with the reference sampling rule instead of the complete bag',
            'checkpoint selection by the validation C-index peak instead of the validation NLL minimum',
            'the reference risk and metric conventions',
        ],
        'endpoint': args.endpoint, 'time_unit': args.time_unit,
        'manifest_sha256': manifest_sha256, 'cohort_input_sha256': cohort_hash,
        'survival_reference_sha256': sha256_file(args.survival_reference),
        'pathway_mapping': pathway_mapping, 'rna_preprocessing': scaler['rna_preprocessing'],
        'model_kwargs': model_kwargs, 'config': asdict(cfg),
        'effective_loss_weights': {**effective_weights(cfg),
                                  **(dense_weights(cfg, cfg.dense_warmup_epochs) if not args.legacy_architecture else {})},
        'minimum_patient_batch': minimum,
        'alpha_surv': args.alpha_surv, 'n_bins': args.n_bins, 'num_patches': args.num_patches,
        'bin_edges': [_json_float(x) for x in edges],
        'patients': {'train': len(train['ids']), 'val': len(validation['ids'])},
        'events': {'train': int(train_event.sum()), 'val': int(val_event.sum())},
        'patches_available': {'train': int(sum(b._source.n_patches for b in train['bags'])),
                               'val': int(sum(b._source.n_patches for b in validation['bags']))},
        'patches_padded': {'train': int(sum(b.padding for b in train['bags'])),
                           'val': int(sum(b.padding for b in validation['bags']))},
        'best_epoch': best_epoch, 'best_validation_c_index': best_cindex,
        'last_epoch': args.epochs, 'last_validation': {key: history[-1][key] for key in final},
        'selected_checkpoint_validation': final,
        'final_validation_semantics': 'Historical alias of selected best checkpoint, not last epoch',
        'final_validation': final, 'history': history,
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'device': args.device, 'wall_seconds': time.perf_counter()-started},
    }
    with (destination/'report.json').open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f'Parity output written to {destination}; best_epoch={best_epoch}')


def _predict(model, cohort, normalized_rna, chunk_size):
    model.eval()
    logits = []
    with torch.no_grad():
        for index, bag in enumerate(cohort['bags']):
            output = _logits(model, [bag], normalized_rna[index:index+1], chunk_size)
            logits.append(output.detach().cpu())
    return torch.cat(logits)


def _evaluate(model, cohort, normalized_rna, labels, censorship, edges, chunk_size, n_bins,
              train_time, train_event, val_time, val_event, alpha=.5):
    """Validation metrics with the censoring reference fitted on the training rows."""
    model.eval()
    logits = _predict(model, cohort, normalized_rna, chunk_size)
    survival = discrete_survival(logits).numpy()
    metrics = parity_metrics(train_time, train_event, val_time, val_event, survival, edges)
    from cancer_parity_audit import audit_parity_metrics
    metrics['finite_edge_audit'] = audit_parity_metrics(train_time, train_event, val_time, val_event, survival, edges)
    metrics['validation_nll'] = float(discrete_nll(
        logits, torch.from_numpy(np.asarray(labels)), torch.from_numpy(np.asarray(censorship)),
        alpha=alpha).item())
    return metrics


if __name__ == '__main__':
    main()
