"""Train the pathway model under the reference protocol so numbers compare.

This runner exists for a like-for-like comparison with the reference pipeline.
It reuses the audited pathway model, the audited streaming forward pass and the
audited auxiliary losses, and changes only the externally defined conventions:

* the survival likelihood becomes the reference discrete-time likelihood over
  train-only quantile bins instead of the continuous piecewise likelihood;
* the patch bag is reduced to the reference patch budget with the reference
  sampling rule;
* the risk score becomes the reference "minus the summed survival";
* the checkpoint is selected by the reference validation C-index peak;
* the reported metrics come from the reference metric implementations.

The auxiliary pathway terms remain part of this repository's method and are
kept enabled by default; disabling them turns the runner into a plain
pathology-plus-RNA model.  Nothing here weakens the existing protocol: this is
a separate entry point with its own configuration and its own report.
"""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
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
                                   patient_batches, pathway_objective, validate_config)
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
                        help='Weight of the uncensored term in the discrete likelihood')
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
                        help='Turn every auxiliary correlation term off')
    parser.add_argument('--shuffle-train-labels', action='store_true',
                        help='Permute the training survival labels; the score must collapse to chance')
    parser.add_argument('--freeze-modality', choices=['none', 'pathology', 'rna'], default='none',
                        help='Zero one modality at training time; the remaining one must carry the score')
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--time-unit', required=True)
    parser.add_argument('--split-name', required=True, help='Name of the split, for the report')
    parser.add_argument('--auxiliary-weight', action='append', default=[], metavar='NAME=VALUE',
                        help='Override one auxiliary loss weight; repeatable.  Every override '
                             'switches the auxiliary path on, so the run no longer matches the '
                             'auxiliary-disabled control unless the same override is given there.')
    parser.add_argument('--common-mode', choices=['off', 'uniform', 'stability', 'shuffled'],
                        default=None,
                        help='How the global common subspace is estimated; the default keeps the '
                             'configured value')
    return parser


def _finite_or_none(value):
    """Keep the JSON report free of NaN and infinity."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _parse_auxiliary_overrides(values):
    """Turn ``NAME=VALUE`` strings into a validated weight mapping."""
    from cancer_pathway_stream import LOSS_NAMES
    overrides = {}
    for item in values:
        if '=' not in item:
            raise ValueError(f'Expected NAME=VALUE for --auxiliary-weight, got {item!r}')
        name, _, raw = item.partition('=')
        name = name.strip()
        if name not in LOSS_NAMES:
            raise ValueError(f'Unknown auxiliary loss term {name!r}; expected one of {LOSS_NAMES}')
        try:
            weight = float(raw)
        except ValueError as error:
            raise ValueError(f'Auxiliary weight for {name!r} is not a number') from error
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f'Auxiliary weight for {name!r} must be finite and nonnegative')
        overrides[name] = weight
    return overrides


def _budget(data, num_patches, mode, seed):
    """Wrap every patient bag with the reference patch budget."""
    return dict(data, bags=[BudgetedPatientBag(bag, num_patches, mode, seed)
                            for bag in data['bags']])


class _ZeroBag:
    """A patient bag whose patch rows are all zeros.

    The row count is preserved so that the pooling still sees the same number
    of rows; only the feature content is removed.  This isolates how much of
    the score the pathology branch carries.
    """

    def __init__(self, source, num_patches):
        self.patient_id = source.patient_id
        self.feature_dim = source.feature_dim
        self.view_source = getattr(source, 'view_source', 'single_view')
        self.n_patches = num_patches
        self.padding = num_patches

    def resample(self, seed=0):
        """A frozen bag has no rows to redraw."""
        return self

    def blocks(self, chunk_size):
        for start in range(0, self.n_patches, chunk_size):
            yield ('<parity-budget>', start, min(start+chunk_size, self.n_patches))

    def read_block(self, token, names):
        path, start, stop = token
        if not 0 <= start < stop <= self.n_patches:
            raise ValueError('Invalid block row range')
        return (torch.zeros((stop-start, self.feature_dim), dtype=torch.float32),)

    def assert_unchanged(self):
        return None

    def close(self):
        return None

    @property
    def provenance(self):
        return []


def _freeze_bags(bags, num_patches):
    """Replace every patch row with zeros."""
    return [_ZeroBag(bag, num_patches) for bag in bags]


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
            rows[row['patient_id'].strip()] = (float(row['time']), float(row['event']))
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
    for split in ('train', 'val'):
        for identity in cohort[split]['ids']:
            if identity not in reference:
                raise ValueError(f'{identity} has no survival reference row')
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

    # The null control permutes the training labels only.  The validation
    # labels stay untouched, so a model that still reaches a high validation
    # C-index would be reading something other than the training outcome.
    if args.shuffle_train_labels:
        generator = np.random.default_rng(args.seed)
        permutation = generator.permutation(len(train_time))
        train_time = train_time[permutation]
        train_event = train_event[permutation]
        labels['train'] = torch.from_numpy(discretize_labels(train_time, edges))
        censorship['train'] = 1-train_event

    gene_count = len(Path(args.genes).read_text(encoding='utf-8-sig').splitlines())
    pathway_mapping = read_pathway_mapping(args.genes, args.pathways, gene_count, min_genes=1)

    # The reference patch budget: training draws a random subset, evaluation
    # takes evenly spaced rows.
    train = _budget(cohort['train'], args.num_patches, 'train', args.seed)
    validation = _budget(cohort['val'], args.num_patches, 'eval', args.seed)

    defaults = PathwayConfig()
    cfg = PathwayConfig(epochs=args.epochs, patience=args.epochs+1, lr=args.lr,
                        weight_decay=args.weight_decay, seed=args.seed,
                        rna_scale_policy=args.rna_scale_policy)
    overrides = _parse_auxiliary_overrides(args.auxiliary_weight)
    if args.disable_auxiliary and overrides:
        raise ValueError('--disable-auxiliary and --auxiliary-weight contradict each other')
    if args.disable_auxiliary:
        from cancer_pathway_stream import LOSS_NAMES
        cfg = PathwayConfig(**{**asdict(cfg),
                               **{name: 0. for name in LOSS_NAMES},
                               'common_mode': 'off'})
    elif overrides:
        # Only the named terms change; every other term keeps the configured
        # weight, so an explicit override of every term reproduces the defaults.
        cfg = PathwayConfig(**{**asdict(cfg), **overrides})
    if args.common_mode is not None:
        cfg = PathwayConfig(**{**asdict(cfg), 'common_mode': args.common_mode})
    minimum = validate_config(cfg, 'feature_noise', args.patient_batch_size, args.patch_chunk_size)
    scaler = fit_rna_scaler(train['rna'], cfg.rna_scale_policy, cfg.rna_min_std)
    normalized = transform_rna(train['rna'], scaler)
    validation_normalized = transform_rna(validation['rna'], scaler)

    # The modality control removes one input from both the training batches and
    # the evaluation, so the remaining modality is the only source of signal.
    if args.freeze_modality == 'rna':
        normalized = torch.zeros_like(normalized)
        validation_normalized = torch.zeros_like(validation_normalized)
    if args.freeze_modality == 'pathology':
        train['bags'] = _freeze_bags(train['bags'], args.num_patches)
        validation['bags'] = _freeze_bags(validation['bags'], args.num_patches)

    dimension = train['bags'][0].feature_dim
    model_kwargs = dict(path_dim=dimension, rna_dim=train['rna'].shape[1],
                        pathway_indices=[item['indices'] for item in pathway_mapping['pathways']],
                        hidden=cfg.hidden, common=cfg.common, local_dim=cfg.local_dim,
                        local_common=cfg.local_common, bins=args.n_bins,
                        interaction_strength=cfg.interaction_strength,
                        match_temperature=cfg.match_temperature)
    torch.manual_seed(args.seed)
    model = PathwayLocalSurvival(**model_kwargs).to(device)
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
        total, patients, skipped = 0.0, 0, 0
        term_sums = {}
        term_counts = {}
        diagnostic_sums = {}
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
            loss, stats = pathway_objective(model, batch, None, cfg, args.patch_chunk_size,
                                            'feature_noise', args.seed*1000+epoch,
                                            survival_loss=survival_loss)
            loss.backward()
            optimizer.step()
            total += float(stats['train_nll'])*len(indices)
            patients += len(indices)
            skipped += int(bool(stats['common_skipped']))
            for name, value in stats['terms'].items():
                if value is None:
                    continue
                term_sums[name] = term_sums.get(name, 0.0) + float(value)
                term_counts[name] = term_counts.get(name, 0) + 1
            for name, diagnostic in stats.get('pathway_diagnostics', {}).items():
                if not isinstance(diagnostic, dict):
                    continue
                for key, value in diagnostic.items():
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        continue
                    bucket = diagnostic_sums.setdefault(name, {})
                    bucket[key] = bucket.get(key, 0.0) + float(value)
        scheduler.step()
        validation_metrics = _evaluate(model, validation, validation_normalized, labels['val'],
                                       censorship['val'], edges, args.patch_chunk_size,
                                       args.n_bins, train_time, train_event, val_time,
                                       val_event, args.alpha_surv)
        record = {'epoch': epoch, 'train_nll': total/max(1, patients),
                  'common_skipped_batches': skipped,
                  'seconds': time.perf_counter()-epoch_start, **validation_metrics}
        if term_sums:
            record['auxiliary_terms'] = {
                name: {'mean': term_sums[name]/term_counts[name], 'batches': term_counts[name]}
                for name in sorted(term_sums)}
        if diagnostic_sums:
            record['pathway_diagnostics'] = {
                name: {key: value/max(1, term_counts.get(name, 1)) for key, value in sorted(bucket.items())}
                for name, bucket in sorted(diagnostic_sums.items())}
        record['matching_normalized_entropy_mean'] = _finite_or_none(
            stats.get('matching_normalized_entropy_mean'))
        record['interaction_abs_mean'] = _finite_or_none(stats.get('interaction_abs_mean'))
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
        if validation_metrics['c_index'] is not None and validation_metrics['c_index'] > best_cindex:
            best_cindex = validation_metrics['c_index']
            best_epoch = epoch
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best is None:
        raise RuntimeError('No validation C-index could be computed for any epoch')
    model.load_state_dict(best, strict=True)
    final = _evaluate(model, validation, validation_normalized, labels['val'],
                      censorship['val'], edges, args.patch_chunk_size, args.n_bins,
                      train_time, train_event, val_time, val_event, args.alpha_surv)

    destination.mkdir(parents=True, exist_ok=False)
    torch.save({'architecture': 'PathwayLocalSurvival', 'model_state': model.state_dict(),
                'model_kwargs': model_kwargs, 'pathway_mapping': pathway_mapping,
                'rna_mean': scaler['rna_mean'], 'rna_scale': scaler['rna_scale'],
                'rna_mask': scaler['rna_mask'], 'rna_train_std': scaler['rna_train_std'],
                'rna_preprocessing': scaler['rna_preprocessing'],
                'bin_edges': [_json_float(x) for x in edges], 'config': asdict(cfg),
                'manifest_sha256': manifest_sha256, 'cohort_input_sha256': cohort_hash},
               destination/'checkpoint.pt')
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
    report = {
        'status': 'research_only',
        'controls': {
            'shuffled_train_labels': bool(args.shuffle_train_labels),
            'frozen_modality': args.freeze_modality,
            'auxiliary_weight_overrides': overrides,
            'common_mode': cfg.common_mode,
            'note': ('a shuffled-label run is a null control: the validation score must '
                     'collapse to chance.  A frozen-modality run removes one input '
                     'from both the training and the evaluation.  An auxiliary-weight '
                     'override switches the auxiliary path on; the term means and the '
                     'pathway diagnostics are recorded per epoch.'),
        },
        'protocol': parity_report_protocol(edges, args.num_patches, args.endpoint,
                                            args.split_name, args.n_bins),
        'architecture': 'PathwayLocalSurvival',
        'conventions_changed_against_the_main_pipeline': [
            'discrete-time likelihood over train-only quantile bins instead of the continuous likelihood',
            'a fixed patch budget with the reference sampling rule instead of the complete bag',
            'checkpoint selection by the validation C-index peak instead of the validation NLL minimum',
            'the reference risk and metric conventions',
        ],
        'endpoint': args.endpoint, 'time_unit': args.time_unit,
        'manifest_sha256': manifest_sha256, 'cohort_input_sha256': cohort_hash,
        'pathway_mapping': pathway_mapping, 'rna_preprocessing': scaler['rna_preprocessing'],
        'model_kwargs': model_kwargs, 'config': asdict(cfg),
        'effective_loss_weights': effective_weights(cfg), 'minimum_patient_batch': minimum,
        'alpha_surv': args.alpha_surv, 'n_bins': args.n_bins, 'num_patches': args.num_patches,
        'bin_edges': [_json_float(x) for x in edges],
        'patients': {'train': len(train['ids']), 'val': len(validation['ids'])},
        'events': {'train': int(train_event.sum()), 'val': int(val_event.sum())},
        'patches_available': {'train': int(sum(b._source.n_patches for b in train['bags'])),
                               'val': int(sum(b._source.n_patches for b in validation['bags']))},
        'patches_padded': {'train': int(sum(b.padding for b in train['bags'])),
                           'val': int(sum(b.padding for b in validation['bags']))},
        'best_epoch': best_epoch, 'best_validation_c_index': best_cindex,
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
    metrics['validation_nll'] = float(discrete_nll(
        logits, torch.from_numpy(np.asarray(labels)), torch.from_numpy(np.asarray(censorship)),
        alpha=alpha).item())
    return metrics


if __name__ == '__main__':
    main()
