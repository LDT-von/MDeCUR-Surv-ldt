"""Five-fold BLCA comparison runner: plan, run and summary.

The four multimodal adapters (PIBD, CAMR, DeReF and DIMAF) share this
driver's data preparation, binning and metric calls. Archived DCT numbers
remain a separately labelled historical reference, never a re-run. The
retired RNA-only and WSI-only baselines are no longer runnable arms.

The driver has three commands:

``plan``
    Freeze the inputs, the upstream source hashes, the code hashes and the
    protocol constants of one fold into an immutable JSON plan.

``run``
    Rebuild the plan from the inputs, compare it byte-for-byte with the stored
    plan, train one arm, score it with ``parity_metrics`` and write a report.

``summary``
    Aggregate the per-fold reports into mean +/- std of the C-index, IBS and
    iAUC, and refuse incomplete, duplicated or mismatched runs.

Leakage boundary
----------------
* Bin edges, RNA standardization and DIMAF prototypes are
  fitted on the training rows of the current fold only.
* The outer test partition is rejected before any file is opened.
* The manifest is authoritative and is never rewritten.
"""
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

import comparison_sources as sources
from cancer_parity import (discrete_nll, discrete_survival, dct_bin_edges, dct_risk,
                           parity_metrics, DISCRETE_NLL_VERSION)
from cancer_parity_audit import audit_parity_metrics
from cancer_parity_bag import BudgetedPatientBag
from cancer_pathway import read_pathway_mapping
from comparison_protocol import (ALPHA_SURV, ENDPOINT, FOLDS, N_BINS, NUM_PATCHES,
                                 PROTOCOL, SHARED_SETTINGS, TIME_UNIT,
                                 close_bags, evaluate_predictions, load_fold,
                                 logits_to_survival, sha256_file)


PROTOCOL_VERSION = 'onefold_comparison_v1_single_fold'
SOURCE_MODULES = ('comparison_5fold', 'comparison_sources', 'comparison_protocol', 'comparison_camr',
                  'comparison_pibd', 'comparison_deref', 'comparison_dimaf',
                  'decur_losses', 'cancer_parity', 'cancer_parity_audit',
                  'cancer_parity_bag', 'cancer_pathway', 'cancer_rna',
                  'cancer_stream_data', 'cancer_single_view', 'cancer_metrics')

MULTIMODAL_MODELS = ('pibd_2m', 'camr_2m', 'deref_2m', 'dimaf_2m')
MODELS = MULTIMODAL_MODELS
DEFAULT_MODEL = 'mdecur_dense'

SETTINGS = dict(
    folds=list(FOLDS), epochs=30, seed=3, patient_batch_size=20,
    lr=5e-4, weight_decay=5e-4, scheduler='CosineAnnealingLR(T_max=30)',
    rna_scale_policy='none', num_patches=NUM_PATCHES, n_bins=N_BINS,
    alpha_surv=ALPHA_SURV, endpoint=ENDPOINT, time_unit=TIME_UNIT,
    selection='maximum validation Harrell C, earliest exact tie',
    patch_resample='train-only random rows without replacement, sorted',
)

# The DIMAF prototypes and the omic group definitions are fitted from the
# training rows of the current fold; these are the only data-dependent
# inputs besides the bin edges and the RNA scaler.
PATHWAY_SOURCES = {
    'pibd': 'protocol GMT (combine_signatures) mapped by exact gene symbol',
    'deref': 'upstream DeReF csv/signatures.csv six functional families',
    'dimaf': 'protocol GMT (combine_signatures) mapped by exact gene symbol',
}


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


def source_hashes():
    return {name: digest(importlib.import_module(name).__file__) for name in SOURCE_MODULES}


def runtime_versions():
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__,
            'sksurv': importlib.import_module('sksurv').__version__}


# --------------------------------------------------------------------- pathway

def load_pathway_groups(genes_path, gmt_path, rna_dim, min_genes=5):
    """Pathway gene groups for PIBD and DIMAF, fitted from names only."""
    mapping = read_pathway_mapping(genes_path, gmt_path, rna_dim, min_genes)
    return mapping


def load_deref_groups(path, gene_names, min_genes=5):
    """The six upstream DeReF functional families mapped by exact gene symbol.

    The upstream ``csv/signatures.csv`` holds one column per functional
    family.  Only gene names and the ordered RNA schema are used, never
    expression values or labels.
    """
    import pandas as pd
    signatures = pd.read_csv(path)
    index = {gene: i for i, gene in enumerate(gene_names)}
    groups, skipped = [], []
    for column in signatures.columns:
        members = [g.strip() for g in signatures[column].dropna().unique() if str(g).strip()]
        matched = sorted(index[g] for g in members if g in index)
        if len(matched) < min_genes:
            skipped.append({'family': column, 'source_genes': len(members),
                            'matched_genes': len(matched)})
        else:
            groups.append({'family': column, 'indices': matched,
                           'source_genes': len(members), 'matched_genes': len(matched)})
    if len(groups) != 6:
        raise ValueError(f'DeReF needs exactly 6 usable functional families, got {len(groups)}')
    return {'groups': groups, 'skipped': skipped, 'signatures_sha256': sha256_file(path)}


def load_gene_names(genes_path, rna_dim):
    genes = Path(genes_path).read_text(encoding='utf-8-sig').splitlines()
    genes = [g.strip() for g in genes]
    if len(genes) != rna_dim or len(set(genes)) != len(genes) or any(not g for g in genes):
        raise ValueError('Ordered gene names must match the RNA dimension exactly')
    return genes


# ------------------------------------------------------------------ the plan

def make_plan(fold, manifest, survival_reference, genes_path, gmt_path,
              deref_signatures, device='cuda'):
    """Freeze every input, code hash and protocol constant of one fold."""
    if fold not in FOLDS:
        raise ValueError(f'Unknown fold {fold}')
    cohort = load_fold(manifest, survival_reference, fold)
    try:
        gene_names = load_gene_names(genes_path, cohort.train_rna.shape[1])
        pathway = load_pathway_groups(genes_path, gmt_path, cohort.train_rna.shape[1])
        deref = load_deref_groups(deref_signatures, gene_names)
        upstream = sources.verify_upstream()
        plan = {
            'protocol': PROTOCOL,
            'protocol_version': PROTOCOL_VERSION,
            'settings': SETTINGS,
            'shared_settings': SHARED_SETTINGS,
            'fold': fold,
            'models': list(MODELS),
            'available_models': list(MODELS),
            'unavailable_models': {
                'murrenet': 'Verified author implementation not available; no placeholder model.',
                'mohd': 'Verified author implementation and hierarchical input contract not available.'},
            'inputs': cohort.metadata(),
            'pathway_mapping': {
                'gene_sha256': pathway['gene_sha256'],
                'pathway_sha256': pathway['pathway_sha256'],
                'pathways': [{k: v for k, v in item.items() if k != 'indices'}
                             | {'indices_sha256': json_hash(item['indices'])}
                             for item in pathway['pathways']],
                'pathway_count': len(pathway['pathways']),
                'source': PATHWAY_SOURCES['pibd'],
            },
            'deref_groups': {
                'signatures_sha256': deref['signatures_sha256'],
                'groups': [{k: v for k, v in item.items() if k != 'indices'}
                           | {'indices_sha256': json_hash(item['indices'])}
                           for item in deref['groups']],
                'skipped': deref['skipped'],
                'source': PATHWAY_SOURCES['deref'],
            },
            'upstream_sources': upstream,
            'upstream_metadata': {name: sources.upstream_metadata(name)
                                  for name in sources.UPSTREAM},
            'source_sha256': source_hashes(),
            'versions': runtime_versions(),
            'device_requested': device,
            'files': {
                'manifest': str(Path(manifest).resolve()),
                'survival_reference': str(Path(survival_reference).resolve()),
                'genes': str(Path(genes_path).resolve()),
                'gmt': str(Path(gmt_path).resolve()),
                'deref_signatures': str(Path(deref_signatures).resolve()),
            },
            'hash_seed_state': {
                'python_hash_seed_env': os.environ.get('PYTHONHASHSEED'),
                'hash_randomization_active': bool(sys.flags.hash_randomization),
                'sampler_depends_on_python_hash': False},
            'fit_boundary': {
                'bin_edges': 'train uncensored times only',
                'rna_scaler': 'train rows only',
                'dimaf_prototypes': 'train-fold patch features only',
                'validation_rows': 'scored only',
                'outer_test': 'never opened',
            },
        }
        return plan
    finally:
        for bag in cohort.train_bags + cohort.val_bags:
            if hasattr(bag, 'close'):
                bag.close()


# ------------------------------------------------------------------- training

def build_model(model_id, plan, cohort, gene_names, pathway, deref, device):
    """Instantiate one arm with the frozen plan dimensions."""
    path_dim = cohort.train_bags[0].feature_dim
    rna_dim = cohort.train_rna.shape[1]
    if model_id == 'pibd_2m':
        from comparison_pibd import PIBD2M
        model = PIBD2M(path_dim=path_dim, rna_dim=rna_dim,
                       omic_indices=[item['indices'] for item in pathway['pathways']],
                       bins=N_BINS, projection_dim=128, sample_num=16,
                       seed=SETTINGS['seed'], num_patches=NUM_PATCHES)
        return model.to(device), model.adaptation_metadata()
    if model_id == 'deref_2m':
        from comparison_deref import DeReF2M
        model = DeReF2M(path_dim=path_dim, rna_dim=rna_dim,
                        omic_indices=[item['indices'] for item in deref['groups']],
                        bins=N_BINS, seed=SETTINGS['seed'])
        return model.to(device), model.adaptation_metadata()
    if model_id == 'dimaf_2m':
        from comparison_dimaf import (DIMAF2M, DIMAF_PROTOTYPE_PATCHES_PER_PROTO,
                                     fit_prototypes, sample_prototype_rows)
        model = DIMAF2M(path_dim=path_dim, rna_dim=rna_dim,
                        omic_indices=[item['indices'] for item in pathway['pathways']],
                        bins=N_BINS, n_proto=16, seed=SETTINGS['seed'])
        # The upstream prototype step samples a bounded number of patch rows
        # from the training slides and clusters those.  The rows are read from
        # the bags in bounded blocks, because the complete training bags hold
        # far more rows than this host can materialise at once.
        sample = sample_prototype_rows(
            cohort.train_bags, n_proto=16,
            n_patches_per_proto=DIMAF_PROTOTYPE_PATCHES_PER_PROTO,
            feature_dim=path_dim, seed=SETTINGS['seed'])
        prototypes = fit_prototypes(sample, n_proto=16, n_iter=50,
                                     feature_dim=path_dim, seed=SETTINGS['seed'])
        del sample
        model.attach_prototypes(prototypes)
        return model.to(device), model.adaptation_metadata()
    if model_id == 'camr_2m':
        from comparison_camr import CAMR2M
        model = CAMR2M(path_dim, rna_dim, bins=N_BINS)
        return model.to(device), model.adaptation_metadata()
    raise ValueError(f'Unknown model {model_id}')


def materialize_bag(bag, num_patches=NUM_PATCHES, chunk_size=NUM_PATCHES):
    """Read the complete fixed patch budget, including explicit zero padding.

    The bag must already be budget-aware (``BudgetedPatientBag``), so the row
    count is exactly the protocol budget for every patient.
    """
    arrays = [bag.read_block(block, ('features_reference',))[0]
              for block in bag.blocks(chunk_size)]
    patches = torch.cat(arrays) if arrays else torch.zeros((0, bag.feature_dim))
    if patches.shape[0] != num_patches:
        raise ValueError(f'Expected a {num_patches}-row patch budget, got {patches.shape[0]}')
    return patches


def _batch_patches(bags, indices):
    return [materialize_bag(bags[i]) for i in indices]


def forward_arm(model_id, model, bags, rna, indices, device, labels=None, censorship=None):
    """One forward pass in the uniform arm interface.

    ``rna`` already contains exactly the rows of ``indices``; ``bags`` is the
    full bag list and ``indices`` selects the matching patients.
    """
    patches = _batch_patches(bags, indices)
    if model_id == 'pibd_2m':
        return model(patches, rna, labels, censorship)
    if model_id in ('camr_2m', 'deref_2m', 'dimaf_2m'):
        return model(patches, rna)
    raise ValueError(f'Unknown model {model_id}')


def evaluate_arm(model_id, model, cohort, device, val_bags):
    """Score the validation partition with the shared metric call."""
    if hasattr(model, 'eval'):
        model.eval()
    logits = []
    with torch.no_grad():
        for index in range(len(val_bags)):
            output = forward_arm(model_id, model, val_bags,
                                 cohort.val_rna[index:index+1], [index], device)
            logits.append(output['logits'].detach().cpu())
    logits = torch.cat(logits)
    survival = logits_to_survival(logits)
    metrics = evaluate_predictions(cohort, survival)
    metrics['finite_edge_audit'] = audit_parity_metrics(
        cohort.train_time, cohort.train_event, cohort.val_time, cohort.val_event,
        survival, cohort.bin_edges)
    return metrics, survival, logits


def run_model(plan_path, plan_sha256, model_id, out, device='cuda', smoke=False,
              epochs=None):
    """Train one arm of one fold and write its report."""
    if model_id not in MODELS:
        raise ValueError('Unknown model; no implicit substitute')
    destination = Path(out)
    if destination.exists():
        raise FileExistsError(destination)
    if digest(plan_path) != plan_sha256:
        raise ValueError('Plan SHA mismatch')
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    rebuilt = make_plan(plan['fold'], plan['files']['manifest'],
                        plan['files']['survival_reference'], plan['files']['genes'],
                        plan['files']['gmt'], plan['files']['deref_signatures'],
                        plan.get('device_requested', 'cuda'))
    if rebuilt != plan:
        raise ValueError('Plan no longer matches the inputs, code, protocol or runtime')

    cohort = load_fold(plan['files']['manifest'], plan['files']['survival_reference'],
                       plan['fold'])
    started = time.perf_counter()
    try:
        destination.mkdir(parents=True, exist_ok=False)
        torch_device = torch.device(device)
        if torch_device.type == 'cuda':
            torch.zeros(1, device=torch_device)
            torch.cuda.synchronize(torch_device)
        torch.manual_seed(SETTINGS['seed'])
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        gene_names = load_gene_names(plan['files']['genes'], cohort.train_rna.shape[1])
        pathway = load_pathway_groups(plan['files']['genes'], plan['files']['gmt'],
                                      cohort.train_rna.shape[1])
        deref = load_deref_groups(plan['files']['deref_signatures'], gene_names)
        model, metadata = build_model(model_id, plan, cohort, gene_names,
                                      pathway, deref, torch_device)

        count_epochs = 1 if smoke else int(epochs or SETTINGS['epochs'])
        history, best_metrics, best_survival, best_epoch = [], None, None, None
        labels = torch.from_numpy(cohort.train_labels)
        events = torch.from_numpy(cohort.train_event)
        train_bags = [BudgetedPatientBag(b, NUM_PATCHES, 'train', SETTINGS['seed'])
                      for b in cohort.train_bags]
        val_bags = [BudgetedPatientBag(b, NUM_PATCHES, 'eval', SETTINGS['seed'])
                    for b in cohort.val_bags]

        main_parameters = (list(model.main_parameters())
                           if hasattr(model, 'main_parameters') else list(model.parameters()))
        discriminator_parameters = (list(model.discriminator_parameters())
                                    if hasattr(model, 'discriminator_parameters') else [])
        parameters = main_parameters+discriminator_parameters
        optimizer = torch.optim.AdamW(main_parameters, lr=SETTINGS['lr'],
                                      weight_decay=SETTINGS['weight_decay'])
        discriminator_optimizer = (torch.optim.AdamW(discriminator_parameters,
                                                      lr=SETTINGS['lr'],
                                                      weight_decay=SETTINGS['weight_decay'])
                                   if discriminator_parameters else None)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=count_epochs)
        generator = torch.Generator().manual_seed(SETTINGS['seed'])
        for epoch in range(1, count_epochs+1):
            model.train()
            for bag in train_bags:
                bag.resample(3000+epoch)
            order = torch.randperm(len(train_bags), generator=generator)
            losses, n = {}, 0
            for start in range(0, len(order), SETTINGS['patient_batch_size']):
                indices = order[start:start+SETTINGS['patient_batch_size']].tolist()
                selected = torch.as_tensor(indices)
                rna_batch = cohort.train_rna[selected].to(torch_device)
                label_batch = labels[selected].to(torch_device)
                censor_batch = (1-events[selected]).to(torch_device)
                output = forward_arm(model_id, model, train_bags, rna_batch,
                                     indices, torch_device, label_batch, censor_batch)
                survival_loss = discrete_nll(output['logits'].double(), label_batch,
                                             censor_batch.double(), alpha=ALPHA_SURV)
                loss = survival_loss
                components = {'survival': survival_loss}
                if discriminator_optimizer is not None:
                    # The D-only step runs first and detaches the encoders, so
                    # the adversarial term cannot move the shared encoders.
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    discriminator_loss = model.discriminator_loss(output)
                    if not torch.isfinite(discriminator_loss):
                        raise FloatingPointError('Nonfinite discriminator objective')
                    discriminator_loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for p in discriminator_parameters):
                        raise FloatingPointError('Nonfinite discriminator gradient')
                    discriminator_optimizer.step()
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    components['discriminator'] = discriminator_loss
                auxiliary = model.auxiliary_losses(output) if hasattr(model, 'auxiliary_losses') else {}
                for name, value in auxiliary.items():
                    weight = getattr(model, 'auxiliary_weights', {}).get(name, 1.0)
                    loss = loss + weight*value
                    components[name] = value
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training objective')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in main_parameters):
                    raise FloatingPointError('Nonfinite parameter gradient')
                optimizer.step()
                for name, value in components.items():
                    losses[name] = losses.get(name, 0.)+len(indices)*float(
                        value.detach() if torch.is_tensor(value) else value)
                n += len(indices)
            scheduler.step()
            metrics, survival, _ = evaluate_arm(model_id, model, cohort,
                                                torch_device, val_bags)
            history.append({'epoch': epoch,
                            'train_terms': {k: v/n for k, v in losses.items()},
                            'learning_rate_after_step': scheduler.get_last_lr()[0],
                            **metrics})
            print(json.dumps({'fold': plan['fold'], 'model': model_id,
                              'epoch': epoch, 'c_index': metrics['c_index'],
                              'ibs': metrics['ibs'], 'iauc': metrics['iauc']},
                             allow_nan=False), flush=True)
            if best_metrics is None or metrics['c_index'] > best_metrics['c_index']:
                best_metrics, best_survival, best_epoch = metrics, survival.copy(), epoch
            last_metrics, last_survival = metrics, survival

        result = {
            'protocol': PROTOCOL, 'protocol_version': PROTOCOL_VERSION,
            'fold': plan['fold'], 'model_id': model_id, 'metadata': metadata,
            'plan_sha256': plan_sha256, 'settings': plan['settings'],
            'bin_edges': plan['inputs']['bin_edges'],
            'loss_version': DISCRETE_NLL_VERSION,
            'patients': plan['inputs']['patients'], 'events': plan['inputs']['events'],
            'source_sha256': plan['source_sha256'],
            'upstream_sources': plan['upstream_sources'],
            'epochs_completed': count_epochs, 'smoke_only': smoke,
            'test_evaluated': False, 'outer_test_used': False,
            'best_epoch': best_epoch, 'last_epoch': count_epochs,
            'best_validation': best_metrics, 'last_validation': last_metrics,
            'history': history,
            'parameters': sum(p.numel() for p in model.parameters())
                         if hasattr(model, 'parameters') else None,
            'wall_seconds': time.perf_counter()-started,
            'limits': [
                'Validation-selected fold scores; not an independent outer test estimate.',
                'Adapted methods are not original-paper result reproductions.',
                'Shared/unique structure does not imply prognostic value.',
            ],
        }
        write_new(destination/'report.json', result)
        _save_predictions(destination/'val_predictions_best.csv',
                          cohort.val_ids, best_survival)
        return result
    finally:
        for bag in list(cohort.train_bags)+list(cohort.val_bags):
            if hasattr(bag, 'close'):
                bag.close()


def _save_predictions(path, ids, survival):
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient_id', 'risk']+[f'survival_bin_{k}' for k in range(N_BINS)])
        writer.writerows([p, float(r), *map(float, s)]
                          for p, r, s in zip(ids, dct_risk(survival), survival))


# -------------------------------------------------------------------- summary

DCT_REFERENCE_ARCHIVE = Path(__file__).resolve().parent / 'reference' / 'dct_5fold_archive.json'


def load_dct_reference(path=None):
    """Load the archived DCT reference numbers for the protocol folds.

    The reference implementation is not re-run here.  Its numbers are read
    from a committed archive so that the summary can place every arm on the
    same table, and they are aggregated with the same mean/std rule.

    The committed archive was produced under the five-fold schedule and
    therefore holds five folds, while the current protocol declares only the
    folds it runs.  The loader **requires the archive to cover every protocol
    fold** and then selects exactly those folds, so a missing fold can never
    masquerade as a complete one and extra archived folds are simply not
    used.  The returned block states which folds were selected, so a
    single-fold row can never be read as a five-fold mean.
    """
    from comparison_protocol import FOLDS

    target = Path(path) if path else DCT_REFERENCE_ARCHIVE
    if not target.is_file():
        raise FileNotFoundError(f'Missing DCT reference archive: {target}')
    payload = json.loads(target.read_text(encoding='utf-8'))
    folds = payload.get('folds')
    if not isinstance(folds, list) or not folds:
        raise ValueError('DCT reference archive has no folds')
    by_fold = {}
    for record in folds:
        fold = int(record['fold'])
        if fold in by_fold:
            raise ValueError(f'Duplicate fold {fold} in the DCT reference archive')
        by_fold[fold] = record
    missing = sorted(set(FOLDS) - set(by_fold))
    if missing:
        raise ValueError(f'DCT reference archive must cover folds {sorted(FOLDS)}; '
                         f'missing {missing}')
    ordered = [by_fold[fold] for fold in FOLDS]
    return {'archive_id': payload.get('archive_id'),
            'archive_path': str(target.resolve()),
            'protocol': payload.get('protocol'),
            'source': payload.get('source'),
            'limits': payload.get('limits', []),
            'official_model_reproduction': True,
            're_run': False,
            'archive_fold_count': len(by_fold),
            'selected_folds': list(FOLDS),
            'selection_note': ('The archive holds the five-fold schedule; only the '
                               'protocol folds are read. The reference row is a '
                               'single-fold value under the reduced schedule, not a '
                               'five-fold mean.'),
            'folds': ordered}


def _metric_rows(model_id, folds, metrics_by_fold):
    """Mean/std rows for one arm from its per-fold metric dicts.

    ``metrics_by_fold`` maps a fold index to the metric dictionary of that
    fold.  The same rule is applied to every arm, including the archived DCT
    reference, so the table is comparable by construction.

    Under the reduced one-fold schedule the ``mean`` is the single fold's
    value and the ``std`` is zero.  The row records ``fold_count`` so a
    reader cannot mistake it for a five-fold aggregate.
    """
    rows = []
    for key in ('c_index', 'ibs', 'iauc'):
        values = [metrics_by_fold[fold].get(key) for fold in folds]
        if any(value is None for value in values):
            rows.append({'model': model_id, 'metric': key, 'mean': None, 'std': None,
                         'per_fold': values, 'folds': list(folds),
                         'fold_count': len(folds),
                         'note': 'metric undefined in at least one fold'})
            continue
        array = np.asarray(values, dtype=np.float64)
        rows.append({'model': model_id, 'metric': key, 'mean': float(array.mean()),
                     'std': float(array.std(ddof=0)), 'per_fold': [float(v) for v in values],
                     'folds': list(folds), 'fold_count': len(folds)})
    return rows


def summarize(run_dirs, out, dct_reference=None):
    """Aggregate per-fold reports into mean +/- std across the protocol folds."""
    records = []
    for directory in run_dirs:
        records.append(json.loads((Path(directory)/'report.json').read_text(encoding='utf-8')))
    if not records:
        raise ValueError('No run reports to summarize')
    by_model = {}
    for record in records:
        if record.get('smoke_only') or record.get('test_evaluated'):
            raise ValueError('Refuse smoke or test-evaluated runs in a comparison summary')
        by_model.setdefault(record['model_id'], []).append(record)
    rows = []
    for model_id, group in sorted(by_model.items()):
        folds = sorted(int(r['fold']) for r in group)
        if len(set(folds)) != len(folds):
            raise ValueError(f'Duplicate fold in {model_id}')
        if set(folds) != set(FOLDS):
            raise ValueError(f'{model_id} does not cover folds {FOLDS}: got {folds}')
        if len({r['plan_sha256'] for r in group}) != 1:
            raise ValueError(f'{model_id} used more than one plan')
        ordered = sorted(group, key=lambda x: x['fold'])
        rows.extend(_metric_rows(model_id, folds,
                                 {int(r['fold']): r['best_validation'] for r in ordered}))
    reference_block = None
    if dct_reference is not None:
        archive = load_dct_reference(dct_reference)
        folds = [int(record['fold']) for record in archive['folds']]
        rows.extend(_metric_rows('dct_reference', folds,
                                 {int(record['fold']): record for record in archive['folds']}))
        reference_block = {k: v for k, v in archive.items() if k != 'folds'}
    result = {'protocol': PROTOCOL, 'protocol_version': PROTOCOL_VERSION,
              'shared_settings': SHARED_SETTINGS, 'rows': rows,
              'models': sorted(set(list(by_model)+(['dct_reference'] if reference_block else []))),
              'dct_reference': reference_block,
              'metric_directions': {'c_index': 'higher', 'ibs': 'lower', 'iauc': 'higher'},
              'limits': [f'Mean and population std over the {len(FOLDS)} protocol fold(s); '
                         f'the schedule was reduced from five folds to {list(FOLDS)}, so the '
                         'spread is not a five-fold estimate.',
                         'No significance test is performed; differences are not claims of superiority.',
                         'All arms share the same binning, likelihood, splits and metric call.',
                         'The DCT reference row is an archived reference-implementation result, not a re-run in this repository.']}
    write_new(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--fold', type=int, required=True, choices=FOLDS)
    p.add_argument('--manifest', required=True)
    p.add_argument('--survival-reference', required=True)
    p.add_argument('--genes', required=True)
    p.add_argument('--gmt', required=True)
    p.add_argument('--deref-signatures', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    p = sub.add_parser('run')
    p.add_argument('--plan', required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--model', required=True, choices=MODELS)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--smoke', action='store_true')
    p = sub.add_parser('summary')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--dct-reference', default=None,
                   help='Archived DCT reference archive; it is loaded, never re-run.')
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    if args.action == 'plan':
        write_new(args.out, make_plan(args.fold, args.manifest, args.survival_reference,
                                      args.genes, args.gmt, args.deref_signatures,
                                      args.device))
    elif args.action == 'run':
        run_model(args.plan, args.plan_sha256, args.model, args.out, args.device,
                  args.smoke, args.epochs)
    else:
        summarize(args.runs, args.out, dct_reference=args.dct_reference)
    print(json.dumps({'action': args.action, 'out': args.out}), flush=True)


if __name__ == '__main__':
    main()
