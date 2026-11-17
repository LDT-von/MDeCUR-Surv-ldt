"""Explicit-split adapter for exploratory method 3.1 patient-cohort pilots.

This module does not open patient files or invent a train/validation split. The
caller supplies aligned bags, RNA and outcomes, with disjoint patient IDs.
No assumption is made that the supplied evaluation split is an independent test.
"""
from dataclasses import asdict, dataclass
import copy
import numpy as np
import torch

from cancer_morphid_model import FrozenMorphologyBank
from cancer_parity import dct_bin_edges, discretize_labels, discrete_nll, discrete_survival
from cancer_rna import fit_rna_scaler, transform_rna
from method3_1_model import Method31, _ids
from method3_1_interaction_features import GlobalResidualInteractions
from method3_1_guarded_pipeline import GuardedMethod31Pipeline


@dataclass(frozen=True)
class PilotConfig:
    """Fixed ahead of use; values are engineering defaults, not clinical optima."""
    seed: int = 3
    clusters: int = 2
    latent_dim: int = 3
    projection_dim: int = 4
    max_fit_patches: int = 2560
    per_patient_cap: int = 16
    ssl_epochs: int = 35
    bins: int = 4
    probe_folds: int = 4
    head_epochs: int = 160
    rna_policy: str = 'clamp'


def _checked(split, name):
    if not isinstance(split, dict) or not {'bags', 'rna', 'ids', 'time', 'event'} <= set(split):
        raise ValueError(f'{name} needs bags, rna, ids, time and event')
    bags = list(split['bags'])
    ids = _ids(split['ids'], len(bags))
    if not bags:
        raise ValueError(f'{name} cannot be empty')
    rna = torch.as_tensor(split['rna']).detach().cpu().double()
    time = torch.as_tensor(split['time']).detach().cpu().double().reshape(-1)
    event = torch.as_tensor(split['event']).detach().cpu().double().reshape(-1)
    if (rna.ndim != 2 or rna.shape[0] != len(ids) or rna.shape[1] < 1
            or len(time) != len(ids) or len(event) != len(ids)
            or not torch.isfinite(rna).all() or not torch.isfinite(time).all()
            or not torch.isfinite(event).all() or (time < 0).any()
            or ((event != 0) & (event != 1)).any()):
        raise ValueError(f'{name} has malformed or nonfinite features/outcomes')
    for identity, bag in zip(ids, bags):
        if hasattr(bag, 'patient_id') and str(bag.patient_id) != identity:
            raise ValueError(f'{name} patient identity and bag disagree')
    return {'bags': bags, 'ids': ids, 'rna': rna, 'time': time, 'event': event}


def _identity_disjoint(groups):
    seen = set()
    for ids in groups:
        if seen.intersection(ids):
            raise ValueError('Train, selection, gate and evaluation IDs must be disjoint')
        seen.update(ids)


def fit_pilot(train, selection, gate, evaluation_ids, config=PilotConfig()):
    """Fit once using explicit disjoint splits, and reserve evaluation IDs.

    Each of train/selection/gate has bags, rna, ids, time and event (1 means
    observed event). Evaluation features/outcomes are not accepted here.
    The caller provides an already agreed patient-level split. A small or
    event-poor gate is closed by the underlying gate, not silently recycled.
    """
    if not isinstance(config, PilotConfig):
        raise TypeError('config must be a PilotConfig')
    tr, sel, cal = (_checked(x, n) for x, n in ((train, 'train'),
                      (selection, 'selection'), (gate, 'gate')))
    evaluation_ids = _ids(evaluation_ids, len(evaluation_ids))
    if not evaluation_ids:
        raise ValueError('At least one evaluation identity must be reserved')
    _identity_disjoint((tr['ids'], sel['ids'], cal['ids'], evaluation_ids))
    dim = tr['rna'].shape[1]
    if any(x['rna'].shape[1] != dim for x in (sel, cal)):
        raise ValueError('RNA dimensions differ across patient splits')
    edges = dct_bin_edges(tr['time'].numpy(), tr['event'].numpy(), config.bins)
    scaler = fit_rna_scaler(tr['rna'], policy=config.rna_policy)
    for split in (tr, sel, cal):
        split['rna'] = transform_rna(split['rna'], scaler)
        split['labels'] = torch.from_numpy(discretize_labels(split['time'].numpy(), edges))
        split['censorship'] = 1 - split['event']
    bank = FrozenMorphologyBank.fit(tr['bags'], ids=tr['ids'], clusters=config.clusters,
        latent_dim=config.latent_dim, max_fit_patches=config.max_fit_patches,
        per_patient_cap=config.per_patient_cap, seed=config.seed)
    base = Method31(bank, rna_dim=dim, projection_dim=config.projection_dim,
                    bins=config.bins, seed=config.seed)
    base.pretrain(tr['bags'], tr['rna'], tr['ids'], epochs=config.ssl_epochs)
    # Raises on an inadequately represented cluster; do not drop/merge silently.
    base.fit_probes(tr['bags'], tr['rna'], tr['ids'], folds=config.probe_folds)
    mapper = GlobalResidualInteractions(base).fit(tr['bags'], tr['rna'], tr['ids'],
                                                   folds=config.probe_folds)
    model = GuardedMethod31Pipeline(mapper)
    args = lambda x: (x['bags'], x['rna'], x['ids'], x['labels'], x['censorship'])
    result = model.fit(tr['labels'], tr['censorship'], tr['ids'], selection=args(sel),
                       gate=args(cal), test_ids=evaluation_ids, epochs=config.head_epochs)
    return {'model': model, 'rna_scaler': scaler, 'bin_edges': edges,
            'fit_result': result, 'config': config,
            'disclosure': 'Exploratory pilot; evaluation independence must be established separately.'}


def evaluate_pilot(fitted, evaluation):
    """Evaluate reserved IDs; caller prevents repeated test-informed choices.

    The API permits repeated calls and subsets of reserved IDs. The caller must
    establish cohort independence and control how often outcomes are inspected.
    """
    val = _checked(evaluation, 'evaluation')
    normalized = transform_rna(val['rna'], fitted['rna_scaler'])
    labels = torch.from_numpy(discretize_labels(val['time'].numpy(), fitted['bin_edges']))
    logits = fitted['model'].predict(val['bags'], normalized, val['ids'])
    return {'logits': logits, 'survival': discrete_survival(logits),
            'nll_alpha0': float(discrete_nll(logits, labels, 1-val['event'], alpha=0.)),
            'n_patients': len(val['ids']), 'events': int(val['event'].sum())}


def checkpoint_pilot(fitted):
    """Snapshot includes patient IDs and kernel training features: keep private.

    This deliberately performs no filesystem or network writes. Never publish a
    patient-trained checkpoint, and never deserialize untrusted pickle data.
    """
    return {'version': 'method3_1_patient_pilot_v1', 'model': fitted['model'].checkpoint(),
            'rna_scaler': copy.deepcopy(fitted['rna_scaler']),
            'bin_edges': np.array(fitted['bin_edges'], copy=True),
            'config': asdict(fitted['config']), 'fit_result': copy.deepcopy(fitted['fit_result']),
            'disclosure': fitted['disclosure'],
            'sensitivity': 'contains patient IDs and patient-derived kernel support/training features'}


def restore_pilot(snapshot):
    """Restore an already trusted in-memory checkpoint snapshot."""
    if not isinstance(snapshot, dict) or snapshot.get('version') != 'method3_1_patient_pilot_v1':
        raise ValueError('Unsupported pilot checkpoint')
    return {'model': GuardedMethod31Pipeline.from_checkpoint(snapshot['model']),
            'rna_scaler': copy.deepcopy(snapshot['rna_scaler']),
            'bin_edges': np.array(snapshot['bin_edges'], copy=True),
            'config': PilotConfig(**snapshot['config']),
            'fit_result': copy.deepcopy(snapshot['fit_result']), 'disclosure': snapshot['disclosure']}
