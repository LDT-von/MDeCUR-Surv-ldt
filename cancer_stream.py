"""Lazy cached-UNI training: patient batches and checkpointed patch summaries.

Patch chunking preserves the objective within a fixed patient batch. Patient
minibatching changes the population used for each correlation estimate.
"""
from copy import deepcopy
from functools import partial
import math

import torch
from torch.utils.checkpoint import checkpoint

from cancer_model import CancerDeCUR, piecewise_nll
from cancer_train import effective_loss_weights
from model import correlation_loss, stability_weighted_common, DegenerateCorrelation


def _merge_moments(left, right, fraction):
    """Merge normalized centered moments, retaining all parameter gradients."""
    ma, mb, va, vb, cov = left
    na, nb, wa, wb, cross = right
    da, db = na - ma, nb - mb
    t = fraction
    return (ma + t*da, mb + t*db,
            (1-t)*va + t*wa + t*(1-t)*da.square(),
            (1-t)*vb + t*wb + t*(1-t)*db.square(),
            (1-t)*cov + t*cross + t*(1-t)*da[:, None]*db[None, :])


def _chunk_summary(model, bag, block, names, include_local, device_token):
    parameter = next(model.parameters())
    arrays = bag.read_block(block, names)
    encoded = [model.patch_encoder(x.to(device=parameter.device, dtype=parameter.dtype)) for x in arrays]
    summary = []
    for features in encoded:
        scores = model.attention(features).squeeze(-1)
        summary.extend((scores.logsumexp(0), (scores.softmax(0)[:, None]*features).sum(0)))
    if include_local:
        # Float64 centered accumulation avoids E[z^2]-E[z]^2 cancellation.
        a, b = (model.local_projector(f).double() for f in encoded[1:])
        ma, mb = a.mean(0), b.mean(0)
        ac, bc = a-ma, b-mb
        summary.extend((ma, mb, ac.square().mean(0), bc.square().mean(0), ac.T@bc/len(a)))
    return tuple(summary)


def _stream_patient(model, bag, chunk_size, include_views, include_local, checkpoint_chunks):
    names = ('features_reference', 'features_a', 'features_b') if include_views else ('features_reference',)
    summaries = None
    moments = None
    seen = 0
    device = next(model.parameters()).device
    for block in bag.blocks(chunk_size):
        # Partial binds this exact immutable block; no loop-variable closure.
        fn = partial(_chunk_summary, model, bag, block, names, include_local)
        token = torch.empty(0, device=device)
        if checkpoint_chunks and torch.is_grad_enabled():
            values = checkpoint(fn, token, use_reentrant=False, preserve_rng_state=True)
        else:
            values = fn(token)
        count = block[2] - block[1]
        if count < 1:
            raise ValueError('A stream block must be nonempty')
        parts = [(values[2*i], values[2*i+1]) for i in range(len(names))]
        if summaries is None:
            summaries = parts
        else:
            merged = []
            for (ll, lm), (rl, rm) in zip(summaries, parts):
                log_total = torch.logaddexp(ll, rl)
                merged.append((log_total, (ll-log_total).exp()*lm + (rl-log_total).exp()*rm))
            summaries = merged
        if include_local:
            current = values[2*len(names):]
            moments = current if moments is None else _merge_moments(moments, current, count/(seen+count))
        seen += count
    if seen != bag.n_patches or summaries is None:
        raise ValueError('Stream blocks must cover the full patient bag')
    return [value[1] for value in summaries], moments


def _local_from_patients(moments, bags, eps=1e-4, off_weight=.005):
    if len(moments) < 2:
        raise ValueError('Local correlation requires at least two patients')
    ma = torch.stack([m[0] for m in moments]).mean(0)
    mb = torch.stack([m[1] for m in moments]).mean(0)
    va = torch.stack([m[2] + (m[0]-ma).square() for m in moments]).mean(0)
    vb = torch.stack([m[3] + (m[1]-mb).square() for m in moments]).mean(0)
    cov = torch.stack([m[4] + (m[0]-ma)[:, None]*(m[1]-mb)[None, :] for m in moments]).mean(0)
    corr = cov / (va+eps).sqrt()[:, None] / (vb+eps).sqrt()[None, :]
    diagonal = corr.diagonal()
    off = corr-torch.diag_embed(diagonal)
    loss = (diagonal-1).square().sum() + off_weight*off.square().sum()
    return loss, {'patients': len(bags), 'matched_patches': sum(b.n_patches for b in bags),
                  'variance_min': torch.minimum(va.min(), vb.min()).detach().item(),
                  'low_variance_dimensions': int(((va < eps) | (vb < eps)).sum()),
                  'diagonal_mean': diagonal.detach().mean().item()}


def stream_forward(model, bags, rna, chunk_size=2048, include_views=False,
                   include_local=False, checkpoint_chunks=True, rna_views=None):
    """Compute exact complete-bag summaries; never return all patch attention.

    A/B features are read only for training auxiliary objectives. RNA view noise
    must be sampled by the caller before checkpointed patch functions execute.
    """
    if not isinstance(chunk_size, int) or chunk_size < 1 or not bags or len(bags) != len(rna):
        raise ValueError('Positive chunk_size and one RNA row per nonempty bag required')
    if include_local and (not include_views or model.local_projector is None):
        raise ValueError('Local summaries require aligned views and a local projector')
    if rna_views is not None and (len(rna_views) != 2 or any(v.shape != rna.shape for v in rna_views)):
        raise ValueError('RNA views must match the patient batch')
    parameter = next(model.parameters())
    rna = rna.to(device=parameter.device, dtype=parameter.dtype)
    pooled, moments = [], []
    for bag in bags:
        values, local = _stream_patient(model, bag, chunk_size, include_views, include_local, checkpoint_chunks)
        pooled.append(values)
        if include_local:
            moments.append(local)
    hp = torch.stack([p[0] for p in pooled])
    hr = model.rna_encoder(rna)
    p, r = (model.path_projector(hp), model.rna_projector(hr)) if model.prediction_space == 'projected' else (hp, hr)
    out = {'log_rates': model.head(torch.cat((p, r), 1)), 'path': p, 'rna': r,
           'path_hidden': hp, 'rna_hidden': hr}
    if include_views:
        g1, g2 = (rna, rna) if rna_views is None else [v.to(device=parameter.device, dtype=parameter.dtype) for v in rna_views]
        out['auxiliary'] = {
            'path1': model.path_projector(torch.stack([p[1] for p in pooled])),
            'path2': model.path_projector(torch.stack([p[2] for p in pooled])),
            'rna1': model.rna_projector(model.rna_encoder(g1)),
            'rna2': model.rna_projector(model.rna_encoder(g2)),
        }
    if include_local:
        out['local_loss'], out['local_diagnostics'] = _local_from_patients(moments, bags)
    return out


def _objective(model, batch, starts, cfg, chunk_size):
    weights = effective_loss_weights(cfg)
    g = batch['rna']
    use_views = any(weights.values())
    views = (g + cfg.rna_noise*torch.randn_like(g), g + cfg.rna_noise*torch.randn_like(g)) if use_views else None
    out = stream_forward(model, batch['bags'], g, chunk_size, use_views,
                         bool(weights['local']), rna_views=views)
    nll = piecewise_nll(out['log_rates'], batch['time'], batch['event'], starts)
    loss = nll
    stats = {'train_nll': nll.item(), 'patients': len(g), 'common_loss': None,
             'path_within_loss': None, 'rna_within_loss': None, 'unique_loss': None,
             'local_loss': None, 'local_diagnostics': None, 'ess': None,
             'weight_mean': None, 'weight_min': None, 'weight_std': None, 'skipped': False}
    if use_views:
        aux = out['auxiliary']
        for key, left, right, target in (
                ('path_within', aux['path1'], aux['path2'], 1),
                ('rna_within', aux['rna1'], aux['rna2'], 1),
                ('unique', aux['path1'][:, cfg.common:], aux['rna1'][:, cfg.common:], 0)):
            if weights[key]:
                term = correlation_loss(left, right, target)
                loss = loss + weights[key]*term
                stats[key+'_loss'] = term.item()
        if weights['local']:
            loss = loss + weights['local']*out['local_loss']
            stats.update(local_loss=out['local_loss'].item(), local_diagnostics=out['local_diagnostics'])
        if weights['common']:
            try:
                common, diagnostic = stability_weighted_common(
                    aux['path1'][:, :cfg.common], aux['path2'][:, :cfg.common],
                    aux['rna1'][:, :cfg.common], aux['rna2'][:, :cfg.common],
                    strength=0 if cfg.common_mode == 'uniform' else cfg.strength,
                    min_weight=cfg.min_weight, shuffle_weights=cfg.common_mode == 'shuffled')
                loss = loss + weights['common']*common
                w = diagnostic['weights']
                stats.update(common_loss=common.item(), ess=diagnostic['ess'].item(),
                             weight_mean=w.mean().item(), weight_min=w.min().item(),
                             weight_std=w.std(unbiased=False).item())
            except DegenerateCorrelation:
                stats['skipped'] = True
    if not torch.isfinite(loss):
        raise FloatingPointError('Streaming training loss is not finite')
    stats['total_loss'] = loss.item()
    return loss, stats


def _patient_batches(order, size, require_pairs):
    if size < (2 if require_pairs else 1):
        raise ValueError('Auxiliary correlations require patient_batch_size >= 2')
    batches = list(order.split(size))
    if require_pairs and len(batches[-1]) == 1:
        if len(batches) == 1:
            raise ValueError('Auxiliary correlations require at least two patients')
        batches[-2] = torch.cat((batches[-2], batches[-1]))
        batches.pop()
    return batches


def _validate_stream_cohort(cohort):
    required = ('ids', 'bags', 'rna', 'time', 'event')
    if any(k not in cohort for k in required):
        raise ValueError('Missing streaming cohort fields')
    n = len(cohort['ids'])
    if (not n or len(set(cohort['ids'])) != n or len(cohort['bags']) != n
            or cohort['rna'].ndim != 2 or len(cohort['rna']) != n
            or cohort['time'].shape != (n,) or cohort['event'].shape != (n,)
            or not torch.isfinite(cohort['rna']).all()):
        raise ValueError('Invalid streaming patient rows')


def train_stream_model(train, validation, starts, cfg, patient_batch_size=16,
                       patch_chunk_size=2048, device='cpu'):
    """Select by all-validation reference NLL; test patients are not accepted."""
    if cfg.path_views != 'cached_uni' or cfg.patch_keep != 1:
        raise ValueError('Streaming training requires cached_uni and patch_keep=1')
    if cfg.common_mode not in ('off', 'uniform', 'stability', 'shuffled'):
        raise ValueError('Invalid common mode')
    weights = effective_loss_weights(cfg)
    if any(not math.isfinite(v) or v < 0 for v in (*weights.values(), cfg.lambda_within, cfg.lambda_common, cfg.strength)):
        raise ValueError('Loss coefficients must be finite and nonnegative')
    if (cfg.epochs < 1 or cfg.patience < 1 or not math.isfinite(cfg.lr) or cfg.lr <= 0
            or not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0
            or not math.isfinite(cfg.rna_noise) or cfg.rna_noise < 0 or not 0 < cfg.min_weight <= 1
            or type(patch_chunk_size) is not int or patch_chunk_size < 1
            or type(patient_batch_size) is not int or patient_batch_size < 1
            or type(cfg.local_dim) is not int or cfg.local_dim < 1):
        raise ValueError('Invalid streaming training configuration')
    _validate_stream_cohort(train)
    _validate_stream_cohort(validation)
    if set(train['ids']) & set(validation['ids']):
        raise ValueError('Patient overlap between training and validation')
    dim = train['bags'][0].feature_dim
    if (any(b.feature_dim != dim or b.n_patches < 1 for b in train['bags']+validation['bags'])
            or train['rna'].shape[1] != validation['rna'].shape[1]):
        raise ValueError('Incompatible streaming feature dimensions')
    for cohort in (train, validation):
        piecewise_nll(torch.zeros(len(cohort['ids']), len(starts)), cohort['time'], cohort['event'], starts)
    _patient_batches(torch.arange(len(train['ids'])), patient_batch_size, any(weights.values()))
    torch.manual_seed(cfg.seed)
    model_kwargs = dict(path_dim=dim, rna_dim=train['rna'].shape[1], hidden=cfg.hidden,
                        common=cfg.common, bins=len(starts), prediction_space=cfg.prediction_space,
                        local_dim=cfg.local_dim if cfg.lambda_local else None)
    model = CancerDeCUR(**model_kwargs).to(device)
    mean = train['rna'].mean(0)
    scale = train['rna'].std(0, unbiased=False).clamp_min(1e-6)
    normalized = (train['rna']-mean)/scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history, best_state, best_epoch, stale, skipped = [], None, 0, 0, 0
    best = math.inf
    fitted = dict(model=model, rna_mean=mean, rna_scale=scale, starts=starts.clone(),
                  model_kwargs=model_kwargs, effective_loss_weights=weights,
                  stream_config=dict(patient_batch_size=patient_batch_size,
                                     patch_chunk_size=patch_chunk_size, device=str(device),
                                     checkpoint_chunks=True, local_moment_dtype='float64',
                                     correlation_scope='patient_minibatch'))
    generator = torch.Generator().manual_seed(cfg.seed+1009)
    for epoch in range(1, cfg.epochs+1):
        model.train()
        records = []
        order = torch.randperm(len(train['ids']), generator=generator)
        for indices in _patient_batches(order, patient_batch_size, any(weights.values())):
            optimizer.zero_grad(set_to_none=True)
            batch = dict(bags=[train['bags'][i] for i in indices.tolist()],
                         rna=normalized[indices].to(device), time=train['time'][indices].to(device),
                         event=train['event'][indices].to(device))
            loss, stats = _objective(model, batch, starts.to(device), cfg, patch_chunk_size)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Streaming parameter gradient is not finite')
            optimizer.step()
            records.append(stats)
            skipped += int(stats['skipped'])
        predicted = predict_stream(fitted, validation)['log_rates']
        val = piecewise_nll(predicted, validation['time'], validation['event'], starts).item()
        if not math.isfinite(val):
            raise FloatingPointError('Streaming validation loss is not finite')
        n = len(train['ids'])
        history.append(dict(epoch=epoch, validation_nll=val,
                            train_nll=sum(r['patients']*r['train_nll'] for r in records)/n,
                            total_loss=sum(r['patients']*r['total_loss'] for r in records)/n,
                            batch_patient_counts=[r['patients'] for r in records], batches=records))
        if val < best:
            best, best_epoch, best_state, stale = val, epoch, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    fitted.update(best_epoch=best_epoch, validation_nll=best, history=history, skipped_common=skipped)
    return fitted


def predict_stream(fitted, cohort):
    """Reference-only prediction, with labels and augmented caches unused."""
    model = fitted['model']
    model.eval()
    if not cohort['bags'] or len(cohort['bags']) != len(cohort['rna']):
        raise ValueError('One RNA row per patient bag is required')
    g = (cohort['rna']-fitted['rna_mean'])/fitted['rna_scale']
    outputs = []
    with torch.no_grad():
        for i, bag in enumerate(cohort['bags']):
            outputs.append(stream_forward(model, [bag], g[i:i+1],
                                          fitted['stream_config']['patch_chunk_size'], checkpoint_chunks=False))
    return {k: torch.cat([o[k].cpu() for o in outputs]) for k in outputs[0]}
