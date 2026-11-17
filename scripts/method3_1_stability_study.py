"""Pre-registered, independently generated CPU-only method 3.1 stability study."""
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cancer_morphid_model import FrozenMorphologyBank, discrete_nll  # noqa: E402
from method3_1_model import Method31  # noqa: E402
from method3_1_interaction_features import GlobalResidualInteractions  # noqa: E402
from method3_1_strict_pipeline import Method31StrictPipeline  # noqa: E402
from method3_1_active_head import ActiveHierarchicalSurvivalHead  # noqa: E402
from method3_1_clr_conditioning import CLRConditioner  # noqa: E402
from method3_1_independent_synthetic import make_independent_splits  # noqa: E402

SEEDS = tuple(range(1301, 1321))
PRIMARY = ('xor', 'composition_only', 'composition_plus_rna', 'null', 'redundant_nonlinear')
SECONDARY = ('xor_smooth', 'xor_time_varying', 'additive', 'rna_only', 'path_only',
             'heteroscedastic_null', 'composition_interaction')
FINITE = tuple(itertools.product((0., .1), (.1, 1.), (.1, 1.)))
GATED = FINITE + ((0., None, None), (.1, None, None))
CLR_GRID = tuple((x, None, None) for x in (0., .01, .1, 1., 10.))
SIZES = {'train': 160, 'validation': 120, 'test': 360}


def nll(logits, part):
    return float(discrete_nll(logits, part['labels'], part['censorship'], alpha=0.))


def evaluate(name, tr, va, te, tr_blocks, va_blocks, te_blocks, *, gated=False, clr_only=False):
    has_interaction = tr_blocks['interaction'].shape[1] > 0 and not clr_only
    head = ActiveHierarchicalSurvivalHead(bins=4, interaction_mode='per_bin',
        min_select_epoch=0 if gated or clr_only else 5, require_interaction=has_interaction and not gated)
    candidates = CLR_GRID if clr_only else GATED if gated else FINITE
    result = head.fit(tr_blocks, tr['labels'], tr['censorship'], tr['ids'], va_blocks,
        va['labels'], va['censorship'], va['ids'], candidates=candidates, epochs=160, lr=.03, eval_every=5)
    prediction = head.predict(te_blocks, te['ids'])
    row = {'arm': name, 'validation_nll': result['val_nll'], 'train_nll': result['train_nll'],
           'test_nll': nll(prediction, te), 'selection': result, 'dimensions': head.dimensions,
           'nominal_parameters': sum(v.numel() for v in head.weights.values())+head.bias.numel()}
    if has_interaction:
        standard = (te_blocks['interaction']-head.mean['interaction'])/head.scale['interaction']
        contribution = standard @ head.weights['interaction'].T
        row['interaction_weight_norm'] = float(head.weights['interaction'].norm())
        row['interaction_logit_std'] = float(contribution.std(unbiased=False))
        row['zero_interaction_test_nll'] = nll(prediction-contribution, te)
        # A new tensor isolates the gradient of the survival likelihood from
        # the penalty; fixed other blocks and trained head remain unchanged.
        weight = head.weights['interaction'].clone().requires_grad_(True)
        blocks = head._normalized(tr_blocks)
        train_logits = (blocks['clr'] @ head.weights['clr'].T + blocks['other'] @ head.weights['other'].T
                        + blocks['interaction'] @ weight.T + head.bias)
        likelihood = discrete_nll(train_logits, tr['labels'], tr['censorship'], alpha=0.)
        row['survival_only_interaction_grad_norm'] = float(torch.autograd.grad(likelihood, weight)[0].norm())
    return row, head


def case_data(case, seed):
    return make_independent_splits(seed, case, n_train=SIZES['train'],
                                   n_val=SIZES['validation'], n_test=SIZES['test'])


def run_one(case, seed):
    data, _, meta = case_data(case, seed)
    tr, va, te = (data[k] for k in ('train', 'validation', 'test'))
    groups = [set(p['ids']) for p in (tr, va, te)]
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
        raise ValueError('Patient split overlap')
    bank = FrozenMorphologyBank.fit(tr['bags'], ids=tr['ids'], clusters=2, latent_dim=3,
        max_fit_patches=2560, per_patient_cap=16, seed=seed)
    base = Method31(bank, rna_dim=2, projection_dim=4, bins=4, seed=seed)
    base.pretrain(tr['bags'], tr['rna'], tr['ids'], epochs=35)
    base.fit_probes(tr['bags'], tr['rna'], tr['ids'], folds=4)
    mapper = GlobalResidualInteractions(base).fit(tr['bags'], tr['rna'], tr['ids'], folds=4)
    # Register and prepare train/validation/test features once; no outcomes are
    # passed into the bank, SSL, KRR or feature mapper.
    map_tr = mapper.training_features()
    map_va = mapper.validation_features(va['bags'], va['rna'], va['ids'])
    map_te = mapper.test_features(te['bags'], te['rna'], te['ids'])
    enc_tr, rna_tr, c_tr = base.cached
    _, enc_va, rna_va = base._input(va['bags'], va['rna'], va['ids'])
    _, enc_te, rna_te = base._input(te['bags'], te['rna'], te['ids'])
    # Strict _blocks ignores cluster_c by design; passing None makes it
    # impossible for this runner to accidentally inject RNA-conditioned c.
    parts = ((map_tr, enc_tr, rna_tr, c_tr), (map_va, enc_va, rna_va, None),
             (map_te, enc_te, rna_te, None))
    prototype = Method31StrictPipeline(mapper, mode='tanh_outer', expand_clr=True)
    prototype._fit_moments()
    by_mode = {}
    for mode in ('none', 'diagonal', 'outer', 'tanh_outer'):
        prototype.mode = mode
        by_mode[mode] = [prototype._blocks(*values) for values in parts]
    reference = by_mode['none']
    for mode in ('diagonal', 'outer', 'tanh_outer'):
        for idx in range(3):
            if not all(torch.equal(reference[idx][k], by_mode[mode][idx][k]) for k in ('clr', 'other')):
                raise RuntimeError('Strict main-effect inputs differ across arms')
        if by_mode[mode][0]['interaction'].shape[1] != 6:
            raise RuntimeError('Preset interaction dimension is not 6')
    if reference[0]['interaction'].shape[1] != 0:
        raise RuntimeError('No-interaction control has a nonempty interaction')
    rows, fitted = [], {}
    for name, mode, gated in (('strict_none','none',False), ('old_diagonal','diagonal',False),
                               ('global_outer','outer',False), ('new_bounded','tanh_outer',False),
                               ('new_gated','tanh_outer',True)):
        row, head = evaluate(name, tr, va, te, *by_mode[mode], gated=gated)
        rows.append(row); fitted[name] = head

    # Compare a frozen new model to removal and patient-wise shuffling of the
    # explicit interaction block. RNA-only main effects remain unchanged;
    # residuals retain their original RNA-conditioned definition.
    new_head = fitted['new_bounded']
    u, g = map_te['global_residual_standardized'], map_te['rna_standardized']
    changed = dict(by_mode['tanh_outer'][2])
    changed['interaction'] = (torch.tanh(u).unsqueeze(2)*torch.tanh(torch.roll(g, 1, 0)).unsqueeze(1)).flatten(1)
    mismatch = nll(new_head.predict(changed, te['ids']), te)
    next(row for row in rows if row['arm']=='new_bounded')['interaction_only_mismatch_test_nll'] = mismatch

    # Both composition-only controls share head budget/scale; nonlinear CLR
    # uses the same expanded training-only basis already supplied to models.
    clr_base = reference
    for name, nonlinear in (('clr_linear',False), ('clr_nonlinear',True)):
        comp = []
        for block in clr_base:
            clr = block['clr'] if nonlinear else block['clr'][:, :base.k]
            comp.append({'clr': clr, 'other': block['other'][:, :0],
                         'interaction': block['interaction'][:, :0]})
        row, _ = evaluate(name, tr, va, te, *comp, clr_only=True)
        rows.append(row)

    if case.startswith('composition_'):
        train, val, test = by_mode['tanh_outer']
        conditioner = CLRConditioner(alpha=.01).fit(train['clr'][:, :base.k], train['other'], tr['ids'],
                                                   folds=4, seed=seed)
        conditioned = [conditioner.training_features(tr['ids']),
            conditioner.validation_features(val['clr'][:, :base.k], val['other'], va['ids']),
            conditioner.test_features(test['clr'][:, :base.k], test['other'], te['ids'])]
        for original, transformed in zip((train, val, test), conditioned):
            if not torch.allclose(original['clr'], transformed['clr'], atol=1e-12, rtol=0.):
                raise RuntimeError('CLR expansion differs between conditioning arms')
            transformed['interaction'] = original['interaction']
        row, _ = evaluate('clr_conditioned_gated', tr, va, te, *conditioned, gated=True)
        row['conditioner_metadata'] = {'alpha': .01, 'folds': 4,
            'train': 'OOF conditional on outer training representation', 'test': 'fold-model ensemble'}
        rows.append(row)

    return {'seed': seed, 'case': case, 'metadata': meta,
        'events': {k: int((1-data[k]['censorship']).sum()) for k in data},
        'split_ids_sha256': {k: hashlib.sha256('\n'.join(data[k]['ids']).encode()).hexdigest() for k in data},
        'global_probe_folds': mapper.metadata['split_audit'], 'outcomes': rows}


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0,1,2))
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    if (args.worker is None) == (not args.merge):
        raise ValueError('Specify exactly one of --worker or --merge')
    watched = ('method3_1_model.py','method3_1_strict_pipeline.py','method3_1_interaction_pipeline.py',
        'method3_1_interaction_features.py','method3_1_active_head.py','method3_1_batched_head.py',
        'method3_1_hierarchical_head.py','method3_1_clr_conditioning.py','method3_1_independent_synthetic.py',
        'method3_1_regularized.py','cancer_morphid_model.py','morphid_probe.py','cancer_parity.py',
        'scripts/method3_1_stability_study.py','METHOD3_1_STABILITY_PREREG_20261117.md')
    hashes = {x: hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in watched}
    rows = []
    if args.merge:
        for worker in range(3):
            path = ROOT/'results'/f'method3_1_stability_worker_{worker}_20261117.json'
            saved = json.loads(path.read_text(encoding='utf8'))
            if saved['source_sha256'] != hashes or saved['worker'] != worker:
                raise ValueError(f'Incompatible worker receipt {path}')
            rows.extend(saved['runs'])
        expected = {(seed,case) for seed in SEEDS for case in PRIMARY+(SECONDARY if seed <= 1310 else ())}
        actual = [(r['seed'],r['case']) for r in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError('Missing, extra or duplicated worker runs')
        rows.sort(key=lambda r:(r['seed'], (PRIMARY+SECONDARY).index(r['case'])))
    else:
        partial = ROOT/'results'/f'method3_1_stability_worker_{args.worker}_partial_20261117.json'
        if partial.exists():
            raise FileExistsError(f'Refusing to replace an existing partial receipt: {partial}')
        for seed in SEEDS:
            if (seed-SEEDS[0]) % 3 != args.worker:
                continue
            for case in PRIMARY+(SECONDARY if seed <= 1310 else ()):
                try:
                    row = run_one(case, seed)
                    vals = {x['arm']:x['test_nll'] for x in row['outcomes']}
                    print(f"{seed} {case} diag={vals['old_diagonal']:.4f} new={vals['new_bounded']:.4f} "
                          f"none={vals['strict_none']:.4f} clr={vals['clr_nonlinear']:.4f}",flush=True)
                except Exception:
                    row = {'seed':seed,'case':case,'failure':traceback.format_exc()}
                    print(f"{seed} {case} FAILED {row['failure']}",flush=True)
                rows.append(row)
                # Each worker exclusively owns its own partial receipt. Keeping
                # it allows the completed diagnostics to survive interruption.
                partial.write_text(json.dumps({'worker':args.worker,'source_sha256':hashes,'runs':rows},
                    ensure_ascii=False,indent=2,allow_nan=False)+'\n', encoding='utf8')
                gc.collect()
    if any(hashlib.sha256((ROOT/x).read_bytes()).hexdigest() != h for x,h in hashes.items()):
        raise RuntimeError('Sources changed during the study; do not interpret receipt')
    summary = {}
    for case in PRIMARY+SECONDARY:
        chosen = [r for r in rows if r['case']==case and 'failure' not in r]
        arms = sorted({a['arm'] for r in chosen for a in r['outcomes']})
        values = {a:[next(x['test_nll'] for x in r['outcomes'] if x['arm']==a) for r in chosen] for a in arms}
        pairs = {}
        for left,right in (('new_bounded','old_diagonal'),('global_outer','old_diagonal'),
                           ('new_bounded','global_outer'),('new_bounded','strict_none'),
                           ('new_gated','clr_nonlinear'),('clr_conditioned_gated','new_gated')):
            if left not in values or right not in values:
                continue
            differences = [a-b for a,b in zip(values[left],values[right])]
            interval = None
            if len(differences)>1:
                m = float(np.mean(differences)); rad = float(stats.t.ppf(.975,len(differences)-1)*stats.sem(differences))
                interval = [m-rad,m+rad]
            pairs[f'{left}_minus_{right}'] = {'per_seed':differences,'n':len(differences),
                'mean':float(np.mean(differences)),'t95_interval':interval,'wins':sum(v<0 for v in differences)}
        summary[case] = {'mean_nll':{a:float(np.mean(v)) for a,v in values.items() if v}, 'paired':pairs}
    result = {'synthetic_only':True, 'worker':args.worker, 'source_sha256':hashes,
        'seeds':SEEDS,'primary':PRIMARY,'secondary_first_ten':SECONDARY,'sizes':SIZES,
        'finite_candidates':FINITE,'gated_candidates':GATED, 'failures':sum('failure' in r for r in rows),
        'runs':rows, 'summary':summary, 'protocol':'METHOD3_1_STABILITY_PREREG_20261117.md'}
    filename = (f'method3_1_stability_worker_{args.worker}_20261117.json' if args.worker is not None
                else 'method3_1_stability_20261117.json')
    path = ROOT/'results'/filename
    with path.open('x', encoding='utf8') as out:
        out.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(f'Wrote {path}; groups={len(rows)}, failures={result["failures"]}',flush=True)


if __name__ == '__main__':
    main()
