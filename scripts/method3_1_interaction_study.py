"""Pre-registered method 3.1 synthetic interaction and composition study."""
import gc
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cancer_morphid_model import FrozenMorphologyBank, discrete_nll  # noqa: E402
from method3_1_model import Method31  # noqa: E402
from method3_1_regularized import RegularizedMethod31, training_hazard_logits  # noqa: E402
from method3_1_interaction_features import GlobalResidualInteractions  # noqa: E402
from method3_1_interaction_pipeline import Method31InteractionPipeline  # noqa: E402
from method3_1_batched_head import BatchedHierarchicalSurvivalHead  # noqa: E402
from method3_1_interaction_data import make_interaction_splits  # noqa: E402
from method3_1_synthetic_data import make_composition_splits  # noqa: E402

SEEDS = (401, 409, 419, 421, 431)
CASES = ('xor', 'shared', 'additive', 'rna_only', 'path_only', 'null',
         'redundant_nonlinear', 'composition_signal', 'composition_null')
ARM_MODES = (('none', 'none', 'per_bin'), ('diagonal', 'diagonal', 'per_bin'),
             ('outer', 'outer', 'per_bin'), ('tanh_outer', 'tanh_outer', 'per_bin'),
             ('raw_outer', 'raw_outer', 'per_bin'), ('tanh_shared', 'tanh_outer', 'shared'))
SIZES = {'train': 160, 'validation': 120, 'test': 360}
GRID = tuple(itertools.product((0., .1), (.1, 1.), (.1, 1.))) + ((0., None, None), (.1, None, None))
CLR_GRID = tuple((x, None, None) for x in (0., .01, .1, 1., 10.))


def loss(logits, split):
    return float(discrete_nll(logits, split['labels'], split['censorship'], alpha=0.))


def generate(case, seed):
    kwargs = {'n_train': SIZES['train'], 'n_val': SIZES['validation'], 'n_test': SIZES['test']}
    if case.startswith('composition_'):
        return make_composition_splits(seed, composition_effect=2.5 if case == 'composition_signal' else 0., **kwargs)
    return make_interaction_splits(seed, case, **kwargs)


def assess_arm(name, pipeline, tr, va, te, *, candidates):
    selection = pipeline.fit(tr['labels'], tr['censorship'], tr['ids'],
        va['bags'], va['rna'], va['ids'], va['labels'], va['censorship'], candidates=candidates)
    output = pipeline.predict(te['bags'], te['rna'], te['ids'])
    return {'arm': name, 'train_nll': selection['train_nll'], 'validation_nll': selection['val_nll'],
            'test_nll': loss(output, te), 'selection': selection,
            'head_dimensions': pipeline.head.dimensions,
            'head_nominal_parameter_count': sum(v.numel() for v in pipeline.head.weights.values())+pipeline.head.bias.numel(),
            'interaction_weight_norm': float(pipeline.head.weights['interaction'].norm()),
            'interaction_mode': pipeline.head.interaction_mode}


def run_one(case, seed):
    data, _, meta = generate(case, seed)
    tr, va, te = (data[x] for x in ('train', 'validation', 'test'))
    sets = [set(x['ids']) for x in (tr, va, te)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Synthetic split IDs overlap')
    bank = FrozenMorphologyBank.fit(tr['bags'], ids=tr['ids'], clusters=2, latent_dim=3,
        max_fit_patches=2560, per_patient_cap=16, seed=seed)
    base = Method31(bank, rna_dim=2, projection_dim=4, bins=4, seed=seed)
    base.pretrain(tr['bags'], tr['rna'], tr['ids'], epochs=35)
    base.fit_probes(tr['bags'], tr['rna'], tr['ids'], folds=4)
    mapper = GlobalResidualInteractions(base).fit(tr['bags'], tr['rna'], tr['ids'], folds=4)
    results, reference_blocks = [], None
    for name, mode, interaction_mode in ARM_MODES:
        model = Method31InteractionPipeline(mapper, mode=mode, interaction_mode=interaction_mode)
        model.head = BatchedHierarchicalSurvivalHead(base.bins, interaction_mode=interaction_mode)
        result = assess_arm(name, model, tr, va, te, candidates=GRID)
        blocks = model.training_blocks()
        if reference_blocks is None:
            reference_blocks = {'clr': blocks['clr'], 'other': blocks['other']}
        elif not all(torch.equal(blocks[k], reference_blocks[k]) for k in ('clr', 'other')):
            raise RuntimeError('Interaction comparison arms do not have identical main inputs')
        if mode == 'none' and blocks['interaction'].shape[1] != 0:
            raise RuntimeError('No-interaction arm has an interaction input')
        results.append(result)
    clr = Method31InteractionPipeline(mapper, mode='none')
    clr.head = BatchedHierarchicalSurvivalHead(base.bins)
    results.append(assess_arm('nature_clr', clr, tr, va, te, candidates=CLR_GRID))
    previous = RegularizedMethod31(base, arm='full')
    selection = previous.fit(tr['labels'], tr['censorship'], tr['ids'],
        va['bags'], va['rna'], va['ids'], va['labels'], va['censorship'])
    results.append({'arm': 'previous_full', 'train_nll': loss(previous.train_logits(), tr),
        'validation_nll': selection['validation_nll'], 'test_nll': loss(previous.predict(te['bags'],te['rna'],te['ids']),te),
        'selection': selection, 'head_nominal_parameter_count': previous.weight.numel()+previous.bias.numel()})
    intercept = training_hazard_logits(tr['labels'], tr['censorship'], 4)
    results.append({'arm': 'intercept', 'train_nll': loss(intercept.expand(len(tr['ids']), -1), tr),
        'validation_nll': loss(intercept.expand(len(va['ids']), -1), va),
        'test_nll': loss(intercept.expand(len(te['ids']), -1), te),
        'selection': {'rule': 'training smoothed bin hazards'}, 'head_nominal_parameter_count': 4})
    return {'case': case, 'seed': seed, 'metadata': meta,
            'events': {name: int((1-data[name]['censorship']).sum()) for name in data},
            'split_ids_sha256': {name: hashlib.sha256('\n'.join(data[name]['ids']).encode()).hexdigest() for name in data},
            'global_probe': {'folds': len(mapper.probe.models), 'split_audit': mapper.metadata['split_audit'],
                             'condition': 'OOF conditional on shared outer-training bank and SSL'},
            'outcomes': results}


def paired_summary(rows, left, right):
    diffs = []
    for row in rows:
        values = {x['arm']: x['test_nll'] for x in row['outcomes']}
        diffs.append(values[left]-values[right])
    if len(diffs) < 2:
        return {'n': len(diffs), 'per_seed': diffs, 'mean': None, 't95_interval': None}
    mean = float(np.mean(diffs))
    radius = float(stats.t.ppf(.975, len(diffs)-1)*stats.sem(diffs))
    return {'n': len(diffs), 'per_seed': diffs, 'mean': mean,
            't95_interval': [mean-radius, mean+radius]}


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS, help='Run one fixed seed to a separate receipt')
    parser.add_argument('--merge', action='store_true', help='Merge all five unchanged seed receipts')
    args = parser.parse_args()
    if args.seed is not None and args.merge:
        raise ValueError('Use --seed or --merge, not both')
    watched = ('method3_1_model.py', 'method3_1_regularized.py', 'method3_1_hierarchical_head.py',
        'method3_1_interaction_features.py', 'method3_1_interaction_pipeline.py', 'method3_1_batched_head.py',
        'method3_1_interaction_data.py', 'method3_1_synthetic_data.py', 'cancer_morphid_model.py',
        'morphid_probe.py', 'cancer_parity.py', 'scripts/method3_1_interaction_study.py',
        'METHOD3_1_INTERACTION_PREREG_20261117.md')
    hashes = {x: hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in watched}
    selected_seeds = (args.seed,) if args.seed is not None else SEEDS
    rows = []
    if args.merge:
        for seed in SEEDS:
            path = ROOT/'results'/f'method3_1_interaction_seed_{seed}_20261117.json'
            saved = json.loads(path.read_text(encoding='utf8'))
            if (saved['source_sha256'] != hashes or saved['seeds'] != [seed]
                    or [r['case'] for r in saved['runs']] != list(CASES)
                    or any(r['seed'] != seed for r in saved['runs'])):
                raise ValueError(f'Incompatible or incomplete fixed-seed receipt: {path}')
            rows.extend(saved['runs'])
    else:
        for seed in selected_seeds:
            for case in CASES:
                try:
                    row = run_one(case, seed)
                    values = {x['arm']: x['test_nll'] for x in row['outcomes']}
                    print(f"{seed} {case}: none={values['none']:.4f}, tanh={values['tanh_outer']:.4f}, "
                          f"clr={values['nature_clr']:.4f}, null={values['intercept']:.4f}", flush=True)
                except Exception:
                    row = {'case': case, 'seed': seed, 'failure': traceback.format_exc()}
                    print(f"{seed} {case}: FAILED {row['failure']}", flush=True)
                rows.append(row)
                gc.collect()
    if any(hashlib.sha256((ROOT/x).read_bytes()).hexdigest() != v for x, v in hashes.items()):
        raise RuntimeError('Source changed during the run; output is invalid')
    pairs = (('tanh_outer','none'),('outer','none'),('diagonal','none'),('raw_outer','outer'),
             ('tanh_outer','diagonal'),('tanh_shared','tanh_outer'),('tanh_outer','nature_clr'),
             ('none','nature_clr'),('tanh_outer','previous_full'),('tanh_outer','intercept'))
    summary = {}
    for case in CASES:
        selected = [x for x in rows if x['case']==case and 'failure' not in x]
        values = {name: [next(a['test_nll'] for a in row['outcomes'] if a['arm']==name) for row in selected]
                  for name in [x[0] for x in ARM_MODES]+['nature_clr','previous_full','intercept']}
        summary[case] = {'mean_test_nll': {k: (float(np.mean(v)) if v else None) for k,v in values.items()},
            'paired': {f'{a}_minus_{b}': paired_summary(selected, a, b) for a,b in pairs}}
    result = {'synthetic_only': True, 'seeds': selected_seeds, 'cases': CASES, 'split_sizes': SIZES,
        'penalty_candidates': GRID, 'clr_penalty_candidates': CLR_GRID, 'epochs': 160, 'eval_every': 5,
        'source_sha256': hashes, 'protocol': 'METHOD3_1_INTERACTION_PREREG_20261117.md',
        'failures': sum('failure' in row for row in rows), 'summary': summary, 'runs': rows,
        'limitations': ['Synthetic evidence only; no patient data were opened.',
            'Global residual OOF is conditional on shared outer-training morphology/SSL basis.',
            'Interaction is a predictive function, not PID, independence, or causal interaction proof.',
            'Validation selects penalty/epoch among several candidates; held-out tests are necessary.',
            'Shared-bin coefficients impose a different inductive bias and have fewer parameters.']}
    filename = (f'method3_1_interaction_seed_{args.seed}_20261117.json' if args.seed is not None
                else 'method3_1_interaction_study_20261117.json')
    path = ROOT/'results'/filename
    with path.open('x', encoding='utf8') as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(f'Wrote {path}; failures={result["failures"]}', flush=True)


if __name__ == '__main__':
    main()
