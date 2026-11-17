"""Pre-registered, CPU synthetic null controls for CLR conditioning.

Use --smoke for seed 9999 with no output file; --worker 0/1/2 owns a unique
subset of 1401-1410 and a unique result receipt; --merge validates and merges
all workers. No patient paths, data or remote systems are accessed.
"""
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
from method3_1_conditioned_pipeline import CLRConditionedStrictPipeline  # noqa: E402
from method3_1_active_head import ActiveHierarchicalSurvivalHead  # noqa: E402
from method3_1_independent_synthetic import make_independent_splits  # noqa: E402

SEEDS = tuple(range(1401, 1411))
CASES = ('null', 'heteroscedastic_null', 'redundant_nonlinear')
SIZES = {'train': 160, 'validation': 120, 'test': 360}
FINITE = tuple(itertools.product((0., .1), (.1, 1.), (.1, 1.)))
GATED = FINITE + ((0., None, None), (.1, None, None))
# Repetition matches computational candidate count, not independent hypotheses.
CLR_GRID = tuple((v, None, None) for v in (0., .01, .1, 1., 10.))*2
WATCHED = ('method3_1_model.py', 'method3_1_strict_pipeline.py', 'method3_1_interaction_pipeline.py',
           'method3_1_interaction_features.py', 'method3_1_active_head.py', 'method3_1_batched_head.py',
           'method3_1_hierarchical_head.py', 'method3_1_clr_conditioning.py',
           'method3_1_conditioned_pipeline.py', 'method3_1_independent_synthetic.py',
           'method3_1_regularized.py', 'cancer_morphid_model.py', 'morphid_probe.py', 'cancer_parity.py',
           'scripts/method3_1_condition_negative_study.py', 'METHOD3_1_CONDITION_NEGATIVE_PREREG_20261117.md')


def loss(logits, data):
    return float(discrete_nll(logits, data['labels'], data['censorship'], alpha=0.))


def _features_for_test(model, part):
    blocks = model._fresh_blocks(part['bags'], part['rna'], part['ids'], validation=False)
    if isinstance(model, CLRConditionedStrictPipeline):
        blocks = model._apply_conditioner(blocks, part['ids'], stage='test')
    return blocks


def assess(name, model, parts, candidates):
    tr, va, te = (parts[k] for k in ('train', 'validation', 'test'))
    model.head = ActiveHierarchicalSurvivalHead(bins=4, min_select_epoch=0, require_interaction=False)
    selection = model.fit(tr['labels'], tr['censorship'], tr['ids'],
        va['bags'], va['rna'], va['ids'], va['labels'], va['censorship'],
        candidates=candidates, epochs=160, lr=.03, eval_every=5)
    test_blocks = _features_for_test(model, te)
    output = model.head.predict(test_blocks, te['ids'])
    # Actual user-facing prediction must follow exactly the same path.
    if not torch.equal(output, model.predict(te['bags'], te['rna'], te['ids'])):
        raise RuntimeError('Pipeline prediction differs from audited test blocks')
    head = model.head
    row = {'arm': name, 'training_nll': float(selection['train_nll']),
           'validation_nll': float(selection['val_nll']), 'test_nll': loss(output, te),
           'selection': selection, 'dimensions': head.dimensions,
           'nominal_parameters': sum(x.numel() for x in head.weights.values())+head.bias.numel(),
           'candidate_evaluations': len(candidates),
           'unique_candidate_evaluations': len(set(candidates))}
    if head.dimensions['interaction']:
        normed = (test_blocks['interaction']-head.mean['interaction'])/head.scale['interaction']
        contribution = normed @ head.weights['interaction'].T
        norm = float(head.weights['interaction'].norm())
        spread = float(contribution.std(unbiased=False))
        row.update(interaction_weight_norm=norm, interaction_logit_std=spread,
            interaction_selected=bool(selection['epoch'] > 0 and selection['penalties'][2] is not None
                                      and norm > 1e-8 and spread > 1e-8),
            zero_standardized_interaction_test_nll=loss(output-contribution, te),
            # This gradient may be nonzero even when selection hard-disabled a
            # block; in that case it is a counterfactual, not an update audit.
            survival_only_interaction_gradient_norm=head.survival_interaction_gradient_norm(
                model.training_blocks(), tr['labels'], tr['censorship'], tr['ids']),
            survival_gradient_was_trainable=bool(selection['penalties'][2] is not None and selection['epoch'] > 0))
    else:
        row.update(interaction_selected=False, interaction_weight_norm=0., interaction_logit_std=0.,
                   zero_standardized_interaction_test_nll=loss(output, te),
                   survival_only_interaction_gradient_norm=None, survival_gradient_was_trainable=False)
    return row


def run_one(seed, case):
    parts, _, metadata = make_independent_splits(seed, case, n_train=SIZES['train'],
                                                 n_val=SIZES['validation'], n_test=SIZES['test'])
    tr, va, te = (parts[k] for k in ('train', 'validation', 'test'))
    identities = [set(p['ids']) for p in (tr, va, te)]
    if identities[0]&identities[1] or identities[0]&identities[2] or identities[1]&identities[2]:
        raise ValueError('Synthetic patient splits overlap')
    bank = FrozenMorphologyBank.fit(tr['bags'], ids=tr['ids'], clusters=2, latent_dim=3,
        max_fit_patches=2560, per_patient_cap=16, seed=seed)
    base = Method31(bank, rna_dim=2, projection_dim=4, bins=4, seed=seed)
    base.pretrain(tr['bags'], tr['rna'], tr['ids'], epochs=35)
    base.fit_probes(tr['bags'], tr['rna'], tr['ids'], folds=4)
    mapper = GlobalResidualInteractions(base).fit(tr['bags'], tr['rna'], tr['ids'], folds=4)
    ordinary = Method31StrictPipeline(mapper, mode='tanh_outer', expand_clr=True)
    conditioned = CLRConditionedStrictPipeline(mapper, mode='tanh_outer', conditioner_alpha=.01)
    composition = Method31StrictPipeline(mapper, mode='none', expand_clr=True)
    rows = [assess('strict_gated', ordinary, parts, GATED),
            assess('clr_conditioned_gated', conditioned, parts, GATED),
            assess('nonlinear_clr_only', composition, parts, CLR_GRID)]
    x, y = ordinary.training_blocks(), conditioned.training_blocks()
    if not torch.equal(x['clr'], y['clr']) or not torch.equal(x['interaction'], y['interaction']):
        raise RuntimeError('Conditional vs ordinary CLR/interaction training inputs differ')
    if not torch.equal(x['clr'], composition.training_blocks()['clr']):
        raise RuntimeError('Composition comparator CLR training basis differs')
    # The observed outcomes have not been used by this OOF reconstruction.
    audits = conditioned.conditioner.metadata['split_audit']
    if any(set(f['train_ids']) & set(f['validation_ids']) for f in audits):
        raise RuntimeError('Conditional KRR fold overlap')
    return {'seed': seed, 'case': case, 'metadata': metadata,
            'events': {name: int((1-part['censorship']).sum()) for name, part in parts.items()},
            'split_ids_sha256': {name: hashlib.sha256('\n'.join(part['ids']).encode()).hexdigest()
                                  for name, part in parts.items()},
            'conditional_probe': {'config': conditioned.conditioner.config,
                                  'survival_labels_used': False,
                                  'train_estimator': 'fold OOF given shared outer-training representation',
                                  'test_estimator': 'mean fold-model ensemble',
                                  'split_audit': audits}, 'outcomes': rows}


def summarize(rows):
    summary = {}
    for case in CASES:
        present = [r for r in rows if r['case'] == case and 'failure' not in r]
        case_values = {}
        for arm in ('strict_gated', 'clr_conditioned_gated', 'nonlinear_clr_only'):
            entries = [next(o for o in r['outcomes'] if o['arm'] == arm) for r in present]
            case_values[arm] = {'n': len(entries),
                'mean_test_nll': float(np.mean([e['test_nll'] for e in entries])) if entries else None,
                'interaction_selected': sum(e['interaction_selected'] for e in entries)}
        diff = []
        for row in present:
            arms = {o['arm']: o for o in row['outcomes']}
            diff.append(arms['clr_conditioned_gated']['test_nll']-arms['strict_gated']['test_nll'])
        interval = None
        if len(diff) > 1:
            center = float(np.mean(diff)); radius = float(stats.t.ppf(.975, len(diff)-1)*stats.sem(diff))
            interval = [center-radius, center+radius]
        summary[case] = {'arms': case_values, 'paired_conditioned_minus_strict': {
            'per_seed': diff, 'mean': float(np.mean(diff)) if diff else None, 't95_interval': interval}}
    return summary


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--smoke', action='store_true', help='Run only exploratory seed 9999, write no receipt')
    options.add_argument('--worker', type=int, choices=(0, 1, 2))
    options.add_argument('--merge', action='store_true')
    parser.add_argument('--smoke-case', choices=CASES, default=None,
                        help='One smoke scenario; defaults to null, only used with --smoke')
    args = parser.parse_args()
    if args.smoke_case is not None and not args.smoke:
        raise ValueError('--smoke-case is available only with --smoke')
    hashes = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in WATCHED}
    rows = []
    if args.merge:
        for worker in range(3):
            path = ROOT/'results'/f'method3_1_condition_negative_worker_{worker}_20261117.json'
            prior = json.loads(path.read_text(encoding='utf8'))
            if prior['worker'] != worker or prior['source_sha256'] != hashes:
                raise ValueError(f'Incompatible worker receipt: {path}')
            rows.extend(prior['runs'])
        expected = {(seed, case) for seed in SEEDS for case in CASES}
        actual = [(row['seed'], row['case']) for row in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError('Worker results contain missing, extra or duplicate groups')
        rows.sort(key=lambda r: (r['seed'], CASES.index(r['case'])))
    else:
        seeds = (9999,) if args.smoke else tuple(s for s in SEEDS if (s-SEEDS[0]) % 3 == args.worker)
        partial = (None if args.smoke else
                   ROOT/'results'/f'method3_1_condition_negative_worker_{args.worker}_partial_20261117.json')
        if partial is not None and partial.exists():
            raise FileExistsError(f'Will not replace another partial receipt: {partial}')
        run_cases = ((args.smoke_case or 'null',) if args.smoke else CASES)
        for seed in seeds:
            for case in run_cases:
                try:
                    row = run_one(seed, case)
                    by_arm = {x['arm']: x['test_nll'] for x in row['outcomes']}
                    print(f"{seed} {case}: conditioned={by_arm['clr_conditioned_gated']:.4f}, "
                          f"strict={by_arm['strict_gated']:.4f}, CLR={by_arm['nonlinear_clr_only']:.4f}", flush=True)
                except Exception:
                    row = {'seed': seed, 'case': case, 'failure': traceback.format_exc()}
                    print(f"{seed} {case} FAILED {row['failure']}", flush=True)
                rows.append(row)
                if partial is not None:
                    partial.write_text(json.dumps({'worker': args.worker, 'source_sha256': hashes, 'runs': rows},
                                                  ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
                gc.collect()
    if any(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected for path, expected in hashes.items()):
        raise RuntimeError('A watched source changed during execution; do not interpret results')
    result = {'synthetic_only': True, 'worker': args.worker, 'seeds': SEEDS, 'cases': CASES,
              'sizes': SIZES, 'source_sha256': hashes, 'gated_candidates': GATED,
              'clr_candidates': CLR_GRID, 'failures': sum('failure' in r for r in rows),
              'runs': rows, 'summary': summarize(rows),
              'protocol': 'METHOD3_1_CONDITION_NEGATIVE_PREREG_20261117.md',
              'limitations': ['Only synthetic data; no patient data read.',
                 'Conditional OOF is conditional on the shared outer-training representation; test uses a fold ensemble.',
                 'A gradient for a hard-disabled interaction is a counterfactual, not proof it was trained.',
                 'The standardized-block zeroing diagnostic is not a causal intervention on a patient modality.',
                 'Five unique CLR-only penalties are repeated for computational matching; choice complexity differs.']}
    if args.smoke:
        print(f'Smoke only; seed=9999, groups={len(rows)}, failures={result["failures"]}; no receipt written.', flush=True)
        if result['failures']:
            raise RuntimeError('Synthetic smoke failed; see traceback above')
        return
    filename = (f'method3_1_condition_negative_worker_{args.worker}_20261117.json' if args.worker is not None
                else 'method3_1_condition_negative_20261117.json')
    path = ROOT/'results'/filename
    with path.open('x', encoding='utf8') as out:
        out.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(f'Wrote {path}; groups={len(rows)}, failures={result["failures"]}', flush=True)


if __name__ == '__main__':
    main()
