"""Pre-registered CPU-only independent-gate survival mechanism study."""
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
from method3_1_active_head import ActiveHierarchicalSurvivalHead  # noqa: E402
from method3_1_guarded_pipeline import GuardedMethod31Pipeline, OFF_CANDIDATES, ON_CANDIDATES  # noqa: E402
from method3_1_holdout_gate import evaluate_holdout_gate  # noqa: E402
from method3_1_independent_synthetic import make_independent_splits  # noqa: E402

SEEDS = tuple(range(1601, 1621))
PRIMARY = ('xor', 'null', 'heteroscedastic_null', 'composition_only', 'composition_plus_rna')
SECONDARY = ('redundant_nonlinear', 'additive', 'composition_interaction')
SIZES = {'train':160, 'selection':120, 'gate':120, 'test':360}
CLR = ((0.,None,None),(.1,None,None))
OLD_GRID = ON_CANDIDATES + CLR
CORRECTED_GRID = OLD_GRID + tuple((a,b,None) for a,b in itertools.product((0.,.1),(.1,1.)))
CLR_GRID = tuple((v,None,None) for v in (0.,.01,.1,1.,10.))
WATCHED = ('method3_1_model.py','method3_1_strict_pipeline.py','method3_1_interaction_pipeline.py',
 'method3_1_interaction_features.py','method3_1_active_head.py','method3_1_batched_head.py',
 'method3_1_hierarchical_head.py','method3_1_holdout_gate.py','method3_1_guarded_pipeline.py',
 'method3_1_independent_synthetic.py','method3_1_regularized.py','cancer_morphid_model.py',
 'morphid_probe.py','cancer_parity.py','scripts/method3_1_gate_study.py','METHOD3_1_GATE_PREREG_20261117.md')


def _slice(part, start, end):
    return {k:v[start:end] for k,v in part.items()}


def _tuple(part):
    return part['bags'], part['rna'], part['ids'], part['labels'], part['censorship']


def _loss(logits, part):
    return discrete_nll(logits, part['labels'], part['censorship'], alpha=0., reduction='none')


def _candidate_predict(head, state, blocks):
    x = head._normalized(blocks)
    return (x['clr']@state['weights']['clr'].T + x['other']@state['weights']['other'].T
            + x['interaction']@state['weights']['interaction'].T + state['bias'])


def _row(name, head, state, test_blocks, part, selection, prediction=None):
    logits = _candidate_predict(head, state, test_blocks) if prediction is None else prediction
    normalized = head._normalized(test_blocks)['interaction']
    contribution = normalized@state['weights']['interaction'].T
    loss = _loss(logits, part)
    return {'arm':name, 'test_nll':float(loss.mean()), 'test_patientwise_nll':loss.tolist(),
            'interaction_weight_norm':float(state['weights']['interaction'].norm()),
            'interaction_patient_axis_std_by_bin':contribution.std(0,unbiased=False).tolist(),
            'zero_interaction_test_nll':float(_loss(logits-contribution,part).mean()),
            'selection':selection}


def _selected(head):
    return {'weights':{k:v.clone() for k,v in head.weights.items()},'bias':head.bias.clone()}


def run_one(seed, case):
    data, _, metadata = make_independent_splits(seed, case, n_train=160, n_val=240, n_test=360)
    tr, val, te = data['train'], data['validation'], data['test']
    sel, gate = _slice(val,0,120), _slice(val,120,240)
    ids = [set(x['ids']) for x in (tr,sel,gate,te)]
    if any(ids[i]&ids[j] for i in range(4) for j in range(i)):
        raise RuntimeError('Synthetic split IDs overlap')
    bank = FrozenMorphologyBank.fit(tr['bags'], ids=tr['ids'], clusters=2, latent_dim=3,
        max_fit_patches=2560, per_patient_cap=16, seed=seed)
    base = Method31(bank,rna_dim=2,projection_dim=4,bins=4,seed=seed)
    base.pretrain(tr['bags'],tr['rna'],tr['ids'],epochs=35)
    base.fit_probes(tr['bags'],tr['rna'],tr['ids'],folds=4)
    mapper = GlobalResidualInteractions(base).fit(tr['bags'],tr['rna'],tr['ids'],folds=4)
    guarded = GuardedMethod31Pipeline(mapper,mode='tanh_outer')
    fitted = guarded.fit(tr['labels'],tr['censorship'],tr['ids'],selection=_tuple(sel),gate=_tuple(gate),
                         test_ids=te['ids'],retain_candidates=True)
    off,on = guarded._diagnostic_heads
    pooled_blocks = guarded.builder._fresh_blocks(val['bags'],val['rna'],val['ids'],validation=True)
    test_blocks = guarded.builder._fresh_blocks(te['bags'],te['rna'],te['ids'],validation=False)
    train_blocks = guarded.builder._train_blocks(mapper.training_features())
    gate_blocks = {k:v[120:] for k,v in pooled_blocks.items()}
    # A single batched fit runs exactly the old and newly added candidates;
    # their independent Adam trajectories are checked by the existing tests.
    joint = ActiveHierarchicalSurvivalHead(bins=4,min_select_epoch=0)
    joint_result = joint.fit(train_blocks,tr['labels'],tr['censorship'],tr['ids'],
        pooled_blocks,val['labels'],val['censorship'],val['ids'],candidates=CORRECTED_GRID)
    clr_head = ActiveHierarchicalSurvivalHead(bins=4,min_select_epoch=0)
    clr_result = clr_head.fit(train_blocks,tr['labels'],tr['censorship'],tr['ids'],
        pooled_blocks,val['labels'],val['censorship'],val['ids'],candidates=CLR_GRID)
    old_idx = min(range(len(OLD_GRID)),key=lambda i:joint_result['candidates'][i]['val_nll'])
    old_state = joint.candidate_states[old_idx]
    old_sel = joint_result['candidates'][old_idx]
    test_off, test_on = off.predict(test_blocks,te['ids']),on.predict(test_blocks,te['ids'])
    gate_off, gate_on = off.predict(gate_blocks,gate['ids']),on.predict(gate_blocks,gate['ids'])
    gate_kw = dict(train_ids=tr['ids'],validation_ids=sel['ids'],test_ids=te['ids'])
    loose = evaluate_holdout_gate(gate_off,gate_on,gate['labels'],gate['censorship'],gate['ids'],
                                   alpha=.20,min_gain=0.,**gate_kw)
    direct = evaluate_holdout_gate(gate_off,gate_on,gate['labels'],gate['censorship'],gate['ids'],
                                    alpha=.50,min_gain=0.,**gate_kw)
    expected = test_on if fitted['gate']['enabled'] else test_off
    observed = guarded.predict(te['bags'],te['rna'],te['ids'])
    if not torch.equal(expected,observed):
        raise RuntimeError('Production guarded prediction differs from selected frozen candidate')
    rows = [
        _row('legacy240',joint,old_state,test_blocks,te,old_sel),
        _row('corrected240',joint,_selected(joint),test_blocks,te,joint_result),
        _row('strict_gate120',guarded.head,_selected(guarded.head),test_blocks,te,fitted['gate'],observed),
        _row('loose_gate120',on if loose['enabled'] else off,_selected(on if loose['enabled'] else off),
             test_blocks,te,loose,test_on if loose['enabled'] else test_off),
        _row('direct_gate120',on if direct['enabled'] else off,_selected(on if direct['enabled'] else off),
             test_blocks,te,direct,test_on if direct['enabled'] else test_off),
        _row('always_on120',on,_selected(on),test_blocks,te,fitted['on_selection'],test_on),
        _row('always_off120',off,_selected(off),test_blocks,te,fitted['off_selection'],test_off),
        _row('nature_clr240',clr_head,_selected(clr_head),test_blocks,te,clr_result)]
    for row in rows:
        row['interaction_active'] = bool(row['interaction_weight_norm']>1e-8 and
                                          max(row['interaction_patient_axis_std_by_bin'])>1e-8)
    return {'seed':seed,'case':case,'metadata':metadata,'events':{k:int((1-p['censorship']).sum())
            for k,p in (('train',tr),('selection',sel),('gate',gate),('test',te))},
            'split_ids_sha256':{k:hashlib.sha256('\n'.join(p['ids']).encode()).hexdigest()
            for k,p in (('train',tr),('selection',sel),('gate',gate),('test',te))},
            'decision':{'strict':fitted['gate'],'loose':loose,'direct':direct},
            'on_selection':fitted['on_selection'],'off_selection':fitted['off_selection'],
            'gate_outcomes_used_for_refit':fitted['gate_outcomes_used_for_refit'],'outcomes':rows}


def _summary(rows):
    result = {}
    for case in PRIMARY+SECONDARY:
        group = [r for r in rows if r['case']==case and 'failure' not in r]
        arms = ('legacy240','corrected240','strict_gate120','loose_gate120','direct_gate120',
                'always_on120','always_off120','nature_clr240')
        values = {name:[next(o for o in row['outcomes'] if o['arm']==name) for row in group] for name in arms}
        summaries = {name:{'n':len(v),'mean_nll':float(np.mean([x['test_nll'] for x in v])) if v else None,
            'interaction_active':sum(x['interaction_active'] for x in v)} for name,v in values.items()}
        pairs = {}
        for left,right in (('strict_gate120','legacy240'),('strict_gate120','corrected240'),
                           ('strict_gate120','always_on120'),('strict_gate120','always_off120'),
                           ('corrected240','legacy240'),('loose_gate120','strict_gate120'),
                           ('direct_gate120','strict_gate120')):
            diff = [a['test_nll']-b['test_nll'] for a,b in zip(values[left],values[right])]
            ci = None
            if len(diff)>1:
                mean = float(np.mean(diff)); radius = float(stats.t.ppf(.975,len(diff)-1)*stats.sem(diff))
                ci = [mean-radius,mean+radius]
            pairs[left+'_minus_'+right] = {'n':len(diff),'mean':float(np.mean(diff)) if diff else None,
                'per_seed':diff,'t95_interval':ci}
        result[case] = {'arms':summaries,'paired':pairs}
    return result


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=int,choices=(0,1,2))
    parser.add_argument('--merge',action='store_true')
    args = parser.parse_args()
    if (args.worker is None)==(not args.merge):
        raise ValueError('Specify exactly one of --worker or --merge')
    hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in WATCHED}
    rows = []
    if args.merge:
        for worker in range(3):
            path = ROOT/'results'/f'method3_1_gate_worker_{worker}_20261117.json'
            data = json.loads(path.read_text(encoding='utf8'))
            if data['worker']!=worker or data['source_sha256']!=hashes:
                raise ValueError(f'Incompatible receipt {path}')
            rows.extend(data['runs'])
        expected = {(s,c) for s in SEEDS for c in PRIMARY+(SECONDARY if s<=1610 else ())}
        actual = [(r['seed'],r['case']) for r in rows]
        if len(actual)!=len(expected) or set(actual)!=expected:
            raise ValueError('Missing, extra or duplicate groups')
        rows.sort(key=lambda r:(r['seed'],(PRIMARY+SECONDARY).index(r['case'])))
    else:
        partial = ROOT/'results'/f'method3_1_gate_worker_{args.worker}_partial_20261117.json'
        if partial.exists():
            raise FileExistsError('Refusing to overwrite a partial receipt')
        for seed in SEEDS:
            if (seed-SEEDS[0])%3 != args.worker:
                continue
            for case in PRIMARY+(SECONDARY if seed<=1610 else ()):
                try:
                    row = run_one(seed,case)
                    print(f"{seed} {case} strict={row['decision']['strict']['enabled']} "
                          f"gain={row['decision']['strict']['mean_gain']}",flush=True)
                except Exception:
                    row={'seed':seed,'case':case,'failure':traceback.format_exc()}
                    print(f"{seed} {case} FAILED {row['failure']}",flush=True)
                rows.append(row)
                partial.write_text(json.dumps({'worker':args.worker,'source_sha256':hashes,'runs':rows},
                    ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
                gc.collect()
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise RuntimeError('Source changed; do not interpret this run')
    output={'synthetic_only':True,'worker':args.worker,'source_sha256':hashes,'seeds':SEEDS,
            'primary':PRIMARY,'secondary_first_ten':SECONDARY,'sizes':SIZES,
            'failures':sum('failure' in r for r in rows),'runs':rows,'summary':_summary(rows),
            'protocol':'METHOD3_1_GATE_PREREG_20261117.md'}
    filename = (f'method3_1_gate_worker_{args.worker}_20261117.json' if args.worker is not None
                else 'method3_1_gate_study_20261117.json')
    path=ROOT/'results'/filename
    with path.open('x',encoding='utf8') as handle:
        handle.write(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(f'Wrote {path}; groups={len(rows)} failures={output["failures"]}',flush=True)


if __name__=='__main__':
    main()
