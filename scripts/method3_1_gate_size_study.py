"""CPU-only pre-registered test of independent gate sample-size tradeoffs."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy import stats
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cancer_morphid_model import FrozenMorphologyBank  # noqa:E402
from method3_1_model import Method31  # noqa:E402
from method3_1_interaction_features import GlobalResidualInteractions  # noqa:E402
from method3_1_active_head import ActiveHierarchicalSurvivalHead  # noqa:E402
from method3_1_guarded_pipeline import GuardedMethod31Pipeline  # noqa:E402
from method3_1_holdout_gate import evaluate_holdout_gate  # noqa:E402
from method3_1_independent_synthetic import make_independent_splits  # noqa:E402
from scripts.method3_1_gate_study import (_slice,_tuple,_selected,_candidate_predict,_row,
                                          OLD_GRID,CORRECTED_GRID,CLR_GRID)  # noqa:E402

SEEDS=tuple(range(1801,1821))
PRIMARY=('xor','null','heteroscedastic_null','composition_only','composition_plus_rna')
SECONDARY=('redundant_nonlinear','composition_interaction')
SIZES={'train':160,'selection':120,'gate':480,'test':360}
WATCHED=('method3_1_model.py','method3_1_strict_pipeline.py','method3_1_interaction_pipeline.py',
 'method3_1_interaction_features.py','method3_1_active_head.py','method3_1_batched_head.py',
 'method3_1_hierarchical_head.py','method3_1_holdout_gate.py','method3_1_guarded_pipeline.py',
 'method3_1_independent_synthetic.py','method3_1_regularized.py','cancer_morphid_model.py',
 'morphid_probe.py','cancer_parity.py','scripts/method3_1_gate_study.py',
 'scripts/method3_1_gate_size_study.py','METHOD3_1_GATE_SIZE_PREREG_20261117.md')


def run_one(seed,case):
    data,_,metadata=make_independent_splits(seed,case,n_train=160,n_val=600,n_test=360)
    tr,val,te=data['train'],data['validation'],data['test']
    sel,gate=_slice(val,0,120),_slice(val,120,600)
    groups=[set(x['ids']) for x in (tr,sel,gate,te)]
    if any(groups[i]&groups[j] for i in range(4) for j in range(i)):
        raise RuntimeError('Synthetic patient splits overlap')
    bank=FrozenMorphologyBank.fit(tr['bags'],ids=tr['ids'],clusters=2,latent_dim=3,
                                   max_fit_patches=2560,per_patient_cap=16,seed=seed)
    base=Method31(bank,rna_dim=2,projection_dim=4,bins=4,seed=seed)
    base.pretrain(tr['bags'],tr['rna'],tr['ids'],epochs=35)
    base.fit_probes(tr['bags'],tr['rna'],tr['ids'],folds=4)
    mapper=GlobalResidualInteractions(base).fit(tr['bags'],tr['rna'],tr['ids'],folds=4)
    guarded=GuardedMethod31Pipeline(mapper,mode='tanh_outer')
    result=guarded.fit(tr['labels'],tr['censorship'],tr['ids'],selection=_tuple(sel),gate=_tuple(gate),
                       test_ids=te['ids'],retain_candidates=True)
    off,on=guarded._diagnostic_heads
    vb=guarded.builder._fresh_blocks(val['bags'],val['rna'],val['ids'],validation=True)
    tb=guarded.builder._fresh_blocks(te['bags'],te['rna'],te['ids'],validation=False)
    xb=guarded.builder._train_blocks(mapper.training_features())
    gate_blocks={k:v[120:] for k,v in vb.items()}
    gate_off,gate_on=off.predict(gate_blocks,gate['ids']),on.predict(gate_blocks,gate['ids'])
    test_off,test_on=off.predict(tb,te['ids']),on.predict(tb,te['ids'])
    decisions={}
    for n in (120,240,480):
        decisions[str(n)]=evaluate_holdout_gate(gate_off[:n],gate_on[:n],gate['labels'][:n],
            gate['censorship'][:n],gate['ids'][:n],train_ids=tr['ids'],validation_ids=sel['ids'],
            test_ids=te['ids'],alpha=.05,min_gain=.005)
    if decisions['480']['enabled']!=result['gate']['enabled']:
        raise RuntimeError('Production gate differs from independent replay')
    actual=guarded.predict(te['bags'],te['rna'],te['ids'])
    if not torch.equal(actual,test_on if decisions['480']['enabled'] else test_off):
        raise RuntimeError('Production pipeline did not use its selected frozen head')
    joint=ActiveHierarchicalSurvivalHead(bins=4,min_select_epoch=0)
    joint_result=joint.fit(xb,tr['labels'],tr['censorship'],tr['ids'],vb,val['labels'],val['censorship'],val['ids'],
                           candidates=CORRECTED_GRID)
    clr=ActiveHierarchicalSurvivalHead(bins=4,min_select_epoch=0)
    clr_result=clr.fit(xb,tr['labels'],tr['censorship'],tr['ids'],vb,val['labels'],val['censorship'],val['ids'],
                       candidates=CLR_GRID)
    old_index=min(range(len(OLD_GRID)),key=lambda i:joint_result['candidates'][i]['val_nll'])
    rows=[_row('legacy600',joint,joint.candidate_states[old_index],tb,te,joint_result['candidates'][old_index]),
          _row('corrected600',joint,_selected(joint),tb,te,joint_result),
          _row('nature_clr600',clr,_selected(clr),tb,te,clr_result),
          _row('always_on120',on,_selected(on),tb,te,result['on_selection'],test_on),
          _row('always_off120',off,_selected(off),tb,te,result['off_selection'],test_off)]
    for n in (120,240,480):
        selected=on if decisions[str(n)]['enabled'] else off
        pred=test_on if decisions[str(n)]['enabled'] else test_off
        rows.append(_row(f'gate{n}',selected,_selected(selected),tb,te,decisions[str(n)],pred))
    for row in rows:
        row['interaction_active']=bool(row['interaction_weight_norm']>1e-8 and
                                        max(row['interaction_patient_axis_std_by_bin'])>1e-8)
    return {'seed':seed,'case':case,'metadata':metadata,'decisions':decisions,
        'events':{k:int((1-p['censorship']).sum()) for k,p in (('train',tr),('selection',sel),('gate',gate),('test',te))},
        'split_ids_sha256':{k:hashlib.sha256('\n'.join(p['ids']).encode()).hexdigest()
            for k,p in (('train',tr),('selection',sel),('gate',gate),('test',te))},
        'on_selection':result['on_selection'],'off_selection':result['off_selection'],
        'gate_outcomes_used_for_refit':result['gate_outcomes_used_for_refit'],'outcomes':rows}


def summarize(rows):
    out={}
    arms=('gate120','gate240','gate480','legacy600','corrected600','nature_clr600','always_on120','always_off120')
    for case in PRIMARY+SECONDARY:
        present=[r for r in rows if r['case']==case and 'failure' not in r]
        values={a:[next(x for x in r['outcomes'] if x['arm']==a) for r in present] for a in arms}
        means={a:{'n':len(v),'mean_nll':float(np.mean([x['test_nll'] for x in v])) if v else None,
                   'interaction_active':sum(x['interaction_active'] for x in v)} for a,v in values.items()}
        pairs={}
        for a,b in (('gate480','always_on120'),('gate480','corrected600'),('gate480','always_off120'),
                    ('gate120','always_on120'),('gate240','always_on120'),('corrected600','legacy600')):
            diffs=[x['test_nll']-y['test_nll'] for x,y in zip(values[a],values[b])]
            ci=None
            if len(diffs)>1:
                m=float(np.mean(diffs)); rad=float(stats.t.ppf(.975,len(diffs)-1)*stats.sem(diffs)); ci=[m-rad,m+rad]
            pairs[a+'_minus_'+b]={'n':len(diffs),'mean':float(np.mean(diffs)) if diffs else None,
                                   'per_seed':diffs,'t95_interval':ci}
        out[case]={'arms':means,'paired':pairs}
    return out


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    options=parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--worker',type=int,choices=(0,1,2)); options.add_argument('--merge',action='store_true')
    args=parser.parse_args()
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in WATCHED}
    rows=[]
    if args.merge:
        for worker in range(3):
            path=ROOT/'results'/f'method3_1_gate_size_worker_{worker}_20261117.json'
            data=json.loads(path.read_text(encoding='utf8'))
            if data['worker']!=worker or data['source_sha256']!=hashes:
                raise ValueError(f'Incompatible receipt: {path}')
            rows.extend(data['runs'])
        expected={(s,c) for s in SEEDS for c in PRIMARY+(SECONDARY if s<=1810 else ())}
        actual=[(r['seed'],r['case']) for r in rows]
        if len(actual)!=len(expected) or set(actual)!=expected:
            raise ValueError('Missing, extra or duplicate groups')
        rows.sort(key=lambda r:(r['seed'],(PRIMARY+SECONDARY).index(r['case'])))
    else:
        partial=ROOT/'results'/f'method3_1_gate_size_worker_{args.worker}_partial_20261117.json'
        if partial.exists():
            raise FileExistsError('Refusing to replace an existing partial receipt')
        for seed in SEEDS:
            if (seed-SEEDS[0])%3!=args.worker:
                continue
            for case in PRIMARY+(SECONDARY if seed<=1810 else ()):
                try:
                    row=run_one(seed,case)
                    print(f"{seed} {case}: gates120/240/480="+
                          '/'.join(str(row['decisions'][str(n)]['enabled']) for n in (120,240,480)),flush=True)
                except Exception:
                    row={'seed':seed,'case':case,'failure':traceback.format_exc()}
                    print(f"{seed} {case} FAILED {row['failure']}",flush=True)
                rows.append(row)
                partial.write_text(json.dumps({'worker':args.worker,'source_sha256':hashes,'runs':rows},
                    ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
                gc.collect()
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise RuntimeError('Source changed during study; do not interpret')
    result={'synthetic_only':True,'worker':args.worker,'source_sha256':hashes,'seeds':SEEDS,
            'primary':PRIMARY,'secondary_first_ten':SECONDARY,'sizes':SIZES,
            'failures':sum('failure' in r for r in rows),'runs':rows,'summary':summarize(rows),
            'protocol':'METHOD3_1_GATE_SIZE_PREREG_20261117.md'}
    filename=(f'method3_1_gate_size_worker_{args.worker}_20261117.json' if args.worker is not None
              else 'method3_1_gate_size_study_20261117.json')
    path=ROOT/'results'/filename
    with path.open('x',encoding='utf8') as output:
        output.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(f'Wrote {path}; groups={len(rows)} failures={result["failures"]}',flush=True)


if __name__=='__main__':
    main()
