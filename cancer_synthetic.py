"""Generate simulated WSI/RNA survival data and compare fixed control arms.

This script never reads real patients; model selection uses synthetic validation.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import numpy as np
import torch
from cancer_model import piecewise_nll, survival_at
from cancer_train import CancerConfig, train_cancer_model, predict, view_batch
from model import stability_weighted_common, DegenerateCorrelation


def generate(seed, scenario, n=360, patches=12):
    if scenario not in ('shared','complementary','noisy_path','unstable_prognostic'):
        raise ValueError('Unknown scenario')
    rng=np.random.default_rng(seed)
    shared,u,v = rng.normal(size=(3,n))
    group = rng.integers(0,2,n).astype(bool)
    base=np.stack([shared,u,rng.normal(size=n),rng.normal(size=n)],1)
    sigma=np.full(n,.25)
    if scenario in ('noisy_path','unstable_prognostic'):
        sigma[group]=2.5
    images=base[:,None,:] + rng.normal(size=(n,patches,4))*sigma[:,None,None]
    rna=np.stack([shared,v,rng.normal(size=n),rng.normal(size=n)],1)+rng.normal(0,.15,(n,4))
    if scenario == 'shared':
        risk=.9*shared
    elif scenario == 'unstable_prognostic':
        risk=.5*shared+.7*v+1.4*u*group
    else:
        risk=.6*shared+.8*u+.8*v
    hazard=np.exp(np.clip(-1.2+risk,-5,3))
    event_time=rng.exponential(1/hazard)
    censor_time=rng.exponential(1/.22,n)
    observed=np.minimum(event_time,censor_time)
    event=(event_time<=censor_time).astype('float32')
    return {'patches':torch.tensor(images,dtype=torch.float32),
            'rna':torch.tensor(rna,dtype=torch.float32),
            'mask':torch.ones((n,patches),dtype=torch.bool),
            'time':torch.tensor(observed,dtype=torch.float32),
            'event':torch.tensor(event), 'group':group,
            'ids':[f'synthetic-{i}' for i in range(n)]}


def subset(d,indices,modality='both'):
    out={k: ([v[i] for i in indices] if k=='ids' else v[indices]) for k,v in d.items() if k != 'group'}
    if modality == 'path':
        out['rna'] = torch.zeros_like(out['rna'])
    elif modality == 'rna':
        out['patches'] = torch.zeros_like(out['patches'])
    return out


def concordance(time,event,risk):
    comparable=0
    score=0.0
    for i in range(len(time)):
        for j in range(len(time)):
            if event[i] == 1 and time[i] < time[j]:
                comparable += 1
                score += float(risk[i]>risk[j]) + .5*float(risk[i]==risk[j])
    return None if not comparable else score/comparable


def run(seed,scenario,cfg):
    data=generate(seed,scenario)
    order=np.random.default_rng(seed+900).permutation(len(data['ids']))
    tr,va,te=order[:180],order[180:240],order[240:]
    starts=torch.tensor([0.,1.,3.])  # fixed in advance, in arbitrary simulated units
    variants={
        'path':('path',replace(cfg,common_mode='off',lambda_within=0,lambda_unique=0)),
        'rna':('rna',replace(cfg,common_mode='off',lambda_within=0,lambda_unique=0)),
        'fusion':('both',replace(cfg,common_mode='off',lambda_within=0,lambda_unique=0)),
        'decur':('both',replace(cfg,common_mode='uniform')),
        'shuffled':('both',replace(cfg,common_mode='shuffled')),
        'stability':('both',replace(cfg,common_mode='stability')),
    }
    results={}
    for name,(modality,setting) in variants.items():
        train,valid,test=(subset(data,idx,modality) for idx in (tr,va,te))
        fit=train_cancer_model(train,valid,starts,setting)
        out=predict(fit,{k:v for k,v in test.items() if k not in ('time','event')})
        losses=piecewise_nll(out['log_rates'],test['time'],test['event'],starts,'none')
        risk=(1-survival_at(out['log_rates'],torch.tensor([2.]),starts)[:,0]).numpy()
        group=data['group'][te]
        record={'nll':losses.mean().item(),
                'c_index_at_2':concordance(test['time'].numpy(),test['event'].numpy(),risk),
                'group_nll':[losses[torch.tensor(group==q)].mean().item() for q in (False,True)],
                'group_n':[int((group==q).sum()) for q in (False,True)],
                'best_epoch':fit['best_epoch'],'skipped_common':fit['skipped_common'],
                'history':fit['history']}
        if name == 'stability':
            torch.manual_seed(seed+5000)
            x=test['patches']
            g=(test['rna']-fit['rna_mean'])/fit['rna_scale']
            (x1,g1,m1),(x2,g2,m2)=view_batch(x,g,test['mask'],setting)
            with torch.no_grad():
                p1,r1,_=fit['model'].encode(x1,g1,m1)
                p2,r2,_=fit['model'].encode(x2,g2,m2)
                try:
                    _,diag=stability_weighted_common(p1[:,:setting.common],p2[:,:setting.common],
                        r1[:,:setting.common],r2[:,:setting.common],strength=setting.strength,
                        min_weight=setting.min_weight)
                    w=diag['weights'].numpy()
                    record['group_weight']=[float(w[group==q].mean()) for q in (False,True)]
                    record['weight_std']=float(w.std())
                except DegenerateCorrelation:
                    record['group_weight']=None
                    record['weight_std']=None
        results[name]=record
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,23,37])
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--lambda-common',type=float,default=.001)
    parser.add_argument('--strength',type=float,default=2.)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    cfg=CancerConfig(epochs=args.epochs,patience=12,patch_keep=.6,lambda_within=.001,
                     lambda_common=args.lambda_common,lambda_unique=.001,strength=args.strength)
    report={'purpose':'synthetic only; not cancer patients','config':asdict(cfg),
            'runtime':{'torch':torch.__version__,'python':platform.python_version()},'runs':{}}
    for scenario in ('shared','complementary','noisy_path','unstable_prognostic'):
        report['runs'][scenario]={}
        for seed in args.seeds:
            report['runs'][scenario][str(seed)]=run(seed,scenario,replace(cfg,seed=seed))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2)
    print(f'Wrote synthetic cancer-survival stress test: {output}')


if __name__ == '__main__':
    main()
