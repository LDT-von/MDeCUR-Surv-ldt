"""Explicit research CLI; does not discover or download patient data."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import torch
from cancer_data import load_manifest
from cancer_model import piecewise_nll, survival_at
from cancer_train import CancerConfig, train_cancer_model, predict
from cancer_metrics import harrell_c, ipcw_brier


def cohort_fingerprint(cohort):
    """Hash ordered split IDs, processed input tensors and labels."""
    h=hashlib.sha256()
    for split in ('train','val','test'):
        d=cohort[split]
        h.update(split.encode('utf-8'))
        for patient in d['ids']:
            b=patient.encode('utf-8');h.update(len(b).to_bytes(8,'little'));h.update(b)
        for name in ('patches','mask','rna','time','event'):
            array=d[name].contiguous().numpy()
            h.update(name.encode());h.update(str(array.shape).encode());h.update(str(array.dtype).encode());h.update(array.tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--out',required=True,help='New output directory; existing directories are not overwritten')
    parser.add_argument('--starts',required=True,nargs='+',type=float,help='Fixed interval starts in time units; first must be 0')
    parser.add_argument('--horizon',required=True,type=float,help='Prediction horizon in same time units')
    parser.add_argument('--time-unit',required=True,help='E.g. years; record provenance explicitly')
    parser.add_argument('--endpoint',required=True,help='E.g. overall survival and its exact baseline')
    parser.add_argument('--mode',choices=['off','uniform','stability','shuffled'],required=True,
                        help='Choose a prespecified ablation; off disables only the common term')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--lambda-common',type=float,default=.001)
    parser.add_argument('--lambda-within',type=float,default=.001)
    parser.add_argument('--lambda-unique',type=float,default=.001)
    parser.add_argument('--strength',type=float,default=2.)
    args=parser.parse_args()
    destination=Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    if args.horizon < 0:
        raise ValueError('horizon must be nonnegative')
    manifest=Path(args.manifest).resolve()
    manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    cohort=load_manifest(manifest)
    input_hash=cohort_fingerprint(cohort)
    cfg=CancerConfig(seed=args.seed,epochs=args.epochs,common_mode=args.mode,
                     lambda_common=args.lambda_common,lambda_within=args.lambda_within,
                     lambda_unique=args.lambda_unique,strength=args.strength)
    starts=torch.tensor(args.starts,dtype=torch.float32)
    fitted=train_cancer_model(cohort['train'],cohort['val'],starts,cfg)
    test=cohort['test']
    outputs=predict(fitted,{k:v for k,v in test.items() if k not in ('event','time')})
    with torch.no_grad():
        nll=piecewise_nll(outputs['log_rates'],test['time'],test['event'],starts).item()
        survival=survival_at(outputs['log_rates'],torch.tensor([args.horizon]),starts)[:,0]
    ci=harrell_c(test['time'].numpy(),test['event'].numpy(),1-survival.numpy())
    brier=ipcw_brier(cohort['train']['time'].numpy(),cohort['train']['event'].numpy(),
                     test['time'].numpy(),test['event'].numpy(),survival.numpy(),args.horizon)
    destination.mkdir(parents=True,exist_ok=False)
    torch.save({'model_state':fitted['model'].state_dict(),
                'path_dim':cohort['train']['patches'].shape[-1],
                'rna_dim':cohort['train']['rna'].shape[-1],
                'rna_mean':fitted['rna_mean'],'rna_scale':fitted['rna_scale'],
                'starts':starts,'config':asdict(cfg)},destination/'checkpoint.pt')
    with (destination/'test_predictions.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['patient_id','survival_at_horizon'])
        writer.writerows(zip(test['ids'],survival.tolist()))
    report={'status':'research_only','manifest_sha256':manifest_hash,'cohort_tensor_sha256':input_hash,'config':asdict(cfg),
            'endpoint':args.endpoint,'time_unit':args.time_unit,'interval_starts':args.starts,
            'prediction_horizon':args.horizon,'best_epoch':fitted['best_epoch'],
            'validation_nll':fitted['validation_nll'],'test_nll':nll,
            'c_index_risk_at_horizon':ci,'ipcw_brier_at_horizon':brier,
            'patients':{k:len(v['ids']) for k,v in cohort.items()},
            'events':{k:int(v['event'].sum()) for k,v in cohort.items()},
            'skipped_common':fitted['skipped_common'],'history':fitted['history'],
            'runtime':{'python':platform.python_version(),'torch':torch.__version__},
            'warning':'Cohort provenance, time origin and split validity must be independently audited.'}
    with (destination/'report.json').open('w',encoding='utf-8') as handle:
        json.dump(report,handle,indent=2,ensure_ascii=False)
    print(f'Research output written to {destination}')


if __name__ == '__main__':
    main()
