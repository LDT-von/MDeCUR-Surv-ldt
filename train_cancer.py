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
        for name in ('patches_view1','patches_view2'):
            if name in d:
                array=d[name].contiguous().numpy()
                h.update(name.encode());h.update(str(array.shape).encode());h.update(array.tobytes())
        if 'view_provenance' in d:
            h.update(json.dumps(d['view_provenance'],sort_keys=True).encode())
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
    parser.add_argument('--patience',type=int,default=15,help='Stop after this many epochs without validation-NLL improvement')
    parser.add_argument('--hidden',type=int,default=16)
    parser.add_argument('--common',type=int,default=8)
    parser.add_argument('--training-backend',choices=['padded','stream'],default='padded')
    parser.add_argument('--validation-only',action='store_true',
                        help='stream: load/train/validate only; no test inputs or test evaluation')
    parser.add_argument('--patient-batch-size',type=int,default=16,
                        help='stream: patients per correlation batch; a final singleton is merged')
    parser.add_argument('--patch-chunk-size',type=int,default=2048,
                        help='stream: maximum patch rows read/encoded per block')
    parser.add_argument('--device',default='cpu',help='stream: explicit torch device; no automatic GPU selection')
    parser.add_argument('--lambda-common',type=float,default=.001)
    parser.add_argument('--lambda-within',type=float,default=.001)
    parser.add_argument('--lambda-path-within',type=float,default=None,
                        help='Pathology within coefficient; default is lambda-within/2')
    parser.add_argument('--lambda-rna-within',type=float,default=None,
                        help='RNA within coefficient; default is lambda-within/2')
    parser.add_argument('--lambda-local',type=float,default=0.,
                        help='Patient-balanced same-patch pathology loss; requires cached_uni')
    parser.add_argument('--local-dim',type=int,default=16)
    parser.add_argument('--prediction-space',choices=['projected','encoder'],default='projected',
                        help='encoder predicts before auxiliary projection; projected preserves the historical baseline')
    parser.add_argument('--modality',choices=['both','pathology','rna'],default='both',
                        help='Survival-only unimodal controls on the same paired cohort; both preserves the default')
    parser.add_argument('--lambda-unique',type=float,default=.001)
    parser.add_argument('--strength',type=float,default=2.)
    parser.add_argument('--path-views',choices=['legacy_patch_dropout','cached_uni','single_view'],default='legacy_patch_dropout',
                        help='cached_uni: verified image views; single_view: external mmap PT survival-only stream baseline')
    parser.add_argument('--patch-keep',type=float,default=None)
    parser.add_argument('--rna-noise',type=float,default=.05,
                        help='Existing RNA feature perturbation; not a validated biological augmentation')
    parser.add_argument('--rna-scale-policy',choices=['clamp','mask'],default='clamp',
                        help='mask: exclude training-low-variance genes at every split; clamp preserves historical scaling')
    parser.add_argument('--rna-min-std',type=float,default=1e-6,
                        help='Positive train-only std threshold/floor; saved with RNA statistics and mask')
    args=parser.parse_args()
    destination=Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    if args.horizon < 0:
        raise ValueError('horizon must be nonnegative')
    cfg=CancerConfig(seed=args.seed,epochs=args.epochs,patience=args.patience,common_mode=args.mode,
                     hidden=args.hidden,common=args.common,
                     lambda_common=args.lambda_common,lambda_within=args.lambda_within,
                     lambda_unique=args.lambda_unique,strength=args.strength,
                     lambda_path_within=args.lambda_path_within,lambda_rna_within=args.lambda_rna_within,
                     lambda_local=args.lambda_local,local_dim=args.local_dim,prediction_space=args.prediction_space,
                     path_views=args.path_views,rna_noise=args.rna_noise,
                     rna_scale_policy=args.rna_scale_policy,rna_min_std=args.rna_min_std,
                     modality=args.modality,
                     patch_keep=args.patch_keep if args.patch_keep is not None else
                                (1. if args.path_views in ('cached_uni','single_view') else .8))
    if args.training_backend == 'stream':
        from cancer_stream import train_stream_model, predict_stream, validate_stream_views
        validate_stream_views(cfg)
    elif args.validation_only or args.path_views == 'single_view':
        raise ValueError('--validation-only and single_view require --training-backend stream')
    manifest=Path(args.manifest).resolve()
    manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if args.training_backend == 'stream':
        from cancer_stream_data import load_stream_manifest, stream_cohort_fingerprint
        cohort=load_stream_manifest(manifest,path_views=args.path_views,include_test=not args.validation_only)
        input_hash=stream_cohort_fingerprint(cohort,path_views=args.path_views)
        fingerprint_scheme='single-view-pt-stream-v1' if args.path_views == 'single_view' else 'verified-cache-stream-v1'
    else:
        if args.device != 'cpu':
            raise ValueError('Explicit device selection is supported by the stream backend')
        cohort=load_manifest(manifest,path_views=args.path_views)
        input_hash=cohort_fingerprint(cohort)
        fingerprint_scheme='padded-cohort-tensors-v1'
    starts=torch.tensor(args.starts,dtype=torch.float32)
    if args.training_backend == 'stream':
        fitted=train_stream_model(cohort['train'],cohort['val'],starts,cfg,
                                  patient_batch_size=args.patient_batch_size,
                                  patch_chunk_size=args.patch_chunk_size,device=args.device)
        prediction_function=predict_stream
    else:
        fitted=train_cancer_model(cohort['train'],cohort['val'],starts,cfg)
        prediction_function=predict
    evaluation_split='val' if args.validation_only else 'test'
    evaluation=cohort[evaluation_split]
    outputs=prediction_function(fitted,{k:v for k,v in evaluation.items() if k not in ('event','time')})
    with torch.no_grad():
        nll=piecewise_nll(outputs['log_rates'],evaluation['time'],evaluation['event'],starts).item()
        survival=survival_at(outputs['log_rates'],torch.tensor([args.horizon]),starts)[:,0]
    ci=harrell_c(evaluation['time'].numpy(),evaluation['event'].numpy(),1-survival.numpy())
    brier=ipcw_brier(cohort['train']['time'].numpy(),cohort['train']['event'].numpy(),
                     evaluation['time'].numpy(),evaluation['event'].numpy(),survival.numpy(),args.horizon)
    destination.mkdir(parents=True,exist_ok=False)
    torch.save({'model_state':fitted['model'].state_dict(),
                'model_kwargs':fitted['model_kwargs'],
                'path_dim':fitted['model_kwargs']['path_dim'],
                'rna_dim':cohort['train']['rna'].shape[-1],
                'rna_mean':fitted['rna_mean'],'rna_scale':fitted['rna_scale'],
                'rna_mask':fitted['rna_mask'],'rna_train_std':fitted['rna_train_std'],
                'rna_preprocessing':fitted['rna_preprocessing'],
                'starts':starts,'config':asdict(cfg),
                'training_backend':args.training_backend,'stream_config':fitted.get('stream_config'),
                'validation_only':args.validation_only,'evaluation_split':evaluation_split,
                'effective_loss_weights':fitted['effective_loss_weights']},destination/'checkpoint.pt')
    prediction_name='validation_predictions.csv' if args.validation_only else 'test_predictions.csv'
    with (destination/prediction_name).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['patient_id','survival_at_horizon'])
        writer.writerows(zip(evaluation['ids'],survival.tolist()))
    report={'status':'research_only','manifest_sha256':manifest_hash,'cohort_input_sha256':input_hash,'config':asdict(cfg),
            'cohort_tensor_sha256':input_hash if args.training_backend == 'padded' else None,
            'cohort_fingerprint_scheme':fingerprint_scheme,'training_backend':args.training_backend,
            'stream_config':fitted.get('stream_config'),
            'model_kwargs':fitted['model_kwargs'],'effective_loss_weights':fitted['effective_loss_weights'],
            'modality':args.modality,
            'rna_preprocessing':fitted['rna_preprocessing'],
            'evaluation_split':evaluation_split,'test_evaluated':not args.validation_only,
            'feature_source':'external_single_view_pt_encoder_unverified' if args.path_views == 'single_view' else args.path_views,
            'endpoint':args.endpoint,'time_unit':args.time_unit,'interval_starts':args.starts,
            'prediction_horizon':args.horizon,'best_epoch':fitted['best_epoch'],
            'validation_nll':fitted['validation_nll'],
            'patients':{k:len(v['ids']) for k,v in cohort.items()},
            'events':{k:int(v['event'].sum()) for k,v in cohort.items()},
            'skipped_common':fitted['skipped_common'],'history':fitted['history'],
            'view_provenance':{k:v.get('view_provenance') for k,v in cohort.items()},
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'device':args.device},
            'warning':'Cohort provenance, time origin and split validity must be independently audited.'}
    if args.validation_only:
        report.update(validation_c_index_risk_at_horizon=ci,validation_ipcw_brier_at_horizon=brier)
    else:
        report.update(test_nll=nll,c_index_risk_at_horizon=ci,ipcw_brier_at_horizon=brier)
    with (destination/'report.json').open('w',encoding='utf-8') as handle:
        json.dump(report,handle,indent=2,ensure_ascii=False)
    print(f'Research output written to {destination}')


if __name__ == '__main__':
    main()
