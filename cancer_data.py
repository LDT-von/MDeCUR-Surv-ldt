"""Explicit, feature-only cancer cohort interface. Does not discover data."""
import csv
from pathlib import Path
import numpy as np
import torch


def load_manifest(manifest, path_views='legacy_patch_dropout'):
    """Load per-patient features and explicit splits from the supplied CSV.

    Legacy rows: patient_id,split,wsi_npy,rna_npy,time,event. WSI is [N,D],
    RNA [G]. cached_uni instead requires wsi_h5 containing verified reference
    and A/B views. Time must already use the documented baseline and unit.
    """
    if path_views not in ('legacy_patch_dropout','cached_uni'):
        raise ValueError('Unknown pathology view source')
    path=Path(manifest).resolve()
    with path.open('r',encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        required={'patient_id','split','rna_npy','time','event'}
        required.add('wsi_h5' if path_views == 'cached_uni' else 'wsi_npy')
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Manifest missing required columns')
        rows=list(reader)
    if not rows:
        raise ValueError('Manifest is empty')
    seen=set()
    seen_slides=set()
    cache_signatures=set()
    groups={'train':[], 'val':[], 'test':[]}
    patch_dim=gene_dim=None
    for row in rows:
        ident=row['patient_id'].strip()
        split=row['split'].strip().lower()
        if not ident or ident in seen or split not in groups:
            raise ValueError('Each patient must have a unique nonempty ID and valid split')
        seen.add(ident)
        rna_path=(path.parent / row['rna_npy']).resolve()
        views=provenance=None
        if path_views == 'cached_uni':
            from uni_views.cache import load_patient_cache
            wsi,views,provenance=load_patient_cache(row['wsi_h5'],path.parent,ident,split == 'train')
            for entry in provenance:
                if entry['slide_id'] in seen_slides:
                    raise ValueError('A slide appears in more than one patient or split')
                seen_slides.add(entry['slide_id'])
                cache_signatures.add((entry['encoder_sha256'],entry['recipe_sha256']))
            if len(cache_signatures) > 1:
                raise ValueError('Cohort uses different UNI weights or preprocessing')
        else:
            wsi_path=(path.parent / row['wsi_npy']).resolve()
            wsi=np.load(wsi_path,allow_pickle=False)
        rna=np.load(rna_path,allow_pickle=False)
        if wsi.ndim != 2 or wsi.shape[0] < 1 or rna.ndim != 1:
            raise ValueError('WSI must be [patches,features], RNA must be [features]')
        if not np.issubdtype(wsi.dtype,np.number) or not np.issubdtype(rna.dtype,np.number):
            raise ValueError('Features must be numeric')
        if not np.isfinite(wsi).all() or not np.isfinite(rna).all():
            raise ValueError('Nonfinite features: supply explicitly preprocessed input')
        if patch_dim is None:
            patch_dim,gene_dim=wsi.shape[1],rna.shape[0]
        if wsi.shape[1] != patch_dim or rna.shape[0] != gene_dim or not patch_dim or not gene_dim:
            raise ValueError('Feature dimensions must agree across all patients')
        time=float(row['time'])
        event=float(row['event'])
        if not np.isfinite(time) or time < 0 or event not in (0,1):
            raise ValueError('Invalid time or event')
        groups[split].append((ident,wsi.astype(np.float32),rna.astype(np.float32),time,event,views,provenance))
    if any(not group for group in groups.values()):
        raise ValueError('All three explicit train, val and test partitions are required')
    result={}
    for split,group in groups.items():
        nmax=max(len(x[1]) for x in group)
        x=torch.zeros((len(group),nmax,patch_dim),dtype=torch.float32)
        mask=torch.zeros((len(group),nmax),dtype=torch.bool)
        for i,row in enumerate(group):
            wsi=row[1]
            x[i,:len(wsi)]=torch.from_numpy(wsi)
            mask[i,:len(wsi)]=True
        result[split]={'ids':[r[0] for r in group], 'patches':x, 'mask':mask,
                       'rna':torch.from_numpy(np.stack([r[2] for r in group])),
                       'time':torch.tensor([r[3] for r in group],dtype=torch.float32),
                       'event':torch.tensor([r[4] for r in group],dtype=torch.float32)}
        if path_views == 'cached_uni':
            result[split]['view_provenance']=[r[6] for r in group]
            if split == 'train':
                a,b=torch.zeros_like(x),torch.zeros_like(x)
                for i,row in enumerate(group):
                    length=len(row[1])
                    a[i,:length]=torch.from_numpy(row[5][0])
                    b[i,:length]=torch.from_numpy(row[5][1])
                result[split].update(patches_view1=a,patches_view2=b)
    return result
