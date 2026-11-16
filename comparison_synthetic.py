"""Synthetic-data helpers for the five-fold comparison tests.

These build a fake cohort with the same *shape* as the real one -- a manifest
CSV, a double-precision survival reference CSV, per-patient RNA ``.npy`` files
and patch ``.pt`` bags -- so the protocol layer, the adapters and the runner
can be exercised on CPU without touching any patient data.

The synthetic patients carry a known, deliberately weak signal so that a test
can distinguish "the pipeline runs" from "the pipeline fits something".
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


def write_synthetic_cohort(root, *, patients=40, feature_dim=16, rna_dim=32,
                           seed=0, pathway_groups=3, genes=None):
    """Create a synthetic train/val cohort on disk.

    Returns a dict with the manifest path, the survival reference path, the
    ordered gene names and the expected split membership.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if genes is None:
        genes = [f'SYN{i:04d}' for i in range(rna_dim)]
    gene_path = root/'genes.txt'
    gene_path.write_text('\n'.join(genes)+'\n', encoding='utf-8')

    # Three pathways over disjoint gene blocks.
    block = rna_dim//pathway_groups
    gmt_lines = []
    pathway_indices = []
    for index in range(pathway_groups):
        members = list(range(index*block, (index+1)*block))
        pathway_indices.append(members)
        gmt_lines.append('\t'.join([f'PATH_{index}', f'PATH_{index}_ID']+[genes[m] for m in members]))
    gmt_path = root/'pathways.gmt'
    gmt_path.write_text('\n'.join(gmt_lines)+'\n', encoding='utf-8')

    feature_dir = root/'pt_files'
    feature_dir.mkdir(exist_ok=True)
    manifest_rows = []
    reference_rows = []
    train_count = int(patients*0.75)
    for index in range(patients):
        identity = f'SYN{index:03d}'
        split = 'train' if index < train_count else 'val'
        patches = rng.normal(0, 1, size=(8, feature_dim)).astype(np.float32)
        # A weak signal in the first patch coordinate drives the outcome.
        signal = patches[0, 0]
        time = float(np.clip(30+signal*3, .5, 80))
        event = float(rng.random() < .6)
        torch.save(torch.from_numpy(patches), feature_dir/f'{identity}.pt')
        rna = rng.normal(0, 1, size=rna_dim).astype(np.float32)
        rna[0] = signal*.5
        np.save(root/f'{identity}_rna.npy', rna)
        manifest_rows.append({'patient_id': identity, 'split': split,
                              'wsi_pt': f'pt_files/{identity}.pt',
                              'rna_npy': f'{identity}_rna.npy',
                              'time': f'{time:.6f}', 'event': f'{event:.1f}'})
        reference_rows.append({'patient_id': identity, 'split': split,
                               'time': repr(time), 'event': repr(event)})

    manifest_path = root/'manifest.csv'
    with manifest_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    reference_path = root/'survival_reference.csv'
    with reference_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)
    return {'root': root, 'manifest': manifest_path, 'survival_reference': reference_path,
            'genes': gene_path, 'gmt': gmt_path, 'genes_list': genes,
            'pathway_indices': pathway_indices,
            'train_ids': [r['patient_id'] for r in manifest_rows if r['split'] == 'train'],
            'val_ids': [r['patient_id'] for r in manifest_rows if r['split'] == 'val'],
            'feature_dim': feature_dim, 'rna_dim': rna_dim}
