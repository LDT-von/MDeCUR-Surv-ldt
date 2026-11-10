"""Verified lazy UNI H5 bags with bounded reads and explicit patient splits.

Checksums are verified once at construction; each later block read checks the
file identity, size and timestamps before and after I/O. Files must stay fixed
throughout forward/backward replay. No complete WSI feature bag is retained.
"""
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from uni_views.core import FEATURE_DIM, json_hash, verify_output
from cancer_single_view import PTPatientBag


FEATURE_NAMES = ('features_reference', 'features_a', 'features_b')


def _stamp(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class _Slide:
    path: str
    rows: int
    stamp: tuple


class H5PatientBag:
    """One patient's ordered slides; block tokens are immutable path/start/stop."""
    feature_dim = FEATURE_DIM
    view_source = 'cached_uni'

    def __init__(self, patient_id, cell, manifest_dir):
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError('A nonempty patient identity is required')
        if not isinstance(cell, str) or not cell.strip():
            raise ValueError('wsi_h5 must be a path or a nonempty JSON list of paths')
        names = json.loads(cell) if cell.strip().startswith('[') else [cell.strip()]
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError('wsi_h5 must be a path or a nonempty JSON list of paths')
        paths = tuple(str((Path(manifest_dir) / name).resolve()) for name in names)
        if len(set(paths)) != len(paths):
            raise ValueError('Duplicate slide files in a patient bag')
        self.patient_id = patient_id
        slides, provenance, identities, slide_ids, signatures = [], [], set(), set(), set()
        for path in paths:
            before = _stamp(path)
            identity = before[:2] if before[1] else ('path', path.casefold())
            if identity in identities:
                raise ValueError('Duplicate physical slide files in a patient bag')
            identities.add(identity)
            meta = verify_output(path, require_reference=True)
            if meta['patient_id'] != patient_id:
                raise ValueError('Cached slide belongs to a different patient')
            if meta['slide_id'] in slide_ids:
                raise ValueError('The same slide was listed more than once')
            slide_ids.add(meta['slide_id'])
            if meta['encoder'] != 'uni_v1' or meta['recipe_sha256'] != json_hash(meta['recipe']):
                raise ValueError('Encoder/preprocessing provenance does not match')
            signatures.add((meta['encoder_sha256'], meta['recipe_sha256']))
            with h5py.File(path, 'r') as handle:
                checksums = {name: str(handle[name].attrs['sha256']) for name in FEATURE_NAMES}
            if _stamp(path) != before:
                raise ValueError('Cached feature file changed during verification')
            slides.append(_Slide(path, meta['rows'], before))
            provenance.append({key: meta[key] for key in (
                'slide_id', 'rows', 'coords_sha256', 'output_coords_sha256',
                'run_sha256', 'encoder_sha256', 'recipe_sha256')})
            provenance[-1]['feature_sha256'] = checksums
        if len(signatures) != 1:
            raise ValueError('Patient slides use different UNI weights or preprocessing')
        self._slides = tuple(slides)
        self._by_path = {slide.path: slide for slide in slides}
        self._provenance = provenance
        self.paths = paths
        self.file_identities = tuple((slide.stamp[:2] if slide.stamp[1]
                                     else ('path', slide.path.casefold())) for slide in slides)
        self.n_patches = sum(slide.rows for slide in slides)

    @property
    def provenance(self):
        return copy.deepcopy(self._provenance)

    def assert_unchanged(self):
        for slide in self._slides:
            if _stamp(slide.path) != slide.stamp:
                raise ValueError('Cached feature file changed after verification')

    def blocks(self, chunk_size):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        for slide in self._slides:
            for start in range(0, slide.rows, chunk_size):
                yield (slide.path, start, min(start + chunk_size, slide.rows))

    def read_block(self, token, names):
        if not isinstance(token, tuple) or len(token) != 3:
            raise ValueError('A block token must be a (path,start,stop) tuple')
        path, start, stop = token
        if not isinstance(path, str) or path not in self._by_path:
            raise ValueError('Block path is not part of this patient bag')
        slide = self._by_path[path]
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= slide.rows:
            raise ValueError('Invalid block row range')
        if (not isinstance(names, (list, tuple)) or not names
                or len(set(names)) != len(names) or any(name not in FEATURE_NAMES for name in names)):
            raise ValueError('Choose explicit reference/A/B feature datasets')
        if _stamp(path) != slide.stamp:
            raise ValueError('Cached feature file changed after verification')
        with h5py.File(path, 'r') as handle:
            arrays = tuple(handle[name][start:stop] for name in names)
        if _stamp(path) != slide.stamp:
            raise ValueError('Cached feature file changed during block read')
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError('Nonfinite features in cached block')
        return tuple(torch.from_numpy(array) for array in arrays)


def load_stream_manifest(manifest, path_views='cached_uni', include_test=True):
    """Load explicit patient inputs without padding; optionally exclude test I/O.

    Test IDs/split names in the CSV are still checked for duplicate identities,
    but their feature/RNA paths and labels are not accessed in validation-only mode.
    """
    if path_views not in ('cached_uni', 'single_view'):
        raise ValueError('Choose cached_uni or single_view for streaming input')
    feature_column = 'wsi_h5' if path_views == 'cached_uni' else 'wsi_pt'
    bag_class = H5PatientBag if path_views == 'cached_uni' else PTPatientBag
    path = Path(manifest).resolve()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'patient_id', 'split', feature_column, 'rna_npy', 'time', 'event'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Manifest missing required columns')
        rows = list(reader)
    if not rows:
        raise ValueError('Manifest is empty')
    groups = {split: [] for split in (('train', 'val', 'test') if include_test else ('train', 'val'))}
    seen_ids, seen_slides, seen_paths, seen_files, signatures = set(), set(), set(), set(), set()
    gene_dim = None
    for row in rows:
        ident, split = row['patient_id'].strip(), row['split'].strip().lower()
        if not ident or ident in seen_ids or split not in ('train', 'val', 'test'):
            raise ValueError('Each patient must have a unique nonempty ID and valid split')
        seen_ids.add(ident)
        if split == 'test' and not include_test:
            continue
        bag = bag_class(ident, row[feature_column], path.parent)
        if seen_paths.intersection(bag.paths) or seen_files.intersection(bag.file_identities):
            raise ValueError('A physical slide appears in more than one patient or split')
        seen_paths.update(bag.paths)
        seen_files.update(bag.file_identities)
        for entry in bag.provenance:
            if entry['slide_id'] in seen_slides:
                raise ValueError('A slide appears in more than one patient or split')
            seen_slides.add(entry['slide_id'])
            if path_views == 'cached_uni':
                signatures.add((entry['encoder_sha256'], entry['recipe_sha256']))
        if path_views == 'cached_uni' and len(signatures) != 1:
            raise ValueError('Cohort uses different UNI weights or preprocessing')
        rna = np.load((path.parent / row['rna_npy']).resolve(), allow_pickle=False)
        if (rna.ndim != 1 or not rna.size or not np.issubdtype(rna.dtype, np.number)
                or np.iscomplexobj(rna) or not np.isfinite(rna).all()):
            raise ValueError('RNA must be a finite nonempty real numeric vector')
        if gene_dim is None:
            gene_dim = len(rna)
        if len(rna) != gene_dim:
            raise ValueError('RNA feature dimensions must agree across all patients')
        rna = rna.astype(np.float32)
        if not np.isfinite(rna).all():
            raise ValueError('RNA cannot be represented as finite float32')
        time, event = float(row['time']), float(row['event'])
        if not np.isfinite(time) or time < 0 or event not in (0., 1.):
            raise ValueError('Invalid time or event')
        if not np.isfinite(np.float32(time)):
            raise ValueError('Time cannot be represented as finite float32')
        groups[split].append((ident, bag, rna, time, event))
    if any(not group for group in groups.values()):
        raise ValueError('All requested explicit train/val/test partitions must be nonempty')
    if len({bag.feature_dim for group in groups.values() for _, bag, *_ in group}) != 1:
        raise ValueError('Single-view feature dimensions must agree across all loaded patients')
    return {split: {'ids': [r[0] for r in group], 'bags': [r[1] for r in group],
                    'rna': torch.from_numpy(np.stack([r[2] for r in group])),
                    'time': torch.tensor([r[3] for r in group], dtype=torch.float32),
                    'event': torch.tensor([r[4] for r in group], dtype=torch.float32),
                    'view_provenance': [r[1].provenance for r in group]}
            for split, group in groups.items()}


def stream_cohort_fingerprint(cohort, path_views='cached_uni'):
    """Hash ordered identities, RNA, labels and content checksums, not paths."""
    if path_views not in ('cached_uni', 'single_view'):
        raise ValueError('Invalid streaming fingerprint source')
    domain = b'cancer-verified-stream-v1' if path_views == 'cached_uni' else b'cancer-single-view-stream-v1'
    digest = hashlib.sha256(domain)
    for split in ('train', 'val', 'test'):
        if split not in cohort:
            continue
        data = cohort[split]
        digest.update(split.encode('utf-8'))
        for ident, bag in zip(data['ids'], data['bags']):
            bag.assert_unchanged()
            encoded = ident.encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
            encoded = json.dumps(bag.provenance, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False).encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
        for name in ('rna', 'time', 'event'):
            array = data[name].detach().cpu().contiguous().numpy()
            digest.update(name.encode('ascii'))
            digest.update(str(array.shape).encode('ascii'))
            digest.update(str(array.dtype).encode('ascii'))
            digest.update(array.tobytes(order='C'))
    return digest.hexdigest()
