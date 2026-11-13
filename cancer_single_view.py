"""Memory-mapped external single-view tensors, with no inferred encoder identity.

Patient ownership comes from the explicit manifest. Source files are hashed,
but a content hash cannot establish their clinical or encoder provenance.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

import torch


def _stamp(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class _PTSlide:
    path: str
    rows: int
    stamp: tuple


class PTPatientBag:
    """An ordered complete patient bag backed by safe torch mmap loads.

    Only block copies are returned; callers cannot modify mapped source tensors.
    Files without torch mmap support are rejected without an eager-load fallback.
    """
    view_source = 'single_view'

    def __init__(self, patient_id, cell, manifest_dir):
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError('A nonempty patient identity is required')
        if not isinstance(cell, str) or not cell.strip():
            raise ValueError('wsi_pt must be a path or a nonempty JSON list of paths')
        names = json.loads(cell) if cell.strip().startswith('[') else [cell.strip()]
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name.strip() for name in names)):
            raise ValueError('wsi_pt must be a path or a nonempty JSON list of paths')
        paths = tuple(str((Path(manifest_dir) / name).resolve()) for name in names)
        if len(set(paths)) != len(paths):
            raise ValueError('Duplicate slide files in a patient bag')
        self.patient_id = patient_id
        slides, mapped, provenance, identities, slide_ids = [], {}, [], set(), set()
        dimensions = set()
        for path in paths:
            before = _stamp(path)
            identity = before[:2] if before[1] else ('path', path.casefold())
            if identity in identities or Path(path).stem in slide_ids:
                raise ValueError('Duplicate physical slide or slide identity in a patient bag')
            identities.add(identity)
            slide_ids.add(Path(path).stem)
            try:
                tensor = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            except (RuntimeError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as error:
                raise ValueError('Single-view .pt requires safe weights_only=True mmap loading; '
                                 'convert unsupported files explicitly, no eager fallback') from error
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tensor.ndim != 2 or min(tensor.shape) < 1 or not tensor.is_floating_point()):
                raise ValueError('Single-view .pt must contain one nonempty real floating [N,D] tensor')
            dimensions.add(tensor.shape[1])
            digest = hashlib.sha256()
            with open(path, 'rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            if _stamp(path) != before:
                raise ValueError('Single-view feature file changed during verification')
            slides.append(_PTSlide(path, len(tensor), before))
            mapped[path] = tensor.detach()
            provenance.append(dict(slide_id=Path(path).stem, rows=len(tensor),
                                   feature_dim=tensor.shape[1], dtype=str(tensor.dtype),
                                   source_kind='external_single_view_pt', encoder_status='unverified',
                                   patient_mapping_source='manifest', file_sha256=digest.hexdigest()))
        if len(dimensions) != 1:
            raise ValueError('Single-view feature dimensions must agree within a patient')
        self.feature_dim = dimensions.pop()
        self.paths = paths
        self.file_identities = tuple(slide.stamp[:2] if slide.stamp[1]
                                     else ('path', slide.path.casefold()) for slide in slides)
        self.n_patches = sum(slide.rows for slide in slides)
        self._slides = tuple(slides)
        self._by_path = {slide.path: slide for slide in slides}
        self._mapped = mapped
        self._provenance = provenance

    def close(self):
        """Release mapped storages when a long-lived caller finishes this bag."""
        self._mapped.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    @property
    def provenance(self):
        return copy.deepcopy(self._provenance)

    def assert_unchanged(self):
        for slide in self._slides:
            if _stamp(slide.path) != slide.stamp:
                raise ValueError('Single-view feature file changed after verification')

    def blocks(self, chunk_size):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        for slide in self._slides:
            for start in range(0, slide.rows, chunk_size):
                yield (slide.path, start, min(start + chunk_size, slide.rows))

    def read_rows(self, rows):
        """Read an explicit list of row indices across the whole patient bag.

        This is the batched counterpart of ``read_block``: it resolves every
        global row to its slide once and copies the rows in a single pass per
        slide, instead of issuing one block read per row.
        """
        if not self._mapped:
            raise ValueError('Single-view bag is closed')
        if not isinstance(rows, (list, tuple, range)) or len(rows) == 0:
            raise ValueError('read_rows needs a nonempty sequence of row indices')
        offsets, cursor = [], 0
        for slide in self._slides:
            offsets.append((slide.path, cursor, cursor+slide.rows))
            cursor += slide.rows
        pieces = []
        for row in rows:
            if type(row) is not int or not 0 <= row < cursor:
                raise ValueError('Row index is outside the patient bag')
            for path, base, stop in offsets:
                if base <= row < stop:
                    slide = self._by_path[path]
                    if _stamp(path) != slide.stamp:
                        raise ValueError('Single-view feature file changed after verification')
                    pieces.append((path, row-base))
                    break
            else:
                raise ValueError('Row index is outside the patient bag')
        grouped = {}
        for path, local in pieces:
            grouped.setdefault(path, []).append(local)
        arrays = []
        for path, locals_ in grouped.items():
            slide = self._by_path[path]
            if _stamp(path) != slide.stamp:
                raise ValueError('Single-view feature file changed after verification')
            tensor = self._mapped[path]
            index = torch.as_tensor(locals_, dtype=torch.long)
            arrays.append(tensor.index_select(0, index).to(dtype=torch.float32, copy=True))
            if _stamp(path) != slide.stamp:
                raise ValueError('Single-view feature file changed during block read')
        stacked = torch.cat(arrays, dim=0)
        if not torch.isfinite(stacked).all():
            raise ValueError('Nonfinite or float32-overflow features in single-view block')
        return stacked

    def read_block(self, token, names):
        if not self._mapped:
            raise ValueError('Single-view bag is closed')
        if not isinstance(names, (tuple, list)) or tuple(names) != ('features_reference',):
            raise ValueError('Single-view features have reference only; no A/B views are available')
        if not isinstance(token, tuple) or len(token) != 3:
            raise ValueError('A block token must be a (path,start,stop) tuple')
        path, start, stop = token
        if not isinstance(path, str) or path not in self._by_path:
            raise ValueError('Block path is not part of this patient bag')
        slide = self._by_path[path]
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= slide.rows:
            raise ValueError('Invalid block row range')
        if _stamp(path) != slide.stamp:
            raise ValueError('Single-view feature file changed after verification')
        array = self._mapped[path][start:stop].to(dtype=torch.float32, copy=True)
        if _stamp(path) != slide.stamp:
            raise ValueError('Single-view feature file changed during block read')
        if not torch.isfinite(array).all():
            raise ValueError('Nonfinite or float32-overflow features in single-view block')
        return (array,)
