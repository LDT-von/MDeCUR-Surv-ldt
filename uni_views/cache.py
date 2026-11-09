"""Verified cached UNI views -> one explicitly ordered patient bag."""
import json
from pathlib import Path

import h5py
import numpy as np

from .core import json_hash, verify_output


def load_patient_cache(cell, manifest_dir, patient_id, include_views):
    """CSV cell is one H5 path, or a JSON array of slide paths in bag order.

    A fresh unaugmented reference is required: matching row counts alone cannot
    certify alignment or preprocessing equivalence with a downloaded .pt cache.
    """
    names = json.loads(cell) if cell.strip().startswith("[") else [cell.strip()]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("wsi_h5 must be a path or a nonempty JSON list of paths")
    paths = [(Path(manifest_dir) / name).resolve() for name in names]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate slide files in a patient bag")
    base, first, second, provenance = [], [], [], []
    signatures = set()
    slides = set()
    for path in paths:
        meta = verify_output(path, require_reference=True)
        if meta["patient_id"] != patient_id:
            raise ValueError("Cached slide belongs to a different patient")
        if meta["slide_id"] in slides:
            raise ValueError("The same slide was listed more than once")
        slides.add(meta["slide_id"])
        signatures.add((meta["encoder_sha256"], meta["recipe_sha256"]))
        if meta["encoder"] != "uni_v1" or meta["recipe_sha256"] != json_hash(meta["recipe"]):
            raise ValueError("Encoder/preprocessing provenance does not match")
        with h5py.File(path, "r") as handle:
            base.append(handle["features_reference"][:])
            if include_views:
                first.append(handle["features_a"][:])
                second.append(handle["features_b"][:])
        provenance.append({"slide_id": meta["slide_id"], "rows": meta["rows"],
                           "coords_sha256": meta["coords_sha256"],
                           "run_sha256": meta["run_sha256"],
                           "encoder_sha256": meta["encoder_sha256"],
                           "recipe_sha256": meta["recipe_sha256"]})
    if len(signatures) != 1:
        raise ValueError("Patient slides use different UNI weights or preprocessing")
    views = (np.concatenate(first), np.concatenate(second)) if include_views else None
    return np.concatenate(base), views, provenance
