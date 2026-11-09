"""WSI/coordinate alignment, lossless D4 views and bounded feature extraction.

No tissue segmentation, patient split creation, downloads or model training.
Only CLAM-style level-0 coordinates/patches are supported in this first version.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

import h5py
import numpy as np
from PIL import Image, ImageOps


FEATURE_DIM = 1024
SCHEMA = "uni-full-patch-views-v1"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype="<i8").tobytes(order="C")).hexdigest()


def load_recipe(path):
    recipe = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    keys = {"encoder", "resize_sizes", "interpolation", "mean", "std"}
    if set(recipe) != keys or recipe["encoder"] != "uni_v1":
        raise ValueError("Recipe must explicitly specify UNI v1 and all preprocessing fields")
    sizes = recipe["resize_sizes"]
    if (not isinstance(sizes, list) or not sizes or sizes[-1] != 224 or
            any(type(n) is not int or n < 1 for n in sizes)):
        raise ValueError("resize_sizes must be positive square sizes ending in 224")
    if recipe["interpolation"] not in ("bilinear", "bicubic"):
        raise ValueError("interpolation must be bilinear or bicubic")
    for key in ("mean", "std"):
        values = recipe[key]
        if (not isinstance(values, list) or len(values) != 3 or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)):
            raise ValueError(f"{key} must contain three finite numbers")
    if min(recipe["std"]) <= 0:
        raise ValueError("Standard deviations must be positive")
    return recipe


def load_slides(manifest):
    path = Path(manifest).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {"patient_id", "slide_id", "wsi_path", "coords_h5"}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError(f"Extraction manifest requires {sorted(fields)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Extraction manifest is empty")
    slides, seen, wsi_sources, coord_sources = [], set(), set(), set()
    for row in rows:
        patient, slide = row["patient_id"].strip(), row["slide_id"].strip()
        if not patient or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", slide):
            raise ValueError("Explicit patient_id and a filename-safe slide_id are required")
        if slide in seen:
            raise ValueError(f"Duplicate slide_id: {slide}")
        seen.add(slide)
        item = {"patient_id": patient, "slide_id": slide}
        for key in ("wsi_path", "coords_h5"):
            if not row[key].strip():
                raise ValueError(f"Empty {key}: {slide}")
            item[key] = str((path.parent / row[key].strip()).resolve())
        # Multiple slides per patient are valid, duplicate physical inputs are not.
        if item["wsi_path"] in wsi_sources or item["coords_h5"] in coord_sources:
            raise ValueError("The same WSI or coordinate file was listed twice")
        if any(Path(item[key]).stem != slide for key in ("wsi_path", "coords_h5")):
            raise ValueError("slide_id must exactly match both WSI and coordinate filename stems")
        wsi_sources.add(item["wsi_path"])
        coord_sources.add(item["coords_h5"])
        slides.append(item)
    return slides


def read_coordinates(path):
    with h5py.File(path, "r") as handle:
        ds = handle["coords"]
        if ds.ndim != 2 or ds.shape[1] != 2 or ds.shape[0] < 1 or ds.dtype.kind not in "iu":
            raise ValueError("coords must be a nonempty integer [N,2] dataset")
        coords = ds[:]
        if (coords < 0).any() or (coords > np.iinfo(np.int64).max).any():
            raise ValueError("Coordinates must be nonnegative signed-int64 values")
        attrs = dict(ds.attrs)
    size, level = attrs.get("patch_size"), attrs.get("patch_level")
    if np.ndim(size) != 0 or size is None or float(size) != int(size) or int(size) < 1:
        raise ValueError("coords.attrs.patch_size must be an explicit positive integer")
    if level is None or np.ndim(level) != 0 or float(level) != 0:
        raise ValueError("Only explicit patch_level=0 is supported; do not guess a conversion")
    coords = np.asarray(coords, dtype=np.int64)
    dims = attrs.get("level_dim")
    if dims is not None:
        dims = np.asarray(dims)
        if dims.shape != (2,) or dims.dtype.kind not in "iu" or (dims <= 0).any():
            raise ValueError("Recorded level_dim must be positive integer [width,height]")
        dims = [int(n) for n in dims]
    return coords, {"patch_size": int(size), "patch_level": 0,
                    "recorded_level_dim": dims, "coords_sha256": array_hash(coords)}


def inspect_slides(slides, limit=None):
    """Read coordinates/file metadata only; never decode WSI pixels or load UNI."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    reports, errors = [], []
    for slide in slides:
        try:
            for key in ("wsi_path", "coords_h5"):
                if not Path(slide[key]).is_file():
                    raise FileNotFoundError(slide[key])
            coords, geometry = read_coordinates(slide["coords_h5"])
            n = len(coords) if limit is None else min(limit, len(coords))
            stat = Path(slide["wsi_path"]).stat()
            reports.append({**slide, **geometry, "source_rows": len(coords), "rows": n,
                            "pilot": n != len(coords),
                            "coords_file_sha256": file_hash(slide["coords_h5"]),
                            "wsi_size": stat.st_size, "wsi_mtime_ns": stat.st_mtime_ns})
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            errors.append({"slide_id": slide["slide_id"], "error": str(exc)})
    return reports, errors


def validate_geometry(slide, coords, geometry, edge_policy):
    if edge_policy not in ("error", "pad-white"):
        raise ValueError("Unknown edge policy")
    dims = tuple(int(n) for n in slide.dimensions)
    recorded = geometry["recorded_level_dim"]
    if recorded is not None and tuple(recorded) != dims:
        raise ValueError(f"Actual WSI dimensions {dims} differ from recorded {recorded}")
    if len(dims) != 2 or min(dims) < 1:
        raise ValueError("Invalid WSI dimensions")
    size = geometry["patch_size"]
    if (coords[:, 0] >= dims[0]).any() or (coords[:, 1] >= dims[1]).any():
        raise ValueError("A patch origin is outside the WSI; cannot repair by padding")
    edge = (coords[:, 0] > dims[0] - size) | (coords[:, 1] > dims[1] - size)
    if edge.any() and edge_policy == "error":
        raise ValueError(f"{int(edge.sum())} boundary patches need an explicit pad-white decision; no rows skipped")
    props = dict(slide.properties)
    return {"actual_level0_dim": dims, "edge_rows": int(edge.sum()),
            "mpp_x": props.get("openslide.mpp-x"), "mpp_y": props.get("openslide.mpp-y"),
            "objective_power": props.get("openslide.objective-power")}


def view_codes(seed, slide_id, row):
    """Independent uniform D4 draws; equal draws are valid and recorded."""
    digest = hashlib.sha256(json.dumps([int(seed), slide_id, int(row)]).encode()).digest()
    return digest[0] % 8, digest[1] % 8


def transform_image(image, code):
    if code not in range(8):
        raise ValueError("D4 code must be 0..7")
    if code >= 4:
        image = ImageOps.mirror(image)
    rotation = code % 4
    if rotation:
        image = image.transpose({1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                                 3: Image.Transpose.ROTATE_270}[rotation])
    return image


def read_patch(slide, xy, size):
    # OpenSlide coords are level-0 (x,y). Preserve all rows and composite alpha.
    image = slide.read_region(tuple(int(n) for n in xy), 0, (size, size)).convert("RGBA")
    if image.size != (size, size):
        raise ValueError("Reader returned a patch with an unexpected size")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, image).convert("RGB")


def pixel_tensor_array(image, recipe):
    interpolation = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC}
    if image.width != image.height:
        raise ValueError("Only square patches are supported; no implicit crop")
    for size in recipe["resize_sizes"]:
        image = image.resize((size, size), interpolation[recipe["interpolation"]])
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray(recipe["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(recipe["std"], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((array - mean) / std)


def verify_output(path, expected=None, require_reference=False, allow_pilot=False):
    """Check completion, row identity and feature checksums in bounded chunks."""
    with h5py.File(path, "r") as handle:
        meta = json.loads(handle.attrs["metadata_json"])
        if handle.attrs.get("complete", 0) != 1 or meta.get("schema") != SCHEMA:
            raise ValueError(f"Incomplete or unsupported feature file: {path}")
        if meta["pilot"] and not allow_pilot:
            raise ValueError("Pilot subsets cannot be used as full patient bags")
        if expected is not None:
            for key in ("run_sha256", "patient_id", "slide_id", "rows", "coords_sha256"):
                if meta[key] != expected[key]:
                    raise ValueError(f"Resume/source mismatch: {key}")
        n = meta["rows"]
        if (type(n) is not int or n < 1 or meta["source_rows"] < n or
                meta["pilot"] != (n != meta["source_rows"])):
            raise ValueError("Invalid feature row provenance")
        if require_reference and not meta["include_reference"]:
            raise ValueError("An unaugmented reference is required for training integration")
        coords = handle["coords"]
        if coords.shape != (n, 2):
            raise ValueError("Coordinate identity was changed")
        coordinate_hash = hashlib.sha256()
        for start in range(0, n, 2048):
            coordinate_hash.update(np.asarray(coords[start:start + 2048], dtype="<i8").tobytes(order="C"))
        if coordinate_hash.hexdigest() != meta["output_coords_sha256"]:
            raise ValueError("Coordinate identity was changed")
        if not meta["pilot"] and meta["output_coords_sha256"] != meta["coords_sha256"]:
            raise ValueError("Full bag coordinates differ from the source")
        row_index = handle["row_index"]
        if row_index.shape != (n,):
            raise ValueError("Feature rows were reordered or omitted")
        for start in range(0, n, 2048):
            stop = min(start + 2048, n)
            if not np.array_equal(row_index[start:stop], np.arange(start, stop, dtype=np.int64)):
                raise ValueError("Feature rows were reordered or omitted")
        for name in ("transform_a", "transform_b"):
            codes = handle[name]
            index = 0 if name == "transform_a" else 1
            if codes.shape != (n,):
                raise ValueError("Transform record does not match the deterministic view recipe")
            for start in range(0, n, 2048):
                stop = min(start + 2048, n)
                expected_codes = [view_codes(meta["seed"], meta["slide_id"], i)[index]
                                  for i in range(start, stop)]
                if not np.array_equal(codes[start:stop], expected_codes):
                    raise ValueError("Transform record does not match the deterministic view recipe")
        names = ["features_a", "features_b"]
        if require_reference or meta["include_reference"]:
            names.append("features_reference")
        for name in names:
            ds = handle[name]
            if ds.shape != (n, FEATURE_DIM) or ds.dtype != np.dtype("float32"):
                raise ValueError(f"Invalid UNI v1 features: {name}")
            checksum = hashlib.sha256()
            for start in range(0, n, 2048):
                block = ds[start:start + 2048]
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite features: {name}")
                checksum.update(block.tobytes(order="C"))
            if ds.attrs.get("sha256") != checksum.hexdigest():
                raise ValueError(f"Feature checksum mismatch: {name}")
    return meta


def extract_slide(item, recipe, encoder, slide_factory, destination, run_sha256,
                  encoder_sha256, seed=17, batch_size=16, include_reference=False,
                  edge_policy="error"):
    """encoder receives a float32 [images,3,224,224] NumPy array, returns [images,1024]."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    coords, geometry = read_coordinates(item["coords_h5"])
    stat = Path(item["wsi_path"]).stat()
    if stat.st_size != item["wsi_size"] or stat.st_mtime_ns != item["wsi_mtime_ns"]:
        raise ValueError("WSI file metadata changed since preflight")
    if (file_hash(item["coords_h5"]) != item["coords_file_sha256"] or
            array_hash(coords) != item["coords_sha256"] or len(coords) != item["source_rows"]):
        raise ValueError("Coordinates changed since preflight")
    n = item["rows"]
    if not 0 < n <= len(coords):
        raise ValueError("Invalid row budget")
    with slide_factory(item["wsi_path"]) as slide:
        actual = validate_geometry(slide, coords, geometry, edge_policy)
        meta = {**item, **actual, "schema": SCHEMA, "run_sha256": run_sha256,
                "encoder_sha256": encoder_sha256, "recipe": recipe,
                "recipe_sha256": json_hash(recipe), "encoder": "uni_v1",
                "include_reference": include_reference, "seed": seed,
                "edge_policy": edge_policy, "output_coords_sha256": array_hash(coords[:n])}
        temporary = destination.with_name(destination.name + f".partial.{uuid.uuid4().hex}")
        started = time.perf_counter()
        with h5py.File(temporary, "x") as handle:
            handle.attrs["complete"] = 0
            handle.create_dataset("coords", data=coords[:n])
            handle.create_dataset("row_index", data=np.arange(n, dtype=np.int64))
            codes = np.array([view_codes(seed, item["slide_id"], i) for i in range(n)], dtype=np.uint8)
            handle.create_dataset("transform_a", data=codes[:, 0])
            handle.create_dataset("transform_b", data=codes[:, 1])
            names = (["features_reference"] if include_reference else []) + ["features_a", "features_b"]
            datasets = {name: handle.create_dataset(name, (n, FEATURE_DIM), dtype="float32",
                         chunks=(min(n, 256), FEATURE_DIM)) for name in names}
            hashes = {name: hashlib.sha256() for name in names}
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                images = {name: [] for name in names}
                for row in range(start, stop):
                    image = read_patch(slide, coords[row], geometry["patch_size"])
                    if include_reference:
                        images["features_reference"].append(pixel_tensor_array(image, recipe))
                    for name, code in zip(("features_a", "features_b"), codes[row]):
                        images[name].append(pixel_tensor_array(transform_image(image, int(code)), recipe))
                output = np.asarray(encoder(np.stack([image for name in names for image in images[name]])))
                count = stop - start
                if output.shape != (count * len(names), FEATURE_DIM) or not np.isfinite(output).all():
                    raise ValueError("UNI must return finite [images,1024] embeddings")
                output = output.astype(np.float32)
                if not np.isfinite(output).all():
                    raise ValueError("Embeddings overflowed float32 output storage")
                for index, name in enumerate(names):
                    block = np.ascontiguousarray(output[index * count:(index + 1) * count])
                    datasets[name][start:stop] = block
                    hashes[name].update(block.tobytes(order="C"))
            seconds = time.perf_counter() - started
            meta.update(elapsed_seconds=seconds, patches_per_second=n / max(seconds, 1e-9),
                        images_per_second=n * len(names) / max(seconds, 1e-9),
                        identical_transform_pairs=int((codes[:, 0] == codes[:, 1]).sum()))
            for name in names:
                datasets[name].attrs["sha256"] = hashes[name].hexdigest()
            handle.attrs["metadata_json"] = json.dumps(meta, ensure_ascii=False, allow_nan=False)
            handle.attrs["complete"] = 1
            handle.flush()
        # Atomic publication without replacing an existing name, including a
        # concurrent external writer. Both names are on the same filesystem.
        os.link(temporary, destination)
        temporary.unlink()
    return meta
