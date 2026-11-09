"""Explicit plan/doctor/extract commands. No automatic data or weight discovery."""
import argparse
from contextlib import nullcontext
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .core import (FEATURE_DIM, SCHEMA, extract_slide, file_hash, inspect_slides,
                   json_hash, load_recipe, load_slides,
                   read_coordinates, validate_geometry, verify_output)


def open_slide(path):
    try:
        import openslide
    except (ImportError, OSError) as exc:
        raise RuntimeError("OpenSlide Python package and native library are required; see uni_views/README.md") from exc
    return openslide.OpenSlide(path)


class FrozenUNI:
    def __init__(self, checkpoint, device, amp):
        import torch
        import timm
        self.torch, self.device, self.amp = torch, torch.device(device), amp
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU or CUDA devices are supported")
        if self.device.type != "cuda" and amp != "none":
            raise ValueError("Mixed precision is CUDA-only here")
        self.model = timm.create_model("vit_large_patch16_224", pretrained=False,
                         img_size=224, patch_size=16, init_values=1e-5,
                         num_classes=0, dynamic_img_size=True)
        # Fail on old PyTorch rather than silently using executable pickle loading.
        if "weights_only" not in inspect.signature(torch.load).parameters:
            raise RuntimeError("This PyTorch lacks weights_only; use a supported isolated environment")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False).eval().to(self.device)

    def __call__(self, array):
        torch = self.torch
        tensor = torch.from_numpy(array).to(self.device)
        precision = nullcontext() if self.amp == "none" else torch.autocast(
            "cuda", dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[self.amp])
        with torch.inference_mode(), precision:
            result = self.model(tensor)
        return result.float().cpu().numpy()


def environment():
    result = {"python": sys.version, "packages": {}}
    for package in ("numpy", "h5py", "Pillow", "torch", "torchvision", "timm", "openslide-python"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    return result


def preflight_geometry(items, edge_policy):
    errors, reports = [], []
    for item in items:
        try:
            coords, geometry = read_coordinates(item["coords_h5"])
            with open_slide(item["wsi_path"]) as slide:
                reports.append({"slide_id": item["slide_id"],
                                **validate_geometry(slide, coords, geometry, edge_policy)})
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append({"slide_id": item["slide_id"], "error": str(exc)})
    return reports, errors


def runtime_check(device_name, amp):
    """Import/check the runtime without building UNI or forwarding any images."""
    import torch
    import timm
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or CUDA devices are supported")
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError("PyTorch must support weights_only=True")
    result = {"device": str(device), "amp": amp, "torch_cuda": torch.version.cuda,
              "timm": timm.__version__}
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA device is unavailable")
        props = torch.cuda.get_device_properties(index)
        result.update(gpu=props.name, total_memory_bytes=props.total_memory,
                      capability=torch.cuda.get_device_capability(index),
                      compiled_arches=torch.cuda.get_arch_list())
        if amp == "bfloat16":
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("This CUDA device does not support requested bfloat16")
    elif amp != "none":
        raise ValueError("Mixed precision is CUDA-only here")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "doctor", "extract"))
    parser.add_argument("--manifest", required=True, help="Explicit slide/patient/WSI/coordinate CSV")
    parser.add_argument("--recipe", required=True, help="Explicit preprocessing JSON, not inferred from old features")
    parser.add_argument("--checkpoint", help="Local UNI v1 pytorch_model.bin; no download")
    parser.add_argument("--out", help="New run directory; --resume verifies existing completed slides")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16, help="Original patches per batch; 2x or 3x images")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--amp", choices=("none", "float16", "bfloat16"), default="none")
    parser.add_argument("--include-reference", action="store_true", help="Also save unaugmented features (50%% extra payload)")
    parser.add_argument("--edge-policy", choices=("error", "pad-white"), default="error")
    parser.add_argument("--limit-patches", type=int, help="Pilot only: first K rows per slide; never a full patient bag")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.limit_patches is not None and args.limit_patches < 1):
        parser.error("Batch size and patch limit must be positive")
    recipe = load_recipe(args.recipe)
    slides = load_slides(args.manifest)
    items, errors = inspect_slides(slides, args.limit_patches)
    count = sum(s["rows"] for s in items)
    sets = 3 if args.include_reference else 2
    report = {"action": args.action, "schema": SCHEMA, "recipe": recipe, "slides": items,
              "errors": errors, "patches": count, "feature_arrays": sets,
              "feature_payload_bytes": count * FEATURE_DIM * 4 * sets,
              "payload_note": "float32 feature arrays only; add HDF5 metadata/coords and temporary headroom",
              "environment": environment()}
    if args.action == "plan":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if errors else 0
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        report["errors"].append({"error": "An existing local UNI v1 checkpoint is required"})
    try:
        report["runtime"] = runtime_check(args.device,args.amp)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        report["errors"].append({"error": str(exc)})
    if not errors:
        geometry, failures = preflight_geometry(items, args.edge_policy)
        report.update(geometry=geometry)
        report["errors"].extend(failures)
    if args.action == "doctor" or report["errors"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report["errors"] else 0
    if not args.out:
        parser.error("extract requires --out")
    destination = Path(args.out).resolve()
    # Never allow feature outputs inside source WSI/coordinate folders.
    for item in items:
        for name in ("wsi_path", "coords_h5"):
            source_dir = Path(item[name]).parent
            if destination == source_dir or source_dir in destination.parents:
                raise ValueError("Choose an output directory outside the source data directories")
    encoder_sha = file_hash(args.checkpoint)
    contract = {"schema": SCHEMA, "manifest_sha256": file_hash(args.manifest),
                "recipe": recipe, "encoder_sha256": encoder_sha, "slides": items,
                "seed": args.seed, "include_reference": args.include_reference,
                "batch_size": args.batch_size, "amp": args.amp, "device": args.device,
                "edge_policy": args.edge_policy, "environment": report["environment"],
                "implementation_sha256": {name: file_hash(Path(__file__).parent / name)
                                          for name in ("cli.py", "core.py")}}
    signature = json_hash(contract)
    if args.resume:
        previous = json.loads((destination / "run.json").read_text(encoding="utf-8"))
        if previous.get("run_sha256") != signature:
            raise ValueError("Resume contract differs: inputs, checkpoint, recipe, environment or options changed")
    elif destination.exists():
        raise FileExistsError(destination)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / "run.json").open("x", encoding="utf-8") as stream:
            json.dump({"run_sha256": signature, "contract": contract}, stream, ensure_ascii=False, indent=2)
    lock = destination / ".extract.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"run_sha256": signature, "pid": os.getpid()}, stream)
    try:
        completed, pending = [], []
        for item in items:
            path = destination / (item["slide_id"] + ".h5")
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                verify_output(path, {**item, "run_sha256": signature}, allow_pilot=True)
                completed.append(item["slide_id"])
            else:
                pending.append(item)
        needed = sum(s["rows"] for s in pending) * FEATURE_DIM * 4 * sets
        free = shutil.disk_usage(destination).free
        if free < needed * 1.1 + 64 * 1024 ** 2:
            raise OSError("Insufficient free disk space for features plus headroom")
        started = time.perf_counter()
        encoder = FrozenUNI(args.checkpoint, args.device, args.amp) if pending else None
        for item in pending:
            meta = extract_slide(item, recipe, encoder, open_slide,
                         destination / (item["slide_id"] + ".h5"), signature, encoder_sha,
                         seed=args.seed, batch_size=args.batch_size,
                         include_reference=args.include_reference, edge_policy=args.edge_policy)
            print(json.dumps({"completed": item["slide_id"], "rows": meta["rows"],
                              "seconds": meta["elapsed_seconds"],
                              "images_per_second": meta["images_per_second"]}), flush=True)
            completed.append(item["slide_id"])
        print(json.dumps({"complete_slides": completed, "wall_seconds": time.perf_counter() - started,
                          "out": str(destination)}, ensure_ascii=False))
    finally:
        lock.unlink()
    return 0
