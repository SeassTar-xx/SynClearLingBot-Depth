#!/usr/bin/env python3
"""Run LingBot-Depth v0.5 on a manifest of ClearDepth-compatible RGB-D samples."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from depth_io import read_depth, read_intrinsics, read_jsonl, read_rgb, write_jsonl


def safe_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest; input_depth must not be GT depth")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--apply-mask", action="store_true", help="Replace predicted-mask-rejected pixels with inf")
    parser.add_argument("--native-sdpa", action="store_true", help="Use repository native PyTorch SDPA wrapper")
    parser.add_argument("--disable-depth-masking", action="store_true", help="Compatibility option; not the default formal protocol")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not (args.official_repo / "mdm/model/v2.py").is_file():
        raise FileNotFoundError(f"Expected official LingBot source below {args.official_repo}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    if args.disable_depth_masking:
        print("WARNING: depth-token masking is disabled; do not treat this as the default formal protocol.", file=sys.stderr)

    sys.path.insert(0, str(args.official_repo))
    from mdm.model.v2 import MDMModel

    samples = list(read_jsonl(args.manifest))
    if not samples:
        raise ValueError("Manifest has no samples")
    if args.limit is not None:
        samples = samples[: args.limit]
    required = {"sample_id", "rgb", "input_depth", "input_depth_unit"}
    for item in samples:
        missing = required - item.keys()
        if missing:
            raise ValueError(f"{item.get('sample_id', '<unknown>')} missing fields: {sorted(missing)}")
        if item.get("input_depth") == item.get("gt_depth"):
            raise ValueError(f"{item['sample_id']}: input_depth and gt_depth must be different files")

    model = MDMModel.from_pretrained(args.checkpoint).to(args.device).eval()
    if args.native_sdpa:
        model.enable_pytorch_native_sdpa()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    rows: list[dict] = []

    for index, item in enumerate(samples, start=1):
        rgb = read_rgb(item["rgb"])
        depth = read_depth(item["input_depth"], item["input_depth_unit"])
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"{item['sample_id']}: RGB {rgb.shape[:2]} and input depth {depth.shape} are not aligned")
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(args.device)
        depth_tensor = torch.from_numpy(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)).unsqueeze(0).to(args.device)
        intrinsics = None
        if "intrinsics" in item and item["intrinsics"] is not None:
            intrinsics_np = read_intrinsics(item["intrinsics"]).copy()
            if not item.get("intrinsics_normalized", False):
                intrinsics_np[0, :] /= depth.shape[1]
                intrinsics_np[1, :] /= depth.shape[0]
            intrinsics = torch.from_numpy(intrinsics_np).unsqueeze(0).to(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = model.infer(
            image, depth_in=depth_tensor, intrinsics=intrinsics,
            resolution_level=args.resolution_level, apply_mask=args.apply_mask,
            enable_depth_mask=not args.disable_depth_masking,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        prediction = output["depth"].detach().float().cpu().numpy()[0]
        prediction_path = prediction_dir / f"{safe_name(item['sample_id'])}.npy"
        np.save(prediction_path, prediction)
        row = {
            "sample_id": item["sample_id"], "prediction": str(prediction_path.resolve()),
            "input_depth": item["input_depth"], "input_depth_unit": item["input_depth_unit"],
            "gt_depth": item.get("gt_depth"), "gt_depth_unit": item.get("gt_depth_unit", "m"),
            "transparent_mask": item.get("transparent_mask"), "elapsed_seconds": time.perf_counter() - started,
            "prediction_valid_ratio": float((np.isfinite(prediction) & (prediction > 0)).mean()),
        }
        rows.append(row)
        print(f"[{index}/{len(samples)}] {item['sample_id']} -> {prediction_path.name}")

    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "manifest": str(args.manifest.resolve()),
        "num_samples": len(rows), "device": args.device, "resolution_level": args.resolution_level,
        "apply_mask": args.apply_mask, "native_sdpa": args.native_sdpa,
        "disable_depth_masking": args.disable_depth_masking,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
