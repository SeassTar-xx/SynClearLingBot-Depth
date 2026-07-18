#!/usr/bin/env python3
"""Fail fast on formal ClearDepth/LingBot manifest errors before GPU inference."""
from __future__ import annotations

import argparse
from pathlib import Path

from depth_io import read_depth, read_intrinsics, read_jsonl, read_mask, read_rgb


def require_file(value: str | None, label: str, sample_id: str) -> None:
    if value is None or not Path(value).is_file():
        raise FileNotFoundError(f"{sample_id}: missing {label}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    samples = list(read_jsonl(args.manifest))
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("Manifest is empty")
    seen = set()
    for item in samples:
        sample_id = item["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        for field in ("rgb", "input_depth", "input_depth_unit", "gt_depth"):
            if field not in item:
                raise ValueError(f"{sample_id}: missing required field {field}")
        if item["input_depth"] == item["gt_depth"]:
            raise ValueError(f"{sample_id}: input_depth equals gt_depth (label leakage)")
        require_file(item["rgb"], "rgb", sample_id)
        require_file(item["input_depth"], "input_depth", sample_id)
        require_file(item["gt_depth"], "gt_depth", sample_id)
        rgb = read_rgb(item["rgb"])
        input_depth = read_depth(item["input_depth"], item["input_depth_unit"])
        gt = read_depth(item["gt_depth"], item.get("gt_depth_unit", "m"))
        if rgb.shape[:2] != input_depth.shape or rgb.shape[:2] != gt.shape:
            raise ValueError(f"{sample_id}: RGB/input/GT shapes differ: {rgb.shape[:2]}, {input_depth.shape}, {gt.shape}")
        if item.get("intrinsics"):
            require_file(item["intrinsics"], "intrinsics", sample_id)
            read_intrinsics(item["intrinsics"])
        if item.get("transparent_mask"):
            require_file(item["transparent_mask"], "transparent_mask", sample_id)
            if read_mask(item["transparent_mask"]).shape != gt.shape:
                raise ValueError(f"{sample_id}: transparent_mask is not aligned to GT")
        valid_input = float((input_depth > 0).mean())
        valid_gt = float((gt > 0).mean())
        print(f"OK {sample_id}: shape={gt.shape}, input_coverage={valid_input:.4f}, gt_coverage={valid_gt:.4f}")
    print(f"Validated {len(samples)} sample(s) without GT leakage.")


if __name__ == "__main__":
    main()
