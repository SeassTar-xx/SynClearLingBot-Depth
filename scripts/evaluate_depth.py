#!/usr/bin/env python3
"""Evaluate metric-depth predictions against GT depth from a prediction JSONL index."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from depth_io import read_depth, read_jsonl, read_mask, valid_depth_mask


def compute(prediction: np.ndarray, ground_truth: np.ndarray, region: np.ndarray | None = None) -> dict:
    valid = valid_depth_mask(prediction, ground_truth)
    if region is not None:
        valid &= region
    total = int(region.sum()) if region is not None else prediction.size
    if not valid.any():
        return {"valid_pixels": 0, "coverage": 0.0, "mae_m": None, "rmse_m": None, "absrel": None, "delta1": None}
    pred, gt = prediction[valid], ground_truth[valid]
    ratio = np.maximum(pred / gt, gt / pred)
    error = pred - gt
    return {
        "valid_pixels": int(valid.sum()), "coverage": float(valid.sum() / total),
        "mae_m": float(np.mean(np.abs(error))), "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "absrel": float(np.mean(np.abs(error) / gt)), "delta1": float(np.mean(ratio < 1.25)),
    }


def concatenate_metric(predictions: list[np.ndarray], ground_truths: list[np.ndarray]) -> dict:
    if not predictions:
        return {"valid_pixels": 0, "coverage": None, "mae_m": None, "rmse_m": None, "absrel": None, "delta1": None}
    return compute(np.concatenate(predictions), np.concatenate(ground_truths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-index", type=Path, required=True, help="predictions.jsonl produced by run_lingbot_on_cleardepth.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-alignment", choices=["none", "median"], default="none")
    args = parser.parse_args()
    rows = list(read_jsonl(args.prediction_index))
    if not rows:
        raise ValueError("Prediction index has no rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample, aggregates = [], {"all": ([], []), "transparent": ([], []), "background": ([], [])}

    for row in rows:
        if not row.get("gt_depth"):
            raise ValueError(f"{row['sample_id']} has no gt_depth")
        pred = np.load(row["prediction"]).astype(np.float32)
        gt = read_depth(row["gt_depth"], row.get("gt_depth_unit", "m"))
        if pred.shape != gt.shape:
            raise ValueError(f"{row['sample_id']}: prediction {pred.shape} and GT {gt.shape} differ")
        scale = 1.0
        valid = valid_depth_mask(pred, gt)
        if args.scale_alignment == "median":
            if not valid.any():
                raise ValueError(f"{row['sample_id']}: no pixels for median scale alignment")
            scale = float(np.median(gt[valid]) / np.median(pred[valid]))
            pred = pred * scale
        regions: dict[str, np.ndarray | None] = {"all": None}
        if row.get("transparent_mask"):
            transparent = read_mask(row["transparent_mask"])
            if transparent.shape != gt.shape:
                raise ValueError(f"{row['sample_id']}: transparent mask {transparent.shape} and GT {gt.shape} differ")
            regions.update({"transparent": transparent, "background": ~transparent})
        sample = {"sample_id": row["sample_id"], "scale": scale}
        for name, region in regions.items():
            sample[name] = compute(pred, gt, region)
            selected = valid_depth_mask(pred, gt) if region is None else valid_depth_mask(pred, gt) & region
            if selected.any():
                aggregates[name][0].append(pred[selected])
                aggregates[name][1].append(gt[selected])
        per_sample.append(sample)

    aggregate = {name: concatenate_metric(*values) for name, values in aggregates.items()}
    result = {
        "protocol": {"scale_alignment": args.scale_alignment, "valid_mask": "finite(pred) & finite(gt) & (pred > 0) & (gt > 0)"},
        "num_samples": len(per_sample), "per_sample": per_sample, "aggregate_pixel_weighted": aggregate,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = ["sample_id", "region", "scale", "valid_pixels", "coverage", "mae_m", "rmse_m", "absrel", "delta1"]
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in per_sample:
            for region, values in sample.items():
                if region not in {"all", "transparent", "background"}:
                    continue
                writer.writerow({"sample_id": sample["sample_id"], "region": region, "scale": sample["scale"], **values})
        for region, values in aggregate.items():
            writer.writerow({"sample_id": "pixel_weighted_aggregate", "region": region, "scale": None, **values})
    print(json.dumps(result["aggregate_pixel_weighted"], indent=2))


if __name__ == "__main__":
    main()
