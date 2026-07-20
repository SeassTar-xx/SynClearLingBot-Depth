#!/usr/bin/env python3
"""Shared read-only and path-safety utilities for the evaluation pipeline."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ROOT", str(PROJECT_ROOT))).expanduser().resolve()
if ROOT != PROJECT_ROOT:
    raise RuntimeError(f"ROOT must identify this repository: expected {PROJECT_ROOT}, got {ROOT}")
RAW_SUBSET = ROOT / "data/syncleardepth/raw_subset"
SPLIT_FILE = ROOT / "data/syncleardepth/test_split/test_samples.jsonl"
RAW_CACHE = ROOT / "data/syncleardepth/normalized/raw_depth_sgbm_fx500_v2"
CALIBRATION = {
    "baseline_m": 0.12,
    "fx_px": 500.0,
    "fy_px": 497.36842105263156,
    "cx_px": 640.0,
    "cy_px": 360.0,
    "focal_length_mm": 2.1,
    "image_width": 1280,
    "image_height": 720,
}
SGBM = {
    "min_disparity": 0,
    "num_disparities": 320,
    "block_size": 5,
    "p1": 200,
    "p2": 800,
    "disp12_max_diff": 1,
    "uniqueness_ratio": 8,
    "speckle_window_size": 100,
    "speckle_range": 2,
    "mode": "SGBM_3WAY",
}


def require_project_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"path escapes project root: {path}") from exc
    return resolved


def ensure_directory(path: Path) -> Path:
    require_project_path(path)
    current = ROOT
    for part in path.relative_to(ROOT).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"refuse symlink in writable path: {current}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_member_path(base: Path, member: str) -> Path:
    pp = PurePosixPath(member)
    if pp.is_absolute() or ".." in pp.parts or not pp.parts:
        raise ValueError(f"unsafe archive member: {member!r}")
    target = base.joinpath(*pp.parts)
    require_project_path(target)
    current = base
    for part in pp.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise RuntimeError(f"refuse symlink component: {current}")
    return target


def load_samples(path: Path = SPLIT_FILE) -> list[dict[str, Any]]:
    require_project_path(path)
    samples = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError(f"invalid or duplicated sample_id at line {number}")
            seen.add(sample_id)
            samples.append(row)
    if not samples:
        raise ValueError(f"no samples in {path}")
    return sorted(samples, key=lambda row: row["sample_id"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def raw_cache_file(sample_id: str, cache_root: Path = RAW_CACHE) -> Path:
    pp = PurePosixPath(sample_id)
    if pp.is_absolute() or ".." in pp.parts or not pp.parts:
        raise ValueError(f"unsafe sample id: {sample_id!r}")
    return require_project_path(cache_root.joinpath(*pp.parts).with_suffix(".npz"))


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read RGB image: {path}")
    return image


def read_depth_exr(path: Path) -> np.ndarray:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"cannot read EXR depth: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return np.asarray(depth, dtype=np.float32)


def stereo_raw_depth(left_bgr: np.ndarray, right_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if left_bgr.shape[:2] != right_bgr.shape[:2]:
        raise ValueError(f"stereo size mismatch: {left_bgr.shape} vs {right_bgr.shape}")
    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right = right_bgr if right_bgr.ndim == 2 else cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    matcher = cv2.StereoSGBM_create(
        minDisparity=SGBM["min_disparity"],
        numDisparities=SGBM["num_disparities"],
        blockSize=SGBM["block_size"],
        P1=SGBM["p1"], P2=SGBM["p2"],
        disp12MaxDiff=SGBM["disp12_max_diff"],
        uniquenessRatio=SGBM["uniqueness_ratio"],
        speckleWindowSize=SGBM["speckle_window_size"],
        speckleRange=SGBM["speckle_range"],
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(left, right).astype(np.float32) / 16.0
    valid = np.isfinite(disparity) & (disparity > 1.0) & (disparity < SGBM["num_disparities"] - 2)
    raw = np.zeros(disparity.shape, dtype=np.float32)
    raw[valid] = CALIBRATION["fx_px"] * CALIBRATION["baseline_m"] / disparity[valid]
    raw[~np.isfinite(raw)] = 0.0
    return raw, valid


def cache_configuration() -> dict[str, Any]:
    return {
        "format_version": 2,
        "calibration": CALIBRATION,
        "stereo_sgbm": SGBM,
        "split_sha256": sha256_file(SPLIT_FILE),
    }


def metric_masks(pred: np.ndarray, gt: np.ndarray, segmentation: np.ndarray | None = None) -> dict[str, np.ndarray]:
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0.0) & (gt > 0.1) & (gt < 20.0)
    result = {"all": valid}
    if segmentation is not None:
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        if segmentation.shape != gt.shape:
            raise ValueError("segmentation/depth size mismatch")
        transparent = valid & (segmentation > 0)
        result["transparent"] = transparent
        result["background"] = valid & ~transparent
    return result


def error_sums(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    err = pred[mask].astype(np.float64) - gt[mask].astype(np.float64)
    return {"count": int(err.size), "abs_sum": float(np.abs(err).sum()), "sq_sum": float(np.square(err).sum())}


def aggregate_sums(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    count = sum(int(row[key]["count"]) for row in rows)
    abs_sum = sum(float(row[key]["abs_sum"]) for row in rows)
    sq_sum = sum(float(row[key]["sq_sum"]) for row in rows)
    return {"pixel_count": count, "rmse_m": float(np.sqrt(sq_sum / count)) if count else None, "mae_m": abs_sum / count if count else None}
