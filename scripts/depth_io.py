"""Shared, dependency-light I/O helpers for formal ClearDepth evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line and not line.startswith("#"):
                item = json.loads(line)
                if "sample_id" not in item:
                    raise ValueError(f"{path}:{line_no} has no sample_id")
                yield item


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_depth(path: str | Path, unit: str = "m") -> np.ndarray:
    """Read a 2-D depth map and convert it to metres when requested."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read depth map: {path}")
    if depth.ndim == 3:
        if depth.shape[-1] != 3 or not np.allclose(depth[..., 0], depth[..., 1]) or not np.allclose(depth[..., 0], depth[..., 2]):
            raise ValueError(f"Expected a scalar/repeated-channel depth map: {path}, shape={depth.shape}")
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth map: {path}, shape={depth.shape}")
    depth = depth.astype(np.float32, copy=False)
    if unit == "mm":
        depth = depth / 1000.0
    elif unit != "m":
        raise ValueError(f"Unsupported depth unit {unit!r}; use m or mm")
    return depth


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def read_intrinsics(value: str | list[list[float]] | np.ndarray) -> np.ndarray:
    matrix = np.loadtxt(value, dtype=np.float32) if isinstance(value, str) else np.asarray(value, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"Intrinsics must be 3x3, received {matrix.shape}")
    return matrix


def valid_depth_mask(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    return np.isfinite(prediction) & np.isfinite(ground_truth) & (prediction > 0) & (ground_truth > 0)
