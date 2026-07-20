#!/usr/bin/env python3
"""Cache StereoSGBM raw depth by fixed sample ID without recomputing matching entries."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from eval_common import RAW_CACHE, RAW_SUBSET, ROOT, cache_configuration, canonical_json_sha256, ensure_directory, load_samples, raw_cache_file, read_rgb, safe_member_path, stereo_raw_depth


def check_cached(path: Path, expected_fingerprint: str) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            raw, valid = archive["raw"], archive["valid"]
            fingerprint = str(archive["config_fingerprint"].item())
        return raw.dtype == np.float32 and raw.ndim == 2 and valid.shape == raw.shape and fingerprint == expected_fingerprint
    except Exception:
        return False


def write_cache(path: Path, raw: np.ndarray, valid: np.ndarray, fingerprint: str) -> None:
    ensure_directory(path.parent)
    part = path.with_name(path.name + ".part")
    if part.exists():
        if part.is_symlink(): raise RuntimeError(f"refuse symlink temporary file: {part}")
        part.unlink()
    with part.open("wb") as handle:
        np.savez_compressed(handle, raw=raw.astype(np.float32, copy=False), valid=valid.astype(np.uint8, copy=False), config_fingerprint=np.asarray(fingerprint))
        handle.flush(); os.fsync(handle.fileno())
    os.replace(part, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="确保固定 SynClearDepth 子集的 raw depth 缓存存在")
    parser.add_argument("--cache", type=Path, default=RAW_CACHE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cache = args.cache.resolve(strict=False)
    if ROOT not in (cache, *cache.parents): raise SystemExit(f"缓存必须位于项目根目录：{cache}")
    if cache.exists() and cache.is_symlink(): raise SystemExit(f"拒绝符号链接缓存目录：{cache}")
    ensure_directory(cache)
    config = cache_configuration(); fingerprint = canonical_json_sha256(config)
    config_path = cache / "cache_config.json"
    if config_path.exists():
        if config_path.is_symlink(): raise SystemExit(f"拒绝符号链接配置：{config_path}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if canonical_json_sha256(previous) != fingerprint:
            raise SystemExit("已存在的 raw depth 缓存参数或样本清单不同；为避免静默混用，未覆盖。请使用新的缓存目录版本。")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = load_samples()
    pending = [row for row in rows if not check_cached(raw_cache_file(row["sample_id"], cache), fingerprint)]
    print(json.dumps({"samples": len(rows), "cache": str(cache), "present_valid": len(rows)-len(pending), "pending": len(pending), "config_fingerprint": fingerprint}, ensure_ascii=False))
    if args.check_only or not pending:
        return
    usage = shutil.disk_usage(ROOT)
    conservative_bytes = len(pending) * 1280 * 720 * 5
    if usage.free < conservative_bytes + 2 * 1024**3:
        raise SystemExit(f"磁盘空间不足：可用 {usage.free} bytes，保守缓存需求约 {conservative_bytes} bytes 加 2 GiB 余量。")
    def build(row):
        left = read_rgb(safe_member_path(RAW_SUBSET, row["left_rgb_member"]))
        right = cv2.imread(str(safe_member_path(RAW_SUBSET, row["right_rgb_member"])), cv2.IMREAD_GRAYSCALE)
        if right is None: raise RuntimeError(f"cannot read right RGB: {row['right_rgb_member']}")
        raw, valid = stereo_raw_depth(left, right)
        write_cache(raw_cache_file(row["sample_id"], cache), raw, valid, fingerprint)
        return row["sample_id"], int(valid.sum()), int(raw.nbytes + valid.nbytes)
    started = time.time(); total_valid = total_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, (sample_id, valid_pixels, bytes_written) in enumerate(pool.map(build, pending), 1):
            total_valid += valid_pixels; total_bytes += bytes_written
            if index % 25 == 0 or index == len(pending): print(f"{index}/{len(pending)} raw depth 已缓存")
    actual_bytes = sum(path.stat().st_size for path in cache.rglob("*.npz") if not path.is_symlink())
    manifest = {"cache": str(cache.relative_to(ROOT)), "config": config, "config_fingerprint": fingerprint, "total_samples": len(rows), "cached_samples": len(rows), "pending_processed": len(pending), "cache_disk_bytes": actual_bytes, "created_at": time.strftime("%FT%TZ", time.gmtime())}
    (cache / "raw_depth_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"processed": len(pending), "valid_pixels": total_valid, "uncompressed_bytes_written": total_bytes, "cache_disk_bytes": actual_bytes, "seconds": round(time.time()-started, 1)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
