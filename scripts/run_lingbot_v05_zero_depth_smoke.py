#!/usr/bin/env python3
"""Run reproducible LingBot-Depth v0.5 completion smoke tests without editing official code."""
import argparse, json, os, shutil, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read RGB: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0)


def load_depth_meters(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"cannot read depth: {path}")
    if raw.dtype == np.uint16:
        depth = raw.astype(np.float32) / 1000.0
        unit = "uint16 millimetres converted to metres"
    else:
        depth = raw.astype(np.float32)
        unit = str(raw.dtype)
    return depth, unit


def to_visual(depth):
    x = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(x) & (x > 0)
    gray = np.zeros(x.shape, np.uint8)
    if valid.any():
        lo, hi = np.percentile(x[valid], [2, 98])
        hi = max(hi, lo + 1e-6)
        gray[valid] = np.clip((x[valid] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def depth_u16(depth):
    x = np.asarray(depth, dtype=np.float32)
    out = np.zeros(x.shape, np.uint16)
    valid = np.isfinite(x) & (x > 0)
    out[valid] = np.clip(np.rint(x[valid] * 1000.0), 0, 65535).astype(np.uint16)
    return out


def write_ply(path, points, rgb, limit=200000):
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    c = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    keep = np.isfinite(p).all(axis=1)
    p, c = p[keep], c[keep]
    if len(p) > limit:
        idx = np.linspace(0, len(p) - 1, limit, dtype=np.int64)
        p, c = p[idx], c[idx]
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(p)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, color in zip(p, c):
            f.write(f"{xyz[0]:.7g} {xyz[1]:.7g} {xyz[2]:.7g} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def run_case(model, image, rgb, depth, out_dir, args, intrinsics=None, apply_mask=False, label=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.current_rgb, out_dir / "input_rgb.png")
    np.save(out_dir / "input_depth_m.npy", depth)
    cv2.imwrite(str(out_dir / "input_depth_mm.png"), depth_u16(depth))
    cv2.imwrite(str(out_dir / "input_depth_viz.png"), to_visual(depth))
    depth_t = torch.from_numpy(depth).to(device=args.device)
    if intrinsics is None:
        k_t = None
    else:
        # The official README expects normalized K and a batch dimension.
        k_norm = intrinsics.astype(np.float32).copy()
        k_norm[0, :] /= depth.shape[1]
        k_norm[1, :] /= depth.shape[0]
        k_t = torch.from_numpy(k_norm).unsqueeze(0).to(device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    out = model.infer(image, depth_in=depth_t, intrinsics=k_t, resolution_level=args.resolution_level,
                      apply_mask=apply_mask, use_fp16=not args.disable_amp, enable_depth_mask=False)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    pred = out["depth"].detach().float().cpu().numpy()
    np.save(out_dir / "pred_depth_m.npy", pred)
    cv2.imwrite(str(out_dir / "pred_depth_mm.png"), depth_u16(pred))
    cv2.imwrite(str(out_dir / "pred_depth_viz.png"), to_visual(pred))
    mask = out.get("mask")
    mask_fraction = None
    if mask is not None:
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        np.save(out_dir / "pred_mask.npy", mask_np)
        cv2.imwrite(str(out_dir / "pred_mask.png"), mask_np * 255)
        mask_fraction = float(mask_np.mean())
    if out.get("points") is not None:
        pts = out["points"].detach().float().cpu().numpy()
        np.save(out_dir / "points.npy", pts)
        write_ply(out_dir / "points.ply", pts, rgb)
    finite = np.isfinite(pred) & (pred > 0)
    summary = {
        "label": label, "checkpoint": str(args.checkpoint), "device": args.device,
        "resolution_level": args.resolution_level, "apply_mask": apply_mask,
        "enable_depth_mask": False, "intrinsics_supplied": intrinsics is not None,
        "input_depth_valid_fraction": float(np.isfinite(depth).astype(np.float32)[depth > 0].sum() / depth.size),
        "output_depth_valid_fraction": float(finite.mean()), "output_depth_min_m": float(pred[finite].min()) if finite.any() else None,
        "output_depth_max_m": float(pred[finite].max()) if finite.any() else None,
        "output_mask_true_fraction": mask_fraction, "elapsed_seconds": elapsed,
        "output_keys": sorted(out.keys()),
    }
    write_json(out_dir / "summary.json", summary)
    return pred, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resolution-level", type=int, default=9)
    ap.add_argument("--disable-amp", action="store_true")
    ap.add_argument("--clear-rgb", type=Path, action="append", default=[])
    args = ap.parse_args()
    if not args.checkpoint.is_file(): raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available() and args.device.startswith("cuda"): raise RuntimeError("CUDA is not available")
    sys.path.insert(0, str(args.repo))
    from mdm.model.v2 import MDMModel
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    model = MDMModel.from_pretrained(args.checkpoint).to(args.device).eval()
    model.enable_pytorch_native_sdpa()
    config = torch.load(args.checkpoint, map_location="cpu", weights_only=True).get("model_config", {})
    write_json(args.out / "model_config.json", config)
    write_json(args.out / "runtime.json", {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "xformers_disabled": os.environ.get("XFORMERS_DISABLED"),
        "checkpoint": str(args.checkpoint), "checkpoint_bytes": args.checkpoint.stat().st_size,
        "native_sdpa_enabled": True, "depth_masking_enabled": False})
    ex = args.repo / "examples" / "0"
    args.current_rgb = ex / "rgb.png"
    rgb, image = load_rgb(args.current_rgb)
    image = image.to(args.device)
    raw_depth, depth_unit = load_depth_meters(ex / "raw_depth.png")
    k = np.loadtxt(ex / "intrinsics.txt", dtype=np.float32)
    write_json(args.out / "official_example_input.json", {"rgb": str(args.current_rgb), "raw_depth": str(ex / "raw_depth.png"), "depth_interpretation": depth_unit, "intrinsics": k.tolist()})
    try:
        model.infer(image, depth_in=None, resolution_level=args.resolution_level, apply_mask=False, enable_depth_mask=False)
        none_result = {"unexpected": "depth_in=None completed"}
    except Exception as exc:
        none_result = {"expected_failure": True, "exception_type": type(exc).__name__, "message": str(exc)}
    write_json(args.out / "control_depth_none.json", none_result)
    standard, _ = run_case(model, image, rgb, raw_depth, args.out / "official_standard_raw_depth_K", args, k, False, "official raw depth + K")
    zeros = np.zeros_like(raw_depth, dtype=np.float32)
    zero_no_k, _ = run_case(model, image, rgb, zeros, args.out / "official_zero_depth_no_K", args, None, False, "official RGB + all-zero depth, no K")
    zero_k, _ = run_case(model, image, rgb, zeros, args.out / "official_zero_depth_with_K", args, k, False, "official RGB + all-zero depth + K")
    zero_mask, _ = run_case(model, image, rgb, zeros, args.out / "official_zero_depth_apply_mask", args, None, True, "official RGB + all-zero depth, apply_mask=True")
    finite_pair = np.isfinite(zero_no_k) & np.isfinite(zero_k)
    compare = {"finite_overlap_fraction": float(finite_pair.mean()),
               "max_abs_depth_difference_m": float(np.max(np.abs(zero_no_k[finite_pair] - zero_k[finite_pair]))) if finite_pair.any() else None}
    write_json(args.out / "zero_depth_K_comparison.json", compare)
    for rgb_path in args.clear_rgb:
        args.current_rgb = rgb_path
        clear_rgb, clear_image = load_rgb(rgb_path)
        clear_image = clear_image.to(args.device)
        clear_zeros = np.zeros(clear_rgb.shape[:2], dtype=np.float32)
        safe = rgb_path.parent.name
        run_case(model, clear_image, clear_rgb, clear_zeros, args.out / f"cleardepth_{safe}_zero_depth_no_K", args, None, False, f"ClearDepth RGB + all-zero depth: {safe}")

if __name__ == "__main__":
    main()
