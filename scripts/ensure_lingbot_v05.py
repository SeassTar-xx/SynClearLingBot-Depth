#!/usr/bin/env python3
"""Ensure that the official LingBot-Depth v0.5 checkpoint is available locally."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ROOT", str(PROJECT_ROOT))).expanduser().resolve()
if ROOT != PROJECT_ROOT: raise RuntimeError(f"ROOT must identify this repository: expected {PROJECT_ROOT}, got {ROOT}")
TARGET = ROOT / "models/lingbot-depth-v0.5"; MODEL = TARGET / "model.pt"
MODEL_ID = "Robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
SOURCE = "https://www.modelscope.cn/Robbyant/lingbot-depth-pretrain-vitl-14-v0.5.git"
EXPECTED_SIZE = 1284837952
EXPECTED_SHA256 = "b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23"
def project_path(path):
    resolved = path.resolve(strict=False)
    if ROOT not in (resolved, *resolved.parents): raise RuntimeError(f"path outside project: {path}")
    return resolved
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
def verify(path):
    if not path.is_file() or path.is_symlink(): raise RuntimeError(f"missing or symlink checkpoint: {path}")
    size, digest = path.stat().st_size, sha256(path)
    if size != EXPECTED_SIZE or digest != EXPECTED_SHA256: raise RuntimeError(f"checkpoint failed official identity check: size={size}, sha256={digest}")
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {"model", "model_config"}
        if not required.issubset(payload): raise RuntimeError(f"checkpoint keys missing: {required-set(payload)}")
    except Exception as exc: raise RuntimeError("checkpoint cannot be loaded with torch weights_only=True") from exc
    return {"size_bytes": size, "sha256": digest}
def main():
    parser = argparse.ArgumentParser(description="确保官方 LingBot-Depth v0.5 模型存在"); parser.add_argument("--direct", action="store_true", help="克隆时忽略环境代理")
    args = parser.parse_args()
    project_path(TARGET.parent); TARGET.parent.mkdir(parents=True, exist_ok=True)
    if MODEL.exists(): details = verify(MODEL); print(json.dumps({"state": "present_verified", "checkpoint": str(MODEL), **details}, ensure_ascii=False))
    else:
        if TARGET.exists() and TARGET.is_symlink(): raise RuntimeError(f"refuse symlink model directory: {TARGET}")
        TARGET.mkdir(parents=True, exist_ok=True)
        temporary = TARGET.parent / f".lingbot-v05-download-{os.getpid()}"; project_path(temporary)
        if temporary.exists(): shutil.rmtree(temporary)
        env = os.environ.copy()
        if args.direct:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"): env.pop(key, None)
        try:
            subprocess.run(["git", "clone", "--depth", "1", SOURCE, str(temporary)], check=True, env=env)
            details = verify(temporary / "model.pt")
            os.replace(temporary / "model.pt", MODEL)
            print(json.dumps({"state": "downloaded_verified", "checkpoint": str(MODEL), **details}, ensure_ascii=False))
        finally:
            if temporary.exists(): shutil.rmtree(temporary)
    active = ROOT / "models/ACTIVE_MODEL.txt"; project_path(active)
    active.write_text("\n".join(["model_name=LingBot-Depth-v0.5", f"model_id={MODEL_ID}", f"checkpoint={MODEL}", f"sha256={EXPECTED_SHA256}", f"activated_at={time.strftime('%FT%TZ', time.gmtime())}", ""]), encoding="utf-8")
if __name__ == "__main__": main()
