#!/usr/bin/env python3
"""确保按固定 test_samples 清单选择性保存 SynClearDepth 子集。"""
from __future__ import annotations
import argparse
import concurrent.futures
import io
import json
import os
import time
import urllib.request
import zipfile
import zlib

from eval_common import RAW_SUBSET, ROOT, SPLIT_FILE, ensure_directory, load_samples, safe_member_path

ARCHIVE_URL = "https://tams.informatik.uni-hamburg.de/research/datasets/cleardepth_dataset/transparent_dataset1.zip"
REQUIRED_FIELDS = ("left_rgb_member", "right_rgb_member", "gt_depth_member", "segmentation_member", "scene_info_member")

class RangeReader(io.RawIOBase):
    def __init__(self, url, size, timeout, direct):
        self.url, self.size, self.timeout = url, size, timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
        self.position = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=io.SEEK_SET):
        self.position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset
        self.position = max(0, min(self.size, self.position)); return self.position
    def readinto(self, buffer):
        if self.position >= self.size: return 0
        end = min(self.size - 1, self.position + len(buffer) - 1)
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}", "User-Agent": "SynClearLingBot-Depth/1.0"})
        with self.opener.open(request, timeout=self.timeout) as response:
            data = response.read()
            if response.status != 206: raise RuntimeError(f"server did not honor Range request (HTTP {response.status})")
        buffer[:len(data)] = data; self.position += len(data)
        return len(data)

def member_set(samples):
    members = set()
    for row in samples:
        for field in REQUIRED_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and value: members.add(value)
            else: raise ValueError(f"{row['sample_id']} lacks required field {field}")
    return sorted(members)

def existing_missing(members):
    return [member for member in members if not (lambda p: p.is_file() and not p.is_symlink() and p.stat().st_size > 0)(safe_member_path(RAW_SUBSET, member))]

def fetch_member(reader, info, member, retries):
    target = safe_member_path(RAW_SUBSET, member)
    ensure_directory(target.parent)
    if target.exists() and not target.is_symlink() and target.stat().st_size == info.file_size:
        with target.open("rb") as handle:
            crc = 0
            for block in iter(lambda: handle.read(1024 * 1024), b""): crc = zlib.crc32(block, crc)
        if (crc & 0xffffffff) == info.CRC: return "verified"
    part = target.with_name(target.name + ".part")
    if part.exists() and not part.is_symlink(): part.unlink()
    for attempt in range(1, retries + 1):
        try:
            data = reader.open(info).read()
            if len(data) != info.file_size or (zlib.crc32(data) & 0xffffffff) != info.CRC: raise RuntimeError("size or CRC mismatch")
            with part.open("wb") as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            os.replace(part, target); return "downloaded"
        except Exception:
            if part.exists(): part.unlink()
            if attempt == retries: raise
            time.sleep(attempt)

def main():
    parser = argparse.ArgumentParser(description="按固定样本 ID 确保 SynClearDepth 选择性子集存在")
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0); parser.add_argument("--direct", action="store_true", help="忽略环境代理，直连官方 UHH 地址")
    args = parser.parse_args()
    ensure_directory(RAW_SUBSET)
    samples, members = load_samples(SPLIT_FILE), None
    members = member_set(samples); missing = existing_missing(members)
    print(json.dumps({"samples": len(samples), "required_members": len(members), "already_present": len(members)-len(missing), "missing": len(missing)}, ensure_ascii=False))
    if not missing:
        print("SynClearDepth 子集完整，未发起网络下载。"); return
    request = urllib.request.Request(ARCHIVE_URL, method="HEAD", headers={"User-Agent": "SynClearLingBot-Depth/1.0"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.direct else urllib.request.build_opener()
    with opener.open(request, timeout=args.timeout) as response:
        size, ranges = int(response.headers["Content-Length"]), response.headers.get("Accept-Ranges", "")
    if "bytes" not in ranges.lower(): raise RuntimeError("官方源未声明支持 HTTP Range；为避免下载完整 ZIP 已停止。")
    reader = zipfile.ZipFile(RangeReader(ARCHIVE_URL, size, args.timeout, args.direct))
    infos = {info.filename: info for info in reader.infolist()}
    absent = [member for member in missing if member not in infos]
    if absent: raise RuntimeError(f"清单成员不在官方 ZIP 中，示例：{absent[:3]}")
    log_path = ROOT / "logs/syncleardepth_subset_download.log"; ensure_directory(log_path.parent)
    def one(member): return member, fetch_member(reader, infos[member], member, args.retries), infos[member].file_size
    downloaded = verified = 0; started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, log_path.open("a", encoding="utf-8") as log:
        for index, (member, state, count) in enumerate(pool.map(one, missing), 1):
            downloaded += state == "downloaded"; verified += state == "verified"
            log.write(json.dumps({"time": time.strftime("%FT%TZ", time.gmtime()), "member": member, "state": state, "bytes": count}, ensure_ascii=False) + "\n"); log.flush()
            if index % 25 == 0 or index == len(missing): print(f"{index}/{len(missing)} 缺失成员已处理")
    reader.close(); print(json.dumps({"downloaded": downloaded, "verified": verified, "seconds": round(time.time()-started, 1)}, ensure_ascii=False))
if __name__ == "__main__": main()
