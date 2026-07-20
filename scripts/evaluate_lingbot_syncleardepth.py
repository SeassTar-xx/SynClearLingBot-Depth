#!/usr/bin/env python3
"""Reproduce LingBot-Depth RMSE/MAE on SynClearDepth using cached raw depth."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time
from pathlib import Path
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
import torch
from eval_common import CALIBRATION, RAW_CACHE, RAW_SUBSET, ROOT, SGBM, aggregate_sums, cache_configuration, error_sums, load_samples, metric_masks, raw_cache_file, read_depth_exr, read_rgb, safe_member_path

MODEL = ROOT / "models/lingbot-depth-v0.5/model.pt"

def load_raw(path: Path):
    if not path.is_file() or path.is_symlink(): raise RuntimeError(f"缺少 raw depth 缓存：{path}；请先执行 ensure_raw_depth.py")
    with np.load(path, allow_pickle=False) as archive:
        return archive["raw"].astype(np.float32, copy=False), archive["valid"].astype(bool, copy=False)

def aggregate(records):
    valid=[record for record in records if record.get("status")=="ok"]
    output={"completed":len(valid)}
    for region in ("all","transparent","background"):
        rows=[]
        for record in valid:
            pixels = record[f"{region}_pixels"]
            abs_sum = record.get(f"{region}_abs_sum")
            sq_sum = record.get(f"{region}_sq_sum")
            if abs_sum is None: abs_sum = record[f"{region}_mae_m"] * pixels
            if sq_sum is None: sq_sum = record[f"{region}_rmse_m"] ** 2 * pixels
            rows.append({region:{"count":pixels,"abs_sum":abs_sum,"sq_sum":sq_sum}})
        output[region]=aggregate_sums(rows, region)
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"runs/lingbot_v05_syncleardepth_test_20pct")
    parser.add_argument("--raw-cache", type=Path, default=RAW_CACHE)
    parser.add_argument("--max-samples",type=int,default=0); parser.add_argument("--resolution-level",type=int,default=9)
    parser.add_argument("--device",default="cuda:0"); parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(); output=args.output.resolve(strict=False); cache=args.raw_cache.resolve(strict=False)
    if ROOT not in (output,*output.parents) or ROOT not in (cache,*cache.parents): raise SystemExit("输出和缓存必须位于项目根目录内")
    output.mkdir(parents=True,exist_ok=True)
    rows=load_samples(); rows=sorted(rows,key=lambda row:hashlib.sha256((str(20260718)+":"+row["sample_id"]).encode()).hexdigest())
    if args.max_samples: rows=rows[:args.max_samples]
    results_path=output/"per_sample_metrics.jsonl"; done={}
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line: record=json.loads(line); done[record["sample_id"]]=record
    config={"dataset":"SynClearDepth","model_checkpoint":str(MODEL),"calibration":CALIBRATION,"raw_depth_cache":str(cache.relative_to(ROOT)),"raw_depth_cache_config":cache_configuration(),"raw_depth_formula":"raw_depth_m = fx_px * baseline_m / disparity_px; invalid disparity -> 0","stereo_sgbm":SGBM,"metric_mask":"finite GT depth in [0.1,20] m and finite positive prediction","apply_mask":False,"resolution_level":args.resolution_level,"seed":20260718}
    (output/"run_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    pending=[row for row in rows if done.get(row["sample_id"],{}).get("status")!="ok"]
    if pending:
        if not MODEL.is_file() or MODEL.is_symlink(): raise SystemExit(f"缺少模型：{MODEL}")
        if not torch.cuda.is_available() and args.device.startswith("cuda"): raise SystemExit("CUDA PyTorch 不可用")
        from mdm.model.v2 import MDMModel
        device=torch.device(args.device); model=MDMModel.from_pretrained(MODEL).to(device).eval(); torch.backends.cuda.matmul.allow_tf32=True
        intr=np.array([[CALIBRATION["fx_px"],0,CALIBRATION["cx_px"]],[0,CALIBRATION["fy_px"],CALIBRATION["cy_px"]],[0,0,1]],np.float32); intr[0,[0,2]]/=1280; intr[1,[1,2]]/=720
        intr_t=torch.from_numpy(intr).to(device).unsqueeze(0)
        mode="a" if args.resume else "w"
        with results_path.open(mode,encoding="utf-8") as handle:
            for index,row in enumerate(pending,1):
                started=time.time(); record={"sample_id":row["sample_id"],"scene":row["scene"],"sequence":row["sequence"],"status":"ok"}
                try:
                    left=read_rgb(safe_member_path(RAW_SUBSET,row["left_rgb_member"])); gt=read_depth_exr(safe_member_path(RAW_SUBSET,row["gt_depth_member"])); seg=cv2.imread(str(safe_member_path(RAW_SUBSET,row["segmentation_member"])),cv2.IMREAD_GRAYSCALE)
                    raw,stereo_valid=load_raw(raw_cache_file(row["sample_id"],cache))
                    if seg is None or left.shape[:2]!=gt.shape or raw.shape!=gt.shape or seg.shape!=gt.shape: raise RuntimeError("输入尺寸或读取失败")
                    rgb=cv2.cvtColor(left,cv2.COLOR_BGR2RGB); image_t=torch.from_numpy(rgb).to(device=device,dtype=torch.float32).permute(2,0,1).unsqueeze(0).div_(255); raw_t=torch.from_numpy(raw).to(device=device,dtype=torch.float32)
                    with torch.inference_mode(): pred=model.infer(image_t,depth_in=raw_t,resolution_level=args.resolution_level,apply_mask=False,intrinsics=intr_t)["depth"].squeeze().float().cpu().numpy()
                    masks=metric_masks(pred,gt,seg); record.update({"stereo_valid_fraction":float(stereo_valid.mean()),"raw_depth_median_m":float(np.median(raw[stereo_valid])) if stereo_valid.any() else None})
                    for region,mask in masks.items():
                        sums=error_sums(pred,gt,mask); record.update({f"{region}_pixels":sums["count"],f"{region}_abs_sum":sums["abs_sum"],f"{region}_sq_sum":sums["sq_sum"],f"{region}_mae_m":sums["abs_sum"]/sums["count"] if sums["count"] else None,f"{region}_rmse_m":(sums["sq_sum"]/sums["count"])**0.5 if sums["count"] else None})
                except Exception as exc:
                    record.update({"status":"failed","error":f"{type(exc).__name__}: {exc}"}); torch.cuda.empty_cache()
                record["elapsed_seconds"]=time.time()-started; handle.write(json.dumps(record,ensure_ascii=False)+"\n"); handle.flush(); os.fsync(handle.fileno())
                print(f"{index}/{len(pending)} {record['sample_id']} {record['status']} {record['elapsed_seconds']:.2f}s",flush=True)
    latest={}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line: record=json.loads(line); latest[record["sample_id"]]=record
    summary=aggregate(list(latest.values())); summary.update({"target_samples":len(rows),"failed_samples":sum(latest.get(row["sample_id"],{}).get("status")!="ok" for row in rows)})
    (output/"metrics_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (output/"metrics_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["region","pixels","mae_m","rmse_m"])
        for region in ("all","transparent","background"): writer.writerow([region,summary[region]["pixel_count"],summary[region]["mae_m"],summary[region]["rmse_m"]])
    print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__": main()
