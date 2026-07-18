# Formal SynClearDepth evaluation protocol

## Goal

Evaluate LingBot-Depth v0.5 as a depth refinement/completion stage on SynClearDepth without presenting GT depth to the model.

## Required per-sample fields

Use `configs/cleardepth_manifest.example.jsonl` as the schema. Every JSONL row requires:

- `sample_id`
- `rgb`: rectified **left** RGB image
- `input_depth`: metric depth aligned to that left RGB, derived independently of GT
- `input_depth_unit`: `m` or `mm`
- `gt_depth`: left-view ClearDepth GT, used only by the evaluator

Optional fields are `transparent_mask`, `intrinsics`, `intrinsics_normalized`, and `gt_depth_unit`.

## Input construction

The recommended initial protocol is an independent calibrated stereo baseline:

1. Rectify left/right RGB using the camera calibration.
2. Estimate left-view disparity with a stereo method independent of ClearDepth.
3. Convert disparity to metric depth, `Z = fB/d`.
4. Mark invalid or occluded estimates as zero/NaN; preserve the remaining noisy/incomplete depth.
5. Put this result in `input_depth`; retain ClearDepth GT exclusively in `gt_depth`.

Do not use GT depth as `input_depth`, including a GT map with randomly removed pixels, for the main benchmark. That is a separate synthetic ablation and must be labelled as such.

## Commands

```bash
cd /mnt/20t/xuxin/depth_io_inspection
source scripts/env.sh
export OPENCV_IO_ENABLE_OPENEXR=1

python scripts/validate_project.py --manifest data/cleardepth/formal_eval.jsonl
python scripts/run_lingbot_on_cleardepth.py \
  --manifest data/cleardepth/formal_eval.jsonl \
  --checkpoint models/lingbot-depth-v0.5/v0.5/model.pt \
  --official-repo repos/lingbot-depth \
  --output-dir outputs/formal_eval \
  --device cuda:0 --resolution-level 9
python scripts/evaluate_depth.py \
  --prediction-index outputs/formal_eval/predictions.jsonl \
  --output-dir outputs/formal_eval/metrics \
  --scale-alignment none
```

The default inference path retains LingBot depth-token masking. `--native-sdpa` or `--disable-depth-masking` are compatibility switches and must be recorded if used; they are not the default formal protocol.

## Metrics

The evaluator computes whole-image MAE, RMSE, AbsRel, delta1, and valid-pixel coverage. If `transparent_mask` is supplied as a binary transparent-object mask, it also computes transparent-region and background metrics. It writes `metrics.json` and `metrics.csv`.

For a strict comparison to the ClearDepth paper, additionally convert predicted metric depth to disparity with the correct `fB`, then implement its disparity metrics (AvgErr, RMS, Bad-0.5/1/2/4) under the same calibration, crop, and mask policy.
