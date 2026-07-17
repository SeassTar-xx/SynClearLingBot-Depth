# LingBot-Depth v0.5 upgrade and smoke report

## Scope and checkpoint identity

The active checkpoint is **LingBot-Depth-v0.5**, ModelScope ID `Robbyant/lingbot-depth-pretrain-vitl-14-v0.5`:

- Path: `models/lingbot-depth-v0.5/v0.5/model.pt`
- Size: 1,284,837,952 bytes
- SHA-256: `b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23`
- Active pointer: `models/ACTIVE_MODEL.txt`
- Download manifest: `manifests/lingbot_v05_checkpoint.json`

The earlier v0.1 checkpoint payload at `models/lingbot-depth-v0.5/model.pt` was intentionally deleted on 2026-07-17 after v0.5 verification and activation. Its former SHA-256 was `6ab1da5822e4fea712202616d1f3b683ce4b2f7f82ea58fb3f5ebd7cfae9c0e0`; `models/V01_CHECKPOINT_DELETED.txt` records the deletion. The nested `v0.5/` directory contains the only retained model payload.

Official code is pinned at commit `f3a237e434ae987bc38281476d6cfb5df3e4d739`; no file below `repos/lingbot-depth/` was edited.

## Runtime

The run used one idle NVIDIA A100 40 GB (GPU 0), Python 3.12.8, PyTorch `2.10.0+cu128`, CUDA 12.8. No system, shared conda environment, PyTorch, CUDA, or model directory was modified. All caches and outputs were redirected under this project root through `scripts/env.sh`.

The available xFormers package had incompatible CUDA extensions for this Torch build. The smoke wrapper therefore set `XFORMERS_DISABLED=1`, invoked the repository-provided `model.enable_pytorch_native_sdpa()`, and passed `enable_depth_mask=False`. This is a compatible native-PyTorch execution path, but it is an explicit smoke-test compatibility setting rather than a claim of the official xFormers/nested-token benchmark configuration. See `runs/lingbot_v05_smoke/CODE_PATH_ANALYSIS.md`.

## Code-path findings

- The released `MDMModel.forward()` asserts that `depth is not None`; it is RGB-D, not a released RGB-only inference API.
- Its RGB-D encoder maps `NaN`/`inf` and non-positive input depth to invalid/zero values.
- Intrinsics are only consumed after depth prediction to make `points`; they do not feed the transformer. The official README requires normalized K and a batch dimension.
- `apply_mask=True` replaces model-mask-rejected depth (and points when K exists) with `inf`; `False` returns the unmasked regression output.
- All-zero depth removes all depth patches under default depth-token masking. The native-SDPA fallback disables that masking to avoid unavailable xFormers nested-token operations; this limitation is material for interpreting the zero-depth test.

## Executed smoke matrix

| Case | Result | Key observation |
|---|---|---|
| `depth_in=None` control | Expected `AssertionError` | Confirms depth is required in this release. |
| Official example RGB + raw depth + normalized K | Passed | Output `depth`, `mask`, and `points`; 0.7355 s inference; predicted unmasked depth 0.9531–7.2729 m. |
| Official RGB + all-zero depth, no K | Passed | Output `depth` and `mask`; 0.1633 s; depth 0.8721–7.1563 m. |
| Official RGB + all-zero depth + K | Passed | Output adds `points`; depth matches the no-K run exactly (`max_abs_difference=0.0`). |
| Official RGB + all-zero depth, `apply_mask=True` | Passed | The predicted mask was all valid on this image, so the finite-depth outcome matched the unmasked case. |
| ClearDepth frame 0023 RGB + all-zero depth | Passed | Qualitative prediction only; 0.7188–1.1641 m. |
| ClearDepth frame 0038 RGB + all-zero depth | Passed | Qualitative prediction only; 0.6289–1.5000 m. |

Machine-readable summaries are stored with each case. The raw stdout/stderr were empty on the successful second attempt. The first failed wrapper attempt, caused by unnormalized/unbatched K in this project wrapper, is preserved at `logs/lingbot_v05_smoke_attempt1_wrapper_K_error.stderr.log`; the corrected run log is `logs/lingbot_v05_smoke.stderr.log`.

## ClearDepth interpretation and limits

The two ClearDepth inputs are copied left RGBs only from the pilot set. They were paired with literal all-zero depth maps strictly to check API behavior and save predictions. This does **not** supply LingBot-Depth's intended aligned raw sensor/stereo depth input, and no ClearDepth GT depth, segmentation mask, intrinsics, or stereo-derived raw depth was used in the run. Consequently, no ClearDepth metric is reported and this is not a comparable ClearDepth baseline.

A valid quantitative protocol needs an input depth source that does not leak ClearDepth GT labels: e.g. calibrated stereo matching from left/right RGB plus intrinsics/baseline, or a provided raw depth product. It should retain GT depth only for evaluation and calculate the ClearDepth disparity metrics after a documented depth-to-disparity conversion/calibration. LingBot's native depth metrics (RMSE, MAE, AbsRel, delta accuracy) are also appropriate when metric depth and valid masks are available.

## Artifacts

- Interactive/static index: `runs/lingbot_v05_smoke/gallery_index.html`
- All arrays, PNGs, masks, summaries, and PLY files: `runs/lingbot_v05_smoke/`
- Reproducible wrapper: `scripts/run_lingbot_v05_zero_depth_smoke.py`
- Exact inference-path analysis: `runs/lingbot_v05_smoke/CODE_PATH_ANALYSIS.md`
- Runtime metadata and full checkpoint config: `runs/lingbot_v05_smoke/runtime.json`, `runs/lingbot_v05_smoke/model_config.json`

## Exact rerun command

```bash
cd /mnt/20t/xuxin/depth_io_inspection
source scripts/env.sh
export TRANSFORMERS_CACHE="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME/datasets" XFORMERS_DISABLED=1 CUDA_VISIBLE_DEVICES=0
/mnt/8t/xhm/miniconda3/envs/belief_training/bin/python scripts/run_lingbot_v05_zero_depth_smoke.py \
  --repo repos/lingbot-depth \
  --checkpoint models/lingbot-depth-v0.5/v0.5/model.pt \
  --out runs/lingbot_v05_smoke \
  --device cuda:0 --resolution-level 9 \
  --clear-rgb data/cleardepth_examples/transparent_dataset__bathroom001_0003_circle000__0023/Image0023_L.png \
  --clear-rgb data/cleardepth_examples/transparent_dataset__bathroom001_0007_circle001__0038/Image0038_L.png
```
