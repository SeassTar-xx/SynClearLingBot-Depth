# I/O guide

| System | Input | Direct output | Local status |
|---|---|---|---|
| ClearDepth | rectified left/right RGB + calibration | left disparity | only original RGB and GT reference depth are local; no official prediction |
| LingBot-Depth | RGB + raw/incomplete depth + normalized intrinsics | refined metric depth and organized `[H,W,3]` points | no model output yet |

Raw depth is an incomplete sensor/stereo measurement, unlike complete GT. Disparity is stereo shift; metric depth needs calibration. LingBot consumes RGB-D, so no right image; ClearDepth consumes both images. `gt_*` files are reference data, never predictions.
