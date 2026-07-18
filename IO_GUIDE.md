# I/O guide

| Component | Required input | Output | Use of ClearDepth GT |
|---|---|---|---|
| Stereo baseline | Rectified left/right RGB and calibration | Left-view disparity or metric depth | Never input |
| LingBot-Depth v0.5 | Left RGB + aligned raw/incomplete metric depth; optional normalized intrinsics | Refined metric depth; optional points/mask | Never input |
| Evaluator | LingBot prediction + GT depth; optional transparent-object mask | Depth metrics | Evaluation only |

For calibrated stereo, `Z = fB / d`, where `d` is disparity in pixels, `f` is focal length in pixels, and `B` is the baseline in metres. Ensure that the resulting metric depth is aligned with the left RGB before passing it to LingBot.
