# Start here

1. Obtain the full SynClearDepth data and place it below `data/cleardepth/` (ignored by Git).
2. Prepare a manifest from `configs/cleardepth_manifest.example.jsonl`.
3. Produce an aligned, non-GT-leaking raw depth map for every left RGB image, for example with an independent calibrated stereo baseline.
4. Run `scripts/validate_project.py` on the manifest.
5. Run `scripts/run_lingbot_on_cleardepth.py`.
6. Evaluate the generated prediction index using `scripts/evaluate_depth.py`.

Do not use GT depth as the `input_depth` field. It is reserved for evaluation.
