# ClearDepth / LingBot-Depth evaluation

This repository provides a clean, manifest-driven framework for evaluating **LingBot-Depth v0.5** on ClearDepth without leaking GT depth into the model input.

The formal protocol is:

```text
rectified ClearDepth left/right RGB + calibration
  -> independent stereo baseline
  -> aligned raw/incomplete metric depth
  -> LingBot-Depth (left RGB + input depth)
  -> predicted metric depth
  -> GT-only evaluation
```

Start with [Formal evaluation guide](reports/FORMAL_EVALUATION.md). The repository excludes all datasets, model weights, third-party clones, run outputs, and logs; see [PUBLISHING_POLICY.md](PUBLISHING_POLICY.md).

## Main entry points

- `configs/cleardepth_manifest.example.jsonl` — one-sample-per-line manifest schema.
- `scripts/run_lingbot_on_cleardepth.py` — manifest-driven LingBot inference.
- `scripts/evaluate_depth.py` — MAE, RMSE, AbsRel, delta1, coverage, and optional foreground/background metrics.
- `scripts/validate_project.py` — validates a formal manifest before GPU use.
- `scripts/env.sh` — redirects caches and temporary files into this project root.
