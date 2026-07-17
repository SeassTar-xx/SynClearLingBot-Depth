# ClearDepth / LingBot-Depth isolated inspection

> Public-source release: datasets, checkpoints, run artifacts, and third-party
> source clones referenced below remain intentionally excluded. See
> [PUBLISHING_POLICY.md](PUBLISHING_POLICY.md).

This directory contains an isolated LingBot-Depth v0.5 checkpoint upgrade and real GPU smoke test, plus a small copied ClearDepth RGB-only pilot input set. The active checkpoint is recorded in `models/ACTIVE_MODEL.txt`.

Start with `reports/LingBotDepth_v0.5_upgrade_and_smoke.md` and `runs/lingbot_v05_smoke/gallery_index.html`. The ClearDepth zero-depth outputs are qualitative API smoke artifacts only, not a valid benchmark evaluation; see the report for the required non-leaking raw-depth protocol.
