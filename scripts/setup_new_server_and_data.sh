#!/bin/sh
set -eu
ROOT=/home/liuke/xuxin/depth_io_inspection
. "$ROOT/scripts/env.sh"
python "$ROOT/scripts/inspect_new_server.py"
python "$ROOT/scripts/migrate_paths.py"
python "$ROOT/scripts/download_lingbot_v05.py"
python "$ROOT/scripts/index_syncleardepth_archive.py"
python "$ROOT/scripts/build_syncleardepth_sample_manifest.py"
python "$ROOT/scripts/build_syncleardepth_test_split.py" --all-samples "$ROOT/manifests/syncleardepth_all_samples.jsonl" --ratio .20 --seed 20260718 --split-name test_20pct_v1 --group-key auto --list-only
