#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/20t/xuxin/depth_io_inspection
[ "$(realpath -e "$ROOT")" = "$ROOT" ] && [ ! -L "$ROOT" ]
export HOME="$ROOT/home" TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/.cache" PIP_CACHE_DIR="$ROOT/.cache/pip" HF_HOME="$ROOT/.cache/huggingface" HUGGINGFACE_HUB_CACHE="$ROOT/.cache/huggingface/hub" MODELSCOPE_CACHE="$ROOT/.cache/modelscope" TORCH_HOME="$ROOT/.cache/torch" MPLCONFIGDIR="$ROOT/.cache/matplotlib" CUDA_CACHE_PATH="$ROOT/.cache/cuda" PYTHONPYCACHEPREFIX="$ROOT/.cache/pycache"
