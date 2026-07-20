#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
source "${SCRIPT_DIR}/env.sh"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/ensure_syncleardepth_subset.py
"${PYTHON_BIN}" scripts/ensure_lingbot_v05.py
"${PYTHON_BIN}" scripts/ensure_raw_depth.py --workers 2
"${PYTHON_BIN}" scripts/evaluate_lingbot_syncleardepth.py --resume
