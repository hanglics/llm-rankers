#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR=${1:-"results/analysis/topdown_bubblesort_dualend_bubblesort_qualitative"}
TOPDOWN_NAME=${2:-"topdown_bubblesort"}
DUALEND_NAME=${3:-"dualend_bubblesort"}
shift || true
DEFAULT_PYTHON_BIN="$(command -v python3 || command -v python)"
if [ -x "./ranker_env/bin/python" ]; then
    DEFAULT_PYTHON_BIN="./ranker_env/bin/python"
fi
PYTHON_BIN=${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}

RESULT_DIRS=(
    "results/flan-t5-large-dl19"
    "results/flan-t5-large-dl20"
    "results/flan-t5-xl-dl19"
    "results/flan-t5-xl-dl20"
    "results/flan-t5-xxl-dl19"
    "results/flan-t5-xxl-dl20"
    "results/qwen3-4b-dl19"
    "results/qwen3-4b-dl20"
    "results/qwen3-8b-dl19"
    "results/qwen3-8b-dl20"
    "results/qwen3-14b-dl19"
    "results/qwen3-14b-dl20"
    "results/qwen3.5-4b-dl19"
    "results/qwen3.5-4b-dl20"
    "results/qwen3.5-9b-dl19"
    "results/qwen3.5-9b-dl20"
    "results/qwen3.5-27b-dl19"
    "results/qwen3.5-27b-dl20"
)

"${PYTHON_BIN}" analysis/when_dualend_helps.py \
    --result_dirs "${RESULT_DIRS[@]}" \
    --topdown_name "${TOPDOWN_NAME}" \
    --dualend_name "${DUALEND_NAME}" \
    --output_dir "${OUTPUT_DIR}"
