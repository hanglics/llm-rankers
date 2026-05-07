#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR=${1:-"results/analysis/pareto"}
shift || true
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python)}

if [ "$#" -gt 0 ]; then
    "${PYTHON_BIN}" analysis/quality_cost_pareto.py --output_dir "${OUTPUT_DIR}" --result_dirs "$@"
else
    "${PYTHON_BIN}" analysis/quality_cost_pareto.py --output_dir "${OUTPUT_DIR}"
fi
