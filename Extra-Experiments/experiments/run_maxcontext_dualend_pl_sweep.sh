#!/bin/bash
# Submit Study B (passage-length sweep) jobs via the MaxContext base launcher.
#
# Usage:
#   bash experiments/run_maxcontext_dualend_pl_sweep.sh <model> <dataset> <run_path> <output_dir> \
#       [device] [scoring] [pool_size]
#
# Set POOL_SIZE to the predeclared Study A choice before launch.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL=${1:-"Qwen/Qwen3-4B"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/maxcontext_dualend_pl_sweep/qwen3-4b-dl19"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
POOL_SIZE=${7:-30}

mkdir -p logs "${OUTPUT_DIR}"

MODEL_TAG=$(echo "${MODEL}" | tr '/.' '-' | tr '[:upper:]' '[:lower:]')
DATASET_TAG=$(echo "${DATASET}" | sed 's#[^[:alnum:]]#-#g' | tr '[:upper:]' '[:lower:]')

for PASSAGE_LENGTH in 64 128 256 512; do
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/pl${PASSAGE_LENGTH}"
    JOB_TAG="mcpl-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}-pl${PASSAGE_LENGTH}"
    echo "Submitting ${JOB_TAG}"
    sbatch -J "${JOB_TAG}" \
        -o "logs/${JOB_TAG}-%j.out" \
        -e "logs/${JOB_TAG}-%j.err" \
        experiments/run_maxcontext_dualend.sh \
        "${MODEL}" "${DATASET}" "${RUN_PATH}" "${RUN_OUTPUT_DIR}" \
        "${DEVICE}" "${SCORING}" "${POOL_SIZE}" "${PASSAGE_LENGTH}"
done
