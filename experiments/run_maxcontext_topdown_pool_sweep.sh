#!/bin/bash
# Submit Study A (pool-size sweep) jobs via the MaxContext TopDown base launcher.
#
# Usage:
#   bash experiments/run_maxcontext_topdown_pool_sweep.sh <model> <dataset> <run_path> <output_dir> \
#       [device] [scoring] [passage_length]

set -e

MODEL=${1:-"Qwen/Qwen3-4B"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/maxcontext_topdown_pool_sweep/qwen3-4b-dl19"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
PASSAGE_LENGTH=${7:-512}

mkdir -p logs "${OUTPUT_DIR}"

MODEL_TAG=$(echo "${MODEL}" | tr '/.' '-' | tr '[:upper:]' '[:lower:]')
DATASET_TAG=$(echo "${DATASET}" | sed 's#[^[:alnum:]]#-#g' | tr '[:upper:]' '[:lower:]')

for POOL_SIZE in 10 20 30 40 50; do
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/pool${POOL_SIZE}"
    JOB_TAG="mctdpool-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}"
    echo "Submitting ${JOB_TAG}"
    sbatch -J "${JOB_TAG}" \
        -o "logs/${JOB_TAG}-%j.out" \
        -e "logs/${JOB_TAG}-%j.err" \
        experiments/run_maxcontext_topdown.sh \
        "${MODEL}" "${DATASET}" "${RUN_PATH}" "${RUN_OUTPUT_DIR}" \
        "${DEVICE}" "${SCORING}" "${POOL_SIZE}" "${PASSAGE_LENGTH}"
done
