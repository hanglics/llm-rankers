#!/bin/bash --login
# Same-call worst-signal regularization: TopDown bubblesort with local worst demotion.

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module load anaconda3/2023.09-0 || true
    source "${EBROOTANACONDA3:-/dev/null}/etc/profile.d/conda.sh" 2>/dev/null || true
    module load cuda/12.2 || true
fi

cd /scratch/project/neural_ir/hang/llm-rankers 2>/dev/null || cd "$(dirname "$0")/.."

MODEL=${1:-"Qwen/Qwen3-8B"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/qwen3-8b-dl19"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
NUM_CHILD=${7:-3}
K=${8:-10}
HITS=${9:-100}
PASSAGE_LENGTH=${10:-512}

if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    :
elif [ -f /scratch/project/neural_ir/hang/llm-rankers/ranker_env/bin/activate ]; then
    source /scratch/project/neural_ir/hang/llm-rankers/ranker_env/bin/activate
fi

mkdir -p "${OUTPUT_DIR}"
ANALYSIS_DIR="results/analysis/$(basename "${OUTPUT_DIR}")"
mkdir -p "${ANALYSIS_DIR}"

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

STEM="samecall_regularized_bubblesort"
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python)}

"${PYTHON_BIN}" run.py \
    run --model_name_or_path "${MODEL}" \
        --ir_dataset_name "${DATASET}" \
        --run_path "${RUN_PATH}" \
        --save_path "${OUTPUT_DIR}/${STEM}.txt" \
        --device "${DEVICE}" \
        --scoring "${SCORING}" \
        --hits "${HITS}" \
        --passage_length "${PASSAGE_LENGTH}" \
        --log_comparisons "${ANALYSIS_DIR}/${STEM}_comparisons.jsonl" \
    setwise --num_child "${NUM_CHILD}" \
            --method bubblesort \
            --k "${K}" \
            --direction samecall_regularized \
    2>&1 | tee "${OUTPUT_DIR}/${STEM}.log"
