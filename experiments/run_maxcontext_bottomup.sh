#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --job-name=mcbu
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --account=a_ai_collab
#SBATCH --exclude=bun116,bun073

set -eo pipefail

module load anaconda3/2023.09-0
: "${EBROOTANACONDA3:?EBROOTANACONDA3 is not set after module load anaconda3/2023.09-0}"
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
# module load cuda/12.2.0  # disabled: not registered in SLURM batch env on
#                          # this cluster (was silently failing pre-`set -e`).
#                          # PyTorch in conda env brings its own bundled CUDA.
# CONDA_ENV is resolved per-model by the dispatcher (submit_max_context_jobs.sh
# / submit_emnlp_jobs.sh) and propagated via sbatch --export=ALL. Default is
# ranker_env (Qwen3 family + pyserini); qwen35_env is used for Qwen3.5,
# Llama-3.1, and Ministral-3 model families.
CONDA_ENV="${CONDA_ENV:-/scratch/project/neural_ir/hang/llm-rankers/ranker_env}"
conda activate "$CONDA_ENV"
PYTHON="${CONDA_PREFIX:-$CONDA_ENV}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Error: selected CONDA_ENV python is not executable: $PYTHON" >&2
  echo "CONDA_ENV=$CONDA_ENV" >&2
  echo "CONDA_PREFIX=${CONDA_PREFIX:-}" >&2
  exit 2
fi
echo "[launcher] CONDA_ENV=$CONDA_ENV" >&2
echo "[launcher] CONDA_PREFIX=${CONDA_PREFIX:-}" >&2
echo "[launcher] PYTHON=$PYTHON" >&2
"$PYTHON" -c 'import sys, ir_datasets; print("[launcher] sys.executable=" + sys.executable); print("[launcher] ir_datasets=" + ir_datasets.__file__)' >&2
cd /scratch/project/neural_ir/hang/llm-rankers

# MaxContext BottomUp launcher.
# This launcher forces:
#   --direction maxcontext_bottomup
#   --hits == --k == POOL_SIZE
#   --query_length 128
#   --num_permutation 1
#   --method selection
# num_child is overridden internally by the ranker to pool_size-1 (so each
# comparison batch sees the entire pool). The CLI value below is a
# self-documenting placeholder; any positive integer is functionally equivalent.
#
# Usage:
#   bash experiments/run_maxcontext_bottomup.sh <model> <dataset> <run_path> <output_dir> \
#       [device] [scoring] [pool_size] [passage_length]

MODEL=${1:-"Qwen/Qwen3-4B"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/maxcontext_bottomup/qwen3-4b-dl19"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
POOL_SIZE=${7:-10}
PASSAGE_LENGTH=${8:-512}

mkdir -p "${OUTPUT_DIR}"

# ANALYSIS_LOG_DIR override avoids basename collisions when OUTPUT_DIR is
# nested deeper than the launcher's default layout (e.g. EMNLP main matrix
# uses pool{NN}/ as the leaf basename and would collapse across models).
if [[ -n "${ANALYSIS_LOG_DIR:-}" ]]; then
  ANALYSIS_DIR="${ANALYSIS_LOG_DIR}"
else
  ANALYSIS_DIR="results/analysis/position_bias_maxcontext/$(basename "${OUTPUT_DIR}")"
fi
mkdir -p "${ANALYSIS_DIR}"

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

CONDITION_ARGS=()
if [[ -n "${SHUFFLE:-}" && "${SHUFFLE}" != "0" ]]; then
    CONDITION_ARGS+=(--shuffle)
fi
if [[ -n "${REVERSE:-}" && "${REVERSE}" != "0" ]]; then
    CONDITION_ARGS+=(--reverse)
fi

"${PYTHON}" run.py \
    run --model_name_or_path "${MODEL}" \
        --ir_dataset_name "${DATASET}" \
        --run_path "${RUN_PATH}" \
        --save_path "${OUTPUT_DIR}/maxcontext_bottomup.txt" \
        --device "${DEVICE}" \
        --scoring "${SCORING}" \
        --hits "${POOL_SIZE}" \
        --query_length 128 \
        --passage_length "${PASSAGE_LENGTH}" \
        "${CONDITION_ARGS[@]}" \
        --log_comparisons "${ANALYSIS_DIR}/maxcontext_bottomup_comparisons.jsonl" \
    setwise --num_child "${POOL_SIZE}" \
            --method selection \
            --k "${POOL_SIZE}" \
            --num_permutation 1 \
            --direction maxcontext_bottomup \
    2>&1 | tee "${OUTPUT_DIR}/maxcontext_bottomup.log"
