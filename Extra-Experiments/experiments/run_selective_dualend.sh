#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --job-name=selective-dualend
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=20:00:00
#SBATCH --account=a_ai_collab
#SBATCH --exclude=bun116

# Selective DualEnd: TopDown sorting with DualEnd prompts only on uncertain / shortlist windows.

set -eo pipefail

module load anaconda3/2023.09-0
# module load java/21.0.8
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
METHOD=${11:-"heapsort"}
GATE_STRATEGY=${12:-"hybrid"}
SHORTLIST_SIZE=${13:-20}
MARGIN_THRESHOLD=${14:-0.15}

mkdir -p "${OUTPUT_DIR}"
ANALYSIS_DIR="results/analysis/selective-dualend/selective-dualend-${SCORING}-${GATE_STRATEGY}/$(basename "${OUTPUT_DIR}")"
mkdir -p "${ANALYSIS_DIR}"

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

STEM="selective_dualend_${METHOD}"

"${PYTHON}" run.py \
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
            --method "${METHOD}" \
            --k "${K}" \
            --direction selective_dualend \
            --gate_strategy "${GATE_STRATEGY}" \
            --shortlist_size "${SHORTLIST_SIZE}" \
            --uncertainty_percentile "${MARGIN_THRESHOLD}" \
    2>&1 | tee "${OUTPUT_DIR}/${STEM}.log"
