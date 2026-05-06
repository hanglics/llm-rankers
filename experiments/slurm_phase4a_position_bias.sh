#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --job-name=p4a-posbias
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=20:00:00
#SBATCH --account=a_ai_collab
#SBATCH --exclude=bun116

# Phase 4A: Position Bias Analysis — requires GPU re-runs with comparison logging
#
# Usage:
#   sbatch experiments/slurm_phase4a_position_bias.sh [model] [passage_length]
#
# Examples:
#   sbatch experiments/slurm_phase4a_position_bias.sh google/flan-t5-large 128
#   sbatch experiments/slurm_phase4a_position_bias.sh google/flan-t5-xl 128
#   sbatch experiments/slurm_phase4a_position_bias.sh google/flan-t5-xxl 128
#   sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3-4B 512
#   sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3-8B 512
#   sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3-14B 512
#   sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3.5-4B 512

set -eo pipefail

module load anaconda3/2023.09-0
: "${EBROOTANACONDA3:?EBROOTANACONDA3 is not set after module load anaconda3/2023.09-0}"
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
module load cuda/12.2.0
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

MODEL=${1:-"google/flan-t5-xl"}
PASSAGE_LENGTH=${2:-128}

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

# Derive short model name for paths
MODEL_SHORT=$(echo ${MODEL} | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')

ANALYSIS_DIR="results/analysis/position_bias/${MODEL_SHORT}-dl19"
mkdir -p ${ANALYSIS_DIR}

DATASET="msmarco-passage/trec-dl-2019/judged"
RUN_PATH="runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"

echo "=============================================="
echo "Phase 4A: Position Bias Analysis"
echo "Model: ${MODEL}"
echo "Passage Length: ${PASSAGE_LENGTH}"
echo "Output: ${ANALYSIS_DIR}"
echo "=============================================="

# [1/3] TopDown Heapsort
echo ""
echo ">>> [1/3] TopDown Heapsort (with comparison logging)"
"${PYTHON}" run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${ANALYSIS_DIR}/topdown_heapsort.txt \
        --device cuda --scoring generation --hits 100 --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/topdown_heapsort_comparisons.jsonl \
    setwise --num_child 3 --method heapsort --k 10 --direction topdown \
    2>&1 | tee ${ANALYSIS_DIR}/topdown_heapsort.log

# [2/3] BottomUp Heapsort
echo ""
echo ">>> [2/3] BottomUp Heapsort (with comparison logging)"
"${PYTHON}" run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${ANALYSIS_DIR}/bottomup_heapsort.txt \
        --device cuda --scoring generation --hits 100 --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/bottomup_heapsort_comparisons.jsonl \
    setwise --num_child 3 --method heapsort --k 10 --direction bottomup \
    2>&1 | tee ${ANALYSIS_DIR}/bottomup_heapsort.log

# [3/3] DualEnd Cocktail
echo ""
echo ">>> [3/3] DualEnd Cocktail (with comparison logging)"
"${PYTHON}" run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${ANALYSIS_DIR}/dualend_bubblesort.txt \
        --device cuda --scoring generation --hits 100 --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/dualend_bubblesort_comparisons.jsonl \
    setwise --num_child 3 --method bubblesort --k 10 --direction dualend \
    2>&1 | tee ${ANALYSIS_DIR}/dualend_bubblesort.log

# Analyze position bias (runs on CPU, quick)
echo ""
echo ">>> Analyzing position bias..."
"${PYTHON}" analysis/position_bias.py \
    --log ${ANALYSIS_DIR}/*_comparisons.jsonl \
    --output ${ANALYSIS_DIR}/position_bias_results.txt

echo ""
echo "=============================================="
echo "Phase 4A Complete for ${MODEL}"
echo "Results: ${ANALYSIS_DIR}/position_bias_results.txt"
echo "=============================================="
