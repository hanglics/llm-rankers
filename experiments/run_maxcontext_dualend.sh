#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --job-name=mcde
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=7-00:00:00
#SBATCH --account=a_ai_collab
#SBATCH --exclude=bun116

module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
module load cuda/12.2.0
conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
cd /scratch/project/neural_ir/hang/llm-rankers

# MaxContext DualEnd launcher.
# This launcher forces:
#   --direction maxcontext_dualend
#   --hits == --k == POOL_SIZE
#   --query_length 128
#   --num_permutation 1
#   --method selection
# num_child is ignored internally by the ranker and is passed only as a placeholder.
#
# Usage:
#   bash experiments/run_maxcontext_dualend.sh <model> <dataset> <run_path> <output_dir> \
#       [device] [scoring] [pool_size] [passage_length]

MODEL=${1:-"Qwen/Qwen3-4B"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/maxcontext_dualend/qwen3-4b-dl19"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
POOL_SIZE=${7:-10}
PASSAGE_LENGTH=${8:-512}

mkdir -p "${OUTPUT_DIR}"

ANALYSIS_DIR="results/analysis/position_bias_maxcontext/$(basename "${OUTPUT_DIR}")"
mkdir -p "${ANALYSIS_DIR}"

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

python run.py \
    run --model_name_or_path "${MODEL}" \
        --ir_dataset_name "${DATASET}" \
        --run_path "${RUN_PATH}" \
        --save_path "${OUTPUT_DIR}/maxcontext_dualend.txt" \
        --device "${DEVICE}" \
        --scoring "${SCORING}" \
        --hits "${POOL_SIZE}" \
        --query_length 128 \
        --passage_length "${PASSAGE_LENGTH}" \
        --log_comparisons "${ANALYSIS_DIR}/maxcontext_dualend_comparisons.jsonl" \
    setwise --num_child 3 \
            --method selection \
            --k "${POOL_SIZE}" \
            --num_permutation 1 \
            --direction maxcontext_dualend \
    2>&1 | tee "${OUTPUT_DIR}/maxcontext_dualend.log"
