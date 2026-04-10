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

# Same-call worst-signal regularization: TopDown bubblesort with local worst demotion.

module load anaconda3/2023.09-0
# module load java/21.0.8
source $EBROOTANACONDA3/etc/profile.d/conda.sh
module load cuda/12.2
conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
# conda activate /scratch/project/neural_ir/hang/llm-rankers/qwen35_env
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

mkdir -p "${OUTPUT_DIR}"
ANALYSIS_DIR="results/analysis/samecall-regularized-${SCORING}/$(basename "${OUTPUT_DIR}")"
mkdir -p "${ANALYSIS_DIR}"

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

STEM="samecall_regularized_bubblesort"

python run.py \
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
