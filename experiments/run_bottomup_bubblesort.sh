#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=512G
#SBATCH --job-name=bubb
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=10:00:00
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/project/neural_ir/hang/llm-rankers/logs/qwen3-4b-3-10-100-512-dl20-bottomup_bubblesort.output
#SBATCH -e /scratch/project/neural_ir/hang/llm-rankers/logs/qwen3-4b-3-10-100-512-dl20-bottomup_bubblesort.error

module load anaconda3
conda deactivate
module load cuda/12.2
conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
cd /scratch/project/neural_ir/hang/llm-rankers

MODEL=${1:-"Qwen/Qwen3-4B"}
DATASET=${2:-"msmarco-passage/trec-dl-2020/judged"}
RUN_PATH=${3:-"/scratch/project/neural_ir/hang/llm-rankers/runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt"}
OUTPUT_DIR=${4:-"/scratch/project/neural_ir/hang/llm-rankers/results/extended_setwise/qwen3-4b-3-10-100-512-dl20"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
NUM_CHILD=${7:-3}
K=${8:-10}
HITS=${9:-100}
PASSAGE_LENGTH=${10:-512}
BIDIRECTION_WEIGHTED_ALPHA=${11:-0.7}

mkdir -p ${OUTPUT_DIR}

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/bottomup_bubblesort.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method bubblesort \
            --k ${K} \
            --direction bottomup \
    2>&1 | tee ${OUTPUT_DIR}/bottomup_bubblesort.log