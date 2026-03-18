#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --job-name=likelihood
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --account=a_ai_collab

module load anaconda3
conda deactivate
module load cuda/12.2
conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
cd /scratch/project/neural_ir/hang/llm-rankers

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

MODEL=${1:-"google/flan-t5-xl"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"/scratch/project/neural_ir/hang/llm-rankers/runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"/scratch/project/neural_ir/hang/llm-rankers/results/flan-t5-xl-dl19-likelihood"}
DEVICE=${5:-"cuda"}
NUM_CHILD=${6:-3}
K=${7:-10}
HITS=${8:-100}
PASSAGE_LENGTH=${9:-128}

mkdir -p ${OUTPUT_DIR}

echo "=== Likelihood Scoring Experiments ==="
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"

# TopDown-Heap with likelihood
echo ""
echo ">>> [1/2] TopDown-Heap (likelihood)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/topdown_heapsort.txt \
        --device ${DEVICE} \
        --scoring likelihood \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method heapsort \
            --k ${K} \
            --direction topdown \
    2>&1 | tee ${OUTPUT_DIR}/topdown_heapsort.log

# DualEnd-Heap with likelihood (reads max+min from distribution)
echo ""
echo ">>> [2/2] DualEnd-Heap (likelihood)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/dualend_heapsort.txt \
        --device ${DEVICE} \
        --scoring likelihood \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method heapsort \
            --k ${K} \
            --direction dualend \
    2>&1 | tee ${OUTPUT_DIR}/dualend_heapsort.log

echo ""
echo "=== Likelihood experiments complete ==="
