#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --job-name=likelihood
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=a_ai_collab

module load anaconda3/2023.09-0
# module load java/21.0.8
source $EBROOTANACONDA3/etc/profile.d/conda.sh
module load cuda/12.2.0
conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
# conda activate /scratch/project/neural_ir/hang/llm-rankers/qwen35_env
cd /scratch/project/neural_ir/hang/llm-rankers

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

MODEL=${1:-"google/flan-t5-xl"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/flan-t5-xl-dl19-likelihood"}
DEVICE=${5:-"cuda"}
NUM_CHILD=${6:-3}
K=${7:-10}
HITS=${8:-100}
PASSAGE_LENGTH=${9:-128}

mkdir -p ${OUTPUT_DIR}

ANALYSIS_DIR="results/analysis/likelihood/$(basename ${OUTPUT_DIR})"

mkdir -p ${ANALYSIS_DIR}

echo "=== Likelihood Scoring Experiments ==="
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"

# TopDown-Heap with likelihood
echo ""
echo ">>> [1/4] TopDown-Heap (likelihood)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/topdown_heapsort.txt \
        --device ${DEVICE} \
        --scoring likelihood \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/topdown_heapsort_comparisons.jsonl \
    setwise --num_child ${NUM_CHILD} \
            --method heapsort \
            --k ${K} \
            --direction topdown \
    2>&1 | tee ${OUTPUT_DIR}/topdown_heapsort.log

# BottomUp-Heap with likelihood
echo ""
echo ">>> [2/4] BottomUp-Heap (likelihood)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/bottomup_heapsort.txt \
        --device ${DEVICE} \
        --scoring likelihood \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/bottomup_heapsort_comparisons.jsonl \
    setwise --num_child ${NUM_CHILD} \
            --method heapsort \
            --k ${K} \
            --direction bottomup \
    2>&1 | tee ${OUTPUT_DIR}/bottomup_heapsort.log

# DualEnd-Bubblesort with likelihood (reads max+min from the best-label distribution)
echo ""
echo ">>> [3/4] DualEnd-Bubblesort (likelihood)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/dualend_bubblesort.txt \
        --device ${DEVICE} \
        --scoring likelihood \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/dualend_bubblesort_comparisons.jsonl \
    setwise --num_child ${NUM_CHILD} \
            --method bubblesort \
            --k ${K} \
            --direction dualend \
    2>&1 | tee ${OUTPUT_DIR}/dualend_bubblesort.log

# DualEnd-Selection with likelihood (same heuristic best/min reuse, different sorter)
echo ""
echo ">>> [4/4] DualEnd-Selection (likelihood)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/dualend_selection.txt \
        --device ${DEVICE} \
        --scoring likelihood \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/dualend_selection_comparisons.jsonl \
    setwise --num_child ${NUM_CHILD} \
            --method selection \
            --k ${K} \
            --direction dualend \
    2>&1 | tee ${OUTPUT_DIR}/dualend_selection.log

echo ""
echo "=== Likelihood experiments complete ==="
