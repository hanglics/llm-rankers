#!/bin/bash
# Extended Setwise Ranking Experiments
# Paper: "Beyond Best Selection: Bidirectional Strategies for LLM-Based Setwise Ranking"
#
# Usage: bash experiments/run_extended_setwise.sh <model> <dataset> <run_path> <output_dir>
# Example: bash experiments/run_extended_setwise.sh google/flan-t5-xl msmarco-passage/trec-dl-2019 runs/bm25_dl19.txt results/

# Models:
# google/flan-t5-large
# google/flan-t5-xl 3B
# google/flan-t5-xxl 11B
# Qwen/Qwen3-4B
# Qwen/Qwen3-8B
# Qwen/Qwen3-14B
# Qwen/Qwen3.5-4B  # requires a Transformers build with qwen3_5 support
#
# Notes:
# - For Flan-T5 models, a smaller passage length (e.g. 64) is usually cleaner.
# - Qwen3-family chat templates support enable_thinking=False in the ranker code.

set -e

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

MODEL=${1:-"google/flan-t5-xl"}
DATASET=${2:-"msmarco-passage/trec-dl-2019"}
RUN_PATH=${3:-"runs/run.msmarco-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/extended_setwise"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
NUM_CHILD=${7:-3}
K=${8:-10}
HITS=${9:-100}
PASSAGE_LENGTH=${10:-128}

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "Extended Setwise Ranking Experiments"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Scoring: ${SCORING}"
echo "num_child: ${NUM_CHILD}, k: ${K}, hits: ${HITS}"
echo "=============================================="

# --- Baseline: Standard Top-Down Setwise (Heapsort) ---
echo ""
echo ">>> [1/8] Standard Top-Down Setwise (Heapsort)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/topdown_heapsort.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method heapsort \
            --k ${K} \
            --direction topdown \
    2>&1 | tee ${OUTPUT_DIR}/topdown_heapsort.log

# --- Baseline: Standard Top-Down Setwise (Bubblesort) ---
echo ""
echo ">>> [2/8] Standard Top-Down Setwise (Bubblesort)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/topdown_bubblesort.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method bubblesort \
            --k ${K} \
            --direction topdown \
    2>&1 | tee ${OUTPUT_DIR}/topdown_bubblesort.log

# --- Bottom-Up Setwise (Heapsort) ---
echo ""
echo ">>> [3/8] Bottom-Up Setwise (Heapsort)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/bottomup_heapsort.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method heapsort \
            --k ${K} \
            --direction bottomup \
    2>&1 | tee ${OUTPUT_DIR}/bottomup_heapsort.log

# --- Bottom-Up Setwise (Bubblesort) ---
echo ""
echo ">>> [4/8] Bottom-Up Setwise (Bubblesort)"
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

# --- Dual-End Setwise (Bubblesort / Cocktail Shaker) ---
echo ""
echo ">>> [5/8] Dual-End Setwise (Bubblesort / Cocktail Shaker)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/dualend_bubblesort.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method bubblesort \
            --k ${K} \
            --direction dualend \
    2>&1 | tee ${OUTPUT_DIR}/dualend_bubblesort.log

# --- Dual-End Setwise (Selection Sort) ---
echo ""
echo ">>> [6/8] Dual-End Setwise (Double-Ended Selection Sort)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/dualend_selection.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method selection \
            --k ${K} \
            --direction dualend \
    2>&1 | tee ${OUTPUT_DIR}/dualend_selection.log

# --- Bidirectional Ensemble (RRF) ---
echo ""
echo ">>> [7/8] Bidirectional Ensemble (RRF)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/bidirectional_rrf.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method heapsort \
            --k ${K} \
            --direction bidirectional \
            --fusion rrf \
    2>&1 | tee ${OUTPUT_DIR}/bidirectional_rrf.log

# --- Bidirectional Ensemble (Weighted, alpha=0.7) ---
echo ""
echo ">>> [8/8] Bidirectional Ensemble (Weighted, alpha=0.7)"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/bidirectional_weighted.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method heapsort \
            --k ${K} \
            --direction bidirectional \
            --fusion weighted \
            --alpha 0.7 \
    2>&1 | tee ${OUTPUT_DIR}/bidirectional_weighted.log

echo ""
echo "=============================================="
echo "All experiments complete! Results in ${OUTPUT_DIR}/"
echo "=============================================="

# --- Evaluate all runs ---
echo ""
echo ">>> Evaluating all runs..."

if [ "${DATASET}" = "msmarco-passage/trec-dl-2019" ]; then
    QRELS="dl19-passage"
elif [ "${DATASET}" = "msmarco-passage/trec-dl-2020" ]; then
    QRELS="dl20-passage"
else
    echo "Unknown dataset for evaluation: ${DATASET}"
    exit 0
fi

echo ""
echo "Results Summary (NDCG@10):"
echo "==========================================="
for f in ${OUTPUT_DIR}/*.txt; do
    name=$(basename $f .txt)
    score=$(python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 ${QRELS} ${f} 2>/dev/null | grep "ndcg_cut_10" | awk '{print $3}')
    printf "%-35s %s\n" "${name}" "${score}"
done
echo "==========================================="

# Print efficiency stats from logs
echo ""
echo "Efficiency Summary (Avg per query):"
echo "==========================================="
for f in ${OUTPUT_DIR}/*.log; do
    name=$(basename $f .log)
    comparisons=$(grep "Avg comparisons" $f | awk '{print $3}')
    prompt_tokens=$(grep "Avg prompt tokens" $f | awk '{print $4}')
    time=$(grep "Avg time per query" $f | awk '{print $6}')
    printf "%-35s comps: %s  tokens: %s  time: %s\n" "${name}" "${comparisons}" "${prompt_tokens}" "${time}"
done
echo "==========================================="
