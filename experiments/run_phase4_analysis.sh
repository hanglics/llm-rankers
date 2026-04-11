#!/bin/bash
# Phase 4: Analysis Experiments
# Runs all post-hoc analyses and position bias experiments
#
# Usage: bash experiments/run_phase4_analysis.sh [model] [device] [passage_length]
# Example: bash experiments/run_phase4_analysis.sh google/flan-t5-xl cuda
# Example: bash experiments/run_phase4_analysis.sh Qwen/Qwen3-8B cuda 512

set -e

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

MODEL=${1:-"google/flan-t5-xl"}
DEVICE=${2:-"cuda"}
PASSAGE_LENGTH=${3:-""}

# Derive short model name for paths
MODEL_SHORT=$(echo ${MODEL} | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')

if [ -z "${PASSAGE_LENGTH}" ]; then
    case "${MODEL}" in
        Qwen/*)
            PASSAGE_LENGTH=512
            ;;
        *)
            PASSAGE_LENGTH=128
            ;;
    esac
fi

echo "=============================================="
echo "Phase 4: Analysis Experiments"
echo "Model: ${MODEL}"
echo "Passage Length: ${PASSAGE_LENGTH}"
echo "=============================================="

# ============================================================
# 4A: Position Bias Analysis (requires re-runs with logging)
# ============================================================
echo ""
echo ">>> 4A: Position Bias Analysis"
echo "    Re-running 3 methods on DL19 with comparison logging..."

ANALYSIS_DIR="results/analysis/${MODEL_SHORT}-dl19"
mkdir -p ${ANALYSIS_DIR}

# TopDown Heapsort
echo "    [1/3] TopDown Heapsort..."
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
        --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
        --save_path ${ANALYSIS_DIR}/topdown_heapsort.txt \
        --device ${DEVICE} --scoring generation --hits 100 --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/topdown_heapsort_comparisons.jsonl \
    setwise --num_child 3 --method heapsort --k 10 --direction topdown \
    2>&1 | tee ${ANALYSIS_DIR}/topdown_heapsort.log

# BottomUp Heapsort
echo "    [2/3] BottomUp Heapsort..."
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
        --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
        --save_path ${ANALYSIS_DIR}/bottomup_heapsort.txt \
        --device ${DEVICE} --scoring generation --hits 100 --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/bottomup_heapsort_comparisons.jsonl \
    setwise --num_child 3 --method heapsort --k 10 --direction bottomup \
    2>&1 | tee ${ANALYSIS_DIR}/bottomup_heapsort.log

# DualEnd Cocktail
echo "    [3/3] DualEnd Cocktail..."
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
        --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
        --save_path ${ANALYSIS_DIR}/dualend_bubblesort.txt \
        --device ${DEVICE} --scoring generation --hits 100 --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/dualend_bubblesort_comparisons.jsonl \
    setwise --num_child 3 --method bubblesort --k 10 --direction dualend \
    2>&1 | tee ${ANALYSIS_DIR}/dualend_bubblesort.log

# Analyze position bias
echo ""
echo "    Analyzing position bias..."
python analysis/position_bias.py \
    --log ${ANALYSIS_DIR}/*_comparisons.jsonl \
    --output ${ANALYSIS_DIR}/position_bias_results.txt

# ============================================================
# 4B: Query Difficulty Stratification (post-hoc, all models)
# ============================================================
echo ""
echo ">>> 4B: Query Difficulty Stratification"

DIFF_DIR="results/analysis/query_difficulty"
mkdir -p ${DIFF_DIR}

for RESULTS_DIR in results/*-dl19; do
    [ -d "${RESULTS_DIR}" ] || continue
    MODEL_NAME=$(basename ${RESULTS_DIR} | sed 's/-dl19$//')

    TD="${RESULTS_DIR}/topdown_heapsort.txt"
    BU="${RESULTS_DIR}/bottomup_heapsort.txt"
    DE="${RESULTS_DIR}/dualend_bubblesort.txt"

    if [ -f "${TD}" ] && [ -f "${BU}" ]; then
        CMD="python analysis/query_difficulty.py --topdown ${TD} --bottomup ${BU}"
        CMD="${CMD} --bm25_run runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt --qrels dl19-passage"
        [ -f "${DE}" ] && CMD="${CMD} --dualend ${DE}"
        echo "    ${MODEL_NAME} (DL19)"
        eval ${CMD} > "${DIFF_DIR}/${MODEL_NAME}_dl19.txt" 2>&1
    fi
done

for RESULTS_DIR in results/*-dl20; do
    [ -d "${RESULTS_DIR}" ] || continue
    MODEL_NAME=$(basename ${RESULTS_DIR} | sed 's/-dl20$//')

    TD="${RESULTS_DIR}/topdown_heapsort.txt"
    BU="${RESULTS_DIR}/bottomup_heapsort.txt"
    DE="${RESULTS_DIR}/dualend_bubblesort.txt"

    if [ -f "${TD}" ] && [ -f "${BU}" ]; then
        CMD="python analysis/query_difficulty.py --topdown ${TD} --bottomup ${BU}"
        CMD="${CMD} --bm25_run runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt --qrels dl20-passage"
        [ -f "${DE}" ] && CMD="${CMD} --dualend ${DE}"
        echo "    ${MODEL_NAME} (DL20)"
        eval ${CMD} > "${DIFF_DIR}/${MODEL_NAME}_dl20.txt" 2>&1
    fi
done

echo "    Results saved to ${DIFF_DIR}/"

# ============================================================
# 4C: Ranking Agreement (post-hoc, all models)
# ============================================================
echo ""
echo ">>> 4C: Ranking Agreement"

AGREE_DIR="results/analysis/ranking_agreement"
mkdir -p ${AGREE_DIR}

for RESULTS_DIR in results/*-dl19; do
    [ -d "${RESULTS_DIR}" ] || continue
    MODEL_NAME=$(basename ${RESULTS_DIR} | sed 's/-dl19$//')

    TD="${RESULTS_DIR}/topdown_heapsort.txt"
    BU="${RESULTS_DIR}/bottomup_heapsort.txt"
    DE="${RESULTS_DIR}/dualend_bubblesort.txt"

    if [ -f "${TD}" ] && [ -f "${BU}" ]; then
        CMD="python analysis/ranking_agreement.py --topdown ${TD} --bottomup ${BU}"
        [ -f "${DE}" ] && CMD="${CMD} --dualend ${DE}"
        echo "    ${MODEL_NAME}: pairwise agreement (DL19)"
        eval ${CMD} > "${AGREE_DIR}/${MODEL_NAME}_dl19.txt" 2>&1
    fi
done

for RESULTS_DIR in results/*-dl20; do
    [ -d "${RESULTS_DIR}" ] || continue
    MODEL_NAME=$(basename ${RESULTS_DIR} | sed 's/-dl20$//')

    TD="${RESULTS_DIR}/topdown_heapsort.txt"
    BU="${RESULTS_DIR}/bottomup_heapsort.txt"
    DE="${RESULTS_DIR}/dualend_bubblesort.txt"

    if [ -f "${TD}" ] && [ -f "${BU}" ]; then
        CMD="python analysis/ranking_agreement.py --topdown ${TD} --bottomup ${BU}"
        [ -f "${DE}" ] && CMD="${CMD} --dualend ${DE}"
        echo "    ${MODEL_NAME}: pairwise agreement (DL20)"
        eval ${CMD} > "${AGREE_DIR}/${MODEL_NAME}_dl20.txt" 2>&1
    fi
done

echo "    Results saved to ${AGREE_DIR}/"

# ============================================================
# 4D: Per-Query Wins Analysis (post-hoc, all models)
# ============================================================
echo ""
echo ">>> 4D: Per-Query Wins Analysis"

WINS_DIR="results/analysis/per_query_wins"
mkdir -p ${WINS_DIR}

for RESULTS_DIR in results/*-dl19; do
    [ -d "${RESULTS_DIR}" ] || continue
    MODEL_NAME=$(basename ${RESULTS_DIR} | sed 's/-dl19$//')

    TD="${RESULTS_DIR}/topdown_heapsort.txt"
    BU="${RESULTS_DIR}/bottomup_heapsort.txt"
    DE="${RESULTS_DIR}/dualend_bubblesort.txt"
    BIDIR="${RESULTS_DIR}/bidirectional_rrf.txt"
    PV="${RESULTS_DIR}/permvote_p2_heapsort.txt"

    if [ -f "${TD}" ] && [ -f "${BU}" ]; then
        CMD="python analysis/per_query_analysis.py --topdown ${TD} --bottomup ${BU} --qrels dl19-passage"
        [ -f "${DE}" ] && CMD="${CMD} --dualend ${DE}"
        [ -f "${BIDIR}" ] && CMD="${CMD} --bidir_rrf ${BIDIR}"
        [ -f "${PV}" ] && CMD="${CMD} --permvote ${PV}"
        echo "    ${MODEL_NAME} (DL19)"
        eval ${CMD} > "${WINS_DIR}/${MODEL_NAME}_dl19.txt" 2>&1
    fi
done

for RESULTS_DIR in results/*-dl20; do
    [ -d "${RESULTS_DIR}" ] || continue
    MODEL_NAME=$(basename ${RESULTS_DIR} | sed 's/-dl20$//')

    TD="${RESULTS_DIR}/topdown_heapsort.txt"
    BU="${RESULTS_DIR}/bottomup_heapsort.txt"
    DE="${RESULTS_DIR}/dualend_bubblesort.txt"
    BIDIR="${RESULTS_DIR}/bidirectional_rrf.txt"
    PV="${RESULTS_DIR}/permvote_p2_heapsort.txt"

    if [ -f "${TD}" ] && [ -f "${BU}" ]; then
        CMD="python analysis/per_query_analysis.py --topdown ${TD} --bottomup ${BU} --qrels dl20-passage"
        [ -f "${DE}" ] && CMD="${CMD} --dualend ${DE}"
        [ -f "${BIDIR}" ] && CMD="${CMD} --bidir_rrf ${BIDIR}"
        [ -f "${PV}" ] && CMD="${CMD} --permvote ${PV}"
        echo "    ${MODEL_NAME} (DL20)"
        eval ${CMD} > "${WINS_DIR}/${MODEL_NAME}_dl20.txt" 2>&1
    fi
done

echo "    Results saved to ${WINS_DIR}/"

# ============================================================
# 4E: Dual-End Parse Success Rate (from logs)
# ============================================================
echo ""
echo ">>> 4E: Dual-End Parse Success Rate"

PARSE_DIR="results/analysis/parse_success"
mkdir -p ${PARSE_DIR}

for RESULTS_DIR in results/*-dl19 results/*-dl20; do
    [ -d "${RESULTS_DIR}" ] || continue
    MODEL_NAME=$(basename ${RESULTS_DIR})

    if ls ${RESULTS_DIR}/*dualend*.log 1>/dev/null 2>&1 || ls ${RESULTS_DIR}/*dual*.log 1>/dev/null 2>&1; then
        echo "    ${MODEL_NAME}"
        bash analysis/parse_success_rate.sh "${RESULTS_DIR}" \
            > "${PARSE_DIR}/${MODEL_NAME}.txt" 2>&1
    fi
done

echo "    Results saved to ${PARSE_DIR}/"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=============================================="
echo "Phase 4 Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  Position bias:       ${ANALYSIS_DIR}/position_bias_results.txt"
echo "  Query difficulty:    ${DIFF_DIR}/"
echo "  Ranking agreement:   ${AGREE_DIR}/"
echo "  Per-query wins:      ${WINS_DIR}/"
echo "  Parse success:       ${PARSE_DIR}/"
