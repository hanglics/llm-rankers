#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=p4-posthoc
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --account=a_ai_collab
#SBATCH --exclude=bun116

# Phase 4B/C/D/E: Post-hoc analyses — NO GPU needed
# Runs on existing Phase 1-3 results, CPU only.
#
# Usage:
#   sbatch experiments/slurm_phase4bce_posthoc.sh
#   # or run directly:
#   bash experiments/slurm_phase4bce_posthoc.sh

module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
cd /scratch/project/neural_ir/hang/llm-rankers

export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

echo "=============================================="
echo "Phase 4B/C/D/E: Post-hoc Analyses"
echo "=============================================="

# ============================================================
# 4B: Query Difficulty Stratification
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
# 4C: Ranking Agreement
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
# 4D: Per-Query Wins Analysis
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
# 4E: Dual-End Parse Success Rate
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
echo "Phase 4B/C/D/E Complete!"
echo "=============================================="
echo "  Query difficulty:    ${DIFF_DIR}/"
echo "  Ranking agreement:   ${AGREE_DIR}/"
echo "  Per-query wins:      ${WINS_DIR}/"
echo "  Parse success:       ${PARSE_DIR}/"
