#!/bin/bash
# Evaluate all result directories
# Usage: bash experiments/eval_all.sh [results_dir1] [results_dir2] ...
# If no args, evaluates all directories under results/

set -e

export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

DIRS="${@:-$(find results/ -mindepth 1 -maxdepth 1 -type d 2>/dev/null)}"

if [ -z "${DIRS}" ]; then
    echo "No results directories found. Pass directories as arguments or create results under results/."
    exit 0
fi

for DIR in ${DIRS}; do
    if [ ! -d "${DIR}" ]; then
        echo "Skipping ${DIR} (not a directory)"
        continue
    fi

    # Determine qrels from directory name
    if echo "${DIR}" | grep -qi "dl19"; then
        QRELS="dl19-passage"
        DATASET_LABEL="DL19"
    elif echo "${DIR}" | grep -qi "dl20"; then
        QRELS="dl20-passage"
        DATASET_LABEL="DL20"
    else
        echo "Skipping ${DIR} (cannot determine qrels from name — include 'dl19' or 'dl20')"
        continue
    fi

    RESULTS_FILE="${DIR}/results.txt"
    > "${RESULTS_FILE}"

    {
    echo ""
    echo "=============================================="
    echo "Results for: ${DIR} (${DATASET_LABEL})"
    echo "=============================================="
    echo ""
    echo "NDCG@10 and MAP@100:"
    echo "-------------------------------------------"
    printf "%-40s %-10s %-10s\n" "Method" "NDCG@10" "MAP@100"
    echo "-------------------------------------------"
    for f in ${DIR}/*.txt; do
        name=$(basename $f .txt)
        [[ "${name}" == "results" ]] && continue
        ndcg=$(python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 ${QRELS} ${f} 2>/dev/null | grep "ndcg_cut_10" | awk '{print $3}')
        map=$(python -m pyserini.eval.trec_eval -c -l 2 -m map_cut.100 ${QRELS} ${f} 2>/dev/null | grep "map_cut_100" | awk '{print $3}')
        printf "%-40s %-10s %-10s\n" "${name}" "${ndcg}" "${map}"
    done
    echo "-------------------------------------------"

    # Efficiency stats from logs
    if ls ${DIR}/*.log 1>/dev/null 2>&1; then
        echo ""
        echo "Efficiency Summary (Avg per query):"
        echo "-------------------------------------------"
        printf "%-40s %-8s %-10s %-8s\n" "Method" "Comps" "Tokens" "Time(s)"
        echo "-------------------------------------------"
        for f in ${DIR}/*.log; do
            name=$(basename $f .log)
            comps=$(grep "Avg comparisons" $f 2>/dev/null | awk '{print $3}')
            tokens=$(grep "Avg prompt tokens" $f 2>/dev/null | awk '{print $4}')
            time=$(grep "Avg time per query" $f 2>/dev/null | awk '{print $5}')
            printf "%-40s %-8s %-10s %-8s\n" "${name}" "${comps}" "${tokens}" "${time}"
        done
        echo "-------------------------------------------"
    fi

    # Parse warnings count
    if ls ${DIR}/*.log 1>/dev/null 2>&1; then
        echo ""
        echo "Warnings / Parse Failures:"
        echo "-------------------------------------------"
        for f in ${DIR}/*.log; do
            name=$(basename $f .log)
            unexpected=$(grep -c "Unexpected output" $f 2>/dev/null || true)
            dual_parse=$(grep -c "Could not reliably parse dual" $f 2>/dev/null || true)
            if [ "${unexpected}" -gt 0 ] || [ "${dual_parse}" -gt 0 ]; then
                echo "  ${name}: unexpected=${unexpected}, dual_parse_fail=${dual_parse}"
            fi
        done
        echo "-------------------------------------------"
    fi
    } | tee "${RESULTS_FILE}"

    echo ""
    echo "Results saved to ${RESULTS_FILE}"
done
