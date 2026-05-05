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
        LEVEL=2
    elif echo "${DIR}" | grep -qi "dl20"; then
        QRELS="dl20-passage"
        DATASET_LABEL="DL20"
        LEVEL=2
    elif echo "${DIR}" | grep -qi "beir-dbpedia\|dbpedia-entity"; then
        QRELS="beir-v1.0.0-dbpedia-entity-test"; DATASET_LABEL="BEIR DBPedia"; LEVEL=1
    elif echo "${DIR}" | grep -qi "beir-nfcorpus\|nfcorpus"; then
        QRELS="beir-v1.0.0-nfcorpus-test"; DATASET_LABEL="BEIR NFCorpus"; LEVEL=1
    elif echo "${DIR}" | grep -qi "beir-scifact\|scifact"; then
        QRELS="beir-v1.0.0-scifact-test"; DATASET_LABEL="BEIR SciFact"; LEVEL=1
    elif echo "${DIR}" | grep -qi "beir-trec-covid\|trec-covid"; then
        QRELS="beir-v1.0.0-trec-covid-test"; DATASET_LABEL="BEIR TREC-COVID"; LEVEL=1
    elif echo "${DIR}" | grep -qi "beir-touche2020\|webis-touche2020"; then
        QRELS="beir-v1.0.0-webis-touche2020-test"; DATASET_LABEL="BEIR Touche 2020"; LEVEL=1
    elif echo "${DIR}" | grep -qi "beir-fiqa\|fiqa"; then
        QRELS="beir-v1.0.0-fiqa-test"; DATASET_LABEL="BEIR FiQA"; LEVEL=1
    else
        echo "Skipping ${DIR} (cannot determine qrels from name — include 'dl19', 'dl20', or a supported BEIR tag)"
        continue
    fi

    RESULTS_FILE="${DIR}/results.txt"
    > "${RESULTS_FILE}"

    {
    echo ""
    echo "================================================================================================================"
    echo "Results for: ${DIR} (${DATASET_LABEL})"
    echo "================================================================================================================"
    echo ""
    echo "nDCG@10, nDCG@100, MAP@10, MAP@100, and Recall@1000:"
    echo "----------------------------------------------------------------------------------------------------------------"
    printf "%-32s %-10s %-10s %-10s %-10s %-10s\n" "Method" "nDCG@10" "nDCG@100" "MAP@10" "MAP@100" "Recall@1000"
    echo "----------------------------------------------------------------------------------------------------------------"
    for f in ${DIR}/*.txt; do
        name=$(basename $f .txt)
        [[ "${name}" == "results" ]] && continue
        ndcg10=$(python -m pyserini.eval.trec_eval -c -l ${LEVEL:-2} -m ndcg_cut.10 ${QRELS} ${f} 2>/dev/null | grep "ndcg_cut_10" | awk '{print $3}')
        ndcg100=$(python -m pyserini.eval.trec_eval -c -l ${LEVEL:-2} -m ndcg_cut.100 ${QRELS} ${f} 2>/dev/null | grep "ndcg_cut_100" | awk '{print $3}')
        map10=$(python -m pyserini.eval.trec_eval -c -l ${LEVEL:-2} -m map_cut.10 ${QRELS} ${f} 2>/dev/null | grep "map_cut_10" | awk '{print $3}')
        map100=$(python -m pyserini.eval.trec_eval -c -l ${LEVEL:-2} -m map_cut.100 ${QRELS} ${f} 2>/dev/null | grep "map_cut_100" | awk '{print $3}')
        recall1000=$(python -m pyserini.eval.trec_eval -c -l ${LEVEL:-2} -m recall.1000 ${QRELS} ${f} 2>/dev/null | grep "recall_1000" | awk '{print $3}')
        printf "%-32s %-10s %-10s %-10s %-10s %-10s\n" "${name}" "${ndcg10}" "${ndcg100}" "${map10}" "${map100}" "${recall1000}"
    done
    echo "----------------------------------------------------------------------------------------------------------------"

    # Efficiency stats from logs
    if ls ${DIR}/*.log 1>/dev/null 2>&1; then
        echo ""
        echo "Efficiency Summary (Avg per query):"
        echo "----------------------------------------------------------------------------------------------------------------"
        printf "%-32s %-8s %-10s %-8s\n" "Method" "Comps" "Tokens" "Time(s)"
        echo "----------------------------------------------------------------------------------------------------------------"
        for f in ${DIR}/*.log; do
            name=$(basename $f .log)
            comps=$(grep "Avg comparisons" $f 2>/dev/null | awk '{print $3}')
            tokens=$(grep "Avg prompt tokens" $f 2>/dev/null | awk '{print $4}')
            time=$(grep "Avg time per query" $f 2>/dev/null | awk '{print $5}')
            printf "%-32s %-8s %-10s %-8s\n" "${name}" "${comps}" "${tokens}" "${time}"
        done
        echo "----------------------------------------------------------------------------------------------------------------"
    fi

    if ls ${DIR}/*.log 1>/dev/null 2>&1; then
        echo ""
        echo "Method-Specific Counters (Avg per query):"
        echo "--------------------------------------------------------------------------------------------------------------------------------"
        printf "%-32s %-10s %-10s %-14s %-14s %-14s\n" "Method" "Dual" "Single" "RobustWin" "ExtraOrd" "RegWorst"
        echo "--------------------------------------------------------------------------------------------------------------------------------"
        for f in ${DIR}/*.log; do
            name=$(basename $f .log)
            dual=$(grep "Avg dual invocations" $f 2>/dev/null | awk '{print $4}')
            single=$(grep "Avg single invocations" $f 2>/dev/null | awk '{print $4}')
            robust=$(grep "Avg order-robust windows" $f 2>/dev/null | awk '{print $4}')
            extra=$(grep "Avg extra orderings" $f 2>/dev/null | awk '{print $4}')
            regworst=$(grep "Avg regularized worst moves" $f 2>/dev/null | awk '{print $5}')
            printf "%-32s %-10s %-10s %-14s %-14s %-14s\n" "${name}" "${dual:-0}" "${single:-0}" "${robust:-0}" "${extra:-0}" "${regworst:-0}"
        done
        echo "--------------------------------------------------------------------------------------------------------------------------------"
    fi

    # Parse warnings count
    if ls ${DIR}/*.log 1>/dev/null 2>&1; then
        echo ""
        echo "Warnings / Parse Failures:"
        echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        printf "%-32s %-20s %-20s %-20s %-20s %-20s %-20s\n" "Name" "Unexpected Output" "Dual Parse Fail" "Exceed Input Length" "Same Best&Worst" "Only One Parsable" "Partial Dual Parse"
        echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        for f in ${DIR}/*.log; do
            name=$(basename $f .log)
            unexpected=$(grep -c "Unexpected output" $f 2>/dev/null || true)
            dual_parse=$(grep -c "Could not reliably parse dual" $f 2>/dev/null || true)
            length_exceed=$(grep -c "exceeds model limit" $f 2>/dev/null || true)
            best_worst_same=$(grep -c "best and worst are the same" $f 2>/dev/null || true)
            parse_one=$(grep -c "Could only parse one label" $f 2>/dev/null || true)
            partial_dual=$(grep -c "Partial dual parse" $f 2>/dev/null || true)
            printf "%-32s %-20s %-20s %-20s %-20s %-20s %-20s\n" "${name}" "${unexpected}" "${dual_parse}" "${length_exceed}" "${best_worst_same}" "${parse_one}" "${partial_dual}"
        done
        echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    fi
    } | tee "${RESULTS_FILE}"

    echo ""
    echo "Results saved to ${RESULTS_FILE}"
done
