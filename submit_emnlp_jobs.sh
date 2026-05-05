#!/usr/bin/env bash
# Submit one EMNLP main-matrix cell, optionally across all six pool sizes.

set -euo pipefail

METHOD=""
MODEL=""
DATASET=""
POOL_SIZE="50"
TAG=""
DRY_RUN=0
MAX_JOBS=100
OUTPUT_ROOT=""

usage() {
  cat <<'USAGE'
Usage: submit_emnlp_jobs.sh --method METHOD --model HF_ID --dataset DATASET --tag TAG [options]

Options:
  --method METHOD     One of topdown_bubblesort, topdown_heapsort,
                      bottomup_bubblesort, bottomup_heapsort,
                      maxcontext_topdown, maxcontext_bottomup,
                      maxcontext_dualend.
  --model HF_ID       HuggingFace model id.
  --dataset DATASET   dl19, dl20, beir-dbpedia, beir-nfcorpus, beir-scifact,
                      beir-trec-covid, beir-touche2020, or beir-fiqa.
  --pool-size N|all   Pool size (default: 50). Use "all" for 10,20,30,40,50,100.
  --tag TAG           Main-matrix tag under results/emnlp/main/.
  --output-root DIR   Override output root; default is results/emnlp/main/TAG.
  --max-jobs N        Refuse to submit if expanded job count exceeds N.
  --dry-run           Print commands instead of submitting sbatch jobs.
  -h | --help         Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) [[ $# -ge 2 ]] || { echo "Error: --method requires a value" >&2; exit 2; }; METHOD="$2"; shift 2 ;;
    --model) [[ $# -ge 2 ]] || { echo "Error: --model requires a value" >&2; exit 2; }; MODEL="$2"; shift 2 ;;
    --dataset) [[ $# -ge 2 ]] || { echo "Error: --dataset requires a value" >&2; exit 2; }; DATASET="$2"; shift 2 ;;
    --pool-size) [[ $# -ge 2 ]] || { echo "Error: --pool-size requires a value" >&2; exit 2; }; POOL_SIZE="$2"; shift 2 ;;
    --tag) [[ $# -ge 2 ]] || { echo "Error: --tag requires a value" >&2; exit 2; }; TAG="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || { echo "Error: --output-root requires a value" >&2; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
    --max-jobs) [[ $# -ge 2 ]] || { echo "Error: --max-jobs requires a value" >&2; exit 2; }; MAX_JOBS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$METHOD" ]] || { echo "Error: --method is required" >&2; usage >&2; exit 2; }
[[ -n "$MODEL" ]] || { echo "Error: --model is required" >&2; usage >&2; exit 2; }
[[ -n "$DATASET" ]] || { echo "Error: --dataset is required" >&2; usage >&2; exit 2; }
[[ -n "$TAG" || -n "$OUTPUT_ROOT" ]] || { echo "Error: --tag is required unless --output-root is set" >&2; usage >&2; exit 2; }

case "$DATASET" in
  dl19)
    DATASET_PATH="msmarco-passage/trec-dl-2019/judged"
    BM25_RUN="runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"
    DATASET_TAG="dl19"
    ;;
  dl20)
    DATASET_PATH="msmarco-passage/trec-dl-2020/judged"
    BM25_RUN="runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt"
    DATASET_TAG="dl20"
    ;;
  beir-dbpedia)
    DATASET_PATH="beir/dbpedia-entity/test"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.dbpedia-entity.txt"
    DATASET_TAG="beir-dbpedia"
    ;;
  beir-nfcorpus)
    DATASET_PATH="beir/nfcorpus/test"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.nfcorpus.txt"
    DATASET_TAG="beir-nfcorpus"
    ;;
  beir-scifact)
    DATASET_PATH="beir/scifact/test"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.scifact.txt"
    DATASET_TAG="beir-scifact"
    ;;
  beir-trec-covid)
    DATASET_PATH="beir/trec-covid"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.trec-covid.txt"
    DATASET_TAG="beir-trec-covid"
    ;;
  beir-touche2020)
    DATASET_PATH="beir/webis-touche2020/v2"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.webis-touche2020.txt"
    DATASET_TAG="beir-touche2020"
    ;;
  beir-fiqa)
    DATASET_PATH="beir/fiqa/test"
    BM25_RUN="runs/bm25/run.beir.bm25-flat.fiqa.txt"
    DATASET_TAG="beir-fiqa"
    ;;
  *) echo "Error: unsupported --dataset '$DATASET'" >&2; exit 2 ;;
esac

case "$METHOD" in
  topdown_heapsort) METHOD_ARG="heapsort"; DIRECTION="topdown"; K_MODE="fixed"; ;;
  topdown_bubblesort) METHOD_ARG="bubblesort"; DIRECTION="topdown"; K_MODE="fixed"; ;;
  bottomup_heapsort) METHOD_ARG="heapsort"; DIRECTION="bottomup"; K_MODE="fixed"; ;;
  bottomup_bubblesort) METHOD_ARG="bubblesort"; DIRECTION="bottomup"; K_MODE="fixed"; ;;
  maxcontext_topdown) METHOD_ARG="selection"; DIRECTION="maxcontext_topdown"; K_MODE="pool"; ;;
  maxcontext_bottomup) METHOD_ARG="selection"; DIRECTION="maxcontext_bottomup"; K_MODE="pool"; ;;
  maxcontext_dualend) METHOD_ARG="selection"; DIRECTION="maxcontext_dualend"; K_MODE="pool"; ;;
  *) echo "Error: unsupported --method '$METHOD'" >&2; exit 2 ;;
esac
# K_MODE=="pool" → MaxContext direction. The ranker overrides num_child to
# pool_size-1 internally; we pass POOL_SIZE at the CLI for self-documentation.
# K_MODE=="fixed" → standard setwise; num_child=2 is the actual operative value.

if [[ "$POOL_SIZE" == "all" ]]; then
  POOL_SIZES=(10 20 30 40 50 100)
elif [[ "$POOL_SIZE" =~ ^[0-9]+$ ]]; then
  POOL_SIZES=("$POOL_SIZE")
else
  echo "Error: --pool-size must be an integer or all" >&2
  exit 2
fi

if [[ "${#POOL_SIZES[@]}" -gt "$MAX_JOBS" ]]; then
  echo "Would submit ${#POOL_SIZES[@]} jobs; --max-jobs cap is ${MAX_JOBS}" >&2
  exit 2
fi

MODEL_TAG="$(printf '%s' "${MODEL##*/}" | tr './' '-' | tr '[:upper:]' '[:lower:]')"
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="results/emnlp/main/${TAG}"
fi

JOB_COUNT=0
for N in "${POOL_SIZES[@]}"; do
  printf -v POOL_TAG "pool%02d" "$N"
  OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_TAG}/${DATASET_TAG}/${METHOD}/${POOL_TAG}"
  SAVE_PATH="${OUTPUT_DIR}/${METHOD}.txt"
  LOG_PATH="${OUTPUT_DIR}/${METHOD}.log"
  COMPARISON_LOG="${OUTPUT_DIR}/${METHOD}_comparisons.jsonl"
  if [[ "$K_MODE" == "pool" ]]; then
    K_VALUE="$N"
    NUM_CHILD="$N"
  else
    K_VALUE=10
    NUM_CHILD=2
  fi

  RUN_LINE="python3 run.py run --model_name_or_path ${MODEL} --tokenizer_name_or_path ${MODEL} --run_path ${BM25_RUN} --save_path ${SAVE_PATH} --ir_dataset_name ${DATASET_PATH} --hits ${N} --query_length 128 --passage_length 512 --device cuda --scoring generation --log_comparisons ${COMPARISON_LOG} setwise --num_child ${NUM_CHILD} --method ${METHOD_ARG} --k ${K_VALUE} --num_permutation 1 --direction ${DIRECTION}"

  JOB_COUNT=$((JOB_COUNT + 1))
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "$RUN_LINE"
  else
    mkdir -p "$OUTPUT_DIR"
    sbatch \
      --job-name="emnlp-${METHOD}-${N}" \
      --partition=gpu_cuda \
      --qos=gpu \
      --gres=gpu:h100:1 \
      --cpus-per-task=4 \
      --mem=512G \
      --time=08:00:00 \
      --account=a_ai_collab \
      --output="${OUTPUT_DIR}/slurm-%j.out" \
      --wrap="cd $(pwd) && ${RUN_LINE} 2>&1 | tee ${LOG_PATH}"
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: ${JOB_COUNT} job(s) would be submitted." >&2
else
  echo "Submitted ${JOB_COUNT} job(s)." >&2
fi
