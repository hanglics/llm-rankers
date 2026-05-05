#!/usr/bin/env bash
# Evaluate EMNLP main-matrix outputs next to their .txt files.

set -euo pipefail

METHOD=""
MODEL=""
DATASET=""
POOL_SIZE="50"
TAG=""
OUTPUT_ROOT=""
DRY_RUN=0
FORCE=0

usage() {
  cat <<'USAGE'
Usage: eval_emnlp_jobs.sh --method METHOD --model HF_ID --dataset DATASET --tag TAG [options]

Options mirror submit_emnlp_jobs.sh:
  --method METHOD
  --model HF_ID
  --dataset DATASET
  --pool-size N|all   Pool size (default: 50). Use "all" for 10,20,30,40,50,100.
  --tag TAG           Main-matrix tag under results/emnlp/main/.
  --output-root DIR   Override output root; default is results/emnlp/main/TAG.
  --force             Re-evaluate existing .eval files.
  --dry-run           Print eval commands without invoking pyserini.
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
    --force) FORCE=1; shift ;;
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
  dl19) DATASET_TAG="dl19"; QRELS="dl19-passage"; LEVEL=2 ;;
  dl20) DATASET_TAG="dl20"; QRELS="dl20-passage"; LEVEL=2 ;;
  beir-dbpedia) DATASET_TAG="beir-dbpedia"; QRELS="beir-v1.0.0-dbpedia-entity-test"; LEVEL=1 ;;
  beir-nfcorpus) DATASET_TAG="beir-nfcorpus"; QRELS="beir-v1.0.0-nfcorpus-test"; LEVEL=1 ;;
  beir-scifact) DATASET_TAG="beir-scifact"; QRELS="beir-v1.0.0-scifact-test"; LEVEL=1 ;;
  beir-trec-covid) DATASET_TAG="beir-trec-covid"; QRELS="beir-v1.0.0-trec-covid-test"; LEVEL=1 ;;
  beir-touche2020) DATASET_TAG="beir-touche2020"; QRELS="beir-v1.0.0-webis-touche2020-test"; LEVEL=1 ;;
  beir-fiqa) DATASET_TAG="beir-fiqa"; QRELS="beir-v1.0.0-fiqa-test"; LEVEL=1 ;;
  *) echo "Error: unsupported --dataset '$DATASET'" >&2; exit 2 ;;
esac

case "$METHOD" in
  topdown_bubblesort|topdown_heapsort|bottomup_bubblesort|bottomup_heapsort|maxcontext_topdown|maxcontext_bottomup|maxcontext_dualend) ;;
  *) echo "Error: unsupported --method '$METHOD'" >&2; exit 2 ;;
esac

if [[ "$POOL_SIZE" == "all" ]]; then
  POOL_SIZES=(10 20 30 40 50 100)
elif [[ "$POOL_SIZE" =~ ^[0-9]+$ ]]; then
  POOL_SIZES=("$POOL_SIZE")
else
  echo "Error: --pool-size must be an integer or all" >&2
  exit 2
fi

MODEL_TAG="$(printf '%s' "${MODEL##*/}" | tr './' '-' | tr '[:upper:]' '[:lower:]')"
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="results/emnlp/main/${TAG}"
fi

OK=0
FAIL=0
SKIP=0
MISSING=0
PLANNED=0

for N in "${POOL_SIZES[@]}"; do
  printf -v POOL_TAG "pool%02d" "$N"
  OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_TAG}/${DATASET_TAG}/${METHOD}/${POOL_TAG}"
  RESULT="${OUTPUT_DIR}/${METHOD}.txt"
  EVAL_FILE="${OUTPUT_DIR}/${METHOD}.eval"
  EVAL_CMD="python -m pyserini.eval.trec_eval -q -l ${LEVEL} -m ndcg_cut.3,5,10,20,30,40,50 ${QRELS} ${RESULT}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "$EVAL_CMD"
    echo "  --> ${EVAL_FILE}"
    PLANNED=$((PLANNED + 1))
    continue
  fi

  if [[ ! -f "$RESULT" ]]; then
    echo "[MISSING] ${RESULT}" >&2
    MISSING=$((MISSING + 1))
    continue
  fi
  if [[ -f "$EVAL_FILE" && "$FORCE" -ne 1 ]]; then
    echo "[skip] ${EVAL_FILE} already exists (--force to overwrite)" >&2
    SKIP=$((SKIP + 1))
    continue
  fi

  if OUTPUT="$(python -m pyserini.eval.trec_eval -q -l "$LEVEL" \
                -m ndcg_cut.3,5,10,20,30,40,50 \
                "$QRELS" "$RESULT" 2>&1)"; then
    BODY="$(printf '%s\n' "$OUTPUT" | awk '/^Results:[[:space:]]*$/{found=1; next} found')"
    if [[ -z "$BODY" ]]; then
      echo "[fail] no Results marker for ${RESULT}" >&2
      printf '%s\n' "$OUTPUT" >&2
      FAIL=$((FAIL + 1))
      continue
    fi
    printf '%s\n' "$BODY" > "$EVAL_FILE"
    OK=$((OK + 1))
  else
    echo "[fail] trec_eval failed for ${RESULT}" >&2
    printf '%s\n' "$OUTPUT" >&2
    FAIL=$((FAIL + 1))
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: would evaluate ${PLANNED} file(s)." >&2
else
  echo "Eval summary: ok=${OK} failed=${FAIL} skipped=${SKIP} missing=${MISSING}" >&2
  [[ "$FAIL" -eq 0 ]] || exit 1
fi
