#!/usr/bin/env bash
# Submit one EMNLP main-matrix cell, optionally across all six pool sizes.
#
# Reuses experiments/run_*.sh as the SLURM job scripts so module loads, conda
# env activation, and the cd into the project root all happen inside the job
# exactly as they do for the IDEA_007 dispatcher (submit_max_context_jobs.sh).
# Per-cell behavior is steered via three sbatch --export env vars:
#
#   CONDA_ENV          ranker_env (Qwen3 + pyserini) for Qwen3-* models;
#                      qwen35_env for Qwen3.5 / Llama-3.1 / Ministral-3.
#   ANALYSIS_LOG_DIR   directory for *_comparisons.jsonl, set to OUTPUT_DIR so
#                      different (model, dataset, pool) cells don't collide on
#                      the launcher's default basename-derived path.

set -euo pipefail

METHOD=""
MODEL=""
DATASET=""
POOL_SIZE="50"
TAG=""
DRY_RUN=0
MAX_JOBS=100
OUTPUT_ROOT=""
TIME_LIMIT="08:00:00"
SHUFFLE=0
REVERSE=0

usage() {
  cat <<'USAGE'
Usage: submit_emnlp_jobs.sh --method METHOD --model HF_ID --dataset DATASET --tag TAG [options]

Options:
  --method METHOD     One of topdown_bubblesort, topdown_heapsort,
                      bottomup_bubblesort, bottomup_heapsort,
                      maxcontext_topdown, maxcontext_bottomup,
                      maxcontext_dualend.
  --model HF_ID       HuggingFace model id. Family-mapped to a conda env:
                        Qwen/Qwen3-*                 -> ranker_env
                        Qwen/Qwen3.5-*               -> qwen35_env
                        meta-llama/Meta-Llama-3.1-*  -> qwen35_env
                        mistralai/Ministral-3-*      -> qwen35_env
                        anything else                -> ranker_env (fallback)
  --dataset DATASET   dl19, dl20, beir-dbpedia, beir-nfcorpus, beir-scifact,
                      beir-trec-covid, beir-touche2020, or beir-fiqa.
  --pool-size N|all   Pool size (default: 50). Use "all" for 10,20,30,40,50,100.
  --tag TAG           Main-matrix tag under results/emnlp/main/.
  --output-root DIR   Override output root; default is results/emnlp/main/TAG.
  --max-jobs N        Refuse to submit if expanded job count exceeds N
                      (default: 100).
  --time-limit HMS    SLURM --time directive override (default: 08:00:00).
  --shuffle           MaxContext-only: per-round shuffle of the remaining pool.
  --reverse           MaxContext-only: per-round reverse of the remaining pool.
  --dry-run           Print sbatch commands instead of submitting.
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
    --time-limit) [[ $# -ge 2 ]] || { echo "Error: --time-limit requires a value" >&2; exit 2; }; TIME_LIMIT="$2"; shift 2 ;;
    --shuffle) SHUFFLE=1; shift ;;
    --reverse) REVERSE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$METHOD" ]] || { echo "Error: --method is required" >&2; usage >&2; exit 2; }
[[ -n "$MODEL" ]] || { echo "Error: --model is required" >&2; usage >&2; exit 2; }
[[ -n "$DATASET" ]] || { echo "Error: --dataset is required" >&2; usage >&2; exit 2; }
[[ -n "$TAG" || -n "$OUTPUT_ROOT" ]] || { echo "Error: --tag is required unless --output-root is set" >&2; usage >&2; exit 2; }

if [[ "$SHUFFLE" -eq 1 && "$REVERSE" -eq 1 ]]; then
  echo "Error: --shuffle and --reverse are mutually exclusive" >&2
  exit 2
fi

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

# Map method -> launcher script + k policy.
# K_MODE=="pool" : MaxContext direction. Launcher takes
#                  (MODEL DS RUN_PATH OUTPUT_DIR DEVICE SCORING POOL_SIZE PASSAGE_LENGTH);
#                  ranker overrides num_child to pool_size-1 internally.
# K_MODE=="fixed": standard setwise. Bigram launcher takes
#                  (MODEL DS RUN_PATH OUTPUT_DIR DEVICE SCORING NUM_CHILD K HITS PASSAGE_LENGTH METHOD);
#                  EMNLP standard methods use num_child=2 (Setwise WS=3), k=10, hits=pool_size.
case "$METHOD" in
  topdown_heapsort)    LAUNCHER="experiments/run_topdown_bigram.sh";    K_MODE="fixed"; SETWISE_METHOD="heapsort"   ;;
  topdown_bubblesort)  LAUNCHER="experiments/run_topdown_bigram.sh";    K_MODE="fixed"; SETWISE_METHOD="bubblesort" ;;
  bottomup_heapsort)   LAUNCHER="experiments/run_bottomup_bigram.sh";   K_MODE="fixed"; SETWISE_METHOD="heapsort"   ;;
  bottomup_bubblesort) LAUNCHER="experiments/run_bottomup_bigram.sh";   K_MODE="fixed"; SETWISE_METHOD="bubblesort" ;;
  maxcontext_topdown)  LAUNCHER="experiments/run_maxcontext_topdown.sh";  K_MODE="pool"  ;;
  maxcontext_bottomup) LAUNCHER="experiments/run_maxcontext_bottomup.sh"; K_MODE="pool"  ;;
  maxcontext_dualend)  LAUNCHER="experiments/run_maxcontext_dualend.sh";  K_MODE="pool"  ;;
  *) echo "Error: unsupported --method '$METHOD'" >&2; exit 2 ;;
esac

if [[ "$K_MODE" != "pool" && ( "$SHUFFLE" -eq 1 || "$REVERSE" -eq 1 ) ]]; then
  echo "Error: --shuffle/--reverse are supported only for MaxContext methods" >&2
  exit 2
fi

# Resolve conda env from model family.
MODEL_BASE="${MODEL##*/}"
case "$MODEL_BASE" in
  Qwen3.5-*|Meta-Llama-3.1-*|Ministral-3-*)
    CONDA_ENV_PATH="/scratch/project/neural_ir/hang/llm-rankers/qwen35_env" ;;
  *)
    CONDA_ENV_PATH="/scratch/project/neural_ir/hang/llm-rankers/ranker_env" ;;
esac

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

CONDITION_SUFFIX=""
CONDITION_EXPORT=""
if [[ "$SHUFFLE" -eq 1 ]]; then
  CONDITION_SUFFIX="_shuffle"
  CONDITION_EXPORT=",SHUFFLE=1"
elif [[ "$REVERSE" -eq 1 ]]; then
  CONDITION_SUFFIX="_reverse"
  CONDITION_EXPORT=",REVERSE=1"
fi

JOB_COUNT=0
for N in "${POOL_SIZES[@]}"; do
  printf -v POOL_TAG "pool%02d" "$N"
  POOL_TAG="${POOL_TAG}${CONDITION_SUFFIX}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_TAG}/${DATASET_TAG}/${METHOD}/${POOL_TAG}"

  if [[ "$K_MODE" == "pool" ]]; then
    LAUNCHER_ARGS=("$MODEL" "$DATASET_PATH" "$BM25_RUN" "$OUTPUT_DIR" cuda generation "$N" 512)
  else
    # EMNLP standard methods: NUM_CHILD=2 (Setwise WS=3), K=10, HITS=N=pool_size, PL=512, METHOD=heap/bubble.
    LAUNCHER_ARGS=("$MODEL" "$DATASET_PATH" "$BM25_RUN" "$OUTPUT_DIR" cuda generation 2 10 "$N" 512 "$SETWISE_METHOD")
  fi

  EXPORT_VARS="ALL,CONDA_ENV=${CONDA_ENV_PATH},ANALYSIS_LOG_DIR=${OUTPUT_DIR}${CONDITION_EXPORT}"

  JOB_COUNT=$((JOB_COUNT + 1))
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'sbatch'
    printf ' --job-name=emnlp-%s-%s' "$METHOD" "$N"
    printf ' --time=%s' "$TIME_LIMIT"
    printf ' --output=%s/slurm-%%j.out' "$OUTPUT_DIR"
    printf ' --export=%s' "$EXPORT_VARS"
    printf ' %s' "$LAUNCHER" "${LAUNCHER_ARGS[@]}"
    printf '\n'
  else
    mkdir -p "$OUTPUT_DIR"
    sbatch \
      --job-name="emnlp-${METHOD}-${N}" \
      --time="$TIME_LIMIT" \
      --output="${OUTPUT_DIR}/slurm-%j.out" \
      --export="$EXPORT_VARS" \
      "$LAUNCHER" "${LAUNCHER_ARGS[@]}"
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: ${JOB_COUNT} job(s) would be submitted." >&2
else
  echo "Submitted ${JOB_COUNT} job(s)." >&2
fi
