#!/usr/bin/env bash
# submit_max_context_jobs.sh
#
# Submit all 35 sbatch jobs from Need_to_Run_Max_Context.txt with one command,
# parameterised by run tag, model, and dataset (DL19 / DL20). Everything else
# (BM25 run path, pool sizes {10,20,30,40,50}, passage length, scoring mode,
# sort algorithms, output directory layout) stays fixed.
#
# Usage:
#   ./submit_max_context_jobs.sh [--tag TAG] [--model MODEL]
#                                [--dataset DL19|DL20] [--dry-run] [-h|--help]
#
# Examples:
#   ./submit_max_context_jobs.sh                                            # all defaults
#   ./submit_max_context_jobs.sh --tag test_run_v2 --model Qwen/Qwen3-8B
#   ./submit_max_context_jobs.sh --tag prod_v1 --model Qwen/Qwen3.5-9B \
#                                --dataset DL20
#   ./submit_max_context_jobs.sh --dry-run                                  # print only

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults (match the original Need_to_Run_Max_Context.txt baseline)
# -----------------------------------------------------------------------------
TAG="test_run_v1"
MODEL="Qwen/Qwen3-4B"
DATASET="DL19"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_max_context_jobs.sh [options]

Options:
  --tag TAG          Run-tag fragment to substitute for "test_run_v1" in
                     the output directory paths (default: test_run_v1).
  --model MODEL      HuggingFace model identifier, e.g. "Qwen/Qwen3-4B"
                     (default: Qwen/Qwen3-4B). The model tag in output
                     directories is derived as the lower-cased basename
                     after the final "/" (e.g. Qwen/Qwen3.5-9B -> qwen3.5-9b).
  --dataset NAME     One of DL19 or DL20 (default: DL19). Mapped to:
                       DL19 -> msmarco-passage/trec-dl-2019/judged
                       DL20 -> msmarco-passage/trec-dl-2020/judged
                     and the corresponding BM25 run path under runs/bm25/.
  --dry-run          Print sbatch commands instead of submitting them.
  -h | --help        Show this help and exit.

Submits 35 sbatch jobs (7 method blocks x 5 pool sizes {10,20,30,40,50}):
  - Original TopDown-Heap   (WS=4)
  - Original TopDown-Bubble (WS=4)
  - Original TopDown-Heap   (WS=PS)
  - Original TopDown-Bubble (WS=PS)
  - MaxContext-TopDown
  - MaxContext-DualEnd     (writes to phase1/, all others write to baseline/)
  - MaxContext-BottomUp
USAGE
}

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      [[ $# -ge 2 ]] || { echo "Error: --tag requires a value" >&2; exit 2; }
      TAG="$2"; shift 2 ;;
    --model)
      [[ $# -ge 2 ]] || { echo "Error: --model requires a value" >&2; exit 2; }
      MODEL="$2"; shift 2 ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "Error: --dataset requires a value" >&2; exit 2; }
      DATASET="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      usage >&2
      exit 2 ;;
  esac
done

# -----------------------------------------------------------------------------
# Resolve dataset shortcut -> dataset path, BM25 run path, dataset tag
# -----------------------------------------------------------------------------
case "$DATASET" in
  DL19|dl19)
    DATASET_PATH="msmarco-passage/trec-dl-2019/judged"
    BM25_RUN="runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"
    DATASET_TAG="dl19"
    ;;
  DL20|dl20)
    DATASET_PATH="msmarco-passage/trec-dl-2020/judged"
    BM25_RUN="runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt"
    DATASET_TAG="dl20"
    ;;
  *)
    echo "Error: --dataset must be DL19 or DL20 (got: '$DATASET')" >&2
    exit 2
    ;;
esac

# -----------------------------------------------------------------------------
# Derive model tag for the output-directory path: basename of the HF id, lowercased.
# Matches the convention used in Need_to_Run_Max_Context.txt:
#   Qwen/Qwen3-4B    -> qwen3-4b
#   Qwen/Qwen3.5-9B  -> qwen3.5-9b
# -----------------------------------------------------------------------------
MODEL_BASE="${MODEL##*/}"
MODEL_TAG="$(printf '%s' "$MODEL_BASE" | tr '[:upper:]' '[:lower:]')"

OUT_PREFIX="results/maxcontext_dualend/${TAG}"
RUN_BASELINE="${OUT_PREFIX}/baseline/${MODEL_TAG}-${DATASET_TAG}"
RUN_PHASE1="${OUT_PREFIX}/phase1/${MODEL_TAG}-${DATASET_TAG}"

POOL_SIZES=(10 20 30 40 50)

# -----------------------------------------------------------------------------
# Submission helper: dry-runs print, real runs ensure the output directory
# exists then submit. Tracks total jobs submitted in $JOB_COUNT.
#
# Convention used by every block below: the 4th positional argument to the
# launcher is OUTPUT_DIR. Inside this function $1 is the launcher itself, so
# the OUTPUT_DIR is at $5. Pre-creating it here fails fast on permission /
# typo issues at submission time instead of hours later when the job runs.
# -----------------------------------------------------------------------------
JOB_COUNT=0

submit() {
  JOB_COUNT=$((JOB_COUNT + 1))
  local output_dir="${5:?submit() expects OUTPUT_DIR as the 4th launcher arg}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'sbatch'
    printf ' %s' "$@"
    printf '\n'
  else
    mkdir -p "$output_dir"
    sbatch "$@"
  fi
}

cat <<INFO >&2
==> submit_max_context_jobs.sh
    tag         : ${TAG}
    model       : ${MODEL}     (model_tag = ${MODEL_TAG})
    dataset     : ${DATASET}   (path = ${DATASET_PATH}; tag = ${DATASET_TAG})
    bm25 run    : ${BM25_RUN}
    output base : ${OUT_PREFIX}
    mode        : $([[ $DRY_RUN -eq 1 ]] && echo "DRY-RUN (no sbatch)" || echo "submitting")
INFO

# =============================================================================
# Block 1 - Original TopDown-Heap (WS=4)
# =============================================================================
for N in "${POOL_SIZES[@]}"; do
  submit experiments/run_topdown_bigram.sh \
    "$MODEL" \
    "$DATASET_PATH" \
    "$BM25_RUN" \
    "${RUN_BASELINE}/original/ws-4/top${N}" \
    cuda generation 3 "$N" "$N" 512 heapsort
done

# =============================================================================
# Block 2 - Original TopDown-Bubble (WS=4)
# =============================================================================
for N in "${POOL_SIZES[@]}"; do
  submit experiments/run_topdown_bigram.sh \
    "$MODEL" \
    "$DATASET_PATH" \
    "$BM25_RUN" \
    "${RUN_BASELINE}/original/ws-4/top${N}" \
    cuda generation 3 "$N" "$N" 512 bubblesort
done

# =============================================================================
# Block 3 - Original TopDown-Heap (WS=PS)  [num_child = pool_size - 1]
# =============================================================================
for N in "${POOL_SIZES[@]}"; do
  submit experiments/run_topdown_bigram.sh \
    "$MODEL" \
    "$DATASET_PATH" \
    "$BM25_RUN" \
    "${RUN_BASELINE}/original/ws-ps/top${N}" \
    cuda generation $((N - 1)) "$N" "$N" 512 heapsort
done

# =============================================================================
# Block 4 - Original TopDown-Bubble (WS=PS)  [num_child = pool_size - 1]
# =============================================================================
for N in "${POOL_SIZES[@]}"; do
  submit experiments/run_topdown_bigram.sh \
    "$MODEL" \
    "$DATASET_PATH" \
    "$BM25_RUN" \
    "${RUN_BASELINE}/original/ws-ps/top${N}" \
    cuda generation $((N - 1)) "$N" "$N" 512 bubblesort
done

# =============================================================================
# Block 5 - MaxContext-TopDown
# =============================================================================
for N in "${POOL_SIZES[@]}"; do
  submit experiments/run_maxcontext_topdown.sh \
    "$MODEL" \
    "$DATASET_PATH" \
    "$BM25_RUN" \
    "${RUN_BASELINE}/max-context/topdown/top${N}" \
    cuda generation "$N" 512
done

# =============================================================================
# Block 6 - MaxContext-DualEnd        (NOTE: writes under phase1/, not baseline/)
# =============================================================================
for N in "${POOL_SIZES[@]}"; do
  submit experiments/run_maxcontext_dualend.sh \
    "$MODEL" \
    "$DATASET_PATH" \
    "$BM25_RUN" \
    "${RUN_PHASE1}/top${N}" \
    cuda generation "$N" 512
done

# =============================================================================
# Block 7 - MaxContext-BottomUp
# =============================================================================
for N in "${POOL_SIZES[@]}"; do
  submit experiments/run_maxcontext_bottomup.sh \
    "$MODEL" \
    "$DATASET_PATH" \
    "$BM25_RUN" \
    "${RUN_BASELINE}/max-context/bottomup/top${N}" \
    cuda generation "$N" 512
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "" >&2
  echo "==> DRY-RUN done. ${JOB_COUNT} sbatch jobs would have been submitted." >&2
else
  echo "" >&2
  echo "==> Submitted ${JOB_COUNT} sbatch jobs." >&2
fi
