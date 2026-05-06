#!/usr/bin/env bash
# Phase A EMNLP smoke gate: 3 models x 7 methods x dl19 x pools {50,100}.

set -euo pipefail

# Activate pyserini environment for the inline python3 calls in verify_cell
# and to keep the smoke script's environment aligned with eval_emnlp_jobs.sh /
# eval_max_context_jobs.sh. Tolerate non-HPC environments where `module` is
# unavailable — the user is expected to have ranker_env active manually then.
# Per-cell SLURM jobs spawned by submit_emnlp_jobs.sh resolve their own
# CONDA_ENV per model family (qwen35_env for Qwen3.5/Llama-3.1/Ministral-3),
# so this preamble is only the head-node activation.
if command -v module >/dev/null 2>&1; then
  module load anaconda3/2023.09-0
  source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
  module load cuda/12.2.0
  conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
fi

DRY_RUN=0
EVAL_ONLY=0
VERIFY_ONLY=0
TAG="phase_a"
OUTPUT_ROOT="results/emnlp/smoke/phase_a"

usage() {
  cat <<'USAGE'
Usage: scripts/smoke_emnlp_models.sh [--dry-run] [--eval-only] [--verify-only]

Submits the 42 Phase A smoke jobs unless --eval-only or --verify-only is passed.
Outputs are written under results/emnlp/smoke/phase_a/{model_tag}/dl19/{method}/pool{50,100}/.

Use --eval-only after the jobs complete to write .eval files next to the .txt
outputs, then --verify-only to apply the method-aware smoke checks.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

MODELS=(
  "Qwen/Qwen3.5-9B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "mistralai/Ministral-3-8B-Instruct-2512"
)

METHODS=(
  "topdown_bubblesort"
  "topdown_heapsort"
  "bottomup_bubblesort"
  "bottomup_heapsort"
  "maxcontext_topdown"
  "maxcontext_bottomup"
  "maxcontext_dualend"
)

SMOKE_POOLS=(50 100)
# dl19 has 43 judged queries; this script is dl19-only per Phase A.
N_QUERIES=43

model_tag() {
  printf '%s' "${1##*/}" | tr './' '-' | tr '[:upper:]' '[:lower:]'
}

verify_cell() {
  local model="$1"
  local method="$2"
  local pool_size="$3"
  local tag
  tag="$(model_tag "$model")"
  local dir="${OUTPUT_ROOT}/${tag}/dl19/${method}/pool${pool_size}"
  local txt="${dir}/${method}.txt"
  local eval_file="${dir}/${method}.eval"
  local log="${dir}/${method}.log"
  local expected_lines=$((N_QUERIES * pool_size))

  [[ -f "$txt" ]] || { echo "[FAIL] missing ${txt}" >&2; return 1; }
  [[ -f "$eval_file" ]] || { echo "[FAIL] missing ${eval_file}" >&2; return 1; }
  [[ -f "$log" ]] || { echo "[FAIL] missing ${log}" >&2; return 1; }

  local lines
  lines="$(wc -l < "$txt" | tr -d ' ')"
  [[ "$lines" == "$expected_lines" ]] || { echo "[FAIL] ${txt}: expected ${expected_lines} lines, got ${lines}" >&2; return 1; }

  python3 - "$txt" "$eval_file" "$N_QUERIES" <<'PY'
import sys
from collections import defaultdict
run_path, eval_path, expected_qids_raw = sys.argv[1:]
expected_qids = int(expected_qids_raw)
top_docs = defaultdict(list)
with open(run_path) as handle:
    for line in handle:
        parts = line.split()
        if len(parts) < 4:
            raise SystemExit(f"bad run line: {line!r}")
        qid, docid, rank = parts[0], parts[2], int(parts[3])
        if rank <= 10:
            top_docs[qid].append(docid)
if len(top_docs) != expected_qids:
    raise SystemExit(f"expected {expected_qids} qids, got {len(top_docs)}")
for qid, docs in top_docs.items():
    if len(docs) != 10 or len(set(docs)) != 10:
        raise SystemExit(f"qid {qid} top10 is not 10 distinct docids")
ndcg = None
with open(eval_path) as handle:
    for line in handle:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "ndcg_cut_10" and parts[1] == "all":
            ndcg = float(parts[2])
            break
if ndcg is None:
    raise SystemExit("missing ndcg_cut_10 all")
if ndcg <= 0:
    raise SystemExit(f"ndcg_cut_10 all is not positive: {ndcg}")
PY

  if grep -Eq "Traceback|ERROR|exceeds model limit" "$log"; then
    echo "[FAIL] ${log}: contains Traceback/ERROR/exceeds model limit" >&2
    return 1
  fi

  case "$method" in
    maxcontext_*)
      grep -q "Avg parse fallbacks: 0" "$log" || { echo "[FAIL] ${log}: missing zero parse fallback counter" >&2; return 1; }
      grep -q "Avg numeric out-of-range fallbacks: 0" "$log" || { echo "[FAIL] ${log}: missing zero numeric out-of-range counter" >&2; return 1; }
      ;;
  esac

  echo "[OK] ${tag} ${method} pool${pool_size}"
}

print_summary() {
  echo "[smoke] processed ${#MODELS[@]} × ${#METHODS[@]} × ${#SMOKE_POOLS[@]} = $((${#MODELS[@]} * ${#METHODS[@]} * ${#SMOKE_POOLS[@]})) cells"
}

if [[ "$VERIFY_ONLY" -eq 0 && "$EVAL_ONLY" -eq 0 ]]; then
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      for ps in "${SMOKE_POOLS[@]}"; do
        args=(
          bash submit_emnlp_jobs.sh
          --method "$method"
          --model "$model"
          --dataset dl19
          --pool-size "$ps"
          --tag "$TAG"
          --output-root "$OUTPUT_ROOT"
        )
        if [[ "$DRY_RUN" -eq 1 ]]; then
          args+=(--dry-run)
        fi
        "${args[@]}"
      done
    done
  done
  print_summary
fi

if [[ "$EVAL_ONLY" -eq 1 ]]; then
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      for ps in "${SMOKE_POOLS[@]}"; do
        args=(
          bash eval_emnlp_jobs.sh
          --method "$method"
          --model "$model"
          --dataset dl19
          --pool-size "$ps"
          --tag "$TAG"
          --output-root "$OUTPUT_ROOT"
        )
        if [[ "$DRY_RUN" -eq 1 ]]; then
          args+=(--dry-run)
        fi
        "${args[@]}"
      done
    done
  done
  print_summary
fi

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  failures=0
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      for ps in "${SMOKE_POOLS[@]}"; do
        verify_cell "$model" "$method" "$ps" || failures=$((failures + 1))
      done
    done
  done
  [[ "$failures" -eq 0 ]] || exit 1
  print_summary
fi
