#!/usr/bin/env bash

set -uo pipefail

usage() {
  cat <<'EOF'
Usage: ./eval_results.sh <path>

Recursively finds .txt files under <path>, skips results.txt, infers the dataset
from the path (for example, dl19 from flan-t5-large-dl19), and writes
trec_eval output to a sibling .eval file.

Environment overrides:
  TREC_EVAL_BIN  Path to the trec_eval binary
  QRELS_DIR      Path to the directory containing qrels.<dataset>.txt files
EOF
}

trec_eval_bin="${TREC_EVAL_BIN:-/scratch/project/neural_ir/hang/echo-embeddings/trec_eval/trec_eval}"
qrels_dir="${QRELS_DIR:-/scratch/project/neural_ir/hang/echo-embeddings/qrels}"

infer_dataset() {
  local path="$1"

  if [[ "$path" =~ (dl[0-9]{2}) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi

  return 1
}

looks_like_trec_run() {
  local file="$1"
  local first_nonempty

  first_nonempty="$(awk 'NF { print; exit }' "$file")"

  [[ -n "$first_nonempty" ]] || return 1
  [[ "$first_nonempty" =~ ^[^[:space:]]+[[:space:]]+Q0[[:space:]]+[^[:space:]]+[[:space:]]+[0-9]+[[:space:]]+[-+]?[0-9]*([.][0-9]+)?([eE][-+]?[0-9]+)?[[:space:]]+[^[:space:]]+$ ]]
}

evaluate_file() {
  local result_file="$1"
  local dataset
  local qrels_file
  local eval_file

  if ! dataset="$(infer_dataset "$result_file")"; then
    printf 'Skipping %s: could not infer dataset from path\n' "$result_file" >&2
    return 2
  fi

  if ! looks_like_trec_run "$result_file"; then
    printf 'Skipping %s: does not look like a TREC run file\n' "$result_file" >&2
    return 2
  fi

  qrels_file="$qrels_dir/qrels.${dataset}.txt"
  if [[ ! -f "$qrels_file" ]]; then
    printf 'Skipping %s: missing qrels file %s\n' "$result_file" "$qrels_file" >&2
    return 2
  fi

  eval_file="${result_file%.txt}.eval"

  printf 'Evaluating %s -> %s\n' "$result_file" "$eval_file"
  if "$trec_eval_bin" \
    -q \
    -l 2 \
    -m ndcg_cut.10,100 \
    -m map_cut.10,100 \
    -m recall.1000 \
    "$qrels_file" \
    "$result_file" > "$eval_file"; then
    return 0
  fi

  rm -f "$eval_file"
  printf 'Failed %s\n' "$result_file" >&2
  return 1
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

target_path="$1"

if [[ ! -e "$target_path" ]]; then
  printf 'Error: path not found: %s\n' "$target_path" >&2
  exit 1
fi

if [[ ! -x "$trec_eval_bin" ]]; then
  printf 'Error: trec_eval not found or not executable: %s\n' "$trec_eval_bin" >&2
  exit 1
fi

if [[ ! -d "$qrels_dir" ]]; then
  printf 'Error: qrels directory not found: %s\n' "$qrels_dir" >&2
  exit 1
fi

evaluated=0
skipped=0
failed=0

if [[ -f "$target_path" ]]; then
  if [[ "$target_path" != *.txt || "$(basename "$target_path")" == "results.txt" ]]; then
    printf 'Error: when passing a file, it must be a .txt file other than results.txt\n' >&2
    exit 1
  fi

  if evaluate_file "$target_path"; then
    ((evaluated+=1))
  else
    case $? in
      1) ((failed+=1)) ;;
      2) ((skipped+=1)) ;;
    esac
  fi
else
  while IFS= read -r -d '' result_file; do
    if evaluate_file "$result_file"; then
      ((evaluated+=1))
    else
      case $? in
        1) ((failed+=1)) ;;
        2) ((skipped+=1)) ;;
      esac
    fi
  done < <(find "$target_path" -type f -name '*.txt' ! -name 'results.txt' -print0)
fi

printf '\nEvaluated: %d\nSkipped: %d\nFailed: %d\n' "$evaluated" "$skipped" "$failed"

if ((failed > 0)); then
  exit 1
fi
