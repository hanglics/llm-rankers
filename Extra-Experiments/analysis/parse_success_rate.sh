#!/bin/bash
# Compute dual-end parse success rate from log files
# Usage: bash analysis/parse_success_rate.sh results/flan-t5-xl-dl19

shopt -s nullglob

DIR=${1:-"results/flan-t5-xl-dl19"}

echo "=== Dual-End Parse Success Rate ==="
echo "Directory: ${DIR}"
echo ""

for f in "$DIR"/*dual*.log; do
    [ -f "$f" ] || continue
    name=$(basename -- "$f" .log)

    dual_fail=$(grep -c "Could not reliably parse dual" "$f" 2>/dev/null)
    unexpected=$(grep -c "Unexpected output" "$f" 2>/dev/null)
    same_best_worst=$(grep -c "best and worst are the same" "$f" 2>/dev/null)
    length_exceed=$(grep -c "exceeds model limit" "$f" 2>/dev/null)
    parse_one=$(grep -c "Could only parse one label" "$f" 2>/dev/null)
    partial_dual=$(grep -c "Partial dual parse" "$f" 2>/dev/null)

    avg_comps=$(awk '/Avg comparisons/ {print $3; exit}' "$f")

    echo "${name}:"
    echo "  Avg comparisons/query: ${avg_comps}"
    echo "  Dual parse failures: ${dual_fail}"
    echo "  Unexpected outputs: ${unexpected}"
    echo "  Best==Worst warnings: ${same_best_worst}"
    echo "  Length exceeds: ${length_exceed}"
    echo "  Only parse one: ${parse_one}"
    echo "  Partial dual parse: ${partial_dual}"

    if [ -n "$avg_comps" ] && [ "$avg_comps" != "0" ]; then
        total_warnings=$((dual_fail + unexpected))
        echo "  Total warnings: ${total_warnings}"
    fi
    echo
done
