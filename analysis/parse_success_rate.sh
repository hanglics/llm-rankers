#!/bin/bash
# Compute dual-end parse success rate from log files
# Usage: bash analysis/parse_success_rate.sh results/flan-t5-xl-dl19

DIR=${1:-"results/flan-t5-xl-dl19"}

echo "=== Dual-End Parse Success Rate ==="
echo "Directory: ${DIR}"
echo ""

for f in ${DIR}/*dualend*.log ${DIR}/*dual*.log; do
    [ -f "$f" ] || continue
    name=$(basename $f .log)

    # Count dual parse warnings
    dual_fail=$(grep -c "Could not reliably parse dual" $f 2>/dev/null || echo 0)

    # Count unexpected output warnings
    unexpected=$(grep -c "Unexpected output" $f 2>/dev/null || echo 0)

    # Count best==worst warnings
    same_best_worst=$(grep -c "best and worst are the same" $f 2>/dev/null || echo 0)

    # Get total comparisons
    avg_comps=$(grep "Avg comparisons" $f 2>/dev/null | awk '{print $3}')

    echo "${name}:"
    echo "  Avg comparisons/query: ${avg_comps}"
    echo "  Dual parse failures: ${dual_fail}"
    echo "  Unexpected outputs: ${unexpected}"
    echo "  Best==Worst warnings: ${same_best_worst}"

    if [ -n "${avg_comps}" ] && [ "${avg_comps}" != "0" ]; then
        # Rough estimate — total warnings vs total comparisons
        total_warnings=$((dual_fail + unexpected))
        echo "  Total warnings: ${total_warnings}"
    fi
    echo ""
done
