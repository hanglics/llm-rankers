"""
Position Bias Analysis (§5.4)

Analyzes whether LLMs exhibit position bias when selecting best/worst passages.
Uses comparison logs produced by running with --log_comparisons flag.

Usage:
    # Step 1: Generate comparison logs (re-run with logging)
    python run.py run --model_name_or_path google/flan-t5-xl \
        --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
        --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
        --save_path results/analysis/topdown_heapsort.txt \
        --scoring generation --hits 100 --passage_length 128 \
        --log_comparisons results/analysis/topdown_heapsort_comparisons.jsonl \
        setwise --num_child 3 --method heapsort --k 10 --direction topdown

    # Step 2: Analyze
    python analysis/position_bias.py \
        --log results/analysis/topdown_heapsort_comparisons.jsonl
"""

import argparse
import json
from collections import defaultdict, Counter


def main():
    parser = argparse.ArgumentParser(description="Position bias analysis from comparison logs")
    parser.add_argument("--log", required=True, nargs="+",
                        help="One or more comparison log files (.jsonl)")
    parser.add_argument("--output", default=None,
                        help="Save results to file (optional)")
    args = parser.parse_args()

    # Aggregate across all log files
    all_entries = []
    for log_path in args.log:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))

    if not all_entries:
        print("No comparison entries found.")
        return

    schemes = {entry.get("label_scheme", "letters_a_w") for entry in all_entries}
    if len(schemes) > 1:
        raise ValueError(
            f"Mixed label_scheme values in input logs: {schemes}. "
            "Run analysis separately per scheme."
        )
    scheme = schemes.pop()

    # Separate by comparison type
    by_type = defaultdict(list)
    for entry in all_entries:
        by_type[entry["type"]].append(entry)

    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    def render_label(index: int) -> str:
        if scheme == "letters_a_w":
            return chr(ord('A') + index)
        if scheme == "numeric_1_based":
            return str(index + 1)
        raise ValueError(f"Unsupported label_scheme: {scheme}")

    out("=" * 70)
    out("Position Bias Analysis")
    out(f"Total comparisons: {len(all_entries)}")
    out(f"Label scheme: {scheme}")
    out(f"Types: {', '.join(f'{t}={len(v)}' for t, v in sorted(by_type.items()))}")
    out("=" * 70)

    for comp_type, entries in sorted(by_type.items()):
        out(f"\n--- Type: {comp_type} ({len(entries)} comparisons) ---")

        # Count selection frequency per position
        n_positions = max(len(e["positions"]) for e in entries)
        position_counts = Counter()
        total = 0

        for entry in entries:
            selected = entry["selected"]
            if selected and selected in entry["positions"]:
                idx = entry["positions"].index(selected)
                position_counts[idx] += 1
                total += 1

        if total == 0:
            out("  No valid selections found.")
            continue

        out(f"\n  Selection frequency by position (total={total}):")
        out(f"  {'Position':<12} {'Label':<8} {'Count':>8} {'Freq':>8} {'Expected':>10}")
        out(f"  {'-'*50}")

        for i in range(n_positions):
            count = position_counts.get(i, 0)
            freq = count / total
            expected = 1.0 / n_positions
            label = render_label(i)
            bias = "▲" if freq > expected * 1.3 else ("▼" if freq < expected * 0.7 else " ")
            out(f"  {i:<12} {label:<8} {count:>8} {freq:>8.3f} {expected:>10.3f} {bias}")

        # Chi-squared test (simple)
        expected_count = total / n_positions
        chi2 = sum((position_counts.get(i, 0) - expected_count) ** 2 / expected_count
                    for i in range(n_positions))
        out(f"\n  Chi-squared: {chi2:.2f} (df={n_positions-1})")
        if expected_count < 5:
            out("  Note: expected counts are below 5, so the chi-squared approximation is unreliable.")

        # Position accuracy: was the selected doc the most relevant?
        if any("doc_relevances" in e for e in entries):
            correct = 0
            total_with_rels = 0
            for entry in entries:
                rels = entry.get("doc_relevances")
                if rels is None:
                    continue
                selected = entry["selected"]
                if selected and selected in entry["positions"]:
                    idx = entry["positions"].index(selected)
                    if comp_type in ("best", "dual_best"):
                        # Check if selected is the most relevant
                        if rels[idx] == max(rels):
                            correct += 1
                    elif comp_type in ("worst", "dual_worst"):
                        # Check if selected is the least relevant
                        if rels[idx] == min(rels):
                            correct += 1
                    total_with_rels += 1

            if total_with_rels > 0:
                out(f"\n  Accuracy (selected optimal): {correct}/{total_with_rels} = {correct/total_with_rels:.3f}")

                # Accuracy by position
                out(f"  Accuracy by position:")
                for i in range(n_positions):
                    pos_entries = [(e, e["positions"].index(e["selected"]))
                                   for e in entries
                                   if e.get("doc_relevances") and e["selected"] in e["positions"]
                                   and e["positions"].index(e["selected"]) == i]
                    if pos_entries:
                        pos_correct = 0
                        for e, idx in pos_entries:
                            rels = e["doc_relevances"]
                            if comp_type in ("best", "dual_best"):
                                if rels[idx] == max(rels):
                                    pos_correct += 1
                            elif comp_type in ("worst", "dual_worst"):
                                if rels[idx] == min(rels):
                                    pos_correct += 1
                        out(
                            f"    Position {i} ({render_label(i)}): "
                            f"{pos_correct}/{len(pos_entries)} = {pos_correct/len(pos_entries):.3f}"
                        )

    if args.output:
        with open(args.output, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
