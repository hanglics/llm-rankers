"""
Query Difficulty Stratification Analysis (Table 6)

Stratifies queries by BM25 NDCG@10 difficulty and computes
delta NDCG@10 per method vs TopDown baseline, per tercile.

Usage:
    python analysis/query_difficulty.py \
        --topdown results/flan-t5-xl-dl19/topdown_heapsort.txt \
        --bottomup results/flan-t5-xl-dl19/bottomup_heapsort.txt \
        --dualend results/flan-t5-xl-dl19/dualend_bubblesort.txt \
        --bm25_run runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
        --qrels dl19-passage
"""

import argparse
import subprocess


def get_per_query_ndcg(run_path: str, qrels: str) -> dict:
    """Get NDCG@10 per query using pyserini's trec_eval."""
    result = subprocess.run(
        ["python", "-m", "pyserini.eval.trec_eval",
         "-c", "-l", "2", "-m", "ndcg_cut.10", "-q", qrels, run_path],
        capture_output=True, text=True
    )
    per_query = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "ndcg_cut_10" and parts[1] != "all":
            per_query[parts[1]] = float(parts[2])
    return per_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topdown", required=True, help="TopDown run file")
    parser.add_argument("--bottomup", required=True, help="BottomUp run file")
    parser.add_argument("--dualend", default=None, help="DualEnd run file (optional)")
    parser.add_argument("--bm25_run", required=True, help="BM25 run file")
    parser.add_argument("--qrels", required=True, help="Qrels name (e.g., dl19-passage)")
    args = parser.parse_args()

    # Get per-query NDCG@10 for all methods
    bm25_scores = get_per_query_ndcg(args.bm25_run, args.qrels)
    td_scores = get_per_query_ndcg(args.topdown, args.qrels)
    bu_scores = get_per_query_ndcg(args.bottomup, args.qrels)

    methods = {"TD": td_scores, "BU": bu_scores}

    de_scores = None
    if args.dualend:
        de_scores = get_per_query_ndcg(args.dualend, args.qrels)
        methods["DE"] = de_scores

    # Common queries across all methods
    common_qids = set(bm25_scores) & set(td_scores) & set(bu_scores)
    if de_scores:
        common_qids &= set(de_scores)
    common_qids = sorted(common_qids)

    if not common_qids:
        print("No common queries found. Check run files and qrels.")
        return

    # Sort by BM25 difficulty
    sorted_by_bm25 = sorted(common_qids, key=lambda q: bm25_scores[q])
    n = len(sorted_by_bm25)
    tercile_size = n // 3

    terciles = {
        "Hard": sorted_by_bm25[:tercile_size],
        "Medium": sorted_by_bm25[tercile_size:2 * tercile_size],
        "Easy": sorted_by_bm25[2 * tercile_size:],
    }

    # Build header
    method_names = ["TD", "BU"]
    if de_scores:
        method_names.append("DE")
    delta_names = ["BU-TD"]
    if de_scores:
        delta_names.append("DE-TD")

    header_parts = [f"{'Tercile':<10}", f"{'N':>4}", f"{'BM25 range':>20}"]
    for m in method_names:
        header_parts.append(f"{m + ' NDCG':>12}")
    for d in delta_names:
        header_parts.append(f"{d:>10}")
    header = "  ".join(header_parts)

    print(f"\nQuery Difficulty Analysis ({len(common_qids)} queries)")
    print(f"Qrels: {args.qrels}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label in ["Easy", "Medium", "Hard"]:
        qids = terciles[label]
        bm25_vals = [bm25_scores[q] for q in qids]
        bm25_range = f"{min(bm25_vals):.3f}-{max(bm25_vals):.3f}"

        td_mean = sum(td_scores[q] for q in qids) / len(qids)
        bu_mean = sum(bu_scores[q] for q in qids) / len(qids)
        delta_bu_td = bu_mean - td_mean

        parts = [f"{label:<10}", f"{len(qids):>4}", f"{bm25_range:>20}"]
        parts.append(f"{td_mean:>12.4f}")
        parts.append(f"{bu_mean:>12.4f}")

        if de_scores:
            de_mean = sum(de_scores[q] for q in qids) / len(qids)
            delta_de_td = de_mean - td_mean
            parts.append(f"{de_mean:>12.4f}")

        sign_bu = "+" if delta_bu_td >= 0 else ""
        parts.append(f"{sign_bu}{delta_bu_td:>9.4f}")

        if de_scores:
            sign_de = "+" if delta_de_td >= 0 else ""
            parts.append(f"{sign_de}{delta_de_td:>9.4f}")

        print("  ".join(parts))

    print("-" * len(header))

    # Overall
    td_mean = sum(td_scores[q] for q in common_qids) / len(common_qids)
    bu_mean = sum(bu_scores[q] for q in common_qids) / len(common_qids)
    delta_bu_td = bu_mean - td_mean

    parts = [f"{'Overall':<10}", f"{len(common_qids):>4}", f"{'':>20}"]
    parts.append(f"{td_mean:>12.4f}")
    parts.append(f"{bu_mean:>12.4f}")

    if de_scores:
        de_mean = sum(de_scores[q] for q in common_qids) / len(common_qids)
        delta_de_td = de_mean - td_mean
        parts.append(f"{de_mean:>12.4f}")

    sign_bu = "+" if delta_bu_td >= 0 else ""
    parts.append(f"{sign_bu}{delta_bu_td:>9.4f}")
    if de_scores:
        sign_de = "+" if delta_de_td >= 0 else ""
        parts.append(f"{sign_de}{delta_de_td:>9.4f}")

    print("  ".join(parts))

    # Per-query wins
    print(f"\nPer-query wins (TopDown vs BottomUp):")
    td_wins = sum(1 for q in common_qids if td_scores[q] > bu_scores[q])
    bu_wins = sum(1 for q in common_qids if bu_scores[q] > td_scores[q])
    ties = sum(1 for q in common_qids if td_scores[q] == bu_scores[q])
    print(f"  TD wins {td_wins}, BU wins {bu_wins}, Ties {ties}")

    if de_scores:
        print(f"\nPer-query wins (TopDown vs DualEnd):")
        td_wins = sum(1 for q in common_qids if td_scores[q] > de_scores[q])
        de_wins = sum(1 for q in common_qids if de_scores[q] > td_scores[q])
        ties = sum(1 for q in common_qids if td_scores[q] == de_scores[q])
        print(f"  TD wins {td_wins}, DE wins {de_wins}, Ties {ties}")

        print(f"\nPer-query wins (BottomUp vs DualEnd):")
        bu_wins = sum(1 for q in common_qids if bu_scores[q] > de_scores[q])
        de_wins = sum(1 for q in common_qids if de_scores[q] > bu_scores[q])
        ties = sum(1 for q in common_qids if bu_scores[q] == de_scores[q])
        print(f"  BU wins {bu_wins}, DE wins {de_wins}, Ties {ties}")


if __name__ == "__main__":
    main()
