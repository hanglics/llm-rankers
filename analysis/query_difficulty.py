"""
Query Difficulty Stratification Analysis (Table 6)

Stratifies queries by BM25 NDCG@10 difficulty and computes
delta NDCG@10 (BottomUp - TopDown) per tercile.

Usage:
    python analysis/query_difficulty.py \
        --topdown results/flan-t5-xl-dl19/topdown_heapsort.txt \
        --bottomup results/flan-t5-xl-dl19/bottomup_heapsort.txt \
        --bm25_run runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
        --qrels dl19-passage \
        --dataset msmarco-passage/trec-dl-2019/judged
"""

import argparse
import subprocess
import tempfile
import os
from collections import defaultdict


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
    parser.add_argument("--bm25_run", required=True, help="BM25 run file")
    parser.add_argument("--qrels", required=True, help="Qrels name (e.g., dl19-passage)")
    args = parser.parse_args()

    # Get per-query NDCG@10 for all three
    bm25_scores = get_per_query_ndcg(args.bm25_run, args.qrels)
    td_scores = get_per_query_ndcg(args.topdown, args.qrels)
    bu_scores = get_per_query_ndcg(args.bottomup, args.qrels)

    # Common queries
    common_qids = sorted(set(bm25_scores) & set(td_scores) & set(bu_scores))
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

    print(f"\nQuery Difficulty Analysis ({len(common_qids)} queries)")
    print(f"Qrels: {args.qrels}")
    print("=" * 70)
    print(f"{'Tercile':<10} {'N':>4}  {'BM25 range':>20}  {'TD NDCG@10':>12}  {'BU NDCG@10':>12}  {'Delta':>8}")
    print("-" * 70)

    for label in ["Easy", "Medium", "Hard"]:
        qids = terciles[label]
        bm25_vals = [bm25_scores[q] for q in qids]
        td_vals = [td_scores[q] for q in qids]
        bu_vals = [bu_scores[q] for q in qids]
        deltas = [bu_scores[q] - td_scores[q] for q in qids]

        bm25_range = f"{min(bm25_vals):.3f}-{max(bm25_vals):.3f}"
        td_mean = sum(td_vals) / len(td_vals)
        bu_mean = sum(bu_vals) / len(bu_vals)
        delta_mean = sum(deltas) / len(deltas)

        sign = "+" if delta_mean >= 0 else ""
        print(f"{label:<10} {len(qids):>4}  {bm25_range:>20}  {td_mean:>12.4f}  {bu_mean:>12.4f}  {sign}{delta_mean:>7.4f}")

    print("-" * 70)

    # Overall
    all_td = [td_scores[q] for q in common_qids]
    all_bu = [bu_scores[q] for q in common_qids]
    all_delta = [bu_scores[q] - td_scores[q] for q in common_qids]
    td_mean = sum(all_td) / len(all_td)
    bu_mean = sum(all_bu) / len(all_bu)
    delta_mean = sum(all_delta) / len(all_delta)
    sign = "+" if delta_mean >= 0 else ""
    print(f"{'Overall':<10} {len(common_qids):>4}  {'':>20}  {td_mean:>12.4f}  {bu_mean:>12.4f}  {sign}{delta_mean:>7.4f}")

    # Per-query wins
    td_wins = sum(1 for q in common_qids if td_scores[q] > bu_scores[q])
    bu_wins = sum(1 for q in common_qids if bu_scores[q] > td_scores[q])
    ties = sum(1 for q in common_qids if td_scores[q] == bu_scores[q])
    print(f"\nPer-query: TopDown wins {td_wins}, BottomUp wins {bu_wins}, Ties {ties}")


if __name__ == "__main__":
    main()
