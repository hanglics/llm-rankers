"""
Per-Query Analysis

Computes per-query wins/losses between methods and analyzes when
fusion (BiDir) helps vs. hurts compared to individual methods.

Usage:
    python analysis/per_query_analysis.py \
        --topdown results/flan-t5-xl-dl19/topdown_heapsort.txt \
        --bottomup results/flan-t5-xl-dl19/bottomup_heapsort.txt \
        --bidir_rrf results/flan-t5-xl-dl19/bidirectional_rrf.txt \
        --qrels dl19-passage
"""

import argparse
import subprocess


def get_per_query_ndcg(run_path: str, qrels: str) -> dict:
    """Get NDCG@10 per query."""
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
    parser.add_argument("--topdown", required=True)
    parser.add_argument("--bottomup", required=True)
    parser.add_argument("--bidir_rrf", default=None, help="BiDir-RRF run file (optional)")
    parser.add_argument("--permvote", default=None, help="Permutation voting run file (optional)")
    parser.add_argument("--qrels", required=True, help="Qrels name (e.g., dl19-passage)")
    args = parser.parse_args()

    td = get_per_query_ndcg(args.topdown, args.qrels)
    bu = get_per_query_ndcg(args.bottomup, args.qrels)

    common = sorted(set(td) & set(bu))
    n = len(common)

    print(f"\n=== Per-Query Analysis ({n} queries, qrels={args.qrels}) ===\n")

    # TopDown vs BottomUp
    td_wins = [(q, td[q] - bu[q]) for q in common if td[q] > bu[q]]
    bu_wins = [(q, bu[q] - td[q]) for q in common if bu[q] > td[q]]
    ties = [q for q in common if td[q] == bu[q]]

    print("TopDown vs BottomUp:")
    print(f"  TopDown wins: {len(td_wins)} queries (avg margin: {sum(d for _, d in td_wins)/max(len(td_wins),1):.4f})")
    print(f"  BottomUp wins: {len(bu_wins)} queries (avg margin: {sum(d for _, d in bu_wins)/max(len(bu_wins),1):.4f})")
    print(f"  Ties: {len(ties)} queries")

    # Best individual per query
    best_individual = {q: max(td[q], bu[q]) for q in common}
    worst_individual = {q: min(td[q], bu[q]) for q in common}

    if args.bidir_rrf:
        bidir = get_per_query_ndcg(args.bidir_rrf, args.qrels)
        bidir_common = sorted(set(common) & set(bidir))

        helps = [(q, bidir[q] - best_individual[q]) for q in bidir_common if bidir[q] > best_individual[q]]
        hurts = [(q, best_individual[q] - bidir[q]) for q in bidir_common if bidir[q] < worst_individual[q]]
        between = [q for q in bidir_common if worst_individual[q] <= bidir[q] <= best_individual[q]]

        print(f"\nBiDir-RRF vs Best Individual:")
        print(f"  Fusion helps (beats best): {len(helps)} queries (avg gain: {sum(d for _, d in helps)/max(len(helps),1):.4f})")
        print(f"  Fusion hurts (worse than worst): {len(hurts)} queries (avg loss: {sum(d for _, d in hurts)/max(len(hurts),1):.4f})")
        print(f"  Fusion between: {len(between)} queries")

        # Fusion improvement correlates with disagreement
        print("\n  Fusion gain by disagreement:")
        disagree_qids = [q for q in bidir_common if td[q] != bu[q]]
        agree_qids = [q for q in bidir_common if td[q] == bu[q]]

        if disagree_qids:
            disagree_gain = sum(bidir[q] - best_individual[q] for q in disagree_qids) / len(disagree_qids)
            print(f"    Disagree queries ({len(disagree_qids)}): avg delta = {disagree_gain:+.4f}")
        if agree_qids:
            agree_gain = sum(bidir[q] - best_individual[q] for q in agree_qids) / len(agree_qids)
            print(f"    Agree queries ({len(agree_qids)}): avg delta = {agree_gain:+.4f}")

    if args.permvote:
        pv = get_per_query_ndcg(args.permvote, args.qrels)
        pv_common = sorted(set(common) & set(pv))

        if args.bidir_rrf:
            bidir = get_per_query_ndcg(args.bidir_rrf, args.qrels)
            bidir_pv_common = sorted(set(pv_common) & set(bidir))

            bidir_wins = sum(1 for q in bidir_pv_common if bidir[q] > pv[q])
            pv_wins = sum(1 for q in bidir_pv_common if pv[q] > bidir[q])
            ties = sum(1 for q in bidir_pv_common if pv[q] == bidir[q])

            bidir_mean = sum(bidir[q] for q in bidir_pv_common) / len(bidir_pv_common)
            pv_mean = sum(pv[q] for q in bidir_pv_common) / len(bidir_pv_common)

            print(f"\nBiDir-RRF vs PermVote(p=2) (same 2x cost):")
            print(f"  BiDir NDCG@10: {bidir_mean:.4f}")
            print(f"  PermVote NDCG@10: {pv_mean:.4f}")
            print(f"  BiDir wins: {bidir_wins}, PermVote wins: {pv_wins}, Ties: {ties}")


if __name__ == "__main__":
    main()
