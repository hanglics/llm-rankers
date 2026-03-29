"""
Per-Query Analysis

Computes per-query wins/losses between methods and analyzes when
fusion (BiDir) or DualEnd helps vs. hurts compared to individual methods.

Usage:
    python analysis/per_query_analysis.py \
        --topdown results/flan-t5-xl-dl19/topdown_heapsort.txt \
        --bottomup results/flan-t5-xl-dl19/bottomup_heapsort.txt \
        --dualend results/flan-t5-xl-dl19/dualend_bubblesort.txt \
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


def print_pairwise(name1, scores1, name2, scores2, common):
    """Print win/loss between two methods."""
    wins1 = sum(1 for q in common if scores1[q] > scores2[q])
    wins2 = sum(1 for q in common if scores2[q] > scores1[q])
    ties = sum(1 for q in common if scores1[q] == scores2[q])

    mean1 = sum(scores1[q] for q in common) / len(common)
    mean2 = sum(scores2[q] for q in common) / len(common)

    margins1 = [scores1[q] - scores2[q] for q in common if scores1[q] > scores2[q]]
    margins2 = [scores2[q] - scores1[q] for q in common if scores2[q] > scores1[q]]
    avg_margin1 = sum(margins1) / len(margins1) if margins1 else 0
    avg_margin2 = sum(margins2) / len(margins2) if margins2 else 0

    print(f"\n{name1} vs {name2}:")
    print(f"  {name1:<15} NDCG@10={mean1:.4f}  wins {wins1} queries (avg margin: {avg_margin1:.4f})")
    print(f"  {name2:<15} NDCG@10={mean2:.4f}  wins {wins2} queries (avg margin: {avg_margin2:.4f})")
    print(f"  Ties: {ties} queries")


def print_fusion_analysis(fusion_name, fusion_scores, td, bu, common):
    """Analyze whether fusion beats best individual method."""
    best_individual = {q: max(td[q], bu[q]) for q in common}
    worst_individual = {q: min(td[q], bu[q]) for q in common}

    fusion_common = sorted(set(common) & set(fusion_scores))
    if not fusion_common:
        return

    helps = [(q, fusion_scores[q] - best_individual[q])
             for q in fusion_common if fusion_scores[q] > best_individual[q]]
    hurts = [(q, best_individual[q] - fusion_scores[q])
             for q in fusion_common if fusion_scores[q] < worst_individual[q]]
    between = [q for q in fusion_common
               if worst_individual[q] <= fusion_scores[q] <= best_individual[q]]

    print(f"\n{fusion_name} vs Best Individual (TD or BU):")
    print(f"  Beats best: {len(helps)} queries (avg gain: {sum(d for _, d in helps)/max(len(helps),1):.4f})")
    print(f"  Worse than worst: {len(hurts)} queries (avg loss: {sum(d for _, d in hurts)/max(len(hurts),1):.4f})")
    print(f"  Between: {len(between)} queries")

    # Fusion improvement vs disagreement
    print(f"\n  {fusion_name} gain by TD-BU disagreement:")
    disagree_qids = [q for q in fusion_common if td[q] != bu[q]]
    agree_qids = [q for q in fusion_common if td[q] == bu[q]]

    if disagree_qids:
        gain = sum(fusion_scores[q] - best_individual[q] for q in disagree_qids) / len(disagree_qids)
        print(f"    Disagree queries ({len(disagree_qids)}): avg delta = {gain:+.4f}")
    if agree_qids:
        gain = sum(fusion_scores[q] - best_individual[q] for q in agree_qids) / len(agree_qids)
        print(f"    Agree queries ({len(agree_qids)}): avg delta = {gain:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topdown", required=True)
    parser.add_argument("--bottomup", required=True)
    parser.add_argument("--dualend", default=None, help="DualEnd run file (optional)")
    parser.add_argument("--bidir_rrf", default=None, help="BiDir-RRF run file (optional)")
    parser.add_argument("--permvote", default=None, help="Permutation voting run file (optional)")
    parser.add_argument("--qrels", required=True, help="Qrels name (e.g., dl19-passage)")
    args = parser.parse_args()

    td = get_per_query_ndcg(args.topdown, args.qrels)
    bu = get_per_query_ndcg(args.bottomup, args.qrels)

    common = sorted(set(td) & set(bu))
    n = len(common)

    print(f"\n=== Per-Query Analysis ({n} queries, qrels={args.qrels}) ===")

    # --- Core pairwise comparisons ---
    print_pairwise("TopDown", td, "BottomUp", bu, common)

    de = None
    if args.dualend:
        de = get_per_query_ndcg(args.dualend, args.qrels)
        de_common = sorted(set(common) & set(de))
        if de_common:
            print_pairwise("TopDown", td, "DualEnd", de, de_common)
            print_pairwise("BottomUp", bu, "DualEnd", de, de_common)

    # --- Fusion analysis ---
    if args.bidir_rrf:
        bidir = get_per_query_ndcg(args.bidir_rrf, args.qrels)
        print_fusion_analysis("BiDir-RRF", bidir, td, bu, common)

    # --- DualEnd vs fusion (DualEnd uses ~same cost as TD alone, fusion uses 2x) ---
    if de and args.bidir_rrf:
        bidir = get_per_query_ndcg(args.bidir_rrf, args.qrels)
        de_bidir_common = sorted(set(common) & set(de) & set(bidir))
        if de_bidir_common:
            print_pairwise("DualEnd", de, "BiDir-RRF", bidir, de_bidir_common)
            print("  (Note: DualEnd uses ~1x cost, BiDir uses 2x cost)")

    # --- PermVote comparison ---
    if args.permvote:
        pv = get_per_query_ndcg(args.permvote, args.qrels)
        pv_common = sorted(set(common) & set(pv))

        if pv_common:
            print_pairwise("TopDown", td, "PermVote(p=2)", pv, pv_common)

        if args.bidir_rrf:
            bidir = get_per_query_ndcg(args.bidir_rrf, args.qrels)
            bidir_pv_common = sorted(set(pv_common) & set(bidir))
            if bidir_pv_common:
                print_pairwise("BiDir-RRF", bidir, "PermVote(p=2)", pv, bidir_pv_common)
                print("  (Both use 2x cost)")

        if de:
            de_pv_common = sorted(set(pv_common) & set(de))
            if de_pv_common:
                print_pairwise("DualEnd", de, "PermVote(p=2)", pv, de_pv_common)
                print("  (DualEnd uses ~1x cost, PermVote uses 2x cost)")

    # --- Summary table ---
    print(f"\n--- Summary: Mean NDCG@10 ---")
    all_methods = {"TopDown": td, "BottomUp": bu}
    if de:
        all_methods["DualEnd"] = de
    if args.bidir_rrf:
        all_methods["BiDir-RRF"] = get_per_query_ndcg(args.bidir_rrf, args.qrels)
    if args.permvote:
        all_methods["PermVote(p=2)"] = get_per_query_ndcg(args.permvote, args.qrels)

    summary_common = set(common)
    for scores in all_methods.values():
        summary_common &= set(scores)
    summary_common = sorted(summary_common)

    if summary_common:
        print(f"  (over {len(summary_common)} common queries)")
        for name, scores in all_methods.items():
            mean = sum(scores[q] for q in summary_common) / len(summary_common)
            print(f"  {name:<20} {mean:.4f}")


if __name__ == "__main__":
    main()
