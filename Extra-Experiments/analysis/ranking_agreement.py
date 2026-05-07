"""
Ranking Agreement Analysis

Computes top-k overlap and Kendall's tau between all pairs of ranking methods.
Compares TopDown vs BottomUp vs DualEnd (and optionally more).

Usage:
    python analysis/ranking_agreement.py \
        --topdown results/flan-t5-xl-dl19/topdown_heapsort.txt \
        --bottomup results/flan-t5-xl-dl19/bottomup_heapsort.txt \
        --dualend results/flan-t5-xl-dl19/dualend_bubblesort.txt
"""

import argparse
from collections import defaultdict
from itertools import combinations


def load_ranking(path: str) -> dict:
    """Load TREC run file into {qid: [(docid, rank), ...]}."""
    rankings = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank = parts[0], parts[1], parts[2], int(parts[3])
                rankings[qid].append((docid, rank))
    # Sort by rank
    for qid in rankings:
        rankings[qid] = sorted(rankings[qid], key=lambda x: x[1])
    return dict(rankings)


def top_k_overlap(ranking1, ranking2, k=10):
    """Compute top-k document overlap."""
    set1 = {doc for doc, _ in ranking1[:k]}
    set2 = {doc for doc, _ in ranking2[:k]}
    return len(set1 & set2)


def kendall_tau(ranking1, ranking2):
    """Compute Kendall's tau between two rankings over common documents."""
    rank_map1 = {doc: rank for doc, rank in ranking1}
    rank_map2 = {doc: rank for doc, rank in ranking2}

    common_docs = sorted(set(rank_map1) & set(rank_map2))
    if len(common_docs) < 2:
        return 0.0

    concordant = 0
    discordant = 0
    for i, j in combinations(range(len(common_docs)), 2):
        doc_i, doc_j = common_docs[i], common_docs[j]
        diff1 = rank_map1[doc_i] - rank_map1[doc_j]
        diff2 = rank_map2[doc_i] - rank_map2[doc_j]
        if diff1 * diff2 > 0:
            concordant += 1
        elif diff1 * diff2 < 0:
            discordant += 1

    n_pairs = concordant + discordant
    if n_pairs == 0:
        return 0.0
    return (concordant - discordant) / n_pairs


def analyze_pair(name1, name2, rankings1, rankings2, common_qids, k):
    """Analyze agreement between a pair of methods."""
    overlaps = []
    taus = []

    for qid in common_qids:
        if qid in rankings1 and qid in rankings2:
            overlap = top_k_overlap(rankings1[qid], rankings2[qid], k)
            tau = kendall_tau(rankings1[qid], rankings2[qid])
            overlaps.append(overlap)
            taus.append(tau)

    if not overlaps:
        return None

    mean_overlap = sum(overlaps) / len(overlaps)
    mean_tau = sum(taus) / len(taus)

    if mean_tau > 0.7:
        agreement = "high"
    elif mean_tau > 0.4:
        agreement = "moderate"
    else:
        agreement = "low"

    return {
        "n_queries": len(overlaps),
        "mean_overlap": mean_overlap,
        "mean_tau": mean_tau,
        "agreement": agreement,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topdown", required=True, help="TopDown run file")
    parser.add_argument("--bottomup", required=True, help="BottomUp run file")
    parser.add_argument("--dualend", default=None, help="DualEnd run file (optional)")
    parser.add_argument("--bidir_rrf", default=None, help="BiDir-RRF run file (optional)")
    parser.add_argument("--k", type=int, default=10, help="k for top-k overlap")
    args = parser.parse_args()

    # Load all available rankings
    methods = {}
    methods["TopDown"] = load_ranking(args.topdown)
    methods["BottomUp"] = load_ranking(args.bottomup)
    if args.dualend:
        methods["DualEnd"] = load_ranking(args.dualend)
    if args.bidir_rrf:
        methods["BiDir-RRF"] = load_ranking(args.bidir_rrf)

    # Common queries across all methods
    common_qids = None
    for name, rankings in methods.items():
        qids = set(rankings.keys())
        common_qids = qids if common_qids is None else common_qids & qids
    common_qids = sorted(common_qids)

    if not common_qids:
        print("No common queries found.")
        return

    method_names = list(methods.keys())

    print(f"\nRanking Agreement Analysis ({len(common_qids)} queries, k={args.k})")
    print("=" * 70)

    # Pairwise comparison summary table
    print(f"\n{'Pair':<25} {'Overlap@' + str(args.k):>12} {'Kendall tau':>12} {'Agreement':>12}")
    print("-" * 65)

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            name1, name2 = method_names[i], method_names[j]
            result = analyze_pair(
                name1, name2,
                methods[name1], methods[name2],
                common_qids, args.k
            )
            if result:
                pair_label = f"{name1} vs {name2}"
                print(f"{pair_label:<25} {result['mean_overlap']:>12.1f} {result['mean_tau']:>12.4f} {result['agreement']:>12}")

    print("-" * 65)

    # Detailed per-query breakdown for TopDown vs BottomUp (always present)
    print(f"\n--- Per-Query Detail: TopDown vs BottomUp ---")
    print(f"{'QID':<15} {'Top-{} Overlap'.format(args.k):>15} {'Kendall tau':>15}")
    print("-" * 50)

    td_rankings = methods["TopDown"]
    bu_rankings = methods["BottomUp"]
    overlaps = []
    taus = []

    for qid in common_qids:
        if qid in td_rankings and qid in bu_rankings:
            overlap = top_k_overlap(td_rankings[qid], bu_rankings[qid], args.k)
            tau = kendall_tau(td_rankings[qid], bu_rankings[qid])
            overlaps.append(overlap)
            taus.append(tau)
            print(f"{qid:<15} {overlap:>15} {tau:>15.4f}")

    if overlaps:
        print("-" * 50)
        mean_overlap = sum(overlaps) / len(overlaps)
        mean_tau = sum(taus) / len(taus)
        print(f"{'Mean':<15} {mean_overlap:>15.1f} {mean_tau:>15.4f}")
        print(f"\nAvg top-{args.k} overlap: {mean_overlap:.1f} / {args.k} ({100*mean_overlap/args.k:.1f}%)")


if __name__ == "__main__":
    main()
