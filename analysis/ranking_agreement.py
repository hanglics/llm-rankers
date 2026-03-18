"""
Ranking Agreement Analysis

Computes top-10 overlap and Kendall's tau between TopDown and BottomUp rankings.

Usage:
    python analysis/ranking_agreement.py \
        --topdown results/flan-t5-xl-dl19/topdown_heapsort.txt \
        --bottomup results/flan-t5-xl-dl19/bottomup_heapsort.txt
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
    # Build rank maps
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topdown", required=True, help="TopDown run file")
    parser.add_argument("--bottomup", required=True, help="BottomUp run file")
    parser.add_argument("--k", type=int, default=10, help="k for top-k overlap")
    args = parser.parse_args()

    td_rankings = load_ranking(args.topdown)
    bu_rankings = load_ranking(args.bottomup)

    common_qids = sorted(set(td_rankings) & set(bu_rankings))
    if not common_qids:
        print("No common queries found.")
        return

    overlaps = []
    taus = []

    print(f"\nRanking Agreement Analysis ({len(common_qids)} queries)")
    print(f"TopDown: {args.topdown}")
    print(f"BottomUp: {args.bottomup}")
    print("=" * 60)
    print(f"{'QID':<15} {'Top-{} Overlap'.format(args.k):>15} {'Kendall tau':>15}")
    print("-" * 60)

    for qid in common_qids:
        overlap = top_k_overlap(td_rankings[qid], bu_rankings[qid], args.k)
        tau = kendall_tau(td_rankings[qid], bu_rankings[qid])
        overlaps.append(overlap)
        taus.append(tau)
        print(f"{qid:<15} {overlap:>15} {tau:>15.4f}")

    print("-" * 60)
    mean_overlap = sum(overlaps) / len(overlaps)
    mean_tau = sum(taus) / len(taus)
    print(f"{'Mean':<15} {mean_overlap:>15.1f} {mean_tau:>15.4f}")
    print(f"\nAvg top-{args.k} overlap: {mean_overlap:.1f} / {args.k} ({100*mean_overlap/args.k:.1f}%)")
    print(f"Avg Kendall's tau: {mean_tau:.4f}")

    # Interpretation
    if mean_tau > 0.7:
        agreement = "high"
    elif mean_tau > 0.4:
        agreement = "moderate"
    else:
        agreement = "low"
    print(f"Agreement level: {agreement}")


if __name__ == "__main__":
    main()
