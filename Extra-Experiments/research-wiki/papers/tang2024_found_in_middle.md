---
type: paper
node_id: paper:tang2024_found_in_middle
title: "Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models"
authors: ["Raphael Tang", "Xinyu Zhang", "Xueguang Ma", "Jimmy Lin", "Ferhan Ture"]
year: 2024
venue: NAACL
external_ids:
  arxiv: "2310.07712"
  doi: null
  s2: null
tags: [position-bias, listwise, self-consistency]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

Permutation self-consistency — re-running the listwise prompt over multiple shuffles of the candidate order and voting — significantly reduces positional bias without touching the model.

## Problem / Gap

LLM listwise rankers are not invariant to candidate order; the same documents in different orders yield different rankings.

## Method

- Sample K permutations of the candidate order; run listwise ranking on each.
- Aggregate via Borda/rank-vote to produce the final ranking.

## Key Results

- +7–18% NDCG for GPT-3.5; +8–16% for LLaMA-2-70B on listwise tasks.
- Mitigates the "lost in the middle" effect at the ranking level.

## Assumptions

- Bias is approximately symmetric under random permutation.
- Voting across permutations cancels per-position bias.

## Limitations / Failure Modes

- K× inference cost.
- Does not help when the comparator itself is informationally weak (the bias dominates).

## Reusable Ingredients

- Permutation voting as a bias-mitigation operator (→ idea:005, and the setwise `--num_repeat` hyperparameter).
- The analytic framework for "symmetric bias + aggregation" designs.

## Open Questions

- What happens when the bias is asymmetric (e.g. dual_worst primacy bias, claim:C5)? Does voting still work or does it entrench the bias?

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Directly motivates `bias_aware_dualend` (idea:005), which runs a small set of controlled orderings on hard windows and votes. Also motivates permutation-voting baselines (PermVote) on the Pareto frontier.
