---
type: experiment
node_id: exp:analysis_pareto
title: "Analysis — quality-cost Pareto frontier (comparisons, tokens, wall-clock)"
status: completed
tests: ["claim:C9"]
results_dir: "results/analysis/pareto/"
source_artifact: "results/analysis/pareto/QUALITY_COST_PARETO.md"
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-27T23:48:07+10:00
---

# Summary

Frontier members:

| Method | Comparisons | Tokens | Wall-clock (s) | Mean NDCG |
|---|---:|---:|---:|---:|
| TD-Heap | 76.5 | ~32K | 28.4 | 0.6851 |
| TD-Bubble | 300 | ~126K | 110.9 | 0.6897 |
| DE-Cocktail | 546 | ~233K | 212.6 | 0.6962 |

Time axis additionally contains PermVote(p=2): ~113s, NDCG 0.6929.

Gap: TD-Bubble → DE-Cocktail = +82% comparisons for +0.0065 NDCG. Target for selective / bias-aware refinements.

## Note

The `TD-Bubble` row (300 comparisons) is from the standard `num_child=3, k=10, hits=100` config and is unaffected. A pre-fix whole-pool `TD-Bubble` run at `hits=k=num_child=10` produced `Avg comparisons: ~6.98` instead of 9 because the local outer clamp interacted with the upstream `last_start` tail-jump and the one-document skip. That exact branch is now patched in `llmrankers/setwise.py`; current verification gives 9 comparisons for `n=10,num_child=10,k=10`.

Do not use the archived pre-fix `~6.98` value as a current efficiency point. For canonical MaxContext whole-pool best-only baselines on the Pareto frontier, use [`exp:maxcontext_topdown_pool_sweep`](maxcontext_topdown_pool_sweep.md). Full analysis: [`FINDINGS.md`](../FINDINGS.md) (2026-04-27 entry); claim note: [`claim:C9`](../claims/C9_pareto_frontier.md).

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
