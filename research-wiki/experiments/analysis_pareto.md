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
updated_at: 2026-04-20T10:35:00+10:00
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

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
