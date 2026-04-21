---
type: claim
node_id: claim:C9
statement: "The quality-cost Pareto frontier contains only TD-Heap, TD-Bubble, and DE-Cocktail (plus PermVote on time-only); the region between TD-Bubble and DE-Cocktail is empty and is the natural target for selective / bias-aware refinements."
status: supported
evidence_strength: high
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Claim

Frontier members (NDCG@10 vs cost, averaged across models × datasets):

| Method | Comparisons | Total tokens | Wall-clock (s) | Mean NDCG |
|---|---:|---:|---:|---:|
| TD-Heap | 76.5 | ~32K | 28.4 | 0.6851 |
| TD-Bubble | 300 | ~126K | 110.9 | 0.6897 |
| DE-Cocktail | 546 | ~233K | 212.6 | 0.6962 |

- On the wall-clock axis, PermVote(p=2) ≈ 113s / 0.6929 adds a frontier point between TD-Heap and TD-Bubble.
- **Frontier gap:** TD-Bubble → DE-Cocktail costs +82% comparisons and +92% wall-clock for +0.0065 NDCG. No method currently bridges this gap efficiently.

## Implication

Refinement methods (idea:004 selective_dualend, idea:005 bias_aware_dualend, idea:006 samecall_regularized) should target this gap. A method landing near (~400 comparisons, ~0.693) would materially advance the frontier.

## Supporting experiments

- exp:analysis_pareto
- (reference artifact: `results/analysis/pareto/QUALITY_COST_PARETO.md`)

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
