---
type: claim
node_id: claim:C9
statement: "The quality-cost Pareto frontier contains only TD-Heap, TD-Bubble, and DE-Cocktail (plus PermVote on time-only); the region between TD-Bubble and DE-Cocktail is empty and is the natural target for selective / bias-aware refinements."
status: supported
evidence_strength: high
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-27T23:48:07+10:00
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

## Note — `TD-Bubble` `Avg comparisons` semantics

The `TD-Bubble` frontier point above (300 comparisons, ~110.9s) is from `exp:main_td_bubble` at `num_child=3, k=10, hits=100` — interpretable.

A pre-fix `TD-Bubble` run under the edge-case config `hits == k == num_child == N` produced a misleadingly low call count. For example, `hits=k=num_child=10` reported `Avg comparisons: 6.9767` instead of the intuitive 9 because the local outer clamp interacted with the upstream `last_start` tail-jump and the one-document skip. That exact whole-pool branch is now patched in `llmrankers/setwise.py`: the local outer clamp is disabled only when `len(ranking) == k == num_child` or `num_child >= len(ranking)`, while the one-document skip remains. Current verification gives 9 comparisons for `n=10,num_child=10,k=10`.

**Implication for paper claims:** do not use the archived pre-fix `6.9767` run as a current efficiency frontier point. When comparing MaxContext methods (idea:007) against TopDown Bubblesort on the comparisons-axis or wall-clock-axis frontier, use `TD-Bubble` at its standard `num_child=3` config (the existing claim:C9 frontier point), or use `MaxContext-TopDown` directly as the canonical whole-pool best-only baseline. Even after the fix, standard `TD-Bubble` is not identical to MaxContext TopDown because MaxContext uses its own prompt/parser path and a two-document BM25 bypass.

Full mechanism analysis: [`FINDINGS.md`](../FINDINGS.md) under the 2026-04-27 entry; experiment-page note: [`exp:main_td_bubble`](../experiments/main_td_bubble.md).

## Supporting experiments

- exp:analysis_pareto
- (reference artifact: `results/analysis/pareto/QUALITY_COST_PARETO.md`)

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
