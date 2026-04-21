---
type: experiment
node_id: exp:maxcontext_dualend_baselines
title: "Matched-hits baselines for MaxContext DualEnd (predeclared depths)"
status: not_submitted
config:
  baseline_methods: ["TD-Heap", "TD-Bubble", "DE-Cocktail", "DE-Selection"]
  pool_size_anchors: [10, 30, 50]
  models: ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B"]
  datasets: ["dl19", "dl20"]
  k: 10
  num_child: "3 (for DE-Cocktail, DE-Selection)"
tests: ["idea:007", "claim:C2", "claim:C9"]
plan_doc: "/Users/hangli/projects/llm-rankers/IDEA_007.md"
origin_skill: manual-backfill
created_at: 2026-04-20T10:55:00+10:00
updated_at: 2026-04-20T10:55:00+10:00
---

# Summary

Mandatory matched-`hits` baselines for IDEA_007. Codex round 1 catch: old `hits=100` runs cannot be subsetted to `hits=50` because `run.py:245` reads only `hits` BM25 docs before any LLM call. Codex round 2 catch: depth must be predeclared; choosing after Study A creates a selection confound.

**Predeclared grid:** `pool_size ∈ {10, 30, 50}` × 6 Qwens × 2 datasets × 4 methods = **144 runs**.

**Budget-restricted alternative** (if GPU budget is tight): `pool_size ∈ {10, 50}` = 96 runs. Loses mid-range comparison; not recommended for headline claim.

**Headline comparison rule:** for each `pool_size ∈ {10, 30, 50}`, MaxContext's NDCG@10 must match or exceed the best baseline at the same `pool_size` (and matched `k=10`) on a bootstrap-CI basis. Cross-`pool_size` comparisons are reported as context only.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
