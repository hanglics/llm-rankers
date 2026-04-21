---
type: experiment
node_id: exp:analysis_per_query_wins
title: "Analysis — per-query win/loss/tie between method families"
status: completed
config:
  unit: "per-query NDCG@10"
tests: ["claim:C2", "claim:C4", "claim:C6"]
results_dir: "results/analysis/per_query_wins/"
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Query-level wins and losses across method pairs. BiDir "between" on ~27 queries average; helps ~12, hurts ~10 per model.

Used to identify queries where DualEnd decisively helps — the qualitative write-up lives at `results/analysis/dualend_qualitative/WHEN_DUALEND_HELPS_SUMMARY.md`.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
