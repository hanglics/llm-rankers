---
type: experiment
node_id: exp:same_method_tables_pending
title: "Pending — same-method/same-sort consolidated result tables"
status: not_submitted
config:
  description: "Per Need_to_Run.txt top priority: build result tables that fix a single (method × sort × model × dataset) cell rather than reporting only best-per-family."
  rationale: "Current wiki and narrative use best-per-family numbers; the same-method tables are required for referees comparing like-with-like."
  source_artifacts: ["Need_to_Run.txt (lines 3-4)"]
tests: ["claim:C2", "claim:C6"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:40:00+10:00
---

# Summary

**User-declared top priority** ahead of running new experiments. The next consolidation pass must produce tables that show, for a fixed (model, dataset, method, sort) combination, the NDCG@10 of every direction (TD, BU, DualEnd, BiDir) — not just the best-in-family value used in the significance test.

Without these tables the paper cannot fairly compare, e.g. BU-Bubblesort vs DualEnd-Bubblesort at the *same* sort.

## Status

Not started. Data exists in `/Users/hangli/projects/llm-rankers/results/{model}-{dataset}/` — just needs aggregation.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
