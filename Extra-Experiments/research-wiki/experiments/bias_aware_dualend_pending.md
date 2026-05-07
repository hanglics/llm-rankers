---
type: experiment
node_id: exp:bias_aware_dualend_pending
title: "Bias-Aware DualEnd — all models (not submitted)"
status: not_submitted
config:
  models: ["flan-t5-xl", "Qwen3-8B", "Qwen3.5-9B"]
  datasets: ["dl19", "dl20"]
  direction: bias_aware_dualend
  method: bubblesort
  gate_strategy: hybrid
  uncertainty_percentile: 0.15
  controlled_orderings: 3
  scoring: ["generation", "likelihood"]
tests: ["idea:005", "claim:C5"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Status

12 runs pending (3 models × 2 datasets × 2 scorings, hybrid gating only).

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
