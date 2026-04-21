---
type: experiment
node_id: exp:ablation_passage_length
title: "Ablation — passage_length ∈ {64, 128, 256, 512}"
status: completed
config:
  grid: "passage_length ∈ {64, 128, 256, 512}"
  methods: ["TopDown", "DualEnd"]
results_dir: "results/ablation-pl/"
tests: ["claim:C2", "claim:C7"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

T5 plateaus at `pl=128`; Qwen plateaus at `pl=256`. Truncation has negligible effect beyond these thresholds.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
