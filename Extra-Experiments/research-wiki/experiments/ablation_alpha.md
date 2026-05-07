---
type: experiment
node_id: exp:ablation_alpha
title: "Ablation — BiDir weighted α ∈ {0.3, 0.5, 0.7, 0.9}"
status: completed
config:
  grid: "alpha ∈ {0.3, 0.5, 0.7, 0.9}"
  method: "BiDir weighted"
  direction: bidirectional
  fusion: weighted
  num_child: 3
  scoring: generation
tests: ["idea:003", "claim:C4"]
results_dir: "results/ablation-alpha/"
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

α=0.9 (90% TopDown) consistently best across models. Result reinforces claim:C4: the more TD we weight, the better — which is equivalent to not using the BU signal at all.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
