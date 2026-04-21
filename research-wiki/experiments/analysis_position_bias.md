---
type: experiment
node_id: exp:analysis_position_bias
title: "Analysis — positional bias across directions (best / worst / dual_best / dual_worst)"
status: completed
config:
  models: 9
  datasets: 2
  windows: "w=4 (labels A/B/C/D)"
  tests: "chi-squared goodness-of-fit vs uniform 0.25"
tests: ["claim:C5", "claim:C1"]
results_dir: "results/analysis/position_bias/"
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Per-position selection frequency broken down by direction and by model. Source of claim:C5 (novel dual_worst primacy reversal).

## Headline numbers (DL19, cross-model range)

| Direction | A-freq | D-freq | Pattern |
|---|---:|---:|---|
| TopDown (best) | 0.23–0.50 | 0.25–0.55 | U-shape |
| BottomUp (worst) | 0.06–0.24 | 0.40–0.63 | Recency dominates |
| DualEnd dual_best | ~flat | 0.27–0.48 | Flatter than single-best |
| DualEnd dual_worst | 0.23–0.43 | 0.10–0.19 | Primacy reversal |

χ² tests reject uniform for all four directions at p < 0.001 across every model.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
