---
type: experiment
node_id: exp:ablation_num_child
title: "Ablation — window size (num_child ∈ {2, 3, 5, 7})"
status: completed
config:
  grid: "num_child ∈ {2, 3, 5, 7}"
  methods: ["TopDown", "DualEnd"]
  models: ["flan-t5-xl", "qwen3-8b", "qwen3.5-9b"]
  datasets: ["DL19"]
tests: ["claim:C7"]
results_dir: "results/ablation-nc/"
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

How does window size interact with model family? T5 degrades with larger windows (context-limited); Qwen is stable or slightly improves.

## DualEnd-Cocktail DL19

| Model | nc=2 | nc=3 | nc=5 | nc=7 |
|---|---:|---:|---:|---:|
| flan-t5-xl | 0.6988 | 0.6884 | 0.6749 | 0.6480 |
| qwen3-8b | 0.7187 | 0.7155 | 0.7224 | 0.7249 |
| qwen3.5-9b | 0.7392 | 0.7370 | 0.7336 | 0.7386 |

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
