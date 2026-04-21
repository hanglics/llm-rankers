---
type: experiment
node_id: exp:main_bidir_wt
title: "Main sweep — Bidirectional Weighted (BiDir-Wt, α=0.7 default)"
status: completed
config:
  direction: bidirectional
  fusion: weighted
  alpha: 0.7
  num_child: 3
  k: 10
  hits: 100
  passage_length: "128 (T5) / 512 (Qwen)"
  scoring: generation
tests: ["idea:003", "claim:C4"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Weighted fusion α·TD + (1−α)·BU. Best variant on 4 of 18 configs; α=0.9 consistently strongest in the α-ablation.

## Results (DL19 NDCG@10 at α=0.7)

| Model | NDCG |
|---|---:|
| flan-t5-large | 0.6147 |
| flan-t5-xl | 0.6810 |
| flan-t5-xxl | 0.6734 |
| qwen3-4b | 0.6608 |
| qwen3-8b | 0.6784 |
| qwen3-14b | 0.7200 |
| qwen3.5-4b | 0.6714 |
| qwen3.5-9b | 0.7087 |
| qwen3.5-27b | 0.7229 |

(The α-ablation shows 0.9 > 0.7 > 0.5 > 0.3 in almost every configuration.)

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
