---
type: experiment
node_id: exp:main_td_bubble
title: "Main sweep — TopDown Bubblesort (TD-Bubble)"
status: completed
config:
  direction: topdown
  method: bubblesort
  num_child: 3
  k: 10
  hits: 100
  passage_length: "128 (T5) / 512 (Qwen)"
  scoring: generation
tests: ["claim:C2", "claim:C6", "claim:C9"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

TopDown setwise + adjacent-pair bubblesort. Best TopDown variant on 4 of 6 T5 configs and 4 of 12 Qwen configs.

## Results (DL19 NDCG@10)

| Model | NDCG |
|---|---:|
| flan-t5-large | 0.6874 |
| flan-t5-xl | 0.6980 |
| flan-t5-xxl | 0.7077 |
| qwen3-4b | 0.6491 |
| qwen3-8b | 0.6794 |
| qwen3-14b | 0.7455 |
| qwen3.5-4b | 0.7108 |
| qwen3.5-9b | 0.7349 |
| qwen3.5-27b | 0.7435 |

## Cost

Mean 300 comparisons / query; ~126K tokens; ~110.9s wall-clock. Frontier member.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
