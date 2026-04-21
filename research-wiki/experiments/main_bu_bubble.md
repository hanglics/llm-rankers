---
type: experiment
node_id: exp:main_bu_bubble
title: "Main sweep — BottomUp Bubblesort (BU-Bubble)"
status: completed
config:
  direction: bottomup
  method: bubblesort
  num_child: 3
  k: 10
  hits: 100
  passage_length: "128 (T5) / 512 (Qwen)"
  scoring: generation
tests: ["idea:001", "claim:C1", "claim:C3"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Worst-selection with adjacent-pair bubblesort. The best BottomUp variant on 6 of 18 configs but still loses to TD every time.

## Results (DL19 NDCG@10)

| Model | NDCG |
|---|---:|
| flan-t5-large | 0.4571 |
| flan-t5-xl | 0.6730 |
| flan-t5-xxl | 0.6936 |
| qwen3-4b | 0.6305 |
| qwen3-8b | 0.6273 |
| qwen3-14b | 0.6702 |
| qwen3.5-4b | 0.6120 |
| qwen3.5-9b | 0.6712 |
| qwen3.5-27b | 0.7336 |

## Cost

Mean ≈ 1683 comparisons / query — the most expensive BU variant (and the most expensive main-sweep variant overall).

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
