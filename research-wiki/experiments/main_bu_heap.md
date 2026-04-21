---
type: experiment
node_id: exp:main_bu_heap
title: "Main sweep — BottomUp Heapsort (BU-Heap)"
status: completed
config:
  direction: bottomup
  method: heapsort
  num_child: 3
  k: 10
  hits: 100
  passage_length: "128 (T5) / 512 (Qwen)"
  scoring: generation
tests: ["idea:001", "claim:C1", "claim:C3", "claim:C8"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Worst-selection with heapsort scaffold. Consistently weaker than TD-Heap; catastrophic on small T5.

## Results (DL19 NDCG@10)

| Model | NDCG | Δ vs TD best |
|---|---:|---:|
| flan-t5-large | 0.2888 | −0.3986 |
| flan-t5-xl | 0.6630 | −0.0350 |
| flan-t5-xxl | 0.6874 | −0.0203 |
| qwen3-4b | 0.6261 | −0.0514 |
| qwen3-8b | 0.6431 | −0.0388 |
| qwen3-14b | 0.6966 | −0.0489 |
| qwen3.5-4b | 0.6158 | −0.0950 |
| qwen3.5-9b | 0.6779 | −0.0570 |
| qwen3.5-27b | 0.7135 | −0.0314 |

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
