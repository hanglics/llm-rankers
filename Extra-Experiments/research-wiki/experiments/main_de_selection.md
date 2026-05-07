---
type: experiment
node_id: exp:main_de_selection
title: "Main sweep — DualEnd Selection (DE-Selection)"
status: completed
config:
  direction: dualend
  method: selection
  num_child: 3
  k: 10
  hits: 100
  passage_length: "128 (T5) / 512 (Qwen)"
  scoring: generation
tests: ["idea:002", "claim:C2", "claim:C6"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Double-ended selection sort with same joint best+worst prompt. Best DualEnd variant on small Qwen models (qwen3-4b, qwen3-8b DL19). Holds the only Bonferroni-significant DualEnd win.

## Results (DL19 NDCG@10)

| Model | NDCG |
|---|---:|
| flan-t5-large | 0.6420 |
| flan-t5-xl | 0.6792 |
| flan-t5-xxl | 0.6974 |
| qwen3-4b | **0.7220** (Bonferroni-sig vs TD-Heap, Δ +0.0446, p=0.010) |
| qwen3-8b | 0.7158 (raw p=0.040; Bonferroni p=0.727) |
| qwen3-14b | 0.7475 |
| qwen3.5-4b | 0.7022 |
| qwen3.5-9b | 0.7309 |
| qwen3.5-27b | 0.7319 |

## Cost

Mean ~406 comparisons / query; cheaper than DE-Cocktail but still ~5.3× TD-Heap.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
