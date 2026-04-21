---
type: experiment
node_id: exp:main_de_cocktail
title: "Main sweep — DualEnd Cocktail (DE-Cocktail)"
status: completed
config:
  direction: dualend
  method: bubblesort
  num_child: 3
  k: 10
  hits: 100
  passage_length: "128 (T5) / 512 (Qwen)"
  scoring: generation
tests: ["idea:002", "claim:C1", "claim:C2", "claim:C5", "claim:C6", "claim:C8", "claim:C9"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Joint best+worst prompt consumed by cocktail-shaker (bubblesort with alternating passes). Best DualEnd variant in 11 of 18 configs; frontier member.

## Results (DL19 NDCG@10)

| Model | NDCG | Δ vs TD best |
|---|---:|---:|
| flan-t5-large | 0.6708 | −0.0165 |
| flan-t5-xl | 0.6884 | −0.0096 |
| flan-t5-xxl | 0.7137 | +0.0060 |
| qwen3-4b | 0.6796 | +0.0021 |
| qwen3-8b | 0.7155 | +0.0337 |
| qwen3-14b | 0.7519 | +0.0064 |
| qwen3.5-4b | 0.7161 | +0.0052 |
| qwen3.5-9b | 0.7370 | +0.0021 |
| qwen3.5-27b | 0.7475 | +0.0026 |

## Cost

Mean 546 comparisons / query; ~233K tokens; ~212.6s wall-clock. Frontier member (7.1× TD-Heap comparisons).

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
