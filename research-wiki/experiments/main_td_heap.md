---
type: experiment
node_id: exp:main_td_heap
title: "Main sweep — TopDown Heapsort (TD-Heap)"
status: completed
config:
  direction: topdown
  method: heapsort
  num_child: 3
  k: 10
  hits: 100
  passage_length: "128 (T5) / 512 (Qwen)"
  scoring: generation
  models: ["flan-t5-large", "flan-t5-xl", "flan-t5-xxl", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b"]
  datasets: ["msmarco-passage/trec-dl-2019", "msmarco-passage/trec-dl-2020"]
tests: ["claim:C2", "claim:C9"]
results_dir_pattern: "results/<model>-<dataset>/ (one subdir per run, e.g. results/flan-t5-xl-dl19/)"
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Reference baseline from `paper:zhuang2024_setwise`. Heapsort over setwise "which is most relevant?" windows. 18 runs total (9 models × 2 datasets).

## Results (NDCG@10)

| Model | DL19 | DL20 |
|---|---:|---:|
| flan-t5-large | 0.6541 | — |
| flan-t5-xl | 0.6901 | — |
| flan-t5-xxl | 0.6846 | — |
| qwen3-4b | 0.6775 | 0.6488 |
| qwen3-8b | 0.6819 | 0.6532 |
| qwen3-14b | 0.7447 | — |
| qwen3.5-4b | 0.7087 | — |
| qwen3.5-9b | 0.7329 | 0.6950 |
| qwen3.5-27b | 0.7449 | — |

(Missing cells are present on disk — filled in the overall significance table, not re-listed here to avoid duplication.)

## Cost

Mean 76.5 comparisons / query; ~32K tokens; ~28.4s wall-clock (cross-model average). Frontier member.

## Tested by (role: reference baseline)

This sweep *is* the baseline the challenger families (BU, DE, BiDir) are measured against.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
