---
type: experiment
node_id: exp:main_bidir_rrf
title: "Main sweep — Bidirectional RRF (BiDir-RRF)"
status: completed
config:
  direction: bidirectional
  fusion: rrf
  num_child: 3
  k: 10
  hits: 100
  passage_length: "128 (T5) / 512 (Qwen)"
  scoring: generation
tests: ["idea:003", "claim:C4", "claim:C8"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Fuse independent TD and BU rankings via Reciprocal Rank Fusion. Best BiDir variant on 4 / 18 configs (never beats best TD).

## Results (DL19 NDCG@10)

| Model | NDCG |
|---|---:|
| flan-t5-large | 0.5820 |
| flan-t5-xl | 0.6845 |
| flan-t5-xxl | 0.6905 |
| qwen3-4b | 0.6814 |
| qwen3-8b | 0.6826 |
| qwen3-14b | 0.7172 |
| qwen3.5-4b | 0.6614 |
| qwen3.5-9b | 0.7101 |
| qwen3.5-27b | 0.7198 |

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
