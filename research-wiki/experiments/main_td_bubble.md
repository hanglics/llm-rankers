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
updated_at: 2026-04-27T23:48:07+10:00
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

## Note — `Avg comparisons` under `hits=k=num_child` whole-pool config

The standard `num_child=3, k=10, hits=100` config above is interpretable and unaffected.

A pre-fix whole-pool run (`hits=10, k=10, num_child=10`) reported `Avg comparisons: 6.977` because of a control-flow interaction in `llmrankers/setwise.py:SetwiseLlmRanker.rerank()`:

- The upstream `last_start += len(window) - 1` tail-jump could fire when a round selected the last item in the window.
- The local outer clamp then moved `last_start` to the outer-loop index, and the local one-document guard skipped later one-doc suffixes.
- In the observed DL19 run, this skipped about 87 comparison opportunities across 43 queries.

This is fixed for the exact whole-pool branch: the local outer clamp is now disabled only when `len(ranking) == k == num_child` or `num_child >= len(ranking)`, while the one-document skip remains. Focused verification gives:

```text
n=10, num_child=10, k=10: no_swap=9, always_tail=9
```

Do not use the archived pre-fix `6.977` value as a current efficiency claim. For canonical MaxContext whole-pool best-only baselines, use `MaxContextTopDownSetwiseLlmRanker` directly (`exp:maxcontext_topdown_pool_sweep`), because MaxContext has its own prompt/parser path and two-document BM25 bypass semantics. Full mechanism analysis in [`research_pipeline_setwise/FINDINGS.md`](../../research_pipeline_setwise/FINDINGS.md) under the 2026-04-27 entry.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
