---
type: experiment
node_id: exp:maxcontext_dualend_pool_sweep
title: "Study A — MaxContext DualEnd pool-size sweep at fixed pl=512"
status: not_submitted
config:
  direction: maxcontext_dualend
  passage_length: 512
  scoring: generation
  qwen_thinking: false
  pool_size_grid: [10, 20, 30, 40, 50]
  models: ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B"]
  datasets: ["dl19", "dl20"]
  invariants: "pool_size == hits == ranker.k; num_permutation == 1; strict_no_truncation; abort_on_bad_parse"
tests: ["idea:007", "claim:C2", "claim:C9"]
plan_doc: "/Users/hangli/projects/llm-rankers/IDEA_007.md"
origin_skill: manual-backfill
created_at: 2026-04-20T10:55:00+10:00
updated_at: 2026-04-20T10:55:00+10:00
---

# Summary

Study A from IDEA_007. 6 Qwens × 5 pool sizes × 2 datasets = **60 runs**. Fixed `passage_length = 512`, `direction = maxcontext_dualend`.

**Hypothesis:** NDCG@10 saturates, not monotonic — plateau point identifies the efficient pool size. That plateau pool is used as fixed parameter in Study B.

**Gate before submission:** Phase 1 (sanity) and Phase 2 (order-robustness pilot) must pass.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
