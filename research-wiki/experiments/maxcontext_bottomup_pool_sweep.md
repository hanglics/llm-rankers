---
type: experiment
node_id: exp:maxcontext_bottomup_pool_sweep
title: "Study A-BU — MaxContext BottomUp pool-size sweep at fixed pl=512"
status: not_submitted
config:
  direction: maxcontext_bottomup
  passage_length: 512
  scoring: generation
  qwen_thinking: false
  pool_size_grid: [10, 20, 30, 40, 50]
  models: ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B"]
  datasets: ["dl19", "dl20"]
  invariants: "pool_size == hits == ranker.k; num_permutation == 1; strict_no_truncation; abort_on_bad_parse"
tests: ["idea:007", "claim:C9"]
plan_doc: "/Users/hangli/projects/llm-rankers/IDEA_007.md"
origin_skill: manual-backfill
created_at: 2026-04-25T00:00:00+10:00
updated_at: 2026-04-25T00:00:00+10:00
---

# Summary

Whole-pool worst-only companion sweep for idea:007. Uses the same pool grid as `exp:maxcontext_dualend_pool_sweep`, but shrinks the live pool by one document per round from the bottom.

This page exists to compare call-count / quality trade-offs across the full MaxContext family under matched pool sizes.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
