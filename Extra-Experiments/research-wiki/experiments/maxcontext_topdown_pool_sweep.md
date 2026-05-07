---
type: experiment
node_id: exp:maxcontext_topdown_pool_sweep
title: "Study A-TD — MaxContext TopDown pool-size sweep at fixed pl=512"
status: not_submitted
config:
  direction: maxcontext_topdown
  passage_length: 512
  scoring: generation
  qwen_thinking: false
  pool_size_grid: [10, 20, 30, 40, 50]
  models: ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B"]
  datasets: ["dl19", "dl20"]
  invariants: "pool_size == hits == ranker.k; num_permutation == 1; strict_no_truncation; structured_numeric_parse + refusal_only_noop (TopDown head); out_of_window_parse_aborts"
tests: ["idea:007", "claim:C9"]
plan_doc: "/Users/hangli/projects/llm-rankers/IDEA_007.md"
origin_skill: manual-backfill
created_at: 2026-04-25T00:00:00+10:00
updated_at: 2026-04-26T00:00:00+10:00
---

# Summary

Whole-pool best-only companion sweep for idea:007. Uses the same pool grid as `exp:maxcontext_dualend_pool_sweep`, but shrinks the live pool by one document per round instead of two.

This page exists to compare call-count / quality trade-offs across the full MaxContext family under matched pool sizes.

## Operational notes

- 2026-04-26 — Initial cluster runs at pool ∈ {20,30,40,50} crashed on Qwen3 refusal-style outputs (`"None of the passages are relevant…"`, `"3 Passage 3 is about… not relevant…"`). Parser was hardened with structured-numeric extraction, a numeric-only refusal regex, refusal-only deterministic no-op (TopDown picks head → no swap), and a window-size-aware prompt suffix. New per-query stat `Avg parse fallbacks:` printed when non-zero. See `research-wiki/log.md` 2026-04-26 entry for the full diff. Re-launch pending.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
