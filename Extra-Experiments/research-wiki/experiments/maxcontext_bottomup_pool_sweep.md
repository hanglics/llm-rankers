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
  invariants: "pool_size == hits == ranker.k; num_permutation == 1; strict_no_truncation; structured_numeric_parse + refusal_only_noop (BottomUp tail); out_of_window_parse_aborts"
tests: ["idea:007", "claim:C9"]
plan_doc: "/Users/hangli/projects/llm-rankers/IDEA_007.md"
origin_skill: manual-backfill
created_at: 2026-04-25T00:00:00+10:00
updated_at: 2026-04-26T00:00:00+10:00
---

# Summary

Whole-pool worst-only companion sweep for idea:007. Uses the same pool grid as `exp:maxcontext_dualend_pool_sweep`, but shrinks the live pool by one document per round from the bottom.

This page exists to compare call-count / quality trade-offs across the full MaxContext family under matched pool sizes.

## Operational notes

- 2026-04-26 — Initial cluster runs at pool ∈ {30,40} crashed on Qwen3 refusal-style outputs ("3 Passage 3 is about… not relevant…" at pool=50). Parser was hardened with structured-numeric extraction, a numeric-only refusal regex, refusal-only deterministic no-op (BottomUp picks tail → no swap), and a window-size-aware prompt suffix. New per-query stat `Avg parse fallbacks:` printed when non-zero. See `research-wiki/log.md` 2026-04-26 entry for the full diff. Re-launch pending.
- BottomUp's bypass impact remains the head of ranking (decides ranks 1 vs 2 at `n_docs=2`); the new refusal-no-op is a separate failure-mode mitigation that pins the live tail when the model truly refuses on a larger window. Both behaviours are exposed via `Avg BM25 bypass` and `Avg parse fallbacks`.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
