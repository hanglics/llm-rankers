---
type: idea
node_id: idea:007
title: "MaxContext family — one-prompt whole-pool selection over ≤50-doc pools"
stage: active
outcome: pending
based_on: ["paper:zhuang2024_setwise", "paper:sato2026_sorting_survey", "paper:zhuang2025_rank_r1"]
target_gaps: ["gap:G1", "gap:G4"]
refines: ["idea:002"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:55:00+10:00
updated_at: 2026-04-20T10:55:00+10:00
---

# Idea

Fit the entire rerank pool (`pool_size ≤ 50`) into a single Qwen prompt. The family now has three variants: ask for best+worst in one call (DualEnd), ask only for the best (TopDown), or ask only for the worst (BottomUp). DualEnd shrinks the live pool by 2 each round; TopDown and BottomUp shrink it by 1.

## Motivation

`DE-Cocktail` / `DE-Selection` (idea:002) win 14/18 configs on TREC DL but cost 5.6–8.9× TD-Heap wall-clock (claim:C2, claim:C6). Codex-verified Pareto frontier (claim:C9) has an empty region between TD-Bubble and DE-Cocktail. MaxContext trades many small joint-elicitation calls for very few large-window joint-elicitation calls — potentially cheaper wall-clock at comparable quality.

## Mechanism (Qwen-generation only)

- `--direction maxcontext_dualend`.
- `--direction maxcontext_topdown`.
- `--direction maxcontext_bottomup`.
- New ranker class `MaxContextDualEndSetwiseLlmRanker` in `llmrankers/setwise_extended.py`.
- New ranker classes `MaxContextTopDownSetwiseLlmRanker` and `MaxContextBottomUpSetwiseLlmRanker` in `llmrankers/setwise_extended.py`.
- Reuses `_double_ended_selection` (`setwise_extended.py:840-942`) with `num_child = pool_size - 1` so the single-group fast-path fires for the whole pool.
- **Numeric labels 1..N** (not letters). Existing `CHARACTERS = ['A'..'W']` capped windows at 23; numeric-label refactor touches prompt construction, parser validation, fallback clamping, JSONL schema, and `analysis/position_bias.py`.
- **Hard invariants** (aborts on violation): Qwen3 / Qwen3.5 generation only (no T5, no likelihood); `pool_size == hits == ranker.k`; `num_permutation == 1`; context-fit preflight passes; zero truncation; every call yields a full-parse (no silent fallback).
- **Preflight** computes max-fit with actual tokenization of the fully rendered prompt (not arithmetic — `setwise.py:662`'s `truncate()` does a tokenize→decode→re-tokenize roundtrip that invalidates closed-form bounds).

## Variants

- **DualEnd** — `type=dual_best` + `type=dual_worst`, call count `floor(N / 2)`.
- **TopDown** — `type=best`, call count `N - 1`.
- **BottomUp** — `type=worst`, call count `N - 1`.

## Predicted outcome

- Matches or beats `DE-Cocktail` on NDCG@10 at matched `hits` (headline comparison rule).
- Lands in the empty region between `TD-Bubble` and `DE-Cocktail` on the **comparisons-axis** and **wall-clock-axis** frontiers.
- **Token axis is expected to be worse** (more prompt tokens per query than DE-Cocktail) — not claimed as a win.

## Status

Active, tests not yet run. Detailed plan in `/Users/hangli/projects/llm-rankers/IDEA_007.md` (Codex-audited 3 rounds, gpt-5.4 xhigh, ready to execute 2026-04-20). Staged execution: Phase 1 sanity → Phase 2 order-pilot → Phase 3 matched-hits regression → Phase 4 Study A + baselines → Phase 5 Study B.

## Tested by

- exp:maxcontext_dualend_pool_sweep (Study A)
- exp:maxcontext_dualend_pl_sweep (Study B)
- exp:maxcontext_dualend_order_pilot (Study C)
- exp:maxcontext_dualend_baselines (matched-hits baselines)
- exp:maxcontext_topdown_pool_sweep
- exp:maxcontext_bottomup_pool_sweep
- exp:maxdoc_dualend_pending (superseded by the above; kept for traceability)

## Related

- Refines idea:002 (DualEnd).
- Addresses gap:G1 (info-per-call) and gap:G4 (efficiency-effectiveness frontier).
- Inspired by paper:zhuang2025_rank_r1 (num_child=19 / 20-way setwise as a known-feasible precedent).
- Inspired by paper:sato2026_sorting_survey (the sorting-as-rerank framing that predicts this works).

## Risks (top four)

1. **Long-context attention degradation** at w=50 (paper:liu2024_lost_in_middle). Study B's control arm is the designed test.
2. **Numeric-label parse fragility at N=50.** Existing parser's silent-default behaviour is a landmine. Abort-on-bad-parse is mandatory.
3. **Order sensitivity.** Study C gates the full matrix (max pairwise NDCG@10 Δ across orderings ≤ 0.01).
4. **Token-frontier framing.** MaxContext uses more prompt tokens than DE-Cocktail; claim:C9's token axis cannot improve. Paper framing must be precise.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
