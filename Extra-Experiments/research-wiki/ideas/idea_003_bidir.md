---
type: idea
node_id: idea:003
title: "Bidirectional ensemble (independent TopDown + BottomUp fusion)"
stage: failed
outcome: negative
based_on: ["paper:tang2024_found_in_middle", "paper:zeng2024_llm_rankfusion"]
target_gaps: ["gap:G1"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Idea

Run TopDown and BottomUp **independently** over the same candidate set, then fuse the two rankings via RRF, CombSUM, or weighted combination. Intuition: if the two directions capture complementary signals, their fusion should beat either alone.

## Mechanism

- `--direction bidirectional`.
- Two independent runs over the BM25 top-100: one with `topdown`, one with `bottomup`.
- Fusion operators: `rrf` (reciprocal rank), `combsum` (score sum), `weighted` (α·TD + (1−α)·BU). α grid: {0.3, 0.5, 0.7, 0.9}.

## Predicted outcome

Rank fusion helps when both input rankings are independently competitive and roughly symmetric in noise (`paper:zeng2024_llm_rankfusion`). So BiDir should narrowly beat TopDown, at least on models where BU is not catastrophic.

## Actual outcome — negative (claim:C4)

| Aspect | Value |
|---|---|
| Family mean Δ vs best TopDown | **−0.0232** |
| Positive deltas | 3 / 18 (all tiny, +0.0008–+0.0068) |
| Bonferroni-significant losses | 3 (`flan-t5-large` DL19, DL20; `qwen3.5-27b` DL20) |
| Best fusion rule | α = 0.9 (90% TopDown) |

- Ranking agreement: TopDown–BottomUp Overlap@10 ≈ 5.04, Kendall τ ≈ 0.859.
- For comparison, TopDown–DualEnd Overlap@10 ≈ 7.01, τ ≈ 0.925 (much higher agreement → why DualEnd fusion-in-prompt works while BiDir fusion-after doesn't).

## Failure notes (why it failed)

1. **BottomUp is systematically biased, not symmetrically noisy.** Rank fusion helps with the latter, hurts with the former — it imports the bias (consistent with `paper:zeng2024_llm_rankfusion`).
2. **Low ranking agreement (overlap ≈ 5 / 10) + weak BU** ⇒ fusion mostly stirs in noise.
3. **Heavy α=0.9 weighting** partly mitigates the harm, but at α=0.9 the ensemble is essentially TopDown alone — no complementarity is being exploited.
4. **Mechanistic implication (claim:C8):** worst-selection information is only useful when *same-call* grounded (idea:002 DualEnd), not when produced independently.

## Lessons learned

- Don't rank-fuse a strong signal with a systematically biased signal. The weakest-link dominates.
- For setwise in particular: signal complementarity must be inside the prompt, not across independent runs.
- BiDir is a coherent **negative result** that supports the project's central story: joint elicitation is the right level for "more information per call".

## Tested by

- exp:main_bidir_rrf
- exp:main_bidir_wt
- exp:ablation_alpha
- exp:analysis_ranking_agreement

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
