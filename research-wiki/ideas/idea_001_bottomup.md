---
type: idea
node_id: idea:001
title: "BottomUp setwise ranking (reverse selection)"
stage: failed
outcome: negative
based_on: ["paper:zhuang2024_setwise", "paper:qin2024_prp"]
target_gaps: ["gap:G1", "gap:G3"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Idea

Reverse the setwise prompt to ask "which passage is LEAST relevant?" and build the ranking from the bottom up with heapsort or bubblesort over the reversed comparator.

## Motivation

If the comparator can be reversed, the "worst" signal would be a complementary rank source to the "best" signal. At minimum it would let us validate the top-down signal; at best it could be fused as an ensemble (→ idea:003) or combined with top-down inside a single prompt (→ idea:002).

## Mechanism

- Same setwise prompt skeleton as `zhuang2024_setwise`, but the question is "Which is the LEAST relevant to the query?" (labels A/B/C/D unchanged).
- `--direction bottomup` in `llmrankers/setwise_extended.py`.
- Sort algorithms: heapsort (`BU-Heap`), bubblesort (`BU-Bubble`).

## Predicted outcome

"Worst" is approximately as easy as "best" for the LLM (assumes symmetric competence). Expected parity or small loss vs `TD-Heap`.

## Actual outcome — negative

| Model | Dataset | TD best | BU best | Δ | Bonferroni |
|-------|---------|--------:|--------:|--:|-----------|
| flan-t5-large | DL19 | 0.6874 (TD-Bubble) | 0.4571 (BU-Bubble) | −0.2302 | **sig loss** |
| flan-t5-large | DL20 | 0.6264 | 0.4116 | −0.2148 | **sig loss** |
| qwen3-14b | DL20 | 0.7044 | 0.6395 | −0.0649 | **sig loss** |
| qwen3.5-4b | DL19 | 0.7108 | 0.6158 | −0.0950 | **sig loss** |
| qwen3.5-4b | DL20 | 0.6713 | 0.5963 | −0.0749 | **sig loss** |
| qwen3.5-9b | DL19 | 0.7349 | 0.6779 | −0.0570 | **sig loss** |

- Family mean Δ vs best TopDown: **−0.0616** across 18 configs.
- **0 / 18 positive deltas**; 6 Bonferroni-significant losses, 0 wins.
- Catastrophic on `flan-t5-large` (−23 NDCG points).

## Failure notes (why it failed — keep for future ideation)

1. **LLM training asymmetry.** Instruction-tuning rewards identifying correct/best/relevant. "Least relevant" is structurally alien; models fall back on position heuristics.
2. **Recency collapse.** BU-alone shows overwhelming recency bias: D-freq 0.40–0.63 for the last position vs 0.25 uniform. The model "picks the last one" as a default (see exp:analysis_position_bias).
3. **Efficiency is worse, not better, for top-k.** For top-10 of 100, we need to exclude 90 documents via BU but only select 10 via TD — so BU is 9× more comparisons for the same useful output.
4. **Not rescued by bigger models.** Even `flan-t5-xxl` and `qwen3.5-27b` show negative deltas; this is not a capacity problem.

## Lessons learned (do not re-try in this form)

- Standalone worst-selection is unreliable. Use it only inside the same prompt as best-selection, where the joint task can ground it (→ idea:002 DualEnd).
- Fusing BU with TD independently imports the BU bias (→ idea:003 BiDir, also failed).
- Worst signal might still be usable as a *local* demotion constraint after the head frontier is decided (→ idea:006 `samecall_regularized`, pending).

## Tested by

- exp:main_bu_heap
- exp:main_bu_bubble
- exp:analysis_significance_tests
- exp:analysis_position_bias

## Related

- Inspired by: paper:zhuang2024_setwise (baseline to reverse)
- Contradicts symmetric-competence assumption implicit in paper:qin2024_prp

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
