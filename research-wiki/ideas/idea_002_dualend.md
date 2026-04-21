---
type: idea
node_id: idea:002
title: "DualEnd setwise ranking (joint best+worst in one call)"
stage: succeeded_partial
outcome: positive_with_caveats
based_on: ["paper:zhuang2024_setwise", "paper:sato2026_sorting_survey"]
target_gaps: ["gap:G1", "gap:G2", "gap:G3"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Idea

In a single setwise prompt, ask the LLM for **both** the most relevant **and** the least relevant passage. Use cocktail-shaker sort (`DE-Cocktail`, bubblesort variant that swaps from both ends) or double-ended selection sort (`DE-Selection`) to consume both outputs per comparison.

## Motivation

A setwise call elicits internal reasoning over all candidates. Extracting only "best" discards information. Asking for "best AND worst" in the same prompt should (a) double the information per call and (b) ground the worst selection with the same evidence as the best, plausibly dodging the failure mode that sank standalone BottomUp (idea:001).

## Mechanism

- **Prompt:** "Among A, B, C, D, which is MOST relevant and which is LEAST relevant?" Output: `Best: X, Worst: Y`.
- **Scoring paths (important — both T5 generation and `likelihood` mode use the same best-only-proxy shortcut; verified in `llmrankers/setwise_extended.py:433-491`):**
  - `generation` on supported causal models (Qwen3 / Qwen3.5): the model decodes a single `Best: X, Worst: Y` string; the parser extracts both labels. This is the only path that performs **true joint elicitation**.
  - `generation` on Flan-T5 (`model_type == 't5'`): T5 cannot reliably emit the dual-format string (512-token context truncation + literal-template echo). The code silently falls back to a **best-only proxy**: run `_build_best_prompt` once, score each label's likelihood, sort, take argmax as `best` and argmin as `worst`. This is one forward pass with zero completion tokens — correct behavior per `RESEARCH_BRIEF.md` line 49, not a bug.
  - `likelihood` scoring (all models, including Qwen): always uses the same best-only-proxy shortcut as T5 generation. Documented explicitly at `setwise_extended.py:481-482` ("this is a best-only proxy for DualEnd, not an exact likelihood model of the full 'Best: X, Worst: Y' output string") and in README lines 423–427.
  - **Consequence:** When you see DualEnd `--scoring likelihood` or DualEnd T5 numbers, the worst signal is derived from the argmin of the best-only label distribution — it is not an independently-scored worst elicitation.
- **Sort algorithms:**
  - `DE-Cocktail` (bubblesort with alternating forward/backward passes; one pass consumes both best and worst per window).
  - `DE-Selection` (double-ended selection sort; places one best and one worst per scan).
- **CLI:** `--direction dualend --method {bubblesort, selection}`.

## Expected outcome

DualEnd dominates both TopDown alone and BottomUp alone. Cost increases because cocktail-shaker has O(n²) comparisons, but information per call doubles.

## Actual outcome — positive with caveats (claim:C2, claim:C6)

- **Wins 14 of 18** model–dataset configs vs best TopDown. **All 12 Qwen configs** favour DualEnd.
- `DE-Cocktail` is the strongest single variant (best in 11 configs).
- Family mean Δ = **+0.0058** over best TopDown; positive in 14/18.
- **Only 1 Bonferroni-significant win:** `qwen3-4b` DL19, `DE-Selection` 0.7220 vs `TD-Heap` 0.6775, Δ +0.0446, p = 0.010 after correction.
- `qwen3-8b` DL19 `DE-Selection` 0.7158 vs TD-Heap 0.6819 has raw p = 0.040 but fails Bonferroni (corrected p = 0.727).
- 0 Bonferroni-significant losses.
- On T5 models `TD-Bubble` remains best on 4 of 6 configs — DualEnd does not dominate T5.

## Why it (partially) works

- Joint prompt forces the model to consider the full relevance spectrum, improving calibration on the "best" selection (bias flattens toward uniform — see claim:C5).
- The "worst" selection is **only** usable when conditioned on same-call best; standalone worst-selection was a disaster (idea:001).

## Cost

- `DE-Cocktail` ≈ 546 comparisons / query vs `TD-Heap`'s ≈ 77 (7.1× more calls).
- Total tokens ≈ 233K vs ≈ 32K for `TD-Heap`.
- Wall-clock: 5.6×–8.9× slower (T5 avg).

## Failure notes (modes to watch)

- Small TREC DL test sets (43 / 54 queries) → most positive deltas are not Bonferroni-significant; framing must be "directional win", not "universal improvement" (claim:C6).
- Only the Qwen-generation path exercises the true joint-elicitation prompt. T5 generation and all `--scoring likelihood` paths collapse to the best-only-proxy shortcut; any quality gain they show is best signal alone. This confound must be disclosed in the paper.
- Gains are modest on the largest models, where TopDown already saturates.

## Refinements (separate ideas)

- idea:004 `selective_dualend` — routed activation only on uncertain windows.
- idea:005 `bias_aware_dualend` — controlled orderings + majority vote on hard windows.
- idea:006 `samecall_regularized` — use worst output only as local demotion.

## Tested by

- exp:main_de_cocktail
- exp:main_de_selection
- exp:analysis_significance_tests
- exp:analysis_position_bias (dual_best, dual_worst bias curves)
- exp:analysis_pareto

## Related

- Inspired by: paper:zhuang2024_setwise, paper:sato2026_sorting_survey (algorithmic language)
- Addresses gap: gap:G1 (information extraction), gap:G2 (position bias), gap:G3 (worst unreliability)

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
