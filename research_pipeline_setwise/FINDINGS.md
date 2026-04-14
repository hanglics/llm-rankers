# Findings

> **Cross-stage discovery log.** Records what you learn during experiments — both research insights about your method/claims and engineering lessons from debugging. Read on every session recovery, so keep entries concise.
>
> **Why this file exists:** Experiments produce discoveries that are critical for future decisions but don't belong in formal experiment reports. Without a central log, these get lost between sessions — and the next session repeats the same mistakes or misses important signals.

---

# Research Findings

> Method-level insights: what works, what doesn't, and why. These directly inform your claims, experiment design, and paper narrative.

## [2026-04-09] The Pareto frontier sharpens the paper's refinement target

- Added [analysis/quality_cost_pareto.py](/Users/hangli/projects/llm-rankers/analysis/quality_cost_pareto.py) and generated [results/analysis/pareto/QUALITY_COST_PARETO.md](/Users/hangli/projects/llm-rankers/results/analysis/pareto/QUALITY_COST_PARETO.md)
- Global mean frontier by comparisons and total tokens contains only **`TD-Heap`**, **`TD-Bubble`**, and **`DE-Cocktail`**
- Global time frontier adds **`PermVote(p=2)`** as a fast-but-weaker point, but the main quality frontier is still `TD-Heap → TD-Bubble → DE-Cocktail`
- Mean NDCG@10 / comparisons / total tokens / time:
  - `TD-Heap`: `.6851`, `76.5`, `32464`, `28.41s`
  - `TD-Bubble`: `.6897`, `300.0`, `126456`, `110.93s`
  - `DE-Cocktail`: `.6962`, `546.0`, `233192`, `212.62s`
- Implication: the next method should not try to "beat everything." It should try to occupy the **empty region between `TD-Bubble` and `DE-Cocktail`** with a cleaner quality-cost tradeoff.

## [2026-04-09] "When DualEnd helps" is model-dependent, which argues for routing rather than unconditional use

- Added [analysis/when_dualend_helps.py](/Users/hangli/projects/llm-rankers/analysis/when_dualend_helps.py) and generated [results/analysis/dualend_qualitative/WHEN_DUALEND_HELPS_SUMMARY.md](/Users/hangli/projects/llm-rankers/results/analysis/dualend_qualitative/WHEN_DUALEND_HELPS_SUMMARY.md)
- Current summary on representative DL19 models:
  - `flan-t5-xl`: mean delta `-0.0017`, Help/Hurt/Tie `20 / 16 / 7`
  - `qwen3-8b`: mean delta `+0.0336`, Help/Hurt/Tie `26 / 12 / 5`
  - `qwen3-14b`: mean delta `+0.0072`, Help/Hurt/Tie `19 / 15 / 9`
- Interpretation:
  - DualEnd is **not** universally stronger than TopDown
  - Qwen-style models show clearer positive help/hurt asymmetry than Flan-T5-XL
  - The right refinement question is therefore **where to spend DualEnd**, not whether to replace TopDown everywhere

## [2026-04-09] The refinement package now has a clean What / Why / How split

- **Selective DualEnd**
  - **What**: TopDown sorting with same-call best-worst prompting only on shortlist or query-locally uncertain windows
  - **Why**: the DualEnd gain looks concentrated in ambiguous windows near the ranking head
  - **How**: gate by shortlist overlap, a query-local percentile over BM25 score spreads, or the hybrid union
- **Order-robust / bias-aware DualEnd**
  - **What**: apply a tiny set of controlled orderings only on hard windows and majority-vote the best / worst labels back into the original order
  - **Why**: the reversed `dual_worst` bias is the strongest mechanistic clue in the current paper
  - **How**: run base / reversed / shifted orderings only when routing says the extra calls are justified
- **Same-call worst-signal regularization**
  - **What**: keep TopDown's head-focused pass, but use the same-call worst output as a local demotion signal inside the current window
  - **Why**: standalone BottomUp is too noisy, but worst may still help when conditioned on the exact same prompt and candidate set as best
  - **How**: promote best as usual, locally push worst to the back, but avoid full backward DualEnd passes

## [2026-04-08] Significance tests narrow the paper claim: DualEnd is directionally strong, but only one win survives correction

- Added a self-contained significance workflow in [analysis/significance_tests.py](/Users/hangli/projects/llm-rankers/analysis/significance_tests.py) and saved the paper-facing report in [research_pipeline_setwise/SIGNIFICANCE_TESTS.md](/Users/hangli/projects/llm-rankers/research_pipeline_setwise/SIGNIFICANCE_TESTS.md)
- Methodology: two-sided paired approximate randomization on saved per-query `ndcg_cut_10` from `.eval` files, 100k samples, paired bootstrap 95% CIs, Bonferroni correction within each family across 18 configs
- **DualEnd vs best TopDown**: positive in 14/18 configs, mean delta `+0.0058`, but only **qwen3-4b DL19** remains significant after Bonferroni (`+0.0446`, adjusted `p=0.010`)
- **BottomUp vs best TopDown**: mean delta `-0.0616`, with **6 Bonferroni-significant losses**
- **BiDir vs best TopDown**: mean delta `-0.0232`, with **3 Bonferroni-significant losses**
- Implication: the paper should frame DualEnd as a robust empirical pattern and the strongest family overall, but not as a universally statistically significant improvement on TREC DL's 43/54-query test sets

## [2026-04-07] DualEnd-Cocktail is the strongest single method across most models (DL19)

- DualEnd-Cocktail achieves the best or near-best NDCG@10 on 7 of 9 models tested on DL19
- Gains over TopDown-Heapsort range from +0.3 (qwen3.5-27b) to +3.4 (qwen3-8b)
- Key results (NDCG@10): flan-t5-xxl .7137 (+2.9 vs TD-Heap), qwen3-8b .7155 (+3.4), qwen3-14b .7519 (+0.7), qwen3.5-9b .7370 (+0.4)
- Exception: qwen3-4b favors DualEnd-Selection (.7220 vs DE-Cocktail .6796); flan-t5-large and flan-t5-xl favor TopDown-Bubblesort (.6874 and .6980 respectively)
- Implication: DualEnd's ability to extract both best and worst in a single LLM call provides a richer signal per comparison, especially for larger models

## [2026-04-07] TopDown-Bubblesort competitive on small T5 models

- On flan-t5-large: TD-Bubble .6874 > DE-Cocktail .6708 > TD-Heap .6541
- On flan-t5-xl: TD-Bubble .6980 > TD-Heap .6901 > DE-Cocktail .6884
- Hypothesis: T5 models with limited context (512 tokens) benefit more from bubblesort's pairwise-adjacent comparison pattern than from DualEnd's more complex dual-output prompt
- Implication: method recommendations should be model-family-aware

## [2026-04-07] BottomUp is unreliable — catastrophic on small models, competitive only on large ones

- flan-t5-large BU-Heap: .2888 (catastrophic, -36.5 vs TD-Heap .6541) — model cannot reliably identify "least relevant"
- flan-t5-xl BU-Heap: .6630 (-2.7 vs TD-Heap)
- flan-t5-xxl BU-Heap: .6874 (+0.3 vs TD-Heap .6846) — competitive at this scale
- qwen3-14b BU-Heap: .6966 (-4.8 vs TD-Heap .7447)
- qwen3.5-27b BU-Heap: .7135 (-3.1 vs TD-Heap .7449); BU-Bubble .7336 is more competitive
- Implication: "worst" prompt is a fundamentally harder task for LLMs. Only the largest models can handle it reliably. Claims about BottomUp must be carefully scoped.

## [2026-04-07] Bidirectional ensemble never beats TopDown

- All BiDir variants (RRF, weighted) produce NDCG@10 scores below TopDown-Heapsort across all models tested on DL19
- Root cause: BottomUp rankings are too noisy (especially on smaller models), so fusing them with TopDown degrades rather than improves
- Decision: BiDir is not a viable strategy for the paper's main claims; report as negative result

## [2026-04-07] Position bias analysis reveals systematic patterns across ALL 9 models (DL19+DL20)

Full position bias analysis completed for all 9 models × 2 datasets (18 result files in `results/analysis/position_bias/*/position_bias_results.txt`).

**Consistent patterns across models:**

- **"best" selection (TopDown)**: U-shaped bias — positions A and D overselected, B and C underselected. Most models: A ~.30-.50, B ~.11-.16, C ~.10-.18, D ~.25-.55. Strongest recency bias on T5-XL (.533 pos D) and Q3.5-4B (.547 pos D). Strongest primacy on Qwen3-4B (.495 pos A).

- **"worst" selection (BottomUp)**: Strong recency bias — position D massively overselected as "worst" across ALL models. D freq ~.40-.63. Positions B and C severely underselected (~.06-.17). This is consistent: models tend to label the LAST passage as worst regardless of content.

- **"dual_best" (DualEnd best output)**: More uniform than single "best", but still shows recency bias for D (.27-.48). Key finding: dual prompting REDUCES position bias compared to single best-selection. Chi-squared values are consistently lower for dual_best than for best.

- **"dual_worst" (DualEnd worst output)**: REVERSED bias pattern — position A (first passage) is most frequently selected as worst (.23-.43), while D (last) is least selected (.10-.19). This is the opposite of the "worst" (BottomUp) pattern where D dominates. The dual prompt context changes which position gets flagged as worst.

**Cross-model summary (DL19, position D frequency):**

| Model | best→D | worst→D | dual_best→D | dual_worst→D |
|-------|--------|---------|-------------|--------------|
| T5-Large | .387 | .473 | .387 | .141 |
| T5-XL | .533 | .631 | .475 | .134 |
| T5-XXL | .361 | .595 | .350 | .145 |
| Qwen3-4B | .247 | .365 | .153 | .180 |
| Qwen3-8B | .338 | .454 | .269 | .112 |
| Qwen3-14B | .411 | .445 | .331 | .124 |
| Q3.5-4B | .547 | .598 | .273 | .189 |
| Q3.5-9B | .368 | .463 | .366 | .101 |
| Q3.5-27B | .431 | .462 | .379 | .128 |

**Key insight for paper**: The dual_worst pattern (primacy bias for worst-in-dual vs recency bias for worst-alone) is a genuinely novel finding. When asked for both best AND worst simultaneously, models shift their "worst" selection toward the FIRST passage rather than the last. This suggests the dual prompt changes the cognitive process — possibly because the model reads for "best" first (recency helps) then assigns "worst" to whatever is leftmost/most salient from the beginning.

## [2026-04-07] Ablation num_child (nc): optimal window size is model-family-dependent

- DualEnd-Cocktail on DL19:
  - flan-t5-xl: nc2=.6988 > nc3=.6884 (default) > nc5=.6749 > nc7=.6480 — smaller window better for T5
  - qwen3-8b: nc7=.7249 > nc5=.7224 > nc2=.7187 — larger window better for Qwen
  - qwen3.5-9b: nc2=.7392 > nc7=.7386 > nc5=.7336 — stable across window sizes
- Implication: T5's 512-token context limit makes larger windows counterproductive (more truncation). Qwen models with 32k+ context benefit from seeing more candidates.

## [2026-04-07] Ablation alpha (BiDir weighted): topdown-heavy weighting consistently best

- Across all models on DL19: alpha=0.9 (90% topdown, 10% bottomup) performs best
- alpha=0.3 (favor bottomup) consistently worst
- Confirms that BottomUp rankings are too noisy to contribute meaningfully, even in ensemble

## [2026-04-07] Ablation passage_length: plateau behavior differs by model family

- T5 models: pl64 < pl128 ~ pl256 ~ pl512 — plateau at 128 tokens (T5 truncates at 512 context anyway)
- Qwen models: pl64 < pl128 < pl256 ~ pl512 — plateau at 256 tokens (longer context can be utilized)
- Implication: default passage_length=128 for T5 and passage_length=512 for Qwen are appropriate

## [2026-04-07] Per-query wins analysis: DualEnd wins more queries than TopDown on capable models

- Flan-T5-XL DL19: TD wins 16, DE wins 20, Ties 7 (DE wins more queries but similar mean NDCG)
- Qwen3-14B DL19: TD wins 15, DE wins 19, Ties 9 (DE wins more queries AND higher mean: .7519 vs .7447)
- TD vs BU consistently: TD wins 24-25 queries, BU wins 9-12. BottomUp rarely wins.
- DualEnd vs BiDir-RRF (Qwen3-14B): DE wins 23, BiDir wins 16 — DualEnd dominates at lower cost
- BiDir-RRF vs best individual: fusion "beats best" on only 8-11 queries, "worse than worst" on 10-11. Net negative.
- Key quote: "DualEnd uses ~1x cost, BiDir uses 2x cost" — DualEnd is strictly better than fusion

## [2026-04-07] Query difficulty stratification: DualEnd helps most on medium-difficulty queries

- Flan-T5-XL DL19: Easy DE-TD=+.0143, Medium DE-TD=-.0103, Hard DE-TD=-.0101 (DE helps on easy)
- Qwen3-14B DL19: Easy DE-TD=-.0058, Medium DE-TD=+.0189, Hard DE-TD=+.0091 (DE helps on medium+hard)
- BottomUp consistently hurts on medium-difficulty queries (BU-TD = -.08 to -.13 for medium tercile)
- Implication: DualEnd's advantage is not query-difficulty-dependent in a consistent direction. The benefit varies by model.

## [2026-04-07] Ranking agreement: DualEnd rankings are much closer to TopDown than BottomUp

- Consistent across all models (DL19):
  - TopDown vs DualEnd: Overlap@10 ≈ 7.3, Kendall tau ≈ 0.93 (high agreement)
  - TopDown vs BottomUp: Overlap@10 ≈ 5.5-5.9, Kendall tau ≈ 0.87-0.88 (moderate-high)
  - BottomUp vs DualEnd: Overlap@10 ≈ 5.9, Kendall tau ≈ 0.88-0.89
- Implication: DualEnd produces rankings most similar to TopDown (the established baseline), suggesting DualEnd's "best" selection dominates the ranking while "worst" selection provides refinement. BottomUp diverges more, explaining why fusion with TopDown doesn't help — the signals aren't complementary, they're just noisier.

## [2026-04-07] DualEnd parse success: near-perfect for T5, minor issues for Qwen3-14B DL20

- All T5 models: 0 parse failures, 0 unexpected outputs across all datasets. Likelihood scoring avoids parsing entirely.
- Qwen3/3.5 small models: near-perfect. Qwen3-4B has 0-2 "only parse one" warnings out of ~33K comparisons.
- Qwen3-14B DL20 has the most issues: 254 "only parse one" + 218 "partial dual parse" for cocktail (out of ~29K comparisons = 0.9%). DL19 is much cleaner (57 + 56).
- Qwen3.5-27B: very clean, 13 "only parse one" out of ~23K comparisons on DL19.
- No model has actual "dual parse failures" — the cascading parser always produces a result.
- Implication: parsing reliability is not a concern for the paper's claims. The <1% partial parse rate is well within noise.

## [2026-04-07] Efficiency trade-offs: DualEnd-Cocktail costs 4-7x more than TopDown-Heapsort

- TopDown-Heapsort: ~77 comparisons, ~32K tokens, ~7s wall time
- DualEnd-Cocktail: ~546 comparisons, ~224K tokens, ~30s (T5) to ~630s (Qwen3.5-27B)
- TopDown-Bubblesort: ~314 comparisons, ~129K tokens
- BottomUp-Heapsort: ~273-325 comparisons, ~111-133K tokens
- BottomUp-Bubblesort: ~1683 comparisons, ~691K tokens (most expensive by far)
- Implication: DualEnd-Cocktail's quality gains (+0.3 to +3.4 NDCG@10) come at significant cost. Paper should frame this as a quality-efficiency trade-off.

---

# Engineering Findings

> Infrastructure, environment, and debugging lessons. Prevents re-debugging the same issues in future sessions.

## [2026-04-09] New DualEnd refinement variants expose their own cost counters for paper-facing audits

- Added three refinement variants in [llmrankers/setwise_extended.py](/Users/hangli/projects/llm-rankers/llmrankers/setwise_extended.py):
  - `SelectiveDualEndSetwiseLlmRanker`
  - `BiasAwareDualEndSetwiseLlmRanker`
  - `SameCallRegularizedSetwiseLlmRanker`
- Added CLI wiring in [run.py](/Users/hangli/projects/llm-rankers/run.py) and log extraction in [experiments/eval_all.sh](/Users/hangli/projects/llm-rankers/experiments/eval_all.sh)
- New counters now appear in logs and `results.txt`:
  - `Avg dual invocations`
  - `Avg single invocations`
  - `Avg order-robust windows`
  - `Avg extra orderings`
  - `Avg regularized worst moves`
- Implication: future runs can be audited as **method behavior**, not just as black-box NDCG/time numbers

## [2026-04-07] BottomUp bubblesort had incorrect top-k sorting

- Problem: bubblesort only performed n-k passes, leaving the top-k results unsorted
- Root cause: the loop termination condition stopped too early — needed n-1 passes to fully sort the top-k portion
- Fix: corrected the pass count to ensure full sorting of the top-k results

## [2026-04-07] BottomUp heapsort mixed "best" and "worst" prompts

- Problem: `_sort_top_k` in BottomUp heapsort was calling the parent class's compare method, which uses "most relevant" prompts instead of "least relevant"
- Root cause: `_sort_top_k` was inherited from the base `SetwiseLlmRanker` and not overridden
- Fix: ensured BottomUp heapsort uses pure "worst" prompts throughout via `compare_worst()` at every comparison stage

## [2026-04-07] compare_worst did not log comparisons

- Problem: comparison logging was missing from the `compare_worst` method, so BottomUp comparison logs were empty
- Fix: added comparison logging to `compare_worst` to match the behavior of `compare` (TopDown)

## [2026-04-07] T5 DualEnd uses likelihood scoring internally

- Problem: DualEnd T5 cocktail reports 0 completion tokens, which looks like a bug
- Root cause: T5 cannot reliably generate "Best: X, Worst: Y" format, so DualEnd T5 uses likelihood-based scoring internally even when `--scoring generation` is set
- Decision: this is correct behavior, not a bug. Documented in CLAUDE.md as a known design decision.

## [2026-04-10] Qwen/Qwen3.5 likelihood now uses teacher-forced short-answer scoring

- Problem: causal label tokens such as `A/B/C/D` are not guaranteed to be stable single tokens across Qwen tokenizers, so a direct next-token-logit implementation would be brittle
- Fix: causal likelihood now scores short continuations like `Passage A` with teacher forcing, which supports TopDown, BottomUp, and heuristic DualEnd while keeping completion tokens at 0
- Scope: causal DualEnd likelihood follows the same heuristic as T5 --- score the best-only label distribution once, then reuse `argmax` as best and `argmin` as worst

## [2026-04-10] Selective / bias-aware gating now uses query-local spread percentiles

- Problem: a fixed absolute BM25 spread cutoff like `0.15` is too dataset- and query-dependent; on tightly clustered rankings it can trigger DualEnd almost everywhere, while on broader score ranges it can be too strict
- Fix: `uncertain` and `hybrid` gating now compare each window's BM25 spread against a query-local percentile threshold computed from sliding windows in the original ranking
- Scope: both `SelectiveDualEndSetwiseLlmRanker` and `BiasAwareDualEndSetwiseLlmRanker` share this routing logic, while the old `--margin_threshold` flag remains as a backward-compatible alias for the percentile value

## [2026-04-10] Joint-signal refinements now document and enforce their real operating regime

- Problem: code audit showed three mismatches between the method descriptions and the actual implementations: (1) likelihood-mode joint prompting is still a best-only proxy, not exact joint scoring; (2) `bias_aware_dualend --method heapsort` silently bypassed the order-robust joint prompt path; (3) same-call regularization could apply an extra worst demotion even when the candidate was still inside the active ranking head
- Fix: documented the likelihood caveat explicitly, rejected `heapsort` for bias-aware DualEnd, disabled shortlist gating for Selective heapsort, and limited the extra same-call worst demotion to candidates already outside the protected ranking head frontier (top-`k` plus one active window)
- Scope: keeps the implemented refinement package aligned with what the paper can honestly claim and removes the most misleading configuration from the CLI surface

## [2026-04-07] Qwen3 thinking models need increased max_new_tokens

- Problem: Qwen3 models emit `<think>...</think>` blocks even with `enable_thinking=False`, consuming output tokens before the actual answer
- Fix: set `max_new_tokens=256` (512 for DualEnd dual-output prompts) to accommodate the thinking overhead
- Cleanup: added `_clean_generation_output()` to strip `<think>...</think>` tags from generated text

## [2026-04-07] Qwen3.5 model loading requires AutoModelForCausalLM

- Problem: loading Qwen3.5 via `Qwen3_5ForConditionalGeneration` fails or produces incorrect results
- Root cause: Qwen3.5 architecture is not a conditional generation model; it must be loaded as a causal LM
- Fix: use `AutoModelForCausalLM` for all Qwen3.5 variants. Requires transformers dev build.

## [2026-04-07] Completion token counting was wrong for causal models

- Problem: token counts for causal (decoder-only) models included input tokens, inflating reported completion token usage
- Root cause: causal model output includes the full sequence (input + generated), but the counting logic treated the entire output as completion tokens
- Fix: subtracted input token count from total output length to get accurate completion token counts
