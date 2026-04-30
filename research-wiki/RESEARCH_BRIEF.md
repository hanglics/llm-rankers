# Research Brief

> **Template for document-based input to `/idea-discovery` or `/research-pipeline`.** Provide detailed context instead of a one-line prompt.

## Problem Statement

LLM-based setwise ranking (Zhuang et al., SIGIR 2024) presents a small set of candidate documents to an LLM and asks "which is the most relevant to the query?" The LLM internally evaluates all candidates in the set, reasoning about relative relevance, yet the method extracts only a single ranking decision per comparison: the winner. All implicit knowledge about which document is worst, and how the remaining candidates relate to each other, is discarded. This is informationally wasteful.

The inefficiency compounds at the sorting algorithm level. Heapsort requires O(n log n) comparisons, each extracting one bit of ordering information. Bubblesort performs O(n^2) pairwise-adjacent swaps, again extracting one decision per call. If we could extract two decisions per LLM call (e.g., both best and worst), we could power richer sorting algorithms (cocktail shaker sort, double-ended selection sort) that place two documents per comparison. More ambitiously, if "worst" selection provides a complementary ranking signal, fusing top-down and bottom-up rankings could improve robustness.

The core research question is: **Can we improve setwise LLM ranking by extracting more information from each comparison?** We explore four strategy families beyond the standard top-down approach:

1. **BottomUp** (select worst, idea:001).
2. **DualEnd** (select best and worst jointly in the same call, idea:002).
3. **Bidirectional ensemble** (fuse independent top-down and bottom-up rankings, idea:003).
4. **MaxContext family** (idea:007 — fit the entire rerank pool, up to 50 docs, into a single Qwen prompt, with DualEnd / TopDown / BottomUp variants). Active and Codex-audited; tests not yet launched.

Our findings show that the **DualEnd family** is strongest overall, but bottom-up selection alone is unreliable, bidirectional fusion degrades performance, and most DualEnd gains are modest on TREC DL's small query sets. Three further refinements are implemented and pending evaluation: **Selective DualEnd** (idea:004), **Bias-aware DualEnd** (idea:005), and **Same-call regularized** (idea:006). The MaxContext family is the freshest direction: it trades many small joint-elicitation calls for very few large-window calls and is designed to land in the empty region between `TD-Bubble` and `DE-Cocktail` on the comparisons-axis and wall-clock-axis frontiers.

## Background

- **Field**: Information Retrieval
- **Sub-area**: LLM-based document reranking / zero-shot neural ranking
- **Key papers I've read**:
  - Zhuang et al. (SIGIR 2024) — Setwise prompting for LLM-based document ranking. Introduces the setwise paradigm with heapsort and bubblesort over "most relevant" prompts.
  - Sun et al. (2023) — RankGPT. Listwise ranking with sliding window, demonstrating LLMs can perform zero-shot passage ranking.
  - Qin et al. (2024) — PRP (Pairwise Ranking Prompting). Systematic study of pairwise LLM ranking with analysis of positional bias.
  - Tang et al. (2024) — Found in the Middle. Documents position bias in LLM-based ranking: models over-attend to first and last positions.
- **What I already tried**: Strategies extending setwise ranking:
  - **BottomUp** (idea:001): Reverse the prompt to ask "which is LEAST relevant?" and build rankings from the bottom up. Implemented with heapsort and bubblesort.
  - **DualEnd** (idea:002): Ask "which is MOST relevant AND which is LEAST relevant?" in a single prompt. Implemented with cocktail shaker sort (bubblesort variant) and double-ended selection sort.
  - **Bidirectional Ensemble** (idea:003): Run TopDown and BottomUp independently, then fuse rankings via Reciprocal Rank Fusion (RRF), CombSUM, or weighted combination.
  - **Selective DualEnd** (idea:004): TopDown sort that upgrades only routed windows to same-call best-worst prompting. Routes by shortlist overlap, query-local BM25 spread percentile, or hybrid union. Implemented; partial results on flan-t5-xl, Qwen runs pending.
  - **Bias-aware DualEnd** (idea:005): Run a tiny set of controlled orderings only on hard windows, then majority-vote labels back into the original order. Designed to exploit the dual_worst primacy reversal documented in claim:C5. Implemented; all 12 runs pending.
  - **Same-call regularized** (idea:006): Use the joint-prompt's worst output only as a local demotion signal once a candidate is already outside the protected ranking head. Implemented; all 12 runs pending.
  - **MaxContext family** (idea:007): Fit the entire rerank pool (`pool_size ≤ 50`) into a single Qwen prompt. Three variants — DualEnd shrinks the pool by 2 each round; TopDown / BottomUp shrink by 1 plus a deterministic BM25 endgame at `n_docs=2`. Codex-audited 3 rounds, ready to execute, 312-run staged matrix not yet launched.
- **What didn't work**:
  - BottomUp is consistently weaker than TopDown. On small models (flan-t5-large), it is catastrophic (NDCG@10 drops from .654 to .289). Even on the largest models, it trails TopDown by 3-5 points. "Worst" selection appears to be a fundamentally harder cognitive task for LLMs.
  - Bidirectional ensemble never beats TopDown. Because BottomUp rankings are too noisy, fusing them with TopDown degrades rather than improves quality. Even with alpha=0.9 (90% TopDown weight), the ensemble underperforms pure TopDown.

## Constraints

- **Compute**: Single NVIDIA H100 80GB GPU (rented via Vast.ai)
- **Timeline**: Phase 1–4 experiments complete (144 main runs + ablations + analysis). Pairwise same-sort tables completed 2026-04-21. MaxContext family (idea:007) staged 312-run matrix not yet launched. Refinement variants (idea:004/005/006) partially or fully pending. Paper revision in progress.
- **Target venue**: ICTIR (primary); later ARR only after a stronger refinement/generalization package
- **Framing constraint** (claim:C10, locked 2026-04-20): the paper is an analysis-driven IR contribution, not a new-state-of-the-art claim. Lead with directional asymmetry and joint elicitation as the contribution. Do not claim universal DualEnd improvement; do not claim DualEnd is more efficient than `TD-Heap`; disclose that T5 / `--scoring likelihood` DualEnd paths are a best-only proxy.

## What I'm Looking For

- [x] Improvement on existing method: setwise LLM ranking (dual-end extraction)
- [x] Diagnostic study / analysis paper: position bias asymmetry, worst-selection difficulty, model-family effects
- [ ] New research direction from scratch
- [ ] Other

## Domain Knowledge

- **LLMs exhibit position bias** in ranking prompts. Documents placed first (primacy) and last (recency) in the candidate set are disproportionately selected. Middle positions are severely underselected (as low as 11-12% when uniform would be 25%).
- **"Worst" selection is harder than "best" selection** for LLMs. This is consistent across model families and scales. Possible explanation: LLMs are predominantly trained on tasks that reward identifying correct/relevant/best answers, not identifying the worst option.
- **Dual-end selection partially mitigates position bias**. When asked for both best and worst simultaneously, the "best" selection shows a more uniform distribution across positions. The "worst" selection in dual-end shows a reversed bias pattern (primacy-heavy rather than recency-heavy), which is a novel finding.
- **Cocktail shaker sort is algorithmically suited for dual-output comparators**. It processes the candidate list in alternating forward/backward passes, naturally accommodating a comparator that returns both a maximum and minimum per window.
- **T5 models can use likelihood scoring internally for DualEnd**. Rather than generating "Best: X, Worst: Y" text (which T5 does unreliably), the implementation scores each candidate's likelihood of being best/worst. This means completion tokens = 0 for DualEnd T5 cocktail, which is correct behavior, not a bug.
- **Optimal window size (num_child) is model-family-dependent**. T5 models with 512-token context limits perform best with smaller windows (nc=2-3). Qwen models with 32k+ context benefit from larger windows (nc=5-7) where more candidates can be compared simultaneously.
- **Efficiency trade-off is real**: DualEnd-Cocktail uses 4-7x more comparisons and tokens than TopDown-Heapsort, and the DualEnd family is 5.6x-8.9x slower than `TD-Heap` on average. The quality gains must be framed as a quality-efficiency trade-off, not a free improvement.
- **Significance is asymmetric**: DualEnd is positive in 14/18 model-dataset configs, but only `qwen3-4b` DL19 survives Bonferroni-corrected significance. By contrast, BottomUp shows 6 corrected losses and BiDir 3 corrected losses. This supports a conservative paper framing.
- **Joint elicitation is the load-bearing contribution** (claim:C8). Three facts jointly identify it: BottomUp alone fails (claim:C1, C3); BiDir fails because BU is biased (claim:C4); DualEnd partially succeeds because worst is co-elicited with best (claim:C2, C5). The cocktail-shaker and double-ended selection sorts are *consumers* of the dual output — necessary plumbing but not the scientific contribution.
- **The Pareto frontier is narrow** (claim:C9). Global mean frontier members are `TD-Heap`, `TD-Bubble`, and `DE-Cocktail`. The empty region between `TD-Bubble` and `DE-Cocktail` (+82% comparisons / +92% wall-clock for +0.0065 NDCG) is the natural target for selective / bias-aware refinements (idea:004/005/006) and for the MaxContext family (idea:007).
- **Pairwise same-sort tables (closed 2026-04-21)** show the cleanest positive finding is DualEnd vs `TD-Bubble` on DL19: 2 Bonferroni-significant DualEnd wins on Qwen3-8B (`DE-Cocktail` and `DE-Selection`). Authoritative numbers in `SIGNIFICANCE_TESTS_PAIRWISE.md`.

## Non-Goals

- We do **NOT** propose a new model architecture. All methods use off-the-shelf LLMs (Flan-T5, Qwen3, Qwen3.5) in zero-shot mode.
- We do **NOT** fine-tune any models. The entire study is zero-shot prompting only.
- We do **NOT** claim efficiency improvements in raw LLM call count. DualEnd uses MORE calls than TopDown-Heapsort. The contribution is in ranking quality per comparison, not fewer comparisons.
- We do **NOT** address listwise or pairwise ranking paradigms. The scope is strictly setwise.

## Existing Results (if any)

Complete Phase 1-4 experiments: 9 models x 2 datasets (TREC DL19, DL20) x 8 methods = 144 main runs, plus ablations (num_child, alpha, passage_length) and analysis (position bias, query difficulty, ranking agreement, per-query wins).

**Representative DL19 results (NDCG@10):**

| Method family / best variant | flan-t5-large | flan-t5-xl | flan-t5-xxl | qwen3-8b | qwen3-14b | qwen3.5-27b |
|---|---|---|---|---|---|---|
| TopDown best | .6874 (`TD-Bubble`) | .6980 (`TD-Bubble`) | .7077 (`TD-Bubble`) | .6819 (`TD-Heap`) | .7455 (`TD-Bubble`) | .7449 (`TD-Heap`) |
| BottomUp best | .4571 (`BU-Bubble`) | .6730 (`BU-Bubble`) | .6936 (`BU-Bubble`) | .6431 (`BU-Heap`) | .6966 (`BU-Heap`) | .7336 (`BU-Bubble`) |
| DualEnd best | .6708 (`DE-Cocktail`) | .6884 (`DE-Cocktail`) | .7137 (`DE-Cocktail`) | .7158 (`DE-Selection`) | .7519 (`DE-Cocktail`) | .7475 (`DE-Cocktail`) |
| Best BiDir | .6147 (`Wt`) | .6845 (`RRF`) | .6905 (`RRF`) | .6826 (`RRF`) | .7200 (`Wt`) | .7229 (`Wt`) |

**Summary of findings:**
- The DualEnd family wins 14/18 model-dataset configurations overall and all 12 Qwen configurations; `DE-Cocktail` is the strongest single variant with 11 wins
- TopDown-Bubblesort remains strongest on 4/6 T5 configurations, so the best recommendation is model-family-dependent
- BottomUp is catastrophic on smaller models and significantly harmful in 6 configurations after Bonferroni correction
- Bidirectional ensemble is usually worse than TopDown and has 3 Bonferroni-significant losses
- DualEnd partially mitigates position bias, with a reversed worst-selection bias pattern under dual prompts
- Quality gains come at substantial extra compute cost and only one DualEnd gain is Bonferroni-significant on the current TREC DL test sets
