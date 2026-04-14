# Research Brief

> **Template for document-based input to `/idea-discovery` or `/research-pipeline`.** Provide detailed context instead of a one-line prompt.

## Problem Statement

LLM-based setwise ranking (Zhuang et al., SIGIR 2024) presents a small set of candidate documents to an LLM and asks "which is the most relevant to the query?" The LLM internally evaluates all candidates in the set, reasoning about relative relevance, yet the method extracts only a single ranking decision per comparison: the winner. All implicit knowledge about which document is worst, and how the remaining candidates relate to each other, is discarded. This is informationally wasteful.

The inefficiency compounds at the sorting algorithm level. Heapsort requires O(n log n) comparisons, each extracting one bit of ordering information. Bubblesort performs O(n^2) pairwise-adjacent swaps, again extracting one decision per call. If we could extract two decisions per LLM call (e.g., both best and worst), we could power richer sorting algorithms (cocktail shaker sort, double-ended selection sort) that place two documents per comparison. More ambitiously, if "worst" selection provides a complementary ranking signal, fusing top-down and bottom-up rankings could improve robustness.

The core research question is: **Can we improve setwise LLM ranking by extracting more information from each comparison?** We explore three strategies beyond the standard top-down approach: bottom-up (select worst), dual-end (select both best and worst simultaneously), and bidirectional ensemble (fuse independent top-down and bottom-up rankings). Our findings now show that the **DualEnd family** is strongest overall, but bottom-up selection alone is unreliable, bidirectional fusion degrades performance, and most DualEnd gains are modest on TREC DL's small query sets.

## Background

- **Field**: Information Retrieval
- **Sub-area**: LLM-based document reranking / zero-shot neural ranking
- **Key papers I've read**:
  - Zhuang et al. (SIGIR 2024) — Setwise prompting for LLM-based document ranking. Introduces the setwise paradigm with heapsort and bubblesort over "most relevant" prompts.
  - Sun et al. (2023) — RankGPT. Listwise ranking with sliding window, demonstrating LLMs can perform zero-shot passage ranking.
  - Qin et al. (2024) — PRP (Pairwise Ranking Prompting). Systematic study of pairwise LLM ranking with analysis of positional bias.
  - Tang et al. (2024) — Found in the Middle. Documents position bias in LLM-based ranking: models over-attend to first and last positions.
- **What I already tried**: Three novel strategies extending setwise ranking:
  - **BottomUp**: Reverse the prompt to ask "which is LEAST relevant?" and build rankings from the bottom up. Implemented with heapsort and bubblesort.
  - **DualEnd**: Ask "which is MOST relevant AND which is LEAST relevant?" in a single prompt. Implemented with cocktail shaker sort (bubblesort variant) and double-ended selection sort.
  - **Bidirectional Ensemble**: Run TopDown and BottomUp independently, then fuse rankings via Reciprocal Rank Fusion (RRF), CombSUM, or weighted combination.
- **What didn't work**:
  - BottomUp is consistently weaker than TopDown. On small models (flan-t5-large), it is catastrophic (NDCG@10 drops from .654 to .289). Even on the largest models, it trails TopDown by 3-5 points. "Worst" selection appears to be a fundamentally harder cognitive task for LLMs.
  - Bidirectional ensemble never beats TopDown. Because BottomUp rankings are too noisy, fusing them with TopDown degrades rather than improves quality. Even with alpha=0.9 (90% TopDown weight), the ensemble underperforms pure TopDown.

## Constraints

- **Compute**: Single NVIDIA H100 80GB GPU (rented via Vast.ai)
- **Timeline**: All experiments complete (144 main runs + ablations + analysis). Paper revision in progress.
- **Target venue**: ICTIR (primary); later ARR only after a stronger refinement/generalization package

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
