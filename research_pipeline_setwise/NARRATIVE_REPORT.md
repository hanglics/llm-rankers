# Narrative Report: Finding the Worst Is Harder Than Finding the Best: Directional Asymmetry in LLM Setwise Ranking

> **Template for Workflow 3 (`/paper-writing`).** Fill in all sections, then run `/paper-writing "NARRATIVE_REPORT.md"`.

## Core Story

LLM-based setwise ranking (Zhuang et al., SIGIR 2024) presents documents in sets and asks the LLM to select the "most relevant" one, then applies sorting algorithms (heapsort, bubblesort) to produce a top-k ranking. A fundamental limitation of this approach is that each LLM comparison extracts only one ranking decision --- the identity of the best document --- while discarding all other information the model implicitly evaluates. This is wasteful: the LLM reads and reasons about every document in the set, yet only one bit of ordinal information is retained per call.

We study three strategies that extract more ranking information per comparison. (1) **BottomUp** reverses the prompt to select the "least relevant" document, building the ranking from the bottom up. This tests whether LLMs are better at identifying irrelevant documents, a hypothesis motivated by the asymmetry between relevance and irrelevance judgments. (2) **DualEnd** asks the LLM to identify both the best and worst documents in a single call, extracting approximately 79% more ranking information per comparison (information-theoretic analysis: log2(w*(w-1)) vs. log2(w) bits for window size w). We pair DualEnd with cocktail shaker sort and double-ended selection sort, both of which naturally exploit simultaneous best-and-worst identification. (3) **Bidirectional ensemble** runs TopDown and BottomUp independently, then fuses rankings via RRF or weighted CombSUM.

Experiments across 9 LLMs spanning three model families (Flan-T5 780M--11B, Qwen3 4B--14B, Qwen3.5 4B--27B) on TREC DL 2019 and 2020 yield a clearer story than the original draft: **the DualEnd family is strongest overall, but its gains are modest on these small test sets**. DualEnd wins 14/18 model-dataset configurations overall and all 12 Qwen configurations; `dualend_bubblesort` is the strongest single variant with 11 wins, while `dualend_selection` is best on the smallest Qwen settings. TopDown-Bubblesort remains strongest on 4/6 T5 configurations. BottomUp is consistently weaker than TopDown, and Bidirectional fusion usually hurts because it imports BottomUp noise. Paired approximate-randomization tests over per-query NDCG@10, with Bonferroni correction across the 18 configs per family, find only one corrected significant DualEnd win (`qwen3-4b` DL19), 6 corrected BottomUp losses, and 3 corrected BiDir losses. The paper should therefore frame DualEnd as a robust empirical pattern and quality-first option, not as a universally significant or compute-free improvement.

The strongest paper framing is now **directional asymmetry in setwise LLM ranking**, not "three equally successful bidirectional strategies." Worst-selection alone is unreliable and heavily biased, but worst-selection inside a joint best-and-worst prompt changes the model's behavior and can become modestly useful. This makes the project strongest as an analysis-driven IR paper with one promising quality-first method and two coherent negative results.

## Paper Positioning

- **Primary venue**: ICTIR
- **Stretch venue after refinement/generalization**: later ARR
- **Core novelty**: worst-selection behaves differently when asked alone versus when asked jointly with best-selection
- **Central positive result**: DualEnd is the strongest family overall, especially on Qwen
- **Central negative results**: BottomUp is unreliable; BiDir fails because it imports BottomUp noise
- **Central caution**: DualEnd is expensive and statistically fragile on TREC DL's small query sets

## Claims

1. **Setwise LLM ranking is directionally asymmetric**: selecting the worst document is not a symmetric mirror of selecting the best. Worst-selection alone is substantially noisier and more position-biased, but asking for best and worst jointly changes the behavior of the model.

2. **The DualEnd family achieves the highest NDCG@10 in the majority of configurations**: DualEnd wins 14/18 model-dataset configurations overall and all 12 Qwen configurations. `dualend_bubblesort` is the strongest single variant with 11 wins, while `dualend_selection` wins the two `qwen3-4b` settings and `qwen3-8b` DL19.

3. **BottomUp is consistently weaker than TopDown**: The best BottomUp variant is below the best TopDown baseline in all 18 configurations, with especially large failures on smaller models (for example, flan-t5-large DL19: `.4571` BU-Bubble vs `.6874` TD-Bubble).

4. **Bidirectional ensemble does not improve on TopDown**: The best BiDir variant exceeds the best TopDown baseline in only 3/18 configurations, all by very small margins (`+0.0008` to `+0.0068`), while averaging `-0.0232` overall.

5. **DualEnd extracts ~79% more information per comparison, but not less total compute**: For the default window size `w=4`, each DualEnd comparison yields `log2(12)=3.58` bits vs `log2(4)=2.0` for single-direction methods. However, end-to-end DualEnd remains more expensive than `TD-Heap` in total time and comparisons.

6. **Position bias patterns differ sharply across directions**: TopDown exhibits a U-shaped best-selection bias, BottomUp shows strong recency bias for worst-selection, and DualEnd flips worst-selection toward primacy while making best-selection more uniform.

7. **Statistical evidence is asymmetric**: DualEnd is directionally positive in 14/18 configs but only one gain survives Bonferroni correction, whereas BottomUp produces 6 corrected losses and BiDir 3. The paper should therefore make conservative significance claims.

8. **The current quality-cost frontier is narrow and interpretable**: at the global mean level, `TD-Heap`, `TD-Bubble`, and `DE-Cocktail` span the main comparison/token frontier, so the correct next method is a selective or bias-aware DualEnd that tries to land between `TD-Bubble` and full `DE-Cocktail`.

## Experiments

### Setup

- **Models**: 9 LLMs across 3 families:
  - Flan-T5: large (780M), xl (3B), xxl (11B) --- encoder-decoder, likelihood scoring
  - Qwen3: 4B, 8B, 14B --- decoder-only, generation for the main tables plus sequence-likelihood follow-ups, thinking disabled in generation mode
  - Qwen3.5: 4B, 9B, 27B --- decoder-only, generation for the main tables plus sequence-likelihood follow-ups
- **Data**: TREC Deep Learning Track 2019 (43 queries) and 2020 (54 queries), re-ranking BM25 top-100
- **Hardware**: NVIDIA A100/H100 GPUs via Vast.ai; single-GPU inference for all models
- **Baselines**: TopDown-Heapsort and TopDown-Bubblesort (original setwise ranker)
- **Hyperparameters**: num_child=3 (window of 4 passages), k=10, passage_length=128 (T5) / 512 (Qwen)

### Experiment 1: Main Ranking Quality (NDCG@10)

Comparison of all 8 methods across 9 models on DL19 and DL20.

**DL19 Results (NDCG@10):**

| Method | T5-large | T5-xl | T5-xxl | Q3-4B | Q3-8B | Q3-14B | Q3.5-4B | Q3.5-9B | Q3.5-27B |
|--------|----------|-------|--------|-------|-------|--------|---------|---------|----------|
| TD-Heap | .6541 | .6901 | .6846 | .6775 | .6819 | **.7447** | .7087 | .7329 | .7449 |
| TD-Bubble | **.6874** | **.6980** | .7077 | .6491 | .6794 | .7455 | .7108 | .7349 | .7435 |
| BU-Heap | .2888 | .6630 | .6874 | .6261 | .6431 | .6966 | .6158 | .6779 | .7135 |
| BU-Bubble | .4571 | .6730 | .6936 | .6305 | .6273 | .6702 | .6120 | .6712 | .7336 |
| DE-Cocktail | .6708 | .6884 | **.7137** | .6796 | **.7155** | **.7519** | **.7161** | **.7370** | **.7475** |
| DE-Selection | .6420 | .6792 | .6974 | **.7220** | .7158 | .7475 | .7022 | .7309 | .7319 |
| BiDir-RRF | .5820 | .6845 | .6905 | .6814 | .6826 | .7172 | .6614 | .7101 | .7198 |
| BiDir-Wt | .6147 | .6810 | .6734 | .6608 | .6784 | .7200 | .6714 | .7087 | .7229 |

**Interpretation**: The main result is better expressed at the family level than at the single-method level. DualEnd wins 14/18 overall, all 12 Qwen configs, and 2/6 T5 configs; `dualend_bubblesort` is the default best variant, while `dualend_selection` matters mainly for the smallest Qwen models. TopDown-Bubblesort remains the strongest T5 baseline.

### Experiment 1b: Statistical Significance

We ran paired approximate-randomization tests directly on the saved per-query `ndcg_cut_10` values in the `.eval` files, with Bonferroni correction across the 18 configs per family. The full artifact is in `research_pipeline_setwise/SIGNIFICANCE_TESTS.md`.

| Family | Mean delta vs best TopDown | Positive deltas | Bonferroni-significant wins | Bonferroni-significant losses |
|--------|-----------------------------|-----------------|-----------------------|-------------------------|
| DualEnd | +0.0058 | 14/18 | 1 | 0 |
| BottomUp | -0.0616 | 0/18 | 0 | 6 |
| BiDir | -0.0232 | 3/18 | 0 | 3 |

**Interpretation**: DualEnd is consistently directional but statistically fragile on 43/54-query TREC DL test sets. Only `qwen3-4b` DL19 survives Bonferroni correction (`+0.0446`, adjusted `p=0.010`), while BottomUp and BiDir show more robust evidence of harm.

### Experiment 2: Efficiency Analysis (Comparisons per Query)

Average number of LLM calls per query for each method.

| Method | Comparisons | Relative to TD-Heap |
|--------|-------------|---------------------|
| TD-Heap | ~77 | 1.0x |
| TD-Bubble | ~310 | 4.0x |
| BU-Heap | ~277 | 3.6x |
| BU-Bubble | ~1683 | 21.9x |
| DE-Cocktail | ~546 | 7.1x |
| DE-Selection | ~406 | 5.3x |
| BiDir-RRF | ~355 | 4.6x |
| BiDir-Wt | ~355 | 4.6x |

**Interpretation**: DualEnd-Cocktail uses about 7x more comparisons than `TD-Heap`, and in wall-clock terms the family averages about `8.89x` (`DE-Cocktail`) and `5.60x` (`DE-Selection`) the `TD-Heap` time. The benefit is quality and richer information per comparison, not fewer total calls or lower end-to-end cost. The codebase now also supports causal likelihood follow-ups for Qwen/Qwen3.5, which remove decoding overhead but do not change this underlying comparison-count gap.

### Experiment 2b: Quality-Cost Pareto Frontier

We added a direct Pareto analysis over the existing TREC DL result folders (`results/analysis/pareto/QUALITY_COST_PARETO.md`) so the paper can talk about budgeted quality rather than isolated averages.

**Global mean frontier members**:

| Frontier | Members |
|----------|---------|
| Comparisons | `TD-Heap`, `TD-Bubble`, `DE-Cocktail` |
| Total tokens | `TD-Heap`, `TD-Bubble`, `DE-Cocktail` |
| Time | `PermVote(p=2)`, `TD-Heap`, `TD-Bubble`, `DE-Cocktail` |

**Interpretation**:
- `TD-Heap` remains the cheap anchor.
- `TD-Bubble` is the strongest middle-cost anchor.
- `DE-Cocktail` is the quality-first anchor.
- The paper's method gap is now precise: the right refinement should try to sit **between `TD-Bubble` and `DE-Cocktail`**, not merely claim a tiny average win over one baseline.

### Experiment 3: num_child Ablation (DualEnd-Cocktail)

Effect of window size on DualEnd-Cocktail quality (DL19).

| num_child | T5-xl | Q3-8B | Q3.5-9B |
|-----------|-------|-------|---------|
| 2 | .6988 | .7187 | .7392 |
| 3 (default) | .6884 | .7155 | .7370 |
| 5 | .6749 | .7224 | .7336 |
| 7 | .6480 | .7249 | .7386 |

**Interpretation**: For T5-xl, smaller windows (nc=2) are better, likely because T5's 512-token context is strained by larger windows. For Qwen models with 32k+ context, performance is relatively stable across window sizes, with slight advantages at nc=2 or nc=7 depending on the model. The default nc=3 is a reasonable compromise.

### Experiment 4: Position Bias Analysis

Full analysis across all 9 models × 2 datasets (18 result files in `results/analysis/position_bias/*/position_bias_results.txt`). Each file contains ~170K comparisons broken into 4 types: best, worst, dual_best, dual_worst.

**Position D (last passage) selection frequency by type (DL19 averages):**

| Type | A (first) | B | C | D (last) | Pattern |
|------|-----------|---|---|----------|---------|
| best (TopDown) | .33 | .13 | .13 | .41 | U-shaped: primacy + recency |
| worst (BottomUp) | .24 | .10 | .13 | .50 | Strong recency for "worst" |
| dual_best | .27 | .24 | .16 | .33 | Weaker bias, more uniform |
| dual_worst | .35 | .22 | .28 | .13 | **Reversed**: primacy for "worst" |

**Key findings**:
- **TopDown (best)**: U-shaped bias — positions A and D overselected, B and C underselected. Strongest on T5-XL (D=.533) and Q3.5-4B (D=.547).
- **BottomUp (worst)**: Extreme recency bias — position D selected as "worst" 40-63% of the time across all models. Models default to labeling the LAST passage as worst.
- **DualEnd dual_best**: More uniform than single-best. Chi-squared values consistently lower. Dual prompting partially mitigates position bias.
- **DualEnd dual_worst**: **Novel finding** — REVERSED bias pattern. Position A (first) is most frequently selected as worst (.23-.43), while D (last) is LEAST selected (.10-.19). This is opposite to standalone worst-selection. The dual context changes which position gets flagged as worst.
- **Interpretation**: When asked for both best AND worst, models shift worst-selection toward the first passage (primacy) rather than the last (recency). This suggests the dual prompt alters the cognitive process.

### Experiment 5: Query Difficulty Stratification

Averaged over all 18 model-dataset configurations, relative to the TopDown baseline:

| Tercile | BU-TD | DE-TD |
|---------|------:|------:|
| Easy | -0.0904 | +0.0155 |
| Medium | -0.0803 | +0.0157 |
| Hard | -0.0897 | +0.0021 |
| Overall | -0.0866 | +0.0111 |

**Key findings**:
- DualEnd helps most on easy and medium queries on average, but the gain shrinks substantially on hard queries.
- BottomUp hurts across all terciles with little evidence of a compensating niche.
- The direction of the DualEnd gain still varies by model, so the difficulty story should be presented as a trend rather than a universal rule.

### Experiment 6: Ranking Agreement Analysis

Pairwise agreement averaged across all 18 model-dataset configurations:

| Pair | Overlap@10 | Kendall tau | Agreement |
|------|-----------:|------------:|-----------|
| TopDown vs DualEnd | 7.01 | 0.9254 | high |
| TopDown vs BottomUp | 5.04 | 0.8589 | high |
| BottomUp vs DualEnd | 5.24 | 0.8767 | high |

**Key findings**:
- DualEnd rankings are much closer to TopDown than BottomUp, which supports the interpretation that DualEnd keeps the useful TopDown signal and adds refinement from the worst-selection side.
- BottomUp is not random, but it is consistently noisier and more divergent than DualEnd.
- The lack of complementarity helps explain why BiDir fusion usually hurts rather than helps.

### Experiment 7: Per-Query Wins Analysis

Average wins across the 18 model-dataset configurations:

| Comparison | First method wins | Second method wins | Ties |
|-----------|------------------:|-------------------:|-----:|
| TopDown vs BottomUp | 31.78 | 13.33 | 3.39 |
| TopDown vs DualEnd | 18.50 | 23.89 | 6.11 |
| BottomUp vs DualEnd | 11.50 | 33.39 | 3.61 |
| DualEnd vs BiDir-RRF | 28.67 | 16.00 | 3.83 |
| BiDir-RRF vs best(TD,BU) | helps 11.78 | hurts 10.06 | between 26.67 |

**Key findings**:
- DualEnd wins more queries than TopDown on average, but many of those margins are small enough that they do not survive multiple-testing correction.
- BottomUp rarely beats TopDown and loses heavily to DualEnd.
- BiDir is close to a wash against the better individual ranking, which is consistent with its negative average delta.

### Experiment 7b: "When DualEnd Helps" Summary

We added a qualitative analysis script that can read live retrieval artifacts (`pyserini` + `ir_datasets`) and emit paper-facing exemplar markdown with query text and passage snippets. The current generated summary covers three representative DL19 model settings:

| Model | Mean delta (DE - TD) | Help / Hurt / Tie |
|-------|----------------------:|------------------:|
| flan-t5-xl | -0.0017 | 20 / 16 / 7 |
| qwen3-8b | +0.0336 | 26 / 12 / 5 |
| qwen3-14b | +0.0072 | 19 / 15 / 9 |

**Interpretation**:
- DualEnd is not universally better than TopDown.
- The strongest positive regime currently looks more like **Qwen-style causal models** than T5-family models.
- In the strongest `qwen3-8b` setting, most positive cases are genuine **relevant-document additions** to the top-k, whereas the `flan-t5-xl` hurt cases lean more toward relevant-document drops or noisy mixed changes.
- This directly motivates **Selective DualEnd**: if the benefit is model- and window-dependent, the method should route the expensive joint prompt only where it is likely to matter.

### Experiment 8: Parse Success Rate

- All T5 models: 0 parse failures (likelihood scoring avoids parsing entirely)
- Qwen3/3.5: <1% partial parse rate. Worst case: Qwen3-14B DL20 with 254 "only parse one" out of ~29K comparisons (0.9%)
- No model has actual "dual parse failures" — cascading parser always produces a result
- Parse reliability is not a practical concern

## Figures

1. **Table 1**: Main results table --- NDCG@10 for all 8 methods across 9 models on DL19 and DL20 (the central result)
2. **Table 2**: Efficiency comparison --- comparisons, tokens, and wall time per query for all methods
3. **Figure 1**: Bar chart or heatmap showing NDCG@10 gains of DualEnd-Cocktail over TopDown-Heapsort baseline across all model-dataset configurations
4. **Figure 2**: Position bias distribution --- stacked bar charts showing selection frequency by candidate position for TopDown, BottomUp, and DualEnd
5. **Table 3**: num_child ablation results for DualEnd-Cocktail
6. **Figure 3**: Query difficulty stratification --- grouped bar chart showing NDCG@10 delta over BM25 for easy/medium/hard queries
7. **Table 4**: Ranking agreement matrix (Kendall tau) between method pairs for a representative model
8. **Figure 4**: Quality-cost Pareto frontier showing NDCG@10 against calls/tokens/time for TopDown, DualEnd, and any refined variant

## Known Weaknesses

- **Efficiency**: DualEnd-Cocktail uses 546 comparisons per query (about 7x `TD-Heap`'s 77), and the DualEnd family is 5.6x-8.9x slower than `TD-Heap` on average. The paper should not imply end-to-end efficiency gains.
- **Statistical power**: The significance tests are now done, but only one DualEnd gain survives Bonferroni correction on these 43/54-query test sets. Most positive deltas still have bootstrap CIs that cross zero.
- **Small query sets**: TREC DL 2019/2020 are standard benchmarks but have limited query counts. BEIR evaluation across diverse domains would strengthen generalization claims.
- **BEIR summary not yet complete**: the representative BEIR runs are already executing remotely, but the paper-facing macro summary and per-dataset delta table are still pending.
- **BottomUp results changed after bug fixes**: Early results showed BottomUp competitive with TopDown. After correcting prompt bugs (ensuring BottomUp exclusively uses "worst" prompts via `compare_worst()`), BottomUp dropped significantly. Some claims from earlier analyses may need revision.
- **DualEnd likelihood is heuristic rather than exact joint-output scoring**: T5 main results already use an internal likelihood shortcut, and the codebase now also provides the same max/min reuse heuristic for Qwen/Qwen3.5 follow-up runs. The same caveat applies to Selective DualEnd, bias-aware DualEnd, and same-call regularization whenever they enter their joint-signal path under `--scoring likelihood`. These likelihood paths are useful efficiency probes, but they are not exact likelihood models of the full `Best: X, Worst: Y` output.
- **No comparison to listwise or pairwise methods**: We only compare within the setwise paradigm. A broader comparison against listwise (RankGPT) and pairwise methods would contextualize the absolute performance levels.
- **Single re-ranking depth**: All experiments re-rank BM25 top-100. Sensitivity to initial retrieval quality and re-ranking depth (top-50, top-200) is not tested.

## Refinement Priorities

The highest-leverage next package is not more BottomUp or BiDir tuning. It is:

1. **Selective DualEnd**: use TopDown by default and invoke DualEnd only on query-locally uncertain windows or a final shortlist to recover a better quality-cost tradeoff.
2. **Order-robust / bias-aware DualEnd**: exploit the dual-worst bias reversal by running a small number of controlled orderings only where query-local uncertainty is high.
3. **Quality-cost Pareto analysis**: show NDCG@10 against calls, tokens, and wall time so the paper can make precise budget-aware claims.
4. **"When DualEnd helps" analysis**: identify the query or window regimes where the joint signal is most useful.
5. **Same-call worst-signal regularization**: test whether the useful part of worst-selection is only the negative constraint available inside the same joint prompt.

These are now implemented as runnable code paths, so the next step is empirical validation rather than more design work.

### What / Why / How For The Refinement Package

- **Selective DualEnd**
  - **What**: a TopDown sort that upgrades only routed windows to same-call best-worst prompting
  - **Why**: the help/hurt summary says the DualEnd gain is concentrated rather than universal
  - **How**: route by shortlist overlap, a query-local percentile gate over BM25 score spreads, or their hybrid union; for heapsort, disable the shortlist gate and keep only uncertainty routing
- **Order-robust / bias-aware DualEnd**
  - **What**: a DualEnd variant that runs a tiny number of controlled orderings only on hard windows
  - **Why**: the reversed `dual_worst` pattern is the strongest mechanism clue in the current paper
  - **How**: vote across base / reversed / shifted orderings, then map the result back into the original window; keep the supported sorts to bubblesort / selection so every run actually exercises the order-robust joint prompt path
- **Same-call worst-signal regularization**
  - **What**: a head-focused TopDown pass that uses worst only as a local demotion signal
  - **Why**: BottomUp is too noisy alone, but worst may help when conditioned on the same evidence as best
  - **How**: promote best as usual, push worst locally to the back, and avoid full backward passes

If these refinements work, the project becomes easier to position as an ICTIR paper now and potentially an ARR-style submission later.

## Related Work

- **Setwise LLM ranking**: Zhuang et al. (SIGIR 2024) introduced the setwise paradigm with heapsort and bubblesort. Our work directly extends this by proposing BottomUp, DualEnd, and Bidirectional strategies within the same framework. We use their codebase as the foundation.
- **Position bias in LLM ranking**: Liu et al. (2024, "Lost in the Middle") showed LLMs attend more to beginning and end of context. Tang et al. (2024) proposed permutation self-consistency to mitigate position bias in listwise ranking. Our DualEnd approach implicitly reduces position bias by requiring the LLM to identify both extremes.
- **Sorting algorithms for LLM ranking**: Sato (2026) surveyed sorting-based approaches for LLM ranking. BlitzRank (2026) proposed efficient sorting strategies. Our cocktail shaker sort and double-ended selection sort are novel algorithmic contributions within this space.
- **Rank fusion**: Cormack & Lynam (2009) introduced Reciprocal Rank Fusion (RRF). We apply RRF and weighted CombSUM to fuse TopDown and BottomUp rankings, finding that fusion does not help when one signal is consistently weaker.
- **LLM-based re-ranking**: Sun et al. (2023, RankGPT) proposed listwise ranking with LLMs. Qin et al. (2024) studied pairwise approaches. Ma et al. (2024) proposed tournament-style ranking. Our work complements these by improving information extraction within the setwise paradigm rather than proposing a new paradigm.

## Proposed Title

Finding the Worst Is Harder Than Finding the Best: Directional Asymmetry in LLM Setwise Ranking

## Target Venue

ICTIR (primary) / later ARR after refinement and broader generalization evidence
