# Paper Plan

> **Template for Workflow 3 — skip planning phase.** Fill in, then run `/paper-writing "PAPER_PLAN.md"`.

## Metadata
- **Title**: Finding the Worst Is Harder Than Finding the Best: Directional Asymmetry in LLM Setwise Ranking
- **Venue**: ICTIR (primary); later ARR only after a stronger refinement/generalization package
- **One-sentence contribution**: We show that setwise LLM ranking is directionally asymmetric: worst-selection alone is unreliable, but joint best-worst elicitation (DualEnd) is the strongest family overall, changes positional bias, and offers a quality-first but expensive refinement.

## Claims-Evidence Matrix
| # | Claim | Evidence | Section |
|---|-------|----------|---------|
| C1 | Setwise LLM ranking is directionally asymmetric: worst-selection alone is not a mirror of best-selection | Table 1 + position bias + agreement analysis | §1 / §6 |
| C2 | The DualEnd family achieves the strongest quality overall, especially on Qwen, but should be framed as quality-first rather than efficiency-first | Table 1 + efficiency table + `SIGNIFICANCE_TESTS.md` | §5 RQ2 |
| C3 | BottomUp is consistently weaker than TopDown | Table 1 + analysis | §5 RQ1 |
| C4 | Bidirectional ensemble does not outperform TopDown because BottomUp is too noisy | Table 1 + agreement + per-query wins | §5 RQ3 |
| C5 | Joint elicitation changes positional bias, including a reversed `dual_worst` pattern | Position bias figure | §6 |
| C6 | DualEnd gains are directionally consistent but statistically fragile on TREC DL, whereas BottomUp/BiDir losses are more robust | `SIGNIFICANCE_TESTS.md` | §5 / §6 |
| C7 | Window size (num_child) interacts differently with model families | Table 4 (ablation) | §5 |
| C8 | The paper's strongest narrative is a mechanism-and-analysis story about joint elicitation, not three equally successful bidirectional methods | Paper synthesis across results | §1 / §7 |
| C9 | The current quality-cost frontier is anchored by `TD-Heap`, `TD-Bubble`, and `DE-Cocktail`, which makes selective / bias-aware DualEnd the correct refinement target | `results/analysis/pareto/QUALITY_COST_PARETO.md` | §6 / §7 |

## Section Plan

### 1. Introduction (~1.5 pages)
- **What**: A study of directional asymmetry in setwise LLM ranking, centered on best-only, worst-only, and joint best-worst elicitation
- **Why**: Standard setwise ranking only asks "which is best?", but asking for the worst is not a symmetric inverse and may reveal different model behavior
- **How**: We compare BottomUp, DualEnd, and Bidirectional strategies to isolate whether worst-selection helps alone, jointly, or only in fusion
- **Result**: DualEnd is strongest overall, but the gains are modest and expensive; the most robust scientific result is the asymmetry between standalone and joint worst-selection

### 2. Related Work (~1 page)
- **LLM ranking paradigms**: pointwise, pairwise, listwise, setwise (Zhuang et al. 2024); gap: all use "best" selection only
- **Sorting algorithms for LLMs**: heapsort, bubblesort for document ranking; gap: no cocktail shaker or double-ended selection sort
- **Position bias in LLM ranking**: lost-in-the-middle (Liu et al. 2024), permutation self-consistency (Tang et al. 2024); gap: bias not studied for worst-only versus joint best-worst prompts
- **Negative evidence in LLM ranking**: motivate why coherent failures of BottomUp and BiDir are scientifically useful rather than dead ends

### 3. Methodology (~2 pages)
- **Problem formulation**: setwise ranking as iterative LLM-based comparison with candidate windows
- **BottomUp**: reverse prompt asking for "least relevant" document; included as a diagnostic test of directional symmetry
- **DualEnd**: single prompt asking for both "most relevant" AND "least relevant"; the central method of interest
- **Bidirectional ensemble**: run TopDown + BottomUp independently, fuse rankings via RRF, CombSUM, or weighted combination; included to test whether the signals are complementary
- **Prompt templates**: concrete prompt examples for each direction (TopDown, BottomUp, DualEnd)

### 4. Algorithms (~1.5 pages)
- **Cocktail shaker sort**: alternating forward (best) and backward (worst) passes for DualEnd bubblesort
- **Double-ended selection sort**: simultaneous min/max extraction for DualEnd selection
- **Selective DualEnd**: keep the TopDown sort structure, but invoke joint best-worst prompting only on shortlist or query-locally uncertain windows; for heapsort, uncertainty routing is cleaner than shortlist routing because heap nodes are not stable rank positions
- **Order-robust / bias-aware DualEnd**: run a tiny set of controlled orderings only on gated windows, then vote back into the original order; keep the supported sorts to bubblesort / selection so the order-robust path is actually exercised
- **Same-call worst-signal regularization**: keep a head-focused TopDown pass and use the same-call worst output only as a local negative constraint once that candidate is already outside the protected ranking head frontier (top-`k` plus one active window)
- **Complexity analysis**: comparison counts for each sorting method across directions
- **Information-theoretic analysis**: bits of information extracted per LLM call; keep as supporting intuition, not the main empirical claim

### 5. Experiments (~3 pages)
- **Setup**: 9 models (Flan-T5 large/xl/xxl, Qwen3 4B/8B/14B, Qwen3.5 4B/9B/27B), 2 datasets (TREC DL19, DL20), 8 methods (TopDown/BottomUp x heapsort/bubblesort + DualEnd cocktail/selection + Bidirectional RRF/CombSUM)
- **Scoring modes**: main paper tables use the established generation setups, while follow-up efficiency analysis can now also use causal likelihood scoring for Qwen/Qwen3.5 via teacher-forced short answer strings such as `Passage A`; for DualEnd and the routed joint-signal refinements, this likelihood path remains a best-only proxy rather than exact joint `Best: X, Worst: Y` scoring
- **RQ1 (BottomUp vs TopDown)**: BottomUp consistently underperforms TopDown across all models and datasets; use this to establish directional asymmetry
- **RQ2 (DualEnd effectiveness + efficiency)**: DualEnd is the strongest family overall, but at substantial extra inference cost; include significance table and scaling analysis across model sizes (C2, C6)
- **RQ3 (Bidirectional ensemble)**: RRF/CombSUM fusion does not outperform standalone TopDown; the signals are not complementary enough to justify fusion (C4)
- **Ablation**: num_child window size interaction with model families (Table 4); alpha weighting + passage length (Table 5)

### 6. Analysis (~1 page)
- **Position bias**: heatmap/bar chart showing bias patterns differ across best/worst/dual prompts (4 types x 4 positions); emphasize the reversed `dual_worst` pattern
- **Query difficulty**: stratified analysis by query difficulty (easy/medium/hard)
- **Ranking agreement**: pairwise agreement between methods (Kendall's tau or similar), supporting the interpretation that DualEnd mostly preserves TopDown's useful signal
- **Quality-cost discussion**: explicit table/figure showing the Pareto tradeoff across TopDown, DualEnd, and any refined variant if available
- **When DualEnd helps**: a small per-query help/hurt summary showing that Qwen-like models benefit more consistently than Flan-T5-XL, motivating routed rather than unconditional DualEnd
- **Refinement package motivation**:
  - **What**: Selective DualEnd, order-robust DualEnd, and same-call worst-signal regularization
  - **Why**: the Pareto frontier and help/hurt summary both say the missing contribution is not "more DualEnd everywhere" but "the same signal, spent where it matters"
  - **How**: gate by shortlist / query-local BM25-spread percentiles, disable shortlist routing in Selective heapsort, add controlled orderings only on hard windows, and reuse worst only inside the same call after the candidate leaves the protected ranking head frontier (top-`k` plus one active window)

### 7. Conclusion (~0.5 pages)
- **Summary**: The strongest conclusion is about directional asymmetry in setwise LLM ranking; DualEnd is the best quality-first variant, `TD-Heap` remains the efficiency baseline, and BottomUp/BiDir are useful negative results
- **Limitations**: evaluated on English TREC datasets only; limited to 9 models; DualEnd T5 uses likelihood internally rather than generation; most positive deltas are not Bonferroni-significant on 43/54-query test sets
- **Future work**: finish the BEIR summary package, benchmark the new Qwen/Qwen3.5 likelihood follow-up, run the already-implemented Selective / bias-aware / same-call variants, and identify the exact regimes where joint elicitation helps enough to justify its cost

## Figure Plan
| # | Type | Description | Auto? |
|---|------|-------------|:-----:|
| Fig 1 | Architecture | Overview diagram: TopDown vs BottomUp vs DualEnd prompt strategies | illustration |
| Fig 2 | Heatmap/bar chart | Position bias patterns across 4 prompt types x 4 positions | matplotlib |
| Fig 3 | Line chart | NDCG@10 vs model size scaling across 9 models | matplotlib |
| Table 1 | Comparison | Main results: 9 models x 8 methods x 2 datasets (NDCG@10, MAP) | LaTeX |
| Table 2 | Comparison | Efficiency: LLM calls, tokens, wall-clock time per method | LaTeX |
| Table 3 | Analysis | Information-theoretic: bits per call for TopDown, BottomUp, DualEnd | LaTeX |
| Table 4 | Ablation | num_child window size x model family interaction | LaTeX |
| Table 5 | Ablation | alpha weighting + passage length sensitivity | LaTeX |
| Fig 4 | Tradeoff curve | Quality-cost Pareto frontier across TopDown, DualEnd, and any refined variant | matplotlib |
| Table 6 | Analysis | "When DualEnd helps" help/hurt summary for 2-3 representative models | LaTeX |

## Key References
1. [Zhuang et al., "Setwise Prompting for LLM Reranking", SIGIR 2024]
2. [Sun et al., "ChatGPT as a passage reranker", 2023]
3. [Qin et al., "Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting", 2024]
4. [Tang et al., "Found in the Middle: Permutation Self-Consistency", NAACL 2024]
5. [Liu et al., "Lost in the Middle", 2024]
6. [Sato, "Survey of Sorting Algorithms for LLMs", 2026]
