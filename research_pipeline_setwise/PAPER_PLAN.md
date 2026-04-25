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
| C2 | The DualEnd family achieves the strongest quality overall on TREC DL19/20, especially on Qwen; out-of-domain claim is provisional pending BEIR generalization | Table 1 + efficiency table + `SIGNIFICANCE_TESTS.md` + `SIGNIFICANCE_TESTS_PAIRWISE.md` | §5 RQ2 |
| C3 | BottomUp is consistently weaker than TopDown | Table 1 + analysis | §5 RQ1 |
| C4 | Bidirectional ensemble does not outperform TopDown because BottomUp is too noisy | Table 1 + agreement + per-query wins | §5 RQ3 |
| C5 | Joint elicitation changes positional bias, including a reversed `dual_worst` pattern | Position bias figure | §6 |
| C6 | DualEnd gains are directionally consistent but statistically fragile on TREC DL, whereas BottomUp/BiDir losses are more robust | `SIGNIFICANCE_TESTS.md` + `SIGNIFICANCE_TESTS_PAIRWISE.md` | §5 / §6 |
| C7 | Window size (num_child) interacts differently with model families | Table 4 (ablation) | §5 |
| C8 | The paper's strongest narrative is a mechanism-and-analysis story about joint elicitation, not three equally successful bidirectional methods | Paper synthesis across results | §1 / §7 |
| C9 | The current quality-cost frontier is anchored by `TD-Heap`, `TD-Bubble`, and `DE-Cocktail`, which makes selective / bias-aware DualEnd and the MaxContext family the correct refinement target | `results/analysis/pareto/QUALITY_COST_PARETO.md` | §6 / §7 |
| C10 | The paper takes an ICTIR-first conservative framing — one modestly effective method (DualEnd) plus two coherent negative results (BottomUp, BiDir); ARR submission is gated on stronger refinement / generalization evidence | `research-wiki/claims/C10_framing_ictir_conservative.md` | §1 / §7 |

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
- **MaxContext family** (idea:007): fit the entire rerank pool (`pool_size ≤ 50`) into a single Qwen prompt. Three variants — DualEnd (best+worst per call, pool shrinks by 2), TopDown (best only, pool shrinks by 1), BottomUp (worst only, pool shrinks by 1). Trades many small joint-elicitation calls for very few large-window calls, designed to land in the empty region between `TD-Bubble` and `DE-Cocktail` on the comparisons-axis and wall-clock-axis frontiers (claim:C9). Qwen-generation only; numeric labels 1..N (not letters); hard invariants on `pool_size == hits == ranker.k`, `num_permutation == 1`, strict no-truncation, abort-on-bad-parse.
- **Prompt templates**: concrete prompt examples for each direction (TopDown, BottomUp, DualEnd, MaxContext-DualEnd)

### 4. Algorithms (~1.5 pages)
- **Cocktail shaker sort**: alternating forward (best) and backward (worst) passes for DualEnd bubblesort
- **Double-ended selection sort**: simultaneous min/max extraction for DualEnd selection
- **Selective DualEnd**: keep the TopDown sort structure, but invoke joint best-worst prompting only on shortlist or query-locally uncertain windows; for heapsort, uncertainty routing is cleaner than shortlist routing because heap nodes are not stable rank positions
- **Order-robust / bias-aware DualEnd**: run a tiny set of controlled orderings only on gated windows, then vote back into the original order; keep the supported sorts to bubblesort / selection so the order-robust path is actually exercised
- **Same-call worst-signal regularization**: keep a head-focused TopDown pass and use the same-call worst output only as a local negative constraint once that candidate is already outside the protected ranking head frontier (top-`k` plus one active window)
- **MaxContext-DualEnd** (idea:007): one prompt over the full live pool asks for both best and worst; reuses `_double_ended_selection` with `num_child = pool_size - 1` so the single-group fast-path fires for the whole pool. Pool shrinks by 2 per round; total `floor(N/2)` LLM calls.
- **MaxContext-TopDown / MaxContext-BottomUp**: one prompt asks for one extreme over the live pool. Pool shrinks by 1 per round. At `n_docs=2` the LLM is bypassed in favor of a deterministic BM25 score tiebreaker — higher-score-wins for TopDown (tail of ranking), lower-score-loses for BottomUp (head of ranking, ranks 1-2). Total `N-2` LLM calls + 1 BM25 bypass per query. The bypass is necessary because at `n_docs=2` the model is semantically unstable for the two-strong-survivors case and tends to refuse or hedge.
- **Complexity analysis**: comparison counts for each sorting method across directions, including MaxContext family at matched `hits ∈ {10, 30, 50}`.
- **Information-theoretic analysis**: bits of information extracted per LLM call; keep as supporting intuition, not the main empirical claim. MaxContext extracts `log2(N · (N-1))` bits per joint call at the first round (e.g., `log2(50·49) ≈ 11.3` for `N=50`).

### 5. Experiments (~3 pages)
- **Setup**: 9 models (Flan-T5 large/xl/xxl, Qwen3 4B/8B/14B, Qwen3.5 4B/9B/27B), 2 datasets (TREC DL19, DL20), 8 methods (TopDown/BottomUp x heapsort/bubblesort + DualEnd cocktail/selection + Bidirectional RRF/CombSUM)
- **Scoring modes**: main paper tables use the established generation setups, while follow-up efficiency analysis can now also use causal likelihood scoring for Qwen/Qwen3.5 via teacher-forced short answer strings such as `Passage A`; for DualEnd and the routed joint-signal refinements, this likelihood path remains a best-only proxy rather than exact joint `Best: X, Worst: Y` scoring
- **RQ1 (BottomUp vs TopDown)**: BottomUp consistently underperforms TopDown across all models and datasets; use this to establish directional asymmetry
- **RQ2 (DualEnd effectiveness + efficiency)**: DualEnd is the strongest family overall, but at substantial extra inference cost; include significance table and scaling analysis across model sizes (C2, C6)
- **RQ3 (Bidirectional ensemble)**: RRF/CombSUM fusion does not outperform standalone TopDown; the signals are not complementary enough to justify fusion (C4)
- **RQ4 (MaxContext family)** [conditional on idea:007 launch]: at matched `hits ∈ {10, 30, 50}`, does whole-pool one-prompt selection match `DE-Cocktail` quality at lower comparisons / wall-clock cost? Per claim:C9, the empty region between `TD-Bubble` and `DE-Cocktail` is the natural target. Token axis is expected to be worse than `DE-Cocktail` and is not claimed as a win.
- **Ablation**: num_child window size interaction with model families (Table 4); alpha weighting + passage length (Table 5); MaxContext pool-size sweep at fixed `pl=512` (Study A); MaxContext pl-sweep at the chosen pool size (Study B); MaxContext order-robustness pilot (Study C launch gate, Bonferroni Δ ≤ 0.01).

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
- **Summary**: The strongest conclusion is about directional asymmetry in setwise LLM ranking; DualEnd is the best quality-first variant, `TD-Heap` remains the efficiency baseline, and BottomUp/BiDir are useful negative results. The contribution is **joint elicitation** (claim:C8), not algorithmic novelty in the sort.
- **Framing** (claim:C10): ICTIR-first conservative — one modestly effective method plus two coherent negative results. ARR submission gated on stronger refinement / generalization evidence.
- **Limitations**: evaluated on English TREC datasets only; limited to 9 models; DualEnd T5 and `--scoring likelihood` paths use a best-only proxy rather than true joint elicitation; most positive deltas are not Bonferroni-significant on 43/54-query test sets
- **Future work**: finish the BEIR summary package; run the already-implemented Selective DualEnd / bias-aware DualEnd / same-call regularized variants (idea:004/005/006); launch the staged 312-run MaxContext family matrix (idea:007) including the MaxContext-TopDown / -BottomUp single-extreme pool sweeps; identify the exact regimes where joint elicitation helps enough to justify its cost.

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
