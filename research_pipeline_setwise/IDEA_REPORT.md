# IDEA REPORT: Novel Prompt Strategies for LLM-Based Setwise Ranking

**Research Direction:** Exploring reverse (bottom-up) and dual-end (simultaneous top-bottom) selection strategies for LLM-based setwise document ranking.

**Date:** March 15, 2026
**Pipeline Stage:** Stage 1 — Idea Discovery (Complete)

---

## Executive Summary

We propose three novel modifications to the LLM-based setwise ranking paradigm (Zhuang et al., SIGIR 2024). The standard approach asks the LLM to select the **most relevant** document from a candidate set, then uses sorting algorithms (heapsort/bubblesort) to build a top-k ranking. We investigate:

1. **Bottom-Up Setwise Ranking**: Select the **least relevant** document each time, building the ranking from bottom to top.
2. **Dual-End Setwise Ranking**: Select **both the most and least relevant** documents simultaneously, building from both ends.
3. **Bidirectional Ensemble**: Run both standard (top-down) and reverse (bottom-up) independently, then fuse rankings.

**Novelty Status:** All three ideas confirmed novel after comprehensive search of arxiv, ACL Anthology, Semantic Scholar, and web (see Novelty Check section).

---

## Idea 1: Bottom-Up Setwise Ranking (Reverse Selection)

### Core Concept

Replace the standard prompt:
```
"Which of the following passages is the most relevant to the query?"
```
with:
```
"Which of the following passages is the least relevant to the query?"
```

Then use sorting algorithms to iteratively extract the **least relevant** documents, building the ranking from the bottom up. The documents that survive the longest (never selected as "worst") end up at the top of the ranking.

### Hypothesis

**H1 (Effectiveness):** LLMs may exhibit different accuracy and bias patterns when identifying the *least* relevant document vs. the *most* relevant. Specifically:
- Identifying clearly irrelevant documents may be a cognitively **easier** task (higher inter-annotator agreement on "clearly bad" vs. "clearly best").
- Position bias may manifest differently for worst-selection vs. best-selection, potentially leading to complementary errors.
- The "elimination" framing may reduce ties and ambiguity in close-relevance scenarios.

**H2 (Efficiency):** For top-k ranking (k << n):
- **Heapsort**: Standard top-down needs O(k) extract-max operations. Bottom-up needs O(n−k) extract-min operations. When k << n, bottom-up is **less efficient** with heapsort.
- **Bubblesort**: The complexity analysis is similar — sinking worst documents requires O((n−k) · n/c) operations vs. O(k · n/c) for standard. **Less efficient** when k << n.
- **Key insight**: Bottom-up is more efficient when k > n/2 (i.e., when you want to rank most of the list). For the typical IR setting (k=10, n=100), bottom-up alone is less efficient.
- **However**, the effectiveness gains (if any) may justify the extra cost, or it may serve as a complementary signal for ensemble.

### Algorithm: Bottom-Up Heapsort

```python
def bottom_up_heapsort(docs, query, k):
    """Build a MIN-heap, extract n-k minimums to identify the top-k."""
    # Build min-heap (least relevant at root)
    build_min_heap(docs)  # LLM selects LEAST relevant as comparator

    # Extract n-k minimums (the worst documents)
    bottom_ranking = []
    for i in range(len(docs) - k):
        bottom_ranking.append(extract_min(docs))  # Remove least relevant

    # Remaining k documents are the top-k (need further sorting)
    # Sort remaining k documents using standard top-down setwise
    top_k = standard_setwise_sort(remaining_docs, k)
    return top_k
```

### Algorithm: Bottom-Up Bubblesort

```python
def bottom_up_bubblesort(docs, query, k):
    """Sink worst documents to the bottom instead of bubbling best to top."""
    # Start from top, push worst down
    for i in range(len(docs) - k):
        for j in range(0, len(docs) - i - 1, step=num_child):
            window = docs[j:j+num_child+1]
            worst = llm_select_least_relevant(query, window)
            # Move worst to end of window (sinks down)
            docs.remove(worst)
            docs.insert(j + num_child, worst)

    # Top k documents remain at the front
    return docs[:k]
```

### Experimental Design

| Parameter | Value |
|-----------|-------|
| Models | Flan-T5-XL, Flan-T5-XXL, Vicuna-7B-v1.5 |
| Datasets | TREC DL 2019, TREC DL 2020 |
| Metrics | NDCG@10, total LLM calls, total tokens |
| Baselines | Standard setwise (heapsort, bubblesort) |
| Sorting | heapsort (num_child=3), bubblesort (num_child=3) |
| k | 10 |
| hits | 100 (rerank top-100 BM25) |
| Permutations | 1 and 3 (to test interaction with position bias) |

### Risk Assessment

- **High risk for efficiency**: Bottom-up with heapsort is theoretically less efficient for top-k (k=10, n=100). Need n-k=90 extractions instead of 10.
- **Medium risk for effectiveness**: Unclear whether "worst selection" is easier or harder for LLMs. Needs pilot study.
- **Mitigation**: If effectiveness is better but efficiency is worse, this still contributes as an ensemble component or for full-ranking scenarios.

### Pilot Priority: ⭐⭐⭐ (Medium)

---

## Idea 2: Dual-End Setwise Ranking (Simultaneous Best-Worst Selection)

### Core Concept

Modify the prompt to ask for **both** selections in a single LLM call:
```
"Given a query '{query}', which of the following passages is the most relevant
and which is the least relevant to the query?

Passage A: "..."
Passage B: "..."
Passage C: "..."
Passage D: "..."

Output the passage label of the most relevant passage and the least relevant
passage in the format: Best: [label], Worst: [label]"
```

Then use a custom **double-ended selection** algorithm: each LLM call removes 2 documents from the unsorted pool (one placed at the top, one at the bottom), building the ranking from both ends simultaneously.

### Hypothesis

**H3 (Effectiveness):**
- Asking for both best AND worst forces the LLM to consider the full relevance spectrum, potentially leading to more calibrated judgments.
- The LLM already implicitly evaluates all documents when selecting the best; the worst selection comes "for free" in terms of the LLM's reasoning (it already knows which is worst).
- Cross-validation between best and worst selections could catch inconsistencies (e.g., if a document is selected as both best in one comparison and worst in another).

**H4 (Efficiency):**
- **2× information per LLM call**: Each call produces two ranking decisions instead of one.
- **Token overhead is minimal**: The prompt is ~10 tokens longer (adding "and the least relevant"). Output is ~5 tokens longer ("Best: A, Worst: C" vs. "A").
- **Total comparisons reduced by ~50%**: For a double-ended selection sort:
  - Standard: O(n) calls to rank n documents (selecting 1 per call).
  - Dual-end: O(n/2) calls (selecting 2 per call).
- **For top-k with heapsort**: Not directly applicable (heapsort structure assumes single comparator). Requires a new sorting algorithm.
- **For custom double-ended selection sort**: k=10 from n=100 requires ~10 rounds of selection (5 from top, 5 from bottom as bonus), with each round comparing a shrinking set.

### Algorithm: Double-Ended Selection Sort

```python
def dual_end_setwise_rank(docs, query, k):
    """Select best and worst simultaneously, building ranking from both ends."""
    top_ranking = []
    bottom_ranking = []
    remaining = list(docs)

    while len(top_ranking) < k and len(remaining) > 1:
        # Compare a window of num_child+1 documents
        if len(remaining) <= num_child + 1:
            window = remaining
        else:
            # Use heapsort-like selection for the comparison set
            window = select_comparison_set(remaining, num_child + 1)

        best, worst = llm_select_best_and_worst(query, window)

        top_ranking.append(best)
        remaining.remove(best)

        if worst != best and len(top_ranking) < k:
            bottom_ranking.insert(0, worst)
            remaining.remove(worst)

    # Final ranking: top_ranking + remaining (middle) + bottom_ranking
    return top_ranking  # For top-k, we primarily care about top_ranking
```

### Algorithm: Dual-End Heapsort Variant

```python
def dual_end_heapsort(docs, query, k):
    """Modified heapsort: each heapify step extracts both max AND min."""
    # Phase 1: Build max-heap as usual
    build_max_heap(docs)

    top_ranking = []
    bottom_ranking = []

    for i in range(k):
        # Extract max (standard)
        max_doc = extract_max(docs)
        top_ranking.append(max_doc)

        # During heapify-down, the "loser" documents are pushed down
        # Track the document that loses ALL comparisons during sift-down
        # That document is a candidate for the bottom of the ranking
        # (This extracts bonus bottom-ranking info from the same heapify)

    return top_ranking
```

### Alternative: Dual-End Bubblesort (Cocktail Shaker Sort)

```python
def dual_end_bubblesort(docs, query, k):
    """Cocktail shaker sort: alternate between bubbling up and sinking down."""
    top_sorted = 0
    bottom_sorted = 0

    while top_sorted < k:
        # Forward pass: bubble best to top
        for j in range(len(docs) - bottom_sorted - 1, top_sorted, -num_child):
            window = docs[max(0, j-num_child):j+1]
            best = llm_select_most_relevant(query, window)
            # Move best to front of window
            move_to_front(docs, best, window_start)
        top_sorted += 1

        # Backward pass: sink worst to bottom
        for j in range(top_sorted, len(docs) - bottom_sorted - 1, num_child):
            window = docs[j:j+num_child+1]
            worst = llm_select_least_relevant(query, window)
            # Move worst to end of window
            move_to_end(docs, worst, window_end)
        bottom_sorted += 1

    return docs[:k]
```

### Experimental Design

| Parameter | Value |
|-----------|-------|
| Models | Flan-T5-XL, Flan-T5-XXL, Vicuna-7B-v1.5 |
| Datasets | TREC DL 2019, TREC DL 2020 |
| Metrics | NDCG@10, total LLM calls, total tokens, latency |
| Baselines | Standard setwise (heapsort, bubblesort) |
| New algorithms | Double-ended selection, dual-end bubblesort (cocktail shaker) |
| num_child | 3, 5, 7 (test impact of set size on dual selection accuracy) |
| k | 10 |
| hits | 100 |

### Key Measurements

1. **Dual selection accuracy**: When the LLM is asked for both best and worst, how often does it agree with separate best-only and worst-only queries?
2. **Per-call information gain**: Measure bits of ranking information extracted per LLM call.
3. **Position bias interaction**: Does asking for both selections change position bias patterns?

### Risk Assessment

- **Low risk for efficiency**: Theoretical ~50% reduction in LLM calls is well-motivated. The only question is whether the parsing works reliably.
- **Medium risk for effectiveness**: Need to verify LLMs can reliably output both selections. Likelihood-based scoring won't work (can only score one token position) — must use generation mode.
- **Mitigation**: Fall back to generation-only scoring. Test parsing robustness across models.

### Pilot Priority: ⭐⭐⭐⭐⭐ (Highest)

---

## Idea 3: Bidirectional Ensemble (Top-Down + Bottom-Up Fusion)

### Core Concept

Run **both** standard setwise ranking (top-down, selecting best) and reverse setwise ranking (bottom-up, selecting worst) independently. Then fuse the two rankings using rank fusion.

```
Standard setwise → Ranking R1 (top-down)
Reverse setwise  → Ranking R2 (bottom-up)
─────────────────────────────────────────
Rank Fusion(R1, R2) → Final Ranking R*
```

### Hypothesis

**H5 (Effectiveness):**
- Top-down and bottom-up selections may capture different aspects of relevance, similar to how "Found in the Middle" (Tang et al., NAACL 2024) uses permutation self-consistency.
- Errors in top-down selection (e.g., mistakenly ranking a mediocre document first) may be caught by bottom-up (which would correctly eliminate it late).
- The ensemble acts as an implicit consistency check: documents ranked highly by both methods are more likely to be truly relevant.
- **Key hypothesis**: The correlation between errors in top-down and bottom-up is low, meaning fusion improves over either alone.

**H6 (Efficiency):**
- This approach **doubles** the computational cost (two full ranking passes).
- However, if it significantly improves effectiveness, it offers a different point on the efficiency-effectiveness Pareto frontier.
- Compare to: permutation voting (also multiplies cost for better results).

### Fusion Methods

1. **Reciprocal Rank Fusion (RRF)**: `score(d) = Σ 1/(k + rank_i(d))` across runs.
2. **CombSUM**: Normalize scores from each run, sum.
3. **CombMNZ**: Like CombSUM but multiply by the number of runs that returned the document.
4. **Weighted fusion**: `score = α · score_topdown + (1−α) · score_bottomup` with α tuned on dev set.

### Experimental Design

| Parameter | Value |
|-----------|-------|
| Models | Flan-T5-XL, Vicuna-7B-v1.5 |
| Datasets | TREC DL 2019, TREC DL 2020 |
| Metrics | NDCG@10, total LLM calls, total tokens |
| Baselines | Standard setwise, permutation voting (num_perm=2) |
| Fusion | RRF, CombSUM, weighted (α ∈ {0.3, 0.5, 0.7}) |
| Sorting | heapsort (num_child=3) |

### Risk Assessment

- **Medium risk for effectiveness**: Ensemble methods usually help, but the question is whether bottom-up is different enough to provide complementary signal.
- **High risk for efficiency**: 2× cost. Must show significant effectiveness gain to justify.
- **Key comparison**: Must beat permutation voting (num_perm=2) which also costs 2× but uses the same top-down approach with different orderings.

### Pilot Priority: ⭐⭐⭐ (Medium)

---

## Novelty Check

### Comprehensive Search Results

| Search Query | Sources Checked | Related Work Found |
|--------------|----------------|-------------------|
| "least relevant" + LLM + ranking | arxiv, Semantic Scholar, Google Scholar, web | **None** |
| "worst document" + selection + LLM + ranking | arxiv, web | **None** |
| "bottom up" + LLM + reranking | arxiv, web | Top-down partitioning (Parry et al., 2024) — different concept |
| "best and worst" + simultaneous + ranking + LLM | arxiv, web | **None** |
| "double ended" + sorting + LLM | arxiv, web | Classical CS only (Wang Min, 2010) — never applied to LLMs |
| "bidirectional" + LLM + ranking | arxiv, web | Bidirectional attention only — different concept |
| "cocktail shaker" + LLM + sort | arxiv, web | **None** |
| Sato (2026) survey on sorting with LLMs | arxiv | Covers heapsort, bubblesort, mergesort, quicksort, insertion sort — does **not** cover any of our ideas |

### Novelty Verdict

| Idea | Novelty Status | Confidence |
|------|---------------|------------|
| Idea 1: Bottom-Up Setwise | ✅ **CONFIRMED NOVEL** | High |
| Idea 2: Dual-End Setwise | ✅ **CONFIRMED NOVEL** | High |
| Idea 3: Bidirectional Ensemble | ✅ **CONFIRMED NOVEL** | High |

**Closest related work:**
- **BlitzRank** (2026): Extracts maximal information from k-wise comparisons via tournament graphs, but does not ask for worst selection or dual-end selection.
- **Permutation self-consistency** (Tang et al., NAACL 2024): Uses multiple permutations to debias, conceptually similar to ensemble (Idea 3) but through different mechanism.
- **AFR-Rank** (2025): Pre-filters irrelevant documents, but uses pointwise scoring, not iterative worst-selection.

---

## Idea Ranking

| Rank | Idea | Effectiveness Potential | Efficiency Potential | Novelty | Implementation Complexity | Overall Score |
|------|------|------------------------|---------------------|---------|--------------------------|--------------|
| **1** | **Dual-End Setwise (Idea 2)** | Medium-High | **High** (~50% fewer calls) | Confirmed | Medium | **⭐⭐⭐⭐⭐** |
| **2** | **Bottom-Up Setwise (Idea 1)** | Medium | Low (worse for top-k) | Confirmed | Low | **⭐⭐⭐** |
| **3** | **Bidirectional Ensemble (Idea 3)** | Medium-High | Low (2× cost) | Confirmed | Low | **⭐⭐⭐** |

### Recommendation

**Pursue Idea 2 (Dual-End Setwise) as the primary contribution**, with Idea 1 (Bottom-Up) as a secondary investigation and Idea 3 (Ensemble) as an analysis experiment.

**Rationale:**
- Idea 2 has the strongest theoretical justification for **both** effectiveness and efficiency improvements.
- It produces a clear, novel algorithmic contribution (double-ended selection sort for LLM ranking).
- The efficiency gain (~50% fewer LLM calls) is a compelling practical benefit.
- Idea 1 provides important empirical analysis (how do LLMs handle worst-selection?) that supports the paper's narrative.
- Idea 3 provides an ablation study showing whether top-down and bottom-up capture complementary signals.

### Suggested Paper Framing

**Title:** "Beyond Best Selection: Bidirectional Strategies for LLM-Based Setwise Ranking"

**Story arc:**
1. Observation: Current setwise ranking only extracts "best" information per LLM call, wasting implicit "worst" knowledge.
2. Research questions: (RQ1) Can LLMs reliably identify least-relevant documents? (RQ2) Can dual-end selection improve efficiency? (RQ3) Do top-down and bottom-up provide complementary signals?
3. Methods: Bottom-up setwise, dual-end setwise, bidirectional ensemble.
4. Key result: Dual-end achieves comparable effectiveness with ~50% fewer LLM calls.
5. Analysis: Position bias patterns differ for best vs. worst selection; ensemble captures complementary signals.

---

## Pilot Experiment Plan

### Phase 1: Feasibility (Immediate)

1. **Prompt validation**: Test bottom-up and dual-end prompts with Flan-T5-XL on 10 TREC DL 2019 queries.
   - Can the model reliably output the least relevant passage?
   - Can the model reliably output both best and worst?
   - What is the parsing success rate?

2. **Position bias analysis**: For 10 queries, compare position bias patterns between:
   - Standard: "most relevant"
   - Reverse: "least relevant"
   - Dual: "most and least relevant"

### Phase 2: Small-Scale Evaluation

3. **Full TREC DL 2019** (43 queries) with Flan-T5-XL:
   - Standard setwise heapsort (baseline)
   - Bottom-up setwise (Idea 1)
   - Dual-end selection sort (Idea 2)
   - Log: NDCG@10, # LLM calls, total tokens

### Phase 3: Scale-Up

4. **Full evaluation** across DL 2019, DL 2020, multiple models.
5. **Ensemble experiment** (Idea 3).
6. **Analysis experiments**: position bias, consistency, error analysis.

---

## Files & References

- **Literature Review:** `/Users/hangli/research_pipeline_setwise/LITERATURE_REVIEW.md`
- **Codebase:** `/Users/hangli/projects/llm-rankers/`
- **Key implementation file:** `/Users/hangli/projects/llm-rankers/llmrankers/setwise.py`
- **This report:** `/Users/hangli/research_pipeline_setwise/IDEA_REPORT.md`
