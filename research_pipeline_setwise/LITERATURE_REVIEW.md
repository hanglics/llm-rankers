# Literature Review: LLM-Based Setwise Ranking for Information Retrieval

**Compiled: March 2026; refreshed April 2026 with project-connection notes (§13).**

> **Companion knowledge base:** the structured wiki at `/Users/hangli/projects/llm-rankers/research-wiki/` indexes 20 papers, 7 ideas, 33 experiments, 10 claims, 5 gaps, and 163 typed edges. Each paper page in this review has a corresponding `papers/<slug>.md` entry; this review is the prose narrative, the wiki is the relational view.

---

## Table of Contents

1. [Introduction and Scope](#1-introduction-and-scope)
2. [Taxonomy of LLM-Based Ranking Approaches](#2-taxonomy-of-llm-based-ranking-approaches)
   - 2.1 Pointwise Methods
   - 2.2 Pairwise Methods
   - 2.3 Listwise Methods
   - 2.4 Setwise Methods
3. [The Setwise Ranking Approach](#3-the-setwise-ranking-approach)
   - 3.1 Original Formulation (Zhuang et al., SIGIR 2024)
   - 3.2 Sorting Algorithms for Setwise Ranking
   - 3.3 Implementation Details
4. [Efficiency and Effectiveness Trade-offs](#4-efficiency-and-effectiveness-trade-offs)
5. [Position Bias and Inconsistency in LLM Ranking](#5-position-bias-and-inconsistency-in-llm-ranking)
6. [Extensions and Variations of Setwise Ranking](#6-extensions-and-variations-of-setwise-ranking)
   - 6.1 Setwise Insertion
   - 6.2 TourRank (Tournament-Based)
   - 6.3 Rank-R1 (Reasoning via Reinforcement Learning)
   - 6.4 REALM (Recursive Bayesian Updates)
   - 6.5 BlitzRank (Tournament Graphs)
7. [Reasoning-Enhanced Rerankers](#7-reasoning-enhanced-rerankers)
8. [Attention-Based and Non-Generative Methods](#8-attention-based-and-non-generative-methods)
9. [Efficiency Improvements](#9-efficiency-improvements)
10. [Open-Source Toolkits and Frameworks](#10-open-source-toolkits-and-frameworks)
11. [Key Open Problems and Future Directions](#11-key-open-problems-and-future-directions)
12. [Reference List](#12-reference-list)

---

## 1. Introduction and Scope

Large Language Models (LLMs) have demonstrated remarkable capabilities as zero-shot passage and document rerankers in information retrieval (IR) systems. Since the seminal work of Sun et al. (2023) on RankGPT, the field has seen rapid development of diverse prompting strategies that leverage LLMs to reorder candidate documents by relevance. This literature review surveys the key approaches, with particular emphasis on **setwise ranking** -- a strategy that asks the LLM to select the most relevant document from a small candidate set, and uses sorting algorithms to produce a full ranking.

This review covers papers from 2023 through early 2026, focusing on:
- The four major prompting paradigms (pointwise, pairwise, listwise, setwise)
- The original setwise approach and its extensions
- Position bias and inconsistency issues
- Efficiency improvements for LLM-based ranking
- Reasoning-enhanced rerankers
- Open-source implementations and toolkits

---

## 2. Taxonomy of LLM-Based Ranking Approaches

The literature identifies four fundamental prompting paradigms for LLM-based zero-shot document ranking:

### 2.1 Pointwise Methods

Pointwise methods score each document independently with respect to the query. The LLM evaluates one query-document pair at a time.

- **MonoT5** (Nogueira et al., 2020): Adapts T5 with a relevance prediction prefix (e.g., "Query: q Document: d Relevant:") and uses the probability of the "true" token as a relevance score. A foundational pointwise reranker.
- **Yes/No Prompting**: The LLM is asked "Is document D relevant to query Q?" and the probability of "Yes" vs. "No" determines the score.
- **Query Likelihood Model (QLM)**: The LLM estimates the likelihood of generating the query given the document.
- **Relevance Generation** (Liang et al., 2022; Sachan et al., 2022): The LLM generates a relevance label or score for each document.

**Strengths**: Highly efficient -- each document requires only one LLM inference. Easily parallelizable.
**Weaknesses**: Poor at capturing relative relevance between documents. Generally the weakest in terms of ranking effectiveness.

### 2.2 Pairwise Methods

Pairwise methods compare two documents at a time, asking the LLM which is more relevant to the query.

- **PRP -- Pairwise Ranking Prompting** (Qin et al., NAACL 2024): Introduces a pairwise comparison strategy using LLMs, achieving competitive results with GPT-4 using smaller models like Flan-UL2 (20B). PRP outperforms ChatGPT-based solutions by 4.2% and pointwise solutions by >10% on average NDCG@10. Supports both all-pairs and sorting-based approaches (heapsort, bubblesort).
- **DuoT5** (Pradeep et al., 2021): A T5-based pairwise reranker from the Expando-Mono-Duo pipeline.

**Strengths**: Superior effectiveness due to direct inter-document comparison. Insensitive to absolute scoring calibration.
**Weaknesses**: O(n^2) comparisons for all-pairs; even with sorting algorithms, requires O(n log n) comparisons. High computational overhead.

### 2.3 Listwise Methods

Listwise methods present multiple documents simultaneously and ask the LLM to produce a complete ranking.

- **RankGPT** (Sun et al., EMNLP 2023): The seminal listwise reranking work. Instructs ChatGPT/GPT-4 to reorder a set of passages by relevance. Uses a **sliding window** strategy to handle context length limitations: a window of w documents slides from the bottom of the ranking to the top, with the LLM reranking each window. Demonstrates that "properly instructed LLMs can deliver competitive, even superior results to state-of-the-art supervised methods." Introduces NovelEval to address data contamination concerns. Shows distillation can compress ranking capabilities into smaller models (440M parameters outperforming 3B supervised models on BEIR).
- **LRL / Zero-Shot Listwise Reranking** (Ma et al., 2023): Concurrent work on listwise prompting with open-source LLMs.
- **Rank-without-GPT** (Zhang et al., ECIR 2025): Builds effective listwise rerankers on open-source LLMs without GPT dependency. Their best model surpasses GPT-3.5 by 13% and achieves 97% of GPT-4 effectiveness. Highlights the need for high-quality listwise training data.
- **Self-Calibrated Listwise Reranking** (Ren et al., WWW 2025): Addresses calibration issues in listwise ranking.
- **PE-Rank** (Liu et al., 2024): Leverages passage embeddings for efficient listwise reranking, using special tokens instead of full passage text to reduce token consumption.

**Strengths**: Captures inter-document relationships. Produces coherent rankings.
**Weaknesses**: Context length constraints require sliding windows. Sensitive to input order (position bias). Expensive per inference due to long prompts.

### 2.4 Setwise Methods

Setwise methods present a small set of candidate documents and ask the LLM to **select the single most relevant document** from the set. This selection is then embedded within a sorting algorithm to produce a full ranking.

- **Setwise Prompting** (Zhuang et al., SIGIR 2024): The original and defining work. Detailed in Section 3.

**Strengths**: Balances effectiveness and efficiency. Fewer tokens per comparison than listwise. More informative per call than pairwise (compares multiple documents simultaneously).
**Weaknesses**: Effectiveness depends on sorting algorithm and set size. Position bias still present within the candidate set.

---

## 3. The Setwise Ranking Approach

### 3.1 Original Formulation (Zhuang et al., SIGIR 2024)

**Paper**: "A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models"
**Authors**: Shengyao Zhuang, Honglei Zhuang, Bevan Koopman, Guido Zuccon
**Venue**: SIGIR 2024 (Full paper)
**arXiv**: 2310.09497

The setwise approach complements the three existing paradigms (pointwise, pairwise, listwise) by introducing a **selection-based** comparison. Given a query and a candidate set of documents {A, B, C, ...}, the LLM is prompted:

```
Given a query "{query}", which of the following passages is the most
relevant one to the query?

Passage A: "{doc_a}"
Passage B: "{doc_b}"
Passage C: "{doc_c}"
...

Output only the passage label of the most relevant passage:
```

The LLM outputs a single label (e.g., "A"), identifying the winner. This comparison is then used as the comparator function within a sorting algorithm (heapsort or bubblesort) to produce a top-k ranking.

**Key findings from the paper:**
1. Pointwise approaches are efficient but have poor effectiveness.
2. Pairwise approaches achieve superior effectiveness but with high computational overhead.
3. Listwise approaches balance effectiveness and efficiency but are constrained by context length.
4. **Setwise provides the best trade-off**: it reduces both the number of LLM inferences and total token consumption compared to pairwise and listwise approaches, while maintaining high effectiveness.
5. The approach is compatible with both generation-based scoring (decoding the output label) and likelihood-based scoring (comparing output token probabilities).

### 3.2 Sorting Algorithms for Setwise Ranking

The setwise approach uses the LLM as a multi-way comparator within classical sorting algorithms:

**Heapsort (default)**:
- Builds an n-ary max-heap where each node is compared against its `num_child` children.
- Each comparison involves `num_child + 1` documents presented to the LLM.
- The LLM selects the most relevant, which becomes the root (winner).
- After extracting the maximum, the heap is re-heapified.
- Only k extract-max operations are needed for top-k ranking.
- Complexity: O(n + k log n) comparisons, each comparing `num_child + 1` documents.

**Bubblesort**:
- Starts from the bottom of the list and "bubbles up" the most relevant documents.
- Each comparison involves a window of `num_child + 1` documents.
- The winner is moved to the front of the window, effectively bubbling to the top.
- Includes an optimization: tracks last-start position to skip unnecessary comparisons.
- Complexity: O(k * n / num_child) comparisons.

The `num_child` parameter (default: 3) controls the "arity" of the comparison -- how many documents the LLM sees at once. Larger values mean fewer total comparisons but longer prompts per call.

### 3.3 Implementation Details

Based on the reference implementation in `ielab/llm-rankers` (GitHub):

- **Model support**: T5-family (Flan-T5), LLaMA-family (Vicuna, LLaMA), and OpenAI API models.
- **Scoring modes**:
  - `generation`: The LLM generates the passage label; the decoded output determines the winner.
  - `likelihood`: The logits of the candidate passage label tokens are compared directly (T5 only). More efficient as no decoding is needed.
- **Permutation voting** (`num_permutation`): To mitigate position bias, documents and labels can be randomly shuffled across multiple permutations. Results are aggregated via majority voting. This reduces the effect of input order on the outcome.
- **Label scheme**: Uses single characters (A, B, C, ...) for standard setwise, and bracketed numbers ([1], [2], ..., [20]) for Rank-R1.

---

## 4. Efficiency and Effectiveness Trade-offs

A central theme across the literature is the tension between ranking quality and computational cost:

| Method | Comparisons (top-k of n) | Tokens per Comparison | Effectiveness |
|--------|--------------------------|----------------------|---------------|
| Pointwise | n (independent) | Short (1 doc) | Low |
| Pairwise (all-pairs) | O(n^2) | Medium (2 docs) | Highest |
| Pairwise (heapsort) | O(n + k log n) | Medium (2 docs) | High |
| Listwise (sliding window) | O(n/step) | Long (window docs) | High |
| Setwise (heapsort) | O(n + k log n) | Medium (c+1 docs) | High |
| Setwise (bubblesort) | O(k * n/c) | Medium (c+1 docs) | High |

**E2R-FLOPs** (Peng et al., EMNLP 2025 Industry): Proposes hardware-independent efficiency metrics for LLM rerankers:
- **RPP** (Ranking Per PetaFLOP): How much ranking quality per PetaFLOP.
- **QPP** (Queries Per PetaFLOP): How many queries processed per PetaFLOP.
These metrics account for model size, unlike proxy metrics like latency or token count.

**Survey on Sorting with LLMs** (Sato, 2026): A comprehensive survey noting that traditional sorting algorithms may not be optimal when the comparison function is an LLM, because:
- LLM comparisons are expensive (time, money, tokens).
- LLM comparisons can be batched (parallelism changes the cost model).
- LLM comparisons are noisy (inconsistent, non-transitive).
- k-wise comparisons (setwise) can extract more information per call than binary (pairwise).

**Are Optimal Algorithms Still Optimal?** (arXiv 2505.24643): Challenges whether classical sorting complexity bounds apply to LLM-based ranking, given that batching and caching fundamentally change the cost model.

---

## 5. Position Bias and Inconsistency in LLM Ranking

### 5.1 Position Bias

LLMs exhibit systematic bias based on the position of documents in the prompt:

- **"Lost in the Middle"** phenomenon (Liu et al., 2024): LLMs tend to attend more to information at the beginning and end of the context, neglecting the middle.
- **"Found in the Middle"** (Tang et al., NAACL 2024): Proposes **permutation self-consistency** to address positional bias. Key idea: repeatedly shuffle the document list, pass each permutation through the LLM, and aggregate rankings to produce an order-independent result. Proves convergence to the true ranking under random perturbations. Achieves 7-18% improvement for GPT-3.5 and 8-16% for LLaMA v2 (70B).
- **"Lost but Not Only in the Middle"** (Hutter et al., ECIR 2025): Systematically investigates positional bias in RAG settings, showing bias depends on document type and LLM architecture.
- **Empirical Study of Position Bias** (Zeng et al., EMNLP 2025 Findings): Introduces Position Sensitivity Index (PSI) as a diagnostic metric for quantifying position bias across retrieval models.
- **Positional Bias in RAG** (Cuconasu et al., 2025): Shows that retrieval pipelines systematically bring distracting passages to top ranks, with >60% of queries containing highly distracting passages in top-10.

### 5.2 Intrinsic Inconsistency

- **LLM-RankFusion** (Zeng et al., NeurIPS 2024 Workshop): Identifies two types of intrinsic inconsistency in LLM-based comparisons:
  1. **Order inconsistency**: Conflicting results when the document order is swapped (e.g., comparing A vs. B gives a different result than B vs. A).
  2. **Transitive inconsistency**: Non-transitive preference triads (A > B, B > C, but C > A).

  Proposes mitigations through in-context learning for order-agnostic comparisons, calibration for preference probability estimation, and aggregation from multiple rankers. These inconsistencies are particularly relevant for sorting-based setwise methods, which assume a consistent comparator.

### 5.3 Mitigation in Setwise Ranking

The `ielab/llm-rankers` implementation addresses position bias through:
- **Permutation voting**: Random shuffling of both document order and label assignment across multiple permutations, with majority voting to determine the winner.
- **Shuffle ranking**: Support for randomizing or reversing the initial input ranking order before reranking.
- **Likelihood-based scoring**: Using token logits instead of generation reduces (but does not eliminate) position effects.

---

## 6. Extensions and Variations of Setwise Ranking

### 6.1 Setwise Insertion (Podolak et al., SIGIR 2025)

**Paper**: "Beyond Reproducibility: Advancing Zero-shot LLM Reranking Efficiency with Setwise Insertion"
**Authors**: Jakub Podolak, Leon Peric, Mina Janicijevic, Roxana Petcu
**Venue**: SIGIR 2025

Extends the original setwise approach by leveraging the **initial document ranking as prior knowledge**. Instead of treating all documents equally in the sorting process, Setwise Insertion prioritizes candidates more likely to improve the ranking, reducing unnecessary comparisons.

**Results**:
- 31% reduction in query processing time
- 23% reduction in model inferences
- Slight improvement in reranking effectiveness
- Validated across Flan-T5, Vicuna, and LLaMA architectures

This approach is analogous to insertion sort with a warm start: since the initial ranking from the first-stage retriever already provides a reasonable ordering, the algorithm can focus on refining rather than building from scratch.

### 6.2 TourRank (Chen et al., WWW 2025)

**Paper**: "TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy"
**Authors**: Yiqun Chen, Qi Liu, Yi Zhang, Weiwei Sun, Xinyu Ma, Wei Yang, Daiting Shi, Jiaxin Mao, Dawei Yin

Draws inspiration from sports tournament brackets:
1. **Multi-stage grouping**: Documents are divided into groups (like tournament pools), reducing input length constraints.
2. **Points system**: Instead of single-elimination, uses accumulated points across rounds for more robust rankings, mitigating the influence of document input order.
3. **Parallelizable**: Group-stage comparisons are independent and can run in parallel.

Delivers state-of-the-art performance but requires more LLM calls (~127 calls) compared to sliding window approaches (~9 calls).

### 6.3 Rank-R1 (Zhuang et al., 2025)

**Paper**: "Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning"
**Authors**: Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, Guido Zuccon

A major extension of the setwise framework that introduces **reasoning capabilities** through reinforcement learning (GRPO). Built on top of the setwise architecture:

- Uses a setwise prompting format with bracketed labels ([1], [2], ..., [20]).
- Trained via GRPO with minimal relevance labels.
- The model generates a **reasoning chain** before selecting the most relevant document.
- Achieves performance comparable to supervised fine-tuning with only **18% of training data**.
- Significantly outperforms zero-shot and supervised approaches on complex queries (BRIGHT dataset).
- Available with LoRA adapters for efficient deployment via vLLM.

Implementation uses `num_child=19` (20-way comparison), much larger than the default setwise setting of 3, enabled by the reasoning model's stronger comprehension.

### 6.4 REALM (Wang et al., EMNLP 2025)

**Paper**: "REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking"
**Authors**: Pinhuan Wang, Zhiqiu Xia, Chunhua Liao, Feiyi Wang, Hang Liu

An uncertainty-aware framework that:
- Models LLM-derived relevance scores as **Gaussian distributions** rather than point estimates.
- Refines rankings through **recursive Bayesian updates**, reducing redundant queries.
- Improves NDCG@10 by 0.7-11.9 while reducing LLM inferences by **23.4-84.4%**.
- Addresses ranking uncertainty by explicitly modeling confidence.

### 6.5 BlitzRank (arXiv 2602.05448)

A principled approach using **tournament graphs** for zero-shot ranking. Unlike classical sorting algorithms that primarily use each comparison to identify a winner (discarding loser information), BlitzRank extracts maximal information from every k-wise comparison by modeling the complete preference graph.

---

## 7. Reasoning-Enhanced Rerankers

A major trend from 2025 onward is integrating reasoning capabilities into LLM rerankers:

| Model | Paradigm | Training | Key Innovation |
|-------|----------|----------|----------------|
| **Rank-R1** (Zhuang et al., 2025) | Setwise | GRPO (RL) | First reasoning setwise reranker; 18% data |
| **Rank1** (arXiv 2502.18418, COLM 2025) | Pointwise | RL + Test-time compute | First reasoning reranker using test-time compute |
| **Rank-K** (arXiv 2505.14432) | Listwise | RL | Test-time reasoning for listwise; 23% improvement over RankZephyr |
| **ReasonRank** (arXiv 2508.07050) | Listwise | SFT + RL | Cold-start SFT then RL; outperforms Rank1 with lower latency |
| **REARANK** (arXiv 2505.20046) | Listwise | RL | Reasoning re-ranking agent |
| **LimRank** (arXiv 2510.23544) | Listwise | RL | Less-is-more reasoning for efficiency |
| **InsertRank** (arXiv 2506.14086) | Listwise | RL | Incorporates BM25 scores to ground reasoning |

The general pattern is:
1. Use supervised fine-tuning (SFT) or direct RL to teach the LLM a reasoning pattern for relevance assessment.
2. The model generates a chain-of-thought before producing its ranking decision.
3. RL (typically GRPO) optimizes for ranking metrics directly.

---

## 8. Attention-Based and Non-Generative Methods

### In-Context Reranking (ICR) (Chen et al., ICLR 2025)

**Paper**: "Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers"
**Authors**: Shijie Chen, Bernal Jimenez Gutierrez, Yu Su

Challenges the assumption that autoregressive generation is necessary for LLM-based reranking. Proposes using **attention patterns** directly:
- More relevant documents receive more attention weight when the LLM processes query tokens.
- No generation needed -- only a single forward pass is required.
- Works with any LLM, including base models without instruction tuning.
- Significantly more efficient than generation-based approaches.

### BlockRank / Scalable In-Context Ranking (NeurIPS 2025)

Identifies exploitable structures in attention:
1. **Inter-document block sparsity**: Attention is dense within document blocks but sparse across them.
2. **Query-document block relevance**: Attention scores from query tokens to document blocks in middle layers correlate with actual relevance.

Uses these structures for efficient blockwise ranking.

---

## 9. Efficiency Improvements

Key strategies for reducing the computational cost of LLM-based ranking:

1. **Sorting algorithm selection**: Heapsort for setwise (O(n + k log n)) vs. all-pairs (O(n^2)) vs. sliding window for listwise.

2. **Prior knowledge exploitation**: Setwise Insertion uses the initial ranking to warm-start the sorting process (31% time reduction).

3. **Pre-filtering**: (arXiv 2406.18740) Filter out clearly irrelevant passages before applying expensive LLM reranking, reducing the candidate set size.

4. **Likelihood-based scoring**: Using output token logits directly instead of autoregressive generation. Avoids the overhead of beam search / sampling.

5. **Passage embedding compression**: PE-Rank replaces full passage text with learned special tokens, dramatically reducing prompt length while preserving ranking quality.

6. **Attention-based methods**: ICR (Chen et al., ICLR 2025) requires only a single forward pass with no generation, offering order-of-magnitude speedups.

7. **Knowledge distillation**: RankGPT demonstrates that ranking capabilities of large proprietary models can be distilled into models 1/100th the size.

8. **Adaptive retrieval**: Rathee et al. (ECIR 2025) propose methods that guide the retrieval process itself using listwise ranker feedback, overcoming the bounded recall problem.

9. **Recursive Bayesian updates**: REALM reduces redundant LLM queries by 23-84% through uncertainty modeling.

10. **Batching and caching**: Modern frameworks exploit vLLM, SGLang, or TensorRT-LLM for efficient batched inference.

---

## 10. Open-Source Toolkits and Frameworks

### ielab/llm-rankers
- **URL**: https://github.com/ielab/llm-rankers
- **Scope**: Reference implementation of pointwise, pairwise, listwise, and setwise ranking.
- **Features**: Supports Flan-T5, Vicuna, LLaMA, OpenAI APIs. Includes Rank-R1 with LoRA adapters via vLLM.
- **Authors**: Shengyao Zhuang (ArvinZhuang) and collaborators.
- **Latest release**: v0.0.3 (March 2025); includes Rank-R1.

### castorini/rank_llm (RankLLM)
- **URL**: https://github.com/castorini/rank_llm
- **Scope**: Comprehensive Python toolkit for reproducible LLM-based reranking, with a focus on listwise methods.
- **Features**: Supports pointwise (MonoT5), pairwise (DuoT5), listwise (RankGPT, RankGemini), reasoning rerankers. Compatible with vLLM, SGLang, TensorRT-LLM, OpenRouter API. Custom prompt templates via YAML.
- **Venue**: SIGIR 2025 (Resource paper).
- **Stars**: 583+ (as of March 2026).

### LLM4Ranking
- **URL**: Referenced in arXiv 2504.07439
- **Scope**: Easy-to-use framework supporting pointwise, pairwise, listwise, and selection-based (TourRank) paradigms in a unified interface.

### hltcoe/llm-heapsort-reranking
- **URL**: https://github.com/hltcoe/llm-heapsort-reranking
- **Scope**: Focused implementation of LLM setwise reranking using heapsort with configurable n-ary comparisons. Supports OpenAI-compatible APIs (including local vLLM).

---

## 11. Key Open Problems and Future Directions

1. **Position bias remains unsolved**: While permutation voting and self-consistency mitigate bias, they multiply computational cost. More efficient debiasing strategies are needed.

2. **Inconsistency in comparisons**: LLM comparisons violate transitivity and order consistency. This fundamentally challenges sorting-based approaches. Better aggregation methods (e.g., rank fusion, Bayesian models) are active research areas.

3. **Optimal sorting algorithms for noisy comparators**: Classical sorting assumes a consistent comparator. When comparisons are noisy, stochastic, and non-transitive, the optimal algorithm design changes. This is an emerging research area (Sato, 2026; BlitzRank).

4. **Reasoning for ranking**: The integration of chain-of-thought reasoning into rerankers (Rank-R1, Rank1, ReasonRank) is a major trend, but raises efficiency concerns due to longer generation. Balancing reasoning depth with latency is an open challenge.

5. **Test-time compute scaling**: While reasoning models benefit from more computation at inference time, initial results (Rank1) show that naive budget-forcing does not reliably improve reranking. Better strategies for allocating test-time compute to ranking are needed.

6. **Bounded recall problem**: All reranking approaches can only reorder documents that were retrieved in the first stage. Adaptive retrieval guided by LLM rerankers (Rathee et al., ECIR 2025) is a promising direction.

7. **Benchmark saturation**: Traditional benchmarks (DL19, BEIR) may no longer differentiate top-performing rerankers. More challenging benchmarks (BRIGHT) and evaluation on complex, multi-faceted queries are needed.

8. **Set selection vs. ranking**: Lee et al. (2024) propose shifting from ranking to **set selection** for RAG, where the goal is to find a collectively informative set of passages rather than an ordered ranking. This reconceptualizes the task.

9. **Efficiency-effectiveness Pareto frontier**: Attention-based methods (ICR) and embedding-based methods (PE-Rank) offer dramatic efficiency gains. Hybrid pipelines combining cheap pre-filtering, attention-based scoring, and targeted LLM reranking for the most ambiguous cases are a promising direction.

10. **Multi-hop and complex queries**: Setwise and listwise approaches struggle with queries requiring synthesis across multiple documents. Set-wise passage selection for multi-hop QA (SetR) is an emerging approach.

---

## 11.5 Connection to This Project (added April 2026)

This review surveys the field; what follows situates this project's contributions and findings against that landscape. Claims and ideas reference the wiki at `research-wiki/`.

### Where this project sits in the taxonomy

This project is **squarely within the setwise paradigm** introduced by Zhuang et al. (SIGIR 2024) (§3). It does not propose a new paradigm; it argues that setwise prompts can be redesigned to extract more information per LLM call. The contribution is **joint elicitation** (claim:C8 in `research-wiki/claims/C8_joint_elicitation_is_contribution.md`), not algorithmic novelty in the sort.

### How this project relates to specific prior work

- **vs. Zhuang et al. 2024 (setwise):** baseline. We extend the prompt to ask for both extremes simultaneously (DualEnd, idea:002), and we extend the live window to the whole rerank pool (MaxContext family, idea:007).
- **vs. TourRank (Chen et al., WWW 2025) and BlitzRank (2026):** parallel "more info per comparison" angle via tournament graphs vs. our joint elicitation angle. TourRank ≈127 LLM calls per query is in the same cost neighborhood as our `DE-Cocktail` (~546 comparisons) and our `MaxContext-DualEnd` at `pool_size=50` (~25 calls). Direct head-to-head comparison is left for future work.
- **vs. Rank-R1 (Zhuang et al. 2025):** Rank-R1 uses `num_child=19` (20-way setwise) as a known-feasible precedent for large-window setwise, validating that current Qwen-class models can compose 20-document prompts coherently. MaxContext idea:007 pushes this to `pool_size=50`.
- **vs. Setwise Insertion (Podolak et al., SIGIR 2025):** orthogonal — Podolak warm-starts the sort using the BM25 ranking as prior; we change the comparator's output. The two ideas can be combined.
- **vs. Lost / Found in the Middle (Liu 2024 / Tang 2024):** our position-bias analysis (claim:C5) shows that joint elicitation flips the worst-selection bias from recency-heavy to primacy-heavy — a `dual_worst` primacy reversal not previously documented.
- **vs. LLM-RankFusion (Zeng et al. 2024):** intrinsic inconsistency results explain why our BiDir ensemble fails (claim:C4): if one of the two rankings has asymmetric bias (BU's recency-heavy `worst→D`), fusion imports that bias rather than averaging it out.
- **vs. Sato (2026 sorting survey):** Sato's framework predicts that double-ended selection is a natural sort for "comparators that emit both a max and a min." We instantiate the comparator (DualEnd) and confirm the sort matches. Sato does not cover dual-output comparators directly.

### Findings the literature does not yet have

1. **Directional asymmetry** (claim:C1): worst-selection alone is unreliable; worst-selection inside a joint best+worst prompt becomes useful. The literature treats best-selection and worst-selection as symmetric variants; we show they are not.
2. **Joint-elicitation shifts position bias** (claim:C5): `dual_worst` primacy reversal vs. `worst` recency bias. Tang (2024) and Liu (2024) document position bias under best-only or read-through prompts; the joint-prompt regime has not been characterized before.
3. **The Pareto frontier between TD-Bubble and DE-Cocktail is empty** (claim:C9). Setwise Insertion claims efficiency gains; TourRank, BlitzRank, Rank-R1 claim quality gains; this is the first explicit characterization of the empty frontier region as a research target.
4. **Conservative ICTIR-first framing** (claim:C10): we treat directional asymmetry as the contribution, with DualEnd as the modest positive method and BottomUp / BiDir as coherent negative results. This is a deliberate framing choice given the statistical fragility of TREC DL on 43/54-query test sets.

### Open questions this project will return to

- **Does the DualEnd directional pattern hold out-of-domain on BEIR?** flan-t5-xl done, qwen3-8b 5/6 done, qwen3.5-9b pending (exp:beir_generalization).
- **Can selective DualEnd, bias-aware DualEnd, or same-call regularized variants land in the Pareto frontier gap?** All three are implemented (idea:004/005/006), most runs pending.
- **Does the MaxContext family beat DE-Cocktail at matched `hits` on the comparisons-axis and wall-clock-axis?** 312-run staged matrix planned (idea:007), not yet launched.
- **Does the `dual_worst` primacy reversal survive controlled re-ordering** (idea:005)? Open empirical question.

---

## 12. Reference List

### Core Setwise Ranking Papers

1. **Zhuang, S., Zhuang, H., Koopman, B., & Zuccon, G.** (2024). A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models. *SIGIR 2024*, 38-47. arXiv:2310.09497.

2. **Podolak, J., Peric, L., Janicijevic, M., & Petcu, R.** (2025). Beyond Reproducibility: Advancing Zero-shot LLM Reranking Efficiency with Setwise Insertion. *SIGIR 2025*. arXiv:2504.10509.

3. **Zhuang, S., Ma, X., Koopman, B., Lin, J., & Zuccon, G.** (2025). Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning. arXiv:2503.06034.

### Foundational LLM Ranking Papers

4. **Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin, D., & Ren, Z.** (2023). Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents. *EMNLP 2023*. arXiv:2304.09542.

5. **Qin, Z., Jagerman, R., Hui, K., Zhuang, H., Wu, J., Yan, L., Shen, J., Liu, T., Liu, J., Metzler, D., Wang, X., & Bendersky, M.** (2024). Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting. *NAACL 2024*. arXiv:2306.17563.

6. **Ma, X., Zhang, X., Pradeep, R., & Lin, J.** (2023). Zero-Shot Listwise Document Reranking with a Large Language Model. arXiv:2305.02156.

7. **Nogueira, R., Jiang, Z., Pradeep, R., & Lin, J.** (2020). Document Ranking with a Pretrained Sequence-to-Sequence Model. *Findings of EMNLP 2020*. arXiv:2003.06713.

8. **Pradeep, R., Nogueira, R., & Lin, J.** (2021). The Expando-Mono-Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence Models. arXiv:2101.05667.

### Position Bias and Inconsistency

9. **Tang, R., Zhang, X., Ma, X., Lin, J., & Ture, F.** (2024). Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models. *NAACL 2024*. arXiv:2310.07712.

10. **Zeng, Y., Tendolkar, O., Baartmans, R., Wu, Q., Chen, L., & Wang, H.** (2024). LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking. *NeurIPS 2024 Workshop*. arXiv:2406.00231.

11. **Zeng, Z., Zhang, D., Li, J., Zou, P., Zhou, Y., & Yang, Y.** (2025). An Empirical Study of Position Bias in Modern Information Retrieval. *Findings of EMNLP 2025*.

12. **Hutter, J., Rau, D., Marx, M., & Kamps, J.** (2025). Lost but Not Only in the Middle: Positional Bias in Retrieval Augmented Generation. *ECIR 2025*, 247-261.

13. **Cuconasu, F., Filice, S., Horowitz, G., Maarek, Y., & Silvestri, F.** (2025). Do RAG Systems Suffer From Positional Bias? arXiv:2505.15561.

### Extensions and Variations

14. **Chen, Y., Liu, Q., Zhang, Y., Sun, W., Ma, X., Yang, W., Shi, D., Mao, J., & Yin, D.** (2025). TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy. *WWW 2025*. arXiv:2406.11678.

15. **Wang, P., Xia, Z., Liao, C., Wang, F., & Liu, H.** (2025). REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking. *EMNLP 2025*. arXiv:2508.18379.

16. **Ren, R., Wang, Y., Zhou, K., Zhao, W.X., Wang, W., Liu, J., Wen, J.-R., & Chua, T.-S.** (2025). Self-Calibrated Listwise Reranking with Large Language Models. *WWW 2025*.

### Reasoning-Enhanced Rerankers

17. **Rank1** (2025). Rank1: Test-Time Compute for Reranking in Information Retrieval. *COLM 2025*. arXiv:2502.18418.

18. **Rank-K** (2025). Rank-K: Test-Time Reasoning for Listwise Reranking. arXiv:2505.14432.

19. **ReasonRank** (2025). ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability. arXiv:2508.07050.

20. **REARANK** (2025). Rearank: Reasoning Re-ranking Agent via Reinforcement Learning. arXiv:2505.20046.

### Efficiency and Attention-Based Methods

21. **Chen, S., Gutierrez, B.J., & Su, Y.** (2025). Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers. *ICLR 2025*. arXiv:2410.02642.

22. **Gupta, N., You, C., Bhojanapalli, S., Kumar, S., Dhillon, I., & Yu, F.** (2025). Scalable In-context Ranking with Generative Models (BlockRank). *NeurIPS 2025*.

23. **Liu, Q., Wang, B., Wang, N., et al.** (2024). Leveraging Passage Embeddings for Efficient Listwise Reranking with Large Language Models (PE-Rank). arXiv:2406.14848.

24. **Peng, Z., Wei, T.-R., Song, T., & Zhao, Y.** (2025). Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers. *EMNLP 2025 Industry*.

### Open-Source LLM Rerankers

25. **Zhang, X., Hofstatter, S., Lewis, P., Tang, R., & Lin, J.** (2025). Rank-without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models. *ECIR 2025*. arXiv:2312.02969.

26. **Sharifymoghaddam, S., Pradeep, R., Slavescu, A., et al.** (2025). RankLLM: A Python Package for Reranking with LLMs. *SIGIR 2025*. arXiv:2505.19284.

### Frameworks and Surveys

27. **Sato, R.** (2026). A Survey on Sorting with Large Language Models. DPC Technical Report DPC-TR-2026-001.

28. **Rathee, M., MacAvaney, S., & Anand, A.** (2025). Guiding Retrieval Using LLM-Based Listwise Rankers. *ECIR 2025*, 230-246.

29. **Lee, D., Jo, Y., Park, H., & Lee, M.** (2024). Shifting from Ranking to Set Selection for Retrieval Augmented Generation (SetR). arXiv:2507.06838.

### Other Relevant Works

30. **BlitzRank** (2026). BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs. arXiv:2602.05448.

31. **InsertRank** (2025). InsertRank: LLMs can reason over BM25 scores to Improve Listwise Reranking. arXiv:2506.14086.

32. **LLM4Ranking** (2025). LLM4Ranking: An Easy-to-use Framework of Utilizing Large Language Models for Document Reranking. arXiv:2504.07439.

---

*This review was compiled from searches across arXiv, ACL Anthology, ACM Digital Library, Springer, and open web sources, supplemented by analysis of the `ielab/llm-rankers` codebase (https://github.com/ielab/llm-rankers).*
