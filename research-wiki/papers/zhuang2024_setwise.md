---
type: paper
node_id: paper:zhuang2024_setwise
title: "A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models"
authors: ["Shengyao Zhuang", "Honglei Zhuang", "Bevan Koopman", "Guido Zuccon"]
year: 2024
venue: SIGIR
external_ids:
  arxiv: "2310.09497"
  doi: null
  s2: null
tags: [setwise, zero-shot-ranking, llm-reranking, foundational]
relevance: core
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

Present small candidate sets to an LLM, ask "which is most relevant", and use heapsort/bubblesort over these comparisons to produce a full ranking — trading token count for quality between pointwise and listwise paradigms.

## Problem / Gap

Prior LLM reranking was split between pointwise (fast, weak), pairwise (strong, expensive: O(n^2) calls), and listwise (RankGPT: hard to fit all documents in context, sensitive to sliding window). No approach sat on the Pareto frontier between effectiveness and efficiency for zero-shot LLMs.

## Method

- **Setwise prompt:** show a small window of `num_child+1` passages (A, B, C, D…); ask the LLM to emit the label of the most relevant.
- **Scoring:** `generation` (sample decoded label) or `likelihood` (score label continuation with teacher forcing).
- **Sorting:** wrap the comparator with classical heapsort (O(n log n)) or bubblesort (O(n^2) adjacent pairs) to produce a full ranking, terminating early once the top-k is stable.
- **Implementation:** Flan-T5 likelihood scoring, open-source GPT/Qwen generation scoring, optional multi-GPU.

## Key Results

- Flan-T5-large TD-Heapsort DL19 ≈ NDCG@10 0.6697; setwise strictly dominates pointwise yes_no (0.6544) with fewer comparisons than pairwise heapsort (0.6571) at the same quality tier.
- Setwise pushes the effectiveness/efficiency Pareto frontier; likelihood scoring on T5 is strictly cheaper than generation for identical quality.

## Assumptions

- LLM comparators are approximately transitive and consistent across permutations (a comparison-sort assumption).
- The "most relevant" question is well posed for small windows; position bias is a secondary concern.
- Top-k is stable under early termination of classical sorts.

## Limitations / Failure Modes

- Comparators are known to be non-transitive in practice — violates sorting-algorithm correctness assumptions.
- Only one decision (identity of best) is extracted per LLM call; implicit knowledge about worst and relative order of the rest is discarded.
- Position bias on small windows (A/B/C/D) is not systematically studied.

## Reusable Ingredients

- The setwise prompt template (labels A, B, C, …).
- `generation` vs `likelihood` dual scoring path (Flan-T5 uses likelihood by default).
- Sorting-as-rerank wrapper (`heapsort` / `bubblesort` over a pluggable comparator).
- Early termination at top-k.

## Open Questions

- Can more than one ranking decision be extracted per comparison? (→ idea:002)
- Does reversed framing ("which is LEAST relevant") give a complementary signal? (→ idea:001, idea:003)
- How does position bias on labels A/B/C/D affect the derived ranking?

## Claims

- claim:C1 (directional asymmetry diagnosed relative to this baseline)
- claim:C2 (DualEnd family vs this baseline)
- claim:C9 (Pareto frontier references TD-Heap and TD-Bubble from this paper)

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Direct foundation. Every method in this repo (`llmrankers/setwise.py`, `llmrankers/setwise_extended.py`) extends the setwise prompt, sorts, and scoring paths introduced here. The project's core question — "can we extract more than one decision per setwise call" — is posed against this baseline.
