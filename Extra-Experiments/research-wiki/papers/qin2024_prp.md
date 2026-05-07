---
type: paper
node_id: paper:qin2024_prp
title: "Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting"
authors: ["Zhen Qin", "Rolf Jagerman", "Kai Hui", "Honglei Zhuang", "Junru Wu", "Le Yan", "Jiaming Shen", "Tianqi Liu", "Jialu Liu", "Donald Metzler", "Xuanhui Wang", "Michael Bendersky"]
year: 2024
venue: "NAACL 2024 (Findings)"
external_ids:
  arxiv: "2306.17563"
  doi: null
  s2: null
tags: [pairwise, llm-reranking, position-bias]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

Pairwise prompting — ask the LLM "which of A, B is more relevant" and aggregate pairwise preferences — is an effective zero-shot reranker, but has significant positional bias and O(n²) cost.

## Problem / Gap

Pointwise LLM ranking wastes relative information; listwise struggles to fit all docs. Pairwise sits in the middle: simple prompt, strong signal, but expensive and biased.

## Method

- **Pairwise prompt:** "which is more relevant: A or B?" (two orderings per pair for bias mitigation).
- **Aggregation:** score each document by its win count; alternatively use pairwise-driven heapsort or bubblesort.
- **Analysis:** two-sided (`A vs B` and `B vs A`) to diagnose positional bias.

## Key Results

- +4.2% NDCG@10 over ChatGPT-listwise on DL19; +10% over pointwise.
- Strong positional bias in pairwise: most models prefer the "first" option.

## Assumptions

- Pairwise preferences are informative despite noise.
- Bias can be averaged out with two orderings.

## Limitations / Failure Modes

- O(n²) cost for "all-pairs" variant.
- Positional bias even after two-ordering trick (bias is asymmetric across models).
- Sorting-based variants inherit the comparator's non-transitivity.

## Reusable Ingredients

- Two-ordering bias-mitigation trick (exploited by idea:005 `bias-aware DualEnd`).
- Pairwise heapsort/bubblesort as baselines.

## Open Questions

- Does position bias remain under setwise framing with more than two candidates? (→ project's position bias analysis)
- Can the bias be exploited rather than merely cancelled?

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Pairwise is the efficiency-worse neighbour of setwise on the Pareto frontier. PRP's positional-bias analysis is a direct precedent for our multi-model position-bias study (exp:analysis_position_bias) and for `bias_aware_dualend` (idea:005).
