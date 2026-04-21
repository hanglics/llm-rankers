---
type: paper
node_id: paper:zeng2024_llm_rankfusion
title: "LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking"
authors: ["Yifan Zeng", "Ojas Tendolkar", "Raymond Baartmans", "Qingyun Wu", "Lizhong Chen", "Huazheng Wang"]
year: 2024
venue: "NeurIPS Workshop (2024) / arXiv"
external_ids:
  arxiv: "2406.00231"
  doi: null
  s2: null
tags: [inconsistency, non-transitivity, rank-fusion]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

LLM comparators exhibit order inconsistency and transitive inconsistency (A>B, B>C, C>A), which breaks the assumptions of classical sorting algorithms used for LLM reranking.

## Problem / Gap

Setwise and pairwise LLM rerankers wrap classical comparison sorts (heapsort, bubblesort) around LLM comparators, but those algorithms assume transitivity. LLMs don't satisfy it.

## Method

- Diagnose inconsistency rates in pairwise LLM comparisons.
- Aggregate inconsistent comparisons via rank-fusion operators (RRF, CombSUM).

## Key Results

- Non-transitive triads are common across GPT-3.5/4 and open LLMs.
- Aggregation reduces but does not eliminate rank variance.

## Assumptions

- Aggregation helps when the noise is roughly symmetric.
- Rank fusion benefits from diverse input rankings.

## Limitations / Failure Modes

- When one of the ranking sources is systematically biased (not noisy-symmetric), fusion imports the bias (exactly what happens in our idea:003 BiDir experiments).

## Reusable Ingredients

- Framing "LLM rankers are non-transitive comparators" as a first-class phenomenon.
- RRF/CombSUM/weighted fusion operators (we use all three in `bidirectional` direction).

## Open Questions

- When does rank fusion help vs hurt? (partially answered by our BiDir negative result — claim:C4)

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Provides the theoretical background for why `bidirectional` fusion *could* have helped in principle — and why it fails in our case: the BottomUp signal is systematically biased, not symmetrically noisy, so fusion imports the bias (claim:C4).
