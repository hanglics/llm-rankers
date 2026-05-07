---
type: paper
node_id: paper:sun2023_rankgpt
title: "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents"
authors: ["Weiwei Sun", "Lingyong Yan", "Xinyu Ma", "Shuaiqiang Wang", "Pengjie Ren", "Zhumin Chen", "Dawei Yin", "Zhaochun Ren"]
year: 2023
venue: EMNLP
external_ids:
  arxiv: "2304.09542"
  doi: null
  s2: null
tags: [listwise, llm-reranking, sliding-window]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

ChatGPT can act as a zero-shot passage reranker when the candidate list is fed as a listwise prompt processed by a sliding window over the initial BM25 ranking.

## Problem / Gap

Before RankGPT, LLMs were used mostly pointwise (expensive per-doc scoring). No robust way to produce a full ranking zero-shot beyond score-aggregation tricks.

## Method

- **Listwise prompt:** show numbered passages, ask the LLM to produce the re-ranked permutation.
- **Sliding window:** process window-by-window over the candidate list, carrying the best candidates forward across passes; `num_repeat` controls number of full sweeps.
- **Distillation recipe:** use ChatGPT outputs to supervise a smaller student ranker.

## Key Results

- ChatGPT-listwise matches or exceeds supervised rerankers on TREC DL and BEIR.
- Distillation into a 440M student produces competitive zero-shot rerankers.

## Assumptions

- The LLM can emit a consistent permutation of window-size candidates in a single response.
- Sliding window preserves top-k under imperfect local rankings.

## Limitations / Failure Modes

- Output parsing is fragile (hallucinated IDs, missing labels).
- Position bias on long context is significant ("lost in the middle" effect compounds in wide windows).
- Cost per window is large because the model has to emit a full permutation, not a single label.

## Reusable Ingredients

- Sliding window over BM25 top-k as a universal reranking scaffold.
- The "distill from a large listwise teacher" recipe for smaller student rankers.

## Open Questions

- Does a smaller set + single-label setwise (zhuang2024_setwise) dominate listwise at the same effectiveness?
- Is listwise position bias worse than setwise position bias?

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Motivates the setwise paradigm by contrast: listwise requires the LLM to emit a full permutation per window; setwise asks only for one label. Our project stays strictly inside the setwise regime and does not compare against listwise methods.
