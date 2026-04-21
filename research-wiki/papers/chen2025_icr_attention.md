---
type: paper
node_id: paper:chen2025_icr_attention
title: "Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers"
authors: ["Shijie Chen", "Bernal Jiménez Gutiérrez", "Yu Su"]
year: 2025
venue: ICLR
external_ids:
  arxiv: "2410.02642"
  doi: null
  s2: null
tags: [attention, zero-shot-ranking, efficiency]
relevance: peripheral
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

Read the LLM's internal attention over query-document tokens to score candidates — no generation required, order-of-magnitude faster than setwise/listwise.

## Problem / Gap

Setwise/listwise rerankers still pay full generation cost per window.

## Method

- Forward pass over [query; docs] once; aggregate attention scores per document.
- No decoding step.

## Key Results

- Matches or exceeds listwise quality at a fraction of the cost.

## Assumptions

- Attention patterns are a reliable proxy for relevance.

## Limitations / Failure Modes

- Requires white-box access to attention weights (API LLMs excluded).
- Orthogonal to our setwise-prompt regime.

## Reusable Ingredients

- The "no generation" efficiency ceiling is a relevant anchor on the Pareto frontier.

## Open Questions

- Can attention-based rerankers match setwise-generation on harder domains (BEIR)?

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Peripheral: a different efficiency axis (no generation) than ours (fewer comparisons / more info per call). Cited as an alternative efficiency strategy.
