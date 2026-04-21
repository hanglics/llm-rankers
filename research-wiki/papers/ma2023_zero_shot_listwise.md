---
type: paper
node_id: paper:ma2023_zero_shot_listwise
title: "Zero-Shot Listwise Document Reranking with a Large Language Model"
authors: ["Xueguang Ma", "Xinyu Zhang", "Ronak Pradeep", "Jimmy Lin"]
year: 2023
venue: arXiv
external_ids:
  arxiv: "2305.02156"
  doi: null
  s2: null
tags: [listwise, zero-shot-ranking]
relevance: peripheral
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

Listwise zero-shot reranking works on open-source LLMs (not just GPT) with appropriate prompt design and sliding-window scaffolding.

## Problem / Gap

RankGPT relied on ChatGPT; open-source viability was unclear.

## Method

- Listwise sliding window on Flan-T5 / LLaMA variants.

## Key Results

- Competitive with supervised rerankers on BEIR for some domains.

## Assumptions

## Limitations / Failure Modes

- Same output-parsing fragility as RankGPT.

## Reusable Ingredients

- Evidence that listwise works beyond GPT; frames why setwise (zhuang2024_setwise) was a necessary efficiency counter.

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Peripheral; out of scope (listwise). Included for completeness as the concurrent listwise-on-open-LLMs reference point.
