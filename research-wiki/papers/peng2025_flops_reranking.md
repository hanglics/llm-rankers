---
type: paper
node_id: paper:peng2025_flops_reranking
title: "Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers"
authors: ["Zhiyuan Peng", "Ting-Ruen Wei", "Tianming Song", "Yi Zhao"]
year: 2025
venue: "EMNLP Industry"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [efficiency, flops, hardware-independent-metrics]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

Hardware-independent metrics (rerank-FLOPs-per-passage, per-query) are needed to compare LLM rerankers because wall-clock time depends on GPU, batching, and kernel choice.

## Problem / Gap

Efficiency comparisons across LLM rerankers are non-reproducible when reported in wall-clock seconds alone.

## Method

- Define RPP (rerank FLOPs per passage), QPP (per query).
- Derive closed-form estimates from model size × sequence length × comparisons.

## Key Results

- FLOPs-based metrics correlate with but are more robust than wall-clock comparisons.

## Assumptions

- FLOPs proxy for real cost modulo hardware.

## Limitations / Failure Modes

- Ignores memory-bound regimes and KV-cache effects.

## Reusable Ingredients

- RPP / QPP as framing for our Pareto plots.

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Reference for our quality-cost Pareto analysis (exp:analysis_pareto). We report comparisons, tokens, and wall-clock — all three — to stay comparable with this paper's guidance.
