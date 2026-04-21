---
type: paper
node_id: paper:hutter2025_positional_rag
title: "Lost but Not Only in the Middle: Positional Bias in Retrieval Augmented Generation"
authors: ["Julia Hutter", "David Rau", "Maarten Marx", "Jaap Kamps"]
year: 2025
venue: ECIR
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [position-bias, rag, model-family-effects]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

Position bias in RAG is not a universal U-shape — it depends on model family, document length, and retrieval domain.

## Problem / Gap

"Lost in the middle" had been generalized too broadly; practitioners assumed the U-curve was universal.

## Method

- Controlled placement experiments across multiple model families (GPT, Mistral, Llama) and domains.

## Key Results

- Some models show primacy-only, some show recency-only, some show U-shape.
- Document length modulates the curve.

## Assumptions

## Limitations / Failure Modes

- Studies generation, not explicit ranking decisions.

## Reusable Ingredients

- Empirical framing: expect model-family-dependent bias patterns (supports our claim:C7 about window size × model family).

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Supports our finding that position bias under setwise prompts is direction- and model-family-dependent (claim:C5, claim:C7). In our project, T5 and Qwen differ in bias profile; Hutter et al. would predict this.
