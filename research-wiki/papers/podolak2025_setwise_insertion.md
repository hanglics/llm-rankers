---
type: paper
node_id: paper:podolak2025_setwise_insertion
title: "Beyond Reproducibility: Advancing Zero-shot LLM Reranking Efficiency with Setwise Insertion"
authors: ["Jakub Podolak", "Leon Peric", "Mina Janicijevic", "Roxana Petcu"]
year: 2025
venue: SIGIR
external_ids:
  arxiv: "2504.10509"
  doi: null
  s2: null
tags: [setwise, efficiency, insertion-sort]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

Treat the initial BM25 ranking as a warm-start and use insertion-sort-style setwise comparisons to skip work the classical heapsort/bubblesort wrappers would do redundantly.

## Problem / Gap

Setwise with heapsort/bubblesort ignores the signal already in the first-stage retriever's ranking.

## Method

- Insertion-sort scaffold over a setwise comparator, seeded by the BM25 order.
- Skip comparisons the warm-start already implies.

## Key Results

- ~31% wall-clock reduction and ~23% fewer inferences vs setwise heapsort, with small effectiveness gains.

## Assumptions

- The first-stage ranking is informative at the top.
- Skipping comparisons does not compound into significant top-k errors.

## Limitations / Failure Modes

- Depends heavily on first-stage quality.
- Does not attempt to extract more information per call (orthogonal to our project).

## Reusable Ingredients

- The "warm-start + cheap passes" framing, which inspires `selective_dualend` (idea:004) focusing expensive prompts on uncertain windows only.

## Open Questions

- Can warm-start + DualEnd combine productively (selective DualEnd with warm-start scaffold)?

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

An efficiency-oriented extension of setwise that is orthogonal to our information-extraction angle. Cited as related work and as a precedent for selective/warm-start designs.
