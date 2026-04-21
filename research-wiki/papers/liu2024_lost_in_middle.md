---
type: paper
node_id: paper:liu2024_lost_in_middle
title: "Lost in the Middle: How Language Models Use Long Contexts"
authors: ["Nelson F. Liu", "Kevin Lin", "John Hewitt", "Ashwin Paranjape", "Michele Bevilacqua", "Fabio Petroni", "Percy Liang"]
year: 2024
venue: "TACL"
external_ids:
  arxiv: "2307.03172"
  doi: null
  s2: null
tags: [position-bias, long-context, foundational-bias]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

LLMs disproportionately attend to the beginning and end of their input context and systematically underweight middle positions — the "lost in the middle" phenomenon.

## Problem / Gap

Prior work assumed uniform attention over long contexts for retrieval-augmented generation. The paper shows this is empirically false across model families.

## Method

- Controlled experiments on open-book QA with evidence placed at different positions in the context window.
- Accuracy as a function of evidence position forms a U-curve.

## Key Results

- U-shape accuracy: highest when evidence is at the beginning or end; lowest in the middle.
- Effect persists across model sizes and context lengths.

## Assumptions

- QA accuracy is a faithful proxy for attention to the evidence.

## Limitations / Failure Modes

- Does not disaggregate "primacy vs recency" (we observe in our project that these can differ sharply, see claim:C5).
- Reports on best-only tasks; "worst" selection was not studied.

## Reusable Ingredients

- The U-shape position-accuracy curve as a diagnostic plot style (used in our position-bias analysis notebooks).

## Open Questions

- Does the bias shape change when the task asks for the worst rather than the best? (answered by our project: yes — it shifts to recency-heavy for BU and to primacy-heavy for dual_worst)

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Foundational reference for the position-bias analysis. Our result that `dual_worst` shows reversed primacy bias (claim:C5) extends Liu et al.'s U-shape finding to the multi-objective (best+worst) setwise prompting regime.
