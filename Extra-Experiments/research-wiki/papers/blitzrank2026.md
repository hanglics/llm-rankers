---
type: paper
node_id: paper:blitzrank2026
title: "BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs"
authors: []
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2602.05448"
  doi: null
  s2: null
tags: [setwise, tournament, graph-based, efficiency]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:40:00+10:00
---

# One-line thesis

Formalize tournament-graph reranking over setwise LLM comparators; optimize the bracket structure to maximize information extracted per LLM call.

## Problem / Gap

Tournament bracket rerankers (paper:chen2025_tour_rank) are heuristic; no theory for optimal bracket design.

## Method

- Graph-theoretic formulation of tournament structures.
- Analysis of information per comparison under noisy comparators.

## Key Results

- Principled bracket designs outperform heuristic tournaments.
- Efficiency-quality tradeoffs characterized theoretically.

## Assumptions

- Noise model is known or estimable.

## Limitations / Failure Modes

- Theoretical; empirical gains depend on noise-model fidelity.

## Reusable Ingredients

- Information-per-comparison framing for setwise rerankers.

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Directly adjacent: both this project and BlitzRank aim at "more information per setwise comparison". BlitzRank's mechanism is graph-structural (bracket design); DualEnd (idea:002) is prompt-structural (joint best+worst). A future combination is an open question (Rank-R1 is another axis).
