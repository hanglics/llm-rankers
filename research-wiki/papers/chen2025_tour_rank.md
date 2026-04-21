---
type: paper
node_id: paper:chen2025_tour_rank
title: "TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy"
authors: ["Yiqun Chen", "Qi Liu", "Yi Zhang", "Weiwei Sun", "Xueguang Ma", "Wei Yang", "Daiting Shi", "Jiaxin Mao", "Dawei Yin"]
year: 2025
venue: "WWW"
external_ids:
  arxiv: "2406.11678"
  doi: null
  s2: null
tags: [setwise, tournament, llm-reranking]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:40:00+10:00
---

# One-line thesis

Run a tournament bracket over setwise comparisons — each round advances winners and eliminates losers — to build a full ranking with fewer wasted comparisons than linear sort scaffolds.

## Problem / Gap

Setwise-as-classical-sort (heapsort / bubblesort) does not exploit tournament structure; many comparisons decide already-decided orderings.

## Method

- Tournament bracket over candidate pool with setwise "most relevant" comparisons.
- Multi-round structure with re-seeding between rounds.

## Key Results

- Competitive or better NDCG than setwise baselines on TREC DL and BEIR at lower comparison counts.

## Assumptions

- Setwise comparator is roughly consistent within a round.

## Limitations / Failure Modes

- Same position-bias exposure as any setwise comparator.
- Tournament loses information about losers; top-k stable but full ranking imprecise.

## Reusable Ingredients

- Tournament bracket scaffold — alternative to heapsort/bubblesort/cocktail-shaker.

## Open Questions

- Does TourRank stack with DualEnd's joint elicitation (dual-output per bracket match)?

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Parallel "more information per comparison" research thread: tournament-bracket structure extracts information via round elimination, whereas DualEnd (idea:002) extracts via joint best+worst per prompt. Cited as the closest contemporaneous alternative strategy.
