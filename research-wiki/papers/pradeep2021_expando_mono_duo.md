---
type: paper
node_id: paper:pradeep2021_expando_mono_duo
title: "The Expando-Mono-Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence Models"
authors: ["Ronak Pradeep", "Rodrigo Nogueira", "Jimmy Lin"]
year: 2021
venue: "arXiv"
external_ids:
  arxiv: "2101.05667"
  doi: null
  s2: null
tags: [pipeline, pointwise-then-pairwise, supervised, t5, baseline]
relevance: peripheral
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:40:00+10:00
---

# One-line thesis

A three-stage text-ranking pipeline (document expansion → pointwise MonoT5 → pairwise DuoT5) that establishes a clean decomposition for supervised T5 rerankers.

## Problem / Gap

Pointwise rerankers saturate; pairwise rerankers are expensive; no clean design pattern combining the two.

## Method

- **Expando:** expand documents with predicted queries to improve retrieval.
- **Mono:** MonoT5 pointwise reranking over top-k.
- **Duo:** DuoT5 pairwise reranking on the top-k-of-k after MonoT5.

## Key Results

- +2–4% NDCG over MonoT5 alone with limited DuoT5 application.
- Clean cost-shifting pattern: cheap pointwise prunes, expensive pairwise refines.

## Assumptions

- Supervised T5 training data is available.
- Pipeline staging is monotonically better than single-stage alternatives.

## Limitations / Failure Modes

- Still supervised; not zero-shot.
- O(k²) pairwise at the final stage if not carefully pruned.

## Reusable Ingredients

- The "cheap filter + expensive refine" pattern underlies idea:004 selective_dualend (TD-Bubble default; DualEnd only on shortlist / uncertain windows).
- DuoT5 model checkpoints referenced in llm-rankers README line 246 as pairwise baselines.

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Peripheral; supervised pipeline. Cited because selective activation of expensive rerankers (idea:004) shares its staged-refinement pattern.
