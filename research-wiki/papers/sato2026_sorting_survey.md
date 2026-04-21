---
type: paper
node_id: paper:sato2026_sorting_survey
title: "A Survey on Sorting with Large Language Models"
authors: ["Ryoma Sato"]
year: 2026
venue: "DPC Technical Report DPC-TR-2026-001"
external_ids:
  arxiv: null
  doi: null
  s2: null
  tech_report_id: "DPC-TR-2026-001"
tags: [sorting, survey, llm-comparator]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

Systematic survey of sorting algorithms when the comparator is an LLM, cataloguing heapsort, bubblesort, mergesort, quicksort, and insertion sort under noisy-comparator assumptions.

## Problem / Gap

No centralized reference on when classical sorts break under LLM comparators.

## Method

- Taxonomy of algorithms, cost models, noise-tolerance guarantees.
- Discussion of aggregation/fusion as noise mitigation.

## Key Results

- Classical sorting guarantees collapse under non-transitive noisy comparators.
- Heapsort and mergesort tolerate moderate noise better than bubblesort in simulation.

## Assumptions

- Noise is the dominant failure mode (not systematic bias).

## Limitations / Failure Modes

- Does not discuss multi-output comparators (best+worst per call), which is the setting for idea:002.
- Does not discuss reverse-selection or directional asymmetry.

## Reusable Ingredients

- The framing of "setwise-as-sort" and its noise-tolerance tables.

## Open Questions

- How do classical algorithms behave when the comparator returns two decisions per call (best, worst)?
- What is the right sorting primitive for a dual-output comparator — cocktail shaker, double-ended selection, or something else?

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Provides the taxonomic language ("cocktail shaker sort", "double-ended selection sort") we use for `DE-Cocktail` and `DE-Selection`. The survey explicitly does not cover multi-output comparators — exactly the gap idea:002 targets.
