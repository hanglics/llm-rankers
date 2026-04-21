---
type: paper
node_id: paper:ren2025_self_calibrated_listwise
title: "Self-Calibrated Listwise Reranking with Large Language Models"
authors: ["Ruiyang Ren", "Yuhao Wang", "Kun Zhou", "Wayne Xin Zhao", "Wenjie Wang", "Jing Liu", "Ji-Rong Wen", "Tat-Seng Chua"]
year: 2025
venue: "WWW"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [listwise, calibration, llm-reranking]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:40:00+10:00
---

# One-line thesis

Cross-window calibration for listwise sliding-window rerankers — aligns scores between adjacent windows so top-k from one window can be compared to another.

## Problem / Gap

RankGPT-style sliding window produces per-window rankings; gluing them into a global ranking is heuristic.

## Method

- Global calibration over window outputs via self-consistency signals.

## Key Results

- Improves global NDCG over vanilla sliding window.

## Assumptions

- Window-local rankings are internally consistent.

## Limitations / Failure Modes

- Listwise scope; does not address within-window best-selection bias.

## Reusable Ingredients

- Calibration pattern for multi-window rerankers.

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Peripheral. Included because listwise-with-calibration is the analog to our "route DualEnd to hard windows" pattern (idea:004) — both try to use expensive signal only where it helps.
