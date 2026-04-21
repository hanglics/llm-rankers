---
type: paper
node_id: paper:zhang2025_rank_without_gpt
title: "Rank-without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models"
authors: ["Xinyu Zhang", "Sebastian Hofstatter", "Patrick Lewis", "Raphael Tang", "Jimmy Lin"]
year: 2025
venue: "ECIR"
external_ids:
  arxiv: "2312.02969"
  doi: null
  s2: null
tags: [listwise, open-source-llm, reranking]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:40:00+10:00
---

# One-line thesis

Open-source LLMs, distilled appropriately, can replace GPT-3.5/4 in listwise reranking pipelines without substantial quality loss.

## Problem / Gap

RankGPT was GPT-dependent, limiting reproducibility and inference cost control.

## Method

- Distillation from GPT-3.5/4 into Flan-T5 / LLaMA checkpoints.
- Listwise sliding window on the student.

## Key Results

- Competitive with GPT-based listwise on TREC DL and BEIR.

## Assumptions

- Teacher-student quality gap is closeable for listwise ranking.

## Limitations / Failure Modes

- Student inherits teacher bias.

## Reusable Ingredients

- Open-LLM rerankers are viable — motivates this project's Qwen-based evaluation.

## Open Questions

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Related: supports the design choice to evaluate open LLMs (Qwen3 / Qwen3.5) rather than GPT in this project.
