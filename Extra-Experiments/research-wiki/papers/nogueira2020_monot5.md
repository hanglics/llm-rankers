---
type: paper
node_id: paper:nogueira2020_monot5
title: "Document Ranking with a Pretrained Sequence-to-Sequence Model"
authors: ["Rodrigo Nogueira", "Zhiying Jiang", "Ronak Pradeep", "Jimmy Lin"]
year: 2020
venue: "EMNLP (Findings)"
external_ids:
  arxiv: "2003.06713"
  doi: null
  s2: null
tags: [pointwise, supervised, t5, monoT5, baseline]
relevance: peripheral
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:40:00+10:00
---

# One-line thesis

MonoT5 — fine-tune a sequence-to-sequence model (T5) to emit "true"/"false" relevance tokens for a (query, passage) pair, producing a strong supervised pointwise reranker.

## Problem / Gap

Cross-encoder rerankers were BERT-heavy; T5 had not yet been shown as a strong reranker.

## Method

- Fine-tune T5 on MS MARCO with (query, passage, label) triples.
- Relevance score = P("true") / (P("true") + P("false")).

## Key Results

- Strong supervised baseline on MS MARCO, TREC DL, BEIR.
- Available as `castorini/monot5-*` on HuggingFace; integrated into `llm-rankers/run.py` pointwise mode.

## Assumptions

- Supervised training data is available and representative.

## Limitations / Failure Modes

- Supervised — requires labeled data; out-of-domain degradation without retraining.
- Pointwise — no relative-relevance modeling.

## Reusable Ingredients

- The T5 as scorer pattern, directly reused in Flan-T5 likelihood-scoring in this project.
- The HuggingFace checkpoints (`castorini/monot5-*`) plug into `llm-rankers/run.py` pointwise mode.

## Open Questions

- Does fine-tuning help for setwise LLM ranking, or does zero-shot with larger LLMs dominate? (answered in favor of zero-shot-LLM by paper:zhuang2024_setwise for modern scales)

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Peripheral; out-of-scope (supervised, pointwise). Included because it is the supervised-T5 baseline referenced in llm-rankers README line 128 and it is the closest pre-LLM-era comparison point. We do not run it in this project but cite it for completeness.
