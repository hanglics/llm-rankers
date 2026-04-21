---
type: paper
node_id: paper:zhuang2025_rank_r1
title: "Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning"
authors: ["Shengyao Zhuang", "Xueguang Ma", "Bevan Koopman", "Jimmy Lin", "Guido Zuccon"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2503.06034"
  doi: null
  s2: null
tags: [setwise, reasoning, rlhf, test-time-compute]
relevance: related
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# One-line thesis

RL-tune a setwise reranker to emit a chain-of-thought before picking the most relevant document in a 20-way setwise prompt, matching SFT with ~18% of the data.

## Problem / Gap

Setwise rerankers are zero-shot and have no way to benefit from test-time reasoning.

## Method

- Large setwise windows (num_child=19, i.e. 20 candidates per call).
- RL training (RLAIF-style) on a small supervised dataset to teach chain-of-thought-then-label behaviour.

## Key Results

- Matches supervised fine-tuning with a fraction of the training data.
- Thinking-enabled path improves quality at material inference cost.

## Assumptions

- Reasoning chains help on harder windows.
- The final label is faithful to the reasoning.

## Limitations / Failure Modes

- Thinking mode is expensive at inference; often disabled for practical rerankers (as in the llm-rankers README).
- Orthogonal to the "more information per call" axis.

## Reusable Ingredients

- Large-window (20-way) setwise prompting as a viable regime.
- The "reasoning-then-label" decoding path.

## Open Questions

- Can reasoning improve worst selection (which our project finds unreliable)?
- Is there synergy between reasoning and joint best+worst prompting?

## Claims

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->

## Relevance to This Project

Sister paper in the same lab; a future direction we explicitly do *not* explore (reasoning axis). Our "Rank-R1" directory in the repo is a separate experimental workspace for reasoning rerankers; this wiki focuses on the dual-end / bidirectional axis.
