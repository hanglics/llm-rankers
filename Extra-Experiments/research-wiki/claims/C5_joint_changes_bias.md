---
type: claim
node_id: claim:C5
statement: "Joint best+worst prompting changes the positional bias profile: dual_best is flatter than single-best, and dual_worst shows a reversed primacy bias (opposite of standalone worst-selection)."
status: strongly_supported
evidence_strength: very_high
novelty: true
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Claim

Under setwise prompts with windows of 4 (A/B/C/D), position frequency profiles differ sharply by direction:

| Prompt mode | Position A freq | Position D freq | Shape |
|---|---:|---:|---|
| TopDown (best alone) | 0.23–0.50 | 0.25–0.55 | U-shape (primacy + recency) |
| BottomUp (worst alone) | 0.06–0.24 | 0.40–0.63 | Overwhelming recency |
| DualEnd `dual_best` | ~flat | 0.27–0.48 | Flatter than single-best |
| DualEnd `dual_worst` | 0.23–0.43 | 0.10–0.19 | **Primacy reversal** — novel |

## Novelty

The `dual_worst` primacy reversal is a new observation. Prior work on positional bias in LLM ranking (paper:liu2024_lost_in_middle, paper:tang2024_found_in_middle, paper:hutter2025_positional_rag) studied single-objective prompts only.

## Supporting experiments

- exp:analysis_position_bias (9 models × 2 datasets, chi-squared goodness-of-fit tests)

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
