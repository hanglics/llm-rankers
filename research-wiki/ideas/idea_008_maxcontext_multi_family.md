---
type: idea
node_id: idea:008
title: "MaxContext multi-family extension for EMNLP 2026"
stage: active
outcome: pending
id: idea:008
refines: ["idea:007"]
addresses_gap: ["gap:G1", "gap:G4", "gap:G5"]
tested_by: ["exp:emnlp_phase_b_main_matrix", "exp:emnlp_phase_c_stability", "exp:cross_model_stability"]
status: active
target_venue: EMNLP 2026 short paper
origin_skill: manual-backfill
created_at: 2026-05-05T22:15:00+10:00
updated_at: 2026-05-05T22:15:00+10:00
---

# Idea

Extend IDEA_007's MaxContext family from Qwen3/Qwen3.5 to a multi-family EMNLP matrix covering Qwen3.5, Llama-3.1, and Ministral-3. The algorithm is unchanged; only model-family allowlists, launch/evaluation tooling, and analysis tracks are widened.

## Motivation

IDEA_007 is intentionally Qwen-only. That leaves a single-family confound: MaxContext gains could reflect Qwen-specific long-context behavior, chat-template thinking controls, or parser-compatible output habits. IDEA_008 tests whether the same whole-pool prompting idea generalizes across current open model families.

## Mechanism

- Keep the seven EMNLP methods fixed in `EMNLP_PAPER_PLAN.md`.
- Keep MaxContext strict numeric labels and no-truncation invariants.
- Add Llama-3.1 and Ministral-3 support without changing the Qwen2/Qwen3/Qwen3.5 behavior gates.
- Use `EMNLP_EXPERIMENT_PLAN.md` and `EMNLP_BUDGET.md` as operator gates.

## Tested by

- exp:emnlp_phase_a_smoke
- exp:emnlp_phase_b_main_matrix
- exp:emnlp_phase_c_stability
- exp:emnlp_phase_c_prime_constraint_recheck
- exp:cross_model_stability

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
