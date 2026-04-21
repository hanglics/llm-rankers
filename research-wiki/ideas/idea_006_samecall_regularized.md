---
type: idea
node_id: idea:006
title: "Same-call regularized DualEnd (worst as local demotion only)"
stage: proposed
outcome: pending
based_on: ["paper:zhuang2024_setwise"]
target_gaps: ["gap:G1", "gap:G3"]
refines: ["idea:002"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Idea

Keep `TopDown`'s head-focused structure. Use the worst output from the joint best+worst prompt **only** as a local demotion constraint — and only when the candidate is *outside* the protected ranking head frontier (top-k plus one active window).

## Mechanism

- `--direction samecall_regularized`.
- Inside the active window, do TopDown as usual to pick the best; also read `Worst: Y` from the same prompt.
- Only if Y is outside the top-k protected frontier, demote Y within the current window. No backward bubbling pass.
- Preserves TopDown's head focus. Adds minimal cost: the joint prompt is already paid for; the demotion is a local swap.

## Predicted outcome

- Modest quality lift at near-zero added cost relative to the already-paid joint prompt.
- Documents whether worst signal helps as a *regularizer* (ambiguous-tail constraint) vs as a *ranking source* (failed, idea:003).

## Current status

**Not submitted.** 12 planned runs pending:
- flan-t5-xl: DL19/DL20 × {generation, likelihood} (4).
- Qwen3-8B: DL19/DL20 × {generation, likelihood} (4).
- Qwen3.5-9B: DL19/DL20 × {generation, likelihood} (4).

## Design notes

- The worst-signal demotion is applied **only when the worst candidate is already outside the protected frontier** (README line 426 clarifies this invariant). This avoids poisoning the head with noisy worst calls.
- Under `--scoring likelihood` for causal models, the joint path is a best-only proxy (same caveat as idea:002); the worst-signal regularization degenerates to a heuristic there.

## Open questions

- Is the quality lift over vanilla DualEnd large enough to justify the implementation complexity?
- Does the regularization help more on Qwen (where DualEnd already wins) or T5 (where TD-Bubble is competitive)?

## Tested by

- exp:samecall_regularized_pending

## Related

- Refines idea:002 DualEnd.
- Inspired by paper:zhuang2024_setwise (TopDown head focus).

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
