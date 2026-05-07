---
type: idea
node_id: idea:005
title: "Bias-aware DualEnd (controlled orderings + majority vote on hard windows)"
stage: proposed
outcome: pending
based_on: ["paper:tang2024_found_in_middle", "paper:qin2024_prp"]
target_gaps: ["gap:G2"]
refines: ["idea:002"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Idea

On windows flagged as hard by the routing gate, run a small set of controlled orderings (base, reversed, shifted) and majority-vote the best/worst labels before reinstating the original order. Exploits the novel `dual_worst` primacy reversal (claim:C5) as a robustness lever.

## Mechanism

- `--direction bias_aware_dualend`.
- Gate strategies: `hybrid` (default), `shortlist`, `uncertain`.
- For each hard window: run the setwise prompt on 3 controlled permutations (base, reversed, shifted by ⌈w/2⌉) and majority-vote labels.
- Only supported with `--method bubblesort` or `--method selection` (must exercise the joint prompt path; `heapsort` is excluded).

## Predicted outcome

- Reduces per-window variance where order matters most.
- Tightens DualEnd's significance, at ~3× cost on the routed subset.

## Current status

**Not submitted.** All 12 planned runs are pending:
- flan-t5-xl: DL19/DL20 × {generation, likelihood} × hybrid (4).
- Qwen3-8B: DL19/DL20 × {generation, likelihood} × hybrid (4).
- Qwen3.5-9B: DL19/DL20 × {generation, likelihood} × hybrid (4).

## Open questions

- Does `dual_worst` primacy reversal survive re-orderings (i.e. is it robust or order-dependent)? Majority vote will reveal this.
- Does majority vote help more on T5 or Qwen?

## Tested by

- exp:bias_aware_dualend_pending

## Related

- Refines idea:002 DualEnd.
- Inspired by paper:tang2024_found_in_middle (permutation self-consistency) and paper:qin2024_prp (two-ordering trick).

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
