---
type: idea
node_id: idea:004
title: "Selective DualEnd (route DualEnd only to uncertain or shortlisted windows)"
stage: active_partial
outcome: tbd
based_on: ["paper:podolak2025_setwise_insertion"]
target_gaps: ["gap:G1", "gap:G4"]
refines: ["idea:002"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Idea

Run `TD-Bubble` by default. Invoke `DualEnd` only on windows that are (a) inside the top-k shortlist, (b) uncertain (BM25-score-spread below a query-local percentile), or (c) both (hybrid). Goal: DualEnd's quality lift at a fraction of its 7× cost.

## Mechanism

- `--direction selective_dualend`.
- Gate strategies (`--gate_strategy`): `off`, `shortlist`, `uncertain`, `hybrid`.
- `--uncertainty_percentile 0.15` → invoke DualEnd on the tightest 15% of windows (query-local BM25-spread percentile, not a fixed absolute threshold).
- `--method bubblesort` and `--method selection` supported; `heapsort` disables shortlist gating (heap indices are not rank positions).

## Predicted outcome

Selective DualEnd occupies the empty region on the Pareto frontier between `TD-Bubble` (0.6897 mean NDCG, 300 comparisons) and `DE-Cocktail` (0.6962, 546 comparisons).

## Current status

- **Submitted and finished:** `flan-t5-xl` DL19/DL20 × {generation, likelihood} × {hybrid, shortlist-generation} (6 runs).
- **Not submitted:**
  - `flan-t5-xl` shortlist-likelihood × {DL19, DL20} (2 runs).
  - `flan-t5-xl` uncertain × {gen, lik} × {DL19, DL20} (4 runs).
  - `Qwen3-8B` all 12 variants.
  - `Qwen3.5-9B` all 12 variants.

## Open questions

- Does `hybrid` beat `shortlist` alone? (initial `flan-t5-xl` results pending consolidation)
- Does selective DualEnd close the significance gap where vanilla DualEnd only hit 1 Bonferroni-significant win?

## Tested by

- exp:selective_dualend_flan_t5_xl
- exp:selective_dualend_qwen_pending

## Related

- Refines idea:002 DualEnd.
- Inspired by paper:podolak2025_setwise_insertion (warm-start selective comparisons).

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
