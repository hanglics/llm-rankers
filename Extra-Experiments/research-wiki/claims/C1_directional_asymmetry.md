---
type: claim
node_id: claim:C1
statement: "Setwise LLM ranking is directionally asymmetric: standalone worst-selection is unreliable and heavily biased, whereas worst-selection inside a joint best+worst prompt behaves qualitatively differently and can improve ranking quality."
status: supported
evidence_strength: high
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Claim

Ranking direction matters: "which is best?" is not the mirror of "which is worst?" for LLM comparators. The asymmetry is not a capacity issue — it persists across Flan-T5 {large, xl, xxl} and Qwen {3, 3.5} {4B, 8/9B, 14/27B}.

## Evidence

- **BU catastrophic on small T5:** `flan-t5-large` DL19 BU-Bubble 0.4571 vs TD-Bubble 0.6874, Δ −0.2302 (Bonferroni sig loss).
- **BU consistently weaker across all 18 configs:** family mean Δ = −0.0616; 6 Bonferroni-sig losses; 0 positive deltas.
- **DualEnd wins 14/18 configs:** family mean Δ = +0.0058; 1 Bonferroni-sig win (`qwen3-4b` DL19 DE-Selection Δ +0.0446).
- **Position bias changes direction under joint prompt** (claim:C5):
  - `worst` alone: D-freq 0.40–0.63 (overwhelming recency).
  - `dual_worst`: A-freq 0.23–0.43 (primacy reversal).

## Supporting experiments

- exp:main_bu_heap
- exp:main_bu_bubble
- exp:main_de_cocktail
- exp:main_de_selection
- exp:analysis_significance_tests
- exp:analysis_position_bias

## Status history

- 2026-04-20: **supported** by Phase 1–4 experiments (18 model–dataset configs complete; significance tests done).

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
