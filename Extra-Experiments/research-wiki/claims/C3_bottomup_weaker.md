---
type: claim
node_id: claim:C3
statement: "BottomUp is consistently weaker than TopDown across every one of 18 model-dataset configurations."
status: strongly_supported
evidence_strength: very_high
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Claim

Across all 9 models × 2 datasets, the best BottomUp variant never beats the best TopDown variant; the loss is Bonferroni-significant in 6 configs.

## Evidence

- 0 / 18 positive deltas; 6 Bonferroni-sig losses.
- Family mean Δ = −0.0616.
- Range: flan-t5-large DL19 (Δ −0.2302) ↔ qwen3.5-27b DL19 (Δ −0.0113).

## Supporting experiments

- exp:main_bu_heap
- exp:main_bu_bubble
- exp:analysis_significance_tests

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
