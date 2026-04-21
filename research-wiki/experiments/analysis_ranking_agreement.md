---
type: experiment
node_id: exp:analysis_ranking_agreement
title: "Analysis — ranking agreement (Overlap@10, Kendall τ) between method pairs"
status: completed
tests: ["claim:C4", "claim:C8"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Values below are representative-model numbers (the narrative table in `NARRATIVE_REPORT.md:233` labels them as such). Cross-model ranges differ: `FINDINGS.md:159` reports broader approximate bands. Regenerate aggregate statistics before citing publicly.

| Pair | Overlap@10 (repr.) | Kendall τ (repr.) |
|---|---:|---:|
| TopDown ↔ DualEnd | 7.01 | 0.9254 |
| TopDown ↔ BottomUp | 5.04 | 0.8589 |

Explains why DualEnd fusion-in-prompt works while BiDir fusion-after-the-fact does not (claim:C4): TD and BU agree too little and BU is too biased to provide complementary signal.

## Caveats

- Numbers are for a single representative (model, dataset) combination, not the cross-model mean.
- Cross-model aggregation is a pending consolidation task (see `exp:same_method_tables_pending`).

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
