---
type: experiment
node_id: exp:analysis_ranking_agreement
title: "Analysis — ranking agreement (Overlap@10, Kendall τ) between method pairs"
status: completed
tests: ["claim:C4", "claim:C8"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-30T20:10:00+10:00
---

# Summary

Values below are the consolidated averages reported in [`NARRATIVE.md`](../NARRATIVE.md) and [`FINDINGS.md`](../FINDINGS.md). Regenerate from raw run artifacts before final camera-ready citation, but the current wiki no longer treats the agreement table as missing.

| Pair | Overlap@10 | Kendall τ | Agreement |
|---|---:|---:|
| TopDown ↔ DualEnd | 7.01 | 0.9254 | high |
| TopDown ↔ BottomUp | 5.04 | 0.8589 | high |
| BottomUp ↔ DualEnd | 5.24 | 0.8767 | high |

Explains why DualEnd fusion-in-prompt works while BiDir fusion-after-the-fact does not (claim:C4): TD and BU agree too little and BU is too biased to provide complementary signal.

## Caveats

- The table is sufficient for current narrative planning.
- Final paper numbers should still be regenerated directly from artifacts to avoid citing a copied summary.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
