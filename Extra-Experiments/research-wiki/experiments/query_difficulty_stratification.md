---
type: experiment
node_id: exp:query_difficulty_stratification
title: "Analysis — query-difficulty terciles (BM25-based) × DualEnd Δ"
status: partial
config:
  difficulty_bucketing: "BM25 NDCG@10 terciles per dataset"
  models_measured: "subset of main-sweep models"
  source_artifacts: ["../FINDINGS.md (lines 149-154)"]
tests: ["claim:C2"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-30T20:10:00+10:00
---

# Summary

Stratifies per-query DualEnd vs TopDown delta by query difficulty (easy / medium / hard terciles computed on BM25 NDCG@10 per dataset). The consolidated table is now present in [`NARRATIVE.md`](../NARRATIVE.md) and the source note is preserved in [`FINDINGS.md`](../FINDINGS.md).

| Tercile | BU - TD | DE - TD |
|---|---:|---:|
| Easy | -0.0904 | +0.0155 |
| Medium | -0.0803 | +0.0157 |
| Hard | -0.0897 | +0.0021 |
| Overall | -0.0866 | +0.0111 |

Interpretation: DualEnd helps most on easy and medium queries on average, while the gain shrinks on hard queries. BottomUp hurts across all terciles. The direction still varies by model, so this should be framed as a trend rather than a universal rule.

## Status

Partially consolidated. The aggregate table is in the wiki; per-model / per-dataset tercile tables should still be regenerated before final public citation.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
