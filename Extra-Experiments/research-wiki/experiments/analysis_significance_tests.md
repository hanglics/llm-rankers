---
type: experiment
node_id: exp:analysis_significance_tests
title: "Analysis — paired approximate randomization + Bonferroni, per family vs best TopDown"
status: completed
config:
  test: "paired two-sided approximate randomization, 100000 samples"
  ci: "paired bootstrap 95% CI, 20000 resamples"
  multiple_testing: "Bonferroni within family across 18 configs"
  comparisons: "best challenger vs best TopDown per (model, dataset)"
tests: ["claim:C1", "claim:C2", "claim:C3", "claim:C4", "claim:C6", "claim:C10"]
source_artifacts: ["../SIGNIFICANCE_TESTS.md", "../SIGNIFICANCE_TESTS.json"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Family-level significance table:

| Family | Mean Δ | Positive deltas | Bonferroni wins | Bonferroni losses |
|---|---:|---:|---:|---:|
| DualEnd | +0.0058 | 14/18 | 1 | 0 |
| BottomUp | −0.0616 | 0/18 | 0 | 6 |
| BiDir | −0.0232 | 3/18 | 0 | 3 |

The only Bonferroni-significant win: `qwen3-4b` DL19, `DE-Selection` 0.7220 vs `TD-Heap` 0.6775, Δ +0.0446, raw p < 0.001, Bonferroni p = 0.010.

Full per-config table lives in [`../SIGNIFICANCE_TESTS.md`](../SIGNIFICANCE_TESTS.md).

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
