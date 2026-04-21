---
type: claim
node_id: claim:C6
statement: "DualEnd gains on TREC DL are directionally consistent (14/18 positive) but statistically fragile (only 1 Bonferroni-significant win); BottomUp and BiDir losses are more robust (6 and 3 Bonferroni-sig losses respectively)."
status: strongly_supported
evidence_strength: very_high
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Claim

With TREC DL's small query sets (43 / 54 queries), paired approximate randomization with Bonferroni correction within each challenger family gives:

| Family | Mean Δ vs best TD | Positive deltas | Bonferroni wins | Bonferroni losses |
|---|---:|---:|---:|---:|
| DualEnd | +0.0058 | 14/18 | 1 | 0 |
| BottomUp | −0.0616 | 0/18 | 0 | 6 |
| BiDir | −0.0232 | 3/18 | 0 | 3 |

The only Bonferroni-significant DualEnd win is `qwen3-4b` DL19, `DE-Selection` 0.7220 vs `TD-Heap` 0.6775, Δ +0.0446 (raw p < 0.001, Bonferroni p = 0.010).

Bootstrap 95% CIs for most DualEnd deltas cross zero (e.g. flan-t5-xxl DL19 DE-Cocktail Δ +0.0060, CI [−0.0081, +0.0208]).

## Implication for paper framing

Describe DualEnd gains as a **robust empirical pattern** (directional in 14/18 configs) rather than a universally significant improvement. Highlight the single corrected win and report confidence intervals.

## Supporting experiments

- exp:analysis_significance_tests

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
