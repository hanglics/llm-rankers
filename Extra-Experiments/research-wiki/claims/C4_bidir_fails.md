---
type: claim
node_id: claim:C4
statement: "Bidirectional ensemble (independent TD + BU fusion) does not beat TopDown because the BU input is systematically biased rather than symmetrically noisy; rank fusion imports the bias."
status: strongly_supported
evidence_strength: very_high
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Claim

BiDir fails in aggregate. The best fusion weight is α=0.9 (90% TD), and even there it does not beat pure TD.

## Evidence

- 3 / 18 positive deltas, all tiny (+0.0008–+0.0068).
- 3 Bonferroni-sig losses (flan-t5-large DL19, DL20; qwen3.5-27b DL20).
- Family mean Δ = −0.0232.
- Ranking agreement: TD–BU Overlap@10 ≈ 5.04, Kendall τ ≈ 0.859 (vs TD–DualEnd 7.01 / 0.925).

## Mechanism

Rank-fusion theory (paper:zeng2024_llm_rankfusion) says fusion helps when component rankings carry complementary signal + symmetric noise. BU fails both conditions: low complementarity (overlap only 5/10) and asymmetric bias (overwhelming recency — claim:C5).

## Supporting experiments

- exp:main_bidir_rrf
- exp:main_bidir_wt
- exp:ablation_alpha
- exp:analysis_ranking_agreement

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
