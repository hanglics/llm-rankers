# Research Wiki Index

<!-- AUTO-GENERATED. Regenerate after mutations. -->
<!-- Last regenerated: 2026-04-20T10:55:00+10:00 (post idea:007 addition) -->

## Papers (20)

### Core (foundational to this project)

- [zhuang2024_setwise](papers/zhuang2024_setwise.md) — Setwise prompting for LLM reranking (SIGIR 2024)

### Related (directly informs this project)

- [sun2023_rankgpt](papers/sun2023_rankgpt.md) — RankGPT listwise reranking (EMNLP 2023)
- [qin2024_prp](papers/qin2024_prp.md) — Pairwise Ranking Prompting (NAACL 2024 Findings)
- [tang2024_found_in_middle](papers/tang2024_found_in_middle.md) — Permutation self-consistency (NAACL 2024)
- [liu2024_lost_in_middle](papers/liu2024_lost_in_middle.md) — Lost-in-the-middle position bias (TACL 2024)
- [zeng2024_llm_rankfusion](papers/zeng2024_llm_rankfusion.md) — LLM-RankFusion / intrinsic inconsistency (NeurIPS WS 2024)
- [sato2026_sorting_survey](papers/sato2026_sorting_survey.md) — Sorting with LLMs survey (DPC-TR-2026-001)
- [podolak2025_setwise_insertion](papers/podolak2025_setwise_insertion.md) — Setwise insertion-sort efficiency (SIGIR 2025)
- [zhuang2025_rank_r1](papers/zhuang2025_rank_r1.md) — Rank-R1 reasoning reranker
- [hutter2025_positional_rag](papers/hutter2025_positional_rag.md) — Model-family-dependent RAG bias (ECIR 2025)
- [peng2025_flops_reranking](papers/peng2025_flops_reranking.md) — FLOPs-based reranker efficiency (EMNLP Industry 2025)
- [chen2025_tour_rank](papers/chen2025_tour_rank.md) — Tournament-bracket setwise (WWW 2025)
- [blitzrank2026](papers/blitzrank2026.md) — Principled tournament graphs (arXiv 2026)
- [ren2025_self_calibrated_listwise](papers/ren2025_self_calibrated_listwise.md) — Self-calibrated listwise (WWW 2025)
- [zhang2025_rank_without_gpt](papers/zhang2025_rank_without_gpt.md) — GPT-independent listwise (ECIR 2025)

### Peripheral (pipeline baselines, alternative axes)

- [nogueira2020_monot5](papers/nogueira2020_monot5.md) — MonoT5 supervised pointwise (EMNLP 2020 Findings)
- [pradeep2021_expando_mono_duo](papers/pradeep2021_expando_mono_duo.md) — Supervised staged pipeline (arXiv 2021)
- [rank1_2025](papers/rank1_2025.md) — Test-time compute reranker (COLM 2025)
- [chen2025_icr_attention](papers/chen2025_icr_attention.md) — Attention-based rerankers (ICLR 2025)
- [ma2023_zero_shot_listwise](papers/ma2023_zero_shot_listwise.md) — Zero-shot listwise on open LLMs

## Ideas (7)

- [idea:001 BottomUp](ideas/idea_001_bottomup.md) — failed (reverse-selection alone)
- [idea:002 DualEnd](ideas/idea_002_dualend.md) — succeeded partial (joint best+worst; only Qwen-generation path exercises true joint elicitation)
- [idea:003 BiDir ensemble](ideas/idea_003_bidir.md) — failed (independent TD+BU fusion)
- [idea:004 Selective DualEnd](ideas/idea_004_selective_dualend.md) — active (flan-t5-xl partial)
- [idea:005 Bias-Aware DualEnd](ideas/idea_005_bias_aware_dualend.md) — proposed (all pending)
- [idea:006 Same-Call Regularized](ideas/idea_006_samecall_regularized.md) — proposed (all pending)
- [idea:007 MaxContext family](ideas/idea_007_maxcontext_dualend.md) — active plan (Codex-audited 3 rounds, ready to execute; see `IDEA_007.md`)

## Experiments (33)

### Main sweeps (9 models × 2 datasets = 18 runs each)

- [exp:main_td_heap](experiments/main_td_heap.md) — baseline
- [exp:main_td_bubble](experiments/main_td_bubble.md) — strongest TopDown on 4/6 T5 configs
- [exp:main_bu_heap](experiments/main_bu_heap.md) — never wins
- [exp:main_bu_bubble](experiments/main_bu_bubble.md) — never wins
- [exp:main_de_cocktail](experiments/main_de_cocktail.md) — 11/18 overall wins
- [exp:main_de_selection](experiments/main_de_selection.md) — holds the only Bonferroni-sig DualEnd win
- [exp:main_bidir_rrf](experiments/main_bidir_rrf.md) — never beats TD
- [exp:main_bidir_wt](experiments/main_bidir_wt.md) — best at α=0.9

### Ablations

- [exp:ablation_num_child](experiments/ablation_num_child.md)
- [exp:ablation_alpha](experiments/ablation_alpha.md)
- [exp:ablation_passage_length](experiments/ablation_passage_length.md)

### Analyses

- [exp:analysis_position_bias](experiments/analysis_position_bias.md) — source of claim:C5
- [exp:analysis_significance_tests](experiments/analysis_significance_tests.md)
- [exp:analysis_per_query_wins](experiments/analysis_per_query_wins.md)
- [exp:analysis_ranking_agreement](experiments/analysis_ranking_agreement.md)
- [exp:analysis_pareto](experiments/analysis_pareto.md) — source of claim:C9
- [exp:dualend_parse_success](experiments/dualend_parse_success.md) — diagnostic (DualEnd joint-parse rate)
- [exp:query_difficulty_stratification](experiments/query_difficulty_stratification.md)
- [exp:generalization_weakness](experiments/generalization_weakness.md) — BEIR coverage risk

### Pending / partial

- [exp:selective_dualend_flan_t5_xl](experiments/selective_dualend_flan_t5_xl.md) — 6/12 done
- [exp:selective_dualend_qwen_pending](experiments/selective_dualend_qwen_pending.md) — 24 runs pending
- [exp:bias_aware_dualend_pending](experiments/bias_aware_dualend_pending.md) — 12 runs pending
- [exp:samecall_regularized_pending](experiments/samecall_regularized_pending.md) — 12 runs pending
- [exp:beir_generalization](experiments/beir_generalization.md) — flan-t5-xl done, Qwen3-8B 5/6, Qwen3.5-9B pending
- [exp:likelihood_scoring_pending](experiments/likelihood_scoring_pending.md) — 6 runs pending
- [exp:same_method_tables_pending](experiments/same_method_tables_pending.md) — **completed 2026-04-21**; 12 pairwise tables + SIGNIFICANCE_TESTS_PAIRWISE.{md,json}
- [exp:maxdoc_dualend_pending](experiments/maxdoc_dualend_pending.md) — superseded by idea:007

### idea:007 MaxContext family matrix (plan at `IDEA_007.md`)

- [exp:maxcontext_dualend_pool_sweep](experiments/maxcontext_dualend_pool_sweep.md) — Study A: pool-size sweep, 60 runs
- [exp:maxcontext_dualend_pl_sweep](experiments/maxcontext_dualend_pl_sweep.md) — Study B: pl sweep + dualend-nc3 control arm, 96 runs
- [exp:maxcontext_dualend_order_pilot](experiments/maxcontext_dualend_order_pilot.md) — Study C: order-robustness launch gate, 12 runs
- [exp:maxcontext_dualend_baselines](experiments/maxcontext_dualend_baselines.md) — matched-`hits` predeclared baselines at {10, 30, 50}, 144 runs
- [exp:maxcontext_topdown_pool_sweep](experiments/maxcontext_topdown_pool_sweep.md) — whole-pool best-only sweep
- [exp:maxcontext_bottomup_pool_sweep](experiments/maxcontext_bottomup_pool_sweep.md) — whole-pool worst-only sweep

## Claims (10)

- [claim:C1](claims/C1_directional_asymmetry.md) — directional asymmetry (supported)
- [claim:C2](claims/C2_dualend_strongest.md) — DualEnd strongest overall (supported, statistically fragile per C6)
- [claim:C3](claims/C3_bottomup_weaker.md) — BottomUp consistently weaker (strongly supported)
- [claim:C4](claims/C4_bidir_fails.md) — BiDir fails because BU is biased (strongly supported)
- [claim:C5](claims/C5_joint_changes_bias.md) — joint prompting changes bias; novel dual_worst primacy reversal (strongly supported)
- [claim:C6](claims/C6_statistical_fragility.md) — TREC DL statistical fragility (strongly supported)
- [claim:C7](claims/C7_window_size_model_family.md) — window size × model family interaction (supported, limited)
- [claim:C8](claims/C8_joint_elicitation_is_contribution.md) — joint elicitation, not sort novelty, is contribution (supported)
- [claim:C9](claims/C9_pareto_frontier.md) — Pareto frontier has empty gap between TD-Bubble and DE-Cocktail (supported)
- [claim:C10](claims/C10_framing_ictir_conservative.md) — ICTIR-first, conservative framing (reported / policy)

## Gaps (see [gap_map.md](gap_map.md))

- gap:G1 — information extraction per setwise call
- gap:G2 — position bias under joint prompts
- gap:G3 — asymmetric best-vs-worst competence
- gap:G4 — setwise efficiency-effectiveness frontier
- gap:G5 — model-family-dependent performance

## Derived artifacts

- [query_pack.md](query_pack.md) — compressed context pack for `/idea-creator` and cross-skill retrieval (regenerated on each mutation)

## Graph

- [graph/edges.jsonl](graph/edges.jsonl) — 163 edges across `extends`, `inspired_by`, `addresses_gap`, `tested_by`, `supports`, `invalidates`, `refines`
- [IDEA_007.md](/Users/hangli/projects/llm-rankers/IDEA_007.md) — full idea:007 plan (Codex-audited, ready to execute)
