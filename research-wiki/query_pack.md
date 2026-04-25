# Query Pack

<!-- AUTO-GENERATED, max ~8000 chars. Regenerate after wiki mutations. -->
<!-- Budget: project 300 / gaps 1200 / clusters 1600 / failed 1400 / papers 1800 / chains 900 / unknowns 500 -->
<!-- Last built: 2026-04-20T10:55:00+10:00 (post idea:007 addition) -->

## Project direction

Setwise LLM reranking (paper:zhuang2024_setwise) extracts one decision per call. Strategies tested: reverse-elicit worst (idea:001), co-elicit best+worst (idea:002), independent fusion of TD+BU (idea:003). Active refinements: selective DualEnd (idea:004), bias-aware DualEnd (idea:005), same-call regularized (idea:006), and the MaxContext family (idea:007 — whole-pool TopDown / BottomUp / DualEnd variants on Qwen, plan in IDEA_007.md). Target venue: ICTIR (claim:C10). Conservative framing.

## Top gaps

- **gap:G1** — setwise extracts only one decision per call; worst-signal and relative-order info discarded. Status: partial. Linked ideas: 001 (failed), 002 (partial), 003 (failed), 004 / 006 (pending).
- **gap:G3** — LLMs are asymmetrically competent at best vs worst selection. Status: diagnosed by idea:001 failure and claim:C1; mitigation via co-elicitation (idea:002) partial; same-call regularization (idea:006) pending.
- **gap:G2** — no prior study of position bias under joint best+worst prompting. Status: novel `dual_worst` primacy reversal documented in claim:C5 / exp:analysis_position_bias; mitigation idea:005 pending.
- **gap:G4** — no framework for when expensive setwise variants are justified. Status: frontier mapped (claim:C9); empty region between TD-Bubble and DE-Cocktail is target for idea:004/005/006.
- **gap:G5** — model-family-dependent performance (T5 vs Qwen). Status: pattern identified (claim:C7); global routing pending.

## Paper clusters

- **Setwise paradigm (core):** paper:zhuang2024_setwise (SIGIR 2024) introduces setwise; paper:podolak2025_setwise_insertion (SIGIR 2025) adds warm-start efficiency; paper:zhuang2025_rank_r1 adds reasoning/RL. paper:chen2025_tour_rank and paper:blitzrank2026 propose tournament-graph variants — parallel "more info per comparison" angle. paper:sato2026_sorting_survey (DPC Tech Report) formalizes sorting-over-LLMs but does not cover dual-output comparators.
- **Listwise paradigm (related baseline):** paper:sun2023_rankgpt (EMNLP 2023) introduces RankGPT with sliding window. paper:ma2023_zero_shot_listwise extends to open LLMs. paper:zhang2025_rank_without_gpt makes GPT-independent; paper:ren2025_self_calibrated_listwise adds calibration. paper:rank1_2025 scales reasoning at test-time.
- **Position bias (related):** paper:liu2024_lost_in_middle introduces U-shape; paper:tang2024_found_in_middle mitigates via permutation self-consistency; paper:hutter2025_positional_rag shows model-family variance; paper:zeng2024_llm_rankfusion formalizes non-transitive inconsistency.
- **Pairwise + supervised baselines:** paper:qin2024_prp (NAACL 2024) for pairwise; paper:nogueira2020_monot5 / paper:pradeep2021_expando_mono_duo as pre-LLM supervised rerankers.
- **Efficiency (peripheral):** paper:chen2025_icr_attention (ICLR 2025) attention-based reranker; paper:peng2025_flops_reranking proposes FLOPs metrics.

## Failed / killed ideas — do not re-try in this form

- **idea:001 BottomUp (reverse-selection).** Mean Δ −0.0616; 6 Bonferroni-sig losses; 0 wins. Catastrophic on flan-t5-large (Δ −0.2302). Failure modes: recency collapse (D-freq 0.40–0.63), LLM training asymmetry (trained to identify best, not worst), efficiency is worse for top-k. Mitigation path: **only co-elicit worst with best** (→ idea:002); do not retry standalone worst-selection.
- **idea:003 BiDir ensemble (independent TD + BU fusion).** Mean Δ −0.0232; 0/18 wins above +0.01; 3 Bonferroni-sig losses. Best α = 0.9 (i.e. ~pure TD). Failure mode: rank fusion needs symmetric noise, BU has asymmetric bias ⇒ fusion imports bias. Mitigation path: extraction must be **inside the prompt** (same-call), not across independent runs. Also rules out naive listwise-style permutation fusion over TD + BU pair.
- **Best-only-proxy confound on T5/likelihood DualEnd.** Any DualEnd T5 result or `--scoring likelihood` result is computed from a best-only forward pass (`setwise_extended.py:481-482`) — not a true joint elicitation. If future claims rest on worst-signal utility, must use Qwen generation path only.

## Top papers (ranked)

1. paper:zhuang2024_setwise — direct foundation; every method in this repo extends its prompt + sort path.
2. paper:tang2024_found_in_middle — permutation self-consistency; motivates idea:005.
3. paper:zeng2024_llm_rankfusion — non-transitive LLM comparators; explains why idea:003 imports BU bias.
4. paper:qin2024_prp — pairwise baseline + two-ordering bias trick; motivates idea:005.
5. paper:sato2026_sorting_survey — provides cocktail-shaker / double-ended selection framing; explicitly does not cover dual-output comparators.
6. paper:liu2024_lost_in_middle — base reference for position bias; our claim:C5 extends to joint prompts.
7. paper:podolak2025_setwise_insertion — warm-start selective sort; inspires idea:004 selective activation.
8. paper:chen2025_tour_rank / paper:blitzrank2026 — tournament-graph rival approach to "more info per comparison".
9. paper:hutter2025_positional_rag — model-family-dependent bias; supports claim:C7 / gap:G5.
10. paper:zhuang2025_rank_r1 — reasoning axis; explicitly out of scope for this project.
11. paper:peng2025_flops_reranking — hardware-independent efficiency framing; referenced by exp:analysis_pareto.

## Active limitation → opportunity chains

- BottomUp fails alone → join it with best → DualEnd wins 14/18 (claim:C2) → but expensive → route it only to hard windows (idea:004) → pending.
- DualEnd T5 / likelihood path is a best-only proxy (code confound) → paper must disclose → future worst-signal claims must use Qwen generation only.
- BiDir fails because BU is biased → same-call regularization uses worst only locally, not as a second ranking (idea:006) → pending.
- dual_worst primacy reversal is observed (claim:C5) → exploit via controlled orderings + majority vote (idea:005) → pending.
- TD-Bubble → DE-Cocktail frontier gap is 82% cost for +0.0065 NDCG (claim:C9) → selective / bias-aware variants must land in this region → pending.
- Frontier target on wall-clock + comparisons axes (NOT token axis) → MaxContext family (idea:007): one-prompt whole-pool selection on Qwen at pool_size ≤ 50, with DualEnd / TopDown / BottomUp variants → Codex-audited plan in IDEA_007.md, staged matrix plus two single-extreme pool sweeps → not yet executed.

## Open unknowns

- Does the DualEnd directional pattern (14/18) hold out-of-domain on BEIR? (qwen3-8b 5/6 done; qwen3.5-9b pending)
- Is the `dual_worst` primacy reversal order-robust (idea:005 will reveal) or fragile under re-ordering?
- Can selective DualEnd (idea:004) achieve quality-cost parity with DE-Cocktail at ~400 comparisons?
- Do same-call worst constraints (idea:006) help as a regularizer or only add noise?
- Will same-method / same-sort consolidated tables (top priority in Need_to_Run.txt) change any narrative claim?
