# PIVOT_PLAN — Pivot to MaxContext as the Main Contribution

> **Status:** Draft v3, Codex-ACCEPT (2026-04-28, gpt-5.5 + xhigh, 3 rounds). NOT YET COMMITTED. No code, doc, or experiment changes are executed from this plan; user runs §9 Step 1 to begin the staged decision process.
> **Author:** Hang Li
> **Date:** 2026-04-28
> **Predecessors:** `IDEA_007.md` (MaxContext design spec), `research-wiki/PAPER_PLAN.md` (current paper structure), `research-wiki/claims/C10_framing_ictir_conservative.md` (current framing constraint), `.planning/repo_walk/{CODE,INFRA,RESEARCH}.md` (2026-04-28 repo synthesis)

## 1. Context

The paper currently frames the contribution as **directional asymmetry in LLM setwise ranking**, supported by three families adapted from the upstream `ielab/llm-rankers` codebase plus the original Setwise primitive:

- **TopDown (TD-Heap, TD-Bubble)** — upstream Setwise; cheap baselines.
- **BottomUp (BU-Heap, BU-Bubble)** — adapted from upstream; consistently weaker (claim:C3, 6 Bonferroni-sig losses, 0 wins).
- **DualEnd (DE-Cocktail, DE-Selection)** — joint best+worst per call, drives cocktail-shaker / double-ended-selection sort skeletons. 14/18 directional wins on TREC DL; 1 Bonferroni-sig win (Qwen3-8B DL19); 5.6-8.9× TD-Heap wall-clock.
- **Bidirectional ensemble (BiDir-RRF, BiDir-Weighted)** — independent-fusion of TD + BU; never beats TD (claim:C4).

Phase 1-4 of `EXPERIMENT_PLAN.md` are complete. The pairwise same-method tables (closed 2026-04-21) report the headline positive: **DE-Cocktail + DE-Selection vs TD-Bubble on DL19, Qwen3-8B, 2 Bonferroni-significant DualEnd wins**. The current framing in `claim:C10` is conservative, ICTIR-targeted: one modestly effective method (DualEnd) plus two coherent negative results (BottomUp, BiDir). ARR is gated on stronger refinement / generalization evidence.

**The MaxContext family (idea:007)** was added 2026-04-23 as a refinement targeting the empty Pareto-frontier region between TD-Bubble and DE-Cocktail (claim:C9). It fits the entire rerank pool (`pool_size ≤ 50`) into a single Qwen prompt:

- **MaxContext-DualEnd** — best+worst per call over the whole live pool; pool shrinks by 2; `floor(N/2)` LLM calls.
- **MaxContext-TopDown** — best-only per call; pool shrinks by 1; `N - 2` LLM calls + 1 BM25 bypass at `n_docs=2`.
- **MaxContext-BottomUp** — worst-only per call; pool shrinks by 1; `N - 2` LLM calls + 1 BM25 bypass at `n_docs=2`.

Phase 1 sanity is **green** (15 runs done; algorithm terminates, parses correctly, zero truncation). Recent code work (2026-04-26 → 2026-04-28) hardened the parser against `0`/`51`/`-1` refusal patterns and fixed a TD-Bubble whole-pool short-circuit. Phases 2-5 of the staged matrix (~300 production runs) are queued behind the user's compute budget.

**The proposal:** make MaxContext the **main contribution** of the paper. Recast the upstream-adapted methods (TD-Heap/Bubble, BU-Heap/Bubble, DE-Cocktail/Selection) and BiDir into specific narrative roles (motivation, baseline, ablation) rather than as the paper's central contribution.

**Why now:** MaxContext-DualEnd's `floor(N/2)` calls vs DE-Cocktail's mean 546 represents the kind of order-of-magnitude efficiency story claim:C9 named as the refinement target. Whole-pool joint elicitation is a more novel angle than joint elicitation at `c=4` because it directly stresses the model's long-context attention. And the BU/BiDir negative results, while statistically robust (claim:C3 has the highest evidence_strength in the catalog), are not by themselves a paper headline — they need a **positive** method as the pitch.

**Intended outcome:** a clean paper architecture with MaxContext as the §5 main results, the existing experiments recast to §3 (motivation) and §5 (baseline), and three downstream experiments (Selective / Bias-Aware / Same-Call DualEnd) **scope-deferred** to future work — not "superseded," because each addresses a distinct gap (G1+G4+G5 routing for idea:004; G2 position-bias-mitigation for idea:005; G3 worst-as-regularizer for idea:006). See §7 for the gap-specific framing.

## 2. The Pivot in One Paragraph

The paper's center of gravity moves from "**directional asymmetry diagnosed via four strategy families, with DualEnd as the modestly effective positive case**" to "**MaxContext tests whether the setwise-ranking signal — joint elicitation when applied as DualEnd, or single-extreme selection when applied as TopDown/BottomUp — can be amortized over far fewer sequential LLM calls by fitting the entire rerank pool into a single Qwen prompt; the directional asymmetry, the failure of independent fusion, and the cost-effectiveness ceiling of small-window joint elicitation together motivate why amortization at scale is the right axis to push.**" Adopting the *amortization* framing rather than the stronger *necessity* framing is deliberate: it survives more variant-outcome branches (see §3.5) without committing the headline to a single method.

The candidate headline method is MaxContext-DualEnd; the secondary methods are MaxContext-TopDown / BottomUp (single-extreme variants); the previous DualEnd-Cocktail/Selection and TD-Bubble become the **best-of** matched-hits baseline that MaxContext must beat on the Pareto plot; BU-Bubble/Heap and BiDir-RRF/Weighted become motivation evidence; the never-launched refinement methods (Selective / Bias-Aware / Same-Call) are **scope-deferred** (not "superseded") because they target gaps G2 and G3, not just the C9 efficiency gap — they remain valid future work tied to specific mechanisms.

## 3. Role Recasting (not "demote to side works")

The existing experiments do real narrative work in a MaxContext-led paper. The key insight: **conflating "best candidate method" with "narrative load-bearing evidence" loses information**. Re-cast each existing piece by its role, not by its perceived "main vs side" status:

| Existing work | Role in MaxContext-led paper | Evidence anchor | Stays / Cut |
|---|---|---|---|
| **TD-Heap, TD-Bubble** (upstream Setwise) | Cheapest-cost frontier anchors. Establish the lower envelope of the comparisons-axis frontier. | claim:C9 frontier point (TD-Heap: 76.5 cmp, 0.6851 NDCG; TD-Bubble: 300 cmp, 0.6897 NDCG) | **STAYS** as §5 baseline |
| **BU-Bubble, BU-Heap** | Motivation evidence — naive worst-selection fails; asymmetry is real and severe. **Most statistically robust result in the paper** (6 Bonferroni-sig losses, 0 wins, very_high evidence_strength). | claim:C1, claim:C3 | **STAYS** as §3 motivation. **Do not cut** — losing this loses the most rigorous numbers we have. |
| **BiDir-RRF, BiDir-Weighted** | Motivation evidence — independent fusion can't recover the worst signal; joint signal is necessary. Tested 0.3-0.9 alpha sweep; never beats TD. | claim:C4 (very_high evidence_strength) | **STAYS** as §3 motivation. |
| **DE-Cocktail, DE-Selection** | **Best-of** matched-hits baseline that MaxContext-DualEnd must beat on the Pareto plot. DE-Selection holds the only Bonferroni-significant DualEnd win in the family-level tests and is competitive in some Qwen settings, so reporting only DE-Cocktail invites reviewer pushback. The headline rule (see §8 claim:C11) is "best of {DE-Cocktail, DE-Selection, TD-Bubble} at matched `hits`." DE-Cocktail is reported separately as the current global C9 frontier point. | claim:C2, claim:C9 frontier point (DE-Cocktail: 546 cmp, 212.6s, 0.6962 NDCG); pairwise same-sort evidence in `research-wiki/SIGNIFICANCE_TESTS_PAIRWISE.md` | **STAYS** as §5 primary baseline (best-of). |
| **Selective DualEnd** (idea:004) | Targets gap:G1 (information extraction) + G4 (efficiency frontier) + G5 (model-family interaction). Partial Pareto-axis overlap with MaxContext, but the gating mechanism (shortlist / uncertain / hybrid policies) is orthogonal to whole-pool elicitation. flan-t5-xl 6/12 done; Qwen 24 pending. | gap:G1, G4, G5 (per `research-wiki/ideas/idea_004_selective_dualend.md`) | **SCOPE-DEFER** (not "supersede"). Existing 6 flan-t5-xl results either dropped or moved to a §6 ablation paragraph; pending 24 not launched. Listed in §7 future work. |
| **Bias-Aware DualEnd** (idea:005) | Targets gap:G2 (position bias under joint prompts), not C9-axis. Designed to mitigate the `dual_worst` primacy reversal (claim:C5) via controlled orderings + majority vote. **MaxContext does NOT address gap:G2 directly** (in fact, the Phase 2 order pilot is what tests whether MaxContext is itself order-stable at large windows). 12 pending. | gap:G2 (per `research-wiki/ideas/idea_005_bias_aware_dualend.md`) | **SCOPE-DEFER**. Not launched. Listed in §7 future work tied to claim:C14 (position bias at scale). |
| **Same-Call Regularized** (idea:006) | Targets gap:G3 (asymmetric best-vs-worst competence) via worst-as-local-regularizer outside the head — a different mechanism than joint elicitation. **MaxContext does NOT address gap:G3 directly.** 12 pending. | gap:G3 (per `research-wiki/ideas/idea_006_samecall_regularized.md`) | **SCOPE-DEFER**. Not launched. Listed in §7 future work tied to the same-call worst-signal mechanism. |
| **MaxContext-DualEnd** (idea:007) | **MAIN CONTRIBUTION.** Whole-pool joint elicitation. | claim:C9 (Pareto-frontier filling), claim:C8 (joint elicitation is the contribution) | **STAYS** as §5 headline. |
| **MaxContext-TopDown / BottomUp** (idea:007) | Secondary contribution + ablations isolating the joint vs single-extreme effect at whole-pool scale. | New claims (C11+ candidates, see §10) | **STAYS** as §5 secondary results. |

**Compute budget freed by scope-deferrals:** see §7.5 reconciled accounting. The headline value of the cuts is **narrative focus**, not net compute savings — the MaxContext expansion matrix (Phase 4 alone, ~324 runs across DualEnd + TopDown + BottomUp pool sweeps + 144 baselines) substantially exceeds the freed budget.

## 3.5 Outcome Branch Matrix

The pivot's framing must survive multiple variant outcomes. The amortization framing (§2) is chosen specifically to remain coherent across these branches. After Phase 4 completes, exactly one of the following branches is operative:

| Branch | MaxContext-DualEnd | MaxContext-TopDown | MaxContext-BottomUp | Title (working) | Headline RQ | Surviving claims |
|---|---|---|---|---|---|---|
| **A. Joint amortization wins.** DualEnd ties or beats best baseline at matched hits; TD/BU lag DualEnd at matched hits. | ✅ | ⚠ | ⚠ | "Whole-Pool Joint Elicitation Amortizes the Setwise Signal Across Far Fewer LLM Calls" | RQ3 (DualEnd efficiency) primary | C8 (joint), C9, C11 (efficiency), C13 (long-context joint) |
| **B. Whole-pool selection wins regardless of mode.** TD ≈ DualEnd ≈ BU on quality at matched hits; main story is whole-pool selection at long context, not the joint primitive. | ✅ | ✅ | ✅ | "Whole-Pool Long-Context Selection for LLM Rerankers" | RQ3 + RQ4 co-headline | C9, C11, C13 reframed as selection-not-joint, C12 (single-extreme parity) |
| **C. Single-extreme is the surprise.** TD or BU beats DualEnd; suggests the joint primitive doesn't scale to long context, but selection does. | ⚠ | ✅ | ✅ | "Single-Extreme Whole-Pool Selection Outperforms Joint Elicitation at Long Context" | RQ4 primary; DualEnd becomes contrast | C12 (single-extreme primary), C9, C11 reframed; **claim:C8 is contradicted at scale** — must be reported as a finding, not buried |
| **D. Pool=50 is unstable; smaller pools work.** Phase 2 reveals order-sensitivity at pool=50 but pool=20 or pool=30 is stable. Variant outcomes evaluated at the smaller pool. | varies | varies | varies | "Saturation in Whole-Pool Reranking: a Latency Frontier Result" | RQ3 reframed as pool-size saturation | C9, C11 (with smaller pool), claim:Cnew_saturation |
| **E. MaxContext family fails.** None of the three variants beat DE-Cocktail at any pool size on quality-axis bootstrap CIs. | ❌ | ❌ | ❌ | (revert to current paper) | RQ4 demoted to "we tried, it didn't pan out" | original C1-C10; pivot abandoned |

**Branch decisions are predeclared.** The plan does not permit post-hoc title or claim selection — once Phase 4 results land, the matrix above commits the paper to a specific branch. Each branch's "Surviving claims" list governs which `research-wiki/claims/C*.md` files get committed (see §8).

**Implication for pre-commit doc work.** Because branches A, B, C, D have different headline narratives, the §5.2 doc rewrites cannot be drafted before Phase 4 results determine the branch. Only the preregistration claim *stubs* (C11-C14) can land before Phase 4; the `research-wiki/claims/C10_framing_ictir_conservative.md`, `research-wiki/PAPER_PLAN.md`, and `research-wiki/NARRATIVE.md` rewrites land after.

## 4. New Paper Architecture

### 4.1 Section structure (working draft)

| § | Title (working) | Purpose | Anchor claims | Anchor experiments |
|---|---|---|---|---|
| §1 | Introduction | C9 motivation: the Pareto frontier has an empty refinement region. Whole-pool joint elicitation fills it. | C8, C9 | (overview fig of frontier with MaxContext anchor) |
| §2 | Related Work | Setwise lineage; long-context attention; position bias; efficiency-focused reranking. | gap:G1, G2, G4 | — |
| §3 | Motivation: Directional Asymmetry | Why does the Pareto frontier *have* an empty region? Best vs worst is asymmetric (BU fails, C3); independent fusion can't repair it (BiDir fails, C4); joint elicitation at small windows works but is expensive (DE at c=4, C2 + C6). | C1, C2, C3, C4, C6 | exp:main_bu_*, exp:main_bidir_*, exp:main_de_cocktail, exp:main_de_selection |
| §4 | MaxContext: Whole-Pool Joint Elicitation | The method. Three variants: DualEnd (headline), TopDown / BottomUp (single-extreme isolating). Numeric labels 1..N, deterministic n=2 BM25 endgame, prompt hardening. | C8 (extends), claim:Cnew_maxcontext (proposed) | exp:maxcontext_dualend_pool_sweep, exp:maxcontext_topdown_pool_sweep, exp:maxcontext_bottomup_pool_sweep |
| §5 | Main Results | MaxContext-DualEnd / TopDown / BottomUp vs **best-of {DE-Cocktail, DE-Selection, TD-Bubble}** at matched `hits` (predeclared per §3.5 branch; default {10, 20, 30, 50}) on the comparisons-axis, wall-clock-axis, **and full token-axis breakdown** (prompt + completion + total) per §10's transparent-cost-reporting rule. Single-extreme MaxContext variants reported with same-direction matched-hits baselines (TopDown vs TD-Heap; BottomUp vs BU-Heap) per claim:C12 separation rule. Position bias at large windows (claim:C14 candidate, descriptive). | C11, C12, C9 (extended), C14 | exp:maxcontext_dualend_baselines, exp:maxcontext_dualend_pl_sweep |
| §6 | Analysis | Pool-size saturation (Study A); passage-length sensitivity (Study B); order robustness pilot (Study C); per-query parse fallback rates; long-context attention degradation evidence. | C7 (extends to large windows) | exp:maxcontext_dualend_order_pilot, exp:maxcontext_dualend_pl_sweep |
| §7 | Discussion + Limitations | Qwen-only (per claim:C10 confound); ICTIR conservative framing dropped only if multi-family expansion lands. | C10 (rewrite) | — |
| §8 | Conclusion | — | — | — |

### 4.2 RQ reorganization

**Old RQ structure** (per `research-wiki/PAPER_PLAN.md`):
- RQ1: BottomUp vs TopDown effectiveness?
- RQ2: DualEnd justifies extra compute?
- RQ3: Bidirectional ensemble helps?
- RQ4 (conditional): MaxContext fills C9 gap?

**New RQ structure** (proposed):
- **RQ1 (motivation):** Why does the Pareto frontier shape look the way it does? → C1, C3, C4 (asymmetry + BiDir failure)
- **RQ2 (motivation):** Why is small-window joint elicitation (DE-Cocktail) expensive? → C2, C9 (info-per-call ceiling at c=4)
- **RQ3 (main):** Does whole-pool joint elicitation (MaxContext-DualEnd) match DE-Cocktail effectiveness at lower comparisons / wall-clock cost? → claim:Cnew_efficiency
- **RQ4 (analysis):** Do single-extreme MaxContext variants (TopDown / BottomUp) preserve quality at whole-pool scale, or does long-context attention degrade them? Asymmetric BM25-bypass impact. → claim:Cnew_singleextreme

## 5. Constraints to Honor

### 5.1 claim:C10 confound — Qwen-only joint elicitation

> Only the Qwen-generation code path (`setwise_extended.py:455-474`) exercises true joint elicitation. T5 generation and `--scoring likelihood` collapse to a best-only proxy (argmax/argmin of best-label likelihood).

MaxContext is currently hard-restricted to Qwen3 / Qwen3.5 via:

```python
# llmrankers/setwise_extended.py:23
MAXCONTEXT_ALLOWED_MODEL_TYPES = frozenset({"qwen3", "qwen3_moe", "qwen3_5"})
```

A MaxContext-led paper without expansion is a **6-model Qwen-only paper**. This is acceptable for ICTIR but tighter than the current 9-model story (3 T5 + 3 Qwen3 + 3 Qwen3.5).

**Implication for §3:** the motivation experiments (BU, BiDir, DE-Cocktail/Selection) span all 9 models including T5. The §3 narrative therefore retains the 9-model breadth; only §5 main results are Qwen-only. This is methodologically clean (T5 evidence supports the *asymmetry* claim regardless of MaxContext's scope) and should be explicitly stated in the limitations.

### 5.2 IDEA_007 §7 risk #6 (claim:C10 framing constraint) — staged doc work

> *"This is **not** an automatic ICTIR story upgrade. Even a successful MaxContext only contributes to the efficiency axis; the core setwise-asymmetry narrative (claim:C1, claim:C8) is unchanged."*

The original framing constraint expected MaxContext to be an additional refinement, not a headline. **The pivot deliberately departs from this clause.** Doc work splits into **two stages** to avoid premature commitment:

**Stage 1 (preregistration; lands before Phase 4 launches, after Phase 2+3 pass):**

- Provisional `research-wiki/claims/C11_maxcontext_efficiency.md`, `C12_maxcontext_singleextreme.md`, `C13_long_context_amortization.md`, `C14_position_bias_at_scale.md` claim *stubs* with `status: provisional, evidence_strength: pending, predeclared_2026-04-28`. These predeclare the measurable hypotheses Phase 4 will test (see §8 for the measurable phrasings).
- `IDEA_007.md` §7 risk #6 updated to flag the staged pivot is in-flight (Stage 1 done, Stage 2 pending Phase 4).
- No changes yet to `research-wiki/claims/C10_framing_ictir_conservative.md`, `research-wiki/PAPER_PLAN.md`, `research-wiki/NARRATIVE.md`, or `research-wiki/RESEARCH_BRIEF.md`.

**Stage 2 (commit; lands after Phase 4 results determine the §3.5 branch):**

- `research-wiki/claims/C10_framing_ictir_conservative.md` — branch-specific rewrite. For branch A: framing becomes "ICTIR-floor / EMNLP-ceiling, gated on MaxContext-DualEnd effectiveness." For branch B: same with selection-not-joint emphasis. For branch C: "core asymmetry-at-scale claim contradicts C8 at long context; reported as a finding." For branch D: "saturation/latency frontier result at smaller pool."
- `research-wiki/PAPER_PLAN.md` — branch-specific RQ structure (see §4.2).
- `research-wiki/NARRATIVE.md` — branch-specific narrative.
- `research-wiki/RESEARCH_BRIEF.md` — branch-specific headline.
- `research-wiki/claims/C9_pareto_frontier.md` — add MaxContext frontier point at branch's chosen `pool_size`.
- Provisional C11-C14 stubs upgraded to `status: supported / strongly_supported` with measured `evidence_strength`.

These are doc-only changes (no code impact). The two-stage split ensures: (a) we commit measurable hypotheses before seeing Phase 4 numbers (preregistration discipline; protects against post-hoc rationalization); (b) the load-bearing rewrites of C10 / PAPER_PLAN / NARRATIVE happen only after Phase 4 picks the branch.

### 5.3 Decision gate before commit

Run idea:007 **Phase 2 (expanded; ~16 runs)** + **Phase 3 (4 pairs; 12 runs)** before any cuts in §3 or any Stage 1 doc work in §5.2. Total: **~28 runs, ~56 cluster-hours**. Phase 3 is 12 runs (not 8) because the best-of {DE-Cocktail, DE-Selection, TD-Bubble} comparison rule from claim:C11 requires DE-Selection at matched hits per pair, not just DE-Cocktail.

#### Phase 2 — order pilot + position-bias-at-scale diagnostic

| Models | Datasets | Orderings | Pool sizes | Diagnostic outputs |
|---|---|---|---|---|
| Qwen3-4B + Qwen3.5-9B | DL19 + DL20 | forward, inverse, random | 50 | Per-call: label-position frequencies, selected-doc IDs, parse-fallback location distribution |

Pass criteria (all must hold):

1. **Order stability.** Max pairwise NDCG@10 Δ across {forward, inverse, random} ≤ **0.01** for both models on both datasets (12 runs core; original IDEA_007 §5.3 spec).
2. **Position-bias-at-scale diagnostic** (new, lifted from Codex round-1 MEDIUM-1 finding). Compute over the per-call comparison logs:
   - `selected_position_freq[i]` for `i ∈ [1, pool_size]` per ordering. If any single position captures > 30% of selections under forward ordering, flag as concentration risk (claim:C14 candidate evidence in either direction).
   - `selected_doc_stability` between forward and inverse: fraction of queries where the same docid was selected in both orderings. < 50% = order-dependent doc choice (red flag).
   - `parse_fallback_location_distribution`: if fallbacks concentrate at any pool size, parser hardening may be insufficient at scale.
3. **Order-stable-doc parity sanity** (4 additional runs at pool=20): same models × DL19 × forward + random ordering at pool=20. Establishes that smaller pools are at least as stable as pool=50; if they aren't, something else is wrong.

#### Phase 3 — matched-hits regression (expanded)

| MaxContext-DualEnd | Best-of baseline (mandatory at matched hits) | Models × datasets | Non-inferiority margin |
|---|---|---|---|
| pool=50 | DE-Cocktail nc=3 + DE-Selection nc=3 + (existing TD-Bubble at hits=50 if not already on disk) at matched hits=50 | Qwen3-8B + Qwen3.5-9B × DL19 + DL20 = 4 pairs (**12 runs:** MaxContext + DE-Cocktail + DE-Selection per pair; TD-Bubble counted separately if a fresh run is needed) | Δ NDCG@10 ≥ **−0.003** (point estimate) AND bootstrap-CI lower bound on Δ ≥ **−0.005** |

Pass criteria (must hold for **at least 3 of 4 pairs**):

1. **Non-inferiority on quality.** MaxContext-DualEnd's NDCG@10 ≥ best-of {DE-Cocktail, DE-Selection} NDCG@10 minus the predeclared 0.003 margin (point estimate) AND bootstrap-CI lower bound on the difference is at least −0.005. The 0.003 margin is calibrated against TREC DL bootstrap CIs at 43-54 queries (typical ±0.01).
2. **Cost reduction holds.** MaxContext-DualEnd's `Avg comparisons` ≤ DE-Cocktail's `Avg comparisons`. Wall-clock improvement directional (no formal margin).
3. **Per-call parse-fallback rate** ≤ 5% of `Avg comparisons` (sanity threshold per `research-wiki/FINDINGS.md` 2026-04-26 entry).

The 1-pair gate from v1 was insufficient: a single Qwen3-8B DL19 result could pass "within bootstrap CI" while being materially worse on point estimate. Expanding to 4 pairs across two model sizes and both datasets gives credible coverage against model-specific or dataset-specific quirks.

#### Decision tree

| Phase 2 result | Phase 3 result | Action |
|---|---|---|
| Pass (all 3 sub-criteria) | Pass (≥ 3 of 4 pairs) | **Commit Stage 1 doc work + cuts**; proceed to Phase 4. |
| Pass | Fail (< 3 of 4 pairs) at pool=50 | **Smaller-pool fallback gate** (next row). |
| Fail at pool=50 | (not run yet) | **Smaller-pool fallback gate** (next row). |
| — | — | (smaller-pool fallback) |
| Pass at pool=20 | Pass at pool=20 (4 pairs at pool=20 vs DE-Cocktail hits=20 nc=3) | **Commit pivot at smaller pool**. Branch D in §3.5. |
| Pass at pool=20 | Fail at pool=20 | **Abandon pivot.** Branch E in §3.5. |
| Fail at pool=20 | (not run) | **Abandon pivot.** Branch E. |

The smaller-pool fallback gate adds **~36 more cluster-hours** (Phase 2 at pool=20 = 8 runs; Phase 3 at pool=20 = 4 pairs × 3 methods = 12 runs). This is a contingency budget, only spent if pool=50 fails. The matched-hits baseline grid in §9 must be **predeclared** to include `pool_size ∈ {20, 30, 50}` (not just `{10, 30, 50}`) so a smaller-pool retreat has a defensible baseline at hand.

#### What the gate does NOT do

- Phase 2+3 are go/no-go gates for **launching the pivot decision-process**, not for committing Stage 2 doc rewrites. Stage 2 commits depend on Phase 4 picking a §3.5 branch.
- Phase 2 is still a "smoke gate" per IDEA_007 §5.3 — three orderings cannot rule out higher-order order effects. But with the position-bias-at-scale diagnostic, it now also serves as evidence for / against claim:C14.
- A failed Phase 2 + Phase 3 at both pool=50 and pool=20 reverts to the **current ICTIR-conservative paper** (claim:C10 unchanged, idea:004/005/006 pending runs may still be valuable).

## 6. Llama / Mistral Expansion (optional, gated)

If Phase 2 + Phase 3 pass and the user wants to push for EMNLP / ARR, expanding MaxContext beyond Qwen meaningfully strengthens generalization claims.

### 6.1 Why Llama and Mistral are technically plausible

Both are decoder-only with long context (Llama 3.x: 128K; Mistral varies). Both use generation scoring through the same `compare_both()` path Qwen uses; they should exercise true joint elicitation in principle (unlike T5 encoder-decoder or `--scoring likelihood`, which collapse to best-only proxies).

### 6.2 Required code changes

The set is broader than the v1 plan claimed. Codex round-1 caught that Mistral isn't in `CAUSAL_MODEL_TYPES` (`setwise.py:27-28`), so widening only `MAXCONTEXT_ALLOWED_MODEL_TYPES` would still fail at `SetwiseLlmRanker.__init__`.

1. **`llmrankers/setwise.py:27-28`** — extend `CAUSAL_MODEL_TYPES`:
   ```python
   CAUSAL_MODEL_TYPES = frozenset({"qwen3", "qwen3_moe", "qwen3_5", "llama", "mistral"})
   ```
   (Currently only includes Qwen variants + Llama. Verify Mistral's exact `model_type` string via `AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3").model_type` before committing — empirically `mistral` for the 7B Instruct line, but multi-checkpoint verification is required.)
2. **`llmrankers/setwise_extended.py:23`** — extend MaxContext allowlist:
   ```python
   MAXCONTEXT_ALLOWED_MODEL_TYPES = frozenset({"qwen3", "qwen3_moe", "qwen3_5", "llama", "mistral"})
   ```
3. **`MaxContextDualEndSetwiseLlmRanker._early_reject_non_qwen3`** — rename to `_early_reject_unsupported`; widen the allowed-name regex; mirror to `MaxContextTopDownSetwiseLlmRanker` and `MaxContextBottomUpSetwiseLlmRanker`.
4. **`_chat_template_kwargs` (`setwise.py:190-193`)** — Qwen-specific `enable_thinking=False` only fires when `self.config.model_type in QWEN_MODEL_TYPES`. Llama / Mistral take the empty-dict branch. **No code change**, but verify rendering: each new model's chat template must not insert `<|user|>` / `<|assistant|>` markers that break the numeric-label parser, and must not auto-prepend a system prompt that drifts the model away from "Reply with exactly one passage number from 1 to N."
5. **Prompt footer** (`setwise.py:217-225`, `setwise.py:233-241`) — gated on `numeric_1_based`, scheme-agnostic w.r.t. model family. **No code change.**
6. **Constructor-level smoke test.** Add a new test in `scripts/check_maxcontext_invariants.py` for each new family: `test_maxcontext_topdown_invariants_<family>` constructs the ranker with a Mistral or Llama identifier (mocking the model load via `fake_super_init_factory` in the existing test), confirms `model_type` allowlist passes, confirms no early-reject, confirms `_chat_template_kwargs()` returns expected dict for that family. Catches the "MAXCONTEXT_ALLOWED_MODEL_TYPES extended but CAUSAL_MODEL_TYPES forgotten" class of regression.

### 6.3 Required experiments (sanity then production)

1. **Llama-3-8B and Mistral-7B sanity smoke** — replicate idea:007 Phase 1 sanity (Qwen3-4B DL19 pool=10 pl=512 maxcontext_dualend). Each: 1 run. Gate: zero parse aborts, zero truncations, ranking is a permutation. Estimated 1-2 GPU-hours each.
2. **If sanity passes:** run Phase 2 order pilot for the new model families (4 runs each — DL19 + DL20 × 1 model from each family is sufficient; do not duplicate the 12-run Qwen pilot at scale).
3. **If order-pilot passes:** add Llama and Mistral to Phase 4 Study A pool sweep. Matrix grows from `6 Qwen × 2 datasets × 5 pool sizes = 60` to `~10 models × 2 × 5 = 100` (depending on which Llama / Mistral checkpoints are selected).

### 6.4 Risks specific to multi-family expansion

- **Different refusal patterns.** Qwen's `0` failure was specific; Llama / Mistral may emit *different* out-of-range / refusal patterns the parser doesn't recognize. **The 2026-04-26 → 2026-04-28 parser hardening loop (`_classify_numeric_noop` + `NUMERIC_REFUSAL_REGEX`) was Qwen-specific.** New patterns may need new fixtures in `scripts/check_maxcontext_invariants.py` and possibly new regex extractors.
- **Different long-context degradation profiles.** Llama-3 has documented "lost-in-middle" behavior at >32K; Mistral varies by checkpoint. The Phase 2 order pilot is the right diagnostic per family.
- **Different chat-template idioms.** Verify each model's chat template doesn't insert structural tokens that would interfere with the numeric-label parser (e.g., `<|user|>` markers leaking into the response).
- **Sanity smoke before production.** Do not launch the 100-run multi-family matrix until each new family has passed a single-query sanity check on the failing-prone configs (large pool, thinking-disabled equivalent).

### 6.5 Decision: family selection

If pursued, recommended initial scope:

- **Llama 3.1-8B-Instruct** (battle-tested, 128K context, well-supported by transformers).
- **Mistral 7B Instruct v0.3** (32K context, popular reranking baseline).

Add Llama 3.1-70B and Mixtral 8x7B only if 8B / 7B sanity passes.

## 7. Scope Deferrals (when pivot is committed)

The v1 plan called these "cuts." Codex round-1 (HIGH-5) flagged that idea:005 and idea:006 target gaps G2 and G3 respectively, not just the C9 efficiency axis. The accurate framing is **scope-deferral**, with each deferred idea tied to specific future-work mechanisms a reviewer can ask about.

| Idea / Experiment | Status | Pending runs | Gap targeted | Future-work tie-in | Action on pivot commit |
|---|---|---|---|---|---|
| **idea:004 Selective DualEnd** (`exp:selective_dualend_*`) | flan-t5-xl 6/12 done; Qwen 24 pending | 24 + 6 (likelihood arms; cf. `research-wiki/ideas/idea_004_selective_dualend.md:32-37`) | G1, G4, G5 | "Routing between cheap-but-noisy worst-pick and full DualEnd is orthogonal to whole-pool elicitation; future work can stack selective routing on top of MaxContext." | **Defer.** Existing 6 flan-t5-xl results moved to a one-paragraph §6 ablation. Pending runs not launched. Listed as future work in §7. |
| **idea:005 Bias-Aware DualEnd** (`exp:bias_aware_dualend_pending`) | All 12 pending | 12 | **G2 (position bias under joint prompts)** | "Tied to claim:C5 + C14: order-controlled prompting + majority vote is a position-bias mitigation, not a Pareto-frontier filler. Future work conditional on Phase 2 evidence that MaxContext is itself order-stable enough to base bias-aware variants on." | **Defer.** Not launched. |
| **idea:006 Same-Call Regularized** (`exp:samecall_regularized_pending`) | All 12 pending | 12 | **G3 (asymmetric best-vs-worst competence)** | "Worst-as-local-regularizer outside the head is a different mechanism than joint elicitation; future work investigating whether regularization stacks with whole-pool selection." | **Defer.** Not launched. |

These are **scope-deferrals, not supersessions.** Reviewers asking "why did you not evaluate idea:005 / idea:006?" get a principled answer: they target different mechanisms (G2, G3) and are conditional future work.

## 7.5 Reconciled Compute Accounting

v1 had inconsistent run-count and cluster-hour numbers. This table is the single source of truth:

| Bucket | Runs | Approx cluster-hours (H100, 7d wall is the ceiling — typical run completes in 1-4h) |
|---|---|---|
| **Cuts (deferrals freed)** | | |
| idea:004 Qwen pending (gen + likelihood arms) | 24 | ~36 |
| idea:004 likelihood-shortlist/uncertain on flan-t5-xl | 4 | ~6 |
| idea:005 all variants | 12 | ~18 |
| idea:006 all variants | 12 | ~18 |
| **Subtotal freed** | **52** | **~78** |
| **Decision gate (mandatory before pivot)** | | |
| Phase 2 order pilot at pool=50 | 12 | ~24 |
| Phase 2 position-bias diagnostic at pool=20 (pre-fallback sanity) | 4 | ~8 |
| Phase 3 matched-hits regression (4 pairs × 3 methods = MaxContext + DE-Cocktail + DE-Selection per pair, at pool=50) | 12 | ~24 |
| **Subtotal mandatory** | **28** | **~56** |
| **Smaller-pool fallback (contingent on Phase 2/3 pool=50 fail)** | | |
| Phase 2 order pilot at pool=20 | 8 | ~12 |
| Phase 3 matched-hits at pool=20 (4 pairs × 3 methods) | 12 | ~24 |
| **Subtotal contingent** | **20** | **~36** |
| **MaxContext Phase 4 (production)** | | |
| Study A pool sweep (DualEnd; 6 Qwen × 2 datasets × 5 pool sizes) | 60 | ~120 |
| Phase 4D MaxContext-TopDown pool sweep | 60 | ~120 |
| Phase 4E MaxContext-BottomUp pool sweep | 60 | ~120 |
| Phase 4 baselines at matched hits ∈ {**20** or {10}, 30, 50} × 6 Qwen × 2 datasets × 4 methods (TD-Heap, TD-Bubble, DE-Cocktail, DE-Selection) | 144 | ~288 |
| **Subtotal Phase 4** | **324** | **~648** |
| **MaxContext Phase 5 (production)** | | |
| Study B passage-length sweep (treatment + DualEnd-nc3 control) | 96 | ~192 |
| **Subtotal Phase 5** | **96** | **~192** |
| **Optional Llama / Mistral expansion** | | |
| Sanity smokes (Llama-3-8B + Mistral-7B at pool=10 single config) | 2 | ~4 |
| Phase 2 order pilot per family (DL19+DL20 × 1 model each × 3 orderings) | 12 | ~24 |
| Add to Phase 4 Study A + baselines (per-family expansion if sanity passes) | up to ~80 each family | ~160 each family |
| **Subtotal optional** | **~14 + up to 160 per family** | **~28 + up to ~320 per family** |

**Net new compute on top of cuts:** mandatory gate 28 runs (~56h) + Phase 4-5 production 420 runs (~840h) = ~448 runs (~896h), minus ~78h freed by cuts = **~818h net additive**, before contingencies and family expansion. The framing must be "narrative focus" not "compute parity"; the cuts cannot pay for the expansion.

This corrects v1's understated "30 cluster-hours saved" and "300 production runs" numbers, both of which were too optimistic.

**Cuts that are NOT recommended:**

- **BU-Heap, BU-Bubble.** These are claim:C3's evidence (the most statistically robust result in the catalog). Cutting them deletes the paper's strongest motivation evidence. Keep all completed runs as §3 motivation.
- **BiDir-RRF, BiDir-Weighted.** These are claim:C4's evidence. Same reasoning. Keep as §3.
- **DE-Cocktail, DE-Selection.** These are claim:C9's frontier point and the headline efficiency comparison's baseline. Cutting them removes the "MaxContext beats DE-Cocktail" claim's referent. Keep as §5 baseline.
- **TD-Heap, TD-Bubble.** Frontier anchors. Keep as §5 baseline.

## 8. New Claims Required by the Pivot (preregistered, measurable)

The current claim catalog (C1-C10) supports the asymmetry-led story. The pivot adds preregistered, measurable claims for the MaxContext-led story. v1's drafts were too loose (no non-inferiority margins, no best-baseline rule, causal phrasing on observational data); rewrites below address Codex round-1 MEDIUM-3.

| Proposed | Measurable statement | Phrasing discipline | Evidence required |
|---|---|---|---|
| **claim:C11 (efficiency, non-inferiority)** | At matched `hits` (one of {10, 20, 30, 50} predeclared per branch in §3.5), MaxContext-DualEnd's `Avg comparisons` is at most `floor(hits/2)`, AND MaxContext-DualEnd's NDCG@10 is non-inferior to **the best of {DE-Cocktail nc=3, DE-Selection nc=3, TD-Bubble nc=3}** at the same `hits`, with non-inferiority margin Δ ≥ −0.003 (point estimate) and bootstrap-CI lower bound ≥ −0.005. Headline reported per `(model, dataset, hits)` tuple. | Best-baseline rule fixes the v1 trap of comparing only against DE-Cocktail when DE-Selection holds the only Bonf-sig win. Non-inferiority margin protects against passing on point-estimate noise. | Phase 4 Study A + baselines (60 + 144 runs) |
| **claim:C12 (single-extreme reporting separation)** | MaxContext-TopDown's `n=2` BM25 endgame resolves ranks `N−1` vs `N`; MaxContext-BottomUp's resolves ranks `1` vs `2`. Because NDCG@10 is sensitive to head-of-ranking decisions, the two variants must be reported separately on the Pareto plot, with each variant's NDCG@10 compared to a same-direction matched-hits baseline (TopDown vs TD-Heap; BottomUp vs BU-Heap). | v1 made a causal claim about NDCG impact. Reworded as a reporting-discipline rule (no causal counterfactual without a no-bypass diagnostic, which is out of scope). | Phases 4D + 4E (60 + 60 runs) |
| **claim:C13 (long-context amortization, conditional)** | Conditional on Phase 2 order-robustness (max pairwise NDCG@10 Δ across {forward, inverse, random} ≤ 0.01) and Phase 5 passage-length sweep, MaxContext-DualEnd's signal-extraction-per-LLM-call scales from `c=4` (DE-Cocktail mean 546 calls) to `c ∈ predeclared` (MaxContext mean `floor(c/2)` calls) without quality regression beyond the C11 non-inferiority margin. | v1's "without catastrophic long-context attention degradation" was vague. Reworded as a quantitative conditional tied to a measurable margin. The 3-ordering smoke gate is treated as evidence-against-instability, not as proof of stability. | Phase 2 (16 runs) + Phase 5 (96 runs) |
| **claim:C14 (position bias at scale, descriptive)** | Per-position selection frequencies under MaxContext (numeric labels 1..N) are reported separately from the existing `w=4` letter-scheme frequencies underlying claim:C5. **Whether the `dual_worst` primacy reversal extends to `w=50` is an open question to be reported, not a presupposed claim.** | v1 implied C14 would extend C5. Reworded as descriptive — Phase 4 produces the numbers; the interpretation depends on which of three patterns emerges (extends-C5, contradicts-C5, qualitatively different). | Phase 4 JSONL logs + new analysis stratification (extension of `analysis/position_bias.py`) |
| **claim:C10 (branch-specific rewrite)** | C10's framing rewrite is **branch-specific** (see §3.5 outcome matrix). For branch A: "ICTIR-floor / EMNLP-ceiling, gated on MaxContext-DualEnd effectiveness across multiple model families." For branch C (single-extreme wins): "MaxContext at long context contradicts the joint-elicitation-is-the-contribution claim of original C8; reported as an empirical finding." For branch D: "saturation / latency frontier result at smaller pool." For branch E: original C10 stands. | Codex round-1 HIGH-2 caught that v1's rewrite was branch-agnostic. Now branch-bound. | Phase 4 results (branch determines wording) |

**Stage 1 doc work** lands provisional stubs for C11-C14 (with `status: provisional, predeclared_2026-04-28`) before Phase 4 launches, so the experiments are formally predeclared against measurable hypotheses. Stage 2 doc work lands the C10 rewrite + status upgrades after Phase 4 picks the branch.

## 9. Decision Gate and Execution Order (aligned with §5.3 + §11.1 + §3.5)

**Step 0 (now).** This plan exists; no commits. No SLURM submissions, no doc changes, no code changes.

**Step 1 — Launch expanded gate (mandatory, ~28 runs, ~56 cluster-hours).**
- Phase 2 core: order pilot at `pool=50`, 2 models × 2 datasets × 3 orderings = 12 runs (`experiments/run_maxcontext_dualend_order.sh`).
- Phase 2 sub-criterion 3: pool=20 sanity at `pool=20`, 2 models × DL19 × {forward, random} = 4 runs.
- Phase 3 matched-hits regression at `pool=50`, 4 pairs × 3 methods (MaxContext-DualEnd + DE-Cocktail + DE-Selection) = 12 runs (`run_maxcontext_dualend.sh`, `run_dualend_bubblesort.sh`, `run_dualend_selection.sh`). The DE-Selection runs are required for the best-of {DE-Cocktail, DE-Selection, TD-Bubble} comparison rule from claim:C11.
- Exact commands: see §11.1.

**Step 2 — Evaluate the §5.3 decision tree.**
- All §11.1 pass criteria satisfied (Phase 2 order stability ≤ 0.01 + position-bias diagnostic + pool=20 sanity, Phase 3 best-of non-inferiority ≥ 3 of 4 pairs) → proceed to Step 3 at `pool=50`.
- Phase 2 fails order stability OR Phase 3 fails non-inferiority at `pool=50` → trigger smaller-pool fallback (Step 2b).
- Both fail at `pool=50` → trigger smaller-pool fallback (Step 2b).

**Step 2b — Smaller-pool fallback (contingent, ~20 runs, ~36 cluster-hours).**
- Phase 2 at `pool=20`: 2 models × 2 datasets × 3 orderings = 8 runs.
- Phase 3 at `pool=20`: 4 pairs × 3 methods = 12 runs.
- Pass criteria: same as Step 1, evaluated at `pool=20`. Pass → §3.5 Branch D (saturation result), proceed to Step 3 at `pool=20`. Fail → §3.5 Branch E (abandon pivot); current ICTIR-conservative paper stands; idea:004/005/006 may still be valuable; **stop**.

**Step 3 — Stage 1 doc work (preregistration; doc-only, no code).**
- Write provisional `research-wiki/claims/C{11,12,13,14}_*.md` stubs with `status: provisional, predeclared_2026-04-28`. Each stub uses the measurable phrasings from §8, predeclared against the chosen `pool_size` from Step 2 / 2b.
- Update `IDEA_007.md` §7 risk #6 to flag staged pivot is in-flight.
- Commit scope-deferrals: do **not** launch idea:004 (Qwen 24 + likelihood 4 = 28 pending), idea:005 (12 pending), idea:006 (12 pending). Existing 6 idea:004 flan-t5-xl results held on disk for §6 ablation use.
- **Do not yet rewrite** `research-wiki/claims/C10_framing_ictir_conservative.md`, `research-wiki/PAPER_PLAN.md`, `research-wiki/NARRATIVE.md`, `research-wiki/RESEARCH_BRIEF.md`, or `research-wiki/claims/C9_pareto_frontier.md` — those wait until Phase 4 picks the §3.5 branch.

**Step 4 — Launch Phase 4 production (~324 runs, ~648 cluster-hours).**
- Study A pool sweep (DualEnd): 60 runs.
- Phase 4D TopDown pool sweep: 60 runs.
- Phase 4E BottomUp pool sweep: 60 runs.
- Phase 4B baselines at matched `hits ∈ {pool_size_low, mid, high}` (where `pool_size_low` is `10` if Step 2 passed at pool=50, or `20` if Step 2b passed at pool=20; mid = `30`; high = pool tested) × 6 Qwen × 2 datasets × 4 methods (TD-Heap, TD-Bubble, DE-Cocktail nc=3, DE-Selection nc=3) = 144 runs.
- Run in parallel where SLURM allows.

**Step 5 — Pick the §3.5 branch.**
- Compare MaxContext-DualEnd / TopDown / BottomUp at predeclared `pool_size` against best-of {DE-Cocktail, DE-Selection, TD-Bubble} matched-hits baselines on (NDCG@10, comparisons, total tokens, wall-clock).
- Apply §3.5 branch decision: A (joint amortization wins), B (whole-pool selection wins regardless of mode), C (single-extreme is the surprise), D (smaller-pool saturation result; only entered via Step 2b).
- The branch decision is **predeclared and binding**; no post-hoc title or claim selection.

**Step 6 — Launch Phase 5 passage-length sweep (~96 runs, ~192 cluster-hours).**
- Study B treatment + DualEnd-nc3 control arm at the predeclared `pool_size`.
- Updates claim:C13 evidence (long-context amortization).

**Step 7 — Stage 2 doc work (branch-specific commit).**
- Per §5.2 Stage 2: rewrite `research-wiki/claims/C10_framing_ictir_conservative.md` to the branch-specific wording from §8 (branch A / B / C / D).
- Rewrite `research-wiki/PAPER_PLAN.md`, `research-wiki/NARRATIVE.md`, and `research-wiki/RESEARCH_BRIEF.md` to the branch's title / RQ / narrative.
- Promote C11-C14 stubs from `provisional` to `supported` / `strongly_supported` with measured `evidence_strength`.
- Update `research-wiki/claims/C9_pareto_frontier.md` to add the chosen MaxContext frontier point(s).
- Update `EXPERIMENT_PLAN.md` Pareto takeaway block.

**Step 8 — Update Obsidian vault + project memory with committed framing.**
- `wiki/concepts/MaxContext Reranking.md` — replace "Reporting Caveat" with branch-specific framing.
- `wiki/log.md` — append commit entry.
- `~/.claude/projects/.../memory/MEMORY.md` — add a project memory entry pointing at the committed branch.

**Step 9 — Begin paper drafting against the chosen branch's RQ structure.**

**Optional parallel track (anywhere from Step 4):** Llama / Mistral sanity smokes per §6.3 (2 runs). If green, run per-family Phase 2 (12 runs each); if green, expand Phase 4 matrix per family. Required code changes per §6.2 land **before** the family-expansion sanity smokes, not before Step 1.

**Stop conditions** (any of these halt the pivot):
- Step 2 + 2b both fail → §3.5 Branch E. Current paper stands.
- Step 5 picks Branch C and the user decides "joint elicitation contradicting C8 at scale" is more disruptive than the pivot is worth → revert Stage 1 stubs, keep idea:004/005/006 deferrals as still-deferred future work.
- Codex audit on Stage 2 doc rewrites returns MAJOR_REDESIGN → escalate before commit.

## 10. Risks

| Risk | Mitigation |
|---|---|
| **Long-context attention degradation at `pool=50`.** Primary risk per `paper:liu2024_lost_in_middle` and `paper:hutter2025_positional_rag`. Could undermine the headline efficiency claim. | Phase 2 order pilot is the designed diagnostic; Study B's control arm (DualEnd-nc3 pl-sweep) isolates pool-size vs passage-length effects. |
| **Numeric-label parse fragility at N=50.** `0` failure was caught + fixed; other model families may have different failure modes. | Llama / Mistral sanity smokes; expand `scripts/check_maxcontext_invariants.py` fixtures per family. |
| **Order sensitivity at large windows.** If forward / reverse / random orderings disagree by >0.01 NDCG@10, pool=50 may not be defensible. | IDEA_007 §5.3 contingency: restrict to pool=20 if needed. |
| **Statistical fragility of MaxContext-DualEnd vs DE-Cocktail.** TREC DL has 43 / 54 queries; bootstrap CIs are wide. Even directional wins may not be Bonferroni-significant. | Same statistical fragility constraint as claim:C6; report Bonferroni-corrected significance honestly; do not overstate. |
| **Qwen-only paper if Llama / Mistral expansion doesn't land.** Tighter generalization claim. | Acceptable for ICTIR; explicitly limit scope in §7. EMNLP/ARR upgrade gated on multi-family expansion. |
| **Doc churn if pivot reverts.** All §5.2 doc rewrites are dated; if Phase 2/3 fail and pivot reverts, the rewrites need to be reverted too. | Stage doc rewrites in Step 7 only after Phase 2+3 + Phase 4 land. Do not commit doc changes proactively. |
| **Reviewer pushback on scope-deferral justification.** Reviewers may ask why idea:004/005/006 weren't fully evaluated. | Per §7 scope-deferral framing: idea:004 partial Pareto-axis overlap with MaxContext but the gating mechanism is orthogonal (future-work); idea:005 targets gap:G2 (position bias under joint prompts), tied to claim:C14 future work; idea:006 targets gap:G3 (asymmetric best-vs-worst competence) via worst-as-local-regularizer mechanism, distinct from joint elicitation. Existing 6 flan-t5-xl idea:004 results retained as §6 ablation. **Do not say "supersedes"** — the ideas address different gaps. |
| **Token-axis claim trap.** MaxContext uses *more* prompt tokens than DE-Cocktail. claim:C9 token-axis cannot be improved. | **Report all five cost axes** (`Avg comparisons`, prompt tokens, completion tokens, total tokens, wall-clock) in the main efficiency table. **Frame the claim as "fewer sequential LLM calls and lower wall-clock latency, not token efficiency."** Saying the token-axis claim is "absent" invites reviewer pushback (Codex round-1 HIGH-7); the right response is transparent reporting plus careful framing. |

## 11. Verification Plan

### 11.1 Pre-commit verification (Phase 2 + Phase 3, expanded per §5.3)

```bash
# From cluster login node
cd /scratch/project/neural_ir/hang/llm-rankers

# Phase 2 core: order pilot at pool=50 (12 runs)
for MODEL in Qwen/Qwen3-4B Qwen/Qwen3.5-9B; do
  for DATASET in dl19 dl20; do
    for ORDER in forward inverse random; do
      sbatch experiments/run_maxcontext_dualend_order.sh \
        "$MODEL" "$DATASET" runs/bm25/run.msmarco-v1-passage.bm25-default."$DATASET".txt \
        results/maxcontext_dualend cuda generation 50 512 "$ORDER"
    done
  done
done

# Phase 2 sub-criterion 3: pool=20 sanity (4 runs)
for MODEL in Qwen/Qwen3-4B Qwen/Qwen3.5-9B; do
  for ORDER in forward random; do
    sbatch experiments/run_maxcontext_dualend_order.sh \
      "$MODEL" dl19 runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
      results/maxcontext_dualend cuda generation 20 512 "$ORDER"
  done
done

# Phase 3: matched-hits regression check (4 pairs × 3 methods = 12 runs at pool=50)
for MODEL in Qwen/Qwen3-8B Qwen/Qwen3.5-9B; do
  for DATASET in dl19 dl20; do
    sbatch experiments/run_maxcontext_dualend.sh \
      "$MODEL" "$DATASET" runs/bm25/run.msmarco-v1-passage.bm25-default."$DATASET".txt \
      results/maxcontext_dualend cuda generation 50 512
    # DE-Cocktail at matched hits=50 nc=3
    sbatch experiments/run_dualend_bubblesort.sh \
      "$MODEL" "$DATASET" runs/bm25/run.msmarco-v1-passage.bm25-default."$DATASET".txt \
      results/de_cocktail cuda generation 3 10 50 512
    # DE-Selection at matched hits=50 nc=3 (for best-of comparison per claim:C11)
    sbatch experiments/run_dualend_selection.sh \
      "$MODEL" "$DATASET" runs/bm25/run.msmarco-v1-passage.bm25-default."$DATASET".txt \
      results/de_selection cuda generation 3 10 50 512
  done
done
```

**Pass criteria** (must all hold; per §5.3):

1. All jobs complete; zero parse-fallback **aborts** (refusal-no-op fallbacks ≤ 5% of `Avg comparisons` is fine).
2. **Phase 2 order stability:** max pairwise NDCG@10 Δ across {forward, inverse, random} ≤ 0.01 for both models on both datasets at pool=50.
3. **Phase 2 position-bias diagnostic** (new analysis stratification):
   - `selected_position_freq[i]` distribution per ordering (for claim:C14 evidence).
   - `selected_doc_stability` between forward and inverse: ≥ 50% of queries pick the same docid (sanity floor).
   - `parse_fallback_location_distribution`: no concentration > 30% at any single position.
4. **Phase 2 pool=20 sanity:** pool=20 max pairwise Δ NDCG@10 across {forward, random} ≤ 0.01 (smaller pools should be at least as stable).
5. **Phase 3 non-inferiority (best-of):** for ≥ 3 of 4 (model, dataset) pairs, MaxContext-DualEnd NDCG@10 ≥ best-of {DE-Cocktail, DE-Selection} NDCG@10 with Δ ≥ −0.003 (point estimate) and bootstrap-CI lower bound ≥ −0.005.
6. **Phase 3 cost reduction:** MaxContext-DualEnd `Avg comparisons` ≤ DE-Cocktail at matched hits in all 4 pairs.

### 11.2 Mid-execution verification (Phase 4)

After Phase 4 Study A + baselines complete (~204 runs):

1. `analysis/quality_cost_pareto.py` rerun including MaxContext-DualEnd. MaxContext should land in or near the empty region between TD-Bubble (300 cmp) and DE-Cocktail (546 cmp) on the comparisons-axis frontier.
2. `analysis/significance_tests_pairwise.py` rerun including MaxContext-DualEnd vs **best-of {DE-Cocktail, DE-Selection, TD-Bubble}** at matched hits per claim:C11. Report Bonferroni-corrected p-values per (model, dataset, baseline) tuple.
3. Per-query parse-fallback rate ≤ 5% (sanity threshold per `research-wiki/FINDINGS.md` 2026-04-26 entry).

### 11.3 Post-commit verification (paper draft)

1. Every §3 motivation claim cites a completed experiment (no forward-references).
2. Every §5 main result has a Bonferroni-corrected p-value and a bootstrap CI.
3. claim:C10 rewrite is internally consistent across `research-wiki/claims/C10_framing_ictir_conservative.md`, `IDEA_007.md`, `research-wiki/PAPER_PLAN.md`, and `research-wiki/NARRATIVE.md`.
4. Any T5 reference in §5 is flagged as motivation-only (not main result).
5. All five cost axes (comparisons, prompt tokens, completion tokens, total tokens, wall-clock) are reported in the efficiency table; framing claim is "fewer sequential calls + lower wall-clock latency", not "token efficiency" (per §10 risk + Codex round-1 HIGH-7).
6. Vault `MaxContext Reranking` concept page reflects the new headline framing.
7. `.planning/repo_walk/RESEARCH.md` updated to reflect new RQ structure.

## 12. Out of Scope

- **Code changes beyond §6.2 (Llama/Mistral allowlist + `CAUSAL_MODEL_TYPES` extension + constructor smoke test).** The MaxContext implementation is stable as of HEAD.
- **Repair-prompt retry path.** Deferred per IDEA_007 §3.2 for Qwen given current ≤5% fallback rates. **Becomes in-scope if a new model family's sanity smoke (§6.3) shows parse-fallback rate above 5%** of `Avg comparisons` — repair prompt would then be required for that family's production runs (Codex round-1 LOW-2).
- **Prefix-allowed-tokens generation.** Already reverted; not revisited.
- **Pool sizes > 50.** Out of scope per IDEA_007 §10.
- **BEIR coverage expansion.** Pending Qwen3.5-9B BEIR runs (24 from `Need_to_Run.txt`) are independent of the pivot decision and continue on their own track.
- **Rank-R1 sub-project integration.** Separate paper; not part of this pivot.
- **No-bypass diagnostic for claim:C12.** A counterfactual ("what would NDCG@10 be if `n_docs=2` BM25 endgame were replaced with an LLM call?") would strengthen the causal phrasing of v1's C12. Out of scope: claim:C12 is reworded as a reporting-discipline rule, not a causal claim.
- **T5 leakage guard.** Reminder, not new scope: any §5 main-result claim involving MaxContext is **Qwen-generation-only** unless Llama/Mistral expansion lands. T5 evidence is restricted to §3 motivation. Paper-writing must enforce this; reviewers will check.

## 13. Audit Trail

- **v1** (2026-04-28): drafted from session discussion. NEEDS_REVISION verdict from Codex round 1 (3 BLOCKER, 5 HIGH, 4 MEDIUM, 2 LOW).
- **v2** (2026-04-28): addresses round-1 findings.
  - BLOCKER-1: Phase 3 expanded from 1 pair to 4 pairs (Qwen3-8B + Qwen3.5-9B × DL19 + DL20); non-inferiority margin Δ ≥ −0.003 added; pass requires ≥ 3 of 4 pairs.
  - BLOCKER-2: smaller-pool fallback gate added; pool=20 added to baseline grid as predeclared contingency.
  - HIGH-1: §3.5 outcome branch matrix added (5 branches: A/B/C/D/E); each branch's title/RQ/surviving-claims predeclared.
  - HIGH-2: rewrite timing split into Stage 1 (preregistration before Phase 4) and Stage 2 (commit after Phase 4 picks branch).
  - HIGH-3: best-of {DE-Cocktail, DE-Selection, TD-Bubble} baseline rule replaces single-baseline DE-Cocktail comparison.
  - HIGH-4: idea:004/005/006 reframed as "scope-deferrals" with gap-specific future-work tie-ins; idea:005 → claim:C14 (G2), idea:006 → same-call worst-signal mechanism (G3).
  - HIGH-5: token-axis stance changed from "absent" to "report all five cost axes; frame claim as comparisons + wall-clock, not token efficiency."
  - MEDIUM-1: position-bias-at-scale diagnostic added to Phase 2 (selected_position_freq, selected_doc_stability, parse_fallback_location_distribution).
  - MEDIUM-2: Llama/Mistral code scope corrected — `CAUSAL_MODEL_TYPES` (`setwise.py:27-28`) extension now mandatory, not just `MAXCONTEXT_ALLOWED_MODEL_TYPES`; constructor smoke test in `scripts/check_maxcontext_invariants.py` added.
  - MEDIUM-3: claims C11-C14 rewritten with measurable margins, conditional phrasing, best-baseline rule.
  - MEDIUM-4: §7.5 Reconciled Compute Accounting added; v1's run counts and cluster-hour numbers were inconsistent.
  - LOW-1: T5 leakage guard added to §12 out-of-scope.
  - LOW-2: repair-prompt out-of-scope made conditional on per-family sanity fallback rate.
- **v3** (this document, 2026-04-28; Codex-ACCEPT round 3): addresses round-2 findings. Strategic content unchanged.
  - BLOCKER (round-2): §9 fully rewritten to align with §5.3 expanded gate, §3.5 branch matrix, and §5.2 Stage 1/Stage 2 split. v1 execution order (12 + 1, immediate cuts, single-stage doc rewrite) replaced with the §5.3 decision tree + Stage 1/Stage 2 split.
  - MEDIUM (round-2): Phase 3 run count reconciled. Best-of {DE-Cocktail, DE-Selection, TD-Bubble} requires DE-Selection per pair, so Phase 3 = 4 pairs × 3 methods = 12 runs (~24 cluster-hours), not 8 runs. §5.3, §7.5, §9 aligned to 12. Smaller-pool fallback also reconciled to 4 pairs × 3 methods = 12 runs (~24 cluster-hours).
  - LOW (round-2): stale §4 wording aligned to best-of baseline rule + token-axis transparent reporting; stale §10 wording replaced "supersedes" with "scope-defers" and gap-specific framing.

## 14. Reference Paths

- `IDEA_007.md` — current MaxContext design spec
- `MAX_CONTEXT_EXPERIMENT_PLAN.md` — operator command sheet for staged matrix
- `EXPERIMENT_PLAN.md` — current paper experiment plan (will be rewritten in Step 7)
- `research-wiki/PAPER_PLAN.md` — current paper structure (will be rewritten)
- `research-wiki/NARRATIVE.md` — current narrative (will be rewritten)
- `research-wiki/RESEARCH_BRIEF.md` — current paper brief (will be rewritten)
- `research-wiki/FINDINGS.md` — running discovery log (append, do not rewrite)
- `research-wiki/claims/C10_framing_ictir_conservative.md` — framing constraint (will be rewritten)
- `research-wiki/claims/C9_pareto_frontier.md` — Pareto claim (will add MaxContext frontier point)
- `research-wiki/ideas/idea_007_maxcontext_dualend.md` — idea page (status update only)
- `llmrankers/setwise_extended.py:23` — `MAXCONTEXT_ALLOWED_MODEL_TYPES` (optional Llama/Mistral extension)
- `.planning/repo_walk/{CODE,INFRA,RESEARCH}.md` — comprehensive repo synthesis (2026-04-28)
- `.planning/idea_007_topdown_parser/` — recent parser-fix audit-loop artifacts (Codex investigation + 3 audit rounds + implementation log)
