# EMNLP_SHORT_PLAN.md — MaxContext short-paper plan (v2)

> **Status:** v1 — addresses Codex round-1 NEEDS_MAJOR_REVISION findings (2026-05-02). Sole contribution = MaxContext family. Paper is a 4-page ACL/EMNLP short submission. All pre-idea:007 narrative is dropped.
> **Companion:** [`EMNLP_SHORT_EXPERIMENT_PLAN.md`](EMNLP_SHORT_EXPERIMENT_PLAN.md)
> **Venue rules:** ACL Rolling Review short — 4 pages of content; references unlimited; Limitations required after Conclusion (no page cost, but cannot carry new results); Ethics optional (no page cost); appendices do not count toward limit but reviewers are not required to read them — every load-bearing claim must live in main text.

---

## 1. One-sentence contribution

> *We introduce **MaxContext**, a zero-shot setwise reranker whose every LLM call prompts over the current whole rerank pool ($N \le 50$), and show that asking for the best **and** the worst document jointly per call (DualEnd) reranks any pool in $\lfloor N/2 \rfloor$ **LLM calls** — roughly **half** the **LLM-call** count of best-only or worst-only whole-pool selection — without losing nDCG@10 on TREC DL and BEIR.*

The half-cost framing is the headline, but it is framed strictly on the **LLM-call axis**. Concretely, at the matched operating point $W = k = N$:

| Variant | LLM calls | BM25 bypass | Serial decision steps | Pool shrinkage per call |
|---|---|---|---|---|
| **MaxContext-DualEnd** | $\lfloor N/2 \rfloor$ | 0 | $\lfloor N/2 \rfloor$ (+ 1 trivial placement when $N$ odd) | 2 |
| **MaxContext-TopDown** | $N{-}2$ | 1 | $N{-}1$ | 1 |
| **MaxContext-BottomUp** | $N{-}2$ | 1 | $N{-}1$ | 1 |

For $N=10$: $5$ vs $8$ LLM calls (5/8 ≈ $0.625$); for $N=50$: $25$ vs $48$ LLM calls (25/48 ≈ $0.521$). The "half" descriptor is approximate at small $N$ and asymptotically tight at large $N$. **Avoid the unqualified phrase "half the dependent calls"** — TopDown / BottomUp emit one *deterministic* BM25 decision at the very end which is not an LLM call. The plan and paper consistently say "LLM calls".

For odd $N$ (e.g. $N=11$), the DualEnd loop runs $\lfloor N/2 \rfloor$ times and a single unpaired document is placed without an additional LLM call — the formula stays $\lfloor N/2 \rfloor$, not $\lceil N/2 \rceil$.

The single-extreme variants (TopDown / BottomUp) are *controls* against which the half-cost claim is measured, not co-contributions.

## 2. Research questions

Two RQs, both answerable from the same matrix.

- **RQ1 (cost-quality, primary).** At $W = k = N$, is MaxContext-DualEnd's nDCG@10 *non-inferior* to MaxContext-TopDown and MaxContext-BottomUp under a predeclared margin (see §3), at the same hits and across the Tier-1 (model, dataset) cells? Symmetrically: is the LLM-call gap deterministic and as large as predicted?
- **RQ2 (generalisation).** Does the half-cost gap with non-inferior quality hold across (a) Tier-1 four-checkpoint multi-family lineup spanning Qwen3.5, Llama-3.1, Mistral / Ministral, and (b) six datasets — TREC DL19, DL20, BEIR-{dbpedia, nfcorpus, scifact, trec-covid}?

Order-robustness, position-bias-at-scale, passage-length sweeps, and the Tier-2 size-sweep (smaller / larger Qwen3.5, additional Mistral checkpoints, BEIR-touche2020, BEIR-fiqa) are appendix-only.

## 3. Headline claims (preregistered, measurable)

Four claims, each preregistered with a margin and a statistical decision rule.

| ID | Statement | Margin / metric | Test |
|---|---|---|---|
| **H1** | At $W{=}k{=}N{=}50$, MaxContext-DualEnd's nDCG@10 is *non-inferior* to MaxContext-TopDown (and separately to MaxContext-BottomUp) on each Tier-1 (model, dataset) cell. Define $\Delta = \text{nDCG@10}(\text{DualEnd}) - \text{nDCG@10}(\text{baseline})$ and margin $\delta = 0.01$. Test $H_0: \Delta \le -\delta$ vs $H_1: \Delta > -\delta$ on per-query $\Delta$; reject $H_0$ at $\alpha{=}0.05$. Reported alongside paired-bootstrap 95% CI on $\Delta$. | one-sided paired permutation, 10K samples; per-cell verdict reported, multiplicity-corrected per H1$_\text{TD}$ and H1$_\text{BU}$ comparator families separately |
| **H2** | MaxContext-DualEnd uses $\lfloor N/2 \rfloor$ mean LLM calls per query; MaxContext-TopDown and MaxContext-BottomUp use $N{-}2$ mean LLM calls plus one BM25 bypass. | exact, deterministic from algorithm — no test required. **Reported descriptively** in the cost-axes table. | n/a |
| **H3** | The H1 verdict is positive on $\ge 80\%$ of Tier-1 cells (= $\ge 20$ out of 24 model-dataset pairs at $N{=}50$), separately for H1$_\text{TD}$ and H1$_\text{BU}$. | per-cell H1 verdict, BH-FDR adjusted within each comparator family of 24 cells (so two FDR runs total) | BH-FDR at $q{=}0.05$ |
| **H4** | Across 10 independent re-runs on Qwen3-4B / DL19 with the **input-order seed** varying across runs, the **input-order stability** statistics for nDCG@10 satisfy: per-cell SD $\le 0.005$, max-min range $\le 0.015$, and worst-pair $|\Delta| \le 0.015$, at every $N \in \{10,20,30,40,50\}$. | per-cell SD, range, worst-pair $|\Delta|$ | empirical, no test |

Notes on H1's hypothesis direction (Codex round-2 §4):

- $\Delta > 0$ means DualEnd is *better*; $\Delta < 0$ means DualEnd is *worse*. Non-inferiority asks "is DualEnd not meaningfully worse than baseline by more than $\delta$?" — i.e. is $\Delta > -\delta$? So the alternative is $H_1: \Delta > -\delta$, and the null is $H_0: \Delta \le -\delta$. Rejecting $H_0$ → DualEnd is non-inferior to baseline at margin $\delta$. (This is the textbook non-inferiority framing; the v0 / v1 wording inverted it and is corrected here.)
- The per-cell test is one-sided. Failing to reject $H_0$ is *not* evidence of inferiority; it's an inconclusive cell, annotated as ▲ in Table 1.

Notes on H3 multiplicity:

- Two comparator families (H1$_\text{TD}$ and H1$_\text{BU}$), each with 24 cells. BH-FDR is applied within each family at $q=0.05$. Bonferroni is reported in the appendix table for transparency. Holm-Bonferroni is documented as an alternative we considered (Codex round-2 §4 prefers Holm-Bonferroni for *per-cell* H1 claims; we use BH-FDR because H3 asserts $\ge 80\%$ of cells, an FDR-shaped claim).
- The aggregate H3 verdict is "supported" iff $\ge 20$ of 24 cells reject $H_0$ in each comparator family.

H4's variance source is **input-order seed plumbing** — see [`EMNLP_SHORT_EXPERIMENT_PLAN.md`](EMNLP_SHORT_EXPERIMENT_PLAN.md) §3 / Phase 3 for the prerequisite code change. Renamed from the v1 wording "run-to-run stability" to "input-order stability" per Codex round-2 §5.

H2 is a deterministic property of the algorithm; the half-cost numbers go into the abstract / introduction without a hypothesis test. H1 + H3 carry the empirical load. H4 is the methodology guard.

## 4. Branch resilience (outcome scenarios)

Seven branches, predeclared. The paper architecture survives all of them with under 10% rewrite.

- **Branch A — H1 + H3 + H4 all supported.** Headline framing unchanged. *(Most-favourable.)*
- **Branch B — H1 holds for some $N$ but fails at $N{=}50$.** Long-context degradation hypothesis. Title softens to "Half-Cost Reranking up to a Pool-Size Saturation Point"; main-paper Pareto figure highlights the saturation point.
- **Branch C — H1 holds on Qwen but fails on Llama / Mistral.** Family-fragility hypothesis. Headline retained as a Qwen-class result; non-Qwen runs reported in appendix as evidence of family-specific parser / chat-template fragility, with §10 limitation widened.
- **Branch D — Call count win confirmed but wall-clock / total-tokens not improved.** Pareto figure x-axis switches to wall-clock / per-query token cost; framing becomes "matched-quality LLM-call reduction; total wall-clock and tokens reported transparently and depend on per-call latency".
- **Branch E — H1 inconclusive (CIs too wide).** Title becomes a measured-tradeoff paper; abstract reports point estimates and CIs without a non-inferiority verdict; H3 reframed as "directional on $X / 24$ cells, with CIs that do not exclude either direction on $24-X$ cells".
- **Branch F — One single-extreme variant beats DualEnd outright.** Headline becomes "Whole-Pool Single-Extreme Selection Outperforms Joint Elicitation at Long Context"; H1 contradicted as an empirical finding, half-cost claim retained as a separate property of the dominated method.
- **Branch G — H4 fails (input-order stability outside thresholds).** Methodology limitation; main-paper numbers are reported as means over 5 runs with explicit SD / range bars, not as point estimates. Stability table moves to main paper.

## 5. Section plan — page budget (4 pages content)

The v0 plan over-allocated 4-page content; this version trims aggressively per Codex round-1 §1.

| § | Section | Words (target) | Pages (target) | Content |
|---|---|---|---|---|
| 1 | Title + Abstract | 150 words | 0.4 (title block + abstract on first column above section 1) | One-paragraph hook with H1 + H2 numbers (filled at camera-ready) |
| 2 | Introduction | 500 words | 0.6 | Setwise call-cost motivation; introduce MaxContext + three variants; preview H2 numbers; 3-bullet contributions list |
| 3 | Related Work | 250 words | 0.3 | One paragraph each: setwise lineage (Setwise + extensions: TourRank, Rank-R1, Setwise Insertion), long-context attention + position bias (lost-in-the-middle, found-in-the-middle), efficiency reporting (E2R-FLOPs). One paragraph contrast vs RankGPT-listwise *(call-count framing of MaxContext is not vs listwise)*. |
| 4 | MaxContext (method) | 400 words + algorithm box (5–7 lines) | 0.7 | Algorithm 1 (whole-pool DualEnd shrink-by-2 loop with BM25 endgame); three variants with their LLM-call counts; numeric labels 1..N; hard invariants in one inline sentence; out-of-scope: $N>50$, T5 / likelihood path. |
| 5 | Experimental Setup | 250 words + 1 tiny model table | 0.4 | Datasets (DL19/20 + 4 BEIR), Tier-1 four-checkpoint lineup with rationale, $N \in \{10,20,30,40,50\}$, $W{=}k{=}N$, scoring=generation, statistical protocol (one-sided paired permutation + BH-FDR + paired-bootstrap CIs, 10K samples each), 10× stability sanity. |
| 6 | Results | 350 words + Table 1 + Figure 1 | 0.9 | **Table 1 = compressed signed-$\Delta$ form** (per §6 below): 4 model rows × {DL19, DL20, BEIR-mean, BEIR-min} column blocks × {$\Delta_\text{TD}$, $\Delta_\text{BU}$} per block, each cell `+0.0XX✓ / ▲ / ✗` for the H1 verdict. Full 4×6×3 raw nDCG@10 + CI table moves to appendix. **Figure 1** = Pareto plot (mean LLM calls vs nDCG@10), top panel DL19+DL20 mean over $N \in \{10..50\}$, bottom panel BEIR-mean over $N \in \{10, 30, 50\}$; caption flags asymmetric $N$ sets. Two short prose paragraphs: H1 verdict, H3 verdict. |
| 7 | Conclusion | 80 words | 0.1 | One paragraph; no future-work bloat. |
| — | Limitations | 250 words | (no count) | Pool-size cap is *context-window- and passage-length-dependent* (soft, not hard); Qwen-Llama-Mistral generation-only scope; small-query TREC DL fragility; BM25-endgame position asymmetry; input-order sensitivity caveat from H4. |
| — | Ethics | optional | (no count) | Skip unless reviewer-flagged. |
| — | Appendix | full sweep tables, prompt templates, hyperparameters, position-bias-at-scale, order-pilot, pl-sweep, Tier-2 size sweep, per-BEIR-domain breakdown, 10× per-query SD, original Setwise direct-baseline smoke (§8) | (no count) | Promote anything that a reviewer might call "missing" to the appendix; never as load-bearing for a main claim. |

**Page accounting:** Title + abstract + §2 + §3 + §4 + §5 + §6 + §7 ≈ $0.4 + 0.6 + 0.3 + 0.7 + 0.4 + 0.9 + 0.1 = 3.4$ pages. Leaves $\sim 0.6$ pages of slack for typography (spacing around equations, section headers, etc.). H4 sanity micro-table moves to appendix to free this slack.

## 6. Figure / table allocation

- **Figure 1 (main, single column, two panels stacked).** Pareto plot, x-axis = mean LLM calls per query, y-axis = mean nDCG@10. Top panel: DL19+DL20 mean. Bottom panel: BEIR-mean (over the 4 Tier-1 BEIR domains). Three line series (DualEnd, TopDown, BottomUp). DL panel shows all $N \in \{10,20,30,40,50\}$; **BEIR panel shows only $N \in \{10, 30, 50\}$** to match the experiment-plan trim — the figure caption explicitly notes that the two panels use different $N$ point sets.
- **Table 1 (main, double column).** **Compressed form** (per Codex round-2 §1) — full 4×6×3 matrix moves to appendix. Rows = 4 Tier-1 models. Column blocks = {DL19, DL20, BEIR-mean, BEIR-min}. Per-block columns = $\Delta_\text{TD} = \text{DualEnd} - \text{TopDown}$ and $\Delta_\text{BU} = \text{DualEnd} - \text{BottomUp}$ at $N{=}50$. Cell entry = $\Delta$ as a signed nDCG@10 difference, with annotation ✓ / ▲ / ✗ for the H1 verdict on each $\Delta$. Reading the table: ✓ everywhere → DualEnd non-inferior, ✗ → significant gap. The full per-cell raw nDCG@10 + CIs lives in the appendix master table.
- **Algorithm 1 (main, single column).** 5-line MaxContext-DualEnd pseudocode (port from `paper/v1/introduction.tex`).
- **Appendix figures/tables.** Full 4×6×3 nDCG@10 table with CIs and verdicts; per-BEIR-domain breakdown; 10× stability per-query SD + range + worst-pair $|\Delta|$; order-robustness pilot; pl-sweep; Tier-2 size sweep; original Setwise vs MaxContext-TopDown smoke (per Codex round-1 §10); Bonferroni and Holm-Bonferroni adjusted $p$-value tables alongside BH-FDR.

## 7. What survives from `paper/v1/`

Per the Explore digest: §2 of `paper/v1/introduction.tex` (paradigms) and §4 (MaxContext framing) port nearly verbatim with compression. Entire v1 §3 motivation (BU / BiDir / DE-Cocktail) is **deleted**. Related Work shrinks from 1460 to 250 words. Experimental Setup shrinks from 2207 to 250 words + tiny table — every launch-gate / compute-hour detail moves to appendix.

## 8. Novelty positioning vs prior work

Per Codex round-1 §10, original Setwise must be cited as a *direct comparator*, not just lineage.

- **vs Setwise (Zhuang et al. 2024).** Original Setwise uses heapsort or bubblesort over $c{+}1{=}4$ windows, with letter labels A–D. MaxContext-TopDown is *not* "Setwise-Selection at $c{+}1{=}N$" because (a) MaxContext uses **numeric labels 1..N** (Setwise's letter alphabet has 23 elements and is not directly extensible past $N{>}23$), (b) **strict no-truncation** preflight + abort-on-bad-parse policies, (c) the **deterministic BM25 endgame** at the final 2-document pair, and (d) MaxContext-TopDown writes its own logging path (`type=best`, numeric `label_scheme`). An appendix smoke at $N{=}10$ on a single (model, dataset) cell directly compares Setwise-Selection at $c{+}1{=}N{=}10$ against MaxContext-TopDown to expose any latent equivalence.
- **vs TourRank.** TourRank scales setwise via tournament structure (multi-round bracket, accumulated points) at $c{+}1 \le 20$ candidates per LLM call. MaxContext changes the *window size*, not the *bracket* — orthogonal axes, can in principle stack.
- **vs Setwise Insertion.** Warm-starts from BM25 to skip redundant comparisons inside small windows. MaxContext skips the small-window assumption itself.
- **vs Rank-R1.** Uses an RL-trained reasoning chain inside a $c{+}1 = 20$ window. Different mechanism (training the comparator) on a different axis (per-call quality, not call count).
- **vs RankGPT / listwise.** Listwise asks for a permutation in **one** call, so its "calls per pool" is $1$. **MaxContext does *not* claim a call-count win against listwise.** Our framing is: MaxContext is a setwise-style alternative to listwise that emits short structured outputs (one or two integer labels per call rather than a permutation), avoids the parser fragility of listwise on long candidate lists, and beats single-extreme whole-pool setwise on call count. Listwise is a different efficiency-quality regime.
- **vs lost-in-the-middle / permutation self-consistency.** PSC mitigates positional bias but multiplies calls, undoing MaxContext's cost win. We treat order as an *empirical robustness check* (appendix order pilot at $N{=}50$, three orderings) not a mitigation strategy.
- **vs E2R-FLOPs.** We adopt the all-axes reporting recommendation (LLM calls, prompt tokens, completion tokens, total tokens, wall-clock) but frame our claim narrowly on the *LLM-call-count* and *wall-clock* axes — not on tokens, which MaxContext spends more of per call.

## 9. Risk register (short paper specific)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 4-page budget violated | medium | desk-reject | Aggressive use of appendix; prose-by-numbers page check before submission; Table 1 + Figure 1 sized to fit one column / spanning columns respectively |
| Llama / Mistral parse-fragility blows up at $N{=}50$ | medium | matrix shrinks to Qwen-only | Pre-launch sanity smokes (experiment plan §3 / Phase 1); fall back to Branch C if persistent |
| Half-cost claim contradicted on a small subset | medium | branch B / E | Predeclared branch language survives §4 |
| Reviewer asks "what about TourRank as a baseline?" | high | rebuttal cost | §8 positions explicitly; appendix has one matched-budget TourRank smoke per experiment plan §5 |
| Reviewer asks "isn't MaxContext-TopDown just Setwise selection at large $c$?" | high | direct rebuttal needed | §8 explicit difference list + appendix smoke per Codex round-1 §10 |
| Reviewer asks "why no permutation self-consistency?" | medium | rebuttal cost | One paragraph in §3 / §6 explaining the cost trade-off + appendix order pilot |
| 10× stability run reveals non-deterministic behaviour | medium | branch G | Investigate before any further launch (deterministic-algorithm flags); H4 reported as measured SD if persistent |
| BEIR-mean obscures per-domain regression | medium | reviewer pushback | Always report BEIR-min in main table alongside BEIR-mean; per-domain breakdown in appendix |
| Mistral / Ministral identifier ambiguity | high | matrix incomplete | Experiment plan §1 pins exact HF identifiers + revisions before launch |
| Submit/eval path inconsistency (`ws-3` vs `ws-4`) | high | broken eval | Already on the pre-launch checklist (experiment plan §7); fix before any Phase 2 / 3 launch |

## 10. Out-of-scope (firmly)

- Any pre-idea:007 method as a result, not even as a baseline. The v2 paper's baseline is **MaxContext-TopDown** and **MaxContext-BottomUp**; not TD-Heap, TD-Bubble, DE-Cocktail, DE-Selection. Matched-hits comparisons against the small-window family belong to the long-paper version, not v2.
- T5 / encoder-decoder backbones (likelihood-scoring path collapses joint elicitation to a best-only proxy).
- Pools $> 50$.
- Permutation self-consistency.
- TourRank / BlitzRank as primary baselines (different budget knobs; appendix-only smoke).
- Listwise (RankGPT-style) as primary baseline (different efficiency regime).
- Tier-2 size sweep results in main paper (appendix only).
- Tier-2 BEIR domains (touche2020, fiqa) in main paper (appendix only).

## 11. Audit trail

- v0 (2026-05-02 morning) — initial draft.
- v1 (2026-05-02 afternoon) — addresses Codex round-1 NEEDS_MAJOR_REVISION:
  - §1: explicit LLM-call accounting table; "half the LLM calls" not "half the dependent calls"; concrete N=10 / N=50 breakdown.
  - §3: H1 reframed to one-sided paired permutation test against $\delta = -0.01$ (not $p > 0.05$).
  - §3: H3 multiplicity correction switched to BH-FDR.
  - §4: outcome-branch matrix expanded from 3 to 7 (added long-context-saturation, family-fragility, wall-clock-not-improved, statistically-inconclusive, stability-fail branches).
  - §5: Results section budget revised to 0.9 pages with main Table 1 + Figure 1 spec; Conclusion compressed to 0.1.
  - §8: original Setwise added as direct comparator with explicit difference list + appendix smoke.
  - §8: RankGPT framing corrected — MaxContext does *not* claim a call-count win against listwise.
  - §10: pool-size limitation softened ("context-window- and passage-length-dependent").
  - §9: risk register adds Setwise-equivalence reviewer ask, submit/eval path mismatch, identifier ambiguity.
- v2 (2026-05-02 evening) — addresses Codex round-2 NEEDS_MAJOR_REVISION:
  - §1: DualEnd LLM-call formula corrected from $\lceil N/2 \rceil$ to $\lfloor N/2 \rfloor$ (matches the algorithm for odd $N$; even-$N$ values unchanged).
  - §3 H1: **non-inferiority hypothesis direction corrected.** Now $H_0: \Delta \le -\delta$ vs $H_1: \Delta > -\delta$ — rejecting $H_0$ supports non-inferiority. v1 had the direction inverted.
  - §3 H3: explicit two-comparator multiplicity (FDR run separately for H1$_\text{TD}$ and H1$_\text{BU}$), each over 24 cells.
  - §3 H4: renamed "run-to-run stability" → "input-order stability"; predeclared SD + max-min range + worst-pair $|\Delta|$ thresholds (Codex round-2 §5).
  - §6: Table 1 redesigned as a **compressed signed-$\Delta$ form** (4 models × {DL19, DL20, BEIR-mean, BEIR-min} × {$\Delta_\text{TD}$, $\Delta_\text{BU}$}) to fit 4-page budget; full 4×6×3 nDCG@10 + CI matrix moves to appendix.
  - §6 Figure 1: BEIR panel uses only $N \in \{10, 30, 50\}$ to match experiment-plan trim; caption explicitly flags the asymmetric $N$ sets.

---

*Companion:* [`EMNLP_SHORT_EXPERIMENT_PLAN.md`](EMNLP_SHORT_EXPERIMENT_PLAN.md) — concrete model lineup with pinned HF identifiers, dataset matrix, statistical protocol, 10× stability plan, compute budget recomputed against actual BEIR query counts, and Phase-by-Phase staging.
