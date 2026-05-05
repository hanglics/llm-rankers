# EMNLP_SHORT_EXPERIMENT_PLAN.md — experiments for the v2 short paper

> **Status:** v1 — addresses Codex round-1 NEEDS_MAJOR_REVISION findings (2026-05-02). Companion to [`EMNLP_SHORT_PLAN.md`](EMNLP_SHORT_PLAN.md).
> **Goal:** support claims H1–H4 from the short-paper plan with the smallest matrix that survives reviewer pushback in 4 pages.
> **Hardware target:** NVIDIA H100 80 GB on Vast.ai, single-GPU per job.

---

## 1. Model lineup (Tier 1 = main paper, Tier 2 = appendix only)

The user's nine-checkpoint lineup is generous for a short paper. Splitting into a **Tier 1** subset for the main-paper headline and a **Tier 2** for the appendix keeps the 4-page narrative defensible.

| Family | Tier 1 (main paper) | Tier 2 (appendix) | Pinned HF identifier (revision pinned at launch) |
|---|---|---|---|
| Qwen3.5 | Qwen3.5-4B, Qwen3.5-9B | Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-27B | `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` (Tier 1); `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-27B` (Tier 2). Per Codex round-2 §7, the user's original 0.8B / 2B labels do exist as Qwen3.5 release names; v1's "correction" to 0.5B / 1.5B was wrong and is reverted here. |
| Llama 3.1 | Llama-3.1-8B-Instruct | — | `meta-llama/Llama-3.1-8B-Instruct`. Llama-70B out of scope. |
| Ministral 3 | Ministral-3-8B-Instruct-2512 | Ministral-3-3B-Instruct-2512, Ministral-3-14B-Instruct-2512 | `mistralai/Ministral-3-8B-Instruct-2512` (Tier 1); Tier 2 = `mistralai/Ministral-3-3B-Instruct-2512` and `mistralai/Ministral-3-14B-Instruct-2512`. Per Codex round-2 §7, "Mistral 3" is the Ministral 3 family at 3B/8B/14B (not the older `Ministral-8B-Instruct-2410` from October 2024). |

**Tier 1 final lineup (4 checkpoints).** `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Ministral-3-8B-Instruct-2512`. Three families × ~8B parameter scale dominates. The smaller / larger checkpoints in Tier 2 isolate the *model-scale* axis from the *algorithmic-comparison* axis.

**Defensive rebuttal language for the "why these four" reviewer ask** (per Codex round-1 §6): *"We use four checkpoints as a controlled main-paper matrix: two Qwen3.5 sizes plus one Llama and one Mistral-class model around the 8B scale; the remaining size sweep is appendix-only to avoid conflating model scale with the algorithmic comparison."*

**Pre-launch code change** (mandatory before any Tier-1 / Tier-2 launches; see also §7 checklist):

1. `llmrankers/setwise.py:27-28` — extend `CAUSAL_MODEL_TYPES` to include `mistral` (Llama is already in the existing set).
2. `llmrankers/setwise_extended.py:22` — extend `MAXCONTEXT_ALLOWED_MODEL_TYPES` to include `llama` and `mistral`.
3. `MaxContext{DualEnd,TopDown,BottomUp}SetwiseLlmRanker._early_reject_non_qwen3` (lines around `setwise_extended.py:1209` per Codex) — generalise the regex / allowlist; mirror to all three variants.
4. **Per-family chat-template smoke** — Qwen uses `<|im_start|>` template; Llama uses `<|begin_of_text|>` / `<|eot_id|>`; Mistral uses `[INST]` / `[/INST]`. Each family must emit a clean numeric `Best: <int>, Worst: <int>` (or single-int) label without leaking template tokens into the response. The sanity smoke in Phase 1 covers this.
5. **Per-family refusal-pattern coverage** — the existing `NUMERIC_REFUSAL_REGEX` is Qwen-trained; Llama and Mistral may emit different out-of-range / refusal strings. Run a 5-query smoke at $N{=}50$ per family before any production submission and add new fixtures to `scripts/check_maxcontext_invariants.py` if new patterns appear.
6. **Plumb `--seed` through `run.py`** (currently only `random.seed(929)` at `run.py:26` is hard-coded; no `--seed` argument exists per Codex round-1 §8). Required for the 10× stability protocol in Phase 3.
7. **Add per-query wall-clock and total-token logging** to `run.py` (Codex round-1 §14 notes only `Avg comparisons`, `Avg prompt tokens`, `Avg completion tokens`, and per-query *time* are currently logged; we want `Avg total tokens` + total wall-clock seconds for the cost-axis appendix table).
8. **Fix submit/eval path mismatch** — `submit_max_context_jobs.sh:161,173` writes `original/ws-3/...`; `eval_max_context_jobs.sh:129,133` expects `original/ws-4/...`. Standardise on one (recommend `ws-3` to match the `num_child=2 -> window-of-3` convention).

## 2. Dataset matrix (Tier 1 vs Tier 2)

| Dataset | Tier 1 | Tier 2 | Approx query count | Notes |
|---|---|---|---|---|
| TREC DL19 | ✓ | — | 43 | Primary in-domain |
| TREC DL20 | ✓ | — | 54 | Primary in-domain |
| BEIR-dbpedia-entity | ✓ | — | 400 | Knowledge-base entity retrieval |
| BEIR-nfcorpus | ✓ | — | 323 | Bio-medical, smallest corpus |
| BEIR-scifact | ✓ | — | 300 | Scientific fact verification |
| BEIR-trec-covid | ✓ | — | 50 | High-stakes domain shift, small query set |
| BEIR-touche2020 | — | ✓ | 49 | Argument retrieval (noisy) |
| BEIR-fiqa | — | ✓ | 648 | Financial QA (largest query set) |

**Tier 1 = DL19 + DL20 + 4 BEIR domains** = 6 datasets.

**Existing on-disk runs to reuse without re-launching:** none usable for v2. `flan-t5-xl` BEIR runs are out of scope (no joint elicitation in T5). `qwen3-8b` 5/6 BEIR runs are not directly reusable because the v2 lineup uses **Qwen3.5-9B** (not Qwen3-8B); fresh runs are required.

## 3. Phase-by-phase staging

Each phase must pass before the next launches.

### Phase 0 — Code prerequisites (zero GPU)

Per §1 items 1–8. Verify with:
- `pytest scripts/check_maxcontext_invariants.py` for the existing Qwen invariants.
- New family smokes per `scripts/check_maxcontext_invariants.py` extension.
- Deterministic local count-only smoke for the new families (per the bubblesort-clamp fix pattern).
- `submit_max_context_jobs.sh --dry-run` matches `eval_max_context_jobs.sh --dry-run` expected paths byte-for-byte.

### Phase 1 — Per-family parse-stability smoke (~12 GPU-hours)

Single-config smoke at $N{=}50$, DL19, $pl{=}512$ on:
- `Qwen/Qwen3.5-4B` (re-confirm baseline)
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Ministral-3-8B-Instruct-2512`

per MaxContext-DualEnd, MaxContext-TopDown, MaxContext-BottomUp = 9 jobs.

**Pass criterion:** every job completes; `parse-fallback rate` $\le 5\%$ of LLM calls; no truncation aborts; output is a permutation of the input pool. **Fail handling:** add the missing refusal pattern to the regex, re-run the failing job; if still failing, drop that family from Tier 1 → Branch C in `EMNLP_SHORT_PLAN.md` §4.

### Phase 2 — Tier-1 production matrix

| Variable | Values | Count |
|---|---|---|
| Models | Qwen3.5-4B, Qwen3.5-9B, Llama-3.1-8B-Instruct, Ministral-3-8B-Instruct-2512 | 4 |
| Methods | MaxContext-DualEnd, MaxContext-TopDown, MaxContext-BottomUp | 3 |
| Pool sizes ($N{=}k{=}\text{hits}$) | 10, 20, 30, 40, 50 | 5 |
| Datasets | DL19, DL20, BEIR-dbpedia, BEIR-nfcorpus, BEIR-scifact, BEIR-trec-covid | 6 |
| Total runs | $4 \times 3 \times 5 \times 6$ | **360** |

**Compute estimate (revised per Codex round-1 §9 with realistic per-query and per-dataset numbers):**

The existing project log shows MaxContext-DualEnd-style on Qwen3.5-4B / DL19 takes about 191 s/query. Wall-clock per (model, method, $N$, dataset) is roughly:

$$\text{wall-clock} \approx \tau_{\text{model}}(N) \times \text{queries}_{\text{dataset}}$$

Treat $\tau_{\text{Qwen3.5-4B}}(N{=}50) \approx 191$ s for DualEnd, ~$1.9 \times$ that for TopDown / BottomUp ($N{-}2$ vs $\lfloor N/2 \rfloor$ calls), and $\tau$ scales sub-linearly with $N$.

| Dataset | Queries | Per-job wall-clock at $N{=}50$ DualEnd (Qwen3.5-4B) | Per-job at $N{=}50$ Single-extreme | Approx Phase-2 wall-clock for that dataset (4 models × 5 $N$ × 3 methods, summed, all extrapolated linearly in queries) |
|---|---|---|---|---|
| DL19 | 43 | $\sim 137$ min | $\sim 260$ min | $\sim 80$ H100-h |
| DL20 | 54 | $\sim 172$ min | $\sim 327$ min | $\sim 100$ H100-h |
| BEIR-trec-covid | 50 | $\sim 159$ min | $\sim 302$ min | $\sim 92$ H100-h |
| BEIR-scifact | 300 | $\sim 16$ h | $\sim 30$ h | $\sim 555$ H100-h |
| BEIR-nfcorpus | 323 | $\sim 17$ h | $\sim 32$ h | $\sim 596$ H100-h |
| BEIR-dbpedia | 400 | $\sim 21$ h | $\sim 40$ h | $\sim 738$ H100-h |
| **Subtotal Phase-2** | | | | **$\sim 2160$ H100-h** |

**This is significantly larger than v0's $\sim 720$ H100-h estimate.** The big consumers are the four BEIR datasets with 300–400 queries each. **Codex round-1 §9 was correct: Phase-2 production is closer to $2000+$ H100-hours, not $720$.**

Cost-reduction options to consider before launch:
- **Drop BEIR-dbpedia from Tier 1** (largest single contributor; saves $\sim 740$ H100-h). Trade: cuts Tier-1 BEIR coverage from 4 to 3 domains.
- **Cut Qwen3.5-9B for the BEIR domains only** (keep DL19/20 only on the bigger model). Saves $\sim 540$ H100-h. Trade: BEIR generalisation is then established only on Qwen3.5-4B / Llama-3.1-8B / Ministral-8B (still three families).
- **Run only $N \in \{10, 30, 50\}$ on BEIR** (drop $N{=}20, 40$ for BEIR domains). Saves $\sim 800$ H100-h. Trade: the Pareto curve has fewer points on the BEIR panel.

**Recommended Tier-1 trim:** keep all 4 models on DL19/20 at all 5 $N$ (small datasets are cheap), and use $N \in \{10, 30, 50\}$ on the 4 BEIR domains for all 4 models. This compromise:
- Tier 1 production runs: $4 \times 3 \times 5 \times 2 + 4 \times 3 \times 3 \times 4 = 120 + 144 = 264$ runs.
- Approximate wall-clock: $\sim 80 + 100 + 0.6 \times (92 + 555 + 596 + 738) = \sim 1370$ H100-h.

### Phase 3 — Stability re-runs (10× on Qwen3-4B / DL19)

Per the user's request: 10 independent re-runs of `Need_to_Run_Max_Context.txt` on Qwen3-4B / DL19. Per Codex round-1 §8, MaxContext generation is greedy (`do_sample=False` at `setwise.py:409`), so the 10× variability source is **not** in the LLM decoding step itself — it must come from somewhere else.

**Source-of-variability decision:** use **input-order seed plumbing**. After Phase 0 item 6 (plumb `--seed` through `run.py`), each job runs with a different `--seed` value $\in \{0, 1, \ldots, 9\}$, where `--seed` controls (a) the per-query input-order shuffling under `--shuffle_ranking random`, (b) Python `random.seed()`, (c) NumPy / Torch RNG seeds. **The ranking input order across the 10 runs is therefore different**, isolating *input-order stability* of MaxContext at the Qwen3-4B / DL19 / $N \in \{10..50\}$ scale.

Note the v0 plan's "kernel-determinism" hypothesis is unsupported by current evidence and is dropped.

**Job-scope clarification (Codex round-3 §6).** `Need_to_Run_Max_Context.txt` contains 35 jobs across 7 method blocks: 4 standard-Setwise blocks (TD-Heap-WS=4, TD-Bubble-WS=4, TD-Heap-WS=PS, TD-Bubble-WS=PS) and 3 MaxContext blocks (DualEnd, TopDown, BottomUp). The H4 input-order-stability claim is **scoped to the 15 MaxContext jobs** (3 methods × 5 pool sizes), not all 35. The 20 standard-Setwise jobs are out of scope for the v2 paper anyway and stay out of the H4 claim. The user's "10× the file" request is honoured by re-launching all 35, but only the 15 MaxContext entries feed claim H4.

**Output-directory seeding** (Codex round-3 §8). The 10 runs must not overwrite each other. The recommended layout: `results/maxcontext_dualend/<TAG>/seed-<S>/...` where `<S> ∈ {0..9}`. The submit script needs a `--seed` flag that both seeds the run and routes the output under the seed-specific subdirectory.

**Pass criterion (claim H4):** for each MaxContext (method, pool-size) cell, the nDCG@10 across 10 runs satisfies all three of: SD $\le 0.005$, max−min range $\le 0.015$, and worst-pair $|\Delta| \le 0.015$. **Fail handling:** investigate, then if a threshold is persistently exceeded, frame H4 as a measured input-order-sensitivity number; report SD + range bars on Table 1 / Figure 1 (Branch G in `EMNLP_SHORT_PLAN.md` §4).

**Compute:** $35 \times 10 = 350$ jobs (all 35 re-launched even though only the 15 MaxContext entries feed H4 — keeps `Need_to_Run_Max_Context.txt` as the single source of truth for the launcher). Qwen3-4B / DL19 single-DualEnd-job wall-clock at $N{=}50$ is observed at $\sim 137$ minutes; lighter for smaller $N$ and for the standard-Setwise jobs that don't touch MaxContext. Mean per-job estimate $\sim 60$ minutes, so $\sim 350$ H100-h.

### Phase 4 — Tier 2 (appendix only)

Triggered only if Phases 1–3 are green and budget allows.

- Smaller / larger Qwen3.5: $3 \text{ models} \times 3 \text{ methods} \times 5 \text{ pools} \times 6 \text{ datasets}$, with same trim as Phase 2 = $\sim 200$ jobs at $\sim 720$ H100-h.
- Ministral-3B / Ministral-14B: only if HF identifiers resolve. $2 \text{ models}$ at the same trim = $\sim 130$ jobs at $\sim 480$ H100-h.
- Two extra BEIR domains (touche2020, fiqa): $4 \text{ Tier-1 models} \times 3 \text{ methods} \times 5 \text{ pools} \times 2 \text{ datasets} = 120$ jobs at $\sim 360$ H100-h (fiqa is the largest contributor with 648 queries).

**Phase-4 total upper bound:** $\sim 1560$ H100-hours.

### Phase 5 — Defensive smokes (appendix)

To pre-empt anticipated reviewer asks (per `EMNLP_SHORT_PLAN.md` §9 risk register):

- **Order-robustness pilot at $N{=}50$.** Qwen3.5-4B + Llama-3.1-8B × DL19 × {forward, inverse, random shuffle (seed 0)} × DualEnd = 6 jobs. $\sim 12$ H100-h.
- **TourRank smoke.** Single matched-budget point on Qwen3.5-4B / DL19 with TourRank-10 (or whatever cost matches 25 LLM calls at $N{=}50$). $\sim 5$ H100-h.
- **Original-Setwise direct-baseline smoke** (per Codex round-1 §10). Qwen3.5-4B / DL19 × $c{+}1{=}10$, $k{=}10$ Setwise-Selection (letter alphabet), exactly the configuration that *might* be claimed equivalent to MaxContext-TopDown at $N{=}10$. Single job comparing call count + nDCG@10. $\sim 2$ H100-h.
- **Passage-length sweep.** Qwen3.5-4B × DL19 × {pl 64, 128, 256, 512} × DualEnd = 4 jobs. $\sim 8$ H100-h.

**Phase 5 total:** $\sim 27$ H100-h.

### Total compute envelope (revised)

| Phase | H100-hours (est.) | Cumulative |
|---|---|---|
| Phase 0 | 0 | 0 |
| Phase 1 (family-fragility smoke) | 12 | 12 |
| Phase 2 (Tier 1 trimmed) | 1370 | 1382 |
| Phase 3 (10× stability) | 350 | 1732 |
| Phase 5 (defensive smokes) | 27 | 1759 |
| Phase 4 (Tier 2, optional) | 1560 | 3319 |

**Tier 1 + stability + defensive smokes** = **$\sim 1760$ H100-hours**, roughly 10 GPU-weeks at one H100. This is the minimum to ship a 4-page paper with the headline supported and the obvious reviewer asks pre-empted. **The v0 plan's $\sim 1100$ H100-hour estimate was optimistic; the BEIR query counts dominate the budget.**

## 4. Statistical protocol (revised)

Per Codex round-1 §4 / §5 and round-2 §4.

- **Per-cell H1 test = one-sided paired permutation test.** Define $\Delta = \text{nDCG@10}(\text{DualEnd}) - \text{nDCG@10}(\text{baseline})$ over the per-query series, and margin $\delta = 0.01$. Test
  $$H_0: \Delta \le -\delta \quad \text{vs.} \quad H_1: \Delta > -\delta$$
  Reject $H_0$ at $\alpha{=}0.05$ → cell verdict ✓ "non-inferior". The textbook framing: failing to reject is *not* evidence of inferiority; it's an inconclusive cell, annotated ▲ in Table 1. Implementation: shift the per-query $\Delta$s by $+\delta$, then run a one-sided paired sign-flip permutation test of the null "shifted-mean $\le 0$"; $10{,}000$ resamples per cell. (The v1 wording inverted this hypothesis — corrected here.)
- **Reporting confidence intervals.** Paired bootstrap percentile CIs (95%) on $\Delta$, $10{,}000$ resamples, reported alongside every per-cell point estimate. CIs are *descriptive*, not the primary decision rule. Cells with bootstrap-CI lower bound $\ge -\delta$ but where the permutation test does not reject are annotated ▲. Cells with bootstrap-CI upper bound $\le -\delta$ are annotated ✗.
- **Multiple-testing correction.** **BH-FDR** at $q{=}0.05$, applied **separately within each comparator family** — i.e. one FDR run for the 24 H1$_\text{TD}$ cells, a second FDR run for the 24 H1$_\text{BU}$ cells. Bonferroni and Holm-Bonferroni adjusted $p$-values are also published in the appendix transparency table.
- **Aggregate H3 verdict.** Number of cells with FDR-adjusted-significant non-inferiority verdict $\ge 20$ out of 24, separately for the TD and BU comparator families → claim H3 supported. The "supported" verdict requires the threshold to be met for *both* families; otherwise the verdict reports the family-specific count.
- **Appendix transparency.** Every per-cell point estimate, paired-bootstrap CI, raw $p$-value, BH-FDR-adjusted $p$-value, Holm-Bonferroni-adjusted $p$-value, Bonferroni-adjusted $p$-value, and ✓ / ▲ / ✗ verdict published in an appendix master table.
- **Tier 2 statistics** are reported descriptively only (point estimate + paired bootstrap CI), without a non-inferiority verdict, because Tier-2 cells are appendix-only.

## 5. Cost-axis reporting (per E2R-FLOPs spirit)

Every job logs to its result directory's `*.log` file:

- `Avg comparisons` (mean per-query LLM calls)
- `Avg prompt tokens`, `Avg completion tokens`, `Avg total tokens` (the third requires Phase-0 item 7 plumbing)
- `Avg parse fallbacks` (per-query rate, denominator = `Avg comparisons`)
- `Avg BM25 bypass` (single-extreme variants only)
- `Total wall-clock seconds` and `Avg time per query` (the first requires Phase-0 item 7 plumbing)

All five axes are reported in the appendix master table; main-paper Pareto figure uses **only** `Avg comparisons` (LLM calls, x-axis) vs `nDCG@10` (y-axis) per the half-cost claim. Token axes are flagged in §3 of the short paper as "transparently reported, not the basis for our efficiency claim" (token-axis trap mitigation).

## 6. Reproducibility

- All code released under repo's existing licence; new launchers added under `experiments/`.
- Per-run JSONL log includes: prompt, response, parse status, label scheme, seed, SLURM job ID, model checkpoint hash, per-call call stats.
- Seeds for Phase 3 stability runs ($0, \ldots, 9$) published explicitly in the appendix.
- Hardware profile (H100 80 GB Vast.ai, default kernel determinism settings + `torch.use_deterministic_algorithms(True)` if Phase 3 reveals kernel-level non-determinism) documented in the short paper §3.

## 7. Pre-launch checklist (must all hold before any Phase 2 sbatch)

- [ ] §1 code changes 1–8 merged, tested, smoke passes for all three families.
- [ ] `submit_max_context_jobs.sh` and `eval_max_context_jobs.sh` agree on the `original/ws-X/...` directory naming (the Codex-round-1 §14 `ws-3`/`ws-4` mismatch is fixed).
- [ ] `--seed` argument plumbed through `run.py` and forwarded by `submit_max_context_jobs.sh`; both `--seed` and `--shuffle_ranking random` are propagated for Phase 3 launches.
- [ ] Per-query wall-clock and total-token logging added to `run.py` per §1 item 7.
- [ ] Phase 1 per-family parse-stability smoke passes at $N{=}50$ with $\le 5\%$ fallback rate.
- [ ] Ministral 3 exact HF identifier confirmed (`mistralai/Ministral-3-{3B,8B,14B}-Instruct-2512`) and pinned to a specific revision hash.
- [ ] Qwen3.5 Tier-2 identifiers (`Qwen/Qwen3.5-{0.8B, 2B, 27B}`) confirmed and pinned to revision hashes.
- [ ] `AutoConfig.from_pretrained(...).model_type` verified for each new family before allowlisting; do not assume Ministral resolves to exactly `mistral` (it may be `mistral3` or another string).
- [ ] License / access verification: Llama 3.1 requires HF gated-model access (request through HF before launch); Ministral 3 access status confirmed (the model card metadata governs this — verify rather than assume). `HF_TOKEN` set in cluster env in either case.
- [ ] All hard-coded RNG seeds overridden by the `--seed` argument: `run.py:27` (`random.seed(929)`), `setwise.py:25` (any module-level seed), `setwise_extended.py:20` (any module-level seed). Dump the effective seed set into the run manifest.
- [ ] Phase 3 output directories layout includes `seed-<S>` subdirectories so 10× reruns do not overwrite (e.g. `results/maxcontext_dualend/<TAG>/seed-3/baseline/qwen3-4b-dl19/...`).
- [ ] One **command-level BEIR eval / qrels smoke** (single dataset, single tiny pool) before Phase 2 to confirm `pyserini.eval.trec_eval` resolves the `beir-v1.0.0-...` qrels labels and that the BEIR-v1.0.0 dataset path is reachable.
- [ ] DL19/20 qrels and BEIR qrels paths verified in `pyserini` (`beir-v1.0.0-dbpedia-entity`, `beir-v1.0.0-nfcorpus`, `beir-v1.0.0-scifact`, `beir-v1.0.0-trec-covid`).
- [ ] `submit_max_context_jobs.sh` (or a v2-equivalent) accepts the new dataset shortcuts (BEIR-dbpedia, BEIR-nfcorpus, etc.) and the new model identifiers.
- [ ] `eval_max_context_jobs.sh` is updated to accept the new BEIR qrels labels.
- [ ] **Significance script updated** to implement the margin-shifted one-sided paired permutation test (per §4) and BH-FDR / Holm-Bonferroni / Bonferroni reporting columns.
- [ ] **Run manifest** records exact HF revision hashes for every job (model + tokenizer); manifest committed to repo before launch.
- [ ] Compute budget for Phase 1 + Phase 2 + Phase 3 + Phase 5 reserved (~$1760$ H100-h).

## 8. Out-of-scope

Same as `EMNLP_SHORT_PLAN.md` §10. In particular: no T5, no pre-007 baselines as primary comparators, no listwise / TourRank as primary baselines, no permutation self-consistency, no pools $> 50$.

## 9. Audit trail

- v0 (2026-05-02 morning) — initial draft.
- v1 (2026-05-02 afternoon) — addresses Codex round-1 NEEDS_MAJOR_REVISION:
  - §1: model identifiers pinned (`Qwen/Qwen3.5-{4B,9B}`, `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Ministral-8B-Instruct-2410`); Qwen3.5 small-checkpoint naming corrected (0.5B / 1.5B not 0.8B / 2B); `_early_reject_non_qwen3` mirroring requirement called out at `setwise_extended.py:1209`.
  - §1 / §3 Phase 0: pre-launch code prerequisites expanded — `--seed` plumbing through `run.py` (item 6), per-query wall-clock + total-token logging (item 7), submit/eval `ws-3`/`ws-4` path mismatch fix (item 8).
  - §3 Phase 2: compute estimate revised from $\sim 720$ to $\sim 1370$ H100-h after correcting per-BEIR-dataset query counts (dbpedia 400, nfcorpus 323, scifact 300, fiqa 648) and using the observed Qwen3.5-4B / DL19 wall-clock baseline (191 s / query). Trim: $N \in \{10,30,50\}$ on BEIR domains, full $N \in \{10..50\}$ on DL19/20.
  - §3 Phase 3: stability protocol switched to **input-order seed plumbing** as the variability source (Codex round-1 §8); kernel-determinism hypothesis dropped.
  - §4: H1 per-cell test changed from "$p > 0.05$ failure-to-reject" to "one-sided paired permutation test of $H_0: \Delta \ge -\delta$, reject $H_0$ → non-inferior".
  - §4: multiplicity correction switched from Bonferroni to BH-FDR at $q{=}0.05$.
  - §4: paired-bootstrap CI ($10{,}000$ resamples) added as the primary CI; AR / permutation samples reduced from $100{,}000$ to $10{,}000$.
  - §5: cost-axis logging dependencies aligned with current `run.py` capability + Phase-0 §7 plumbing.
  - §7 pre-launch checklist: ws-3/ws-4 mismatch, `--seed` plumbing, BEIR identifier resolution all listed.
- v2 (2026-05-02 evening) — addresses Codex round-2 NEEDS_MAJOR_REVISION:
  - §1: Ministral 3 identifiers updated to `mistralai/Ministral-3-{3B,8B,14B}-Instruct-2512` (round-1 v1 used the older October-2024 `Ministral-8B-Instruct-2410`, which is *not* the Ministral 3 family).
  - §1: Qwen3.5 Tier-2 identifiers reverted to `Qwen/Qwen3.5-{0.8B, 2B, 27B}` (round-1 v1's "correction" to 0.5B / 1.5B was wrong; the user's original 0.8B / 2B labels do exist).
  - §4: H1 hypothesis direction **inverted to the textbook non-inferiority form** $H_0: \Delta \le -\delta$ vs $H_1: \Delta > -\delta$. v1 had it backwards (rejecting v1's $H_0: \Delta \ge -\delta$ would have supported *inferiority*, not non-inferiority).
  - §4: multiplicity correction explicitly applied **per comparator family** (separate FDR runs for H1$_\text{TD}$ and H1$_\text{BU}$). Holm-Bonferroni and Bonferroni adjusted $p$-values added as appendix transparency columns alongside BH-FDR.
  - §7: pre-launch checklist gains 5 items — `AutoConfig.model_type` verification, license / gated-access checks for Llama and Ministral 3, significance-script update, run-manifest with HF revision hashes, `--shuffle_ranking random` propagation alongside `--seed`.

---

*Companion:* [`EMNLP_SHORT_PLAN.md`](EMNLP_SHORT_PLAN.md) — section plan, claims, novelty positioning, branch resilience, page budget.
