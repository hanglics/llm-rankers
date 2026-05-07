# EMNLP_SHORT_EXPERIMENT_PLAN.md — experiments for the v2 short paper

> **Status:** v4 — aligned to canonical root docs at v8 + multimodal v3 + Phase F v5 (2026-05-07). Companion to [`EMNLP_SHORT_PLAN.md`](EMNLP_SHORT_PLAN.md).
> **Goal:** support claims H1–H4 from the short-paper plan with the smallest matrix that survives reviewer pushback in 4 pages.
> **Hardware target:** NVIDIA H100 80 GB on the project's HPC cluster, single-GPU per job. SLURM dispatchers (`submit_emnlp_jobs.sh`, `submit_emnlp_stability_jobs.sh`, `submit_max_context_jobs.sh`) resolve the conda env per model family (`ranker_env` for Qwen3 / pyserini, `qwen35_env` for Qwen3.5 / Llama-3.1 / Ministral-3) and propagate it via `sbatch --export=ALL,CONDA_ENV=...`.
> **Relationship to v8 + multimodal v3 + Phase F v5:** the canonical root docs are `../EMNLP_EXPERIMENT_PLAN.md` + `../EMNLP_IMPLEMENTATION_PLAN.md` + `../EMNLP_PAPER_DESIGN.md`. The canonical required matrix (9 models × 8 datasets × 7 methods × 6 pool sizes = 3024 main + 1260 stability + 35 prime-recheck + 432 Phase F position-bias = 4793 required jobs across Phases A/B/C/C′/F; Phase D + Phase E are optional Qwen3 add-ons) is the production launcher. The paper's Tier-1 selects **4 of v8's 9 required models** and **3 of v8's 7 methods (the MaxContext family)** for the main-paper headline; the rest are appendix-only.

---

## 1. Model lineup (Tier 1 = main paper, Tier 2 = appendix only)

The user's nine-checkpoint lineup is generous for a short paper. Splitting into a **Tier 1** subset for the main-paper headline and a **Tier 2** for the appendix keeps the 4-page narrative defensible.

| Family                         | Tier 1 (main paper)          | Tier 2 (appendix)                                           | Pinned HF identifier (revision pinned at launch)                                                                                                                                                                                                                                             |
|--------------------------------|------------------------------|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Qwen3.5                        | Qwen3.5-4B, Qwen3.5-9B       | Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-27B                       | `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` (Tier 1); `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-27B` (Tier 2). Note: v8 dropped the trailing `-Instruct` suffix (corrected against the actual HF release names; v1/v2 paper plan had `-Instruct`, fixed in v3).                         |
| Llama 3.1                      | Meta-Llama-3.1-8B-Instruct   | —                                                           | `meta-llama/Meta-Llama-3.1-8B-Instruct` (v8 canonical; v1/v2 used the legacy `meta-llama/Llama-3.1-8B-Instruct` alias, fixed in v3). Llama-70B out of scope.                                                                                                                                 |
| Ministral 3                    | Ministral-3-8B-Instruct-2512 | Ministral-3-3B-Instruct-2512, Ministral-3-14B-Instruct-2512 | `mistralai/Ministral-3-8B-Instruct-2512` (Tier 1); Tier 2 = `mistralai/Ministral-3-3B-Instruct-2512` and `mistralai/Ministral-3-14B-Instruct-2512`. Per Codex round-2 §7, "Mistral 3" is the Ministral 3 family at 3B/8B/14B (not the older `Ministral-8B-Instruct-2410` from October 2024). |
| Qwen3 (optional, v8 Phase D/E) | —                            | Qwen3-{0.6B,1.7B,4B,8B,14B,32B}                             | Used by v8 Phase D (optional Qwen3 main matrix, 1680 jobs) and Phase E (optional Qwen3-8B stability, 350 jobs). Not main paper; appendix-only.                                                                                                                                               |

**Tier 1 final lineup (4 checkpoints).** `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, `meta-llama/Meta-Llama-3.1-8B-Instruct`, `mistralai/Ministral-3-8B-Instruct-2512`. Three families × ~8B parameter scale dominates. The smaller / larger checkpoints in Tier 2 isolate the *model-scale* axis from the *algorithmic-comparison* axis. v8's required matrix is broader (9 models = 5 Qwen3.5 + 1 Llama + 3 Ministral) — Tier 1 is the **paper-headline subset** of v8's required matrix; v8 produces the full 9-model dataset, the paper selects 4 for the main-paper Table 1.

**Defensive rebuttal language for the "why these four" reviewer ask** (per Codex round-1 §6): *"We use four checkpoints as a controlled main-paper matrix: two Qwen3.5 sizes plus one Llama and one Mistral-class model around the 8B scale; the remaining size sweep is appendix-only to avoid conflating model scale with the algorithmic comparison."*

**Pre-launch code change** (mostly DONE per the v8 implementation; see also §7 checklist):

1. ✅ **DONE in v8+multimodal Phase 3a**: `llmrankers/setwise.py:27-40` — text-only causal configs remain in `CAUSAL_MODEL_TYPES`, while Mistral 3 and Qwen3.5 vision-language configs (`mistral3`, `qwen3_5`, `qwen3_5_moe`) use `MULTIMODAL_MODEL_TYPES` with `AutoProcessor` + `AutoModelForImageTextToText`. The IR task passes text only; vision inputs are unused.
2. ✅ **DONE in v8+multimodal Phase 3a**: `llmrankers/setwise_extended.py:22-26` — `MAXCONTEXT_ALLOWED_MODEL_TYPES` widened to `{qwen3, qwen3_moe, qwen3_5, qwen3_5_moe, llama, mistral, mistral3, ministral}`.
3. ✅ **DONE in v8**: `MaxContext{DualEnd,TopDown,BottomUp}SetwiseLlmRanker._early_reject_non_qwen3` renamed to `_early_reject_unsupported_family` (`setwise_extended.py:1219`); class-level `_MAXCONTEXT_NAME_FRAGMENTS` includes Qwen3, Qwen3.5, Llama-3.1, Ministral-3 fragments.
4. **Per-family chat-template smoke** — Qwen uses `<|im_start|>` template; Llama uses `<|begin_of_text|>` / `<|eot_id|>`; Mistral uses `[INST]` / `[/INST]`. Each family must emit a clean numeric `Best: <int>, Worst: <int>` (or single-int) label without leaking template tokens into the response. Runtime smoke covered by `scripts/smoke_emnlp_models.sh` (v8 Phase A: 42 cells = 3 representative models × 7 methods × dl19 × pools {50,100}).
5. **Per-family refusal-pattern coverage** — the existing `NUMERIC_REFUSAL_REGEX` is Qwen-trained; Llama and Mistral may emit different out-of-range / refusal strings. Phase A smoke gates on `Avg parse fallbacks: 0` and `Avg numeric out-of-range fallbacks: 0` for the MaxContext methods on the 3 required families. Any new pattern observed during smoke must be added to `scripts/check_maxcontext_invariants.py` fixtures.
6. ✅ **DONE in Phase F (2026-05-07)**: per-comparison `--shuffle` (fixed seed 929) and `--reverse` flags plumbed for MaxContext methods only; Heap/Bubble methods remain BM25-ordered (canonical `../EMNLP_PAPER_DESIGN.md` v5; `../EMNLP_IMPLEMENTATION_PLAN.md` v4 §7). Phase F now sources input-order H4 evidence via paired forward-vs-reverse / forward-vs-shuffle deltas, superseding the legacy `--seed`-shuffle protocol from paper-plan v1/v2. v8 Phase C continues to run 10 deterministic byte-equality reps as the implementation guard; the deterministic and shuffle-based stability views are now reported separately.
7. **Add per-query wall-clock and total-token logging** to `run.py` — TODO (per Codex round-1 §14; needed for the cost-axis appendix table). v8 implementation has not yet plumbed `Avg total tokens`.
8. ✅ **DONE in v8**: submit/eval path is consistent (`submit_max_context_jobs.sh` and `eval_max_context_jobs.sh` both use `original/ws-3/...` and `original/ws-ps/...`; the legacy `ws-4` mismatch is gone).

**v8 implementation adds** (post paper plan v1/v2):
- `submit_emnlp_jobs.sh`, `eval_emnlp_jobs.sh`, `submit_emnlp_stability_jobs.sh`, `scripts/smoke_emnlp_models.sh`, `scripts/probe_beir_pool100_fit.py`, `analysis/cross_model_stability.py`, `analysis/position_bias_emnlp.py`.
- `submit_max_context_jobs.sh` and `eval_max_context_jobs.sh` gain `--pool-sizes` override (default 5 pools / IDEA_007 byte-equality preserved; pass `"10 20 30 40 50 100"` to add pool=100).
- `--include-standard-bottomup` opt-in flag adds standard BottomUp heap+bubble blocks (default-off path is byte-identical to IDEA_007's 35-cell stability layout).
- `experiments/run_*.sh` launchers gain `CONDA_ENV` env-var support (default ranker_env preserved).

## 2. Dataset matrix (Tier 1 vs Tier 2)

| Dataset             | Tier 1 | Tier 2 | Approx query count | Notes                                     |
|---------------------|--------|--------|--------------------|-------------------------------------------|
| TREC DL19           | ✓      | —      | 43                 | Primary in-domain                         |
| TREC DL20           | ✓      | —      | 54                 | Primary in-domain                         |
| BEIR-dbpedia-entity | ✓      | —      | 400                | Knowledge-base entity retrieval           |
| BEIR-nfcorpus       | ✓      | —      | 323                | Bio-medical, smallest corpus              |
| BEIR-scifact        | ✓      | —      | 300                | Scientific fact verification              |
| BEIR-trec-covid     | ✓      | —      | 50                 | High-stakes domain shift, small query set |
| BEIR-touche2020     | —      | ✓      | 49                 | Argument retrieval (noisy)                |
| BEIR-fiqa           | —      | ✓      | 648                | Financial QA (largest query set)          |

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

### Phase 1 — Per-family parse-stability smoke (subsumed by v8 Phase A)

**v8 alignment:** v8's Phase A smoke (`scripts/smoke_emnlp_models.sh`) subsumes paper-plan Phase 1 with broader scope: 3 representative models (Qwen3.5-9B, Meta-Llama-3.1-8B-Instruct, Ministral-3-8B-Instruct-2512) × 7 methods (4 standard + 3 MaxContext) × dl19 × pools {50, 100} × 1 rep = 42 cells. Use v8 Phase A as the per-family parse-stability gate; paper-plan Phase 1's narrow 9-job version is no longer a separate phase.

```bash
bash scripts/smoke_emnlp_models.sh --dry-run    # verify 42-cell expansion
bash scripts/smoke_emnlp_models.sh              # submit
bash scripts/smoke_emnlp_models.sh --eval-only  # after completion
bash scripts/smoke_emnlp_models.sh --verify-only
```

**Pass criterion:** every cell completes; for MaxContext methods, `Avg parse fallbacks: 0` and `Avg numeric out-of-range fallbacks: 0` in the per-cell `.log`; for all methods, full `.txt` coverage (`n_queries × pool_size` lines), valid top-10 permutation, positive nDCG@10, no `Traceback` / `ERROR` / `exceeds model limit`. **Fail handling:** if a family fails on its pool=100 cells but passes pool=50, drop pool=100 for that family (Branch H). If a family fails parse-stability across both pool sizes, drop the family → Branch C.

**BEIR pool=100 fit probe:** before any Phase B BEIR pool=100 submission, run `python3 scripts/probe_beir_pool100_fit.py`. The probe is tokenizer-only (AutoConfig + AutoTokenizer; no model weights), samples real BEIR corpus passages via `ir_datasets`, and reports per-(family, dataset) `headroom = max_position_embeddings - 4096 - prompt_tokens`. Cells with negative headroom must drop pool=100 before Phase B launches.

### Phase 2 — Tier-1 production matrix (paper-headline subset of v8 Phase B)

| Variable                           | Values                                                                           | Count                                                                       |
|------------------------------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Models                             | Qwen3.5-4B, Qwen3.5-9B, Meta-Llama-3.1-8B-Instruct, Ministral-3-8B-Instruct-2512 | 4                                                                           |
| Methods                            | MaxContext-DualEnd, MaxContext-TopDown, MaxContext-BottomUp                      | 3                                                                           |
| Pool sizes ($N{=}k{=}\text{hits}$) | 10, 20, 30, 40, 50, 100 (gated on Phase A pool=100 smoke + BEIR fit probe)       | 6                                                                           |
| Datasets                           | DL19, DL20, BEIR-dbpedia, BEIR-nfcorpus, BEIR-scifact, BEIR-trec-covid           | 6                                                                           |
| Total runs                         | $4 \times 3 \times 6 \times 6$                                                   | **432** (paper-headline subset; v8 Phase B is a superset at **3024** total) |

v8 Phase B launches all $7 \times 9 \times 8 \times 6 = 3024$ cells. The paper extracts the 432 cells matching this Tier-1 subset for Table 1 and Figure 1; the remaining 2592 cells (5 v8-required models not in Tier 1, 4 standard methods, 2 BEIR datasets in v8 Tier 2) populate the appendix.

**Compute estimate (revised per Codex round-1 §9 with realistic per-query and per-dataset numbers):**

The existing project log shows MaxContext-DualEnd-style on Qwen3.5-4B / DL19 takes about 191 s/query. Wall-clock per (model, method, $N$, dataset) is roughly:

$$\text{wall-clock} \approx \tau_{\text{model}}(N) \times \text{queries}_{\text{dataset}}$$

Treat $\tau_{\text{Qwen3.5-4B}}(N{=}50) \approx 191$ s for DualEnd, ~$1.9 \times$ that for TopDown / BottomUp ($N{-}2$ vs $\lfloor N/2 \rfloor$ calls), and $\tau$ scales sub-linearly with $N$.

| Dataset              | Queries | Per-job wall-clock at $N{=}50$ DualEnd (Qwen3.5-4B) | Per-job at $N{=}50$ Single-extreme | Approx Phase-2 wall-clock for that dataset (4 models × 5 $N$ × 3 methods, summed, all extrapolated linearly in queries) |
|----------------------|---------|-----------------------------------------------------|------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| DL19                 | 43      | $\sim 137$ min                                      | $\sim 260$ min                     | $\sim 80$ H100-h                                                                                                        |
| DL20                 | 54      | $\sim 172$ min                                      | $\sim 327$ min                     | $\sim 100$ H100-h                                                                                                       |
| BEIR-trec-covid      | 50      | $\sim 159$ min                                      | $\sim 302$ min                     | $\sim 92$ H100-h                                                                                                        |
| BEIR-scifact         | 300     | $\sim 16$ h                                         | $\sim 30$ h                        | $\sim 555$ H100-h                                                                                                       |
| BEIR-nfcorpus        | 323     | $\sim 17$ h                                         | $\sim 32$ h                        | $\sim 596$ H100-h                                                                                                       |
| BEIR-dbpedia         | 400     | $\sim 21$ h                                         | $\sim 40$ h                        | $\sim 738$ H100-h                                                                                                       |
| **Subtotal Phase-2** |         |                                                     |                                    | **$\sim 2160$ H100-h**                                                                                                  |

**This is significantly larger than v0's $\sim 720$ H100-h estimate.** The big consumers are the four BEIR datasets with 300–400 queries each. **Codex round-1 §9 was correct: Phase-2 production is closer to $2000+$ H100-hours, not $720$.**

Cost-reduction options to consider before launch:
- **Drop BEIR-dbpedia from Tier 1** (largest single contributor; saves $\sim 740$ H100-h). Trade: cuts Tier-1 BEIR coverage from 4 to 3 domains.
- **Cut Qwen3.5-9B for the BEIR domains only** (keep DL19/20 only on the bigger model). Saves $\sim 540$ H100-h. Trade: BEIR generalisation is then established only on Qwen3.5-4B / Llama-3.1-8B / Ministral-8B (still three families).
- **Run only $N \in \{10, 30, 50\}$ on BEIR** (drop $N{=}20, 40$ for BEIR domains). Saves $\sim 800$ H100-h. Trade: the Pareto curve has fewer points on the BEIR panel.

**Recommended Tier-1 trim** (with pool=100 gated):
- DL19/20: 4 models × 3 MaxContext methods × 6 pools × 2 datasets = **144 runs** (~180 H100-h).
- BEIR: 4 models × 3 methods × $N \in \{10, 30, 50, 100\}$ × 4 domains = **192 runs** (BEIR pool=100 conditional on per-family fit probe pass; ~1100-1500 H100-h depending on per-family pool=100 wall-clock).
- Total Tier-1 paper-headline subset: **336 runs**, ~1300-1700 H100-h.

**Note:** v8 Phase B runs all 3024 cells regardless of paper trim — the trim above is what the paper extracts for main-paper Table 1 / Figure 1. The full 3024-cell matrix is in appendix.

### Phase 3 — Stability re-runs (v8 Phase C: 3 EMNLP families × DL19 × 10 reps)

Per v8 Phase C: 10 independent re-runs on each of the 3 required EMNLP families (`Qwen3.5-9B`, `Meta-Llama-3.1-8B-Instruct`, `Ministral-3-8B-Instruct-2512`) on DL19, launched via `submit_emnlp_stability_jobs.sh` which wraps the default 11-block `submit_max_context_jobs.sh --pool-sizes "10 20 30 40 50 100"` layout.

**Cell counts:**
- Scientific Phase C cells (7 EMNLP methods × 3 models × dl19 × 6 pools × 10 reps) = **1260**.
- Stability-layout submissions (11 method blocks including ws-3/ws-PS TopDown and BottomUp overhead × 6 pools × 10 reps × 3 models) = **1980**.

The H4 input-order-stability claim is **scoped to the 18 MaxContext (method × pool) cells per model = 54 cells across the 3 families**, not all 9 method blocks. The 4 ws-3/ws-PS standard-Setwise blocks per model are launched for IDEA_007 layout compatibility but are out of paper scope.

**Source-of-variability decision (v4 with Phase F):** Phase C and Phase F jointly cover the H4 stability scope:

- **Phase C (deterministic 10× byte-equality)**: 10 reps with `do_sample=False` at `setwise.py:409` measure deterministic-implementation correctness — i.e. that 10 reps produce byte-identical `.eval` files. This is the implementation guard.
- **Phase F (MaxContext-only `--reverse` and `--shuffle` controls)**: per-comparison reverse and fixed-seed-929 shuffle of the remaining BM25 pool, applied only to MaxContext methods. The paired forward-vs-reverse and forward-vs-shuffle deltas measure **input-order sensitivity** — the empirical H4 evidence the paper claims. This supersedes the legacy v1/v2 `--seed`-shuffle protocol; the fixed seed means repeated Phase F stability runs measure system/model nondeterminism, not shuffle-seed variance.

The paper's H4 input-order claim is sourced from Phase F's paired deltas at the per-(model, dataset, method, pool) granularity over the representative subset (3 models × 4 datasets × 3 MaxContext methods × 6 pools × 2 conditions = 432 cells).

**Pass criterion (claim H4 under either interpretation):** for each (family, method, pool-size) cell, the nDCG@10 across 10 runs satisfies all three of: SD $\le 0.005$, max−min range $\le 0.015$, worst-pair $|\Delta| \le 0.015$. Under the byte-equality interpretation these will trivially pass; under the input-order interpretation they're a real test.

**Compute (v8 Phase C):** ~1620 stability-layout jobs across 3 families × 10 reps × 9 method blocks × 6 pools. Qwen3.5-9B / DL19 single-DualEnd-job wall-clock at $N=50$ is estimated at $\sim 27$ min (per `EMNLP_BUDGET.md` ballparks); $N=100$ doubles plus prefill overhead. Mean per-job ≈ $30$-$60$ min, so Phase C is roughly **800–1500 H100-hours** for required EMNLP families. Optional Phase E (Qwen3-8B / DL19 / 5 pools / 10 reps = 350 jobs) adds another ~150-300 H100-hours.

### Phase F — Position-bias controls (MaxContext only)

Phase F adds the evidence for the paper's position-bias discussion without changing the standard Heap/Bubble baselines. It compares Phase B forward outputs against two new MaxContext-only conditions:

- `--reverse`: reverse the remaining BM25 pool before each LLM comparison.
- `--shuffle`: shuffle the remaining pool before each LLM comparison with fixed seed 929.

Required scope: 3 representative models (Qwen3.5-9B, Llama-3.1-8B-Instruct, Ministral-3-8B-Instruct-2512) × 4 representative datasets (DL19, DL20, DBPedia, FiQA) × 3 MaxContext methods × 6 pools × 2 new conditions = **432 jobs**. The fixed seed means repeated Phase F stability runs measure system/model nondeterminism, not shuffle-seed variance. The analysis reports paired within-query nDCG@10 deltas for forward-vs-reverse and forward-vs-shuffle.

### Phase 4 — Tier 2 (appendix only)

Triggered only if Phases 1–3 are green and budget allows.

- Smaller / larger Qwen3.5: $3 \text{ models} \times 3 \text{ methods} \times 5 \text{ pools} \times 6 \text{ datasets}$, with same trim as Phase 2 = $\sim 200$ jobs at $\sim 720$ H100-h.
- Ministral-3B / Ministral-14B: only if HF identifiers resolve. $2 \text{ models}$ at the same trim = $\sim 130$ jobs at $\sim 480$ H100-h.
- Two extra BEIR domains (touche2020, fiqa): $4 \text{ Tier-1 models} \times 3 \text{ methods} \times 5 \text{ pools} \times 2 \text{ datasets} = 120$ jobs at $\sim 360$ H100-h (fiqa is the largest contributor with 648 queries).

**Phase-4 total upper bound:** $\sim 1560$ H100-hours.

### Phase 5 — Defensive smokes (appendix)

To pre-empt anticipated reviewer asks (per `EMNLP_SHORT_PLAN.md` §9 risk register):

- ~~**Order-robustness pilot at $N{=}50$.** Qwen3.5-4B + Llama-3.1-8B × DL19 × {forward, inverse, random shuffle (seed 0)} × DualEnd = 6 jobs. $\sim 12$ H100-h.~~ **Superseded by Phase F (canonical 432-cell representative subset).** The Phase F output (paired forward/reverse/shuffle deltas at the full Tier-1 pool sweep $N \in \{10,20,30,40,50,100\}$) replaces this 6-job pilot at strictly broader scope.
- **TourRank smoke.** Single matched-budget point on Qwen3.5-4B / DL19 with TourRank-10 (or whatever cost matches 25 LLM calls at $N{=}50$). $\sim 5$ H100-h.
- **Original-Setwise direct-baseline smoke** (per Codex round-1 §10). Qwen3.5-4B / DL19 × $c{+}1{=}10$, $k{=}10$ Setwise-Selection (letter alphabet), exactly the configuration that *might* be claimed equivalent to MaxContext-TopDown at $N{=}10$. Single job comparing call count + nDCG@10. $\sim 2$ H100-h.
- **Passage-length sweep.** Qwen3.5-4B × DL19 × {pl 64, 128, 256, 512} × DualEnd = 4 jobs. $\sim 8$ H100-h.

**Phase 5 total:** $\sim 27$ H100-h.

### Total compute envelope (revised)

| Phase                            | H100-hours (est.) | Cumulative |
|----------------------------------|-------------------|------------|
| Phase 0                          | 0                 | 0          |
| Phase 1 (family-fragility smoke) | 12                | 12         |
| Phase 2 (Tier 1 trimmed)         | 1370              | 1382       |
| Phase 3 (10× stability)          | 350               | 1732       |
| Phase 5 (defensive smokes)       | 27                | 1759       |
| Phase 4 (Tier 2, optional)       | 1560              | 3319       |

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
- [x] **Superseded by Phase F:** `--shuffle` (fixed seed 929) and `--reverse` plumbed through `run.py` and forwarded by `submit_emnlp_jobs.sh` / `submit_emnlp_stability_jobs.sh`. The legacy `--seed` + `--shuffle_ranking random` plumbing is no longer required since Phase F's MaxContext-only per-comparison reverse/shuffle now sources input-order H4 evidence.
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
- v3 (2026-05-06) — aligns experiment plan with the canonical v8 EMNLP plan (`../EMNLP_EXPERIMENT_PLAN.md`, `../EMNLP_IMPLEMENTATION_PLAN.md`, `../EMNLP_PAPER_DESIGN.md`):
  - **HF identifier fixes:** Qwen3.5 stripped of trailing `-Instruct` (corrected against actual HF release names; v8's canonical IDs are `Qwen/Qwen3.5-{0.8B,2B,4B,9B,27B}`). Llama-3.1 prefixed with `Meta-` to match v8's `meta-llama/Meta-Llama-3.1-8B-Instruct` canonical form.
  - §1 model lineup: Tier-2 row added for v8 Phase D/E optional Qwen3 family (Qwen3-{0.6B,1.7B,4B,8B,14B,32B}). Note that v8's required matrix is broader (9 models) than the paper's Tier 1 (4 models); paper headline is the 4-model subset.
  - §1 code prerequisites: items 1, 2, 3, 8 marked ✅ DONE in v8 implementation. Items 4, 5 covered by Phase A smoke. Items 6, 7 noted as still-TODO with v8 fallback semantics.
  - §3 Phase 3 stability: rewritten to reflect v8 Phase C scope (3 EMNLP families × DL19 × 10 reps via `submit_emnlp_stability_jobs.sh`); acknowledges that v8 Phase C uses deterministic 10× (byte-equality), while the original paper-plan input-order H4 protocol requires additional `--seed` plumbing (TODO; appendix-only if reviewer demands).
  - §3 Phase 1 / 2: pool sweep extended to include $N=100$ for the 3 required EMNLP families, gated on Phase A pool=100 smoke + BEIR pool=100 fit probe (`scripts/probe_beir_pool100_fit.py`).
  - §7 pre-launch checklist: many items now done by v8 implementation; remaining items flagged.
  - Header: status updated; relationship to v8 plan documented.
- v4 (2026-05-07) — re-aligned to canonical root docs after the EMNLP isolation restructure (`../EMNLP_PAPER_DESIGN.md` v5, `../EMNLP_EXPERIMENT_PLAN.md`, `../EMNLP_IMPLEMENTATION_PLAN.md` v4):
  - **Cross-reference cleanup:** title and root-canonical refs corrected after the over-broad sed during the EMNLP isolation restructure mangled `EXPERIMENT_PLAN.md` paths into `Extra-Experiments/EXPERIMENT_PLAN.md`. All paper-relative refs now point at the post-restructure root locations.
  - §1 code prerequisites item 6 (`--seed` plumbing): flipped from ⚠️ TODO to ✅ DONE — superseded by Phase F's MaxContext-only per-comparison `--shuffle` (fixed seed 929) and `--reverse` flags. v8 Phase C remains deterministic 10× (byte-equality); Phase F now sources input-order H4 evidence via paired forward-vs-reverse / forward-vs-shuffle deltas.
  - §3 Phase 3 source-of-variability decision: rewritten so Phase C and Phase F each carry their own H4 view (deterministic implementation guard vs. input-order sensitivity). The "appendix-only if reviewer demands" recommendation from v3 is dropped — Phase F is now a required canonical phase per `../EMNLP_PAPER_DESIGN.md` §5 and `../EMNLP_IMPLEMENTATION_PLAN.md` §7.
  - §3 Phase 5 order-robustness pilot (6 jobs, ~12 H100-h): struck through as superseded by Phase F's 432-cell representative subset (3 models × 4 datasets × 3 MaxContext methods × 6 pools × 2 conditions). Phase F covers strictly broader scope.
  - §7 pre-launch checklist: `--seed` + `--shuffle_ranking random` propagation item replaced by the Phase F `--shuffle` / `--reverse` plumbing entry, marked done.
  - Multimodal loader (Mistral 3 + Qwen 3.5 vision-language configs through `MULTIMODAL_MODEL_TYPES` + `AutoProcessor` + `AutoModelForImageTextToText`) carried over from canonical `../EMNLP_IMPLEMENTATION_PLAN.md` v3 — already reflected in §1 item 1/2 ✅ DONE markers.
  - No change to compute envelope, statistical protocol, page budget, or H1–H4 thresholds.

---

*Companion:* [`EMNLP_SHORT_PLAN.md`](EMNLP_SHORT_PLAN.md) — section plan, claims, novelty positioning, branch resilience, page budget.
