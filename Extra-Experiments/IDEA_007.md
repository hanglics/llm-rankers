# IDEA_007 — MaxContext family: one-prompt whole-pool selection over ≤50-doc pools

> **Status:** Approved plan, Codex-audited (gpt-5.4, xhigh) through 3 rounds to READY_TO_EXECUTE on 2026-04-20.
> **Refines:** idea:002 (DualEnd).
> **Target gaps:** gap:G1 (information extraction), gap:G4 (efficiency-effectiveness frontier).
> **Inspired by:** paper:zhuang2024_setwise, paper:sato2026_sorting_survey, paper:zhuang2025_rank_r1 (num_child=19 precedent for large setwise windows).

## 1. Context

The current DualEnd family (`idea:002`, `exp:main_de_cocktail` / `main_de_selection`) uses 4-passage windows and costs 5.6–8.9× TD-Heap wall-clock for a +0.0058 mean NDCG@10 gain over best TopDown. `claim:C9` identifies an empty region on the Pareto frontier between TD-Bubble and DE-Cocktail. `Need_to_Run.txt` priority #2 asks for max-context DualEnd on Qwen.

**Cross-link:** EMNLP 2026 multi-family extension lives in `IDEA_008_maxcontext_multi_family.md`. The Qwen-only invariants in this spec remain authoritative; the EMNLP plan documents the deliberate audited widening of `MAXCONTEXT_ALLOWED_MODEL_TYPES` to include Llama-3.1 and Ministral-3.

**Proposal (idea:007):** fit the entire rerank pool into a single Qwen prompt and run one of three MaxContext variants per round: best+worst (DualEnd), best-only (TopDown), or worst-only (BottomUp). DualEnd shrinks the live pool by 2 each round; TopDown and BottomUp shrink by 1. Two orthogonal studies remain: pool-size sweep (does deeper rerank help?) and passage-length sweep (does richer per-passage signal help?).

**Scope gate:** only the Qwen-generation code path exercises true joint elicitation (`setwise_extended.py:455-474`). T5 generation and all `--scoring likelihood` paths collapse to a best-only proxy (documented on `idea:002` and `claim:C10`). This experiment is strictly Qwen + generation.

**Intended outcome:** a Pareto-frontier point that reduces wall-clock (fewer sequential decodes) vs `DE-Cocktail` at comparable NDCG@10, plus the first characterization of how joint elicitation scales from 4-way to 50-way windows on Qwen.

## 2. Algorithm

```
pool := BM25 top-hits          # hits == pool_size, set by CLI
top_idx := 0
bottom_idx := len(pool) - 1
assert pool_size == hits == ranker.k          # hard invariant

while top_idx < bottom_idx:
    window := pool[top_idx .. bottom_idx]     # whole live pool
    best, worst := dual_prompt_numeric(window)
    place best  at rank top_idx;    top_idx    += 1
    place worst at rank bottom_idx; bottom_idx -= 1

if top_idx == bottom_idx:                     # one remaining doc
    place it at rank top_idx
```

- **No early termination.** Every doc gets a rank (needed for both top-k NDCG and per-position bias analysis).
- **No auto-batching, no auto-truncation.** If any prompt would exceed `max_input_tokens - reserved_output_tokens`, the ranker raises.
- Reuses `_double_ended_selection` (`llmrankers/setwise_extended.py:840-942`) with `num_child` overridden internally so the "single-group" fast-path (`setwise_extended.py:860-889`) fires for the whole window. `num_child` is not a user-facing parameter for this direction.

### 2.1 Variants

- **MaxContext DualEnd** — choose best and worst from the whole live pool in one call, place both, shrink by 2. Call count: `floor(N / 2)`.
- **MaxContext TopDown** — choose only the best from the whole live pool, place it at the next top rank, shrink by 1. Uses an `n_docs=2 deterministic BM25 endgame`; call count: `N - 2` LLM calls + 1 BM25 bypass.
- **MaxContext BottomUp** — choose only the worst from the whole live pool, place it at the next bottom rank, shrink by 1. Uses the same `n_docs=2 deterministic BM25 endgame`; call count: `N - 2` LLM calls + 1 BM25 bypass.

## 3. Code changes (this is a refactor, not a prompt tweak)

### 3.1 Numeric labels end-to-end

The existing `CHARACTERS = [...23 letters]` (`setwise.py:32`) is used across prompt construction, parser return, fallback clamping (`setwise_extended.py:658`: `self.CHARACTERS[n_docs-1]`), JSONL logging, and the analysis pipeline. For `N > 23` we need:

- **Prompt:** add `_format_passages_numeric(docs)` emitting `Passage 1: … Passage N: …` and a footer `Output only in the format: Best: [number], Worst: [number]`. Gate on ranker class (not on a general `--direction` switch).
- **Parser:** `_parse_dual_output` / `_try_parse_dual_output` (`setwise_extended.py:555-631`) — tighten the numeric patterns to validate that extracted integers are in `[1, N]`; reject out-of-range with **no silent default**. Today the parser can grab the first two in-range numbers anywhere in the output; this permissive behaviour is a landmine at N=50.
- **Fallback clamping:** audit every `self.CHARACTERS[...]` index site in `setwise_extended.py` (Codex round 1 / 2 identified lines 555, 623, 658, plus sites in `_select_best_and_worst` and the BottomUp / BiDir code paths). Replace with a per-ranker `self._labels` list so the numeric direction uses a 1..N list and the letter direction keeps A..W. **Must be behaviour-preserving for all existing directions.**
- **JSONL logs:** add a `label_scheme` field with values `"letters_a_w"` or `"numeric_1_based"`.
- **Analysis pipeline:** `analysis/position_bias.py:48-59` currently aggregates logs by `type` only. Update to (a) read `label_scheme` per input file; (b) **refuse** mixed-scheme inputs in a single run with a clear error; (c) render numeric (1..N) vs letter positions in separate output tracks.

### 3.2 New ranker class `MaxContextDualEndSetwiseLlmRanker`

In `llmrankers/setwise_extended.py`. Inherits from `DualEndSetwiseLlmRanker` with overrides for prompt building (numeric) and window construction (whole live pool).

Constructor **asserts** (aborts otherwise):

- model family ∈ Qwen3 / Qwen3.5.
- `scoring == 'generation'`.
- `pool_size == hits == ranker.k`.
- `num_permutation == 1` (existing code miscounts `total_compare` if > 1 but `compare_both()` does not actually permute — Codex round 2 catch).
- Context-fit preflight passes (see §3.3).

Internally sets `num_child = pool_size - 1` so `_double_ended_selection`'s single-group branch fires.

Tracks per-call parse status, tokenized prompt length, and a truncation flag.

**Abort-on-bad-parse policy.** Any out-of-range label, duplicate best/worst, or fallback parse triggers immediate run-abort with a diagnostic. **No retry.** `_generate()` (`setwise.py:304`) is greedy-deterministic (`do_sample=False`) so retrying the identical prompt reproduces the same output. If retry is later worth implementing, it must use a *different* repair prompt (e.g. a strict "Output only `Best: <int>, Worst: <int>` with integers in [1, N]"), not the same prompt.

### 3.3 Preflight context-fit check + runtime assertion

Extend `setwise.py:161-221`:

- `compute_max_fit_window(tokenizer, query_text, sample_docs, passage_length, max_input_tokens, reserved_output_tokens=128) -> int` builds the **fully rendered** MaxContext prompt using the runtime's actual `truncate()` helper (`setwise.py:662`) and calls `tokenizer.encode` on it. Arithmetic bounds (`pool_size × passage_length + overhead`) are not sufficient because `truncate()` does a tokenize→decode→re-tokenize roundtrip that drifts from the arithmetic assumption (Codex round 2 catch).
- At ranker construction, assert `pool_size ≤ max_fit_window(...)`. Abort otherwise.
- Inside `_tokenize_inputs` (currently warns + silently clips at `setwise.py:195`), add a `strict_no_truncation` flag. MaxContext sets it; any truncation raises.
- Qwen runs use `enable_thinking=False` (`setwise.py:128`) → output budget is small (~64–128 tokens for `Best: N, Worst: M`), not 512.

### 3.4 CLI (`run.py`)

- Add `'maxcontext_dualend'` to `--direction` choices (~line 357).
- Dispatch in `main()` (~lines 85–172) to `MaxContextDualEndSetwiseLlmRanker`.
- Do **not** overload `--k`. The launcher passes `pool_size` to both `--hits` and `--k`; the dispatch asserts equality.
- Pass `--query_length` explicitly (current launchers default it to 128, not 32).

### 3.5 Launchers

`experiments/run_maxcontext_dualend.sh` (base) + `run_maxcontext_dualend_pool_sweep.sh` + `run_maxcontext_dualend_pl_sweep.sh`. All set `HITS=${POOL_SIZE}` and `K=${POOL_SIZE}` explicitly.

### 3.6 Files to modify

- `llmrankers/setwise.py` — numeric-label helper; `compute_max_fit_window`; strict no-truncation flag; audit of `CHARACTERS` use-sites.
- `llmrankers/setwise_extended.py` — `MaxContextDualEndSetwiseLlmRanker`; replace hard-coded `self.CHARACTERS[...]` indexes with `self._labels`; stricter parser for numeric outputs; abort-on-bad-parse policy; reuse `_double_ended_selection` (840–942) single-group branch.
- `run.py` — direction choice (~357), dispatch (~85–172), `--query_length` in launchers.
- `analysis/position_bias.py` — read `label_scheme`; refuse mixed-scheme inputs; support numeric positions.
- Result JSONL writers — add `label_scheme` field.
- `experiments/run_maxcontext_dualend.sh` + pool / pl sweep launchers (new).
- `research-wiki/` — new `idea_007_maxcontext_dualend.md`, new exp pages, updated `maxdoc_dualend_pending.md`, edges, index, query_pack, log.

## 4. Context fit (actual, not arithmetic)

Qwen non-thinking mode:

- Qwen3-4B / Qwen3-8B / Qwen3-14B: native context **32,768** tokens.
- Qwen3.5-9B: native context **262,144** tokens.
- Output budget with `enable_thinking=False`: ~64–128 tokens.

Worst-case model × pool × pl (Qwen3-4B, pool=50, pl=512, query_length=128):

- Passage content: 50 × 512 = 25,600
- Query: ≤ 128
- Prompt scaffolding: ~300 (instruction + 50 numeric labels + footer)
- Output reserve: 128
- Total: ~26,156 — fits in 32,768.

Any (model, pool, pl) tuple that the preflight helper fails is dropped from the run set before submission.

## 5. Experiments

### 5.1 Study A — pool-size sweep at fixed `pl=512`

|          |                                                                                                     |
|----------|-----------------------------------------------------------------------------------------------------|
| Fixed    | `passage_length = 512`, `direction = maxcontext_dualend`, `scoring = generation`, Qwen non-thinking |
| Variable | `pool_size ∈ {10, 20, 30, 40, 50}`                                                                  |
| Models   | Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B                                  |
| Datasets | DL19, DL20                                                                                          |
| Matrix   | 6 × 5 × 2 = **60 runs**                                                                             |

**Hypothesis:** saturation, not monotonic. Gain from 10 → 20 → 30 → 40 → 50 should flatten; the plateau point is the efficient pool size.

### 5.2 Study B — passage-length sweep at predeclared `pool_size`

|             |                                                                                                                                                                                                |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fixed       | `pool_size` = smallest Study A value within 0.003 NDCG@10 of Study A max (predeclared rule; Codex round 2)                                                                                     |
| Variable    | `passage_length ∈ {64, 128, 256, 512}`                                                                                                                                                         |
| Direction   | `maxcontext_dualend`                                                                                                                                                                           |
| Control arm | same pl sweep with `direction = dualend, num_child = 3` on the same 6 Qwens × 2 datasets — preserves joint best+worst prompting + parser family. TopDown pl-curves are **supplementary only**. |
| Matrix      | 6 × 4 × 2 × 2 arms = **96 runs** (48 of which are the control arm)                                                                                                                             |

**Confound-handling rule:** if shorter `pl` wins in MaxContext but not in the control arm, attribute to long-context attention degradation. If shorter `pl` wins in both, attribute to Qwen's general passage-signal / length sensitivity. Any other pattern → further investigation before claiming anything.

### 5.3 Study C — order-robustness pilot (launch gate)

|           |                                                                            |
|-----------|----------------------------------------------------------------------------|
| Models    | **Qwen3-4B** (tightest context) + **Qwen3.5-9B** (representative mid-size) |
| Datasets  | DL19, DL20                                                                 |
| Orderings | BM25 forward, BM25 reversed, random shuffle (fixed seed)                   |
| Fixed     | `pool_size = 50`, `pl = 512`                                               |
| Matrix    | 2 × 2 × 3 = **12 runs**                                                    |

**This is a smoke gate, not an order-robustness claim.** Three orderings cannot rule out order effects.

**Gate rule:** if max pairwise NDCG@10 Δ across orderings ≤ 0.01 (within typical 43-query TREC DL bootstrap CI), proceed to Studies A and B. If > 0.01, MaxContext is order-sensitive at w=50 — escalate before launching the full matrix. Options: (a) restrict to `pool_size=20` where the prompt fits comfortably within attention limits; (b) pivot to a `bias_aware_dualend`-with-large-windows derivative; (c) investigate the ordering effect as a finding.

### 5.4 Baselines — matched `hits`, predeclared depths

Codex critical correction (round 1): old `hits=100` runs **cannot be subsetted** to `hits=50`, because `run.py:245` reads only `hits` BM25 docs before any LLM call. Different `hits` = different experiments.

Codex critical correction (round 2): choosing baseline depth post-hoc creates a selection confound. Baseline depths are **predeclared** before Study A runs.

**Predeclared baseline grid** — `pool_size ∈ {10, 30, 50}` (cheapest / mid / deepest Study A anchors). For each anchor, on 6 Qwens × 2 datasets, rerun:

- `TD-Heap` at `hits=pool_size, k=10`
- `TD-Bubble` at `hits=pool_size, k=10` (default `num_child=3` from launchers; see comparisons-axis note below if running whole-pool diagnostics)
- `DE-Cocktail` at `hits=pool_size, num_child=3, k=10`
- `DE-Selection` at `hits=pool_size, num_child=3, k=10`

Matrix: 6 × 3 × 2 × 4 = **144 baseline runs**.

**Comparisons-axis note for `TD-Bubble`.** A pre-fix whole-pool `TD-Bubble` run with `hits=k=num_child=10` produced `Avg comparisons: ~6.98` instead of the intuitive 9 because the local outer clamp interacted with the upstream `last_start` tail-jump and the one-document skip. That was a real control-flow effect, not a logging artifact, but it has been fixed for the exact whole-pool branch: `SetwiseLlmRanker.rerank()` now disables only the outer clamp when `len(ranking) == k == num_child` or `num_child >= len(ranking)`, while still skipping one-document windows. Current verification gives 9 comparisons for `n=10,num_child=10,k=10`.

Do not use the archived `~6.98` pre-fix result as an efficiency claim. Also do not describe standard `TD-Bubble` as algorithmically identical to `MaxContext-TopDown`: MaxContext uses the dedicated MaxContext prompt/parser path and an explicit two-document BM25 bypass. Use `MaxContext-TopDown` directly for the canonical whole-pool best-only baseline. Full mechanism: [`research_pipeline_setwise/FINDINGS.md`](research_pipeline_setwise/FINDINGS.md) (2026-04-27 entry); claim note: [`research-wiki/claims/C9_pareto_frontier.md`](research-wiki/claims/C9_pareto_frontier.md).

**Budget-restricted alternative** (if GPU budget is tight): `pool_size ∈ {10, 50}` = 96 runs. Loses mid-range comparison. Not recommended for the headline claim.

**Headline comparison rule:** for each `pool_size ∈ {10, 30, 50}`, MaxContext's NDCG@10 must match or exceed the best baseline at the same `pool_size` (and matched `k=10`) on a bootstrap-CI basis. Comparisons against different `pool_size` baselines are reported as context, not as the primary win.

## 6. Metrics captured per run

- NDCG@10 (primary).
- Total comparisons, total prompt tokens, total completion tokens, wall-clock.
- Per-call parse status (`full_parse`, `abort`).
- Per-position label frequency with `label_scheme: numeric_1_based`. MaxContext TopDown logs `type=best`, MaxContext BottomUp logs `type=worst`, and MaxContext DualEnd logs `type=dual_best` + `type=dual_worst`. This feeds a **separate new analysis page**, not an extension of `exp:analysis_position_bias` (which is w=4-specific). `claim:C5` does not trivially extend to w=50.
- Truncation flag per prompt (must be zero across the run).

## 7. Risks

1. **Long-context attention degradation** (`paper:liu2024_lost_in_middle`). Primary risk. Study B's control arm is the designed test.
2. **Numeric-label parse fragility at N=50.** Existing parser's silent-default behaviour is a landmine. Abort-on-bad-parse is mandatory.
3. **Order sensitivity.** Study C gates the full matrix. The `n_docs=2 deterministic BM25 endgame` is asymmetric: TopDown resolves the tail of the ranking, while BottomUp resolves ranks 1-2.
4. **Scope confound vs `DE-Cocktail hits=100`.** Addressed by matched-`hits` baselines; must never compare MaxContext hits=50 against DE-Cocktail hits=100 directly.
5. **Token-frontier framing.** MaxContext uses *more* prompt tokens than DE-Cocktail. `claim:C9`'s token axis will not improve; only comparisons and wall-clock may. Paper framing must be precise.
6. **`claim:C10` impact.** This is **not** an automatic ICTIR story upgrade. Even a successful MaxContext only contributes to the efficiency axis; the core setwise-asymmetry narrative (`claim:C1`, `claim:C8`) is unchanged.

## 8. Staged execution (do **not** submit the full 312-run matrix at once)

Total: 60 (Study A) + 96 (Study B) + 12 (Study C) + 144 (baselines) = **312 runs**.

| Phase                                | Runs                                                                   | Gate                                                                                                          |
|--------------------------------------|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| 1 — Unit-level sanity                | 1 (Qwen3-4B, DL19, pool=10, pl=512)                                    | algorithm terminates; numeric labels parse 100%; ranking is a permutation; preflight passes; zero truncations |
| 2 — Study C order-robustness pilot   | 12                                                                     | max NDCG@10 Δ across orderings ≤ 0.01                                                                         |
| 3 — Matched-hits regression check    | 1 pair (Qwen3-8B DL19, MaxContext pool=50 vs DE-Cocktail hits=50 nc=3) | MaxContext ≥ DE-Cocktail within bootstrap CI                                                                  |
| 4 — Study A + predeclared baselines  | 60 + 144 = 204                                                         | none (run in parallel)                                                                                        |
| 5 — Study B at predeclared pool_size | 96                                                                     | Phase 4 yielded a defensible pool_size via the selection rule                                                 |

## 9. Verification & wiki follow-up

1. **Parse diagnostic:** run-by-run tally of `full_parse` vs `abort`. Target ≥99% `full_parse`, zero `abort`. Feeds `exp:dualend_parse_success`.
2. **Position-bias plots at w=pool_size:** separate analysis track with `label_scheme: numeric_1_based`; compare to existing w=4 plots as a new claim candidate, not an extension of `claim:C5`.
3. **Pareto plot update:** add pool-sweep and pl-sweep points to `results/analysis/pareto/QUALITY_COST_PARETO.md`. Success = at least one (model, pool_size, pl) lands in the empty region between `TD-Bubble` and `DE-Cocktail` on the **comparisons-axis** and **wall-clock-axis** frontiers (token-axis is not expected to improve and should not be claimed).
4. **Wiki audit:** re-run Codex audit on the new `idea:007` + exp pages once results exist, using the same template as the wiki backfill cycle (3 rounds converged to READY_TO_MERGE on 2026-04-20).

## 10. Out of scope (for this first pass)

- Flan-T5 (context too small; best-only proxy confound across the MaxContext family).
- `--scoring likelihood` on any model for any MaxContext direction (silently degrades to a best-only proxy).
- `pool_size > 50` — requires further context-fit verification and likely different architecture.
- Batched / multi-pass fitting when `pool_size > max_fit`.
- Letter alphabets beyond A-W (numeric is the chosen scheme).
- `bias_aware_dualend` or `samecall_regularized` derivatives of MaxContext — stacks later if the base method works.
- Auto-tuning `pool_size` per query or per model — static sweep first.
- Changing any existing DualEnd / BottomUp / BiDir behaviour; the `CHARACTERS` → `self._labels` refactor must be behaviour-preserving for letter-direction code.
- Llama-3.1 and Ministral-3 families are out of scope for IDEA_007 specifically; they are addressed in IDEA_008.

## 11. Audit trail

- **v1** (draft): MAJOR_REDESIGN_REQUIRED — 23-label alphabet not refactored, post-hoc baseline selection, analysis pipeline not addressed.
- **v2** (round-1 fixes): NEEDS_REVISION — parse-retry no-op, context-fit arithmetic, `num_permutation` miscount.
- **v3** (round-2 fixes): **READY_TO_EXECUTE** (Codex, 2026-04-20).
