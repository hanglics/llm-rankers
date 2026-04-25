# Implementation Status

**Date:** April 25, 2026
**Pipeline Stage:** Stage 2 — Implementation (maintained; idea:007 MaxContext family landed; paper-facing cleanup in progress)

## Current Scope

The codebase now supports the full directional-asymmetry study plus the follow-up routed/joint-signal refinements:

- `TopDown` setwise ranking (baseline)
- `BottomUpSetwiseLlmRanker`
- `DualEndSetwiseLlmRanker`
- `BidirectionalEnsembleRanker`
- `SelectiveDualEndSetwiseLlmRanker`
- `BiasAwareDualEndSetwiseLlmRanker`
- `SameCallRegularizedSetwiseLlmRanker`
- `MaxContextDualEndSetwiseLlmRanker`
- `MaxContextTopDownSetwiseLlmRanker`
- `MaxContextBottomUpSetwiseLlmRanker`

### Implemented (idea:007 — see `/Users/hangli/projects/llm-rankers/IDEA_007.md`)

- `MaxContextDualEndSetwiseLlmRanker` — one-prompt double-ended selection over the whole rerank pool. Qwen-generation only (hard gate); `pool_size == hits == ranker.k` invariant; numeric labels 1..N; context-fit preflight uses actual tokenization; strict parse aborts on bad outputs.
- `MaxContextTopDownSetwiseLlmRanker` — one-prompt best-only selection over the whole live pool. Same Qwen-generation-only contract, numeric labels 1..N, strict parse, and preflight.
- `MaxContextBottomUpSetwiseLlmRanker` — one-prompt worst-only selection over the whole live pool. Same Qwen-generation-only contract, numeric labels 1..N, strict parse, and preflight.

## Core Implementations

### `llmrankers/setwise_extended.py`

- `BottomUpSetwiseLlmRanker`
  - Reverse worst-selection prompt via `compare_worst()`
  - Bottom-up heapsort and bubblesort variants
  - Correct top-k survivor handling after bottom-up extraction

- `DualEndSetwiseLlmRanker`
  - Joint best/worst prompt via `compare_both()`
  - Cocktail-shaker bubblesort and double-ended selection sort
  - Tournament-based best/worst selection for large candidate pools
  - Robust dual-output parsing for generation mode

- `BidirectionalEnsembleRanker`
  - Shared-weight top-down + bottom-up execution
  - `rrf`, `combsum`, and weighted fusion
  - Logging context now propagates to both sub-rankers

- Routed DualEnd refinements
  - `SelectiveDualEndSetwiseLlmRanker`
  - `BiasAwareDualEndSetwiseLlmRanker`
  - `SameCallRegularizedSetwiseLlmRanker`
  - Query-local BM25 score-spread percentile gating

### `llmrankers/setwise.py`

- Added causal-model support via `AutoModelForCausalLM`
- Qwen3 / Qwen3.5 families supported through trusted remote code
- Completion token accounting fixed for causal decoding
- Vicuna v1.5 chat-template condition corrected

### `run.py`

- Directional methods exposed through CLI:
  - `topdown`
  - `bottomup`
  - `dualend`
  - `bidirectional`
  - `selective_dualend`
  - `bias_aware_dualend`
  - `samecall_regularized`
  - `maxcontext_dualend`
  - `maxcontext_topdown`
  - `maxcontext_bottomup`
- Bias-aware DualEnd now rejects unsupported `heapsort`
- Comparison logging hook remains available through `--log_comparisons`

## Important Fixes Since The Initial March Snapshot

### Correctness / behavior

1. Bottom-up bubblesort and heapsort fixes were applied and re-verified.
2. DualEnd compare-both safety and swap bookkeeping were corrected.
3. Same-call worst demotion now only applies outside a protected head frontier.
4. Selective heapsort disables shortlist gating because heap indices are not rank positions.
5. Bias-aware DualEnd is restricted to sort methods that actually exercise the order-robust joint path.

### Logging / analysis

1. `compare_worst()` logging is active for normal worst-selection calls.
2. Tournament worst-selection in DualEnd selection now reuses the same logging/accounting path.
3. Bidirectional comparison logging now propagates to both internal rankers.
4. Position-bias analysis documents when chi-squared expected-count assumptions are weak.

### Model / scoring support

1. Qwen3.5 causal support was added through `AutoModelForCausalLM`.
2. Causal likelihood scoring for short answer strings is supported for Qwen/Qwen3.5 follow-up analysis.
3. DualEnd likelihood is now explicitly documented as a best-only proxy rather than exact joint `Best: X, Worst: Y` likelihood.

### Experiment scripts

1. Main extended setwise sweep defaults to `passage_length=512`.
2. Ablation scripts now choose safer passage-length defaults for Qwen-family models and warn on unusually short overrides.

## Current Caveats

- DualEnd likelihood remains a proxy for joint best/worst scoring.
- Bubble-style refinements remain approximate because they inherit the baseline bubble traversal structure.
- The paper text still needs to stay aligned with the latest significance analysis and per-method win counts.

## Recommended Source Of Truth

For implementation details and recent fixes, use these files first:

- `llmrankers/setwise.py`
- `llmrankers/setwise_extended.py`
- `run.py`
- `research_pipeline_setwise/FINDINGS.md`
- `research_pipeline_setwise/SIGNIFICANCE_TESTS.md`

## Implementation Notes

### 2026-04-21 — `exp:same_method_tables_pending` closed (Need_to_Run priority #1)

- Added `analysis/significance_tests_pairwise.py` (reuses helpers from `analysis/significance_tests.py`).
- Produced 12 pairwise same-sort comparison tables (6 groupings × DL19/DL20) with paired approximate-randomization + Bonferroni correction per (grouping, dataset).
- Authoritative artifacts: `research_pipeline_setwise/SIGNIFICANCE_TESTS_PAIRWISE.{md,json}`. Inlined into `results-display/index.html` under `section id="pairwise-tables"`.
- Headline: cleanest positive finding is DualEnd (`DE-Cocktail` + `DE-Selection`) vs `TD-Bubble` on DL19 — 2 Bonferroni-significant wins on Qwen3-8B. All BU and BiDir groupings confirm directional asymmetry with multiple Bonferroni-significant losses.

### 2026-04-25 — MaxContext family (idea:007) parser hardening + n_docs=2 BM25 endgame

Added MaxContext TopDown and BottomUp to the idea:007 family alongside the existing MaxContext DualEnd. Cluster runs of the shipped TopDown / BottomUp variants on Qwen3-4B / DL19 surfaced three distinct failure modes resolved in two coordinated interventions:

**Failure modes (Codex two-round investigation)**

| Mode | Symptom | Root cause |
|---|---|---|
| (i) Bare out-of-window hallucination | `ValueError: ... Raw text: '3<\|im_end\|>'` at small windows | Model emits a label visible in earlier rounds but not in the current shrinking window. At `n_docs=2`, parser correctly returns None and strict-mode raises. |
| (ii) Silent multi-digit collapse | No abort. `"10"` silently parsed as `"1"`, `"30"` as `"3"`. | `_parse_single_label`'s all-found-char loop iterated single characters against multi-character valid set. Numeric scheme labels had to be handled before this loop fires. |
| (iii) Long-form refusal/hedging | Multi-paragraph reasoning + invented passage labels (e.g. `"Passage 3 is most relevant, the others are equally relevant"`) | Model can't commit at small windows when both surviving docs seem relevant. Existing refusal whitelist missed `"no least relevant"`/`"both are equally relevant"`/`"if there was a Passage"`; numeric fallback greedily grabbed the FIRST integer. |

**Interventions**

1. **Parser hardening** (`llmrankers/setwise.py`, `_parse_single_label`):
   - Multi-digit collapse fix: skip the all-found-char loop for numeric scheme.
   - New numeric-structured-parse stage **before refusal detection** with decisive anchors only (`BEST|WORST|MOST RELEVANT|LEAST RELEVANT|ANSWER|OUTPUT`) — bare `PASSAGE N` deliberately excluded so hypotheticals like `"If there was a Passage 3..."` fall through to refusal handling.
   - Expanded refusal whitelist with anchored phrases (`no least relevant`, `both are equally relevant`, `if there was a passage`, etc.).
   - Refusal-before-numeric-fallback reorder gated to numeric scheme only; letter and bigram schemes keep byte-identical legacy ordering.
   - Strict-mode hook (`strict_no_parse_fallback=True`) now returns None on refusal.

2. **n_docs=2 deterministic BM25 endgame** (`llmrankers/setwise_extended.py`):
   - At the last round when the live window contains exactly 2 docs, MaxContext TopDown and BottomUp skip the LLM and use BM25 score as a deterministic tiebreaker.
   - TopDown: higher score wins; on tie, smaller original BM25 index wins.
   - BottomUp: lower score loses (placed at bottom); on tie, larger original BM25 index becomes worst.
   - Snapshot `orig_pos = {doc.docid: i for i, doc in enumerate(docs)}` taken at selection-method entry so tie-breaking uses input order even after per-round swaps mutate the live ranking.
   - Score-presence guard raises ValueError if any `doc.score` is None or non-finite.
   - New counter `total_bm25_bypass` exposed on each ranker; `run.py` accumulates it across queries and prints `Avg BM25 bypass` only when non-zero.

**Impact asymmetry between TopDown and BottomUp**

This is a research-grade clarification, not a code change:

- **TopDown's last round** picks between the two LEAST relevant remaining docs (TopDown removes best per round; survivors are the worst). The bypass decides ranks N-1 vs N — tail of ranking, low NDCG@10 impact.
- **BottomUp's last round** picks between the two MOST relevant remaining docs (BottomUp removes worst per round; survivors are the best). The bypass decides ranks 1 vs 2 — head of ranking, materially impactful for NDCG@10.

The bypass is therefore not a uniform "tail-only" perturbation; the paper must report TopDown and BottomUp separately and disclose this honestly.

**Backward compatibility**

- `MaxContextDualEndSetwiseLlmRanker` byte-identical (verified by `git show HEAD diff` against the new commit). DualEnd's algorithm shrinks by 2 per round, so n_docs=2 means a single LLM call placing both best and worst — already bypassed by the loop guard.
- All non-MaxContext rankers behavior-identical: the multi-digit-collapse fix is gated on numeric scheme (set only by the new MaxContext classes); refusal-whitelist additions are pure additions; refusal-before-numeric reorder is gated to numeric scheme.

**Expected counters per run**

- TopDown / BottomUp: `Avg comparisons: pool_size - 2`, `Avg BM25 bypass: 1.0`.
- DualEnd: `Avg comparisons: floor(pool_size / 2)` (unchanged), no bypass line printed.

### 2026-04-25 — Earlier on the same day

Added MaxContext TopDown and BottomUp to the idea:007 family. `MaxContextDualEndSetwiseLlmRanker` was left byte-identical; the only behavior-neutral shared changes are two guarded strict-parse hooks that activate only when `strict_no_parse_fallback=True`, which the new MaxContext single-extreme variants set.
