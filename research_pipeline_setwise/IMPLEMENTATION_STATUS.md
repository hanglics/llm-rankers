# Implementation Status

**Date:** April 10, 2026
**Pipeline Stage:** Stage 2 — Implementation (maintained; paper-facing cleanup in progress)

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

## Implementation Note

2026-04-25 — Added MaxContext TopDown and BottomUp to the idea:007 family. `MaxContextDualEndSetwiseLlmRanker` was left byte-identical; the only behavior-neutral shared changes are two guarded strict-parse hooks that activate only when `strict_no_parse_fallback=True`, which the new MaxContext single-extreme variants set.
