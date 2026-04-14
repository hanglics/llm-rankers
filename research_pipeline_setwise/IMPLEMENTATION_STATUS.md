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
