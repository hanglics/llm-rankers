# IDEA_007 — Implementation Plan (v2, post-Codex-round-1)

> **Source spec:** `/Users/hangli/projects/llm-rankers/IDEA_007.md` (v3, Codex-audited READY_TO_EXECUTE 2026-04-20).
> **Prime constraint:** no behavioural change to any existing ranker, pipeline, or result. Every existing command must produce byte-identical `.eval` outputs after the change, verified by a regression gate (§ 9).
> **Audit trail:** v1 → Codex round 1 → NEEDS_REVISION (strict-parse incomplete, launcher log-path collision, regression gate not runnable, Qwen invariant too loose, preflight at init unsafe, method invariant not class-enforced, regression coverage too narrow). v2 incorporates every round-1 fix.

## 1. Strategy: extend, do not refactor

The spec described a global `CHARACTERS → self._labels` refactor across 15+ call sites in `setwise.py`, `setwise_extended.py`, and `analysis/position_bias.py`. That surface area is large enough to risk silent regressions on existing rankers. We avoid it.

**Chosen architecture:** extend via new subclass + additive hook points. Existing code paths are touched only through **additive** flags that default to current behaviour.

- `CHARACTERS` stays a 23-letter class attribute on `SetwiseLlmRanker` — not refactored, not renamed.
- The new `MaxContextDualEndSetwiseLlmRanker` **overrides `self.CHARACTERS` on the instance** after `super().__init__()` with a numeric-label list sized to the run's `pool_size`. All inherited code paths that read `self.CHARACTERS[...]` then see numeric labels without any call-site changes.
- Two new base-class flags, both defaulting to `False`, let us tighten behaviour only inside MaxContext: `strict_no_truncation` (on `SetwiseLlmRanker`) and `strict_no_parse_fallback` (on `DualEndSetwiseLlmRanker`).
- Only the new `maxcontext_dualend` CLI branch is added to `run.py`; the existing dispatch chain is not touched.

## 2. Non-goals (explicit backward-compat guards)

- No change to `CHARACTERS` definition on the base class.
- No behavioural change to `SetwiseLlmRanker`, `BottomUpSetwiseLlmRanker`, `DualEndSetwiseLlmRanker`, `BidirectionalEnsembleRanker`, `SelectiveDualEndSetwiseLlmRanker`, `BiasAwareDualEndSetwiseLlmRanker`, `SameCallRegularizedSetwiseLlmRanker`. Every method signature that exists today stays with the same signature and the same control flow unless the change is guarded by a default-False flag.
- No change to existing launcher scripts (`experiments/run_*.sh`).
- No change to existing `.eval` outputs in `results/`. (See § 9 regression gate.)
- No change to `analysis/significance_tests.py`, `analysis/significance_tests_pairwise.py`, `analysis/per_query_analysis.py`, `analysis/ranking_agreement.py`, `analysis/quality_cost_pareto.py`, `analysis/query_difficulty.py`, `analysis/when_dualend_helps.py`.
- No change to existing JSONL comparison-log fields. The new `label_scheme` field is added **after** existing fields; analysis code that did not know about it continues to work because the new class defaults to `"letters_a_w"` behaviour unless explicitly set.
- `analysis/position_bias.py` gains a `label_scheme` read with default `"letters_a_w"` so existing logs (which lack the field) stay backward-compatible.

## 3. File-by-file changes

### 3.1 `llmrankers/setwise.py`

**Additive only.**

- Add class-level default: `strict_no_truncation: bool = False` on `SetwiseLlmRanker`. Constructor keeps its existing signature; new kwarg is optional with default `False`.
- Modify `_tokenize_inputs` at `setwise.py:192-221` warn path:

  ```python
  if (
      self.max_input_tokens is not None
      and lengths
      and max(lengths) > self.max_input_tokens
  ):
      if self.strict_no_truncation:
          raise ValueError(
              f"Prompt length {max(lengths)} exceeds model input limit "
              f"{self.max_input_tokens} and strict_no_truncation=True."
          )
      if not self._warned_input_truncation:
          # existing warn block, unchanged
          ...
  ```

  Default-False path is the existing warn-and-clip; no existing caller opts in.

- Add a module-level helper with a single canonical signature (Codex round 2: unify with § 6):

  ```python
  def compute_max_fit_window(
      ranker: "SetwiseLlmRanker",
      query: str,
      docs: list,
      reserved_output_tokens: int = 128,
  ) -> tuple[bool, int, int]:
      """Render the full MaxContext prompt through the ranker's actual
      _format_passages + chat template + tokenizer path (same truncate()
      roundtrip at setwise.py:662-663). Return (fits, rendered_length, budget)
      where budget = ranker.max_input_tokens - reserved_output_tokens.
      Pure addition — never called by existing rankers."""
  ```

  This is the single authoritative signature; § 3.2 and § 6 reference this exact form. The older tokenizer-only/int-returning variant from v1 is removed.

- Add an optional `label_scheme` field to `_log_comparison` at `setwise.py:110-125`:

  ```python
  entry = {
      "qid": getattr(self, '_current_qid', None),
      "type": comp_type,
      "positions": positions,
      "selected": selected,
  }
  if docs is not None:
      entry["docids"] = [d.docid for d in docs]
  label_scheme = getattr(self, 'label_scheme', None)
  if label_scheme:
      entry["label_scheme"] = label_scheme
  ```

  Existing rankers do not set `self.label_scheme` → field absent from their logs → existing `analysis/position_bias.py` behaviour unchanged.

### 3.2 `llmrankers/setwise_extended.py`

**Additive only; existing classes unchanged.**

- Add `strict_no_parse_fallback: bool = False` default on `DualEndSetwiseLlmRanker`.
- Modify `_parse_dual_output` (`setwise_extended.py:635-673`) fallback paths:

  ```python
  def _parse_dual_output(self, text, n_docs):
      parsed = self._try_parse_dual_output(text, n_docs)
      if parsed is not None:
          return parsed

      if getattr(self, 'strict_no_parse_fallback', False):
          raise ValueError(
              f"MaxContext dual-output parse failed. Raw text: {text!r}"
          )

      # existing silent-default fallback, unchanged
      ...
  ```

  Default-False path is the current silent-default behaviour. Existing DualEnd callers keep the existing behaviour.

- **Close every silent-default path under strict mode** (Codex round 1 critical): audit every pattern in `_try_parse_dual_output` (`setwise_extended.py:562-633`) — Patterns 1 through 6 — and add validation common to all:
  - Both extracted labels must be in range (letter scheme: in `self.CHARACTERS[:n_docs]`; numeric scheme: in `[1, n_docs]` then resolved via `_num_to_label`).
  - Both labels must be distinct.
  - Under `strict_no_parse_fallback=True`, any violation returns `None` so the caller-level strict raise fires. Under the default (False), today's behaviour is preserved.

- **Harden `compare_both()` under strict mode** (Codex round 1 critical): the duplicate-label safety block at `setwise_extended.py:493-499` currently rewrites `worst` silently when `best == worst`. Under `strict_no_parse_fallback=True`, replace that block with `raise ValueError(f"Duplicate best/worst label {best!r} under strict mode")`. Default-False callers hit the existing rewrite path unchanged.

- **New explicit Qwen3 allowlist constant** at module scope of `setwise_extended.py` (so MaxContext does not inherit the broader `QWEN_MODEL_TYPES` that includes `qwen2`):

  ```python
  # Codex round 1: the plan's contract is "Qwen3 / Qwen3.5 only". The existing
  # QWEN_MODEL_TYPES (setwise.py:24) includes qwen2 for unrelated reasons.
  MAXCONTEXT_ALLOWED_MODEL_TYPES = frozenset({"qwen3", "qwen3_moe", "qwen3_5"})
  ```

- **New class `MaxContextDualEndSetwiseLlmRanker(DualEndSetwiseLlmRanker)`** appended to the file. Minimal surface area:

  ```python
  class MaxContextDualEndSetwiseLlmRanker(DualEndSetwiseLlmRanker):
      def __init__(self, *args, pool_size: int, **kwargs):
          # Early fail-fast: reject clearly non-Qwen3 model names before loading
          # the (large) weights. This is a best-effort belt-and-suspenders check;
          # the authoritative check uses self.config.model_type after super().
          self._early_reject_non_qwen3(kwargs.get("model_name_or_path")
                                        or (args[0] if args else None))
          super().__init__(*args, **kwargs)
          self._assert_maxcontext_invariants(pool_size)
          # Override labels on the instance (not the class) with numeric 1..pool_size.
          self.CHARACTERS = [str(i + 1) for i in range(pool_size)]
          # Force the single-group fast-path of _double_ended_selection.
          self.num_child = pool_size - 1
          # Force the selection-sort path: _double_ended_selection is only
          # reached via method='selection' in DualEndSetwiseLlmRanker.
          self.method = "selection"
          # Hard flags for this direction.
          self.strict_no_truncation = True
          self.strict_no_parse_fallback = True
          # For JSONL log tagging.
          self.label_scheme = "numeric_1_based"
          self._maxcontext_pool_size = pool_size
          # NOTE: no init-time preflight. Real docs exist only at rerank() entry;
          # context-fit is validated there (see rerank() below).

      @staticmethod
      def _early_reject_non_qwen3(model_name: str | None) -> None:
          if not model_name:
              return
          lowered = model_name.lower()
          if ("qwen3" in lowered) or ("qwen3.5" in lowered) or ("qwen3_5" in lowered):
              return
          raise ValueError(
              "MaxContextDualEnd supports Qwen3 / Qwen3.5 only; got "
              f"{model_name!r}. (Qwen2 is explicitly not supported.)"
          )

      def _assert_maxcontext_invariants(self, pool_size: int) -> None:
          if self.config.model_type not in MAXCONTEXT_ALLOWED_MODEL_TYPES:
              raise ValueError(
                  f"MaxContextDualEnd requires a Qwen3 / Qwen3.5 model_type; "
                  f"got {self.config.model_type!r}."
              )
          if self.scoring != "generation":
              raise ValueError(
                  "MaxContextDualEnd requires --scoring generation."
              )
          if self.k != pool_size:
              raise ValueError(f"pool_size={pool_size} but ranker.k={self.k}.")
          if self.num_permutation != 1:
              raise ValueError(
                  "MaxContextDualEnd requires --num_permutation 1 "
                  "(compare_both does not permute)."
              )
          if self.method != "selection":
              raise ValueError(
                  "MaxContextDualEnd requires method='selection' "
                  "(_double_ended_selection is the only supported algorithm)."
              )

      def rerank(self, query, docs):
          if len(docs) != self._maxcontext_pool_size:
              raise ValueError(
                  f"MaxContextDualEnd expects exactly pool_size="
                  f"{self._maxcontext_pool_size} input docs; got {len(docs)}."
              )
          # Codex round 1 high: real preflight lives here, using the actual
          # truncated query + actual top-pool_size docs + the actual chat
          # template overhead. No synthetic "representative doc" proxy.
          self._assert_maxcontext_fits(query, docs)
          return super().rerank(query, docs)

      def _assert_maxcontext_fits(self, query, docs) -> None:
          # Render the exact prompt the runtime will issue for the first
          # iteration (whole pool) using the same truncate() + chat-template
          # path. See compute_max_fit_window() in setwise.py.
          ok, rendered_length, limit = compute_max_fit_window(
              ranker=self,
              query=query,
              docs=docs,
              reserved_output_tokens=128,
          )
          if not ok:
              raise ValueError(
                  f"MaxContextDualEnd preflight failed: rendered prompt is "
                  f"{rendered_length} tokens but the budget is {limit} "
                  f"(max_input_tokens - reserved_output_tokens). "
                  "Reduce --passage_length or --k, or pick a Qwen3.5 variant with larger context."
              )
  ```

  This class:
  - Inherits the existing dualend `compare_both()` — which already uses `self.CHARACTERS[...]`, `self._format_passages()`, `_parse_dual_output()`. Because `self.CHARACTERS` is now numeric, the inherited path emits numeric prompts and parses numeric outputs (Pattern 2 at `setwise_extended.py:589-596` already supports `(\d+)` via `_num_to_label()` at `:555-560`, which returns `self.CHARACTERS[idx]` — numeric strings under our override).
  - Does not add new parse patterns. `strict_no_parse_fallback=True` causes the base `_parse_dual_output` + the `compare_both` duplicate-rewrite block to raise on every silent-default path (§ 3.2 strict-mode hardening).
  - Does not override `_double_ended_selection` — the single-group fast-path at `setwise_extended.py:860-889` fires on every iteration because `unsorted_len <= num_child + 1` reduces to `(pool_size - 2k) <= pool_size` for iteration `k`, which is always true. When only one doc remains (`top_idx == bottom_idx`), the while-loop guard at `:854` exits cleanly.
  - `method = "selection"` is set in `__init__` regardless of what CLI passed, and also asserted, so `run.py` can keep its setup minimal.

### 3.3 `run.py`

**Three additions; no existing branch modified.**

- `run.py:356-358` — add `'maxcontext_dualend'` to `--direction` `choices=` tuple.
- Import the new class near the top of `run.py` alongside existing ranker imports:

  ```python
  from llmrankers.setwise_extended import (
      ...,
      MaxContextDualEndSetwiseLlmRanker,
  )
  ```

- `run.py:85-172` — add one elif branch at the end of the setwise dispatch. Cheap invariants (Codex round 1 medium: fail before loading the model):

  ```python
  elif args.setwise.direction == 'maxcontext_dualend':
      # Cheap invariants hoisted BEFORE the ranker constructor so we fail
      # before loading the model weights.
      if args.run.hits != args.setwise.k:
          raise ValueError(
              "maxcontext_dualend requires --hits == --k (pool_size)."
          )
      if args.run.scoring != "generation":
          raise ValueError(
              "maxcontext_dualend requires --scoring generation."
          )
      if args.setwise.num_permutation != 1:
          raise ValueError(
              "maxcontext_dualend requires --num_permutation 1."
          )
      ranker = MaxContextDualEndSetwiseLlmRanker(
          model_name_or_path=args.run.model_name_or_path,
          tokenizer_name_or_path=args.run.tokenizer_name_or_path,
          device=args.run.device,
          cache_dir=args.run.cache_dir,
          num_child=args.setwise.num_child,   # overridden internally by pool_size - 1
          scoring=args.run.scoring,
          method=args.setwise.method,          # overridden internally to 'selection'
          num_permutation=args.setwise.num_permutation,
          k=args.setwise.k,
          pool_size=args.setwise.k,
      )
  ```

### 3.3A `llmrankers/setwise_extended.py` / `run.py` — 2026-04-25 MaxContext single-extreme extension

Add two new siblings without touching `MaxContextDualEndSetwiseLlmRanker`:

- `MaxContextTopDownSetwiseLlmRanker(SetwiseLlmRanker)` — best-only whole-pool selection, strict parse, numeric labels `1..N`, variant-specific preflight via `_assert_maxcontext_topdown_fits`.
- `MaxContextBottomUpSetwiseLlmRanker(BottomUpSetwiseLlmRanker)` — worst-only whole-pool selection, strict parse, numeric labels `1..N`, variant-specific preflight via `_assert_maxcontext_bottomup_fits`.

Both single-extreme variants now use an `n_docs=2 deterministic BM25 endgame`: they expose `total_bm25_bypass`, make `N-2` LLM calls plus 1 BM25 bypass, and leave `MaxContextDualEndSetwiseLlmRanker` byte-identical.

Both new classes share module-level helpers:

- `_setup_maxcontext_numeric_attrs(ranker, pool_size)`
- `_resolve_maxcontext_label_index(ranker, label, window_len, default)`
- `_assert_maxcontext_topdown_fits(ranker, query, docs)`
- `_assert_maxcontext_bottomup_fits(ranker, query, docs)`

`run.py` also adds:

- `MAXCONTEXT_DIRECTIONS = {"maxcontext_dualend", "maxcontext_topdown", "maxcontext_bottomup"}`
- early `--openai_key` rejection before the OpenAI dispatch
- two additive CLI directions: `maxcontext_topdown`, `maxcontext_bottomup`
- two new dispatch branches mirroring the existing `maxcontext_dualend` path with cheap invariants hoisted before model load

  No other elif branch is modified.

  The error messages (Codex round 1 high on CLI legibility) explicitly name the offending flag so an operator can fix their launcher in one line.

### 3.4 `analysis/position_bias.py`

**Additive + default-preserving.**

At the aggregation step (`analysis/position_bias.py:48-59`), after loading entries:

```python
schemes = {entry.get("label_scheme", "letters_a_w") for entry in all_entries}
if len(schemes) > 1:
    raise ValueError(
        f"Mixed label_scheme values in input logs: {schemes}. "
        "Run analysis separately per scheme."
    )
scheme = schemes.pop()
```

Then branch label rendering at `analysis/position_bias.py:88-92`:

```python
label = (
    chr(ord('A') + i)
    if scheme == "letters_a_w"
    else str(i + 1)
)
```

Existing logs (no `label_scheme` field) default to `"letters_a_w"` and render letter labels — identical to today's behaviour.

### 3.5 New launchers (`experiments/`)

Three files. Same positional-arg convention as `run_dualend_selection.sh` for operator muscle-memory, **but with a distinct logs directory** to avoid colliding with the legacy position-bias pipeline.

Critical: existing launchers write logs to `results/analysis/position_bias/$(basename OUTPUT_DIR)` and `experiments/run_phase4_analysis.sh` globs every `*_comparisons.jsonl` under that path. Putting MaxContext logs there would poison the analysis because our mixed-scheme guard in `analysis/position_bias.py` (§ 3.4) refuses mixed-scheme aggregations. **MaxContext writes logs to a separate directory**:

```
LOG_DIR="results/analysis/position_bias_maxcontext/$(basename ${OUTPUT_DIR})"
```

Launchers:

- `experiments/run_maxcontext_dualend.sh` — base launcher, positional args `MODEL DATASET RUN_PATH OUTPUT_DIR DEVICE SCORING POOL_SIZE PASSAGE_LENGTH`. Enforces `--hits=POOL_SIZE --k=POOL_SIZE --query_length=128 --direction maxcontext_dualend --num_permutation 1 --method selection`. Writes `--log_comparisons` under `position_bias_maxcontext/`.
- `experiments/run_maxcontext_dualend_pool_sweep.sh` — wraps the base over `pool_size ∈ {10, 20, 30, 40, 50}`.
- `experiments/run_maxcontext_dualend_pl_sweep.sh` — wraps the base over `pl ∈ {64, 128, 256, 512}` at predeclared `pool_size`.

No modification to existing launcher scripts.

### 3.6 `MAX_CONTEXT_EXPERIMENT_PLAN.md` (deliverable, created after implementation)

At the end of implementation, a new root-level document `/Users/hangli/projects/llm-rankers/MAX_CONTEXT_EXPERIMENT_PLAN.md` is created. Contents:

- Preamble: pointer back to `IDEA_007.md` and this plan; staged-execution reminder; list of phase gates.
- Phase 1 (Unit sanity): one copy-paste `sbatch` command.
- Phase 2 (Study C order-robustness pilot, 12 runs): copy-paste commands.
- Phase 3 (Matched-hits regression check, 1 pair): copy-paste commands.
- Phase 4 (Study A + predeclared baselines, 204 runs): copy-paste commands grouped by model.
- Phase 5 (Study B, 96 runs including control arm): copy-paste commands.
- Not an executable script — just copy-paste for operator use.

## 4. CLI behaviour contract

- `--direction maxcontext_dualend` requires:
  - `--hits` equals `--k` (pool_size).
  - `--scoring generation`.
  - Qwen3 / Qwen3.5 model (assert at construction).
  - `--num_permutation 1` (default; reject if overridden).
- `--num_child` is ignored / overridden internally. Document this in the launcher file header.
- Existing directions retain every current behaviour. Running any existing `experiments/run_*.sh` before the implementation and after the implementation must produce byte-identical outputs.

## 5. Label-scheme safety net

The design overrides `self.CHARACTERS` only on `MaxContextDualEndSetwiseLlmRanker` instances. Because this is an instance attribute, it **never mutates the class attribute** and cannot leak into other rankers. Each `MaxContext...` instance has its own numeric list of length `pool_size`; each existing ranker instance sees the 23-letter class attribute unchanged. Verified by (a) the Python attribute-resolution rule (instance > class), (b) the init order (`super().__init__()` runs first; the assignment happens on `self` afterwards), and (c) the regression gate in § 9.

Edge case: `SetwiseLlmRanker.__init__` path at `setwise.py:70` (T5 target token ID prep) reads `self.CHARACTERS`. That path is guarded by `model_type == 't5'`. MaxContext asserts Qwen-only and does not go through it. If in doubt, the regression gate catches any leak.

## 6. Preflight context-fit usage

Codex round 1 high: the init-time "representative doc repeated pool_size times" proxy from v1 is dropped. Real passage lengths vary and chat-template overhead (`setwise.py:153` path) is only visible at runtime. Preflight now runs **at the start of `rerank()`**, using the real docs the ranker is about to score.

Signature:

```python
def compute_max_fit_window(
    ranker: SetwiseLlmRanker,
    query: str,
    docs: list,
    reserved_output_tokens: int = 128,
) -> tuple[bool, int, int]:
    """Render the full MaxContext prompt through the ranker's actual
    truncate() + _format_passages + chat-template path; tokenize with the
    ranker's tokenizer; return (fits, rendered_length, budget)."""
```

The helper:

- Calls `ranker._format_passages(docs)` so per-passage truncation matches runtime exactly.
- Applies the ranker's chat template with `enable_thinking=False` so the overhead from `setwise.py:128-130` is counted.
- Tokenizes the final string with `ranker.tokenizer.encode(...)` — no arithmetic.
- Returns `(rendered_length <= max_input_tokens - reserved_output_tokens, rendered_length, budget)`.

If the rendered prompt overflows, the ranker raises with a message naming the concrete budget and a remediation path. `strict_no_truncation=True` remains as a second safety net inside `_tokenize_inputs`; either check alone is sufficient.

Note: we deliberately do not run preflight at `__init__` because the actual per-query passage set determines whether the prompt fits; a same-budget smoke run using synthetic repeated text is misleading on queries with unusually long passages.

## 7. Launcher + CLI integration contract

- Every MaxContext launcher calls `python3 run.py run ... setwise --num_child N_DOCS_UNUSED --direction maxcontext_dualend --k POOL_SIZE --method selection --num_permutation 1`.
- `--num_child` value on the CLI is ignored internally; we pass a placeholder (e.g. `3`) to keep the CLI signature identical to `run_dualend_selection.sh` for operator muscle-memory.
- The launcher header (first 5 lines) must document: "This launcher sets hits=k=POOL_SIZE and requires Qwen + generation + num_permutation=1."

## 8. Ordering of concrete edits

Codex-executable order (exact file writes):

1. `llmrankers/setwise.py` — add `strict_no_truncation` class default; extend `_tokenize_inputs`; add `compute_max_fit_window` helper; extend `_log_comparison` to emit `label_scheme` when set.
2. `llmrankers/setwise_extended.py` — add `strict_no_parse_fallback` class default on `DualEndSetwiseLlmRanker`; extend `_parse_dual_output` fallback paths; add `MaxContextDualEndSetwiseLlmRanker` class at the end of file.
3. `run.py` — add `'maxcontext_dualend'` to `choices`; add import; add dispatch elif at end of setwise branch.
4. `analysis/position_bias.py` — add label_scheme read + mixed-scheme guard + numeric label rendering (default path unchanged).
5. `experiments/run_maxcontext_dualend.sh` — new base launcher.
6. `experiments/run_maxcontext_dualend_pool_sweep.sh` — new sweep wrapper.
7. `experiments/run_maxcontext_dualend_pl_sweep.sh` — new sweep wrapper.
8. `/Users/hangli/projects/llm-rankers/MAX_CONTEXT_EXPERIMENT_PLAN.md` — new copy-paste command document.

Each file written independently; any single file revert leaves the remaining files in a coherent state because additions are gated behind default-False flags or behind the new CLI choice.

## 9. Regression gate (mandatory before claiming success)

Before the implementation is considered complete, run the following **regression check** to prove existing pipelines are unaffected.

### 9.1 Golden snapshots (pre-change) — one run per existing committed family

Codex round 2 correction: the v2 golden list cited paths that do not exist in the workspace (`results/selective-dualend/...`), and the "Flan-T5 default = likelihood" claim is wrong — `experiments/run_topdown_heapsort.sh:25` defaults to `--scoring generation`. The revised golden set uses only `.eval` files that actually exist today and makes the scoring-mode coverage explicit.

**Golden set (verified present on disk 2026-04-21):**

- **TopDown family (generation):** `flan-t5-large-dl19/topdown_heapsort.eval`, `qwen3-8b-dl19/topdown_bubblesort.eval`.
- **BottomUp family (generation):** `qwen3-8b-dl19/bottomup_bubblesort.eval`, `qwen3-8b-dl19/bottomup_heapsort.eval`.
- **DualEnd family (generation):** `qwen3-8b-dl19/dualend_bubblesort.eval`, `qwen3-8b-dl19/dualend_selection.eval`.
- **BiDir family (generation):** `qwen3.5-9b-dl20/bidirectional_rrf.eval`, `qwen3.5-9b-dl20/bidirectional_weighted_a0.7.eval`.
- **PermVote (only available for qwen3-8b-dl19):** `qwen3-8b-dl19/permvote_p2_heapsort.eval`.

That is **9 goldens** covering every committed generation-mode ranker family × both T5 and Qwen × both datasets (DL19 + DL20 via the BiDir pair).

**Coverage the goldens do NOT provide, and explicit substitutes:**

- **`--scoring likelihood` path.** No `.eval` is currently committed under a likelihood-only launcher (`experiments/run_likelihood.sh` has not been submitted per `Need_to_Run.txt`). Substitute: the **invariant check script (§ 9.4)** must include a unit-level assertion that the `_score_label_candidates` / likelihood-proxy code path is **not invoked during a MaxContext run** (MaxContext asserts `scoring == 'generation'`), plus a check that instantiating a generic `DualEndSetwiseLlmRanker(scoring='likelihood')` after the changes behaves identically to before (same return from a fixed set of synthetic inputs).
- **Selective DualEnd, Bias-aware DualEnd, Samecall-Regularized.** No `.eval` is committed for these families. Substitute: the invariant script instantiates each class with a mock config and asserts their `compare_both()` / `compare_worst()` paths still route through the **existing** `_parse_dual_output` code (default `strict_no_parse_fallback=False`) and **existing** `_tokenize_inputs` code (default `strict_no_truncation=False`). Because the v2 plan only adds default-False flags that these classes do not opt into, behaviour is guaranteed unchanged.
- **BEIR runs.** Some `results/beir/**/*.eval` exist, but qrels mapping is not in `experiments/eval_all.sh` (only dl19/dl20 are recognized there). The plan does **not** include BEIR runs in the byte-exact gate; changes to BEIR outputs would be caught only by the invariant-script substitute above.

### 9.2 Post-change rerun + `.eval` regeneration

After Codex applies all edits:

1. Copy each existing `.eval` above to `results/_golden/...` (preserve directory structure).
2. Re-execute the exact launcher command that produced the original `.txt` (each `results/{config}/*.log` starts with the command line; `experiments/run_*.sh` also captures it). Write outputs to a scratch directory `results/_regression/${config}/`.
3. **Regenerate `.eval` from `.txt` via pyserini** (Codex round 1 critical: the original gate forgot this step):

   ```bash
   python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 -m ndcg_cut.100 \
          -m map_cut.10 -m map_cut.100 -m recall.1000 \
          <qrels> results/_regression/${config}/${method}.txt \
          > results/_regression/${config}/${method}.eval
   ```

   Qrels mapping: use `experiments/eval_all.sh` which recognises `dl19`/`dl20` directory names at lines 24–30. No BEIR mapping exists there; BEIR is explicitly out of the byte-exact gate (see § 9.1).
4. `diff -q results/_golden/${config}/${method}.eval results/_regression/${config}/${method}.eval` for every golden. **Any non-empty diff blocks merge.** Ignore `.log` (timestamps) and `.txt` (order may be stable but only `.eval` is audited).

### 9.3 MaxContext smoke-run (Phase 1 of IDEA_007)

```bash
python3 run.py \
  run --model_name_or_path Qwen/Qwen3-4B-FP8 \
      --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
      --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path results/_regression/maxcontext_smoke/maxcontext_dualend.txt \
      --device cuda \
      --scoring generation \
      --hits 10 \
      --passage_length 512 \
      --query_length 128 \
  setwise --num_child 9 \
          --method selection \
          --direction maxcontext_dualend \
          --k 10 \
          --num_permutation 1
```

Codex round 1 critical: `--k` and `--num_permutation` belong under `setwise`, not `run`. The corrected command above puts them under `setwise`.

Expected: the run completes; the output is a valid TREC run file with 10 docs per query; zero parse aborts; zero truncations (log-grep `ERROR`, `Traceback`, `Warning: prompt length`).

### 9.4 Mandatory invariant check script

Codex round 1 low: the script is promoted from optional to mandatory, since there is no pytest suite.

`scripts/check_maxcontext_invariants.py` runs without a GPU (uses lightweight mock configs where possible) and asserts:

- `MaxContextDualEndSetwiseLlmRanker` with T5 `model_type` → raises.
- `MaxContextDualEndSetwiseLlmRanker` with `qwen2` `model_type` → raises (explicit allowlist, not superset).
- `scoring='likelihood'` → raises at dispatch and at class init.
- `hits != k` → dispatch raises.
- `num_permutation=2` → raises.
- `method != 'selection'` → class init raises.
- `_parse_dual_output` with an out-of-range numeric output + `strict_no_parse_fallback=True` → raises.
- `_parse_dual_output` with duplicate best/worst numeric output + strict mode → raises.
- `_parse_dual_output` same inputs, strict mode False → returns silent-default (exactly the existing path).
- `_tokenize_inputs` with a synthetic oversize prompt + `strict_no_truncation=True` → raises.
- `_tokenize_inputs` same input, strict mode False → warn-and-clip (existing).
- `test_maxcontext_topdown_invariants()` covers the new best-only whole-pool variant.
- `test_maxcontext_bottomup_invariants()` covers the new worst-only whole-pool variant.
- **Codex round 2 substitutes for uncommitted `.eval` coverage:**
  - Instantiate a plain `DualEndSetwiseLlmRanker(scoring='likelihood', ...)` with a fixed mock config. Call `_parse_dual_output` on a canned input that currently triggers the silent-default at `setwise_extended.py:672-673`. Assert the return is **identical** to the pre-change silent-default (exact tuple).
  - Instantiate `SelectiveDualEndSetwiseLlmRanker`, `BiasAwareDualEndSetwiseLlmRanker`, `SameCallRegularizedSetwiseLlmRanker` with mock configs. Assert that each has `strict_no_truncation=False` and `strict_no_parse_fallback=False` as runtime attributes (no accidental opt-in via inheritance).
  - Assert `_log_comparison` on any existing ranker produces the **same keyset** as before (no `label_scheme` field unless the ranker sets `self.label_scheme`).

Running the script is required as part of the regression gate (a `make check` style one-liner). CI status: zero failures.

## 10. Deliverables

At the end of implementation, the following files exist and pass the regression gate:

- Modified: `llmrankers/setwise.py`, `llmrankers/setwise_extended.py`, `run.py`, `analysis/position_bias.py`.
- New: `experiments/run_maxcontext_dualend.sh`, `experiments/run_maxcontext_dualend_pool_sweep.sh`, `experiments/run_maxcontext_dualend_pl_sweep.sh`.
- New: `/Users/hangli/projects/llm-rankers/MAX_CONTEXT_EXPERIMENT_PLAN.md` (copy-paste command sheet).
- **New (mandatory):** `scripts/check_maxcontext_invariants.py` (§ 9.4).
- No modifications to any other ranker class, launcher, analysis script, or result file.

`analysis/position_bias.py`'s MaxContext handling is a **compatibility shim** (it aggregates over the shrinking windows into a single `n_positions = max(...)` bin — Codex round 1 medium). The shim is acceptable because position-bias analysis at w=50 is in-scope for a separate future work item per IDEA_007 § 6, not this implementation. The plan explicitly does **not** claim MaxContext's position-bias analysis is paper-ready.

## 11. Out of scope (match IDEA_007 § 10 verbatim)

- Flan-T5 for this direction.
- `--scoring likelihood` for this direction.
- `pool_size > 50`.
- Batched / multi-pass fitting.
- Letter alphabets beyond A-W.
- `bias_aware` / `samecall_regularized` derivatives of MaxContext.
- Auto-tuning `pool_size`.
- Any refactor or change to existing ranker classes, launchers, or analysis scripts.

## 12. Audit trail

- **v1**: extend-not-refactor strategy; instance-level `self.CHARACTERS` override; default-False feature flags; additive `label_scheme` field; backward-compat guards verified against the 15+ anchor points from the anchor exploration.
- **v2** (post Codex round 1): closed silent bad-parse paths across Patterns 1–6 and the `compare_both()` duplicate-rewrite block under strict mode; moved preflight out of init into `rerank()` using actual docs + chat-template overhead; added explicit Qwen3 allowlist (`qwen2` excluded); enforced `method == 'selection'` inside the class; moved cheap invariants to `run.py` before the model loads; expanded regression gate to 9+ goldens covering every existing family and both scoring modes; made the invariant script mandatory; separated MaxContext log directory from legacy position-bias logs to avoid the mixed-scheme guard breaking `experiments/run_phase4_analysis.sh`; fixed the smoke command's `--k` / `--num_permutation` sub-parser placement and added explicit `.txt → .eval` regeneration step.
- **v3** (this doc; post Codex round 2): unified `compute_max_fit_window` to one canonical signature `(ranker, query, docs, reserved_output_tokens) -> (fits, rendered_length, budget)` and removed the older tokenizer-only/int-returning variant from § 3.1; corrected § 9.1 to cite only `.eval` files that actually exist on disk (9 goldens, generation-mode only); removed the false "Flan-T5 topdown covers likelihood" claim; scoped qrels mapping in § 9.2 to dl19/dl20 only and explicitly removed BEIR from the byte-exact gate; added substitute invariant-script coverage for Selective/Bias-aware/Samecall DualEnd and for the likelihood-scoring path (§ 9.4) so the prime constraint holds even for families without committed goldens.
