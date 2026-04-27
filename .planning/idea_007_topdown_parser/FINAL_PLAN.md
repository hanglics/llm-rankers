# Draft Fix Plan v3 — MaxContext `'0'` Out-of-Range Refusal Handling

**Status:** Draft v3 for Codex audit round 3 (Phase 3)
**Source:** Codex investigation report `INVESTIGATION_REPORT.md` + audit rounds 1-2 (gpt-5.5 + xhigh, threadId `019dccd4-2840-70a1-9302-16dc6cb6d186`)
**Author:** Hang Li (orchestrated)
**Date:** 2026-04-27

> **Changelog from v2:**
> - **BLOCKER fix:** initialize `parse_status = "parsed"` and `parse_fallback_reason = None` at the TOP of `compare()`, `compare_worst()`, `compare_both()` — before any scoring-branch dispatch. Prevents `UnboundLocalError` in T5 / likelihood branches (which don't mutate the locals).
> - **HIGH fix:** `MaxContextDualEndSetwiseLlmRanker.rerank` revision shown as INSERTION into the existing method body — preserves the existing length check (`len(docs) != self._maxcontext_pool_size`) and `_assert_maxcontext_fits(query, docs)` call from `setwise_extended.py:1195-1202`.
> - **MEDIUM fix:** DualEnd test layering split into three layers — direct `_classify_numeric_noop` tests, `_parse_dual_output` strict-raise tests (unchanged), and `compare_both` wrapper tests for the soft-fold pair.
> - **LOW fix (1):** numeric out-of-range call-site test loops over ALL 6 inputs (not `[:5]`); counter assertions match `len(raw_inputs)`.
> - **LOW fix (2):** letter-scheme `-1` regression assertion made explicit — preserves current `_parse_single_label('-1', letter_valid) == 'A'` behavior (the bare-integer guard is gated on `numeric_1_based` scheme only, so non-numeric paths are unchanged).
>
> **Changelog from v1 (still applies):**
> - **BLOCKER:** added per-query reset of new fallback counters in `MaxContextDualEndSetwiseLlmRanker.rerank`.
> - **HIGH:** dropped `_last_parse_status` instance attribute; locals passed inline to `_log_comparison`.
> - **HIGH:** test fixtures live in `scripts/check_maxcontext_invariants.py`, not `tests/`.
> - **HIGH:** `_classify_numeric_noop` defensively returns `None` when `label_scheme != "numeric_1_based"`.
> - **MEDIUM:** DualEnd path keeps `raw_output` for diagnostics; uses `cleaned_output` only for parse/classify.
> - **MEDIUM:** DualEnd lexical refusal requires whole-output cleanliness.
> - **MEDIUM:** softened §5 repro pass criteria.
> - **MEDIUM:** updated `test_maxcontext_dualend_byte_identity_snapshot` expected dict.
> - **LOW:** XML-wrapper case explicitly out of scope.
> - **LOW:** corrected `<|endoftext|>` cleaning note.
> - **NIT:** counter init ordering called out.
> - **Over-engineering trim:** Flan-T5 byte-identical rerun is optional, not a gate.

---

## 1. Decision

**Option (e): narrow (a) + (d).** Soft-fold numeric-only out-of-range outputs into the existing refusal pathway, plus a prompt-hardening clause. Concretely:

- Recognize a third class of "no-op refusal" beyond the existing lexical refusal: a cleaned model output that *is* a single signed integer outside `[1, N]`. `0`, `0<|im_end|>`, `0\n`, `  0  `, `-1`, `51` (when `N < 51`) all qualify.
- The fallback policy stays identical to the lexical-refusal branch from commit `7307e9b`: deterministic head-wins (TopDown), tail-stays (BottomUp), position-preserving pair (DualEnd). Increment `total_parse_fallback` and a new reason-specific counter. Log the event in JSONL with a structured `parse_status` field.
- Strengthen the prompt footer with an explicit "Do not output 0" line. Reduces but does not eliminate the failure mode; the parser fix is the load-bearing change.
- **Genuinely garbage outputs (`asdf`, empty, `<|endoftext|>` only, structured out-of-range like `"Passage 51 is most relevant"`) still abort under strict mode.** Preserves the spec's distinction between "refusal" and "corrupted" and protects against silently masking a real parser bug. Note: the existing test at `scripts/check_maxcontext_invariants.py:985-994` already enforces this for structured-but-out-of-window inputs.

**Why not the alternatives:** unchanged from v1.

## 2. Code Changes

### 2.1 New classifier helper — `llmrankers/setwise.py` after line 459

```python
# New class-level constant near NUMERIC_REFUSAL_REGEX
_NUMERIC_ONLY_REGEX = re.compile(r"^\s*(-?\d+)\s*$")

def _classify_numeric_noop(self, raw: str, n_docs: int) -> Optional[str]:
    """
    Recognize a numeric-scheme no-op event.

    Returns 'lexical_refusal' / 'numeric_out_of_range' / None.

    None means the output is not a recognized refusal; caller applies
    strict policy (raise) or lenient fallback per existing rules.

    Defensive: returns None if label_scheme != "numeric_1_based" so
    accidental non-MaxContext callers cannot trigger soft-fold behavior.
    """
    if getattr(self, "label_scheme", None) != "numeric_1_based":
        return None
    cleaned = self._clean_generation_output(raw)
    if re.search(self.NUMERIC_REFUSAL_REGEX, cleaned, flags=re.IGNORECASE):
        return "lexical_refusal"
    m = self._NUMERIC_ONLY_REGEX.match(cleaned)
    if m:
        try:
            value = int(m.group(1))
        except ValueError:
            return None
        if value < 1 or value > n_docs:
            return "numeric_out_of_range"
    return None
```

`_is_numeric_refusal_output` (`setwise.py:457-459`) is **kept** as a wrapper for backward compat (its callers in `_parse_single_label` are minimized — see §2.2). Per Codex audit answer 1.

### 2.2 Guard signed numeric in `_parse_single_label` — `llmrankers/setwise.py:529-534`

The current loose fallback parses `'-1'` as `'1'` because `\b(\d+)\b` matches the trailing `1`. The fix is **scheme-gated** to preserve letter-scheme legacy behavior (Codex round-2 LOW-2):

```python
# Only the numeric scheme guards signed-integer whole-output:
# - Numeric scheme: '-1' / '51' / '0' return None so the caller's
#   classifier can decide (fixes the original bug).
# - Letter/bigram schemes: behavior unchanged ('-1' continues to fall
#   through to the loose \b(\d+)\b fallback, parsing as '1' → label 'A').
if is_numeric_scheme and self._NUMERIC_ONLY_REGEX.match(cleaned):
    return None  # caller will check classify_numeric_noop

num_match = re.search(r"\b(\d+)\b", cleaned)
if num_match:
    idx = int(num_match.group(1)) - 1
    if 0 <= idx < len(valid_chars):
        return valid_chars[idx]
return None
```

Numeric scheme: structured outputs like `"The answer is 25"` still parse to `'25'`. Bare `'-1'`, `'51'` (when N < 51), `'0'` now return `None` for the classifier to decide. Letter scheme: byte-identical to current behavior.

### 2.3 Replace TopDown strict branch — `llmrankers/setwise.py:663-690`

**Critical (Codex round-2 BLOCKER):** initialize `parse_status` and `parse_fallback_reason` at the **top** of `compare()` — before any `if self.scoring == ...` branch. The T5 generation and likelihood branches do not mutate these locals; they must already be defined when `_log_comparison` fires at line 690 regardless of which scoring branch ran.

```python
def compare(self, query, docs):
    # ... existing setup ...
    parse_status = "parsed"
    parse_fallback_reason = None

    if self.scoring == 'generation':
        # ... existing T5 batched-generation branch (no changes; parse_status
        # remains "parsed" by default) ...

        elif self._is_supported_causal_model():
            # ... existing Qwen single-call decode + parse ...
            output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
            if output is None:
    is_numeric = getattr(self, "label_scheme", None) == "numeric_1_based"
    reason = self._classify_numeric_noop(raw_output, len(docs)) if is_numeric else None
    if reason is not None:
        # Deterministic no-op: head wins (no swap in TopDown).
        self.total_parse_fallback = getattr(self, "total_parse_fallback", 0) + 1
        counter_name = f"total_{reason}_fallback"
        setattr(self, counter_name, getattr(self, counter_name, 0) + 1)
        if _DEBUG or getattr(self, "strict_no_parse_fallback", False):
            print(f"[MaxContext] {reason} no-op (best=1). Raw: {raw_output!r}")
        output = self.CHARACTERS[0]
        parse_status = f"{reason}_noop"
        parse_fallback_reason = reason
    elif getattr(self, "strict_no_parse_fallback", False):
        raise ValueError(
            f"MaxContext single-label parse failed. Raw text: {raw_output!r}"
        )
    else:
        output = self._clean_generation_output(raw_output).upper()
        parse_status = "lenient_fallback"
```

Then at line 690 (the existing `_log_comparison("best", ...)` site), pass the locals:

```python
if output in self.CHARACTERS[:len(docs)]:
    self._log_comparison(
        "best", self.CHARACTERS[:len(docs)], output, docs,
        parse_status=parse_status,
        parse_fallback_reason=parse_fallback_reason,
    )
else:
    print(f"Unexpected output: {output}")
```

**Important:** `parse_status` and `parse_fallback_reason` are **local variables**, not `self._last_parse_status`. Per Codex audit finding HIGH-3.

### 2.4 Mirror in BottomUp — `llmrankers/setwise_extended.py:208-237`

Identical structural change in `BottomUpSetwiseLlmRanker.compare_worst`:

- **Top-level locals:** `parse_status = "parsed"`, `parse_fallback_reason = None` defined BEFORE the `if self.scoring == ...` branch (Codex round-2 BLOCKER).
- Default fallback selects `CHARACTERS[len(docs) - 1]` (tail-stays).
- Pass to `_log_comparison("worst", ..., parse_status=..., parse_fallback_reason=...)`.

### 2.5 DualEnd extension — `llmrankers/setwise_extended.py:540-595`

**Top-level locals (Codex round-2 BLOCKER):** initialize `parse_status = "parsed"` and `parse_fallback_reason = None` at the top of `compare_both()` — before any scoring-branch dispatch. The T5 batched generation branch and the likelihood branch do not mutate these locals; they must be defined before `_log_comparison` fires at lines 590-591 regardless of which scoring branch ran.

Wrap the `_parse_dual_output` call with classifier + try/except:

```python
def compare_both(self, query, docs):
    # ... existing setup ...
    parse_status = "parsed"
    parse_fallback_reason = None
    is_numeric = getattr(self, "label_scheme", None) == "numeric_1_based"

    if self.scoring == 'generation':
        # ... existing T5 batched-generation branch unchanged ...

        elif self._is_supported_causal_model():
            # ... existing Qwen decode ...
            raw_output = self.tokenizer.decode(
                output_ids[inputs.input_ids.shape[1]:],
                skip_special_tokens=False,
            ).strip()
            cleaned_output = self._clean_generation_output(raw_output)

            try:
                best, worst = self._parse_dual_output(cleaned_output, len(docs))
            except ValueError:
    if not is_numeric:
        raise
    # Whole-output classifier: only soft-fold when the entire cleaned
    # output is a recognized no-op AND has no embedded structured
    # selector tokens (BEST / WORST / PASSAGE <num>). Structured-but-
    # malformed outputs like "Best: 0, Worst: 7" still abort.
    has_structured = bool(re.search(
        r"\b(BEST|WORST|PASSAGE\s*\d+)\b",
        cleaned_output, flags=re.IGNORECASE,
    ))
    reason = self._classify_numeric_noop(raw_output, len(docs))
    if reason is None or has_structured:
        raise  # truly corrupted output; preserve abort policy
    self.total_parse_fallback = getattr(self, "total_parse_fallback", 0) + 1
    counter_name = f"total_{reason}_fallback"
    setattr(self, counter_name, getattr(self, counter_name, 0) + 1)
    print(f"[MaxContext] dual {reason} no-op "
          f"(best=1, worst={len(docs)}). Raw: {raw_output!r}")
    best = self.CHARACTERS[0]
    worst = self.CHARACTERS[len(docs) - 1]
    parse_status = f"{reason}_noop"
    parse_fallback_reason = reason
```

**Critical diagnostics fix (per Codex MEDIUM-1):** keep `raw_output` for logging; use `cleaned_output` only for parsing/classification. The `print` and the eventual JSONL log show the *uncleaned* raw text so debugging captures the actual model output (e.g., `'0<|im_end|>'`, not `'0'`).

The `if best == worst` strict check at line 581-583 is unaffected (would only matter if `len(docs) == 1`, forbidden by surrounding logic).

Update the `_log_comparison` calls at lines 590-591 to pass `parse_status` and `parse_fallback_reason`:

```python
self._log_comparison(
    "dual_best", self.CHARACTERS[:len(docs)], best, docs,
    parse_status=parse_status,
    parse_fallback_reason=parse_fallback_reason,
)
self._log_comparison(
    "dual_worst", self.CHARACTERS[:len(docs)], worst, docs,
    parse_status=parse_status,
    parse_fallback_reason=parse_fallback_reason,
)
```

Both rows get the same status (the soft-fold pairs them).

### 2.6 Wire `parse_status` into `_log_comparison` — `llmrankers/setwise.py:170-188`

```python
def _log_comparison(self, comp_type: str, positions: list, selected: str,
                    docs: list = None,
                    parse_status: str = None,
                    parse_fallback_reason: str = None):
    """Log a comparison for position bias analysis."""
    log_path = getattr(self, '_comparison_log_path', None)
    if not log_path:
        return
    import json as _json
    entry = {
        "qid": getattr(self, '_current_qid', None),
        "type": comp_type,
        "positions": positions,
        "selected": selected,
    }
    if docs is not None:
        entry["docids"] = [d.docid for d in docs]
    label_scheme = getattr(self, "label_scheme", None)
    if label_scheme:
        entry["label_scheme"] = label_scheme
    if parse_status is not None:
        entry["parse_status"] = parse_status
    if parse_fallback_reason is not None:
        entry["parse_fallback_reason"] = parse_fallback_reason
    with open(log_path, 'a') as f:
        f.write(_json.dumps(entry) + "\n")
```

Per Codex audit answer 5, `analysis/position_bias.py` ignores unknown keys and existing files lacking `parse_status` are tolerated (it reads only `label_scheme`, `type`, `positions`, `selected`). `analysis/parse_success_rate.sh` greps log text, not JSONL fields.

### 2.7 Initialize reason-specific counters — `llmrankers/setwise_extended.py:25-33`

```python
def _setup_maxcontext_numeric_attrs(ranker, pool_size: int) -> None:
    ranker.CHARACTERS = [str(i + 1) for i in range(pool_size)]
    ranker.num_child = pool_size - 1
    ranker.method = "selection"
    ranker.strict_no_truncation = True
    ranker.strict_no_parse_fallback = True
    ranker.total_parse_fallback = 0
    ranker.total_lexical_refusal_fallback = 0
    ranker.total_numeric_out_of_range_fallback = 0
    ranker.label_scheme = "numeric_1_based"
    ranker._maxcontext_pool_size = pool_size
```

Per Codex NIT, this MUST execute in the constructor before any `optional_stat_totals` reads them in `run.py`. Constructor ordering already guarantees this (the helper is called from `__init__` before `rerank` is invoked).

Reset the three counters at:

- `MaxContextTopDownSetwiseLlmRanker._maxcontext_topdown_select` at `setwise_extended.py:1789-1794` (currently resets `total_parse_fallback`; add the two reason-specific counters).
- `MaxContextBottomUpSetwiseLlmRanker._maxcontext_bottomup_select` at `setwise_extended.py:1888-1893`.
- **`MaxContextDualEndSetwiseLlmRanker.rerank` per-query reset (BLOCKER from round 1; INSERTION fix from round 2).** The current `rerank` at `setwise_extended.py:1195-1202` is:

  ```python
  def rerank(self, query: str, docs: List[SearchResult]) -> List[SearchResult]:
      if len(docs) != self._maxcontext_pool_size:
          raise ValueError(
              f"MaxContextDualEnd expects exactly pool_size="
              f"{self._maxcontext_pool_size} input docs; got {len(docs)}."
          )
      self._assert_maxcontext_fits(query, docs)
      return super().rerank(query, docs)
  ```

  The fix is an **insertion** — preserve the length check and the `_assert_maxcontext_fits` call. Add the three counter resets immediately before `super().rerank`:

  ```python
  def rerank(self, query: str, docs: List[SearchResult]) -> List[SearchResult]:
      if len(docs) != self._maxcontext_pool_size:
          raise ValueError(
              f"MaxContextDualEnd expects exactly pool_size="
              f"{self._maxcontext_pool_size} input docs; got {len(docs)}."
          )
      self._assert_maxcontext_fits(query, docs)
      # Reset per-query fallback counters (run.py reads these after each rerank)
      self.total_parse_fallback = 0
      self.total_lexical_refusal_fallback = 0
      self.total_numeric_out_of_range_fallback = 0
      return super().rerank(query, docs)
  ```

**Refactor `MaxContextDualEndSetwiseLlmRanker.__init__` at `setwise_extended.py:1147-1160`** to call `_setup_maxcontext_numeric_attrs(self, pool_size)` instead of duplicating the body. Eliminates drift; the duplication is exactly why DualEnd was missing `total_parse_fallback = 0` before commit `7307e9b`.

### 2.8 Aggregate counters in `run.py` — `run.py:377-384`

Add `total_lexical_refusal_fallback` and `total_numeric_out_of_range_fallback` to whatever stat-aggregation list emits the existing `total_parse_fallback`. Codex's NIT: counters MUST be present as ranker attributes when `optional_stat_totals` (or equivalent) is built — §2.7 guarantees this via constructor init.

### 2.9 Belt-and-suspenders prompt change — `llmrankers/setwise.py:207-212` and `222-227`

For numeric scheme only:

```python
# In _build_best_prompt, replace the existing numeric footer block with:
if getattr(self, "label_scheme", None) == "numeric_1_based":
    prompt += (
        f"\n\nReply with exactly one passage number from 1 to {len(docs)}. "
        "Do not explain. Do not output 0 or any number outside 1 to "
        f"{len(docs)}. If none of the passages are clearly relevant, "
        "still pick the single closest one."
    )

# In _build_worst_prompt, mirror with "irrelevant" / "least relevant" wording.
```

The clause is gated to `label_scheme == "numeric_1_based"`, which is set only by `_setup_maxcontext_numeric_attrs` and the (about-to-be-refactored) DualEnd MaxContext init. No non-MaxContext path sets this scheme; per Codex audit answer 4, the gating is correct.

## 3. Tests — extend `scripts/check_maxcontext_invariants.py`

**No new test files.** Codex audit HIGH-2: there is no `tests/` directory in the repo; the existing test layout is plain-assert functions in `scripts/check_maxcontext_invariants.py` (`wc -l` = 1234 lines). Extend that file.

### 3.1 New fixtures in `test_maxcontext_numeric_parse_fallback` (around line 921)

Add to the existing fixtures list:

| Input (raw) | Pool size N | Expected `_parse_single_label` |
|---|---|---|
| `'0'` | 50 | `None` (now blocked by signed-numeric guard at §2.2) |
| `'0<|im_end|>'` | 50 | `None` |
| `'-1'` | 50 | `None` (regression test for the `-1 → '1'` latent bug) |
| `'51'` | 50 | `None` |
| `'0\n'` | 50 | `None` |
| `'  0  '` | 50 | `None` |

### 3.2 New fixtures in `test_maxcontext_compare_refusal_noop` (around line 974) — Codex round-2 LOW-1

Loop over **all** raw inputs (not `[:5]`); assert counters equal `len(raw_inputs)`:

```python
def test_maxcontext_compare_numeric_out_of_range_noop():
    raw_inputs = [
        "0", "0<|im_end|>", "-1", "51", "0\n", "  0  ",
    ]
    # TopDown: each compare() call should soft-fold to "1"
    td_ranker = build_maxcontext_numeric_compare_stub(raw_inputs, pool_size=50)
    for _ in raw_inputs:
        assert td_ranker.compare("query", make_docs(50)) == "1"
    assert td_ranker.total_parse_fallback == len(raw_inputs)
    assert td_ranker.total_numeric_out_of_range_fallback == len(raw_inputs)
    assert td_ranker.total_lexical_refusal_fallback == 0

    # BottomUp: each compare_worst() call should soft-fold to "50"
    bu_ranker = build_maxcontext_numeric_compare_stub(raw_inputs, pool_size=50)
    for _ in raw_inputs:
        assert bu_ranker.compare_worst("query", make_docs(50)) == "50"
    assert bu_ranker.total_parse_fallback == len(raw_inputs)
    assert bu_ranker.total_numeric_out_of_range_fallback == len(raw_inputs)
    assert bu_ranker.total_lexical_refusal_fallback == 0
```

The implementation will need to thread `parse_status` / `parse_fallback_reason` into the test stub at the same call sites. Codex Phase 4 will write the actual code.

### 3.3 DualEnd fixtures — three layers (Codex round-2 MEDIUM)

The soft-fold behavior lives in `compare_both()`'s try/except wrapper, NOT in `_parse_dual_output` itself. Tests must respect that layering:

**Layer 1 — direct classifier tests (`_classify_numeric_noop`):**

```python
def test_classify_numeric_noop_unit():
    ranker = build_single_label_parser_stub(
        strict=True, label_scheme="numeric_1_based",
        characters=[str(i) for i in range(1, 51)],
    )
    # Numeric out-of-range
    assert ranker._classify_numeric_noop("0", 50) == "numeric_out_of_range"
    assert ranker._classify_numeric_noop("0<|im_end|>", 50) == "numeric_out_of_range"
    assert ranker._classify_numeric_noop("-1", 50) == "numeric_out_of_range"
    assert ranker._classify_numeric_noop("51", 50) == "numeric_out_of_range"
    # Lexical refusal
    assert ranker._classify_numeric_noop("None of the passages are relevant.", 50) == "lexical_refusal"
    assert ranker._classify_numeric_noop("I cannot determine.", 50) == "lexical_refusal"
    # Valid integer (in range) → not a no-op
    assert ranker._classify_numeric_noop("25", 50) is None
    # Structured outputs → not a bare-integer no-op
    assert ranker._classify_numeric_noop("Passage 51 is most relevant", 50) is None
    assert ranker._classify_numeric_noop("The answer is 0.", 50) is None  # not whole-output bare integer
    # Defensive: non-numeric scheme returns None
    legacy_ranker = build_single_label_parser_stub(
        characters=[chr(ord("A") + i) for i in range(23)],
    )
    assert legacy_ranker._classify_numeric_noop("0", 50) is None
```

**Layer 2 — `_parse_dual_output` strict-raise behavior (UNCHANGED):**

The parser itself still raises on malformed input under strict mode — the soft-fold belongs in the wrapper. This must remain true:

```python
def test_parse_dual_output_strict_unchanged():
    numeric_ranker = build_dualend_stub(
        strict=True, label_scheme="numeric_1_based",
        characters=[str(i) for i in range(1, 11)],
    )
    # Bare integer that doesn't conform to "Best: X, Worst: Y" structure
    expect_raises(lambda: numeric_ranker._parse_dual_output("0", 10), ValueError)
    # Structured but invalid label
    expect_raises(lambda: numeric_ranker._parse_dual_output("Best: 0, Worst: 7", 10), ValueError)
    # Already covered at line 880 of the test file
    assert numeric_ranker._try_parse_dual_output("Best: 17, Worst: 42", 10) is None
```

**Layer 3 — `compare_both` wrapper soft-fold (NEW BEHAVIOR):**

```python
def test_maxcontext_dualend_compare_both_noop():
    # Bare integer → soft-fold to (best='1', worst='50')
    raw_inputs = ["0", "0<|im_end|>", "-1", "51", "0\n", "  0  "]
    de_ranker = build_maxcontext_dualend_compare_stub(raw_inputs, pool_size=50)
    for _ in raw_inputs:
        best, worst = de_ranker.compare_both("query", make_docs(50))
        assert best == "1"
        assert worst == "50"
    assert de_ranker.total_parse_fallback == len(raw_inputs)
    assert de_ranker.total_numeric_out_of_range_fallback == len(raw_inputs)

    # Lexical refusal whole-output → soft-fold
    lex_ranker = build_maxcontext_dualend_compare_stub(
        ["None of the passages are relevant."], pool_size=50,
    )
    best, worst = lex_ranker.compare_both("query", make_docs(50))
    assert best == "1"
    assert worst == "50"
    assert lex_ranker.total_lexical_refusal_fallback == 1

    # Structured-but-malformed → still raises (NOT soft-folded)
    err_ranker = build_maxcontext_dualend_compare_stub(
        ["Best: 0, Worst: 7"], pool_size=10,
    )
    expect_raises(
        lambda: err_ranker.compare_both("query", make_docs(10)),
        ValueError,
    )
    # Lexical refusal phrase embedded in structured output → still raises
    embedded_ranker = build_maxcontext_dualend_compare_stub(
        ["Best: 3, Worst: none of the others"], pool_size=10,
    )
    # Either succeeds (parse "Best: 3" / "Worst: none") or raises;
    # it must NOT silently soft-fold to (1, 10).
    # Codex Phase 4 verifies which behavior emerges from the parser.
```

Test stub builder `build_maxcontext_dualend_compare_stub` is new infrastructure — Codex Phase 4 implements it analogously to the existing `build_maxcontext_numeric_compare_stub` at the existing fixture site.

### 3.4 Update `test_maxcontext_dualend_byte_identity_snapshot` at line 428-465

Per Codex MEDIUM-4, the refactor adds three new counter attributes. Update `expected_snapshot`:

```python
expected_snapshot = {
    # ... all existing keys ...
    "total_parse_fallback": 0,
    "total_lexical_refusal_fallback": 0,
    "total_numeric_out_of_range_fallback": 0,
}
```

### 3.5 Letter-scheme regression in `test_single_label_parser_hardening` (around line 889) — Codex round-2 LOW-2

Add an explicit assertion that the new signed-numeric guard does NOT alter letter-scheme behavior. The signed-numeric guard at §2.2 must remain **gated on `is_numeric_scheme`**; letter and bigram schemes continue through the legacy `\b(\d+)\b` numeric fallback exactly as before. For letter-scheme rankers, the caller does NOT classify and falls back to `_clean_generation_output(raw).upper()` (lenient mode) or raises (strict mode).

The intended current letter-scheme behavior for `-1` is `'A'` (because the legacy fallback `\b(\d+)\b` matches the `1` after the minus). Decision: **byte-identical preservation**:

```python
# Existing assertion at scripts/check_maxcontext_invariants.py:917-918:
# assert legacy_letter_ranker._parse_single_label("10", letter_valid) == "J"
# assert legacy_letter_ranker._parse_single_label("A", ["A", "B", "C"]) == "A"
#
# Add:
assert legacy_letter_ranker._parse_single_label("-1", letter_valid) == "A"
```

If preserving this is undesirable (the legacy `-1 → 'A'` is a latent letter-scheme bug too), we'd need a separate decision and broader fix scope. For this fix, keep behavior byte-identical. Codex Phase 4 must NOT introduce the signed-numeric guard for non-numeric schemes.

**Implementation note for §2.2:** the signed-numeric guard at line 529 in v3 must be gated:

```python
# Only gate the signed-integer guard for numeric scheme. For letter/bigram
# schemes, the fallback parses '-1' as '1' (which then maps to label A),
# preserving byte-identical legacy behavior.
if is_numeric_scheme and self._NUMERIC_ONLY_REGEX.match(cleaned):
    return None
```

Updated §2.2 sketch (replaces v2's unconditional guard):

```python
if is_numeric_scheme and self._NUMERIC_ONLY_REGEX.match(cleaned):
    return None  # numeric scheme only; caller will check classify_numeric_noop

num_match = re.search(r"\b(\d+)\b", cleaned)
if num_match:
    idx = int(num_match.group(1)) - 1
    if 0 <= idx < len(valid_chars):
        return valid_chars[idx]
return None
```

## 4. Phase-1 Sanity Rerun

Per `IDEA_007.md` §8 Phase 1, after the code lands run on the SLURM cluster:

```
Qwen3-4B, DL19, pool=10, pl=512, direction=maxcontext_topdown
```

Pass criteria:
- Algorithm terminates
- `total_parse_fallback` ≤ 1% of comparisons (flag ≥ 5%; hard-stop ≥ 20% per Codex investigation §7)
- Ranking is a permutation of input
- Preflight passes (zero truncations)
- JSONL log includes `parse_status` field on every entry; `parse_fallback_reason` on fallback entries only

## 5. Targeted Reproduction (revised pass criteria per Codex MEDIUM-3)

Submit the previously-failing 3 SLURM jobs (Qwen3 TopDown, DL19, pool ∈ {30, 40, 50}, pl=512). Each must:

1. **Complete without raising `ValueError`.**
2. **If the model still emits `0`,** that comparison is logged as `parse_status: numeric_out_of_range_noop` with `parse_fallback_reason: numeric_out_of_range`.
3. **If the model no longer emits `0`** (because the prompt-hardening clause changed behavior), that's also a pass — the prompt change worked.
4. **Outputs are valid permutations** of the input pool.
5. **Metrics recorded** (NDCG@10, total_compare, parse counters).

NDCG comparisons against BottomUp at the same pool size are reported separately, not used as a pass/fail gate. Parser-fix smoke tests should not double as quality gates.

## 6. Open Questions Resolved (from v1)

All six open questions answered by Codex audit round 1:

1. **`_is_numeric_refusal_output`** — keep as wrapper for backward compat. Do not delete.
2. **`_last_parse_status`** — do NOT use; pass locals to `_log_comparison`. Already incorporated above.
3. **Test location** — `scripts/check_maxcontext_invariants.py`, not `tests/`. Already incorporated above.
4. **Prompt gating** — currently correct (numeric-scheme-only); fix preserves this.
5. **JSONL backward compat** — `analysis/position_bias.py` and `analysis/parse_success_rate.sh` both safe.
6. **`_classify_numeric_noop` defensive guard** — added `if label_scheme != "numeric_1_based": return None` at top of helper, AND keep call-site `is_numeric` guard. Belt-and-suspenders.

## 7. Verification (post-implementation)

End-to-end, in order:

1. **Unit-level.** `python scripts/check_maxcontext_invariants.py`. All assertions pass, including the new fixtures from §3.
2. **Latent bug check.** Specific assertion that `_parse_single_label('-1', valid_chars)` returns `None` (was returning `'1'` before §2.2).
3. **Repro.** The 3 originally-failing jobs (Qwen3 TopDown, DL19, pool ∈ {30, 40, 50}, pl=512) complete per §5 criteria.
4. **Phase-1 sanity.** As specified in §4.
5. **Promotion gate.** Once 1-4 pass, idea:007 Phase 4 (Study A pool sweep + predeclared baselines, 204 runs) is unblocked.
6. **Optional confidence check (NOT a gate, per Codex over-engineering trim):** re-run one TD-Heap baseline (Flan-T5-large, DL19, hits=100) and diff JSONL for byte-identity. Skip if time is tight.

## 8. Files to Modify (summary)

| Path | Sections |
|---|---|
| `llmrankers/setwise.py` | new `_NUMERIC_ONLY_REGEX` constant + `_classify_numeric_noop` after line 459; signed-numeric guard at 529-534; replacement at 663-690 (TopDown strict branch + log call); extend `_log_comparison` at 170-188; update prompt footers at 207-212 and 222-227 |
| `llmrankers/setwise_extended.py` | extend `_setup_maxcontext_numeric_attrs` at 25-33 (3 new counters); mirror at 208-237 (BottomUp `compare_worst`); wrap `_parse_dual_output` call at 540-595 (`compare_both`) with `raw_output` preservation; refactor `MaxContextDualEndSetwiseLlmRanker.__init__` at 1147-1160 to call `_setup_maxcontext_numeric_attrs`; **add `MaxContextDualEndSetwiseLlmRanker.rerank` per-query reset (BLOCKER fix)**; reset reason-specific counters at 1789-1794 and 1888-1893 |
| `run.py` | aggregate two new counters around 377-384 |
| `scripts/check_maxcontext_invariants.py` | extend `test_maxcontext_numeric_parse_fallback` (~921), `test_maxcontext_compare_refusal_noop` (~974), and `test_maxcontext_dualend_byte_identity_snapshot` (428-465); add `test_maxcontext_dualend_numeric_out_of_range_noop`; add letter-scheme regression assertion in `test_single_label_parser_hardening` (~889) |

## 9. Out of Scope

- Repair-prompt retry path. Defer until fallback rates exceed 5% threshold per Codex investigation.
- Prefix-allowed-tokens generation. Already reverted; needs separate Qwen3-specific work.
- XML-wrapper out-of-range handling (`<answer>0</answer>`). Per Codex LOW-1, explicitly out of scope; bare-integer only.
- Any change to letter-scheme parsing or non-MaxContext rankers.
- Any change to `position_bias.py` or other `analysis/*.py` consumers.
- Any change to existing JSONL files in `results/`. Forward-only fix.
- Pytest framework introduction. Tests stay in `scripts/check_maxcontext_invariants.py` per Codex HIGH-2.
