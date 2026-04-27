# Codex Investigation Brief — MaxContext TopDown `'0'` Parse Failure

**Repo:** `/Users/hangli/projects/llm-rankers/`
**Branch:** `main` @ `7307e9b` (improve max context parser)
**Spec:** `IDEA_007.md` §3.2 (abort-on-bad-parse policy), §6 (per-call parse status), §9 (verification)
**Owner:** Hang Li
**Date:** 2026-04-27

---

## 1. The Bug

MaxContext TopDown crashes on TREC DL19 for `pool ∈ {30, 40, 50}` at the **same query** (13/43) on Qwen3 with this traceback:

```
 30%|███       | 13/43 [02:15<05:12, 10.41s/it]
 30%|███       | 13/43 [02:24<05:34, 11.14s/it]
Traceback (most recent call last):
  File "/scratch/project_mnt/S0090/hang/llm-rankers/run.py", line 520, in <module>
    main(args)
  File "/scratch/project_mnt/S0090/hang/llm-rankers/run.py", line 399, in main
    reranked_results.append((qid, query, ranker.rerank(query, ranking)))
  File "/scratch/project_mnt/S0090/hang/llm-rankers/llmrankers/setwise_extended.py", line 1841, in rerank
    ordered = self._maxcontext_topdown_select(query, docs)
  File "/scratch/project_mnt/S0090/hang/llm-rankers/llmrankers/setwise_extended.py", line 1822, in _maxcontext_topdown_select
    best_label = self.compare(query, window)
  File "/scratch/project_mnt/S0090/hang/llm-rankers/llmrankers/setwise.py", line 672, in compare
    raise ValueError(
ValueError: MaxContext single-label parse failed. Raw text: '0<|im_end|>'
```

**Critical observations:**

1. **All three pool sizes crash at the same query** (`13/43`) with the same raw output (`'0<|im_end|>'`). This is deterministic given Qwen's `do_sample=False` (greedy decoding at `setwise.py:304`).
2. **MaxContext BottomUp succeeds** for the same `pool ∈ {10, 20, 30, 40, 50}` on the same query 13. Same model, same passages, only the prompt differs (`_build_best_prompt` vs `_build_worst_prompt`).
3. The model is emitting **literal `0`** — out-of-range for `1..N` numeric labels — and the `<|im_end|>` Qwen end-of-turn token is concatenated.

---

## 2. Why the Parser Raises

### 2.1 `_parse_single_label` correctly rejects `'0'`

Location: `llmrankers/setwise.py:461-540`

After `_clean_generation_output('0<|im_end|>')` → `'0'`, the numeric-scheme branch tries patterns including `r"^\s*(\d+)(?:\s|[.,;:!?]|$)"` which matches `'0'`:

```python
# setwise.py:509-520
if is_numeric_scheme:
    numeric_patterns = (
        r"(?:BEST|WORST|MOST\s+RELEVANT|LEAST\s+RELEVANT|ANSWER|OUTPUT)\s*[:\-\s]*(?:PASSAGE\s*)?(\d+)",
        r"^\s*(\d+)(?:\s|[.,;:!?]|$)",                                # ← matches '0'
        r"(?:MOST\s+RELEVANT|LEAST\s+RELEVANT|BEST|WORST|CLOSEST(?:\s+MATCH)?)[^.\n]{0,40}?PASSAGE\s+(\d+)",
        r"PASSAGE\s+(\d+)\s+(?:IS|WAS)\s+(?:THE\s+)?(?:MOST(?:\s+RELEVANT)?|LEAST(?:\s+RELEVANT)?|BEST|WORST|CLOSEST(?:\s+MATCH)?)",
    )
    for pattern in numeric_patterns:
        for match in re.findall(pattern, cleaned, flags=re.IGNORECASE):
            idx = int(match) - 1                           # int('0') - 1 = -1
            if 0 <= idx < len(valid_chars):                # -1 fails range check
                return valid_chars[idx]                     # ← never returns
```

`'0'` → `idx=-1` → fails `0 <= idx`. No return. Falls through.

The fallback (lines 529-534) repeats the search:

```python
num_match = re.search(r"\b(\d+)\b", cleaned)
if num_match:
    idx = int(num_match.group(1)) - 1                       # again -1
    if 0 <= idx < len(valid_chars):
        return valid_chars[idx]
return None                                                  # ← returns None
```

`_parse_single_label('0<|im_end|>', ...)` returns `None`.

### 2.2 `_is_numeric_refusal_output` returns `False` for `'0'`

Location: `setwise.py:447-459`

```python
NUMERIC_REFUSAL_REGEX = (
    r"^\s*(none|no\s+passages?|neither|i\s+cannot|cannot\s+determine|"
    r"cannot\s+pick|cannot\s+decide)\b|"
    r"\bnone\s+of\s+the\s+(passages?|above)\b|"
    r"\bno\s+passages?\s+(is|are)\s+relevant\b|"
    r"\b(correct\s+answer|the\s+answer)\s+is\s*:?\s*none\b|"
    r"\bthere\s+is\s+no\s+(least|most)\s+relevant\b|"
    r"\bif\s+there\s+was\s+a\s+passage\b"
)

def _is_numeric_refusal_output(self, raw: str) -> bool:
    cleaned = self._clean_generation_output(raw)
    return re.search(self.NUMERIC_REFUSAL_REGEX, cleaned, flags=re.IGNORECASE) is not None
```

The regex matches lexical refusals ("none", "cannot determine", "if there was a passage", etc.) but **not bare digits**. `_is_numeric_refusal_output('0<|im_end|>')` → `False`.

### 2.3 The raise site

Location: `setwise.py:655-676` (Qwen generation branch in `compare()`)

```python
raw_output = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:],
                                   skip_special_tokens=False).strip()
output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
if output is None:
    is_numeric = getattr(self, "label_scheme", None) == "numeric_1_based"
    if is_numeric and self._is_numeric_refusal_output(raw_output):
        # Deterministic no-op: head wins (no swap in TopDown)
        self.total_parse_fallback = getattr(self, "total_parse_fallback", 0) + 1
        if _DEBUG or getattr(self, "strict_no_parse_fallback", False):
            print(f"[MaxContext] refusal no-op (best=1). Raw: {raw_output!r}")
        output = self.CHARACTERS[0]
    elif getattr(self, "strict_no_parse_fallback", False):     # ← this branch fires
        raise ValueError(                                       # line 672
            f"MaxContext single-label parse failed. Raw text: {raw_output!r}"
        )
    else:
        output = self._clean_generation_output(raw_output).upper()
```

For `'0<|im_end|>'`:
- `_parse_single_label` returns `None`
- `_is_numeric_refusal_output(raw_output)` returns `False` (bare digit, not a lexical refusal)
- `strict_no_parse_fallback` is `True` (set by `_setup_maxcontext_numeric_attrs`)
- → raises `ValueError`

`MaxContextBottomUp` has the **identical** path at `setwise_extended.py:208-223` in `compare_worst`. The structural code path is symmetric; only the model's response differs.

### 2.4 `strict_no_parse_fallback` setup

Location: `setwise_extended.py:25-33`

```python
def _setup_maxcontext_numeric_attrs(ranker, pool_size: int) -> None:
    ranker.CHARACTERS = [str(i + 1) for i in range(pool_size)]   # ['1', '2', ..., 'N']
    ranker.num_child = pool_size - 1
    ranker.method = "selection"
    ranker.strict_no_truncation = True
    ranker.strict_no_parse_fallback = True
    ranker.total_parse_fallback = 0
    ranker.label_scheme = "numeric_1_based"
    ranker._maxcontext_pool_size = pool_size
```

---

## 3. The Asymmetry — Same Query, Different Outcome

### 3.1 Prompt construction (`setwise.py:199-227`)

Both prompts are constructed by sister functions and append the **identical numeric footer** when `label_scheme == "numeric_1_based"`:

```python
def _build_best_prompt(self, query: str, docs: Sequence[SearchResult]) -> str:
    passages = self._format_passages(docs)
    prompt = (
        f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n'
        + passages
        + "\n\nOutput only the passage label of the most relevant passage:"
    )
    if getattr(self, "label_scheme", None) == "numeric_1_based":
        prompt += (
            f"\n\nReply with exactly one passage number from 1 to {len(docs)}. "
            "Do not explain. If none of the passages are clearly relevant, "
            "still pick the single closest one."
        )
    return prompt

def _build_worst_prompt(self, query: str, docs: Sequence[SearchResult]) -> str:
    # ... identical structure with "least relevant" instead of "most relevant"
    if getattr(self, "label_scheme", None) == "numeric_1_based":
        prompt += (
            f"\n\nReply with exactly one passage number from 1 to {len(docs)}. "
            "Do not explain. If none of the passages are clearly irrelevant, "
            "still pick the single least relevant one."
        )
    return prompt
```

The prompts **explicitly instruct the model** to reply with `1..N` and to "still pick" even if nothing matches. Yet the model emits `0` for the "most relevant" prompt at large pools, while the same model on the same query, same docs, with "least relevant" instead, emits a valid `1..N`.

### 3.2 The window passed to `compare()`

`MaxContextTopDownSetwiseLlmRanker._maxcontext_topdown_select` (`setwise_extended.py:1789-1832`) walks the pool top-down:

```python
n = len(ranking)
top_idx = 0
while n - top_idx > 1:
    window_len = n - top_idx
    if window_len <= 2:
        # ... BM25 endgame for last 2 docs
    else:
        window = ranking[top_idx:]                  # whole live pool
        best_label = self.compare(query, window)
        best_window_pos = _resolve_maxcontext_label_index(
            self, best_label, window_len, default=0
        )
    # ... swap into top_idx position, advance
```

At pool=30, the first comparison sees a 30-doc window. At pool=50, the first sees 50 docs. The crash at query 13 implies the model's response distribution flips at this query for `most_relevant` prompts when the window is large.

`MaxContextBottomUp` does the symmetric thing from the bottom (`setwise_extended.py:1881-1899`).

---

## 4. Recent Commit History (relevant only)

| SHA | Title | Effect |
|---|---|---|
| `7307e9b` | improve max context parser | Added `NUMERIC_REFUSAL_REGEX` + `_is_numeric_refusal_output`. Added refusal → deterministic head/tail no-op for both `compare` and `compare_worst`. Added numeric prompt footer. **This is the current code.** |
| `c892d00` | fix max context single end parse | Added 4 numeric patterns to `_parse_single_label` (the `^\s*(\d+)…` pattern is here). |
| `74a6162` | add topdown/bottomup max context runs | Created `MaxContextTopDownSetwiseLlmRanker` + `MaxContextBottomUpSetwiseLlmRanker`. Set `strict_no_parse_fallback=True`. Wired the raise. |
| `7d94636` | add topdown bigram rank prefix | Orthogonal — added `bigrams_aa_zz` label scheme (676 labels) for plain TopDown, not MaxContext. Not relevant to the `0` bug. |
| `d395adc` | revert "add strict gating for the dual parse" | Rolled back `prefix_allowed_tokens_fn` / token-trie constraint approach because it was incompatible with Qwen3 generation. **Relevant: do not propose a prefix-constraint fix without justifying why this time would work.** |
| `bcc9a07` | revert "fix max context approach passage numbering" | Same revert chain. |

The `7307e9b` diff is the load-bearing change for this bug (its refusal handling exists but does not catch `'0'`).

---

## 5. Spec Citation: `IDEA_007.md` §3.2

> **Abort-on-bad-parse policy.** Any out-of-range label, duplicate best/worst, or fallback parse triggers immediate run-abort with a diagnostic. **No retry.** `_generate()` (`setwise.py:304`) is greedy-deterministic (`do_sample=False`) so retrying the identical prompt reproduces the same output. If retry is later worth implementing, it must use a *different* repair prompt (e.g. a strict "Output only `Best: <int>, Worst: <int>` with integers in [1, N]"), not the same prompt.

This bounds the solution space:

- ✅ Per-spec, an out-of-range label IS supposed to abort (current behavior).
- 🔁 But the spec also distinguishes "true parse failure" (worth aborting) from "model refusal" (worth a soft no-op, as the recent `7307e9b` commit operationalizes for lexical refusals).
- 📌 The question: **is `'0'` (an out-of-range integer) a truly corrupted output or a model-side refusal expressed as 0?**
- ❓ If it's a refusal, the existing soft-no-op pattern should extend to it. If it's a parser/prompt failure, the spec's "different repair prompt" path is the right move.

---

## 6. Ranked Hypotheses for Codex to Evaluate

I want Codex to assess these and recommend a primary fix + optional belt-and-suspenders:

**(a) Soft-fold `'0'` (and any out-of-range positive integer) into the existing refusal pathway.** Treat any cleaned numeric output that parses to a valid integer but falls outside `[1, N]` as a refusal → deterministic head/tail no-op + log + increment `total_parse_fallback`. Pro: smallest delta, matches the spec's distinction between "corrupted" and "refusal". Con: could mask a parser bug if the model is actually trying to communicate something else.

**(b) One-shot repair prompt with explicit "Do not output 0".** When parse fails, call the model again with a strict repair prompt that enumerates `[1, N]` and rejects 0 explicitly. Pro: aligned with the spec's "different repair prompt" guidance. Con: doubles latency for the failure case; risks the model emitting `0` again if the underlying behavior is a confidence collapse.

**(c) Prefix-constrained generation.** Use `prefix_allowed_tokens_fn` to restrict generation to tokens that can start `1..N`. Pro: hard guarantee. Con: previously tried and reverted (`a606221` / `bcc9a07` / `d395adc`) — incompatible with Qwen3 generation as implemented before; needs justification for why this time would work.

**(d) Prompt engineering only.** Add "Do not output 0; only integers in [1, N]" to the existing footer. Pro: zero code complexity. Con: doesn't fix the failure mode if the model still emits 0; needs a fallback anyway.

**(e) Combination.** E.g., (a) + (d): prompt-strengthen plus soft-fold. Or (a) + (b) with a `repair_attempts` counter.

---

## 7. Hard Constraints on Any Fix

1. **Symmetric across the MaxContext family.** TopDown, BottomUp, and DualEnd must all benefit from the same hardening. The dual parser is at `_try_parse_dual_output` / `_parse_dual_output` in `setwise_extended.py` — confirm whether the same `'0'`-class corruption can hit it.
2. **No regression on letter-scheme code.** `CHARACTERS = [A..W]` paths (existing TopDown / BottomUp / DualEnd in `setwise.py` and `setwise_extended.py` for non-MaxContext rankers) must be byte-identical in behavior.
3. **Auditable.** Every fallback event logs to `total_parse_fallback`, prints a debug line, and ideally adds a structured field to the JSONL output so analysis can stratify "true parse" vs "refusal no-op" vs "out-of-range no-op" per query / pool / model.
4. **Greedy-deterministic-safe.** Same-prompt retries are no-ops. Any retry must use a different prompt.
5. **Preserve the spec's distinction.** A genuinely corrupted output (e.g., `'asdf'`, `'<|endoftext|>'` only, `'qux'`) should still abort. The fix should *not* turn the strict abort into a permissive shrug.
6. **No new heavy dependencies.** Code-only change; existing `re`, `transformers`, `torch` are fine.

---

## 8. What I Need From You

Please produce `INVESTIGATION_REPORT.md` with these sections (under 2500 words):

1. **Root-cause analysis.** Why does the model emit `'0<|im_end|>'`? Is `0` a Qwen "I refuse / I don't know" signal at the token level, an off-by-one indexing slip from pretraining (where `0`-indexed lists are common in code corpora), an attention failure at long context, or something else? Cite mechanistic-interpretability or empirical reasoning if possible.
2. **Why TopDown ≠ BottomUp at the same query.** Hypothesis for the asymmetry under "most relevant" vs "least relevant" framing on the same docs.
3. **Ranked recommendations.** Score (a)-(e) above on: spec alignment, fix robustness, future-proofing for DualEnd, complexity, latency cost. Add any options I missed.
4. **Recommended primary fix + optional belt-and-suspenders.** Concrete: which lines in which files change, what the new logic looks like in pseudocode (do not write the full Python — that's Phase 4). Include where the new fallback event gets logged and what JSONL field to add.
5. **Test plan.** Unit tests we can write today for the parser. Specifically: what raw outputs should hit the fallback (e.g., `'0'`, `'0<|im_end|>'`, `'-1'`, `'51'`, `'0\n'`, `'  0  '`), what should still abort (e.g., `'asdf'`, `''`), and what should still parse cleanly (`'1'`, `'25'`, `'50'`, `'<answer>5</answer>'`).
6. **DualEnd extension.** Whether and how the same fix needs to apply to `compare_both` / dual parser. If yes, sketch the equivalent change site.
7. **Open questions.** Anything you can't determine without running an experiment (e.g., "need to log the model's full output distribution for query 13 across pool sizes 10..50 to see when `0` first appears").

**Output:** write the report to `/Users/hangli/projects/llm-rankers/.planning/idea_007_topdown_parser/INVESTIGATION_REPORT.md`. Do not modify any source code in this phase.

---

## 9. Files & Pointers

- `llmrankers/setwise.py` — base ranker. Key: lines 199-227 (prompts), 422-460 (cleaning + refusal predicate), 461-540 (`_parse_single_label`), 655-676 (Qwen generation branch in `compare()`).
- `llmrankers/setwise_extended.py` — extensions. Key: lines 25-55 (`_setup_maxcontext_numeric_attrs`, `_resolve_maxcontext_label_index`), 208-223 (BottomUp `compare_worst` mirror), 1738-1900 (`MaxContextTopDownSetwiseLlmRanker`, `MaxContextBottomUpSetwiseLlmRanker`).
- `IDEA_007.md` — full design spec, especially §3.2.
- `analysis/position_bias.py` — consumer of `label_scheme`. Verify any new JSONL field doesn't break it.
- `run.py:399` — driver; verify the per-call exception handling.
- Recent commits: `git show 7307e9b`, `git show 74a6162` give you the most recent parser changes.
