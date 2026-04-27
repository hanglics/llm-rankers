# Investigation Report - MaxContext TopDown `0` Parse Failure

## 1. Root-cause analysis

The immediate cause is not a parser bug: `0<|im_end|>` cleans to `0` in `llmrankers/setwise.py:422-445`; `_parse_single_label` then treats numeric labels as 1-based and rejects `idx = int("0") - 1 = -1` at `llmrankers/setwise.py:509-520`, then returns `None` at `llmrankers/setwise.py:529-534`. Because `0` is not in `NUMERIC_REFUSAL_REGEX` (`llmrankers/setwise.py:447-459`), strict MaxContext raises at `llmrankers/setwise.py:662-674`.

At the model level, I would not interpret `0` as a Qwen special token. The special token is `<|im_end|>`; `0` is ordinary decoded text. The strongest explanation is a semantic no-answer/refusal behavior expressed through a common "0 = none / no valid choice" convention, reinforced by 0-indexed/code-corpus priors and long-context uncertainty. The deterministic same-query recurrence across pool sizes 30/40/50 says this is a stable greedy attractor for that prompt/query/window, not sampling noise.

One extra parser hazard: `-1` currently risks parsing as `1` through the fallback `re.search(r"\b(\d+)\b", cleaned)` at `llmrankers/setwise.py:529-533`. Any fix for `0` should guard signed numeric-only outputs before the loose fallback.

## 2. Why TopDown != BottomUp at the same query

The code path is structurally symmetric: TopDown decodes in `compare()` and raises at `llmrankers/setwise.py:662-674`; BottomUp mirrors this in `compare_worst()` at `llmrankers/setwise_extended.py:208-223`. The prompt differs only in the semantic task: `_build_best_prompt` asks for "most relevant" at `llmrankers/setwise.py:200-213`, while `_build_worst_prompt` asks for "least relevant" at `llmrankers/setwise.py:215-227`.

The likely asymmetry is task framing under ambiguous retrieval evidence. For a query whose live pool has no clearly relevant passage, "most relevant" can collapse into "none of the above", and `0` is a compact way to express that despite the footer. "Least relevant" almost always has a satisfiable answer in a large pool: some passage is off-topic or less useful. So BottomUp can emit a valid label on the same docs while TopDown refuses. This matches the observation that the failure starts only at large windows, where attention dilution and relevance ambiguity are both worse.

## 3. Ranked recommendations

| Option | Spec alignment | Robustness | DualEnd future-proofing | Complexity | Latency | Rank |
|---|---:|---:|---:|---:|---:|---:|
| (e) Narrow (a) + prompt hardening (d) | 5 | 4 | 4 | 2 | 0 | 1 |
| (a) Soft-fold numeric-only out-of-range | 4 | 4 | 4 | 2 | 0 | 2 |
| (b) One-shot repair prompt | 4 | 3 | 3 | 4 | high on failures | 3 |
| (d) Prompt only | 2 | 2 | 2 | 1 | 0 | 4 |
| (c) Prefix-constrained generation | 3 | 5 if correct | 2 | 5 | 0 | 5 |

Recommendation: implement a narrow version of (a), plus (d) as belt-and-suspenders. Treat only numeric-only out-of-range answers, after special-token cleanup, as `numeric_out_of_range_noop`: `0`, `0<|im_end|>`, `-1`, `51`, `0\n`, `  0  `. Keep arbitrary text such as `asdf`, empty output, and structured-but-invalid text as strict aborts.

Do not use prefix-constrained generation in this phase. Commit `a606221` added `prefix_allowed_tokens_fn` and a token trie over dual continuations; `d395adc` reverted it, and `bcc9a07` is part of the same rollback chain. The diff shows nontrivial prompt-length slicing, EOS handling, and ambiguous BPE paths such as `1` versus `10` where `0` is a valid continuation token. Reintroducing this would need a separate Qwen3-specific generation audit, not a focused parser hardening.

## 4. Recommended primary fix + belt-and-suspenders

Primary fix:

- Add a numeric-output classifier near `llmrankers/setwise.py:457-459`. Pseudocode:

```text
classify_numeric_noop(raw, n_docs):
    cleaned = clean(raw)
    if lexical_refusal(cleaned): return "lexical_refusal_noop"
    if cleaned or simple XML wrapper contains exactly one signed integer:
        value = int(integer)
        if value < 1 or value > n_docs:
            return "numeric_out_of_range_noop"
    return None
```

- In `_parse_single_label` around `llmrankers/setwise.py:522-534`, guard numeric-only signed outputs before the loose `\b(\d+)\b` fallback so `-1` cannot become label `1`. Valid numeric-only values still parse normally.
- Replace the TopDown strict branch at `llmrankers/setwise.py:662-674` with classifier use: if reason exists, increment `total_parse_fallback`, increment a reason-specific counter, print `[MaxContext] {reason} no-op (best=1) ...`, and return `CHARACTERS[0]`; otherwise strict raise remains.
- Mirror this in BottomUp at `llmrankers/setwise_extended.py:210-223`, returning `CHARACTERS[len(docs)-1]`.
- Initialize/reset counters in `_setup_maxcontext_numeric_attrs` at `llmrankers/setwise_extended.py:25-33`, TopDown reset at `llmrankers/setwise_extended.py:1789-1794`, and BottomUp reset at `llmrankers/setwise_extended.py:1888-1893`.

Auditing:

- Extend `_log_comparison` at `llmrankers/setwise.py:170-188` with optional fields `parse_status` and `parse_fallback_reason`. Existing callers default to `parse_status="parsed"`.
- Pass `parse_status="numeric_out_of_range_noop"` or `"lexical_refusal_noop"` at `llmrankers/setwise.py:690` and `llmrankers/setwise_extended.py:237`.
- Add optional aggregate labels in `run.py:377-384`, e.g. `total_numeric_out_of_range_fallback` and `total_refusal_fallback`.
- `analysis/position_bias.py` will not break: it only requires `label_scheme`, `type`, `positions`, and `selected` (`analysis/position_bias.py:48-59`, `analysis/position_bias.py:88-93`). The new fields let later analysis stratify or exclude fallback events.

Belt-and-suspenders prompt change:

- At `llmrankers/setwise.py:207-212` and `llmrankers/setwise.py:222-227`, add "Do not output 0 or any number outside 1 to {len(docs)}." This should reduce fallback rate, but not replace the parser fix.

## 5. Test plan

Add parser/classifier tests to `scripts/check_maxcontext_invariants.py`.

Fallback/no-op for pool size 50:

- `0`
- `0<|im_end|>`
- `-1`
- `51`
- `0\n`
- `  0  `

Still abort under strict mode:

- `asdf`
- ``
- `<|endoftext|>` only
- `Passage 51 is most relevant` unless the team explicitly decides structured out-of-range labels are also refusals
- `Best: 0, Worst: 7` in DualEnd, because this is a malformed structured pair rather than a whole-output refusal

Still parse cleanly:

- `1` -> `1`
- `25` -> `25`
- `50` -> `50`
- `<answer>5</answer>` -> `5`
- `Best: 5` -> `5`
- `The most relevant passage is Passage 23.` -> `23`

Call-site tests:

- TopDown `compare()` on `0<|im_end|>` returns `1` and increments fallback counters.
- BottomUp `compare_worst()` on `0<|im_end|>` returns the active tail label and increments fallback counters.
- Non-refusal garbage still raises `ValueError`.
- Regression for `-1`: must not return label `1`.

## 6. DualEnd extension

Yes, the same class of corruption can hit MaxContext DualEnd. `MaxContextDualEndSetwiseLlmRanker` sets numeric labels and strict parse at `llmrankers/setwise_extended.py:1147-1160`; `compare_both()` decodes and calls `_parse_dual_output` at `llmrankers/setwise_extended.py:557-562`; `_try_parse_dual_output` rejects invalid numeric labels at `llmrankers/setwise_extended.py:695-710` and `llmrankers/setwise_extended.py:759-774`; strict mode raises at `llmrankers/setwise_extended.py:790-793`.

Equivalent fix: wrap the strict `_parse_dual_output` failure in `compare_both()` at `llmrankers/setwise_extended.py:557-562`. If the whole cleaned output is lexical refusal or numeric-only out-of-range, return a position-preserving no-op pair: `(CHARACTERS[0], CHARACTERS[len(docs)-1])`, increment the same fallback counters, and log both `dual_best` and `dual_worst` with the fallback reason. Do not soft-fold partial structured pairs like `Best: 0, Worst: 7`; those should still abort.

Also add `total_parse_fallback` initialization to the DualEnd MaxContext constructor at `llmrankers/setwise_extended.py:1154-1160` or refactor it to call `_setup_maxcontext_numeric_attrs`.

## 7. Open questions

- Need logits or at least generated raw outputs for query 13 across pool sizes 10/20/30/40/50 to see when `0` first becomes the greedy token.
- Need query text, docids, BM25 scores, and qrels for query 13 to confirm whether the pool lacks obvious relevant passages.
- Need fallback-rate thresholds before paper runs. I would flag >5 percent of comparisons and stop at >20 percent.
- Need a policy decision on structured out-of-range labels (`Passage 51`) versus numeric-only out-of-range (`51`). I recommend only numeric-only no-op for now.
