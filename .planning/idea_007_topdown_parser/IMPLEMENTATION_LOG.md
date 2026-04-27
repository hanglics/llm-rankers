# Implementation Log - MaxContext `0` Parser Fix

## File-by-file summary

- `llmrankers/setwise.py`
  - Extended `_log_comparison` around lines 170-198 with optional `parse_status`, `parse_fallback_reason`, and `raw_output` JSONL fields.
  - Hardened numeric-scheme prompt footers in `_build_best_prompt` and `_build_worst_prompt` around lines 217-240 with the explicit "Do not output 0 or any number outside 1 to N" clause.
  - Added `_NUMERIC_ONLY_REGEX` and `_classify_numeric_noop` around lines 459-489. The helper returns `None` for non-`numeric_1_based` schemes.
  - Added the numeric-scheme-only signed-integer guard in `_parse_single_label` around lines 552-567, preserving letter/bigram fallback behavior.
  - Updated `compare` around lines 583-735 to initialize parse locals at the top, soft-fold numeric/lexical no-op outputs only in the Qwen causal-generation branch, increment reason counters, and log parse status.

- `llmrankers/setwise_extended.py`
  - Extended `_setup_maxcontext_numeric_attrs` around lines 25-35 with `total_lexical_refusal_fallback` and `total_numeric_out_of_range_fallback`.
  - Mirrored the TopDown parse-status and no-op logic in `BottomUpSetwiseLlmRanker.compare_worst` around lines 114-250.
  - Wrapped `DualEndSetwiseLlmRanker.compare_both` parsing around lines 518-643. It preserves raw output, parses/classifies on cleaned output, soft-folds whole-output lexical or bare numeric no-ops to `(1, N)`, and keeps structured malformed outputs strict.
  - Refactored `MaxContextDualEndSetwiseLlmRanker.__init__` around lines 1199-1206 to call `_setup_maxcontext_numeric_attrs`.
  - Inserted DualEnd per-query counter resets in `MaxContextDualEndSetwiseLlmRanker.rerank` around lines 1241-1251 while preserving the existing length check and `_assert_maxcontext_fits`.
  - Added reason-counter resets in MaxContext TopDown/BottomUp selection loops around lines 1838-1845 and 1939-1946.

- `run.py`
  - Added the two reason-specific fallback counters to `optional_stat_labels` around lines 377-386.

- `scripts/check_maxcontext_invariants.py`
  - Updated MaxContext compare stubs with reason counters and added a DualEnd compare stub around lines 320-379.
  - Updated `test_maxcontext_dualend_byte_identity_snapshot` around lines 428-500 with the three counter attrs.
  - Added classifier, strict DualEnd parser, numeric out-of-range compare, and DualEnd wrapper no-op tests around lines 927-1151.
  - Extended numeric parser fixtures with `0`, `0<|im_end|>`, `-1`, `51`, `0\n`, and padded `0`; added the letter-scheme `-1 -> A` canary.

## Deviations

- No behavioral deviations from the final plan.
- Additive logging note: `_log_comparison` now accepts optional `raw_output`, and DualEnd fallback rows include the uncleaned raw output. This satisfies hard constraint 6 and is omitted from existing/default log calls.
- Environment note: the default non-activated shell does not have a `python` executable. Verification passed with the repo-provided `ranker_env/bin` prepended to `PATH`, so the invoked executable name was still `python`.

## Verification

Requested command in the default shell:

```text
$ python scripts/check_maxcontext_invariants.py
zsh:1: command not found: python
```

Passing verification command:

```bash
PATH="$PWD/ranker_env/bin:$PATH" python scripts/check_maxcontext_invariants.py
```

Last 50 lines:

```text
WARNING: Using incubator modules: jdk.incubator.vector
Warning: Partial dual parse from 'mangled output', using A as best, D as worst
Warning: Could not parse dual output: '###', defaulting to A and C
[MaxContext] lexical_refusal no-op (best=1). Raw: 'None of the passages are relevant to the query "what is the most popular food in switzerland".'
[MaxContext] lexical_refusal no-op (best=1). Raw: 'None of the passages are relevant to the query "definition of a sigmet".'
[MaxContext] lexical_refusal no-op (worst=50). Raw: 'None of the passages are relevant to the query "definition of a sigmet".'
[MaxContext] numeric_out_of_range no-op (best=1). Raw: '0'
[MaxContext] numeric_out_of_range no-op (best=1). Raw: '0<|im_end|>'
[MaxContext] numeric_out_of_range no-op (best=1). Raw: '-1'
[MaxContext] numeric_out_of_range no-op (best=1). Raw: '51'
[MaxContext] numeric_out_of_range no-op (best=1). Raw: '0'
[MaxContext] numeric_out_of_range no-op (best=1). Raw: '0'
[MaxContext] numeric_out_of_range no-op (worst=50). Raw: '0'
[MaxContext] numeric_out_of_range no-op (worst=50). Raw: '0<|im_end|>'
[MaxContext] numeric_out_of_range no-op (worst=50). Raw: '-1'
[MaxContext] numeric_out_of_range no-op (worst=50). Raw: '51'
[MaxContext] numeric_out_of_range no-op (worst=50). Raw: '0'
[MaxContext] numeric_out_of_range no-op (worst=50). Raw: '0'
[MaxContext] dual numeric_out_of_range no-op (best=1, worst=50). Raw: '0'
[MaxContext] dual numeric_out_of_range no-op (best=1, worst=50). Raw: '0<|im_end|>'
[MaxContext] dual numeric_out_of_range no-op (best=1, worst=50). Raw: '-1'
[MaxContext] dual numeric_out_of_range no-op (best=1, worst=50). Raw: '51'
[MaxContext] dual numeric_out_of_range no-op (best=1, worst=50). Raw: '0'
[MaxContext] dual numeric_out_of_range no-op (best=1, worst=50). Raw: '0'
[MaxContext] dual lexical_refusal no-op (best=1, worst=50). Raw: 'None of the passages are relevant.'
Warning: Could not parse dual output: '###', defaulting to A and C
check_maxcontext_invariants.py: all checks passed
```

## Open follow-ups

- None for the implementation.
- Operationally, run the requested command from an activated environment or with `ranker_env/bin` on `PATH` so `python` and `torch` resolve.
