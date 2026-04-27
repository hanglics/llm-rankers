# Audit Round 1 - Draft Fix Plan

## Verdict

`NEEDS_REVISION`

The core choice, narrow numeric-only out-of-range no-op plus numeric-only prompt hardening, is sound. The plan needs edits before implementation: tests are pointed at a nonexistent `tests/` layout, DualEnd counters would aggregate incorrectly unless reset per query, and parse-status logging should not rely on a sticky instance attribute.

## Findings

| Severity | Location | Problem | Concrete edit |
|---|---|---|---|
| BLOCKER | `DRAFT_PLAN.md:216-220`; `llmrankers/setwise_extended.py:1195-1202`; `run.py:390-405` | DualEnd counters are initialized but not reset per query. `run.py` sums per-query ranker attributes after each `rerank`; inherited `DualEndSetwiseLlmRanker.rerank` resets only compare/token counters at `setwise_extended.py:831-834`. New fallback counters would become cumulative and be over-counted. | In `MaxContextDualEndSetwiseLlmRanker.rerank`, reset `total_parse_fallback`, `total_lexical_refusal_fallback`, and `total_numeric_out_of_range_fallback` immediately before `super().rerank(query, docs)`. Keep TopDown/BottomUp resets as planned. |
| HIGH | `DRAFT_PLAN.md:258-260`, `303-316`, `355`, `369-370`; repo test layout | There is no `tests/` directory. Existing parser/MaxContext invariant tests live in `scripts/check_maxcontext_invariants.py` with plain asserts, including current parser fixtures at `scripts/check_maxcontext_invariants.py:889-994`. | Replace new pytest files with extensions to `scripts/check_maxcontext_invariants.py`; update verification to `python scripts/check_maxcontext_invariants.py`. If pytest is introduced later, make that a separate test-infra change. |
| HIGH | `DRAFT_PLAN.md:117`, `192-196`; `llmrankers/setwise.py:689-690`; `llmrankers/setwise_extended.py:236-237`, `590-591` | `_last_parse_status` is unnecessary mutable state and can go stale across branches. `_log_comparison` already accepts the right parameters in the plan. | Use local variables: initialize `parse_status = "parsed"` and `parse_fallback_reason = None` in each compare method, update them at the fallback site, and pass them directly to `_log_comparison`. Do not add `_last_parse_status`. |
| HIGH | `DRAFT_PLAN.md:138-144`; `llmrankers/setwise_extended.py:557-562` | DualEnd calls `_classify_numeric_noop` unguarded in base `compare_both`. That method is used by non-MaxContext DualEnd variants too; strict tests can construct non-numeric DualEnd instances. | Gate before calling: `is_numeric = getattr(self, "label_scheme", None) == "numeric_1_based"; reason = self._classify_numeric_noop(raw_output, len(docs)) if is_numeric else None`. Also make the helper return `None` for non-`numeric_1_based` as a defensive guard. |
| MEDIUM | `DRAFT_PLAN.md:130-153`; `llmrankers/setwise_extended.py:557-559` | The DualEnd sketch overwrites raw decoded text with cleaned text and prints the cleaned value as `Raw`. This weakens diagnostics, especially for Qwen special tokens. | Use `raw_output = decode(...).strip()`, `cleaned_output = self._clean_generation_output(raw_output)`, parse/classify using either cleaned or raw as appropriate, and print/log `raw_output!r`. |
| MEDIUM | `DRAFT_PLAN.md:31-54`, `130-144` | The classifier’s lexical branch uses a regex search, but §2.5 says DualEnd should soft-fold only when the whole output is a recognized no-op. This mismatch could classify a partially structured malformed output as lexical refusal if it happens to contain a refusal phrase. | For DualEnd, either use a stricter whole-output helper, or guard lexical no-op when explicit selector structure is present (`BEST`, `WORST`, `PASSAGE <num>`). Keep single-extreme behavior compatible with commit `7307e9b`. |
| MEDIUM | `DRAFT_PLAN.md:338-340` | The targeted reproduction requires at least one `numeric_out_of_range_noop` for query 13 after the prompt changes. The prompt is deliberately changed, so the model may stop emitting `0`; that would be success, not failure. The BottomUp NDCG comparison is also too strong for a parser smoke test. | Change pass criteria to: jobs complete, no strict parse crash, any observed `0` is logged as `numeric_out_of_range_noop`, outputs are valid permutations, and metrics are recorded. Compare NDCG separately. |
| MEDIUM | `DRAFT_PLAN.md:428-465` implicit via refactor; `scripts/check_maxcontext_invariants.py:428-465` | Refactoring DualEnd to `_setup_maxcontext_numeric_attrs` changes the exact snapshot test by adding fallback counters. | Update `test_maxcontext_dualend_byte_identity_snapshot` expected fields to include `total_parse_fallback`, `total_lexical_refusal_fallback`, and `total_numeric_out_of_range_fallback`. |
| LOW | `DRAFT_PLAN.md:31-54`; `INVESTIGATION_REPORT.md:41` | My report allowed “simple XML wrapper contains exactly one signed integer”; the draft only handles bare signed integers. `'<answer>5</answer>'` is tested as clean parse, but `'<answer>0</answer>'` is not covered. | Either explicitly drop XML-wrapper no-op as out of scope, or extend `_NUMERIC_ONLY_REGEX`/classifier and add `'<answer>0</answer>'` fallback and `'<answer>5</answer>'` parse tests. |
| LOW | `DRAFT_PLAN.md:281`; `llmrankers/setwise.py:422-445` | The plan says `'<|endoftext|>'` becomes `''` after cleaning. Current cleaner returns `cleaned or stripped`, so special-token-only output returns the original stripped token. | Fix the test note: expected behavior is still strict abort, but not because cleaning returns empty. |
| NIT | `DRAFT_PLAN.md:224-230`; `run.py:377-384` | The plan says counters are “emitted to the run summary”; the code emits them through `optional_stat_labels` only if the ranker has the attributes before the loop. | State explicitly that counters must be initialized in constructors/setup before `optional_stat_totals` is built. |

## Answers to Open Questions

1. Keep `_is_numeric_refusal_output` as a wrapper/helper. Do not delete it in this fix. It is currently used in `_parse_single_label` and the single-extreme call sites; keeping it minimizes churn. If desired, implement it as a lexical-refusal wrapper around the new classifier, but preserve behavior.

2. Do not use `_last_parse_status`. Use local `parse_status` / `parse_fallback_reason` variables and pass them inline to `_log_comparison`.

3. Parser tests should live in `scripts/check_maxcontext_invariants.py`. There is no `tests/` directory in the repo. Extend the existing fixtures around `scripts/check_maxcontext_invariants.py:889-994` and add DualEnd fixtures near the existing dual parser tests.

4. Yes, the prompt-hardening clause is correctly gated to `label_scheme == "numeric_1_based"` in the draft (`DRAFT_PLAN.md:232-252`) and current prompt footer is already numeric-scheme-only (`llmrankers/setwise.py:207-212`, `222-227`).

5. `analysis/*.py` will not break on extra JSONL fields or missing new fields. `analysis/position_bias.py` is the only Python analysis script reading comparison JSONL (`analysis/position_bias.py:36-59`, `88-93`) and ignores unknown keys; existing files without `parse_status` are fine. `analysis/parse_success_rate.sh` greps log text, not JSONL fields.

6. Do not hard-assert inside `_classify_numeric_noop`; that risks crashing accidental non-numeric callers. Prefer both: call-site guard plus an early `return None` unless `label_scheme == "numeric_1_based"`.

## Missed Items

I missed the DualEnd per-query counter reset issue in the original investigation. Because `run.py` sums ranker attributes after each query, any newly initialized DualEnd fallback counters must be reset inside `MaxContextDualEndSetwiseLlmRanker.rerank`, not only in `__init__`.

I also under-specified the test location. This repo currently uses `scripts/check_maxcontext_invariants.py` as the invariant gate rather than pytest files under `tests/`.

## Over-Engineering

The Flan-T5 byte-identical rerun in `DRAFT_PLAN.md:357` is heavier than necessary for this parser unblock. The guardrail is valid, but for the staged matrix I would make `scripts/check_maxcontext_invariants.py` plus the three failed MaxContext jobs the required gate, and leave full baseline reruns as optional confidence checks.

The separate reason-specific aggregate counters are useful but not required to fix the crash. If time is tight, `total_parse_fallback` plus JSONL `parse_fallback_reason` is enough; aggregate reason counters are a convenience for summaries.
