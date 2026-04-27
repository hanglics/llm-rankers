# Audit Round 2 - Draft Fix Plan v2

## Verdict

`NEEDS_REVISION`

v2 resolves the round-1 conceptual issues, but it introduces implementation ambiguities that can cause regressions if Codex follows the snippets literally. The plan is close; the remaining edits are mechanical.

## Resolution Table

| Round-1 finding | Status | v2 resolution |
|---|---|---|
| DualEnd counters over-count without per-query reset | `PARTIAL` | v2 adds a reset in `MaxContextDualEndSetwiseLlmRanker.rerank` (`DRAFT_PLAN.md:263-274`). However, the snippet drops the existing length and context-fit checks from `setwise_extended.py:1195-1202`; see new finding 2. |
| Tests pointed at nonexistent `tests/` directory | `RESOLVED` | “No new test files... extend `scripts/check_maxcontext_invariants.py`” (`DRAFT_PLAN.md:301-304`) and verification uses `python scripts/check_maxcontext_invariants.py` (`DRAFT_PLAN.md:419`). |
| `_last_parse_status` sticky instance state | `RESOLVED` | v2 uses local `parse_status` / `parse_fallback_reason` and passes them inline (`DRAFT_PLAN.md:96-135`, `137-143`, `191-204`). |
| `_classify_numeric_noop` unguarded in base DualEnd | `RESOLVED` | Helper returns `None` unless `label_scheme == "numeric_1_based"` (`DRAFT_PLAN.md:52-56`), and DualEnd has a call-site numeric guard (`DRAFT_PLAN.md:156-164`). |
| DualEnd diagnostic lost raw output | `RESOLVED` | v2 keeps `raw_output`, uses `cleaned_output` for parse, and logs raw text (`DRAFT_PLAN.md:149-187`). |
| DualEnd lexical no-op could catch structured malformed text | `RESOLVED` | v2 checks for `BEST` / `WORST` / `PASSAGE <num>` and raises if present (`DRAFT_PLAN.md:165-175`). |
| Repro criteria required `0` after prompt change | `RESOLVED` | v2 accepts either logged `0` fallback or no further `0` emission (`DRAFT_PLAN.md:392-402`). |
| DualEnd snapshot test not updated | `RESOLVED` | v2 adds the three counter keys to expected snapshot (`DRAFT_PLAN.md:360-370`). |
| XML wrapper ambiguity | `RESOLVED` | v2 explicitly puts XML-wrapper out-of-range handling out of scope (`DRAFT_PLAN.md:437-439`). |
| `<|endoftext|>` cleaning note wrong | `RESOLVED` | v2 changelog states the corrected cleaner behavior (`DRAFT_PLAN.md:17-18`); no bad note remains in test section. |
| Counter init must precede `optional_stat_totals` | `RESOLVED` | v2 calls this out explicitly (`DRAFT_PLAN.md:257`, `278-280`). |

## New Findings

| Severity | Location | Problem | Concrete edit |
|---|---|---|---|
| BLOCKER | `DRAFT_PLAN.md:96-124`, `137-143`, `156-204`; `setwise.py:600-690`; `setwise_extended.py:160-237`, `520-591` | `parse_status` locals are initialized only inside the causal-generation snippets. The shared `_log_comparison` sites also run for T5 generation and likelihood branches. If implementation passes `parse_status=parse_status` at the shared log site without top-level initialization, non-Qwen/non-MaxContext paths can hit `UnboundLocalError`, violating the no-regression constraint. | In `compare()`, `compare_worst()`, and `compare_both()`, initialize `parse_status = "parsed"` and `parse_fallback_reason = None` before the `if self.scoring == ...` branch. Then only mutate them in fallback branches. |
| HIGH | `DRAFT_PLAN.md:263-274`; `setwise_extended.py:1195-1202` | The DualEnd rerank reset sketch replaces the current method body with `return super().rerank(query, docs)`, omitting `len(docs) == pool_size` and `_assert_maxcontext_fits(...)`. If followed literally, it weakens MaxContext invariants. | Show the edit as insertion into the existing method: keep the length check and `_assert_maxcontext_fits(query, docs)`, reset the three fallback counters, then `return super().rerank(query, docs)`. |
| MEDIUM | `DRAFT_PLAN.md:343-358`; `setwise_extended.py:778-793` | The DualEnd test sketch mixes parser-level and wrapper-level behavior. `_parse_dual_output("0", ..., strict=True)` should still raise; the soft-fold belongs in `compare_both()`’s try/except wrapper. | Specify direct classifier tests for `_classify_numeric_noop`, direct `_parse_dual_output` tests that malformed strict outputs still raise, and `compare_both` stub tests for `0` / lexical refusal no-op pairs. |
| LOW | `DRAFT_PLAN.md:323-336` | The numeric out-of-range call-site test lists six raw inputs but loops over `raw_inputs[:5]` and asserts counter value `5`, leaving `"  0  "` untested. | Loop over all `raw_inputs` and assert counters equal `len(raw_inputs)` for both TopDown and BottomUp. |
| LOW | `DRAFT_PLAN.md:373-375`; `setwise.py:536-540` | The letter-scheme `-1` regression expectation is underspecified. Current non-numeric fallback likely parses `-1` as numeric label `A` because `\b(\d+)\b` matches `1`. “Does NOT trigger the new signed-numeric guard” is not an assertion. | Make the expected current behavior explicit, e.g. `assert legacy_letter_ranker._parse_single_label("-1", letter_valid) == "A"` if preserving byte-identical behavior is intended. |

## Implementation-Readiness Check

Not ready to implement without the revisions above. After adding top-level `parse_status` initialization, preserving the existing DualEnd `rerank` invariants, and tightening the test descriptions, the plan has enough detail for Phase 4.

One implementation note: `_log_comparison`’s new parameters should remain optional and default to `None`; existing call sites not touched by MaxContext will continue to work, but any call site updated to pass locals must have those locals initialized on every branch.
