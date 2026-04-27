# Audit Round 3 - Draft Fix Plan v3

## Verdict

`ACCEPT`

v3 resolves the blocking/high issues from round 2 and is implementation-ready. One non-blocking wording nit remains in the test section, noted below.

## Resolution Table

| Round-2 finding | Status | v3 resolution |
|---|---|---|
| `parse_status` locals could be undefined on T5 / likelihood branches | `RESOLVED` | v3 says: “initialize `parse_status = "parsed"` and `parse_fallback_reason = None` at the TOP of `compare()`, `compare_worst()`, `compare_both()` — before any scoring-branch dispatch” (`DRAFT_PLAN.md:8-10`). TopDown details at `DRAFT_PLAN.md:103-111`; BottomUp at `DRAFT_PLAN.md:157-163`; DualEnd at `DRAFT_PLAN.md:165-176`. |
| DualEnd `rerank` reset snippet dropped length/context checks | `RESOLVED` | v3 says the fix is an “INSERTION” and shows the existing length check plus `_assert_maxcontext_fits(query, docs)` preserved before resetting counters (`DRAFT_PLAN.md:292-320`). |
| DualEnd test layering mixed parser behavior with wrapper behavior | `RESOLVED` | v3 splits tests into “Layer 1 — direct classifier tests”, “Layer 2 — `_parse_dual_output` strict-raise behavior”, and “Layer 3 — `compare_both` wrapper soft-fold” (`DRAFT_PLAN.md:392-482`). |
| Numeric out-of-range call-site test skipped one fixture | `RESOLVED` | v3 says “Loop over all raw inputs” and asserts counters equal `len(raw_inputs)` for both TopDown and BottomUp (`DRAFT_PLAN.md:364-388`). |
| Letter-scheme `-1` regression expectation underspecified | `RESOLVED` | v3 makes the expected preservation explicit: `assert legacy_letter_ranker._parse_single_label("-1", letter_valid) == "A"` (`DRAFT_PLAN.md:497-510`) and gates the signed-numeric guard on `is_numeric_scheme` (`DRAFT_PLAN.md:514-535`). |

## New Findings

| Severity | Location | Problem | Concrete edit |
|---|---|---|---|
| NIT | `DRAFT_PLAN.md:497-500` | The prose says the bare-integer guard is “inside `_parse_single_label` regardless of scheme” and “returns `None` for any whole-output bare integer,” which conflicts with the corrected gated guard in §2.2 and `DRAFT_PLAN.md:514-535`. The code sketch and intended assertion are clear, so this is not blocking. | Delete or rewrite that sentence to: “The signed-numeric guard must remain gated on `is_numeric_scheme`; letter/bigram schemes continue through the legacy numeric fallback.” |

## Final Implementation-Readiness Statement

Accepted for Phase 4. The plan now has enough detail for Codex to implement without further design questions:

- Exact source edit sites are listed in `DRAFT_PLAN.md:587-594`.
- The strict/no-op distinction is concrete and scheme-gated.
- TopDown, BottomUp, and DualEnd all have specified behavior.
- Counter initialization and per-query reset are specified.
- JSONL logging fields and analysis compatibility are covered.
- Tests are scoped to the repo’s existing `scripts/check_maxcontext_invariants.py` gate.

The only suggested pre-implementation cleanup is the NIT above; it is documentation clarity, not a design blocker.
