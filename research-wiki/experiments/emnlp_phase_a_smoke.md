---
type: experiment
node_id: exp:emnlp_phase_a_smoke
title: "EMNLP Phase A — smoke gate"
status: not_submitted
config:
  methods: 7
  models: ["Qwen3.5-9B", "Meta-Llama-3.1-8B-Instruct", "Ministral-3-8B-Instruct-2512"]
  datasets: ["dl19"]
  pool_sizes: [50, 100]
  cells: 42
tests: ["idea:008"]
plan_doc: "/Users/hangli/projects/llm-rankers/EMNLP_EXPERIMENT_PLAN.md"
origin_skill: manual-backfill
created_at: 2026-05-05T22:15:00+10:00
updated_at: 2026-05-08T00:00:00+10:00
---

v8 extends Phase A to pool=100 for the three required large-context model families. The original 21 pool=50 cells remain the prime smoke-as-golden layer; the 21 pool=100 cells are supplemental v8 fit/parse goldens after they pass.

## Pass criteria (parse-failure hotfix 2026-05-08)

`scripts/smoke_emnlp_models.sh --verify-only` now asserts the following per
MaxContext cell, in addition to the existing checks
(`Avg parse fallbacks: 0`, `Avg numeric out-of-range fallbacks: 0`, no
`Traceback`/`ERROR`/`exceeds model limit` in the log):

- `Avg parse_failure_strict: 0` — zero unrecovered LLM label-parse failures
  under strict-mode (no BM25 fallback).
- `Avg parse_failure_bm25_fallback: 0` — Phase A runs are submitted **without**
  the `--allow-parse-failure-bm25-fallback` flag, so any non-zero count
  indicates the flag was set in error.

If a cell hits a non-zero `parse_failure_strict`, the smoke gate fails. The
remediation order is: (a) inspect the offending raw_output in the log (it is
truncated into the `ValueError` message), (b) extend
`_parse_single_label` (`llmrankers/setwise.py`) with a covering pattern, (c)
re-run the smoke. Phase B/C/F dispatchers may flip
`--allow-parse-failure-bm25-fallback` on; Phase A must not.
