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
updated_at: 2026-05-05T22:15:00+10:00
---

v8 extends Phase A to pool=100 for the three required large-context model families. The original 21 pool=50 cells remain the prime smoke-as-golden layer; the 21 pool=100 cells are supplemental v8 fit/parse goldens after they pass.
