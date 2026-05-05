---
type: experiment
node_id: exp:emnlp_phase_b_main_matrix
title: "EMNLP Phase B — required main matrix"
status: not_submitted
config:
  methods: 7
  models_required: 9
  datasets: 8
  pool_sizes: [10, 20, 30, 40, 50, 100]
  cells: 3024
tests: ["idea:008", "claim:C7"]
plan_doc: "/Users/hangli/projects/llm-rankers/EMNLP_EXPERIMENT_PLAN.md"
origin_skill: manual-backfill
created_at: 2026-05-05T22:15:00+10:00
updated_at: 2026-05-05T22:15:00+10:00
---

v8 adds pool=100 for Qwen3.5, Llama-3.1, and Ministral-3 required models. Phase B BEIR pool=100 cells are gated by `scripts/probe_beir_pool100_fit.py` before launch.
