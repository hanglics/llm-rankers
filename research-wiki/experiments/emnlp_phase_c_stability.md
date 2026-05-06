---
type: experiment
node_id: exp:emnlp_phase_c_stability
title: "EMNLP Phase C — required cross-family stability"
status: not_submitted
config:
  methods: 7
  models: ["Qwen3.5-9B", "Meta-Llama-3.1-8B-Instruct", "Ministral-3-8B-Instruct-2512"]
  datasets: ["dl19"]
  pool_sizes: [10, 20, 30, 40, 50, 100]
  reps: 10
  cells: 1260
  submitted_cells: 1980
tests: ["idea:008", "claim:C7"]
plan_doc: "/Users/hangli/projects/llm-rankers/EMNLP_EXPERIMENT_PLAN.md"
origin_skill: manual-backfill
created_at: 2026-05-05T22:15:00+10:00
updated_at: 2026-05-05T22:15:00+10:00
---

Launcher-consolidation v3 Phase C reports 1260 scientific seven-method cells. The stability wrapper submits 1980 jobs because the default stability layout now includes the IDEA_007 ws-3/ws-PS TopDown overhead plus standard BottomUp ws-3/ws-PS blocks under `original/bottomup/{ws-3,ws-ps}/`.
