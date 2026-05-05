---
type: experiment
node_id: exp:cross_model_stability
title: "Analysis — cross-model stability"
status: not_submitted
config:
  input_roots: "results/maxcontext_dualend/emnlp_phase_c_required/{model_tag}-dl19/stability-test-runs"
  output_dir: "results/emnlp/analysis/cross_model_stability"
tests: ["idea:008", "claim:C7"]
plan_doc: "/Users/hangli/projects/llm-rankers/EMNLP_EXPERIMENT_PLAN.md"
origin_skill: manual-backfill
created_at: 2026-05-05T22:15:00+10:00
updated_at: 2026-05-05T22:15:00+10:00
---

v8 pool=100 stability rows are grouped by the existing `top{N}` metadata. The CSV remains one combined table across pools; LaTeX exports are emitted per pool as `stability_summary_top50.tex`, `stability_summary_top100.tex`, etc., with `stability_summary.tex` retained as a top50 alias for backward compatibility.
