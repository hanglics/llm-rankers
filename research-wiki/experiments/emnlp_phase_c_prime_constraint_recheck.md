---
type: experiment
node_id: exp:emnlp_phase_c_prime_constraint_recheck
title: "EMNLP Phase C′ — IDEA_007 byte-equality recheck"
status: not_submitted
config:
  model: "Qwen/Qwen3-4B"
  dataset: "dl19"
  layout: "IDEA_007 35-cell stability layout via submit_max_context_jobs.sh --idea007-only"
  cells: 35
tests: ["idea:007", "idea:008"]
plan_doc: "/Users/hangli/projects/llm-rankers/EMNLP_EXPERIMENT_PLAN.md"
origin_skill: manual-backfill
created_at: 2026-05-05T22:15:00+10:00
updated_at: 2026-05-05T22:15:00+10:00
---

Phase C′ uses `submit_max_context_jobs.sh --idea007-only` because the default stability layout now emits 55 jobs. The flag suppresses the four default-on BottomUp ws-3/ws-ps blocks and preserves the historical 35-cell byte-equality diff target.
