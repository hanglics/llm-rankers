---
type: experiment
node_id: exp:maxdoc_dualend_pending
title: "Pending — max-context-window DualEnd (SUPERSEDED by idea:007 / IDEA_007.md)"
status: superseded
config:
  idea: "If Qwen can fit many passages per prompt in its context window, run DualEnd over the whole pool and extract best+worst per call."
  scope: "Qwen3 / Qwen3.5 only (T5 context too small)."
  source_artifacts: ["Need_to_Run.txt (lines 5-6)", "/Users/hangli/projects/llm-rankers/IDEA_007.md"]
tests: ["idea:007"]
superseded_by: "idea:007 (MaxContext DualEnd); see IDEA_007.md for the full plan and exp:maxcontext_dualend_{pool_sweep, pl_sweep, order_pilot, baselines} for the run matrix"
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:55:00+10:00
---

# Summary

Superseded on 2026-04-20 by the formal idea:007 plan at `/Users/hangli/projects/llm-rankers/IDEA_007.md`. The original idea — "fit as many passages as the model allows in a single DualEnd prompt" — is implemented as the `maxcontext_dualend` direction with double-ended selection sort, numeric labels 1..N, predeclared matched-`hits` baselines at {10, 30, 50}, and staged 5-phase execution (sanity → order-pilot → matched-hits regression → Study A + baselines → Study B).

This node is kept for traceability; do not launch anything against it directly. Use the superseding exp pages:

- [exp:maxcontext_dualend_pool_sweep](maxcontext_dualend_pool_sweep.md) — Study A.
- [exp:maxcontext_dualend_pl_sweep](maxcontext_dualend_pl_sweep.md) — Study B + dualend-nc3 control arm.
- [exp:maxcontext_dualend_order_pilot](maxcontext_dualend_order_pilot.md) — Study C launch gate.
- [exp:maxcontext_dualend_baselines](maxcontext_dualend_baselines.md) — matched-`hits` predeclared baselines.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
