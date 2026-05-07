---
type: experiment
node_id: exp:maxcontext_dualend_pl_sweep
title: "Study B — MaxContext DualEnd passage-length sweep + DualEnd-nc3 control arm"
status: not_submitted
config:
  direction: ["maxcontext_dualend", "dualend (control arm, num_child=3)"]
  passage_length_grid: [64, 128, 256, 512]
  pool_size: "predeclared from Study A: smallest pool_size within 0.003 NDCG@10 of Study A max"
  scoring: generation
  models: ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B"]
  datasets: ["dl19", "dl20"]
tests: ["idea:007", "claim:C2", "claim:C7"]
plan_doc: "/Users/hangli/projects/llm-rankers/IDEA_007.md"
origin_skill: manual-backfill
created_at: 2026-04-20T10:55:00+10:00
updated_at: 2026-04-20T10:55:00+10:00
---

# Summary

Study B from IDEA_007. 6 Qwens × 4 pls × 2 datasets × 2 arms = **96 runs** (48 treatment + 48 control).

**Treatment arm:** MaxContext DualEnd at predeclared pool_size.
**Control arm:** existing `direction = dualend, num_child = 3`. Codex round 2 confirmed this is the correct primary control (preserves joint best+worst prompting + parser family). TopDown pl-curves are supplementary context only.

**Confound-handling rule:**
- Shorter pl wins in MaxContext but not control → attribute to long-context attention degradation.
- Shorter pl wins in both → attribute to Qwen's general passage-signal / length sensitivity.
- Any other pattern → further investigation before claiming.

**Gate before submission:** Phase 4 (Study A + baselines) must complete and yield a defensible pool_size via the selection rule.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
