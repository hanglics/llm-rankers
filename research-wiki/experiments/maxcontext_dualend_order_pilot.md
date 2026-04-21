---
type: experiment
node_id: exp:maxcontext_dualend_order_pilot
title: "Study C — MaxContext DualEnd order-robustness pilot (LAUNCH GATE)"
status: not_submitted
config:
  direction: maxcontext_dualend
  pool_size: 50
  passage_length: 512
  orderings: ["BM25 forward", "BM25 reversed", "random shuffle (fixed seed)"]
  models: ["Qwen3-4B", "Qwen3.5-9B"]
  datasets: ["dl19", "dl20"]
  role: "launch_gate_not_proof"
tests: ["idea:007"]
plan_doc: "/Users/hangli/projects/llm-rankers/IDEA_007.md"
origin_skill: manual-backfill
created_at: 2026-04-20T10:55:00+10:00
updated_at: 2026-04-20T10:55:00+10:00
---

# Summary

Study C from IDEA_007. Smoke gate, **not** proof of order-robustness. Three orderings across two models × two datasets = **12 runs**.

- Qwen3-4B included as the tightest-context case — hardest for long-prompt reliability.
- Qwen3.5-9B included as a representative mid-size model.

**Gate rule:** if max pairwise NDCG@10 Δ across the three orderings ≤ 0.01 (within typical bootstrap CI for 43-query TREC DL), proceed to Study A and baselines. If > 0.01, MaxContext is order-sensitive at w=50 — escalate before launching the full matrix.

**Escalation options if the gate fails:**
(a) restrict to pool_size=20 where the prompt fits comfortably within attention limits;
(b) pivot to a `bias_aware_dualend`-with-large-windows derivative that exploits controlled orderings;
(c) treat the ordering effect as a finding in itself and re-plan.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
