---
type: experiment
node_id: exp:selective_dualend_qwen_pending
title: "Selective DualEnd — Qwen3-8B & Qwen3.5-9B DL19/DL20 (not submitted)"
status: not_submitted
config:
  models: ["Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B"]
  datasets: ["dl19", "dl20"]
  direction: selective_dualend
  method: bubblesort
  gate_strategies: ["hybrid", "shortlist", "uncertain"]
  scoring: ["generation", "likelihood"]
  passage_length: 512
tests: ["idea:004"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Status

24 runs pending (2 models × 2 datasets × 3 gates × 2 scorings). Full command list in `Need_to_Run.txt`.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
