---
type: experiment
node_id: exp:selective_dualend_flan_t5_xl
title: "Selective DualEnd — flan-t5-xl DL19/DL20 (partial)"
status: partial
config:
  model: "google/flan-t5-xl"
  datasets: ["dl19", "dl20"]
  direction: selective_dualend
  method: bubblesort
  uncertainty_percentile: 0.15
  hits: 100
  num_child: 3
  k: 10
  passage_length: 128
  submitted_variants:
    - "generation × hybrid × {dl19, dl20}"
    - "likelihood × hybrid × {dl19, dl20}"
    - "generation × shortlist × {dl19, dl20}"
  pending_variants:
    - "likelihood × shortlist × {dl19, dl20}"
    - "generation × uncertain × {dl19, dl20}"
    - "likelihood × uncertain × {dl19, dl20}"
tests: ["idea:004", "claim:C9"]
results_dir: "results/selective-dualend/"
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Status

6 of 12 flan-t5-xl selective-dualend runs complete. NDCG@10 consolidation pending.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
