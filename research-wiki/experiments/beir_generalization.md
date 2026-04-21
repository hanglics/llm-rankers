---
type: experiment
node_id: exp:beir_generalization
title: "BEIR generalization — flan-t5-xl / Qwen3-8B / Qwen3.5-9B across 6 domains"
status: partial
config:
  domains: ["dbpedia-entity", "nfcorpus", "scifact", "trec-covid", "webis-touche2020/v2", "fiqa"]
  methods: ["topdown_bubblesort", "bottomup_bubblesort", "dualend_bubblesort", "bidirectional_rrf"]
  hits: 100
  num_child: 3
  k: 10
  alpha: 0.7
  passage_length: "128 (T5) / 512 (Qwen)"
  scoring: generation
  submitted:
    flan_t5_xl: "all 6 domains × 4 methods"
    qwen3_8b: "5 of 6 domains × 4 methods (fiqa not submitted)"
    qwen3_5_9b: "0 of 6 domains (all not submitted)"
tests: ["idea:001", "idea:002", "idea:003", "claim:C2", "claim:C10"]
results_dir: "results/beir/"
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Summary

Out-of-domain generalization of the 4 main method families. Flan-T5-xl is complete; Qwen3-8B is 5/6 complete; Qwen3.5-9B is fully pending. NDCG@10 aggregation and cross-domain analysis will inform whether the TREC DL DualEnd pattern holds under distribution shift.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
