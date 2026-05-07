---
type: experiment
node_id: exp:generalization_weakness
title: "Analysis — generalization weakness (BEIR coverage incomplete; risk to claim:C2)"
status: active
config:
  risk_domain: "BEIR {dbpedia-entity, nfcorpus, scifact, trec-covid, webis-touche2020/v2, fiqa}"
  coverage_status:
    flan_t5_xl: "6/6 complete"
    qwen3_8b: "5/6 complete (fiqa pending)"
    qwen3_5_9b: "0/6 (all pending)"
  source_artifacts: ["../NARRATIVE.md (lines 240+)", "Need_to_Run.txt"]
tests: ["claim:C2", "claim:C10"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:40:00+10:00
---

# Summary

[`NARRATIVE.md`](../NARRATIVE.md) explicitly calls out that the DualEnd pattern is validated only on TREC DL (43 / 54 queries each). The generalization story therefore rests on BEIR runs that are not yet complete for the full model set.

**Concrete risk:** if DualEnd fails to replicate on BEIR for Qwen3-8B/Qwen3.5-9B, claim:C2 (DualEnd strongest overall) becomes TREC-DL-specific and claim:C10 (ICTIR-first framing) may need to become even more conservative.

## Why this is a wiki node, not just a pending-run entry

Separates the *experiment sweep* (`exp:beir_generalization`) from the *analytical risk* it bears on the wiki's own claims. A round-2 auditor reading claim:C2 should see this linked back and understand that the headline number is provisional out-of-domain.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
