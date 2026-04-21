---
type: experiment
node_id: exp:query_difficulty_stratification
title: "Analysis — query-difficulty terciles (BM25-based) × DualEnd Δ"
status: partial
config:
  difficulty_bucketing: "BM25 NDCG@10 terciles per dataset"
  models_measured: "subset of main-sweep models"
  source_artifacts: ["research_pipeline_setwise/FINDINGS.md (lines 149-154)"]
tests: ["claim:C2"]
origin_skill: manual-backfill
created_at: 2026-04-20T10:40:00+10:00
updated_at: 2026-04-20T10:40:00+10:00
---

# Summary

Stratifies per-query DualEnd vs TopDown delta by query difficulty (easy / medium / hard terciles computed on BM25 NDCG@10 per dataset). FINDINGS.md reports the pattern "DualEnd helps most on medium-difficulty queries for Qwen; model-dependent for T5."

This page exists to hold the numbers when they are regenerated in the consolidation pass — currently only in prose in FINDINGS.md.

## Status

Consolidation pending. Pattern documented narratively; no per-tercile NDCG table in the wiki yet.

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
