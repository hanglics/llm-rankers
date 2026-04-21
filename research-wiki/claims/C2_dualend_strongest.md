---
type: claim
node_id: claim:C2
statement: "On TREC DL19/DL20, the DualEnd family achieves the highest NDCG@10 in 14 of 18 model-dataset configurations (all 12 Qwen configs; 2 of 6 Flan-T5 configs). Out-of-domain generalization (BEIR) is in progress; this claim is TREC-DL-scoped until exp:beir_generalization and exp:generalization_weakness land."
status: supported
evidence_strength: medium
origin_skill: manual-backfill
created_at: 2026-04-20T10:35:00+10:00
updated_at: 2026-04-20T10:35:00+10:00
---

# Claim

On TREC DL19 and DL20, across 9 models and 2 datasets, the DualEnd family (DE-Cocktail, DE-Selection) is the top family in 14/18 configurations.

## Evidence (NDCG@10, DL19)

| Model | TD best | DualEnd best | Δ |
|-------|--------:|-------------:|---:|
| flan-t5-large | 0.6874 (TD-Bubble) | 0.6708 (DE-Cocktail) | −0.0165 |
| flan-t5-xl | 0.6980 (TD-Bubble) | 0.6884 (DE-Cocktail) | −0.0096 |
| flan-t5-xxl | 0.7077 (TD-Bubble) | 0.7137 (DE-Cocktail) | +0.0060 |
| qwen3-4b | 0.6775 (TD-Heap) | **0.7220 (DE-Selection)** | **+0.0446** |
| qwen3-8b | 0.6819 (TD-Heap) | 0.7158 (DE-Selection) | +0.0340 |
| qwen3-14b | 0.7455 (TD-Bubble) | 0.7519 (DE-Cocktail) | +0.0064 |
| qwen3.5-4b | 0.7108 (TD-Bubble) | 0.7161 (DE-Cocktail) | +0.0052 |
| qwen3.5-9b | 0.7349 (TD-Bubble) | 0.7370 (DE-Cocktail) | +0.0021 |
| qwen3.5-27b | 0.7449 (TD-Heap) | 0.7475 (DE-Cocktail) | +0.0026 |

## Caveats (claim:C6)

- Only `qwen3-4b` DL19 survives Bonferroni correction.
- Family mean Δ is small (+0.0058).
- Cost is 5.6×–8.9× wall-clock of TD-Heap.

## Supporting experiments

- exp:main_de_cocktail
- exp:main_de_selection
- exp:analysis_significance_tests

## Connections

<!-- AUTO-GENERATED from graph/edges.jsonl — do not edit manually -->
