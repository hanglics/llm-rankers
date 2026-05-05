---
type: plan
node_id: plan:emnlp_2026_maxcontext
title: "EMNLP 2026 short-paper plan — MaxContext multi-family extension"
status: active
tests: ["idea:008", "claim:C7"]
origin_skill: manual-backfill
created_at: 2026-05-05T22:15:00+10:00
updated_at: 2026-05-05T22:15:00+10:00
---

# EMNLP Paper Plan

## §1 Introduction

- IDEA_007 established MaxContext on Qwen3/Qwen3.5.
- The single-family setup leaves an external-validity concern for claim:C7.
- Contribution: replicate the MaxContext family across Qwen3.5, Llama-3.1, and Ministral-3.

## §2 Methods

Canonical EMNLP methods:

| # | method_tag            | Description                                 |
|---|-----------------------|---------------------------------------------|
| 1 | `topdown_bubblesort`  | Standard TopDown setwise + bubblesort       |
| 2 | `topdown_heapsort`    | Standard TopDown setwise + heapsort         |
| 3 | `bottomup_bubblesort` | Standard BottomUp setwise + bubblesort      |
| 4 | `bottomup_heapsort`   | Standard BottomUp setwise + heapsort        |
| 5 | `maxcontext_topdown`  | MaxContext TopDown (best-only whole-pool)   |
| 6 | `maxcontext_bottomup` | MaxContext BottomUp (worst-only whole-pool) |
| 7 | `maxcontext_dualend`  | MaxContext DualEnd (best+worst whole-pool)  |

Standard DualEnd (`dualend_bubblesort`, `dualend_selection`) is out of EMNLP scope.

## §3 Experiments

- Main matrix: 7 methods × 9 models × 8 datasets × 6 pool sizes = 3024 runs.
- Stability: 7 methods × 3 representative models × dl19 × 6 pool sizes × 10 reps = 1260 scientific cells.
- Datasets: dl19, dl20, BEIR dbpedia-entity, nfcorpus, scifact, trec-covid, webis-touche2020, fiqa.
- Models: five Qwen3.5 models, one Llama-3.1 model, and three Ministral-3 models.
- Significance: paired bootstrap on per-query NDCG@10; Bonferroni across seven methods.

## §4 Results

- Main effectiveness table, including a pool=100 row/column for the required large-context families.
- Stability table by family.
- Position-bias chart, one representative per family.
- Pareto plot extension to all families.

## §5 Discussion

- Failure modes by family.
- Limitations: Qwen3 optional, tokenizer-specific parser sensitivity, and order-robustness not replicated beyond IDEA_007.

## §6 Conclusion

MaxContext's whole-pool prompting is evaluated as a model-family-general reranking pattern rather than a Qwen-only artifact.
