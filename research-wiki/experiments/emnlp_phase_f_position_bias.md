# exp:emnlp_phase_f_position_bias

## Purpose

Measure position dependence in the EMNLP MaxContext methods by comparing the current forward BM25 ordering against per-comparison reverse and shuffle controls.

## Scope

- Methods: `maxcontext_topdown`, `maxcontext_bottomup`, `maxcontext_dualend`
- Models: `Qwen/Qwen3.5-9B`, `meta-llama/Meta-Llama-3.1-8B-Instruct`, `mistralai/Ministral-3-8B-Instruct-2512`
- Datasets: `dl19`, `dl20`, `beir-dbpedia`, `beir-fiqa`
- Pools: 10, 20, 30, 40, 50, 100
- Conditions: forward from Phase B, plus `--reverse` and fixed-seed-929 `--shuffle`

Required new jobs: 432. Optional stability adds 1080 MaxContext-only submissions on DL19 across 10 reps.

## Output Layout

- Forward: `poolNN/`
- Reverse: `poolNN_reverse/`
- Shuffle: `poolNN_shuffle/`

The stability layout mirrors this with `topNN_reverse/` and `topNN_shuffle/`.

## Analysis

`analysis/position_bias_emnlp.py` parses the condition suffix, writes condition-specific position-frequency summaries, and emits paired forward-vs-reverse / forward-vs-shuffle nDCG@10 deltas when matching `.eval` files are present.

## Constraints

Heap/Bubble methods are out of scope. `--shuffle` is not the legacy `--shuffle_ranking`; it shuffles the remaining MaxContext pool before each comparison with fixed seed 929.
