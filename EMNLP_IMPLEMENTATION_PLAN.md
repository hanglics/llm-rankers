---
status: planning
refines: IDEA_007_IMPLEMENTATION_PLAN.md
target_gaps: G1, G4, G5
target_venue: EMNLP 2026 short paper
---

# IDEA_008 — Implementation Plan

> **Source spec:** `IDEA_008_maxcontext_multi_family.md`.
> **Prime constraint:** IDEA_007's default paths remain byte-identical unless explicitly running the audited EMNLP extension gates.
> **Implementation stance:** extend, do not refactor.

## 1. Strategy: Extend, Do Not Refactor

IDEA_008 follows IDEA_007 §1's conservative stance. The core MaxContext implementation remains intact; this plan widens allowlists, adds family-specific admission checks, and adds EMNLP launch/evaluation/analysis files around the existing algorithm.

The core invariants remain:

- `pool_size == hits == ranker.k`.
- `scoring == generation`.
- `num_permutation == 1`.
- numeric labels for MaxContext.
- `strict_no_truncation=True`.
- `strict_no_parse_fallback=True` for MaxContext strict parse paths.

## 2. Non-Goals

- No behavior change to standard methods on Qwen3, Qwen3.5, or T5.
- No behavior change to standard methods from the allowlist split; only Llama-3.1 and Ministral-3 admission is added.
- No change to IDEA_007 phase numbering or `MAX_CONTEXT_EXPERIMENT_PLAN.md` model matrix.
- No change to existing `experiments/run_*.sh` launchers.
- No prompt tuning, decoding sampling, or retry policy for Llama / Ministral unless Phase A exposes a blocker.
- No GPU jobs or HF Hub probing in this documentation/code continuation pass.

## 3. File-by-File Changes

The authoritative line-level audit is the v7 plan §A. This implementation plan records the file surface and points to the corresponding plan sections rather than duplicating each patch.

- `llmrankers/setwise.py` — plan §A.1 plus multimodal loader v3: split causal/Qwen/trust/thinking constants; preserve Qwen2/Qwen3 generation-budget and chat-template behavior; add `_generation_budget()`; route `mistral3`, `qwen3_5`, and `qwen3_5_moe` through `MULTIMODAL_MODEL_TYPES` using `AutoProcessor` + `AutoModelForImageTextToText`.
- `llmrankers/setwise_extended.py` — plan §A.2: widen `MAXCONTEXT_ALLOWED_MODEL_TYPES`; rename `_early_reject_unsupported_family`; update invariant messages; preserve strict numeric MaxContext helpers.
- `run.py` — plan §A.3: update the MaxContext local-model `--openai_key` rejection message.
- `scripts/check_maxcontext_invariants.py` — plan §A.4: add family-admission, trust, thinking-kwargs, generation-budget, and early-reject tests.
- `experiments/eval_all.sh` — plan §A.5: add BEIR qrels mapping and `LEVEL` handling.
- Existing launcher scripts — plan §A.6 plus launcher-consolidation v3: `run_topdown_bigram.sh` gains `ANALYSIS_LOG_DIR`; `run_bottomup_bigram.sh` is added.
- `submit_max_context_jobs.sh`, `eval_max_context_jobs.sh`, `analysis/repeated_run_stability.py` — plan §A.7 / §B.6 / §B.8 plus launcher-consolidation v3: default 11-block stability layout, `--idea007-only` 35-cell gate, and BottomUp ws-3/ws-ps metadata support.
- `submit_emnlp_jobs.sh`, `eval_emnlp_jobs.sh`, `submit_emnlp_stability_jobs.sh`, `scripts/smoke_emnlp_models.sh` — plan §B.4-§B.7: new EMNLP operators.
- `analysis/cross_model_stability.py`, `analysis/position_bias_emnlp.py`, `analysis/position_bias.py` — plan §B.8 / §E.12: cross-family stability and MaxContext-only position-bias analysis.

## 4. CLI Behavior Contract

This extends IDEA_007 §4. All existing directions retain their behavior.

For `maxcontext_topdown`, `maxcontext_bottomup`, and `maxcontext_dualend`:

- Qwen3 and Qwen3.5 remain admitted.
- Llama-3.1 and Ministral-3 are newly admitted.
- Qwen2 remains excluded from MaxContext.
- T5 and likelihood scoring remain excluded from MaxContext.
- `--hits == --k`, `--scoring generation`, and `--num_permutation 1` remain required.
- `--openai_key` remains incompatible with MaxContext because the method requires local tokenization, local context-fit checks, and local comparison logging.

## 5. Label-Scheme Safety Net

Unchanged from IDEA_007 §5. MaxContext uses numeric 1-based labels on the ranker instance; existing letter-scheme rankers retain the class-level A-W labels. Mixed-scheme position-bias logs are still rejected by analysis unless analyzed separately.

## 6. Preflight Context-Fit

Unchanged from IDEA_007 §6. The runtime check renders the actual prompt through the actual tokenizer/chat-template path and verifies it against `max_input_tokens - reserved_output_tokens`. Arithmetic estimates are advisory only.

For IDEA_008 v8, Phase A additionally records each required model's `config.model_type`, context fields, and effective fit at `pool=50` and `pool=100`, `passage_length=512`. The separate BEIR pool=100 probe checks real BEIR passages before Phase B BEIR pool=100 launches.

## 7. Launcher Integration

Main-matrix submission and evaluation use `submit_emnlp_jobs.sh` and `eval_emnlp_jobs.sh` (plan §B.4-§B.5). These scripts invoke `python3 run.py` directly and write comparison JSONL next to the `.txt`, `.eval`, and `.log` outputs.

Stability submission uses `submit_emnlp_stability_jobs.sh` (plan §B.6), which wraps the default 11-block `submit_max_context_jobs.sh` layout. Phase C reports 1260 scientific seven-method cells and submits 1980 stability-layout jobs. Phase C′ passes `--idea007-only` directly to `submit_max_context_jobs.sh` to preserve the 35-cell IDEA_007 byte-equality gate.

Phase A smoke uses `scripts/smoke_emnlp_models.sh` (plan §B.7).

Phase F adds MaxContext-only position-bias controls. `--reverse` and `--shuffle` are valid only for `maxcontext_topdown`, `maxcontext_bottomup`, and `maxcontext_dualend`; Heap/Bubble methods remain BM25-ordered. `--shuffle` uses fixed seed 929 and shuffles the remaining pool before each LLM comparison. This is distinct from legacy `--shuffle_ranking`, which permutes the initial ranking once.

The v8 pool=100 extension adds `--pool-sizes` to the max-context submission/eval wrappers, extends EMNLP `--pool-size all` to `{10,20,30,40,50,100}`, and keeps omitted `--pool-size` at the existing single pool=50 default.

## 8. Ordering of Edits

1. Code.
2. Shell scripts.
3. Root docs.
4. Wiki nodes.
5. Existing wiki index/log/gap/graph updates.

## 9. Regression Gate

IDEA_008 records a **65-cell prime-constraint gate**:

| Component                          | Count | Meaning                                                                                                                                                                   |
|------------------------------------|------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| IDEA_007 goldens                   |     9 | Existing `.eval` goldens from IDEA_007_IMPLEMENTATION_PLAN.md §9.1; they span flan-t5-large, qwen3-8b, and qwen3.5-9b across multiple methods. They are not all Qwen3-4B. |
| Phase C′ canonical stability diffs |    35 | Post-merge Qwen3-4B DL19 default-layout reruns diffed against `results/maxcontext_dualend/qwen3-4b-dl19/stability-test-runs/test_run_v1/`.                                |
| Phase A smoke-as-goldens           |    21 | The v7 pool=50 EMNLP Phase A smoke cells, frozen only after the first successful Phase A run. v8 pool=100 smoke cells are supplemental and stay outside this gate.        |

Phase C′ creates no new smoke-as-golden files. It is only a byte-equality diff against the existing canonical stability v1. Only the 21 v7 pool=50 Phase A cells become prime smoke-as-goldens.

v8 matrix counts:

| Phase                      | Count | Notes                                                                                                             |
|----------------------------|------:|-------------------------------------------------------------------------------------------------------------------|
| Phase A smoke              |    42 | 3 models × 7 methods × dl19 × pools {50,100}.                                                                     |
| Phase B required main      |  3024 | 7 methods × 9 models × 8 datasets × 6 pools.                                                                      |
| Phase C required stability |  1260 | Scientific seven-method count; wrapper submits 1980 stability-layout jobs with default 11-block layout.             |
| Phase C′ prime recheck     |    35 | Unchanged five-pool Qwen3-4B byte-equality control via `--idea007-only`.                                            |
| Required total             |  4361 | A + B + C + C′.                                                                                                   |

The smoke gate is method-aware: all seven methods require full `.txt` coverage, top-10 permutation validity, positive NDCG@10, and clean logs; only the three MaxContext methods require zero parse-fallback and numeric out-of-range counters.

## 10. Deliverables

New root docs:

- `IDEA_008_maxcontext_multi_family.md`
- `IDEA_008_IMPLEMENTATION_PLAN.md`
- `EMNLP_EXPERIMENT_PLAN.md`

New operator and analysis files:

- `submit_emnlp_jobs.sh`
- `eval_emnlp_jobs.sh`
- `submit_emnlp_stability_jobs.sh`
- `scripts/smoke_emnlp_models.sh`
- `experiments/run_bottomup_bigram.sh`
- `analysis/cross_model_stability.py`
- `analysis/position_bias_emnlp.py`
- `scripts/probe_beir_pool100_fit.py`

Modified implementation/support files:

- `llmrankers/setwise.py`
- `llmrankers/setwise_extended.py`
- `run.py`
- `scripts/check_maxcontext_invariants.py`
- `experiments/eval_all.sh`
- `submit_max_context_jobs.sh`
- `eval_max_context_jobs.sh`
- `analysis/repeated_run_stability.py`
- `analysis/position_bias.py`

New wiki files:

- `research-wiki/EMNLP_PAPER_PLAN.md`
- `research-wiki/ideas/idea_008_maxcontext_multi_family.md`
- `research-wiki/experiments/emnlp_phase_a_smoke.md`
- `research-wiki/experiments/emnlp_phase_b_main_matrix.md`
- `research-wiki/experiments/emnlp_phase_c_stability.md`
- `research-wiki/experiments/emnlp_phase_c_prime_constraint_recheck.md`
- `research-wiki/experiments/emnlp_phase_d_qwen3_optional.md`
- `research-wiki/experiments/emnlp_phase_e_qwen3_stability_optional.md`
- `research-wiki/experiments/cross_model_stability.md`
- `research-wiki/papers/llama3_1_paper.md`
- `research-wiki/papers/mistral3_paper.md`
- `research-wiki/papers/qwen3_paper.md`
- `research-wiki/papers/qwen3_5_paper.md`

Modified docs/wiki files:

- `MAX_CONTEXT_EXPERIMENT_PLAN.md`
- `IDEA_007.md`
- `IDEA_007_IMPLEMENTATION_PLAN.md`
- `research-wiki/index.md`
- `research-wiki/log.md`
- `research-wiki/gap_map.md`
- `research-wiki/graph/edges.jsonl`

## 11. Out of Scope

- Flan-T5 for MaxContext.
- `--scoring likelihood` for any MaxContext direction.
- `pool_size > 100`; pool=100 is now in scope only for required Qwen3.5 / Llama-3.1 / Ministral-3 EMNLP Phase A/B/C runs after context-fit gates pass.
- Batched / multi-pass fitting.
- Letter alphabets beyond A-W.
- `bias_aware` / `samecall_regularized` derivatives of MaxContext.
- Auto-tuning `pool_size`.
- Any refactor or behavior change to existing non-MaxContext ranker classes.
- Ministral-7B, Nemo, and Small; use Ministral-3 only.
- Llama-2 family.
- Qwen2 family for MaxContext; Qwen2 remains allowed only for standard methods.
- Bare inner Qwen3.5 text configs (`qwen3_5_text`, `qwen3_5_moe_text`) are not supported as top-level model configs in this refactor; the dispatcher intentionally raises `NotImplementedError` if encountered.

## 12. Audit Trail

- **v3 multimodal loader (2026-05-06):** Phase 3a adds the authoritative `MULTIMODAL_MODEL_TYPES={mistral3,qwen3_5,qwen3_5_moe}` dispatch, `ProcessorTokenizerAdapter`, and a text-only multimodal loader for Mistral 3 / Qwen 3.5 vision-language configs. `qwen3` / `qwen3_moe` remain on the existing causal path for IDEA_007 byte-equality; `qwen3_5_text` / `qwen3_5_moe_text` inner configs are explicitly unsupported.
- **v4 position-bias controls (2026-05-07):** added MaxContext-only `--reverse` and fixed-seed-929 per-comparison `--shuffle` flags, suffixed output leaves (`poolNN_reverse`, `poolNN_shuffle`, `topNN_reverse`, `topNN_shuffle`), and Phase F representative position-bias experiments. Default-off Qwen3 paths and Heap/Bubble launchers remain unchanged.
- **v3 (2026-05-06):** launcher consolidation applied: standard EMNLP TopDown/BottomUp methods use bigram launchers, `run_bottomup_bigram.sh` added, stability default layout expands to 1980 submissions, and Phase C′ remains 35 cells via `--idea007-only`.
- **v2 (2026-05-06):** v8 pool=100 delta applied: Phase A 42, Phase B 3024, Phase C 1260 scientific / 1620 submitted, required total 4361; IDEA_007 65-cell prime gate unchanged.
- **v1 (2026-05-05):** READY_TO_EXECUTE post 6-round Codex audit.
