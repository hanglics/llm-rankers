---
status: planning
refines: IDEA_007.md
target_gaps: G1, G4, G5
target_venue: EMNLP 2026 short paper
---

# IDEA_008 — MaxContext multi-family extension for EMNLP 2026

## 1. Context

IDEA_007 §1 defines MaxContext as a Qwen-generation experiment: fit the whole rerank pool into one prompt and elicit the best document, worst document, or both over the live pool. That Qwen-only scope is deliberate for IDEA_007 and remains authoritative there.

The extension target is the single-family confound. If MaxContext works only on Qwen3 / Qwen3.5, the evidence cannot cleanly separate the algorithm from Qwen-specific chat-template behavior, thinking controls, tokenizer behavior, parser habits, or long-context attention. IDEA_008 keeps the MaxContext algorithm fixed and widens only the model-family admission gate and operator matrix needed for an EMNLP 2026 short-paper replication across Qwen3.5, Llama-3.1, and Ministral-3.

## 2. Algorithm

The algorithm is unchanged from IDEA_007 §2. This spec cross-references IDEA_007 for the authoritative pseudocode and invariants.

- MaxContext TopDown asks for the best document from the whole live pool.
- MaxContext BottomUp asks for the worst document from the whole live pool.
- MaxContext DualEnd asks for best and worst in the same whole-pool call.
- Numeric 1-based labels, strict no-truncation, and strict no-parse-fallback remain the MaxContext defaults.

The EMNLP paper method axis is the canonical seven-method list in `research-wiki/EMNLP_PAPER_PLAN.md`: standard TopDown heap/bubble, standard BottomUp heap/bubble, MaxContext TopDown, MaxContext BottomUp, and MaxContext DualEnd. Standard DualEnd remains part of IDEA_007's baseline history, not a paper method for IDEA_008.

## 3. Code Changes

The line-level implementation is recorded in the v7 plan §A and summarized in `IDEA_008_IMPLEMENTATION_PLAN.md`; this spec does not duplicate that audit trail.

Summary:

- `llmrankers/setwise.py` splits model-family constants while preserving Qwen2 behavior, adds Mistral contingencies to the causal family gate, gates `trust_remote_code`, gates Qwen thinking kwargs, and centralizes generation budgets.
- `llmrankers/setwise_extended.py` widens `MAXCONTEXT_ALLOWED_MODEL_TYPES`, renames `_early_reject_non_qwen3` to `_early_reject_unsupported_family`, and updates MaxContext invariant messages for Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3.
- `run.py` updates the local-model error message for MaxContext with `--openai_key`.
- Evaluation and launcher scripts add BEIR qrels mapping and EMNLP-specific submission, evaluation, smoke, and stability wrappers.
- Analysis adds cross-model stability and EMNLP MaxContext-only position-bias tracks.

## 4. Context Fit

At `pool_size=100`, `passage_length=512`, and `query_length=128`, the rendered prompt is about 55-60K input tokens before output reserve for the required EMNLP families. The existing runtime tokenization preflight remains authoritative; the table below is the admission-level expectation before Phase A `AutoConfig` verification.

| Family      | Required models                 |                                     Native context assumption | Fit at pool=100, pl=512 |
|-------------|---------------------------------|--------------------------------------------------------------:|-------------------------|
| Qwen3       | Optional Qwen3 Phase D/E models |                                                           32K | not used in v8          |
| Qwen3.5     | 0.8B, 2B, 4B, 9B, 27B           |              approximately 262K per IDEA_007 §4 / model cards | gated by Phase A/probe  |
| Llama-3.1   | Meta-Llama-3.1-8B-Instruct      |                                                          128K | gated by Phase A/probe  |
| Ministral-3 | 3B, 8B, 14B Instruct 2512       | assume 128K upper estimate pending Phase A `AutoConfig` probe | gated by Phase A/probe  |

If Phase A reports a shorter effective context, the offending model/pool pair is blocked or dropped to the next lower pool size before Phase B.

## 5. Experiments

The operator command sheet is `EMNLP_EXPERIMENT_PLAN.md`.

Required phases:

- Phase A: 42 smoke cells (`pool_size ∈ {50,100}`).
- Phase B: 3024 main-matrix cells (`pool_size ∈ {10,20,30,40,50,100}`).
- Phase C: 1260 required stability cells on the scientific seven-method axis; 1980 stability-layout submissions using the default 11-block layout.
- Phase C′: 35 IDEA_007 byte-equality recheck cells via `submit_max_context_jobs.sh --idea007-only`.

Optional phases:

- Phase D: 1680 optional Qwen3 main cells.
- Phase E: 350 optional Qwen3-8B stability cells.

## 6. Risks

- Model-specific position bias may change which MaxContext variant is useful by family.
- Family-specific parser drift may surface numeric-output patterns absent from Qwen logs.
- Mistral/Ministral attention quirks may reduce effective context or require loader-specific handling.
- Refusal pattern variance may affect the strict parser and single-extreme fallback telemetry.
- GPU memory ceilings at the 27B class may force reduced context, lower precision, or pool-size drops.

## 7. Out of Scope

- Flan-T5 for MaxContext.
- `--scoring likelihood` for any MaxContext direction.
- `pool_size > 100`; pool=100 is in scope only for Qwen3.5 / Llama-3.1 / Ministral-3 required EMNLP Phase A/B/C runs with native >=128K context and Phase A/probe fit confirmation.
- Batched / multi-pass fitting.
- Letter alphabets beyond A-W.
- `bias_aware` / `samecall_regularized` derivatives of MaxContext.
- Auto-tuning `pool_size`.
- Any refactor or behavior change to existing non-MaxContext ranker classes.
- Ministral-7B, Nemo, and Small; use Ministral-3 only.
- Llama-2 family.
- Qwen2 family for MaxContext; Qwen2 remains allowed only for standard methods.

## 8. Audit Trail

- **v2 (2026-05-06):** v8 extends required EMNLP Phase A/B/C to include pool=100 for Qwen3.5 / Llama-3.1 / Ministral-3; Phase C′ and optional Qwen3 phases remain at the v7 five-pool sweep.
- **v3 (2026-05-06):** standard TopDown/BottomUp launcher consolidation applied; Phase C stability submissions use the default 11-block layout, while Phase C′ keeps the 35-cell gate through `--idea007-only`.
- **v1 (2026-05-05):** READY_TO_EXECUTE post 6-round Codex audit on the EMNLP plan.
