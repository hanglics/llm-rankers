# Experiment Plan: Directional Asymmetry in LLM-Based Setwise Ranking

**Paper:** "Directional Asymmetry in LLM-Based Setwise Ranking: Evidence from Worst-Only, Joint, and Fused Signals"
**Last Updated:** 2026-04-09
**Status:** Phase 1-4 COMPLETE (all re-runs done after 2026-03-29 bug fixes). Significance testing complete. Phase 5 BEIR execution is delegated to the remote GPU runs; the local priority is the review-driven refinement package, paper-facing analysis, and audit.

---

## Table of Contents

1. [Research Questions](#1-research-questions)
2. [Audit Summary: Paper vs. Code Gaps](#2-audit-summary-paper-vs-code-gaps)
3. [Experiment Matrix](#3-experiment-matrix)
4. [Phase 0: Prerequisites](#4-phase-0-prerequisites)
5. [Phase 1: Main Experiments (Table 1 — Main Results)](#5-phase-1-main-experiments)
6. [Phase 2: Efficiency Comparison (Table 3)](#6-phase-2-efficiency-comparison)
7. [Phase 3: Ablation Studies (Tables 4, 5, 5b)](#7-phase-3-ablation-studies)
8. [Phase 4: Analysis Experiments (Tables 6 + Figures)](#8-phase-4-analysis-experiments)
9. [Phase 5: BEIR Evaluation](#9-phase-5-beir-evaluation)
10. [Script Reference](#10-script-reference)
11. [Results Checklist](#11-results-checklist)
12. [Common Mistakes to Avoid](#12-common-mistakes-to-avoid)
13. [Review-Driven Roadmap](#13-review-driven-roadmap)
14. [Immediate TODO Checklist](#14-immediate-todo-checklist)

---

## 1. Research Questions

The paper addresses three research questions. Every experiment must map to at least one RQ.

| RQ | Question | Key Tables/Figures |
|----|----------|--------------------|
| **RQ1** | Does bottom-up (worst) selection achieve different effectiveness than top-down (best) selection? | Tab 1 (main), Tab 6 (difficulty), Position bias analysis |
| **RQ2** | Does dual-end selection improve effectiveness enough to justify its extra compute cost, and can that quality-cost tradeoff be improved? | Tab 1, Tab 3 (efficiency), Tab 4 (num_child ablation) |
| **RQ3** | Does bidirectional ensemble improve over individual methods? | Tab 1, Tab 5 (alpha ablation), Permutation voting comparison |

---

## 2. Audit Summary: Paper vs. Code Gaps

### Critical Misalignments Found

| Issue | Paper Says | Code/Scripts Say | Impact | Status |
|-------|-----------|-----------------|--------|--------|
| **Models** | Flan-T5-XL/XXL, Vicuna | SLURM defaults to Qwen3-4B | **HIGH** | ✅ Plan updated: use 9 models (3 T5, 3 Qwen3, 3 Qwen3.5) |
| **passage_length** | 128 tokens | Scripts use 512 | **HIGH** | ✅ Plan clarifies: **128 for T5** (matches original paper), 512 for Qwen; ablation added |
| **Likelihood scoring** | Tab 1 has likelihood rows | No script | **HIGH** | ✅ `run_likelihood.sh` created |
| **Permutation voting** | §5.3 compares BiDir vs. p=2 | No script | **HIGH** | ✅ `run_permvote_p2.sh` created |
| **num_child ablation** | Tab 4 tests c∈{2,3,5,7} | Not scripted | **MEDIUM** | ✅ `run_ablation_nc.sh` created |
| **Alpha ablation** | Tab 5 tests α∈{0.3,0.5,0.7,0.9} | Only α=0.7 | **MEDIUM** | ✅ `run_ablation_alpha.sh` created |
| **Passage length ablation** | Not in paper yet | Not scripted | **MEDIUM** | ✅ `run_ablation_passage_length.sh` created; paper update needed |
| **CombSUM fusion** | §5.3 mentions CombSUM | Not scripted | **LOW** | ✅ Included in `run_ablation_alpha.sh` |
| **MAP@100 metric** | §4.1 says "also report MAP@100" | Eval only does NDCG@10 | **LOW** | ✅ `eval_all.sh` includes MAP@100 |
| **BM25 baseline** | Tab 1 has BM25 row | Not computed | **LOW** | ✅ Phase 0 instructions added |
| **Position bias analysis** | §5.4 detailed analysis | No logging code | **HIGH** | ⚠️ Requires code changes (Phase 4) |
| **Query difficulty analysis** | Tab 6 stratified by difficulty | No analysis code | **MEDIUM** | ✅ `analysis/query_difficulty.py` created |
| **BottomUp bubblesort bug** | Top-k should be sorted | Top-k was UNSORTED (n-k passes only) | **CRITICAL** | ✅ Fixed 2026-03-29: now does n-1 passes for full sort |
| **BottomUp heapsort mixed prompts** | BottomUp should only use "worst" | Used "best" prompts for top-k sort phase | **HIGH** | ✅ Fixed 2026-03-29: now uses min-heap extraction throughout |
| **BottomUp compare_worst no logging** | Should log for position bias | `compare_worst` had `pass` instead of logging | **HIGH** | ✅ Fixed 2026-03-29: now logs as `type="worst"` |

### Code Implementation Status ✅

All three ranker classes are implemented and parsing bugs are fixed:
- `BottomUpSetwiseLlmRanker` — heapsort + bubblesort ✅
- `DualEndSetwiseLlmRanker` — cocktail shaker + double-ended selection ✅
- `BidirectionalEnsembleRanker` — RRF + CombSUM + Weighted ✅
- `_clean_generation_output` — handles Qwen thinking tokens, T5/Flan-T5 tokens ✅
- `_parse_single_label` — handles brackets, numbers, refusals ✅
- `_try_parse_dual_output` — handles square-bracketed labels + numeric labels ✅
- `_parse_dual_output` — guaranteed-return fallback with numeric + heuristic parsing ✅
- Bubblesort window-shrinking bug — fixed ✅
- `max_new_tokens` — 512 for Qwen dual-end, 256 for Qwen single-label, 64 for non-Qwen causal ✅
- DualEnd T5 generation — uses likelihood internally (single forward pass, no parsing) ✅
- DualEnd strict single-call enforcement — never falls back to 2 separate calls ✅

---

## 3. Experiment Matrix

### Models

```
Encoder-Decoder (T5 family — support generation + likelihood scoring):
  - google/flan-t5-large       (780M)
  - google/flan-t5-xl          (3B)     ← primary model for paper
  - google/flan-t5-xxl         (11B)

Decoder-Only / Causal (Qwen3 family — generation + likelihood scoring, thinking models):
  - Qwen/Qwen3-4B             (4B)
  - Qwen/Qwen3-8B             (8B)
  - Qwen/Qwen3-14B            (14B)

Hybrid Causal (Qwen3.5 — requires transformers dev build with qwen3_5 support):
  - Qwen/Qwen3.5-4B           (4B, loaded via AutoModelForCausalLM)
  - Qwen/Qwen3.5-9B           (9B, loaded via AutoModelForCausalLM)
  - Qwen/Qwen3.5-27B          (27B, loaded via AutoModelForCausalLM)
```

**Total: 9 models** (3 T5 + 3 Qwen3 + 3 Qwen3.5)

### Datasets

```
TREC Deep Learning (primary — have graded qrels, standard benchmarks):
  - msmarco-passage/trec-dl-2019/judged   (43 queries, qrels: dl19-passage)
  - msmarco-passage/trec-dl-2020/judged   (54 queries, qrels: dl20-passage)

BEIR (extended — diverse domains):
  - beir/dbpedia-entity/test     (entity-centric)
  - beir/nfcorpus/test           (medical)
  - beir/scifact/test            (scientific claims)
  - beir/trec-covid              (biomedical)
  - beir/webis-touche2020/v2     (arguments)
  - beir/hotpotqa/test           (multi-hop)
  - beir/quora/test              (duplicate questions)
  - beir/fever/test              (fact verification)
```

### BM25 Run Files (all exist)

```
runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt
runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt
runs/bm25/run.beir.bm25-flat.dbpedia-entity.txt
runs/bm25/run.beir.bm25-flat.nfcorpus.txt
runs/bm25/run.beir.bm25-flat.scifact.txt
runs/bm25/run.beir.bm25-flat.trec-covid.txt
runs/bm25/run.beir.bm25-flat.webis-touche2020.txt
runs/bm25/run.beir.bm25-flat.hotpotqa.txt
runs/bm25/run.beir.bm25-flat.quora.txt
runs/bm25/run.beir.bm25-flat.fever.txt
```

### Methods (8 configurations)

```
  1. TopDown-Heap      (topdown, heapsort)        — baseline
  2. TopDown-Bubble    (topdown, bubblesort)       — baseline
  3. BottomUp-Heap     (bottomup, heapsort)        — RQ1
  4. BottomUp-Bubble   (bottomup, bubblesort)      — RQ1
  5. DualEnd-Cocktail  (dualend, bubblesort)       — RQ2
  6. DualEnd-Selection (dualend, selection)         — RQ2
  7. BiDir-RRF         (bidirectional, heapsort, fusion=rrf)     — RQ3
  8. BiDir-Weighted    (bidirectional, heapsort, fusion=weighted) — RQ3
```

### Default Parameters

```
  - num_child = 3              ← matches original paper (c=3 default)
  - k = 10
  - hits = 100
  - passage_length = 128      ← for Flan-T5 (matches original paper; at 512-token limit)
                    = 512      ← for Qwen3/Qwen3.5 (32k+ context)
  - query_length = 128         (default in run.py; TREC DL queries are short so this rarely matters)
  - scoring = generation       (unless likelihood experiment for T5 models)
  - num_permutation = 1        (unless permutation voting experiment)
```

### Recommended Passage Length by Model

**Matching the original paper (arXiv:2310.09497):**

The original paper uses `num_child=3` with `passage_length=128` for Flan-T5. Both heapsort and bubblesort present `num_child+1 = 4` documents per comparison. With 4×128 = 512 passage tokens plus ~65 tokens overhead (short TREC DL queries), this is at or slightly above the T5 512-token limit. The original paper accepts this marginal truncation — it's part of their methodology.

Their `num_child` ablation (Table 5) calibrates PL to stay near the 512 limit:
```
c=3 → PL=128 → (c+1)×PL = 4×128 = 512
c=5 → PL=85  → 6×85  = 510
c=7 → PL=60  → 8×60  = 480
c=9 → PL=45  → 10×45 = 450
```

| Model Family | Context Limit | `passage_length` | Notes |
|---|---|---|---|
| Flan-T5-large/xl/xxl | 512 tokens | **128** | Matches original paper; marginal truncation expected for some queries |
| Qwen3-4B/8B/14B | 32k+ tokens | **512** | Plenty of room |
| Qwen3.5-4B | 32k+ tokens | **512** | Same as Qwen3 |

**Note**: You may see `"Warning: prompt length NNN exceeds model limit 512"` for T5 models. This is expected — the original paper experiences the same marginal truncation. The code handles it gracefully via `_tokenize_inputs()`. For BEIR datasets with longer queries, consider reducing to `passage_length=100`.

### Experiment Scale Summary

| Scope | Models | Datasets | Methods | Runs |
|-------|--------|----------|---------|------|
| **Phase 1: Core (DL19+DL20)** | 9 models | 2 datasets | 8 methods | 144 |
| **Phase 1B: Likelihood (T5 only)** | 3 T5 models | 2 datasets | 2 methods | 12 |
| **Phase 2: Baselines** | 2-3 models | 2 datasets | 1 method | 4-6 |
| **Phase 3: Ablations** | 1-2 models | 1 dataset | varies | ~20 |
| **Phase 5: BEIR** | 2-3 models | 8 datasets | 4-8 methods | 64-192 |
| **Total estimated** | | | | **~200-340** |

---

## 4. Phase 0: Prerequisites

Complete these **before** running any experiments.

### P0.1: Verify BM25 baselines
```bash
# Evaluate BM25 on DL19 and DL20
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt

python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl20-passage \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt

# Also MAP@100
python -m pyserini.eval.trec_eval -c -l 2 -m map_cut.100 dl19-passage \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt

python -m pyserini.eval.trec_eval -c -l 2 -m map_cut.100 dl20-passage \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt
```

**Expected BM25 NDCG@10**: ~0.506 (DL19), ~0.480 (DL20) from Pyserini docs.

Record in `results/bm25_baselines.txt`.

### P0.2: Verify model downloads
```bash
# Ensure models are cached before submitting batch jobs
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('google/flan-t5-xl')"
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('google/flan-t5-xxl')"
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('lmsys/vicuna-7b-v1.5')"
```

### P0.3: Verify passage_length alignment
**CRITICAL**: Match the original paper's parameters exactly for fair comparison. Use `passage_length=128` for Flan-T5 (matches arXiv:2310.09497) and `passage_length=512` for Qwen3/Qwen3.5.

```bash
# Flan-T5 (passage_length=128, matches original paper):
bash experiments/run_extended_setwise_all.sh google/flan-t5-xl \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/flan-t5-xl-dl19 \
    cuda generation 3 10 100 128
#                              ^^^ passage_length=128 (original paper)

# Qwen3 (passage_length=512, plenty of context room):
bash experiments/run_extended_setwise_all.sh Qwen/Qwen3-4B \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/qwen3-4b-dl19 \
    cuda generation 3 10 100 512
#                              ^^^ passage_length=512
```

### P0.4: Verify code works (smoke test)
Run a quick smoke test with 2 queries to catch any runtime errors:
```bash
# Create a 2-query subset for smoke testing
head -200 runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt > /tmp/smoke_test_run.txt

# Test each direction
for DIR in topdown bottomup dualend; do
    python run.py \
        run --model_name_or_path google/flan-t5-xl \
            --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
            --run_path /tmp/smoke_test_run.txt \
            --save_path /tmp/smoke_${DIR}.txt \
            --scoring generation \
            --hits 100 --passage_length 128 \
        setwise --num_child 3 --method heapsort --k 10 --direction ${DIR}
done

# Test bidirectional
python run.py \
    run --model_name_or_path google/flan-t5-xl \
        --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
        --run_path /tmp/smoke_test_run.txt \
        --save_path /tmp/smoke_bidir.txt \
        --scoring generation \
        --hits 100 --passage_length 128 \
    setwise --num_child 3 --method heapsort --k 10 --direction bidirectional --fusion rrf
```

**Check for**: no Python errors, no "Unexpected output" warnings, reasonable NDCG@10.

---

## 5. Phase 1: Main Experiments

**Goal**: Fill Table 1 (main results) in the paper.
**RQs**: All three.
**Priority**: Highest — all other analyses depend on these runs.

### Naming Convention

Results directory: `results/<model_short>-<dataset_short>/`
- `flan-t5-xl-dl19/`, `flan-t5-xl-dl20/`
- `flan-t5-xxl-dl19/`, `flan-t5-xxl-dl20/`
- `vicuna-7b-dl19/`, `vicuna-7b-dl20/`

### Phase 1 Command Template

For **each model × dataset** combination, run all 8 methods using `run_extended_setwise_all.sh`:

```bash
bash experiments/run_extended_setwise_all.sh \
    <MODEL> <DATASET> <RUN_PATH> <OUTPUT_DIR> \
    cuda generation 3 10 100 <PASSAGE_LENGTH> 0.7
```

Or submit individual SLURM jobs in parallel:
```bash
for SCRIPT in run_topdown_heapsort run_topdown_bubblesort \
              run_bottomup_heapsort run_bottomup_bubblesort \
              run_dualend_bubblesort run_dualend_selection \
              run_bidirectional_rrf run_bidirectional_weighted; do
    sbatch experiments/${SCRIPT}.sh \
        <MODEL> <DATASET> <RUN_PATH> <OUTPUT_DIR> \
        cuda generation 3 10 100 <PASSAGE_LENGTH> 0.7
done
```

### Phase 1A: Flan-T5 Family (Encoder-Decoder)

| Model | DL19 OUTPUT_DIR | DL20 OUTPUT_DIR | PL |
|-------|----------------|----------------|-----|
| `google/flan-t5-large` | `results/flan-t5-large-dl19` | `results/flan-t5-large-dl20` | **128** |
| `google/flan-t5-xl` | `results/flan-t5-xl-dl19` | `results/flan-t5-xl-dl20` | **128** |
| `google/flan-t5-xxl` | `results/flan-t5-xxl-dl19` | `results/flan-t5-xxl-dl20` | **128** |

**passage_length=128** for Flan-T5 — matches the original paper (arXiv:2310.09497). With `num_child=3` (4 passages per prompt), this is at the 512-token limit. Marginal truncation is expected and matches the original methodology.

Each: 8 methods × 2 datasets = 16 runs. Total: **48 runs** across 3 models.

**SLURM notes**:
- Flan-T5-large (780M): `--mem=128G`, `--time=04:00:00`
- Flan-T5-xl (3B): `--mem=256G`, `--time=08:00:00`
- Flan-T5-xxl (11B): `--mem=512G`, `--time=20:00:00`

### Phase 1B: Likelihood Scoring

Likelihood is now supported for both Flan-T5 and Qwen/Qwen3.5 setwise rankers. The shared wrapper below runs `TopDown-Heap`, `BottomUp-Heap`, `DualEnd-Bubblesort`, and `DualEnd-Selection`; for DualEnd, the likelihood path scores the best-only label distribution once and reuses its max/min labels as the heuristic best/worst outputs.

```bash
# For each model × {DL19, DL20}:
bash experiments/run_likelihood.sh \
    <MODEL> <DATASET> <RUN_PATH> <OUTPUT_DIR>-likelihood \
    cuda 3 10 100 128
```

For the original T5 likelihood phase, use `passage_length=128`; for Qwen/Qwen3.5 follow-ups, use `passage_length=512`.

Methods: TD-Heap/BU-Heap/DE-Bubble/DE-Selection

| Model family | DL19 | DL20 |
|-------|------|------|
| flan-t5-xl | 4 runs | 4 runs |
| qwen3-8b | 4 runs | 4 runs |
| qwen3.5-9b | 4 runs | 4 runs |

**Note**: Flan-T5 likelihood reads the final decoder label distribution directly. Causal Qwen/Qwen3.5 likelihood scores short teacher-forced continuations like `Passage A`; this avoids brittle single-token label assumptions and still produces zero completion tokens. For `dualend` and the routed joint-signal refinements, this is still only a best-only proxy, not exact joint `Best: X, Worst: Y` likelihood.

### Phase 1C: Qwen3 Family (Decoder-Only, Thinking Models)

| Model | DL19 OUTPUT_DIR | DL20 OUTPUT_DIR | PL |
|-------|----------------|----------------|-----|
| `Qwen/Qwen3-4B` | `results/qwen3-4b-dl19` | `results/qwen3-4b-dl20` | 512 |
| `Qwen/Qwen3-8B` | `results/qwen3-8b-dl19` | `results/qwen3-8b-dl20` | 512 |
| `Qwen/Qwen3-14B` | `results/qwen3-14b-dl19` | `results/qwen3-14b-dl20` | 512 |

**passage_length=512** for Qwen3 (32k+ context window).

Each: 8 methods × 2 datasets = 16 runs. Total: **48 runs** across 3 models.

**Important Qwen3-specific behavior**:
- Thinking models emit `<think>...</think>` blocks; `max_new_tokens=512` set for dual-end, `256` for single-label
- `skip_special_tokens=False` in code to preserve `<think>` tags for regex stripping
- `enable_thinking=False` passed via `_chat_template_kwargs()`
- DualEnd dual prompt works in a single generation call (parses "Best: X, Worst: Y" from output)

**SLURM notes**:
- Qwen3-4B: `--mem=256G`, `--gres=gpu:h100:1`, `--time=10:00:00`
- Qwen3-8B: `--mem=512G`, `--gres=gpu:h100:1`, `--time=15:00:00`
- Qwen3-14B: `--mem=512G`, `--gres=gpu:h100:1`, `--time=20:00:00`

### Phase 1D: Qwen3.5-4B (Hybrid Causal — Gated DeltaNet + Standard Attention)

| Model | DL19 OUTPUT_DIR | DL20 OUTPUT_DIR | PL |
|-------|----------------|----------------|-----|
| `Qwen/Qwen3.5-4B` | `results/qwen3.5-4b-dl19` | `results/qwen3.5-4b-dl20` | 512 |
| `Qwen/Qwen3.5-9B` | `results/qwen3.5-9b-dl19` | `results/qwen3.5-9b-dl20` | 512 |
| `Qwen/Qwen3.5-27B` | `results/qwen3.5-27b-dl19` | `results/qwen3.5-27b-dl20` | 512 |

3 models × 8 methods × 2 datasets = **48 runs**. ✅ COMPLETE

**Prerequisite**: Requires transformers dev build with `qwen3_5` model type support:

Also need to run:
```bash
mv pyserini pyserini_src
```

After experiments finished, run:
```bash
mv pyserini_src pyserini
```

```bash
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"
```
Verify:
```python
from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
assert config.model_type == "qwen3_5"  # must not fail
```

**Architecture note**: Qwen3.5 is a **decoder-only hybrid** (75% Gated DeltaNet + 25% standard attention), NOT encoder-decoder. The `ForConditionalGeneration` name refers to multimodal capability (integrated ViT), not enc-dec architecture. Loaded via `AutoModelForCausalLM` per official HuggingFace docs.

### Phase 1 Evaluation

Use the unified evaluation script:
```bash
bash experiments/eval_all.sh results/flan-t5-xl-dl19 results/flan-t5-xl-dl20 \
    results/qwen3-4b-dl19 results/qwen3-4b-dl20 ...
```

Or evaluate all at once:
```bash
bash experiments/eval_all.sh results/*/
```

### Phase 1 Final Results (updated 2026-04-07, all bug fixes applied)

> **Bug fixes applied (2026-03-29)**: BottomUp bubblesort now does n-1 passes (full sort),
> BottomUp heapsort uses pure "worst" prompts throughout, compare_worst logs comparisons.
> All results below reflect the corrected code.

**Status**: Phase 1 COMPLETE — all 9 models × 2 datasets × 8 methods re-run with corrected code.

**Key finding — DualEnd-Cocktail is the top performer for capable models:**

| Model | DL19 Best Method | NDCG@10 | DL20 Best Method | NDCG@10 |
|-------|-----------------|---------|-----------------|---------|
| flan-t5-large | topdown_bubblesort | 0.6874 | **dualend_bubblesort** | **0.6308** |
| flan-t5-xl | topdown_bubblesort | 0.6980 | topdown_bubblesort | 0.6868 |
| flan-t5-xxl | **dualend_bubblesort** | **0.7137** | topdown_bubblesort | 0.6959 |
| Qwen3-4B | **dualend_selection** | **0.7220** | **dualend_selection** | **0.6627** |
| Qwen3-8B | **dualend_selection** | **0.7158** | **dualend_bubblesort** | **0.6678** |
| Qwen3-14B | **dualend_bubblesort** | **0.7519** | **dualend_bubblesort** | **0.7051** |
| Qwen3.5-4B | **dualend_bubblesort** | **0.7161** | **dualend_bubblesort** | **0.6768** |
| Qwen3.5-9B | **dualend_bubblesort** | **0.7370** | **dualend_bubblesort** | **0.6984** |
| Qwen3.5-27B | **dualend_bubblesort** | **0.7475** | **dualend_bubblesort** | **0.7186** |

**Observations (post-fix):**

1. **DualEnd dominates for capable models**: DualEnd (cocktail or selection) achieves best NDCG@10 in 14/18 model×dataset configurations.

2. **For T5 models, TopDown-Bubble is competitive**: The DualEnd advantage is smaller for encoder-decoder models. TopDown-Bubble wins on T5-XL (both datasets) and T5-XXL DL20.

3. **BottomUp is consistently weaker than TopDown** (post-fix): BU-Heap underperforms TD-Heap in all configs. Gap is catastrophic for T5-Large (BU-Heap .2888 vs TD-Heap .6541 on DL19), moderate for larger models (Q3.5-27B: .7135 vs .7449). BU-Bubble is slightly better than BU-Heap for larger models but still weaker than TD.

4. **Bidirectional ensemble never beats TopDown**: BiDir-RRF/Weighted always scores below TD-Heap. Weighted with α=0.9 (heavily favoring TopDown) is the best BiDir variant, confirming BottomUp hurts rather than helps.

5. **Position bias differs by selection type**: Best-selection shows strong primacy+recency U-shaped bias. Worst-selection shows strong recency bias. Dual-end shows more uniform patterns, especially for dual_best. This suggests dual prompts partially mitigate position bias.

6. **DualEnd-Selection is surprisingly strong for Qwen3-4B**: The double-ended selection sort beats cocktail shaker for the smallest Qwen model, suggesting smaller models may benefit from the simpler selection protocol.

---

## 6. Phase 2: Efficiency Comparison

**Goal**: Fill Table 3 (efficiency) in the paper.
**RQ**: RQ2

Table 3 data comes directly from Phase 1 logs. No additional runs needed.

### How to Extract

From each `.log` file, grep:
```bash
grep "Avg comparisons" results/flan-t5-xl-dl19/*.log
grep "Avg prompt tokens" results/flan-t5-xl-dl19/*.log
grep "Avg time per query" results/flan-t5-xl-dl19/*.log
```

### Additional Baseline: Permutation Voting (p=2)

This is the key comparison for RQ3 — BiDir costs 2× (runs TopDown + BottomUp), and permutation voting with p=2 also costs 2× (runs TopDown twice with different orderings). Which is a better use of 2× compute?

```bash
# Permutation voting with p=2 on Flan-T5-XL
python run.py \
    run --model_name_or_path google/flan-t5-xl \
        --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
        --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
        --save_path results/flan-t5-xl-dl19/permvote_p2_heapsort.txt \
        --scoring generation --hits 100 --passage_length 128 \
    setwise --num_child 3 --method heapsort --k 10 \
            --direction topdown --num_permutation 2 \
    2>&1 | tee results/flan-t5-xl-dl19/permvote_p2_heapsort.log
```

Repeat for DL20. (Total: 2 runs for Flan-T5-XL, 2 for Flan-T5-XXL, 2 for Qwen3-4B, 2 for Qwen3-8B, 2 for Qwen3.5-4B, 2 for Qwen3.5-9B, total 12 runs.)

---

## 7. Phase 3: Ablation Studies

**Goal**: Fill Tables 4 and 5 in the paper.
**RQs**: RQ2, RQ3

### Ablation 3A: Window Size (num_child) — Table 4

Test `c ∈ {2, 3, 5, 7}` for DualEnd-Cocktail on DL19 with Flan-T5-XL, Qwen3-8B, Qwen3.5-9B.

c=3 is already done in Phase 1. Need c=2, 5, 7. Total 3 models on 1 dataset, 9 runs total:

```bash
for NC in 2 5 7; do
    python run.py \
        run --model_name_or_path google/flan-t5-xl \
            --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
            --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
            --save_path results/ablation-nc/flan-t5-dl19/dualend_cocktail_nc${NC}.txt \
            --scoring generation --hits 100 --passage_length 128 \
        setwise --num_child ${NC} --method bubblesort --k 10 --direction dualend \
        2>&1 | tee results/ablation-nc/flan-t5-dl19/dualend_cocktail_nc${NC}.log
done
```

**What to report**: NDCG@10, #Comps, Parse success % (grep "Warning" count from logs).

### Ablation 3B: Fusion Weight (alpha) — Table 5

Test `α ∈ {0.3, 0.5, 0.7, 0.9}` for BiDir-Weighted on DL19 with Flan-T5-XL, Qwen3-8B, Qwen3.5-9B.

α=0.7 is already done in Phase 1. Need α=0.3, 0.5, 0.9. Total 3 models on 1 dataset, 9 runs total:

```bash
for ALPHA in 0.3 0.5 0.9; do
    python run.py \
        run --model_name_or_path google/flan-t5-xl \
            --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
            --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
            --save_path results/ablation-alpha/flan-t5-xl-dl19/bidir_weighted_a${ALPHA}.txt \
            --scoring generation --hits 100 --passage_length 128 \
        setwise --num_child 3 --method heapsort --k 10 \
                --direction bidirectional --fusion weighted --alpha ${ALPHA} \
        2>&1 | tee results/ablation-alpha/flan-t5-xl-dl19/bidir_weighted_a${ALPHA}.log
done
```

### Ablation 3C: CombSUM Fusion (for §5.3 discussion)

Test with Flan-T5-XL, Qwen3-8B, Qwen3.5-9B models on DL19, total 3 runs (This is included in `run_ablation_alpha.sh` script).

```bash
python run.py \
    run --model_name_or_path google/flan-t5-xl \
        --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
        --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
        --save_path results/ablation-alpha/flan-t5-xl-dl19/bidir_combsum.txt \
        --scoring generation --hits 100 --passage_length 128 \
    setwise --num_child 3 --method heapsort --k 10 \
            --direction bidirectional --fusion combsum \
    2>&1 | tee results/ablation-alpha/flan-t5-xl-dl19/bidir_combsum.log
```

### Ablation 3D: Passage Length — Table 5b (NEW)

Test `passage_length ∈ {64, 128, 256, 512}` to understand the effect of passage truncation on ranking quality.

**Motivation**: Different model families have different context limits. Flan-T5 has a 512-token limit; with `num_child=3` (4 passages), `passage_length=128` is the original paper's setting (at the limit). Qwen3 models have 32k+ context, so `passage_length=512` is feasible. This ablation answers: does more passage text help ranking quality, and does the answer differ for TopDown vs. DualEnd?

**Note on defaults**: Phase 1 already provides results at the default PL for each model family:
- Flan-T5: PL=128 (from Phase 1 — matches original paper)
- Qwen3/Qwen3.5: PL=512 (from Phase 1)

These can be reused in the ablation table — no need to re-run. The ablation script (`run_ablation_passage_length.sh`) tests `PL ∈ {64, 128, 256, 512}`, so it will re-run the default PL as a sanity check and add the non-default values.

**Note for T5**: `passage_length=128` with `num_child=3` is at the 512-token limit (matches original paper methodology). `passage_length=256` will truncate severely — include but note truncation. `passage_length=512` will truncate most of the prompt — included for completeness but may produce garbage results.

**Test on 3 representative models** (one T5, one Qwen3, one Qwen3.5) with TopDown-Heap on DL19:

```bash
# Flan-T5-XL: tests pl={64, 128, 256, 512}
# PL=128 result already exists from Phase 1 (this re-runs for consistency)
# PL=256/512 will show increasing truncation effects
bash experiments/run_ablation_passage_length.sh \
    google/flan-t5-xl \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/ablation-pl/flan-t5-xl-dl19 \
    cuda generation 3 10 100 topdown heapsort

# Qwen3-8B: tests pl={64, 128, 256, 512}
# PL=512 result already exists from Phase 1 (this re-runs for consistency)
bash experiments/run_ablation_passage_length.sh \
    Qwen/Qwen3-8B \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/ablation-pl/qwen3-8b-dl19 \
    cuda generation 3 10 100 topdown heapsort

# Qwen3.5-9B: tests pl={64, 128, 256, 512}
# PL=512 result already exists from Phase 1 (this re-runs for consistency)
bash experiments/run_ablation_passage_length.sh \
    Qwen/Qwen3.5-9B \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/ablation-pl/qwen3.5-9b-dl19 \
    cuda generation 3 10 100 topdown heapsort
```

**Extended**: Also test DualEnd-Cocktail to see if passage length interacts with dual-end parsing:
```bash
bash experiments/run_ablation_passage_length.sh \
    google/flan-t5-xl \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/ablation-pl/flan-t5-xl-dl19-dualend \
    cuda generation 3 10 100 dualend bubblesort

bash experiments/run_ablation_passage_length.sh \
    Qwen/Qwen3-8B \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/ablation-pl/qwen3-8b-dl19-dualend \
    cuda generation 3 10 100 dualend bubblesort

bash experiments/run_ablation_passage_length.sh \
    Qwen/Qwen3.5-9B \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/ablation-pl/qwen3.5-9b-dl19-dualend \
    cuda generation 3 10 100 dualend bubblesort
```

**What to report** (Table 5b):

| PL | Flan-T5-XL TD-Heap | Flan-T5-XL DE-Cocktail | Qwen3-8B TD-Heap | Qwen3-8B DE-Cocktail | Qwen3.5-9B TD-Heap | Qwen3.5-9B DE-Cocktail |
|----|:---:|:---:|:---:|:---:|:---:|:---:|
| 64 | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| 128 | ☐ (**default**, matches paper) | ☐ | ☐ | ☐ | ☐ | ☐ |
| 256 | ☐ (heavy truncation for T5) | ☐ | ☐ | ☐ | ☐ | ☐ |
| 512 | ☐ (severe truncation for T5) | ☐ (**default**) | ☐ (**default**) | ☐ (**default**) | ☐ (**default**) | ☐ (**default**) |

Total: ~24 runs (18 from ablation script + 6 reusable from Phase 1).

---

## 8. Phase 4: Analysis Experiments

**Goal**: Fill position bias analysis (§5.4), query difficulty (Tab 6), and ranking agreement metrics.
**RQs**: All three.
**Status**: IMPLEMENTED — all scripts, SLURM jobs, and code changes are ready.

### SLURM Submission (Recommended)

Submit all Phase 4 jobs at once:
```bash
# Creates logs/ directory, submits 7 GPU jobs (4A) + 1 CPU job (4B-4E)
bash experiments/submit_phase4.sh
```

This submits:
- **Phase 4A is now merged into Phase 1** — no separate GPU jobs needed. Comparison logs are produced automatically during Phase 1 runs.
- **1 CPU job** (`slurm_phase4bce_posthoc.sh`): All post-hoc analyses (4A analysis + 4B/C/D/E) from existing Phase 1-3 results. ~1h.

#### Individual SLURM submissions:
```bash
# Phase 4A — Position bias for a specific model (GPU required):
sbatch experiments/slurm_phase4a_position_bias.sh google/flan-t5-xl 128
sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3-4B 512

# Phase 4B/C/D/E — Post-hoc analyses (CPU only):
sbatch experiments/slurm_phase4bce_posthoc.sh
```

#### Qwen3.5-4B special note:
Edit `slurm_phase4a_position_bias.sh` line 17 to use `qwen35_env`:
```bash
# Comment out ranker_env, uncomment qwen35_env:
# conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
conda activate /scratch/project/neural_ir/hang/llm-rankers/qwen35_env
```

### Local Execution (Alternative)

Run all Phase 4 analyses with a single command (no SLURM):
```bash
# Runs 4A (with model re-runs for logging), 4B-4E (post-hoc from existing results)
bash experiments/run_phase4_analysis.sh google/flan-t5-xl cuda

# For Qwen models:
bash experiments/run_phase4_analysis.sh Qwen/Qwen3-4B cuda
```

### Phase 4 Scripts Reference

| Script | Purpose | GPU? | Time |
|--------|---------|------|------|
| `experiments/submit_phase4.sh` | Submit all Phase 4 SLURM jobs (9 GPU + 1 CPU) | — | Instant |
| `experiments/slurm_phase4a_position_bias.sh` | 4A: Re-run 3 methods with comparison logging + analyze | Yes (H100) | ~20h/model |
| `experiments/slurm_phase4bce_posthoc.sh` | 4B/C/D/E: All post-hoc analyses | No (CPU) | ~1h |
| `experiments/run_phase4_analysis.sh` | All of 4A-4E locally (no SLURM) | Yes | ~1h/model |
| `analysis/position_bias.py` | 4A: Chi-squared position bias from comparison logs | No | <1min |
| `analysis/query_difficulty.py` | 4B: Stratify queries by BM25 difficulty | No | <1min |
| `analysis/ranking_agreement.py` | 4C: TopDown vs BottomUp overlap@k | No | <1min |
| `analysis/per_query_analysis.py` | 4D: Per-query win/loss/tie counts | No | <1min |
| `analysis/parse_success_rate.sh` | 4E: DualEnd parse warnings from logs | No | <1min |

### Phase 4A Job Matrix (9 GPU jobs — all models)

| Model | `sbatch` Command | PL | Output Dir |
|-------|------------------|----|------------|
| `google/flan-t5-large` | `sbatch experiments/slurm_phase4a_position_bias.sh google/flan-t5-large 128` | 128 | `results/analysis/flan-t5-large-dl19/` |
| `google/flan-t5-xl` | `sbatch experiments/slurm_phase4a_position_bias.sh google/flan-t5-xl 128` | 128 | `results/analysis/flan-t5-xl-dl19/` |
| `google/flan-t5-xxl` | `sbatch experiments/slurm_phase4a_position_bias.sh google/flan-t5-xxl 128` | 128 | `results/analysis/flan-t5-xxl-dl19/` |
| `Qwen/Qwen3-4B` | `sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3-4B 512` | 512 | `results/analysis/qwen3-4b-dl19/` |
| `Qwen/Qwen3-8B` | `sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3-8B 512` | 512 | `results/analysis/qwen3-8b-dl19/` |
| `Qwen/Qwen3-14B` | `sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3-14B 512` | 512 | `results/analysis/qwen3-14b-dl19/` |
| `Qwen/Qwen3.5-4B` | `sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3.5-4B 512` | 512 | `results/analysis/qwen3.5-4b-dl19/` |
| `Qwen/Qwen3.5-9B` | `sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3.5-9B 512` | 512 | `results/analysis/qwen3.5-9b-dl19/` |
| `Qwen/Qwen3.5-27B` | `sbatch experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3.5-27B 512` | 512 | `results/analysis/qwen3.5-27b-dl19/` |

> **Note on Bidirectional**: Not included in 4A. Bidirectional runs TopDown + BottomUp
> independently, so its position bias is just the union of the TopDown and BottomUp logs.
> No separate run needed — combine the existing logs for analysis.
> Additionally, `BidirectionalEnsembleRanker` has a known bug where `_comparison_log_path`
> is not propagated to sub-rankers (see audit issue #1).

### Analysis 4A: Position Bias (§5.4) — MERGED INTO PHASE 1

**No separate GPU runs needed.** All Phase 1 scripts now include `--log_comparisons` by
default, writing comparison logs to `results/analysis/<model>-<dataset>/`. This is purely
observational — adding `--log_comparisons` does not affect ranking results (logging is
called after selection decisions).

**Directory structure**:
- Phase 1 ranking results: `results/<model>-<dataset>/<method>.txt`
- Phase 4A comparison logs: `results/analysis/<model>-<dataset>/<method>_comparisons.jsonl`

**Code changes made**:
1. `run.py`: Added `--log_comparisons <path>` flag
2. `setwise.py`: Added `_log_comparison()` method, called in `compare()` for "best" selections
3. `setwise_extended.py`: Added logging in `compare_both()` for "dual_best"/"dual_worst", and in `compare_worst()` for "worst" selections
4. All 6 Phase 1 scripts (topdown/bottomup/dualend × heapsort/bubblesort + dualend selection) now include `--log_comparisons`
5. Bidirectional scripts do NOT log (known bug: log path not propagated to sub-rankers; use TopDown + BottomUp logs separately)

**Log format** (JSONL, one line per comparison):
```json
{"qid": "1037798", "type": "best", "positions": ["A","B","C","D"], "selected": "B", "docids": ["doc1","doc2","doc3","doc4"]}
```

**Comparison types logged per method**:
- TopDown (Heap/Bubble) → `type="best"` (selects most relevant)
- BottomUp (Heap/Bubble) → `type="worst"` (selects least relevant)
- DualEnd (Cocktail/Selection) → `type="dual_best"` + `type="dual_worst"` (selects both)

**After Phase 1 completes**, run the analysis script (CPU only, <1 min per model):
```bash
python analysis/position_bias.py \
    --log results/analysis/<model>-dl19/*_comparisons.jsonl \
    --output results/analysis/<model>-dl19/position_bias_results.txt
```

Note: bubblesort variants use the same prompts as heapsort, so bias patterns should be similar. The sorting algorithm only affects *which* documents appear together, not the selection prompt. Having logs for ALL methods gives a richer analysis.
```bash
# Example for Flan-T5-XL:
python run.py run --model_name_or_path google/flan-t5-xl \
    --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
    --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    --save_path results/analysis/flan-t5-xl-dl19/topdown_heapsort.txt \
    --scoring generation --hits 100 --passage_length 128 \
    --log_comparisons results/analysis/flan-t5-xl-dl19/topdown_heapsort_comparisons.jsonl \
    setwise --num_child 3 --method heapsort --k 10 --direction topdown
```

**Analysis script**: `analysis/position_bias.py`
```bash
python analysis/position_bias.py \
    --log results/analysis/flan-t5-xl-dl19/*_comparisons.jsonl \
    --output results/analysis/flan-t5-xl-dl19/position_bias_results.txt
```

### Analysis 4B: Query Difficulty Stratification (Table 6) — UPDATED

Post-hoc analysis — no new model runs needed. Script: `analysis/query_difficulty.py`

Now includes `--dualend` argument. Compares all 3 methods per difficulty tercile with deltas vs TopDown baseline.

```bash
python analysis/query_difficulty.py \
    --topdown results/flan-t5-xl-dl19/topdown_heapsort.txt \
    --bottomup results/flan-t5-xl-dl19/bottomup_heapsort.txt \
    --dualend results/flan-t5-xl-dl19/dualend_bubblesort.txt \
    --bm25_run runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    --qrels dl19-passage
```

### Analysis 4C: Ranking Agreement — UPDATED

Post-hoc analysis. Script: `analysis/ranking_agreement.py`

Now includes `--dualend` (and optional `--bidir_rrf`). Compares all pairs: TD vs BU, TD vs DE, BU vs DE. Shows whether DualEnd rankings are more similar to TopDown or BottomUp.

```bash
python analysis/ranking_agreement.py \
    --topdown results/flan-t5-xl-dl19/topdown_heapsort.txt \
    --bottomup results/flan-t5-xl-dl19/bottomup_heapsort.txt \
    --dualend results/flan-t5-xl-dl19/dualend_bubblesort.txt
```

### Analysis 4D: Per-Query Wins Analysis — UPDATED

Post-hoc analysis. Script: `analysis/per_query_analysis.py`

Now includes `--dualend` argument. Full pairwise comparisons: TD vs BU, TD vs DE, BU vs DE, plus fusion/permvote analysis with DualEnd as the main method.

```bash
python analysis/per_query_analysis.py \
    --topdown results/flan-t5-xl-dl19/topdown_heapsort.txt \
    --bottomup results/flan-t5-xl-dl19/bottomup_heapsort.txt \
    --dualend results/flan-t5-xl-dl19/dualend_bubblesort.txt \
    --bidir_rrf results/flan-t5-xl-dl19/bidirectional_rrf.txt \
    --permvote results/flan-t5-xl-dl19/permvote_p2_heapsort.txt \
    --qrels dl19-passage
```

### Analysis 4E: Dual-End Parsing Success Rate — IMPLEMENTED

Extracts from Phase 1 logs. Script: `analysis/parse_success_rate.sh`

```bash
bash analysis/parse_success_rate.sh results/flan-t5-xl-dl19
bash analysis/parse_success_rate.sh results/qwen3-4b-dl19
```

---

## 9. Phase 5: BEIR Evaluation

**Goal**: Evaluate generalizability across diverse domains.
**Priority**: After Phase 1 results confirm the methods work on DL19/DL20.

**2026-04-09 note**: The representative BEIR runs are already being handled on the remote GPU server for:
- `TopDown-Bubble`
- `BottomUp-Bubble`
- `DualEnd-Cocktail`
- `BiDir-RRF`

with:
- `google/flan-t5-xl`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3.5-9B`

So there is **no local blocker on BEIR run scripting**. The remaining BEIR work for the paper is light-weight evaluation and synthesis once the run files land.

### BEIR Dataset-to-Run Mapping

| Dataset | BM25 Run File | ir_dataset_name |
|---------|--------------|-----------------|
| DBpedia | `runs/bm25/run.beir.bm25-flat.dbpedia-entity.txt` | `beir/dbpedia-entity/test` |
| NFCorpus | `runs/bm25/run.beir.bm25-flat.nfcorpus.txt` | `beir/nfcorpus/test` |
| SciFact | `runs/bm25/run.beir.bm25-flat.scifact.txt` | `beir/scifact/test` |
| TREC-COVID | `runs/bm25/run.beir.bm25-flat.trec-covid.txt` | `beir/trec-covid` |
| Touche2020 | `runs/bm25/run.beir.bm25-flat.webis-touche2020.txt` | `beir/webis-touche2020/v2` |
| FiQA | `runs/bm25/run.beir.bm25-flat.fiqa.txt` | `beir/fiqa/test` |

### BEIR Evaluation Strategy

Run a **subset of methods** on BEIR (not all 8 — too expensive):
1. TopDown-Bubble (baseline)
2. BottomUp-Bubble (RQ1 comparison)
3. DualEnd-Cocktail (RQ2 primary method)
4. BiDir-RRF (RQ3 primary method)

= 4 methods × 6 datasets = 24 runs per model.

**Recommended models for BEIR**: Start with 3 models:
- `google/flan-t5-xl` (passage_length=128) — representative encoder-decoder
- `Qwen/Qwen3-8B` (passage_length=512) — representative decoder-only
- `Qwen/Qwen3.5-9B` (passage_length=512) — representative stronger decoder-only (**TODO: Not Started Yet**)

Total: **72 runs** for BEIR.

### Minimal BEIR Extras Beyond Effectiveness

Once the remote BEIR run files are available and you have manual `trec_eval` outputs, the paper only needs three additional summaries beyond raw `nDCG@10` / `MAP@10`:

1. **Per-dataset deltas against TopDown-Bubble**
   - `DualEnd-Cocktail - TopDown-Bubble`
   - `BottomUp-Bubble - TopDown-Bubble`
   - `BiDir-RRF - TopDown-Bubble`
2. **Macro win-count summary**
   - On how many of the 8 datasets does DualEnd beat TopDown for each model family?
   - How many clear losses does BottomUp incur?
3. **Family-consistency note**
   - Does the DualEnd pattern hold for both encoder-decoder and causal models, or only for Qwen-style models?

That is enough to support the paper's generalization story. We do **not** need to re-run position-bias analysis, qualitative error analysis, or extra BEIR-specific ablations unless a reviewer explicitly asks.

### BEIR Evaluation

BEIR datasets use ir_datasets qrels, not Pyserini built-in qrels. Use `ir_measures` or `trec_eval` with the qrels from ir_datasets:

```python
import ir_datasets
import ir_measures
from ir_measures import nDCG

dataset = ir_datasets.load("beir/scifact/test")
qrels = dataset.qrels_iter()
# ... evaluate with ir_measures
```

Or export qrels to a file and use standard trec_eval:
```python
import ir_datasets
dataset = ir_datasets.load("beir/scifact/test")
with open("/tmp/scifact_qrels.txt", "w") as f:
    for qrel in dataset.qrels_iter():
        f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
```
Then: `trec_eval -m ndcg_cut.10 /tmp/scifact_qrels.txt results/.../*.txt`

---

## 10. Script Reference

### Existing Scripts

| Script | Purpose | Cluster? |
|--------|---------|----------|
| `experiments/run_extended_setwise_all.sh` | Run all 8 methods + evaluate (no SLURM) | No |
| `experiments/run_topdown_heapsort.sh` | Single method SLURM job | Yes |
| `experiments/run_topdown_bubblesort.sh` | Single method SLURM job | Yes |
| `experiments/run_bottomup_heapsort.sh` | Single method SLURM job | Yes |
| `experiments/run_bottomup_bubblesort.sh` | Single method SLURM job | Yes |
| `experiments/run_dualend_bubblesort.sh` | Single method SLURM job | Yes |
| `experiments/run_dualend_selection.sh` | Single method SLURM job | Yes |
| `experiments/run_bidirectional_rrf.sh` | Single method SLURM job | Yes |
| `experiments/run_bidirectional_weighted.sh` | Single method SLURM job | Yes |

### Phase 2-3 Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `experiments/run_likelihood.sh` | Likelihood scoring for T5 and Qwen/Qwen3.5 models | ✅ Created |
| `experiments/run_permvote_p2.sh` | Permutation voting baseline (p=2) | ✅ Created |
| `experiments/run_ablation_nc.sh` | num_child ablation (c=2,5,7) | ✅ Created |
| `experiments/run_ablation_alpha.sh` | Alpha ablation (α=0.3,0.5,0.9) + CombSUM | ✅ Created |
| `experiments/run_ablation_passage_length.sh` | Passage length ablation (pl=64,128,256,512) | ✅ Created |
| `experiments/eval_all.sh` | Evaluate all results directories | ✅ Created |

### Phase 4 Scripts

| Script | Purpose | GPU? | Status |
|--------|---------|------|--------|
| `experiments/submit_phase4.sh` | Submit all Phase 4 SLURM jobs (9 GPU + 1 CPU) | — | ✅ Updated |
| `experiments/slurm_phase4a_position_bias.sh` | 4A: Re-run 3 methods with `--log_comparisons` + analyze | Yes (H100) | ✅ Created |
| `experiments/slurm_phase4bce_posthoc.sh` | 4B/C/D/E: All post-hoc analyses on existing results | No (CPU) | ✅ Created |
| `experiments/run_phase4_analysis.sh` | All of 4A-4E locally (no SLURM) | Yes | ✅ Created |

### Refinement Package Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `experiments/run_selective_dualend.sh` | Run Selective DualEnd with shortlist / uncertainty gating | ✅ Created |
| `experiments/run_bias_aware_dualend.sh` | Run order-robust / bias-aware DualEnd with controlled orderings | ✅ Created |
| `experiments/run_samecall_regularized.sh` | Run same-call worst-signal regularization | ✅ Created |
| `experiments/run_quality_cost_pareto.sh` | Build the quality-cost Pareto artifacts from existing results | ✅ Created |
| `experiments/run_when_dualend_helps.sh` | Build the qualitative DualEnd-help summary from existing artifacts or live env | ✅ Created |

### Analysis Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `analysis/position_bias.py` | 4A: Chi-squared position bias from comparison logs | ✅ Created |
| `analysis/query_difficulty.py` | 4B: Table 6 — stratify by BM25 difficulty | ✅ Created |
| `analysis/ranking_agreement.py` | 4C: TopDown vs BottomUp overlap@k | ✅ Created |
| `analysis/per_query_analysis.py` | 4D: Per-query win/loss/tie counts | ✅ Created |
| `analysis/parse_success_rate.sh` | 4E: Dual-end parse warnings from logs | ✅ Created |
| `analysis/quality_cost_pareto.py` | Plot / summarize NDCG@10 against calls, tokens, and time | ✅ Created |
| `analysis/when_dualend_helps.py` | Produce model-level help/hurt summaries and, when env allows, qualitative exemplars | ✅ Created |

---

## 11. Results Checklist

Check off each cell as experiments complete. Each cell is one run.

### Table 1: Main Results (8 methods × DL19 + DL20)

For each model below, all 8 methods on both DL19 and DL20:

| Model | DL19 (8 methods) | DL20 (8 methods) |
|-------|:-:|:-:|
| Flan-T5-large | ✅ | ✅ |
| Flan-T5-xl | ✅ | ✅ |
| Flan-T5-xxl | ✅ | ✅ |
| Qwen3-4B | ✅ | ✅ |
| Qwen3-8B | ✅ | ✅ |
| Qwen3-14B | ✅ | ✅ |
| Qwen3.5-4B | ✅ | ✅ |
| Qwen3.5-9B | ✅ | ✅ |
| Qwen3.5-27B | ✅ | ✅ |

#### Likelihood Scoring (T5 models only)
| Model | DL19 (2 methods) | DL20 (2 methods) |
|-------|:-:|:-:|
| Flan-T5-large | ✅ | ✅ |
| Flan-T5-xl | ✅ | ✅ |
| Flan-T5-xxl | ✅ | ✅ |

### Table 3: Efficiency
✅ Extract from Phase 1 logs (no new runs) — data available in results.txt files

### Table 4: num_child Ablation
| c | DL19 NDCG@10 | #Comps | Parse% |
|---|:---:|:---:|:---:|
| 2 | ☐ | ☐ | ☐ |
| 3 | from Phase 1 | — | — |
| 5 | ☐ | ☐ | ☐ |
| 7 | ☐ | ☐ | ☐ |

### Table 5: Alpha Ablation
| α | 0.3 | 0.5 | 0.7 | 0.9 |
|---|:---:|:---:|:---:|:---:|
| DL19 NDCG@10 | ☐ | ☐ | from Phase 1 | ☐ |

### Table 5b: Passage Length Ablation
| PL | Flan-T5-XL TD-Heap | Qwen3-4B TD-Heap | Qwen3-4B DE-Cocktail |
|----|:---:|:---:|:---:|
| 64 | ☐ | ☐ | ☐ |
| 128 | from Phase 1 (**default**) | ☐ | ☐ |
| 256 | ☐ (heavy truncation) | ☐ | ☐ |
| 512 | ☐ (severe truncation) | from Phase 1 (**default**) | from Phase 1 (**default**) |

### Table 6: Query Difficulty
✅ Post-hoc analysis from Phase 1

### Permutation Voting Baseline
| | DL19 | DL20 |
|--|:---:|:---:|
| PermVote (p=2) | 🔄 running | 🔄 running |

### Analysis Sections
- [x] Position bias analysis (§5.4)
- [x] Ranking agreement (top-10 overlap, Kendall's τ)
- [x] Per-query wins analysis
- [x] Dual-end parse success rate
- [x] Quality-cost Pareto analysis
- [x] "When DualEnd helps" summary analysis
- ☐ Error analysis (qualitative)

---

## 12. Common Mistakes to Avoid

### Critical

1. **Wrong passage_length for model**: Flan-T5 models use `passage_length=128` (matching the original paper arXiv:2310.09497). This is at the 512-token limit with `num_child=3` and marginal truncation is expected (same as the original paper). Qwen3/Qwen3.5 models have 32k+ context — use `passage_length=512`. For BEIR datasets with longer queries, consider reducing T5 to `passage_length=100`.

2. **Wrong model in SLURM scripts**: SLURM scripts default to `Qwen/Qwen3-4B`. Always pass the model as the first argument when running other models (Flan-T5, etc.).

3. **Wrong dataset variant**: Use `msmarco-passage/trec-dl-2019/judged` (with `/judged`), not `msmarco-passage/trec-dl-2019`. The `/judged` variant only includes queries with relevance judgments.

4. **Forgetting MAP@100**: The paper says "also report MAP@100". Add `-m map_cut.100` to all eval commands.

5. **BiDir double-counting comparisons**: BidirectionalEnsembleRanker already aggregates stats from both sub-rankers. Don't accidentally double-count when reporting.

6. **DualEnd T5 uses likelihood internally (even in generation mode)**: T5 cannot reliably produce dual-format text output ("Best: X, Worst: Y") — it echoes the template literally. So `compare_both()` automatically uses a single likelihood forward pass (reading max/min from softmax) regardless of `--scoring` setting. This is a SINGLE forward pass (satisfies the single-call constraint), requires no parsing, and uses the shorter "most relevant" prompt that fits within 512 tokens.

7. **DualEnd Qwen3 generation uses `max_new_tokens=512`**: Thinking models need a large token budget because `<think>...</think>` blocks can consume 200+ tokens before the answer. The answer itself is only ~10 tokens. If you still see truncated outputs, increase further. Non-Qwen causal generation uses `max_new_tokens=64`.

8. **Qwen/Qwen3.5 likelihood is sequence-scored, not single-token-logit scored**: the causal likelihood path scores short teacher-forced continuations such as `Passage A` instead of assuming `A/B/C/D` are stable single tokens. For `dualend`, the implementation reuses the best-only distribution and takes `argmax` as best plus `argmin` as worst, matching the existing T5 heuristic.

### Completion Token Reporting for DualEnd T5

**DualEnd Cocktail Shaker (bubblesort) with T5 reports completion tokens = 0.** This is correct and expected. The cocktail shaker sort exclusively calls `compare_both()`, which for T5 uses an internal likelihood forward pass (reading logits, no autoregressive decoding). Since no tokens are generated, completion tokens = 0 is the accurate count.

**DualEnd Selection Sort with T5 reports completion tokens > 0.** This is also correct. Selection sort has a two-stage architecture:
- **Stage 1 — Group comparisons**: Calls `compare_both()` → likelihood forward pass → 0 completion tokens (same as cocktail shaker)
- **Stage 2 — Tournament tiebreakers**: Calls `self.compare()` (parent's T5 generation) for best-of-winners and `_tournament_select_worst()` (T5 generation) for worst-of-losers → these DO generate tokens via autoregressive decoding → non-zero completion tokens

This is a meaningful architectural difference for the paper's efficiency analysis:
- **Cocktail shaker** is purely likelihood-based for T5 — no autoregressive decoding at all, making it computationally cheaper per comparison (single forward pass vs. multi-step generation)
- **Selection sort** mixes likelihood (`compare_both` in groups) with generation (tournament rounds), so it has both forward-pass and decoding costs

When reporting efficiency in the paper, completion tokens should be presented alongside a note explaining that 0 completion tokens for DualEnd cocktail reflects the use of likelihood scoring (single forward pass reading the full label distribution), not a counting error.

### Medium

8. **SLURM log paths are hardcoded**: The individual SLURM scripts have hardcoded `-o` and `-e` paths for qwen3-4b-dl20. Update these when running different models/datasets.

9. **SLURM --mem=512G**: This is probably overkill for Flan-T5-XL (3B). Consider reducing for cluster efficiency.

10. **Vicuna chat template**: The code has a hardcoded chat template for `vicuna` + `v1.5`. This should work but verify the template renders correctly.

11. **random.seed(929)**: Both `setwise.py` and `setwise_extended.py` set `random.seed(929)` at module level. This ensures reproducibility but means re-running produces identical results. Good for paper, but be aware.

### Low

12. **`results.txt` skip in eval loop**: The evaluation loop in `run_extended_setwise_all.sh` skips `results.txt` when iterating `*.txt` files. If you add other non-TREC `.txt` files to the results dir, add skip guards.

13. **Sorting method names**: `method=bubblesort` for DualEnd actually runs cocktail shaker sort. `method=selection` runs double-ended selection sort. The method name in CLI doesn't fully describe the algorithm — the `direction` flag changes behavior.

---

## Execution Order Summary

```
Phase 0: Prerequisites (30 min)
  → Verify BM25 baselines, model downloads, smoke test

Phase 1A: Flan-T5 family × DL19+DL20 × 8 methods (48 runs)
Phase 1B: Flan-T5 family × DL19+DL20 × likelihood (12 runs)
Phase 1C: Qwen3 family × DL19+DL20 × 8 methods (48 runs)
Phase 1D: Qwen3.5-4B × DL19+DL20 × 8 methods (16 runs)
  → After Phase 1: can fill Tab 1, Tab 3, start analyses

Phase 2: Permutation voting baseline (4-6 runs)
  → After Phase 2: can compare BiDir vs PermVote for RQ3

Phase 3: Ablations (~32 runs)
  3A: num_child (c=2,5,7) → Tab 4
  3B: alpha (α=0.3,0.5,0.9) + CombSUM → Tab 5
  3C: passage_length (pl=64,128,256,512) → Tab 5b (NEW)
  → After Phase 3: can fill Tab 4, Tab 5, Tab 5b

Phase 4: Analysis (ALL post-hoc — 4A merged into Phase 1, 4B-4E use Phase 1 results)
  → After Phase 4: can fill Tab 6, §5.4, all analysis sections

Phase 5: BEIR evaluation (64 runs with 2 representative models)
  → After Phase 5: can fill BEIR generalizability table
```

**Total estimated runs**: ~280-310 (can run in parallel on cluster)

⚠️ **Re-run needed (2026-03-29 bug fix)**:
- TopDown results: VALID (keep)
- DualEnd results: VALID (keep)
- BottomUp results: INVALID (re-run — bubblesort unsorted top-k + heapsort mixed prompts)
- Bidirectional results: INVALID (re-run — depends on bottomup)
- Affected methods per model×dataset: bottomup_heapsort, bottomup_bubblesort, bidirectional_rrf, bidirectional_weighted (4 of 8 methods)
- Re-runs needed: 9 models × 2 datasets × 4 methods = **72 runs**
- Phase 2 (permutation voting): VALID (uses topdown only)
- Phase 3A (num_child ablation): partial re-run if bottomup was tested
- Phase 3B (alpha ablation): re-run (uses bidirectional)
- Phase 4A (position bias): re-run bottomup logs
Phase 3-5: pending
**Total estimated GPU-hours**: ~200-400 hours (varies widely by model size; H100 is fast).

---

## Coverage Gap Analysis (2026-03-29)

Cross-referencing IDEA_REPORT hypotheses (H1-H6), research questions (RQ1-RQ3), and paper framing against the experiment plan:

### Must-Have (before submission)

| Gap | IDEA_REPORT Reference | Status | Action |
|-----|----------------------|--------|--------|
| **Statistical significance tests** | Paper framing claims "comparable effectiveness" | ✅ Done | Added [analysis/significance_tests.py](/Users/hangli/projects/llm-rankers/analysis/significance_tests.py) and recorded results in [research_pipeline_setwise/SIGNIFICANCE_TESTS.md](/Users/hangli/projects/llm-rankers/research_pipeline_setwise/SIGNIFICANCE_TESTS.md). Use these results to keep claims conservative. |
| **Dual selection accuracy** (Key Measurement 1 for Idea 2) | "How often does dual agree with separate best-only and worst-only?" | ❌ Missing | New Phase 4F: For 1-2 models, run same queries with TopDown, BottomUp, and DualEnd; compare whether DualEnd's best matches TopDown's best and DualEnd's worst matches BottomUp's worst. Post-hoc from comparison logs. |
| **Efficiency claim validation** (H4) | Old hypothesis: "~50% fewer LLM calls" | ⚠️ Reframed | Current evidence refutes the old efficiency framing. Keep the call-count table, but use it to support a quality-cost tradeoff claim rather than any fewer-calls claim. |
| **MAP@100 metric** | §4.1 "also report MAP@100" | ⚠️ In eval scripts but not checked | Verify eval_all.sh includes `-m map_cut.100` and results are collected. |

### Should-Have (strengthens paper)

| Gap | IDEA_REPORT Reference | Status | Action |
|-----|----------------------|--------|--------|
| **CombMNZ fusion** | Idea 3 lists 4 fusion methods | ❌ Missing (only RRF, CombSUM, Weighted) | Low priority — mention in limitations or add as one-line code change + re-run. |
| **Permutation voting for more models** | RQ3: BiDir vs PermVote(p=2) | ⚠️ Only Flan-T5-XL | Extend to at least 1 Qwen model for cross-family comparison. |
| **Position bias interaction** (H3/Key Measurement 3) | "Does asking for both change bias?" | ⚠️ Phase 4A covers it | Ensure analysis script compares dual-end bias pattern vs. TopDown+BottomUp combined. Already in position_bias.py (separate type analysis). |

### Nice-to-Have (time permitting)

| Gap | IDEA_REPORT Reference | Status | Action |
|-----|----------------------|--------|--------|
| Per-call information gain (bits) | Key Measurement 2 | ❌ Missing | Complex to compute. Could approximate via ranking quality / #calls ratio. |
| BottomUp likelihood scoring | H1 | ❌ Not in Phase 1B | Add BottomUp-Heap likelihood for T5 to compare with TopDown-Heap likelihood. |
| num_child ablation cross-method | Idea 2 experimental design | ⚠️ Only DualEnd | Could add TopDown and BottomUp for 1-2 models if time allows. |
| BEIR evaluation scripts | Phase 5 | ⚠️ Specified but no scripts yet | Create run scripts when Phase 1-4 are done. |
| Error analysis (qualitative) | Analysis checklist | ❌ No methodology | Define what to analyze: failure modes, query types where DualEnd helps/hurts. |

---

## 13. Review-Driven Roadmap

Post-result review changes the project emphasis. The strongest current paper is **not** "a new sorting method clearly beats the baseline." The strongest current paper is:

- a study of **directional asymmetry** in setwise LLM ranking,
- a modest but repeatable **joint best-worst elicitation** improvement,
- a clear negative result for **standalone BottomUp** and **independent BiDir fusion**,
- and a quality-cost tradeoff story rather than an efficiency story.

### Submission Positioning

- **Primary venue**: ICTIR
- **Stretch follow-up**: later ARR only after stronger generalization or a better quality-cost method
- **Core framing**: worst-selection alone is unreliable, but worst-selection inside a joint prompt changes model behavior and can help modestly
- **Do not overclaim**: DualEnd is strongest overall, but only one TREC DL configuration is Bonferroni-significant and the cost is high

### Prioritized Experiment Package

| Priority | Experiment | Why it matters | Minimal scope | Success criterion | Deliverable |
|---|---|---|---|---|---|
| P0 | **Finish BEIR representative evaluation** | Tests whether the DualEnd directional pattern survives beyond small TREC DL query sets | 3 models × 8 datasets × 4 methods: TopDown-Bubble, BottomUp-Bubble, DualEnd-Cocktail, BiDir-RRF | DualEnd is positive on a clear majority of datasets or shows at least one family-consistent win pattern | BEIR summary table + per-dataset deltas + macro win counts |
| P0 | **Quality-cost Pareto analysis** | Directly addresses the biggest reviewer concern: modest gains at high cost | Use existing TREC DL outputs first; compare NDCG@10 against calls, tokens, and wall time | Paper can show where DualEnd is or is not worthwhile under budgeted settings | New figure/table for paper and narrative |
| P0 | **"When DualEnd helps" qualitative analysis** | Converts a weak average-gain story into a mechanism story | Inspect wins/losses for 2-3 representative models; group by query/window type | Produce a small taxonomy with concrete examples | Short analysis section + appendix table |
| P1 | **Selective DualEnd** | Highest-leverage method refinement; may preserve quality while reducing cost | Start from best TopDown baseline and invoke DualEnd only on query-locally uncertain windows or final shortlist | Recover most of the DualEnd gain at much lower cost than full DualEnd | New method + quality-cost table |
| P1 | **Order-robust / bias-aware DualEnd** | Converts the dual-worst bias reversal from analysis into method novelty | Use 2-3 controlled orderings only on query-locally uncertain windows | Improves robustness or average quality enough to justify extra calls | Bias-aware variant + ablation |
| P2 | **Same-call worst-signal regularization** | Tests whether the useful signal is the joint prompt rather than independent worst ranking | Use DualEnd same-call outputs to adjust local ordering or pruning, without standalone BottomUp fusion | Small but consistent gain over vanilla DualEnd or better cost-quality tradeoff | Lightweight method note or appendix |

### 2026-04-09 Status Snapshot

- **P0 BEIR representative evaluation**: execution is already happening remotely; local work is now only the summary table + deltas once the run files arrive.
- **P0 Quality-cost Pareto analysis**: implemented and run. Output: `results/analysis/pareto/QUALITY_COST_PARETO.md`.
- **P0 "When DualEnd helps" analysis**: implemented and run in summary mode. Output: `results/analysis/dualend_qualitative/WHEN_DUALEND_HELPS_SUMMARY.md`.
- **P1 Selective DualEnd**: implemented, CLI wired, shell wrapper created, logging counters exposed, tests passing.
- **P1 Order-robust / bias-aware DualEnd**: implemented, CLI wired, shell wrapper created, logging counters exposed, tests passing.
- **P2 Same-call worst-signal regularization**: implemented, CLI wired, shell wrapper created, logging counters exposed, tests passing.

### Ready-To-Run Commands

#### P0. Quality-Cost Pareto Analysis

Build the summary from all existing TREC DL result folders:

```bash
bash experiments/run_quality_cost_pareto.sh
```

Outputs:
- `results/analysis/pareto/QUALITY_COST_PARETO.md`
- `results/analysis/pareto/all_points.csv`
- `results/analysis/pareto/method_means.csv`
- frontier CSVs and optional plots

Current paper-facing takeaway from the existing results:
- Global comparison/token frontier: `TD-Heap`, `TD-Bubble`, `DE-Cocktail`
- Global time frontier: `PermVote(p=2)`, `TD-Heap`, `TD-Bubble`, `DE-Cocktail`
- This means any refined method should aim to land between `TD-Bubble` and `DE-Cocktail`, not merely beat a weak baseline.

#### P0. "When DualEnd Helps" Analysis

Run the qualitative analysis on the existing representative DL19 result folders:

```bash
bash experiments/run_when_dualend_helps.sh
```

The wrapper now auto-detects `./ranker_env/bin/python` when that environment is
available, so the default command above will use the retrieval-enabled stack and
produce passage-level exemplars in addition to the summary statistics.

Outputs:
- `results/analysis/topdown_heap_dualend_bubble_qualitative/WHEN_DUALEND_HELPS_SUMMARY.md`
- `results/analysis/topdown_heap_dualend_bubble_qualitative/when_dualend_helps_summary.json`
- per-model markdown summaries with query text and passage snippets

Current paper-facing summary from the generated artifact (tested on all models, show a few for example):
- `flan-t5-xl-dl19`: mean delta `-0.0017`, Help/Hurt/Tie `20 / 16 / 7`
- `qwen3-8b-dl19`: mean delta `+0.0336`, Help/Hurt/Tie `26 / 12 / 5`
- `qwen3-14b-dl19`: mean delta `+0.0072`, Help/Hurt/Tie `19 / 15 / 9`
...

Interpretation:
- DualEnd is **not** universally beneficial.
- The strongest qualitative motivation is therefore **selective routing**, not unconditional replacement of TopDown.
- In the live exemplar run, the strongest Qwen gains mostly come from **adding relevant documents** to the top-k, while the T5 hurt cases more often reflect relevant-document drops or mixed noisy set changes.

#### P0. Extra Likelihood Follow-up

**What**: run the new likelihood path on representative models to measure how much of Qwen DualEnd latency comes from autoregressive decoding versus the underlying comparison count and prompt volume.

**Why**: the new causal likelihood implementation removes completion-token overhead and Qwen thinking-block latency, while preserving the same sorting structure. This gives a clean before/after efficiency check for `TopDown`, `BottomUp`, and heuristic `DualEnd`.

Representative commands (each call runs `TopDown-Heap`, `BottomUp-Heap`, `DualEnd-Bubblesort`, and `DualEnd-Selection`): (**TODO: Qwen3.5-9B Not Started Yet**)

```bash
bash experiments/run_likelihood.sh \
  google/flan-t5-xl \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/flan-t5-xl-dl19-likelihood \
  cuda 3 10 100 128

bash experiments/run_likelihood.sh \
  Qwen/Qwen3-8B \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/qwen3-8b-dl19-likelihood \
  cuda 3 10 100 512

bash experiments/run_likelihood.sh \
  Qwen/Qwen3.5-9B \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/qwen3.5-9b-dl19-likelihood \
  cuda 3 10 100 512
```

Each output directory will contain:
- `topdown_heapsort.txt` / `.log`
- `bottomup_heapsort.txt` / `.log`
- `dualend_bubblesort.txt` / `.log`
- `dualend_selection.txt` / `.log`

First comparison targets:
- Qwen likelihood should drive `completion tokens -> 0` for all variants.
- `DualEnd-Bubblesort` and `DualEnd-Selection` latency should drop substantially versus generation-based Qwen DualEnd, but they will still remain much more expensive than `TD-Heap` because the comparison count is unchanged.
- Quality should be reported carefully: causal `DualEnd` likelihood is the same heuristic used by T5 (best-only distribution with max/min reuse), not exact joint-output likelihood.

#### P1. Selective DualEnd

**What**: keep the cheaper TopDown sorting logic, but upgrade only selected windows to a same-call best-worst prompt.

**Why**: the current evidence suggests the DualEnd gain is concentrated in ambiguous windows or near the top-k decision boundary, while full DualEnd pays that cost everywhere.

**How**:
- `--gate_strategy shortlist`: only invoke DualEnd on windows touching the top shortlist
- `--gate_strategy uncertain`: only invoke DualEnd when the window falls into the tightest query-local BM25-spread percentile
- `--gate_strategy hybrid`: union of the two
- `0.15` now means "tightest 15% of windows for this query"; the existing `--margin_threshold` flag is kept as a backward-compatible alias for that percentile value
- `heapsort` remains supported for Selective DualEnd, but its shortlist gate is intentionally disabled because heap node indices are not stable rank positions; use `uncertain` if you want routed heapsort behavior
- for the paper-facing first pass, prefer `bubblesort` because the routing logic is easiest to interpret near the ranking head

Configs:
| Model | Dataset | Method | Sorting | Gating Strategy |
|---|---|---|---|---|
| `google/glan-t5-xl` | DL19 | generation | bubblesort | uncertain |
| `google/glan-t5-xl` | DL19 | likelihood | bubblesort | uncertain |
| `google/glan-t5-xl` | DL19 | generation | bubblesort | shortlist |
| `google/glan-t5-xl` | DL19 | likelihood | bubblesort | shortlist |
| `google/glan-t5-xl` | DL19/20 | generation | bubblesort | hybrid |
| `google/glan-t5-xl` | DL19/20 | likelihood | bubblesort | hybrid |
| `Qwen/Qwen3-8B` | DL19 | generation | bubblesort | uncertain |
| `Qwen/Qwen3-8B` | DL19 | likelihood | bubblesort | uncertain |
| `Qwen/Qwen3-8B` | DL19 | generation | bubblesort | shortlist |
| `Qwen/Qwen3-8B` | DL19 | likelihood | bubblesort | shortlist |
| `Qwen/Qwen3-8B` | DL19/20 | generation | bubblesort | hybrid |
| `Qwen/Qwen3-8B` | DL19/20 | likelihood | bubblesort | hybrid |
| `Qwen/Qwen3.5-9B` | DL19 | generation | bubblesort | uncertain |
| `Qwen/Qwen3.5-9B` | DL19 | likelihood | bubblesort | uncertain |
| `Qwen/Qwen3.5-9B` | DL19 | generation | bubblesort | shortlist |
| `Qwen/Qwen3.5-9B` | DL19 | likelihood | bubblesort | shortlist |
| `Qwen/Qwen3.5-9B` | DL19/20 | generation | bubblesort | hybrid |
| `Qwen/Qwen3.5-9B` | DL19/20 | likelihood | bubblesort | hybrid |

Representative commands: (**TODO: Qwen3.5-9B Not Started Yet**)

```bash
bash experiments/run_selective_dualend.sh \
  google/flan-t5-xl \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/flan-t5-xl-dl19-selective-dualend \
  cuda generation 3 10 100 128 bubblesort hybrid 20 0.15

bash experiments/run_selective_dualend.sh \
  Qwen/Qwen3-8B \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/qwen3-8b-dl19-selective-dualend \
  cuda generation 3 10 100 512 bubblesort hybrid 20 0.15

bash experiments/run_selective_dualend.sh \
  Qwen/Qwen3.5-9B \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/qwen3.5-9b-dl19-selective-dualend \
  cuda generation 3 10 100 512 bubblesort hybrid 20 0.15
```

Then evaluate:

```bash
bash experiments/eval_all.sh results/flan-t5-xl-dl19 results/qwen3-8b-dl19 results/qwen3.5-9b-dl19
```

Use the new log counters in `results.txt`:
- `Avg dual invocations`
- `Avg single invocations`

#### P1. Order-Robust / Bias-Aware DualEnd

**What**: keep DualEnd, but on gated hard windows run a tiny number of controlled orderings and majority-vote the best / worst decision back into the original window.

**Why**: the position-bias analysis shows a reversed `dual_worst` pattern, so the most principled refinement is to spend extra calls only where order sensitivity is most likely to matter.

**How**:
- use the same gating interface as Selective DualEnd
- add `--order_robust_orderings 3` for base / reversed / shifted orderings
- restrict the implementation to `bubblesort` and `selection`; `heapsort` is rejected because it bypasses the order-robust joint prompt path

Representative commands:

```bash
bash experiments/run_bias_aware_dualend.sh \
  google/flan-t5-xl \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/flan-t5-xl-dl19 \
  cuda generation 3 10 100 128 bubblesort hybrid 20 0.15 3

bash experiments/run_bias_aware_dualend.sh \
  Qwen/Qwen3-8B \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/qwen3-8b-dl19 \
  cuda generation 3 10 100 512 bubblesort hybrid 20 0.15 3

bash experiments/run_bias_aware_dualend.sh \
  Qwen/Qwen3.5-9B \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/qwen3.5-9b-dl19 \
  cuda generation 3 10 100 512 bubblesort hybrid 20 0.15 3
```

Then evaluate:

```bash
bash experiments/eval_all.sh results/flan-t5-xl-dl19 results/qwen3-8b-dl19 results/qwen3.5-9b-dl19
```

Use the new log counters in `results.txt`:
- `Avg order-robust windows`
- `Avg extra orderings`

#### P2. Same-Call Worst-Signal Regularization

**What**: keep a head-focused TopDown bubblesort pass, but use the same-call worst output only as a local negative constraint inside each window.

**Why**: BottomUp alone is too noisy, but the worst signal may still be useful **when conditioned on the same prompt and the same candidates as the winning best signal**.

**How**:
- promote the best item as usual
- only demote the worst item locally when that candidate is already outside the protected ranking head frontier (top-`k` plus one active window)
- do not run a full backward DualEnd pass

Representative commands:

```bash
bash experiments/run_samecall_regularized.sh \
  google/flan-t5-xl \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/flan-t5-xl-dl19 \
  cuda generation 3 10 100 128

bash experiments/run_samecall_regularized.sh \
  Qwen/Qwen3-8B \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/qwen3-8b-dl19 \
  cuda generation 3 10 100 512

bash experiments/run_samecall_regularized.sh \
  Qwen/Qwen3.5-9B \
  msmarco-passage/trec-dl-2019/judged \
  runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
  results/qwen3.5-9b-dl19 \
  cuda generation 3 10 100 512
```

Then evaluate:

```bash
bash experiments/eval_all.sh results/flan-t5-xl-dl19 results/qwen3-8b-dl19 results/qwen3.5-9b-dl19
```

Use the new log counter in `results.txt`:
- `Avg regularized worst moves`

### Deprioritized Work

Unless a reviewer specifically demands them, the following are low leverage:

- more standalone BottomUp variants,
- more BiDir alpha or fusion sweeps,
- more effort on information-theoretic "bits per call" without budget-aware results,
- and making all three strategies appear equally central.

The current results do not justify that allocation.

### Recommended Story After This Roadmap

If the next package lands well, the paper should say:

1. Setwise LLM ranking is directionally asymmetric.
2. Worst-selection alone is unreliable and biased.
3. Joint best-worst elicitation changes model behavior and can help.
4. The raw DualEnd method is quality-first but expensive.
5. A selective or bias-aware DualEnd variant is the most promising refined direction.

---

## 14. Immediate TODO Checklist

### Must Do For The Current Paper

- [ ] Finish the representative BEIR subset first, before expanding any new ablation grid
- [x] Build a quality-cost Pareto figure from existing TREC DL results
- [x] Write a short "When DualEnd helps" taxonomy scaffold from per-query wins/losses
- [x] Add an explicit paper-facing note that the strongest story is directional asymmetry, not raw efficiency
- [ ] Keep BottomUp and BiDir in the paper as negative evidence, not co-equal contributions

### Highest-Leverage New Experiments

- [x] Implement **Selective DualEnd**: invoke DualEnd only on query-locally uncertain windows or on a final shortlist
- [ ] Measure Selective DualEnd against best TopDown and full DualEnd on TREC DL first
- [ ] If Selective DualEnd is promising, run the same representative BEIR subset
- [x] Implement **Order-Robust / Bias-Aware DualEnd** if Selective DualEnd alone is not strong enough
- [ ] Re-run significance testing for any refined method that becomes paper-central

### Nice To Have After The Core Package

- [x] Add same-call worst-signal regularization if it is lightweight to implement
- [ ] Extend permutation-voting comparison to one Qwen family model only if still needed
- [ ] Add CombMNZ only if it takes negligible effort and does not delay the main refinement package
