# Experiment Plan: Beyond Best Selection

**Paper:** "Beyond Best Selection: Bidirectional Strategies for Efficient LLM-Based Setwise Ranking"
**Last Updated:** 2026-03-18
**Status:** Pre-experiment (no results yet)

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

---

## 1. Research Questions

The paper addresses three research questions. Every experiment must map to at least one RQ.

| RQ | Question | Key Tables/Figures |
|----|----------|--------------------|
| **RQ1** | Does bottom-up (worst) selection achieve different effectiveness than top-down (best) selection? | Tab 1 (main), Tab 6 (difficulty), Position bias analysis |
| **RQ2** | Does dual-end selection maintain effectiveness while improving efficiency? | Tab 1, Tab 3 (efficiency), Tab 4 (num_child ablation) |
| **RQ3** | Does bidirectional ensemble improve over individual methods? | Tab 1, Tab 5 (alpha ablation), Permutation voting comparison |

---

## 2. Audit Summary: Paper vs. Code Gaps

### Critical Misalignments Found

| Issue | Paper Says | Code/Scripts Say | Impact | Status |
|-------|-----------|-----------------|--------|--------|
| **Models** | Flan-T5-XL/XXL, Vicuna | SLURM defaults to Qwen3-4B | **HIGH** | ✅ Plan updated: use 7 models (3 T5, 3 Qwen3, 1 Qwen3.5) |
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

Decoder-Only / Causal (Qwen3 family — generation only, thinking models):
  - Qwen/Qwen3-4B             (4B)
  - Qwen/Qwen3-8B             (8B)
  - Qwen/Qwen3-14B            (14B)

Hybrid Causal (Qwen3.5 — requires transformers dev build with qwen3_5 support):
  - Qwen/Qwen3.5-4B           (4B, loaded via AutoModelForCausalLM)
```

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
| **Phase 1: Core (DL19+DL20)** | 7 models | 2 datasets | 8 methods | 112 |
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

### Phase 1B: Flan-T5 Likelihood Scoring

Only T5 models support likelihood. Test TopDown-Heap and DualEnd-Heap (likelihood extracts both max and min from the softmax distribution in one forward pass).

```bash
# For each T5 model × {DL19, DL20}:
bash experiments/run_likelihood.sh \
    <MODEL> <DATASET> <RUN_PATH> <OUTPUT_DIR>-likelihood \
    cuda 3 10 100 128
```

| Model | DL19 | DL20 |
|-------|------|------|
| flan-t5-large | 2 runs | 2 runs |
| flan-t5-xl | 2 runs | 2 runs |
| flan-t5-xxl | 2 runs | 2 runs |

Total: **12 runs**.

**Note**: DualEnd with likelihood uses the standard "most relevant" prompt but reads both the max (best) and min (worst) from the softmax distribution. This is "free" dual information with zero extra tokens.

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

8 methods × 2 datasets = **16 runs**.

**Prerequisite**: Requires transformers dev build with `qwen3_5` model type support:
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

Repeat for DL20. (Total: 2 runs for Flan-T5-XL, optionally 2 for Flan-T5-XXL)

---

## 7. Phase 3: Ablation Studies

**Goal**: Fill Tables 4 and 5 in the paper.
**RQs**: RQ2, RQ3

### Ablation 3A: Window Size (num_child) — Table 4

Test `c ∈ {2, 3, 5, 7}` for DualEnd-Cocktail on DL19 with Flan-T5-XL.

c=3 is already done in Phase 1. Need c=2, 5, 7:

```bash
for NC in 2 5 7; do
    python run.py \
        run --model_name_or_path google/flan-t5-xl \
            --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
            --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
            --save_path results/ablation-nc/dualend_cocktail_nc${NC}.txt \
            --scoring generation --hits 100 --passage_length 128 \
        setwise --num_child ${NC} --method bubblesort --k 10 --direction dualend \
        2>&1 | tee results/ablation-nc/dualend_cocktail_nc${NC}.log
done
```

**What to report**: NDCG@10, #Comps, Parse success % (grep "Warning" count from logs).

### Ablation 3B: Fusion Weight (alpha) — Table 5

Test `α ∈ {0.3, 0.5, 0.7, 0.9}` for BiDir-Weighted on DL19 with Flan-T5-XL.

α=0.7 is already done in Phase 1. Need α=0.3, 0.5, 0.9:

```bash
for ALPHA in 0.3 0.5 0.9; do
    python run.py \
        run --model_name_or_path google/flan-t5-xl \
            --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
            --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
            --save_path results/ablation-alpha/bidir_weighted_a${ALPHA}.txt \
            --scoring generation --hits 100 --passage_length 128 \
        setwise --num_child 3 --method heapsort --k 10 \
                --direction bidirectional --fusion weighted --alpha ${ALPHA} \
        2>&1 | tee results/ablation-alpha/bidir_weighted_a${ALPHA}.log
done
```

### Ablation 3C: CombSUM Fusion (for §5.3 discussion)

```bash
python run.py \
    run --model_name_or_path google/flan-t5-xl \
        --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
        --run_path runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
        --save_path results/flan-t5-xl-dl19/bidirectional_combsum.txt \
        --scoring generation --hits 100 --passage_length 128 \
    setwise --num_child 3 --method heapsort --k 10 \
            --direction bidirectional --fusion combsum \
    2>&1 | tee results/flan-t5-xl-dl19/bidirectional_combsum.log
```

### Ablation 3D: Passage Length — Table 5b (NEW)

Test `passage_length ∈ {64, 100, 128, 256, 512}` to understand the effect of passage truncation on ranking quality.

**Motivation**: Different model families have different context limits. Flan-T5 has a 512-token limit; with `num_child=3` (4 passages), `passage_length=100` is the safe max. Qwen3 models have 32k+ context, so `passage_length=512` is feasible. This ablation answers: does more passage text help ranking quality, and does the answer differ for TopDown vs. BottomUp vs. DualEnd?

**Note for T5**: `passage_length=128` with `num_child=3` will cause input truncation (4×128+70=582 > 512). This is still interesting to test — it shows the effect of truncation. `passage_length=256` will truncate even more severely. Include these but note the truncation in the table.

**Test on 2 representative models** (one T5, one Qwen3) with TopDown-Heap on DL19:

```bash
# Flan-T5-XL: test pl={64, 128, 256}
# Note: pl=512 may cause truncation warnings for Flan-T5 (512-token limit)
bash experiments/run_ablation_passage_length.sh \
    google/flan-t5-xl \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/ablation-pl/flan-t5-xl-dl19 \
    cuda generation 3 10 100 topdown heapsort

# Qwen3-4B: test pl={64, 128, 256, 512}
bash experiments/run_ablation_passage_length.sh \
    Qwen/Qwen3-4B \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/ablation-pl/qwen3-4b-dl19 \
    cuda generation 3 10 100 topdown heapsort
```

**Extended**: Also test DualEnd-Cocktail to see if passage length interacts with dual-end parsing:
```bash
bash experiments/run_ablation_passage_length.sh \
    Qwen/Qwen3-4B \
    msmarco-passage/trec-dl-2019/judged \
    runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
    results/ablation-pl/qwen3-4b-dl19-dualend \
    cuda generation 3 10 100 dualend bubblesort
```

**What to report** (Table 5b):

| PL | Flan-T5-XL TD-Heap | Qwen3-4B TD-Heap | Qwen3-4B DE-Cocktail |
|----|:---:|:---:|:---:|
| 64 | ☐ | ☐ | ☐ |
| 100 | ☐ | ☐ | ☐ |
| 128 | ☐ (our default, matches paper) | ☐ | ☐ |
| 256 | ☐ (heavy truncation) | ☐ | ☐ |
| 512 | N/A | ☐ | ☐ |

Total: ~14 runs.

---

## 8. Phase 4: Analysis Experiments

**Goal**: Fill position bias analysis (§5.4), query difficulty (Tab 6), and ranking agreement metrics.
**RQs**: All three.

### Analysis 4A: Position Bias (§5.4)

**Requires code modification** — need to log per-comparison selections.

The analysis needs:
- For each comparison call, record: (query_id, position_labels [A,B,C,D], which was selected, true relevance labels from qrels)
- Aggregate selection frequency by position across all comparisons and queries

**Implementation plan**:
1. Add a `--log_comparisons <path>` flag to `run.py`
2. In `SetwiseLlmRanker.compare()` and `BottomUpSetwiseLlmRanker.compare_worst()`, log each comparison as a JSON line:
   ```json
   {"qid": "...", "type": "best|worst|dual", "positions": ["A","B","C","D"], "selected": "B", "doc_relevances": [3, 2, 0, 1]}
   ```
3. Write a Python analysis script that reads the log and computes:
   - Selection frequency per position for best, worst, dual-best, dual-worst
   - Accuracy (whether selected doc matches qrel-optimal choice)
   - Per-model, per-dataset breakdowns

**Runs needed**: Re-run TopDown-Heap, BottomUp-Heap, DualEnd-Cocktail on DL19 with Flan-T5-XL with comparison logging enabled. (3 runs)

**Note**: This is a **code change** task, not just a script task. Defer until Phase 1 confirms the methods work.

### Analysis 4B: Query Difficulty Stratification (Table 6)

Post-hoc analysis — no new model runs needed. Uses Phase 1 results.

**Script**:
```python
# Pseudocode for difficulty analysis
# 1. Compute BM25 NDCG@10 per query (from qrels + BM25 run)
# 2. Split queries into terciles by BM25 NDCG@10
# 3. For each tercile, compute mean NDCG@10 for TopDown and BottomUp
# 4. Report delta (BottomUp - TopDown) per tercile
```

Will create `analysis/query_difficulty.py` script.

### Analysis 4C: Ranking Agreement

Post-hoc analysis using Phase 1 results:
- **Top-10 overlap**: How many of TopDown's top-10 docs also appear in BottomUp's top-10?
- **Kendall's τ**: Rank correlation over the full 100-doc list between TopDown and BottomUp rankings

```python
# Pseudocode
# For each query:
#   td_top10 = set of top-10 docids from TopDown results
#   bu_top10 = set of top-10 docids from BottomUp results
#   overlap = len(td_top10 & bu_top10)
#   tau = kendall_tau(td_ranking, bu_ranking)
```

Will create `analysis/ranking_agreement.py` script.

### Analysis 4D: Per-Query Wins Analysis

Post-hoc analysis:
- Count queries where BottomUp > TopDown (by NDCG@10) and vice versa
- For BiDir: count queries where fusion helps vs. hurts
- Characterize winning queries

Will create `analysis/per_query_analysis.py` script.

### Analysis 4E: Dual-End Parsing Success Rate

Extract from Phase 1 logs:
```bash
# Count dual parse warnings
grep -c "Could not reliably parse dual output" results/flan-t5-xl-dl19/dualend_bubblesort.log
grep -c "Could not reliably parse dual output" results/flan-t5-xl-dl19/dualend_selection.log

# Total comparisons
grep "Avg comparisons" results/flan-t5-xl-dl19/dualend_bubblesort.log
```

Parse success % = 1 - (warnings / total comparisons across all queries).

---

## 9. Phase 5: BEIR Evaluation

**Goal**: Evaluate generalizability across diverse domains.
**Priority**: After Phase 1 results confirm the methods work on DL19/DL20.

### BEIR Dataset-to-Run Mapping

| Dataset | BM25 Run File | ir_dataset_name |
|---------|--------------|-----------------|
| DBpedia | `runs/bm25/run.beir.bm25-flat.dbpedia-entity.txt` | `beir/dbpedia-entity/test` |
| NFCorpus | `runs/bm25/run.beir.bm25-flat.nfcorpus.txt` | `beir/nfcorpus/test` |
| SciFact | `runs/bm25/run.beir.bm25-flat.scifact.txt` | `beir/scifact/test` |
| TREC-COVID | `runs/bm25/run.beir.bm25-flat.trec-covid.txt` | `beir/trec-covid` |
| Touche2020 | `runs/bm25/run.beir.bm25-flat.webis-touche2020.txt` | `beir/webis-touche2020/v2` |
| HotpotQA | `runs/bm25/run.beir.bm25-flat.hotpotqa.txt` | `beir/hotpotqa/test` |
| Quora | `runs/bm25/run.beir.bm25-flat.quora.txt` | `beir/quora/test` |
| FEVER | `runs/bm25/run.beir.bm25-flat.fever.txt` | `beir/fever/test` |

### BEIR Evaluation Strategy

Run a **subset of methods** on BEIR (not all 8 — too expensive):
1. TopDown-Heap (baseline)
2. BottomUp-Heap (RQ1 comparison)
3. DualEnd-Cocktail (RQ2 primary method)
4. BiDir-RRF (RQ3 primary method)

= 4 methods × 8 datasets = 32 runs per model.

**Recommended models for BEIR**: Start with 2 models:
- `google/flan-t5-xl` (passage_length=128) — representative encoder-decoder
- `Qwen/Qwen3-4B` (passage_length=512) — representative decoder-only

Total: **64 runs** for BEIR.

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

### Scripts Created (this session)

| Script | Purpose | Status |
|--------|---------|--------|
| `experiments/run_likelihood.sh` | Likelihood scoring for T5 models | ✅ Created |
| `experiments/run_permvote_p2.sh` | Permutation voting baseline (p=2) | ✅ Created |
| `experiments/run_ablation_nc.sh` | num_child ablation (c=2,5,7) | ✅ Created |
| `experiments/run_ablation_alpha.sh` | Alpha ablation (α=0.3,0.5,0.9) + CombSUM | ✅ Created |
| `experiments/run_ablation_passage_length.sh` | Passage length ablation (pl=64,128,256,512) | ✅ Created |
| `experiments/eval_all.sh` | Evaluate all results directories | ✅ Created |
| `analysis/query_difficulty.py` | Table 6 analysis | ✅ Created |
| `analysis/ranking_agreement.py` | Ranking agreement metrics | ✅ Created |
| `analysis/per_query_analysis.py` | Per-query win/loss analysis | ✅ Created |
| `analysis/parse_success_rate.sh` | Dual-end parse success % | ✅ Created |

---

## 11. Results Checklist

Check off each cell as experiments complete. Each cell is one run.

### Table 1: Main Results (8 methods × DL19 + DL20)

For each model below, all 8 methods on both DL19 and DL20:

| Model | DL19 (8 methods) | DL20 (8 methods) |
|-------|:-:|:-:|
| Flan-T5-large | ☐ | ☐ |
| Flan-T5-xl | ☐ | ☐ |
| Flan-T5-xxl | ☐ | ☐ |
| Qwen3-4B | ☐ | ☐ |
| Qwen3-8B | ☐ | ☐ |
| Qwen3-14B | ☐ | ☐ |
| Qwen3.5-4B | ☐ | ☐ |

#### Likelihood Scoring (T5 models only)
| Model | DL19 (2 methods) | DL20 (2 methods) |
|-------|:-:|:-:|
| Flan-T5-large | ☐ | ☐ |
| Flan-T5-xl | ☐ | ☐ |
| Flan-T5-xxl | ☐ | ☐ |

### Table 3: Efficiency
☐ Extract from Phase 1 logs (no new runs)

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
| 100 | ☐ | ☐ | ☐ |
| 128 | ☐ (default, matches paper) | ☐ | ☐ |
| 256 | ☐ (heavy truncation) | ☐ | ☐ |
| 512 | N/A | ☐ | ☐ |

### Table 6: Query Difficulty
☐ Post-hoc analysis from Phase 1

### Permutation Voting Baseline
| | DL19 | DL20 |
|--|:---:|:---:|
| PermVote (p=2) | ☐ | ☐ |

### Analysis Sections
- ☐ Position bias analysis (§5.4) — requires code changes
- ☐ Ranking agreement (top-10 overlap, Kendall's τ)
- ☐ Per-query wins analysis
- ☐ Dual-end parse success rate
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

7. **DualEnd Qwen3 uses `max_new_tokens=512`**: Thinking models need a large token budget because `<think>...</think>` blocks can consume 200+ tokens before the answer. The answer itself is only ~10 tokens. If you still see truncated outputs, increase further. Non-Qwen causal models use `max_new_tokens=64`.

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

Phase 4: Analysis (mostly post-hoc, except position bias code changes)
  → After Phase 4: can fill Tab 6, §5.4, all analysis sections

Phase 5: BEIR evaluation (64 runs with 2 representative models)
  → After Phase 5: can fill BEIR generalizability table
```

**Total estimated runs**: ~220-250 (can run in parallel on cluster)
**Total estimated GPU-hours**: ~200-400 hours (varies widely by model size; H100 is fast).
