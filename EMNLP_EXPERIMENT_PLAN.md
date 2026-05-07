# EMNLP 2026 MaxContext Multi-Family Command Sheet

Source documents:
- `EMNLP_PAPER_DESIGN.md`
- `EMNLP_IMPLEMENTATION_PLAN.md`
- `Extra-Experiments/MAX_CONTEXT_Extra-Experiments/EXPERIMENT_PLAN.md` for the unchanged IDEA_007 Qwen-only matrix

This sheet is for the human operator after the implementation gate is green. The EMNLP plan is independent of IDEA_007 phase numbering; do not interleave.

Phase gates:
1. Phase A smoke must pass before Phase B or Phase C submission.
2. `EMNLP_BUDGET.md` and the BEIR pool=100 fit probe must be green before Phase B BEIR pool=100 submission.
3. Phase C′ byte-equality control must pass before interpreting EMNLP results.
4. Optional Phase D/E runs launch only after required Phase B/C results are green or budget-cleared.

Loader architecture gate: Mistral 3 (`model_type=mistral3`) and Qwen 3.5 (`qwen3_5`, `qwen3_5_moe`) are vision-language configs and must use the multimodal loader path (`MULTIMODAL_MODEL_TYPES` in `llmrankers/setwise.py`: `AutoProcessor` + `AutoModelForImageTextToText`). The IR task remains text-only; image inputs are unused. Qwen 3 (`qwen3`, `qwen3_moe`) stays on the existing causal-LM path for byte-equality with IDEA_007.

All commands below run from the cluster login node. If your cluster root differs, replace `/scratch/project/neural_ir/hang/llm-rankers` consistently.

The setup block below defines unexported shell variables and helper functions on your login-node shell. The EMNLP submit/eval scripts re-establish their own environment; the block exists to make copy-paste loops explicit and auditable.

```bash
REPO_ROOT=/scratch/project/neural_ir/hang/llm-rankers
cd "$REPO_ROOT"

mkdir -p logs results/emnlp results/maxcontext_dualend

METHODS=(
  "topdown_bubblesort"
  "topdown_heapsort"
  "bottomup_bubblesort"
  "bottomup_heapsort"
  "maxcontext_topdown"
  "maxcontext_bottomup"
  "maxcontext_dualend"
)

MODELS=(
  "Qwen/Qwen3.5-0.8B"
  "Qwen/Qwen3.5-2B"
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-27B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "mistralai/Ministral-3-3B-Instruct-2512"
  "mistralai/Ministral-3-8B-Instruct-2512"
  "mistralai/Ministral-3-14B-Instruct-2512"
)

OPTIONAL_QWEN3_MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-32B"
)

DATASETS=(
  "dl19"
  "dl20"
  "beir-dbpedia"
  "beir-nfcorpus"
  "beir-scifact"
  "beir-trec-covid"
  "beir-touche2020"
  "beir-fiqa"
)

REQUIRED_POOL_SIZES=(10 20 30 40 50 100)
OPTIONAL_QWEN3_POOL_SIZES=(10 20 30 40 50)

model_tag () {
  basename "$1" | tr '/.' '-' | tr '[:upper:]' '[:lower:]'
}

dataset_path () {
  case "$1" in
    dl19) echo "msmarco-passage/trec-dl-2019/judged" ;;
    dl20) echo "msmarco-passage/trec-dl-2020/judged" ;;
    beir-dbpedia) echo "beir/dbpedia-entity/test" ;;
    beir-nfcorpus) echo "beir/nfcorpus/test" ;;
    beir-scifact) echo "beir/scifact/test" ;;
    beir-trec-covid) echo "beir/trec-covid" ;;
    beir-touche2020) echo "beir/webis-touche2020/v2" ;;
    beir-fiqa) echo "beir/fiqa/test" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

bm25_run () {
  case "$1" in
    dl19) echo "runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt" ;;
    dl20) echo "runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt" ;;
    beir-dbpedia) echo "runs/bm25/run.beir.bm25-flat.dbpedia-entity.txt" ;;
    beir-nfcorpus) echo "runs/bm25/run.beir.bm25-flat.nfcorpus.txt" ;;
    beir-scifact) echo "runs/bm25/run.beir.bm25-flat.scifact.txt" ;;
    beir-trec-covid) echo "runs/bm25/run.beir.bm25-flat.trec-covid.txt" ;;
    beir-touche2020) echo "runs/bm25/run.beir.bm25-flat.webis-touche2020.txt" ;;
    beir-fiqa) echo "runs/bm25/run.beir.bm25-flat.fiqa.txt" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

qrels_label () {
  case "$1" in
    dl19) echo "dl19-passage" ;;
    dl20) echo "dl20-passage" ;;
    beir-dbpedia) echo "beir-v1.0.0-dbpedia-entity-test" ;;
    beir-nfcorpus) echo "beir-v1.0.0-nfcorpus-test" ;;
    beir-scifact) echo "beir-v1.0.0-scifact-test" ;;
    beir-trec-covid) echo "beir-v1.0.0-trec-covid-test" ;;
    beir-touche2020) echo "beir-v1.0.0-webis-touche2020-test" ;;
    beir-fiqa) echo "beir-v1.0.0-fiqa-test" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}
```

## Dataset and Qrels Labels

| dataset_tag       | ir_datasets path                      | BM25 run                                                 | qrels                               | level |
|-------------------|---------------------------------------|----------------------------------------------------------|-------------------------------------|------:|
| `dl19`            | `msmarco-passage/trec-dl-2019/judged` | `runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt` | `dl19-passage`                      |     2 |
| `dl20`            | `msmarco-passage/trec-dl-2020/judged` | `runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt` | `dl20-passage`                      |     2 |
| `beir-dbpedia`    | `beir/dbpedia-entity/test`            | `runs/bm25/run.beir.bm25-flat.dbpedia-entity.txt`        | `beir-v1.0.0-dbpedia-entity-test`   |     1 |
| `beir-nfcorpus`   | `beir/nfcorpus/test`                  | `runs/bm25/run.beir.bm25-flat.nfcorpus.txt`              | `beir-v1.0.0-nfcorpus-test`         |     1 |
| `beir-scifact`    | `beir/scifact/test`                   | `runs/bm25/run.beir.bm25-flat.scifact.txt`               | `beir-v1.0.0-scifact-test`          |     1 |
| `beir-trec-covid` | `beir/trec-covid`                     | `runs/bm25/run.beir.bm25-flat.trec-covid.txt`            | `beir-v1.0.0-trec-covid-test`       |     1 |
| `beir-touche2020` | `beir/webis-touche2020/v2`            | `runs/bm25/run.beir.bm25-flat.webis-touche2020.txt`      | `beir-v1.0.0-webis-touche2020-test` |     1 |
| `beir-fiqa`       | `beir/fiqa/test`                      | `runs/bm25/run.beir.bm25-flat.fiqa.txt`                  | `beir-v1.0.0-fiqa-test`             |     1 |

## Model-Tag Conventions

EMNLP main outputs use lowercase basename with `/` and `.` converted to `-`, for example `Qwen/Qwen3.5-9B` becomes `qwen3-5-9b`.

EMNLP stability outputs reuse `submit_max_context_jobs.sh`, which keeps dots in the lowercase basename, for example `qwen3.5-9b`. Analysis scripts normalize both forms.

## Phase A — Smoke Gate (42 jobs)

One command submits 3 representative models × 7 methods × dl19 × pools {50,100}.

```bash
bash scripts/smoke_emnlp_models.sh --dry-run
bash scripts/smoke_emnlp_models.sh
```

Evaluation / verification after completion:

```bash
bash scripts/smoke_emnlp_models.sh --eval-only
bash scripts/smoke_emnlp_models.sh --verify-only
```

Gate criteria are method-aware: all methods require full `.txt` coverage, valid top-10 permutations, positive NDCG@10, and clean logs; MaxContext methods additionally require zero parse-fallback and numeric out-of-range counters.

Before Phase B BEIR pool=100 launches, run the tokenizer-only BEIR fit probe:

```bash
python3 scripts/probe_beir_pool100_fit.py
```

## Phase B — Required Main Matrix (3024 jobs)

7 methods × 9 models × 8 datasets × 6 pool sizes.

```bash
for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for POOL_SIZE in "${REQUIRED_POOL_SIZES[@]}"; do
        bash submit_emnlp_jobs.sh \
          --method "$METHOD" \
          --model "$MODEL" \
          --dataset "$DATASET" \
          --pool-size "$POOL_SIZE" \
          --tag phase_b_required
      done
    done
  done
done
```

Evaluation after completion:

```bash
for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for POOL_SIZE in "${REQUIRED_POOL_SIZES[@]}"; do
        bash eval_emnlp_jobs.sh \
          --method "$METHOD" \
          --model "$MODEL" \
          --dataset "$DATASET" \
          --pool-size "$POOL_SIZE" \
          --tag phase_b_required
      done
    done
  done
done
```

## Phase C — Required Stability (1260 scientific cells / 1980 submissions)

7 methods × 3 models × dl19 × 6 pool sizes × 10 reps = 1260 scientific cells. The wrapper submits 1980 stability-layout jobs because `submit_max_context_jobs.sh` now emits the 11-block default layout: the IDEA_007 ws-3/ws-PS TopDown overhead plus default-on standard BottomUp ws-3/ws-PS blocks under `original/bottomup/{ws-3,ws-ps}/`.

```bash
for MODEL in \
  "Qwen/Qwen3.5-9B" \
  "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  "mistralai/Ministral-3-8B-Instruct-2512"
do
  bash submit_emnlp_stability_jobs.sh --model "$MODEL" --dataset DL19 --dry-run
  bash submit_emnlp_stability_jobs.sh --model "$MODEL" --dataset DL19
done
```

Evaluation / analysis after completion:

```bash
for MODEL in \
  "Qwen/Qwen3.5-9B" \
  "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  "mistralai/Ministral-3-8B-Instruct-2512"
do
  STABILITY_MODEL_TAG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
  for V in {1..10}; do
    bash eval_max_context_jobs.sh \
      --pool-sizes "10 20 30 40 50 100" \
      --tag "emnlp_phase_c_required/${STABILITY_MODEL_TAG}-dl19/stability-test-runs/test_run_v${V}" \
      --model "$MODEL" \
      --dataset DL19
  done
done

python3 analysis/cross_model_stability.py
```

## Phase C′ — Prime-Constraint Recheck (35 jobs)

Run the IDEA_007-only 35-cell layout for the byte-equality control. The default `submit_max_context_jobs.sh` layout now emits 55 jobs; Phase C′ must pass `--idea007-only`.

```bash
bash submit_max_context_jobs.sh \
  --idea007-only \
  --tag emnlp_phase_c_prime \
  --model Qwen/Qwen3-4B \
  --dataset DL19 \
  --dry-run

bash submit_max_context_jobs.sh \
  --idea007-only \
  --tag emnlp_phase_c_prime \
  --model Qwen/Qwen3-4B \
  --dataset DL19
```

Evaluation and byte-equality diff:

```bash
bash eval_max_context_jobs.sh \
  --idea007-only \
  --tag emnlp_phase_c_prime \
  --model Qwen/Qwen3-4B \
  --dataset DL19

CANONICAL_ROOT="results/maxcontext_dualend/qwen3-4b-dl19/stability-test-runs/test_run_v1"
PHASE_C_ROOT="results/maxcontext_dualend/emnlp_phase_c_prime"
find "$PHASE_C_ROOT" -name "*.eval" -print | while read -r f; do
  rel="${f#${PHASE_C_ROOT}/}"
  diff -q "$f" "$CANONICAL_ROOT/$rel" || echo "DRIFT: $f"
done
```

## Phase D — Optional Qwen3 Main Matrix (1680 jobs)

7 methods × 6 optional Qwen3 models × 8 datasets × 5 pool sizes.

```bash
for MODEL in "${OPTIONAL_QWEN3_MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for POOL_SIZE in "${OPTIONAL_QWEN3_POOL_SIZES[@]}"; do
        bash submit_emnlp_jobs.sh \
          --method "$METHOD" \
          --model "$MODEL" \
          --dataset "$DATASET" \
          --pool-size "$POOL_SIZE" \
          --tag phase_d_qwen3_optional
      done
    done
  done
done
```

Evaluation after completion:

```bash
for MODEL in "${OPTIONAL_QWEN3_MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for POOL_SIZE in "${OPTIONAL_QWEN3_POOL_SIZES[@]}"; do
        bash eval_emnlp_jobs.sh \
          --method "$METHOD" \
          --model "$MODEL" \
          --dataset "$DATASET" \
          --pool-size "$POOL_SIZE" \
          --tag phase_d_qwen3_optional
      done
    done
  done
done
```

## Phase E — Optional Qwen3-8B Stability (350 scientific cells / 550 submissions)

7 methods × Qwen3-8B × dl19 × 5 pool sizes × 10 reps = 350 scientific cells. The stability-layout wrapper emits 550 submissions with the default 11-block layout.

```bash
bash submit_emnlp_stability_jobs.sh \
  --model "Qwen/Qwen3-8B" \
  --dataset DL19 \
  --tag-prefix emnlp_phase_e_qwen3_optional \
  --pool-sizes "10 20 30 40 50" \
  --dry-run

bash submit_emnlp_stability_jobs.sh \
  --model "Qwen/Qwen3-8B" \
  --dataset DL19 \
  --tag-prefix emnlp_phase_e_qwen3_optional \
  --pool-sizes "10 20 30 40 50"
```

Evaluation / analysis after completion:

```bash
MODEL="Qwen/Qwen3-8B"
STABILITY_MODEL_TAG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
for V in {1..10}; do
  bash eval_max_context_jobs.sh \
    --tag "emnlp_phase_e_qwen3_optional/${STABILITY_MODEL_TAG}-dl19/stability-test-runs/test_run_v${V}" \
    --model "$MODEL" \
    --dataset DL19
done

python3 analysis/cross_model_stability.py
```

## Phase F — MaxContext Position-Bias Controls (432 required jobs)

Phase F measures whether MaxContext decisions depend on the order in which the remaining BM25 pool is presented to the LLM. It reuses Phase B `poolNN/` forward outputs and adds two MaxContext-only conditions:

- `--reverse`: reverse the remaining pool before each MaxContext comparison.
- `--shuffle`: shuffle the remaining pool before each MaxContext comparison with fixed seed 929.

This is distinct from legacy `--shuffle_ranking`, which permutes the full initial ranking once. Phase F does not touch Heap/Bubble methods.

Required subset: 3 representative models × 4 representative datasets (`dl19`, `dl20`, `beir-dbpedia`, `beir-fiqa`) × 3 MaxContext methods × 6 pool sizes × 2 new conditions = 432 jobs. Claims should be phrased as representative evidence, not BEIR-wide coverage.

```bash
for MODEL in \
  "Qwen/Qwen3.5-9B" \
  "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  "mistralai/Ministral-3-8B-Instruct-2512"
do
  for DATASET in dl19 dl20 beir-dbpedia beir-fiqa; do
    for METHOD in maxcontext_topdown maxcontext_bottomup maxcontext_dualend; do
      for POOL_SIZE in "${REQUIRED_POOL_SIZES[@]}"; do
        bash submit_emnlp_jobs.sh --reverse \
          --method "$METHOD" --model "$MODEL" --dataset "$DATASET" \
          --pool-size "$POOL_SIZE" --tag phase_f_reverse
        bash submit_emnlp_jobs.sh --shuffle \
          --method "$METHOD" --model "$MODEL" --dataset "$DATASET" \
          --pool-size "$POOL_SIZE" --tag phase_f_shuffle
      done
    done
  done
done
```

Evaluation mirrors submission with the same condition flag and reads suffixed leaves (`pool50_reverse/`, `pool50_shuffle/`):

```bash
bash eval_emnlp_jobs.sh --reverse --method maxcontext_dualend \
  --model "Qwen/Qwen3.5-9B" --dataset dl19 --pool-size 50 --tag phase_f_reverse
bash eval_emnlp_jobs.sh --shuffle --method maxcontext_dualend \
  --model "Qwen/Qwen3.5-9B" --dataset dl19 --pool-size 50 --tag phase_f_shuffle
```

Optional Phase F stability uses the same fixed shuffle seed, so 10 reps measure system/model nondeterminism, not shuffle-seed variance:

```bash
bash submit_emnlp_stability_jobs.sh --reverse --model "Qwen/Qwen3.5-9B" \
  --dataset DL19 --tag-prefix emnlp_phase_f_reverse
bash submit_emnlp_stability_jobs.sh --shuffle --model "Qwen/Qwen3.5-9B" \
  --dataset DL19 --tag-prefix emnlp_phase_f_shuffle
```

## Position Bias and Stability Analysis

```bash
python3 analysis/position_bias_emnlp.py \
  --main-root results/emnlp/main \
  --output-root results/emnlp/analysis/position_bias_emnlp

python3 analysis/cross_model_stability.py
```

`analysis/position_bias_emnlp.py` parses `poolNN`, `poolNN_reverse`, and `poolNN_shuffle` leaves, writes condition-specific position-frequency summaries, and emits paired forward-vs-reverse / forward-vs-shuffle nDCG@10 deltas when matching `.eval` files are present.

## Dry-run validation

The dry-run must intercept both top-level `sbatch` calls made by EMNLP scripts and any nested `sbatch` calls made by wrappers around `submit_max_context_jobs.sh`. The safest approach is to put a stub `sbatch` binary first on `PATH`, then launch a clean non-login bash that preserves that `PATH`.

```bash
# 1. Build the stub binary.
DRYRUN_DIR=$(mktemp -d)
cat > "$DRYRUN_DIR/sbatch" <<'STUB'
#!/usr/bin/env bash
echo "STUB sbatch $*"
STUB
chmod +x "$DRYRUN_DIR/sbatch"
export PATH="$DRYRUN_DIR:$PATH"

# 2. Position yourself at the repo root and start a fresh non-login bash so
# /etc/profile.d/* and ~/.bash_profile cannot rewrite PATH and unshadow the
# stub. A non-login, non-interactive bash inherits the parent's exported PATH
# as-is.
REPO_ROOT=/scratch/project/neural_ir/hang/llm-rankers
cd "$REPO_ROOT"
bash --noprofile --norc

# 3. Inside that bash, verify the stub is still first on PATH before running
# any dry-run command. If the output is not "$DRYRUN_DIR/sbatch", stop and do
# not proceed.
command -v sbatch
type -a sbatch

# 4. Paste the setup block above, then run the EMNLP dry-runs:
bash scripts/smoke_emnlp_models.sh --dry-run
bash submit_emnlp_jobs.sh --dry-run --method maxcontext_dualend --model "Qwen/Qwen3.5-9B" --dataset dl19 --pool-size 100 --tag test
bash submit_emnlp_stability_jobs.sh --dry-run --model "meta-llama/Meta-Llama-3.1-8B-Instruct" --dataset DL19
bash submit_max_context_jobs.sh --dry-run --tag emnlp_phase_c_prime --model Qwen/Qwen3-4B --dataset DL19
```

Expected after the dry-run:

1. `type model_tag`, `type dataset_path`, `type bm25_run`, and `type qrels_label` all resolve in the child bash.
2. Every `STUB sbatch` line contains no local-machine path and uses fully resolved positional arguments after the script name.
3. Nothing actually submits; `squeue -u "$USER"` stays empty.
4. Phase A shows 42 submissions, Phase B one submission per explicit `(model, dataset, method, pool)` cell, Phase C 1260 scientific cells / 1980 stability-layout submissions across wrappers, and Phase C′ 35 submissions via `--idea007-only`.

## Go/No-Go Signal for `EMNLP_BUDGET.md`

Before Phase B, `EMNLP_BUDGET.md` must contain:

| Gate                    | Signal                                                                     | Status  |
|-------------------------|----------------------------------------------------------------------------|---------|
| HF config probe         | all required IDs resolve; `model_type` in allowlist; context length fits   | pending |
| BEIR pool=100 fit probe | `scripts/probe_beir_pool100_fit.py` passes for all 18 family/dataset cells | pending |
| Phase A smoke           | 42/42 pass method-aware verification                                       | pending |
| Phase C′                | 35/35 `.eval` byte-identical to canonical Qwen3-4B stability v1            | pending |
| Budget                  | projected GPU-hours and wall-clock fit available cluster budget            | pending |
| BEIR eval               | all qrels labels resolve locally with `-l 1`                               | pending |

Phase B is blocked until all rows are green.

## Note: EMNLP Plan Numbering

EMNLP plan is independent of IDEA_007 phase numbering; do not interleave.
