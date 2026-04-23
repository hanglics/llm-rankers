# MaxContext DualEnd Experiment Command Sheet

Source documents:
- `IDEA_007_IMPLEMENTATION_PLAN.md` (authoritative implementation plan)
- `IDEA_007.md` (design spec)

This sheet is for the human operator after the implementation gate is green.

Phase gates:
1. Phase 1 must pass before any wider submission.
2. Phase 2 must pass before Study A / baselines.
3. Phase 3 is the matched-hits regression pair.
4. Phase 4 runs Study A plus the predeclared matched-hits baselines.
5. Phase 5 runs Study B after Study A fixes the predeclared pool size.

All commands below assume you are at repo root: `/Users/hangli/projects/llm-rankers`.

```bash
mkdir -p logs results/maxcontext_dualend

DL19_DATASET="msmarco-passage/trec-dl-2019/judged"
DL20_DATASET="msmarco-passage/trec-dl-2020/judged"
DL19_RUN="runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"
DL20_RUN="runs/bm25/run.msmarco-v1-passage.bm25-default.dl20.txt"
QRELS_DL19="dl19-passage"
QRELS_DL20="dl20-passage"

MODELS=(
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-27B"
)

model_tag () {
  echo "$1" | tr '/.' '-' | tr '[:upper:]' '[:lower:]'
}

dataset_short () {
  if [[ "$1" == *"2020"* ]]; then
    echo "dl20"
  else
    echo "dl19"
  fi
}

submit_order_job () {
  local MODEL="$1"
  local DATASET="$2"
  local RUN_PATH="$3"
  local OUTPUT_DIR="$4"
  local ORDERING="$5"
  local ORDER_FLAG=""
  local ORDER_TAG="$ORDERING"
  if [[ "$ORDERING" == "inverse" ]]; then
    ORDER_FLAG="--shuffle_ranking inverse"
  elif [[ "$ORDERING" == "random" ]]; then
    ORDER_FLAG="--shuffle_ranking random"
  fi

  local MODEL_TAG
  MODEL_TAG=$(model_tag "$MODEL")
  local DATASET_TAG
  DATASET_TAG=$(dataset_short "$DATASET")
  local JOB_TAG="mc-order-${MODEL_TAG}-${DATASET_TAG}-${ORDER_TAG}"

  sbatch -J "${JOB_TAG}" \
    -o "logs/${JOB_TAG}-%j.out" \
    -e "logs/${JOB_TAG}-%j.err" \
    --wrap "cd /scratch/project/neural_ir/hang/llm-rankers && \
      module load anaconda3/2023.09-0 && \
      source \$EBROOTANACONDA3/etc/profile.d/conda.sh && \
      module load cuda/12.2.0 && \
      conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env && \
      export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf && \
      export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf && \
      export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini && \
      export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini && \
      mkdir -p \"${OUTPUT_DIR}\" results/analysis/position_bias_maxcontext/$(basename "${OUTPUT_DIR}") && \
      python run.py \
        run --model_name_or_path \"${MODEL}\" \
            --ir_dataset_name \"${DATASET}\" \
            --run_path \"${RUN_PATH}\" \
            --save_path \"${OUTPUT_DIR}/maxcontext_dualend_${ORDER_TAG}.txt\" \
            --device cuda \
            --scoring generation \
            --hits 50 \
            --query_length 128 \
            --passage_length 512 \
            ${ORDER_FLAG} \
            --log_comparisons \"results/analysis/position_bias_maxcontext/$(basename "${OUTPUT_DIR}")/maxcontext_dualend_${ORDER_TAG}_comparisons.jsonl\" \
        setwise --num_child 3 \
                --method selection \
                --k 50 \
                --num_permutation 1 \
                --direction maxcontext_dualend \
        2>&1 | tee \"${OUTPUT_DIR}/maxcontext_dualend_${ORDER_TAG}.log\""
}
```

## Phase 1 — Unit sanity

One GPU run, Qwen3-4B, DL19, pool=10, pl=512.

```bash
sbatch -J mc-sanity-q3-4b-dl19 \
  -o logs/mc-sanity-q3-4b-dl19-%j.out \
  -e logs/mc-sanity-q3-4b-dl19-%j.err \
  experiments/run_maxcontext_dualend.sh \
  "Qwen/Qwen3-4B" \
  "${DL19_DATASET}" \
  "${DL19_RUN}" \
  "results/maxcontext_dualend/phase1/qwen3-4b-dl19" \
  cuda generation 10 512
```

Evaluation after completion:

```bash
python -m pyserini.eval.trec_eval -c -l 2 \
  -m ndcg_cut.10 -m ndcg_cut.100 -m map_cut.10 -m map_cut.100 -m recall.1000 \
  "${QRELS_DL19}" \
  results/maxcontext_dualend/phase1/qwen3-4b-dl19/maxcontext_dualend.txt
```

## Phase 2 — Order-robustness pilot (12 runs)

2 models × 2 datasets × 3 orderings. Forward uses the BM25 order as-is; inverse and random use `--shuffle_ranking`.

```bash
submit_order_job "Qwen/Qwen3-4B"   "${DL19_DATASET}" "${DL19_RUN}" "results/maxcontext_dualend/phase2/qwen3-4b-dl19"   forward
submit_order_job "Qwen/Qwen3-4B"   "${DL19_DATASET}" "${DL19_RUN}" "results/maxcontext_dualend/phase2/qwen3-4b-dl19"   inverse
submit_order_job "Qwen/Qwen3-4B"   "${DL19_DATASET}" "${DL19_RUN}" "results/maxcontext_dualend/phase2/qwen3-4b-dl19"   random
submit_order_job "Qwen/Qwen3-4B"   "${DL20_DATASET}" "${DL20_RUN}" "results/maxcontext_dualend/phase2/qwen3-4b-dl20"   forward
submit_order_job "Qwen/Qwen3-4B"   "${DL20_DATASET}" "${DL20_RUN}" "results/maxcontext_dualend/phase2/qwen3-4b-dl20"   inverse
submit_order_job "Qwen/Qwen3-4B"   "${DL20_DATASET}" "${DL20_RUN}" "results/maxcontext_dualend/phase2/qwen3-4b-dl20"   random
submit_order_job "Qwen/Qwen3.5-9B" "${DL19_DATASET}" "${DL19_RUN}" "results/maxcontext_dualend/phase2/qwen3.5-9b-dl19" forward
submit_order_job "Qwen/Qwen3.5-9B" "${DL19_DATASET}" "${DL19_RUN}" "results/maxcontext_dualend/phase2/qwen3.5-9b-dl19" inverse
submit_order_job "Qwen/Qwen3.5-9B" "${DL19_DATASET}" "${DL19_RUN}" "results/maxcontext_dualend/phase2/qwen3.5-9b-dl19" random
submit_order_job "Qwen/Qwen3.5-9B" "${DL20_DATASET}" "${DL20_RUN}" "results/maxcontext_dualend/phase2/qwen3.5-9b-dl20" forward
submit_order_job "Qwen/Qwen3.5-9B" "${DL20_DATASET}" "${DL20_RUN}" "results/maxcontext_dualend/phase2/qwen3.5-9b-dl20" inverse
submit_order_job "Qwen/Qwen3.5-9B" "${DL20_DATASET}" "${DL20_RUN}" "results/maxcontext_dualend/phase2/qwen3.5-9b-dl20" random
```

Evaluation after completion:

```bash
for FILE in \
  results/maxcontext_dualend/phase2/qwen3-4b-dl19/maxcontext_dualend_forward.txt \
  results/maxcontext_dualend/phase2/qwen3-4b-dl19/maxcontext_dualend_inverse.txt \
  results/maxcontext_dualend/phase2/qwen3-4b-dl19/maxcontext_dualend_random.txt \
  results/maxcontext_dualend/phase2/qwen3-4b-dl20/maxcontext_dualend_forward.txt \
  results/maxcontext_dualend/phase2/qwen3-4b-dl20/maxcontext_dualend_inverse.txt \
  results/maxcontext_dualend/phase2/qwen3-4b-dl20/maxcontext_dualend_random.txt \
  results/maxcontext_dualend/phase2/qwen3.5-9b-dl19/maxcontext_dualend_forward.txt \
  results/maxcontext_dualend/phase2/qwen3.5-9b-dl19/maxcontext_dualend_inverse.txt \
  results/maxcontext_dualend/phase2/qwen3.5-9b-dl19/maxcontext_dualend_random.txt \
  results/maxcontext_dualend/phase2/qwen3.5-9b-dl20/maxcontext_dualend_forward.txt \
  results/maxcontext_dualend/phase2/qwen3.5-9b-dl20/maxcontext_dualend_inverse.txt \
  results/maxcontext_dualend/phase2/qwen3.5-9b-dl20/maxcontext_dualend_random.txt
do
  if [[ "$FILE" == *"dl20"* ]]; then
    QRELS="${QRELS_DL20}"
  else
    QRELS="${QRELS_DL19}"
  fi
  python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 "${QRELS}" "${FILE}"
done
```

## Phase 3 — Matched-hits regression check (1 pair)

Qwen3-8B, DL19, MaxContext pool=50 vs existing DualEnd cocktail at matched `hits=50`.

```bash
sbatch -J mc-reg-max-q3-8b-dl19 \
  -o logs/mc-reg-max-q3-8b-dl19-%j.out \
  -e logs/mc-reg-max-q3-8b-dl19-%j.err \
  experiments/run_maxcontext_dualend.sh \
  "Qwen/Qwen3-8B" \
  "${DL19_DATASET}" \
  "${DL19_RUN}" \
  "results/maxcontext_dualend/phase3/maxcontext/qwen3-8b-dl19" \
  cuda generation 50 512

sbatch -J mc-reg-decocktail-q3-8b-dl19 \
  -o logs/mc-reg-decocktail-q3-8b-dl19-%j.out \
  -e logs/mc-reg-decocktail-q3-8b-dl19-%j.err \
  experiments/run_dualend_bubblesort.sh \
  "Qwen/Qwen3-8B" \
  "${DL19_DATASET}" \
  "${DL19_RUN}" \
  "results/maxcontext_dualend/phase3/dualend_cocktail/qwen3-8b-dl19" \
  cuda generation 3 10 50 512
```

Evaluation after completion:

```bash
python -m pyserini.eval.trec_eval -c -l 2 \
  -m ndcg_cut.10 -m ndcg_cut.100 -m map_cut.10 -m map_cut.100 -m recall.1000 \
  "${QRELS_DL19}" \
  results/maxcontext_dualend/phase3/maxcontext/qwen3-8b-dl19/maxcontext_dualend.txt

python -m pyserini.eval.trec_eval -c -l 2 \
  -m ndcg_cut.10 -m ndcg_cut.100 -m map_cut.10 -m map_cut.100 -m recall.1000 \
  "${QRELS_DL19}" \
  results/maxcontext_dualend/phase3/dualend_cocktail/qwen3-8b-dl19/dualend_bubblesort.txt
```

## Phase 4 — Study A + predeclared baselines (204 runs)

### Phase 4A — Study A pool sweep (60 runs)

6 models × 2 datasets × 5 pool sizes. Each wrapper call below submits 5 MaxContext jobs.

```bash
for MODEL in "${MODELS[@]}"; do
  MODEL_TAG=$(model_tag "${MODEL}")

  sbatch -J "mc-studyA-${MODEL_TAG}-dl19" \
    -o "logs/mc-studyA-${MODEL_TAG}-dl19-%j.out" \
    -e "logs/mc-studyA-${MODEL_TAG}-dl19-%j.err" \
    --wrap "cd /Users/hangli/projects/llm-rankers && \
      bash experiments/run_maxcontext_dualend_pool_sweep.sh \
        \"${MODEL}\" \
        \"${DL19_DATASET}\" \
        \"${DL19_RUN}\" \
        \"results/maxcontext_dualend/phase4/study_a/${MODEL_TAG}-dl19\" \
        cuda generation 512"

  sbatch -J "mc-studyA-${MODEL_TAG}-dl20" \
    -o "logs/mc-studyA-${MODEL_TAG}-dl20-%j.out" \
    -e "logs/mc-studyA-${MODEL_TAG}-dl20-%j.err" \
    --wrap "cd /Users/hangli/projects/llm-rankers && \
      bash experiments/run_maxcontext_dualend_pool_sweep.sh \
        \"${MODEL}\" \
        \"${DL20_DATASET}\" \
        \"${DL20_RUN}\" \
        \"results/maxcontext_dualend/phase4/study_a/${MODEL_TAG}-dl20\" \
        cuda generation 512"
done
```

### Phase 4B — Predeclared matched-hits baselines (144 runs)

Anchors: `pool_size ∈ {10, 30, 50}`. Baselines: `topdown_heapsort`, `topdown_bubblesort`, `dualend_bubblesort`, `dualend_selection`.

```bash
for MODEL in "${MODELS[@]}"; do
  MODEL_TAG=$(model_tag "${MODEL}")

  for DATASET in "${DL19_DATASET}" "${DL20_DATASET}"; do
    if [[ "${DATASET}" == *"2020"* ]]; then
      RUN_PATH="${DL20_RUN}"
      DATASET_TAG="dl20"
    else
      RUN_PATH="${DL19_RUN}"
      DATASET_TAG="dl19"
    fi

    for POOL_SIZE in 10 30 50; do
      BASE_DIR="results/maxcontext_dualend/phase4/baselines/${MODEL_TAG}-${DATASET_TAG}/pool${POOL_SIZE}"

      sbatch -J "mc-tdheap-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}" \
        -o "logs/mc-tdheap-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}-%j.out" \
        -e "logs/mc-tdheap-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}-%j.err" \
        experiments/run_topdown_heapsort.sh \
        "${MODEL}" "${DATASET}" "${RUN_PATH}" "${BASE_DIR}/topdown_heapsort" \
        cuda generation 3 10 "${POOL_SIZE}" 512

      sbatch -J "mc-tdbubble-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}" \
        -o "logs/mc-tdbubble-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}-%j.out" \
        -e "logs/mc-tdbubble-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}-%j.err" \
        experiments/run_topdown_bubblesort.sh \
        "${MODEL}" "${DATASET}" "${RUN_PATH}" "${BASE_DIR}/topdown_bubblesort" \
        cuda generation 3 10 "${POOL_SIZE}" 512

      sbatch -J "mc-decocktail-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}" \
        -o "logs/mc-decocktail-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}-%j.out" \
        -e "logs/mc-decocktail-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}-%j.err" \
        experiments/run_dualend_bubblesort.sh \
        "${MODEL}" "${DATASET}" "${RUN_PATH}" "${BASE_DIR}/dualend_bubblesort" \
        cuda generation 3 10 "${POOL_SIZE}" 512

      sbatch -J "mc-deselect-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}" \
        -o "logs/mc-deselect-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}-%j.out" \
        -e "logs/mc-deselect-${MODEL_TAG}-${DATASET_TAG}-p${POOL_SIZE}-%j.err" \
        experiments/run_dualend_selection.sh \
        "${MODEL}" "${DATASET}" "${RUN_PATH}" "${BASE_DIR}/dualend_selection" \
        cuda generation 3 10 "${POOL_SIZE}" 512
    done
  done
done
```

## Phase 5 — Study B passage-length sweep (96 runs)

Set the predeclared pool size after Study A:

```bash
PREDECLARED_POOL_SIZE=30
CONTROL_METHOD=selection
```

Treatment arm: 48 MaxContext runs. Each wrapper call below submits 4 jobs (`pl ∈ {64, 128, 256, 512}`).

```bash
for MODEL in "${MODELS[@]}"; do
  MODEL_TAG=$(model_tag "${MODEL}")

  sbatch -J "mc-studyB-max-${MODEL_TAG}-dl19" \
    -o "logs/mc-studyB-max-${MODEL_TAG}-dl19-%j.out" \
    -e "logs/mc-studyB-max-${MODEL_TAG}-dl19-%j.err" \
    --wrap "cd /Users/hangli/projects/llm-rankers && \
      bash experiments/run_maxcontext_dualend_pl_sweep.sh \
        \"${MODEL}\" \
        \"${DL19_DATASET}\" \
        \"${DL19_RUN}\" \
        \"results/maxcontext_dualend/phase5/study_b/treatment/${MODEL_TAG}-dl19\" \
        cuda generation ${PREDECLARED_POOL_SIZE}"

  sbatch -J "mc-studyB-max-${MODEL_TAG}-dl20" \
    -o "logs/mc-studyB-max-${MODEL_TAG}-dl20-%j.out" \
    -e "logs/mc-studyB-max-${MODEL_TAG}-dl20-%j.err" \
    --wrap "cd /Users/hangli/projects/llm-rankers && \
      bash experiments/run_maxcontext_dualend_pl_sweep.sh \
        \"${MODEL}\" \
        \"${DL20_DATASET}\" \
        \"${DL20_RUN}\" \
        \"results/maxcontext_dualend/phase5/study_b/treatment/${MODEL_TAG}-dl20\" \
        cuda generation ${PREDECLARED_POOL_SIZE}"
done
```

Control arm: 48 existing DualEnd-nc3 runs at the same predeclared pool size. `CONTROL_METHOD=selection` keeps the selection-sort family aligned with MaxContext while preserving the original DualEnd direction and `num_child=3` control constraint from IDEA_007.

```bash
for MODEL in "${MODELS[@]}"; do
  MODEL_TAG=$(model_tag "${MODEL}")

  for DATASET in "${DL19_DATASET}" "${DL20_DATASET}"; do
    if [[ "${DATASET}" == *"2020"* ]]; then
      RUN_PATH="${DL20_RUN}"
      DATASET_TAG="dl20"
      QRELS="${QRELS_DL20}"
    else
      RUN_PATH="${DL19_RUN}"
      DATASET_TAG="dl19"
      QRELS="${QRELS_DL19}"
    fi

    for PASSAGE_LENGTH in 64 128 256 512; do
      OUTPUT_DIR="results/maxcontext_dualend/phase5/study_b/control/${MODEL_TAG}-${DATASET_TAG}/pl${PASSAGE_LENGTH}"

      if [[ "${CONTROL_METHOD}" == "selection" ]]; then
        sbatch -J "mc-studyB-ctrl-${MODEL_TAG}-${DATASET_TAG}-pl${PASSAGE_LENGTH}" \
          -o "logs/mc-studyB-ctrl-${MODEL_TAG}-${DATASET_TAG}-pl${PASSAGE_LENGTH}-%j.out" \
          -e "logs/mc-studyB-ctrl-${MODEL_TAG}-${DATASET_TAG}-pl${PASSAGE_LENGTH}-%j.err" \
          experiments/run_dualend_selection.sh \
          "${MODEL}" "${DATASET}" "${RUN_PATH}" "${OUTPUT_DIR}" \
          cuda generation 3 10 "${PREDECLARED_POOL_SIZE}" "${PASSAGE_LENGTH}"
      else
        sbatch -J "mc-studyB-ctrl-${MODEL_TAG}-${DATASET_TAG}-pl${PASSAGE_LENGTH}" \
          -o "logs/mc-studyB-ctrl-${MODEL_TAG}-${DATASET_TAG}-pl${PASSAGE_LENGTH}-%j.out" \
          -e "logs/mc-studyB-ctrl-${MODEL_TAG}-${DATASET_TAG}-pl${PASSAGE_LENGTH}-%j.err" \
          experiments/run_dualend_bubblesort.sh \
          "${MODEL}" "${DATASET}" "${RUN_PATH}" "${OUTPUT_DIR}" \
          cuda generation 3 10 "${PREDECLARED_POOL_SIZE}" "${PASSAGE_LENGTH}"
      fi
    done
  done
done
```

Evaluation helper for any completed directory:

```bash
python -m pyserini.eval.trec_eval -c -l 2 \
  -m ndcg_cut.10 -m ndcg_cut.100 -m map_cut.10 -m map_cut.100 -m recall.1000 \
  "${QRELS_DL19}" \
  results/maxcontext_dualend/phase4/study_a/qwen-qwen3-4b-dl19/pool10/maxcontext_dualend.txt
```

Convenience sweeps after runs finish:

```bash
bash experiments/eval_all.sh \
  results/maxcontext_dualend/phase1/qwen3-4b-dl19 \
  results/maxcontext_dualend/phase3/maxcontext/qwen3-8b-dl19 \
  results/maxcontext_dualend/phase3/dualend_cocktail/qwen3-8b-dl19
```

Notes:
- Monitor jobs with `squeue -u $USER`.
- The base MaxContext logs go under `results/analysis/position_bias_maxcontext/...` by design; do not merge them into the legacy `results/analysis/position_bias/...` tree.
- Phase 5 assumes exactly one control method to keep the study at 96 logical runs. The sheet defaults that control to `dualend_selection`.
