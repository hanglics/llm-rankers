#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --job-name=ablation_pl
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=20:00:00
#SBATCH --account=a_ai_collab

module load anaconda3/2023.09-0
# module load java/21.0.8
source $EBROOTANACONDA3/etc/profile.d/conda.sh
module load cuda/12.2.0
conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
# conda activate /scratch/project/neural_ir/hang/llm-rankers/qwen35_env
cd /scratch/project/neural_ir/hang/llm-rankers

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

# Passage length ablation for a given model and method
# Tests passage_length ∈ {64, 128, 256, 512}
#
# Usage:
#   bash experiments/run_ablation_passage_length.sh <model> <dataset> <run_path> <output_dir> \
#       [device] [scoring] [num_child] [k] [hits] [direction] [method]
#
# Example (Flan-T5-XL, TopDown-Heap on DL19):
#   bash experiments/run_ablation_passage_length.sh google/flan-t5-xl \
#       msmarco-passage/trec-dl-2019/judged \
#       runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
#       results/ablation-pl/flan-t5-xl-dl19
#
# Example (Qwen3-4B, DualEnd-Cocktail on DL19):
#   bash experiments/run_ablation_passage_length.sh Qwen/Qwen3-4B \
#       msmarco-passage/trec-dl-2019/judged \
#       runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt \
#       results/ablation-pl/qwen3-4b-dl19 \
#       cuda generation 3 10 100 dualend bubblesort

MODEL=${1:-"google/flan-t5-xl"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/ablation-pl"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
NUM_CHILD=${7:-3}
K=${8:-10}
HITS=${9:-100}
DIRECTION=${10:-"topdown"}
METHOD=${11:-"heapsort"}

mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "Passage Length Ablation"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Direction: ${DIRECTION}, Method: ${METHOD}"
echo "=============================================="

for PL in 64 128 256 512; do
    echo ""
    echo ">>> passage_length=${PL}"
    python run.py \
        run --model_name_or_path ${MODEL} \
            --ir_dataset_name ${DATASET} \
            --run_path ${RUN_PATH} \
            --save_path ${OUTPUT_DIR}/${DIRECTION}_${METHOD}_pl${PL}.txt \
            --device ${DEVICE} \
            --scoring ${SCORING} \
            --hits ${HITS} \
            --passage_length ${PL} \
        setwise --num_child ${NUM_CHILD} \
                --method ${METHOD} \
                --k ${K} \
                --direction ${DIRECTION} \
        2>&1 | tee ${OUTPUT_DIR}/${DIRECTION}_${METHOD}_pl${PL}.log
done

echo ""
echo "=============================================="
echo "Passage length ablation complete!"
echo "=============================================="

# Quick evaluation if we can detect the qrels
if [[ "${DATASET}" = *"dl-2019"* ]]; then
    QRELS="dl19-passage"
elif [[ "${DATASET}" = *"dl-2020"* ]]; then
    QRELS="dl20-passage"
else
    echo "Non-TREC-DL dataset — skipping auto-evaluation"
    exit 0
fi

echo ""
echo "Results Summary:"
echo "-------------------------------------------"
printf "%-45s %-10s\n" "Config" "NDCG@10"
echo "-------------------------------------------"
for PL in 64 128 256 512; do
    f="${OUTPUT_DIR}/${DIRECTION}_${METHOD}_pl${PL}.txt"
    if [ -f "$f" ]; then
        score=$(python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 ${QRELS} ${f} 2>/dev/null | grep "ndcg_cut_10" | awk '{print $3}')
        printf "%-45s %-10s\n" "${DIRECTION}_${METHOD}_pl${PL}" "${score}"
    fi
done
echo "-------------------------------------------"
