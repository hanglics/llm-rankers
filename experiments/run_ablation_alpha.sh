#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --job-name=ablation_alpha
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=20:00:00
#SBATCH --account=a_ai_collab

module load anaconda3/2023.09-0
# module load java/21.0.8
source $EBROOTANACONDA3/etc/profile.d/conda.sh
module load cuda/12.2
conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
# conda activate /scratch/project/neural_ir/hang/llm-rankers/qwen35_env
cd /scratch/project/neural_ir/hang/llm-rankers

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

MODEL=${1:-"google/flan-t5-xl"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/ablation-alpha"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
NUM_CHILD=${7:-3}
K=${8:-10}
HITS=${9:-100}

default_passage_length=128
if [[ "${MODEL,,}" == *"qwen"* ]]; then
    default_passage_length=512
fi
PASSAGE_LENGTH=${10:-${default_passage_length}}

mkdir -p ${OUTPUT_DIR}

echo "=== Alpha Ablation (BiDir-Weighted) ==="
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Passage length: ${PASSAGE_LENGTH}"

if [[ "${MODEL,,}" == *"qwen"* ]] && (( PASSAGE_LENGTH < 512 )); then
    echo "WARNING: Qwen-family runs are usually evaluated with --passage_length 512; got ${PASSAGE_LENGTH}."
fi

# alpha=0.7 is already covered in the main experiments; run 0.3, 0.5, 0.9
for ALPHA in 0.3 0.5 0.9; do
    echo ""
    echo ">>> BiDir-Weighted with alpha=${ALPHA}"
    python run.py \
        run --model_name_or_path ${MODEL} \
            --ir_dataset_name ${DATASET} \
            --run_path ${RUN_PATH} \
            --save_path ${OUTPUT_DIR}/bidir_weighted_a${ALPHA}.txt \
            --device ${DEVICE} \
            --scoring ${SCORING} \
            --hits ${HITS} \
            --passage_length ${PASSAGE_LENGTH} \
        setwise --num_child ${NUM_CHILD} \
                --method heapsort \
                --k ${K} \
                --direction bidirectional \
                --fusion weighted \
                --alpha ${ALPHA} \
        2>&1 | tee ${OUTPUT_DIR}/bidir_weighted_a${ALPHA}.log
done

# Also run CombSUM fusion for comparison
echo ""
echo ">>> BiDir-CombSUM"
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/bidir_combsum.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method heapsort \
            --k ${K} \
            --direction bidirectional \
            --fusion combsum \
    2>&1 | tee ${OUTPUT_DIR}/bidir_combsum.log

echo ""
echo "=== Alpha ablation complete ==="
