#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --job-name=ablation_nc
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=20:00:00
#SBATCH --account=a_ai_collab

module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
module load cuda/12.2
conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
cd /scratch/project/neural_ir/hang/llm-rankers

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

MODEL=${1:-"google/flan-t5-xl"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/ablation-nc"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
K=${7:-10}
HITS=${8:-100}
PASSAGE_LENGTH=${9:-128}

mkdir -p ${OUTPUT_DIR}

echo "=== num_child Ablation (DualEnd-Cocktail) ==="
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"

# c=3 is already covered in the main experiments; run c=2, 5, 7
for NC in 2 5 7; do
    echo ""
    echo ">>> DualEnd-Cocktail with num_child=${NC}"
    python run.py \
        run --model_name_or_path ${MODEL} \
            --ir_dataset_name ${DATASET} \
            --run_path ${RUN_PATH} \
            --save_path ${OUTPUT_DIR}/dualend_cocktail_nc${NC}.txt \
            --device ${DEVICE} \
            --scoring ${SCORING} \
            --hits ${HITS} \
            --passage_length ${PASSAGE_LENGTH} \
        setwise --num_child ${NC} \
                --method bubblesort \
                --k ${K} \
                --direction dualend \
        2>&1 | tee ${OUTPUT_DIR}/dualend_cocktail_nc${NC}.log
done

echo ""
echo "=== num_child ablation complete ==="
