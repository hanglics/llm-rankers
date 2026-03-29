#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --job-name=bdweighted
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

MODEL=${1:-"Qwen/Qwen3-4B"}
DATASET=${2:-"msmarco-passage/trec-dl-2019/judged"}
RUN_PATH=${3:-"runs/bm25/run.msmarco-v1-passage.bm25-default.dl19.txt"}
OUTPUT_DIR=${4:-"results/extended_setwise/qwen3-4b-dl19"}
DEVICE=${5:-"cuda"}
SCORING=${6:-"generation"}
NUM_CHILD=${7:-3}
K=${8:-10}
HITS=${9:-100}
PASSAGE_LENGTH=${10:-512}
BIDIRECTION_WEIGHTED_ALPHA=${11:-0.7}

mkdir -p ${OUTPUT_DIR}

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

# NOTE: Bidirectional does not support --log_comparisons (log path is not
# propagated to sub-rankers). Use TopDown + BottomUp logs separately instead.
python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/bidirectional_weighted_a${BIDIRECTION_WEIGHTED_ALPHA}.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
    setwise --num_child ${NUM_CHILD} \
            --method heapsort \
            --k ${K} \
            --direction bidirectional \
            --fusion weighted \
            --alpha ${BIDIRECTION_WEIGHTED_ALPHA} \
    2>&1 | tee ${OUTPUT_DIR}/bidirectional_weighted_a${BIDIRECTION_WEIGHTED_ALPHA}.log