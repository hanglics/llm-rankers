#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --job-name=tdbg
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --account=a_ai_collab
#SBATCH --exclude=bun116,bun073

module load anaconda3/2023.09-0
source $EBROOTANACONDA3/etc/profile.d/conda.sh
module load cuda/12.2.0
conda activate /scratch/project/neural_ir/hang/llm-rankers/ranker_env
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
METHOD=${11:-"heapsort"}

mkdir -p ${OUTPUT_DIR}

ANALYSIS_DIR="results/analysis/position_bias/$(basename ${OUTPUT_DIR})"
mkdir -p ${ANALYSIS_DIR}

export HF_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export HF_DATASETS_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/hf
export PYSERINI_CACHE=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini
export IR_DATASETS_HOME=/scratch/project/neural_ir/hang/llm-rankers/.cache/pyserini

CHARACTER_SCHEME_ARGS=()
if [ "${NUM_CHILD}" -ge 23 ]; then
    CHARACTER_SCHEME_ARGS=(--character_scheme bigrams_aa_zz)
fi

python run.py \
    run --model_name_or_path ${MODEL} \
        --ir_dataset_name ${DATASET} \
        --run_path ${RUN_PATH} \
        --save_path ${OUTPUT_DIR}/topdown_${METHOD}.txt \
        --device ${DEVICE} \
        --scoring ${SCORING} \
        --hits ${HITS} \
        --passage_length ${PASSAGE_LENGTH} \
        --log_comparisons ${ANALYSIS_DIR}/topdown_${METHOD}_comparisons.jsonl \
    setwise --num_child ${NUM_CHILD} \
            --method ${METHOD} \
            --k ${K} \
            --direction topdown \
            "${CHARACTER_SCHEME_ARGS[@]}" \
    2>&1 | tee ${OUTPUT_DIR}/topdown_${METHOD}.log
