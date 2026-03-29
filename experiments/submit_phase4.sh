#!/bin/bash
# Submit all Phase 4 SLURM jobs
#
# Usage: bash experiments/submit_phase4.sh
#
# This submits:
#   - 9 Phase 4A jobs (one per model, GPU required, ~20h each)
#   - 1 Phase 4B/C/D/E job (post-hoc analysis, CPU only, ~1h)
#
# Phase 4B/C/D/E depends on Phase 1-3 results already being in results/.
# Phase 4A re-runs 3 methods per model with comparison logging, then
# analyzes position bias. These are independent of each other.
#
# For Qwen3.5-4B: uncomment the conda activate line for qwen35_env
# in slurm_phase4a_position_bias.sh before submitting.

set -e

echo "=============================================="
echo "Submitting Phase 4 SLURM Jobs"
echo "=============================================="

# --- Phase 4A: Position Bias (GPU jobs, one per model) ---
echo ""
echo ">>> Phase 4A: Position Bias Analysis (9 GPU jobs)"
echo ""

# Flan-T5 models (passage_length=128)
echo "  Submitting: flan-t5-large..."
sbatch -J p4a-t5l -o logs/phase4a-flan-t5-large-%j.out -e logs/phase4a-flan-t5-large-%j.err \
    experiments/slurm_phase4a_position_bias.sh google/flan-t5-large 128

echo "  Submitting: flan-t5-xl..."
sbatch -J p4a-t5xl -o logs/phase4a-flan-t5-xl-%j.out -e logs/phase4a-flan-t5-xl-%j.err \
    experiments/slurm_phase4a_position_bias.sh google/flan-t5-xl 128

echo "  Submitting: flan-t5-xxl..."
sbatch -J p4a-t5xxl -o logs/phase4a-flan-t5-xxl-%j.out -e logs/phase4a-flan-t5-xxl-%j.err \
    experiments/slurm_phase4a_position_bias.sh google/flan-t5-xxl 128

# Qwen3 models (passage_length=512)
echo "  Submitting: Qwen3-4B..."
sbatch -J p4a-q3-4b -o logs/phase4a-qwen3-4b-%j.out -e logs/phase4a-qwen3-4b-%j.err \
    experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3-4B 512

echo "  Submitting: Qwen3-8B..."
sbatch -J p4a-q3-8b -o logs/phase4a-qwen3-8b-%j.out -e logs/phase4a-qwen3-8b-%j.err \
    experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3-8B 512

echo "  Submitting: Qwen3-14B..."
sbatch -J p4a-q3-14b -o logs/phase4a-qwen3-14b-%j.out -e logs/phase4a-qwen3-14b-%j.err \
    experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3-14B 512

# Qwen3.5 models (passage_length=512)
# NOTE: Edit slurm_phase4a_position_bias.sh to use qwen35_env before submitting
echo "  Submitting: Qwen3.5-4B..."
sbatch -J p4a-q35-4b -o logs/phase4a-qwen3.5-4b-%j.out -e logs/phase4a-qwen3.5-4b-%j.err \
    experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3.5-4B 512

echo "  Submitting: Qwen3.5-9B..."
sbatch -J p4a-q35-9b -o logs/phase4a-qwen3.5-9b-%j.out -e logs/phase4a-qwen3.5-9b-%j.err \
    experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3.5-9B 512

echo "  Submitting: Qwen3.5-27B..."
sbatch -J p4a-q35-27b -o logs/phase4a-qwen3.5-27b-%j.out -e logs/phase4a-qwen3.5-27b-%j.err \
    experiments/slurm_phase4a_position_bias.sh Qwen/Qwen3.5-27B 512

# --- Phase 4B/C/D/E: Post-hoc (CPU job, single) ---
echo ""
echo ">>> Phase 4B/C/D/E: Post-hoc Analyses (1 CPU job)"
echo ""

echo "  Submitting: post-hoc analyses..."
sbatch -o logs/phase4bce-posthoc-%j.out -e logs/phase4bce-posthoc-%j.err \
    experiments/slurm_phase4bce_posthoc.sh

echo ""
echo "=============================================="
echo "All Phase 4 jobs submitted!"
echo "  9 GPU jobs (Phase 4A position bias)"
echo "  1 CPU job  (Phase 4B/C/D/E post-hoc)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "=============================================="
