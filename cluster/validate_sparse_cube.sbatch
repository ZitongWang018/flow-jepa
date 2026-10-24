#!/usr/bin/env bash
#SBATCH --job-name=flow-sparse-validate
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
set -euo pipefail
source "${PYTHON_ENV:?}/bin/activate"
export CPATH="$PYTHON_ENV/include:$PYTHON_ENV/include/python3.10${CPATH:+:$CPATH}"
cd "${REPO_ROOT:?}"
export ARM=sparse336_active192
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "validation_start utc=$(date -u +%FT%TZ)"
nvidia-smi
srun --ntasks=4 python smoke_training.py --device cuda --ddp --sparse
srun --ntasks=4 python full_cube_smoke.py
srun --ntasks=1 --gres=gpu:1 python benchmark_sparse_latency.py --trials 8
echo "validation_complete utc=$(date -u +%FT%TZ)"
