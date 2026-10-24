#!/usr/bin/env bash
#SBATCH --job-name=flow-channel-smoke
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
set -euo pipefail
source "${PYTHON_ENV:?}/bin/activate"
cd "${REPO_ROOT:?}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
nvidia-smi
srun --ntasks=4 python smoke_training.py --device cuda --ddp
if [[ -n "${ORIGINAL_REPO:-}" ]]; then
  srun --ntasks=1 --gres=gpu:1 python audit_baseline.py --device cuda --original-repo "$ORIGINAL_REPO"
fi
if [[ "${FULL_MODEL_SMOKE:-0}" == 1 ]]; then
  srun --ntasks=4 python full_cube_smoke.py
fi
