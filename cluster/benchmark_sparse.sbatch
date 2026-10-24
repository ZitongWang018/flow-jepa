#!/usr/bin/env bash
#SBATCH --job-name=flow-sparse-latency
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
source "${PYTHON_ENV:?}/bin/activate"
export CPATH="$PYTHON_ENV/include:$PYTHON_ENV/include/python3.10${CPATH:+:$CPATH}"
cd "${REPO_ROOT:?}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
python smoke_training.py --device cuda --sparse
python benchmark_sparse_latency.py --trials 8 --precision "${BENCHMARK_PRECISION:-bf16}"
