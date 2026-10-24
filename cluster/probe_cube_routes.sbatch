#!/usr/bin/env bash
#SBATCH --job-name=cube-route-probe
#SBATCH --partition=RTX4090
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
REPO_ROOT="${REPO_ROOT:?}"
DATA_ROOT="${DATA_ROOT:?}"
RUN_ROOT="${RUN_ROOT:?}"
PYTHON_ENV="${PYTHON_ENV:?}"
source "$PYTHON_ENV/bin/activate"
export STABLEWM_HOME="$RUN_ROOT/artifacts"
export LOCAL_DATASET_DIR="$DATA_ROOT"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export MPLCONFIGDIR="$RUN_ROOT/cache/matplotlib"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export ROUTE_PROBE_OUTPUT="$RUN_ROOT/analysis/learned384_epoch20_inference_routes_10_starts.json"
mkdir -p "$(dirname "$ROUTE_PROBE_OUTPUT")" "$RUN_ROOT/tmp/$SLURM_JOB_ID"
export TMPDIR="$RUN_ROOT/tmp/$SLURM_JOB_ID"
cd "$REPO_ROOT"
ckpt="$STABLEWM_HOME/checkpoints/cube-learned384-seed3072/weights_epoch_20.pt"
test -s "$ckpt"
python probe_routes_eval.py --config-name cube \
  "policy=$ckpt" "cache_dir=$DATA_ROOT" eval.num_eval=10 seed=42 flow_seed=42 \
  output.save_video=false \
  "output.dir=$RUN_ROOT/evaluation/cube-learned384-seed3072/probe-10/clean"
test -s "$ROUTE_PROBE_OUTPUT"
