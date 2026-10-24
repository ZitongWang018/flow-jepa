#!/usr/bin/env bash
#SBATCH --job-name=flow-cube-channels
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
set -euo pipefail
REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
PYTHON_ENV="${PYTHON_ENV:?Set PYTHON_ENV}"
ARM="${ARM:-learned384}"
export ARM
source "$PYTHON_ENV/bin/activate"
export CPATH="$PYTHON_ENV/include:$PYTHON_ENV/include/python3.10${CPATH:+:$CPATH}"
export STABLEWM_HOME="$RUN_ROOT/artifacts"
export LOCAL_DATASET_DIR="$DATA_ROOT"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export TMPDIR="$RUN_ROOT/tmp/$SLURM_JOB_ID"
export MPLCONFIGDIR="$RUN_ROOT/cache/matplotlib"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
mkdir -p "$STABLEWM_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" "$MPLCONFIGDIR"
cd "$REPO_ROOT"
test -s "$DATA_ROOT/datasets/cube_single_expert.h5"
test -f "$DATA_ROOT/datasets/cube.ready"
if [[ ! -e "$STABLEWM_HOME/datasets" ]]; then ln -s "$DATA_ROOT/datasets" "$STABLEWM_HOME/datasets"; fi
name="cube-$ARM-seed3072"
mkdir -p "$RUN_ROOT/telemetry"
nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?}" \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv -l 30 > "$RUN_ROOT/telemetry/$name-$SLURM_JOB_ID.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
echo "training_begin arm=$ARM utc=$(date -u +%FT%TZ)"
git rev-parse HEAD
sha256sum channel_routing.py module.py sparse_linear.py jepa.py train.py "config/train/routing/$ARM.yaml"
nvidia-smi
# A short synthetic four-rank update exercises routing, credit and collectives.
smoke_args=()
if [[ "$ARM" == sparse336_active192 ]]; then smoke_args+=(--sparse); fi
srun --ntasks=4 python smoke_training.py --ddp --device cuda "${smoke_args[@]}"
strategy=ddp
if [[ "$ARM" == sparse336_active192 ]]; then strategy=ddp_find_unused_parameters_true; fi
srun --ntasks=4 python train.py \
  data=ogb "+routing=$ARM" \
  "output_model_name=$name" "subdir=$name" seed=3072 \
  trainer.max_epochs=20 trainer.devices=4 "+trainer.strategy=$strategy" \
  +trainer.enable_progress_bar=false +trainer.log_every_n_steps=9 \
  loader.batch_size=32 num_workers=6 \
  model.predictor.causal_self_attention=false model.predictor.dropout=0.1 \
  flow_matching.source=standard_noise flow_matching.source_noise_scale=1.0 \
  loss.sigreg.weight=0.1 \
  "hydra.run.dir=$RUN_ROOT/hydra/$name/rank-\${oc.env:SLURM_PROCID}"
echo "training_end arm=$ARM utc=$(date -u +%FT%TZ)"
for epoch in 1 20; do
  ckpt="$STABLEWM_HOME/checkpoints/$name/weights_epoch_$epoch.pt"
  for condition in clean gaussian; do
    overrides=()
    if [[ "$condition" == gaussian ]]; then overrides+=(perturbation.enabled=true); fi
    srun --ntasks=1 --gres=gpu:1 python eval.py --config-name cube \
      "policy=$ckpt" "cache_dir=$DATA_ROOT" eval.num_eval=50 seed=42 flow_seed=42 \
      output.save_video=false "output.dir=$RUN_ROOT/evaluation/$name/epoch-$epoch/$condition" \
      "${overrides[@]}"
    if [[ "$ARM" == sparse336_active192 && "$epoch" == 20 ]]; then
      srun --ntasks=1 --gres=gpu:1 python eval.py --config-name cube \
        "policy=$ckpt" "cache_dir=$DATA_ROOT" eval.num_eval=50 seed=42 flow_seed=42 \
        output.save_video=false sparse_share_route=true cuda_graph_predictor=true \
        "output.dir=$RUN_ROOT/evaluation/$name/epoch-$epoch/shared-$condition" \
        "${overrides[@]}"
    fi
  done
done
echo "experiment_complete arm=$ARM utc=$(date -u +%FT%TZ)"
python cluster/retain_final_cube_checkpoint.py --run-root "$RUN_ROOT" --arm "$ARM"
