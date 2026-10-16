#!/usr/bin/env bash
#SBATCH --job-name=flow-jepa
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$DATA_ROOT/checkpoints/flow-jepa}"
PYTHON_ENV="${PYTHON_ENV:-}"

if [[ -n "$PYTHON_ENV" ]]; then
  source "$PYTHON_ENV/bin/activate"
fi

export STABLEWM_HOME="$CHECKPOINT_ROOT"
export LOCAL_DATASET_DIR="$DATA_ROOT"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export TMPDIR="$RUN_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$CHECKPOINT_ROOT" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" "$RUN_ROOT/logs"

cd "$REPO_ROOT"
echo "start_utc=$(date -u +%FT%TZ)"
echo "source_commit=$(git rev-parse HEAD)"
nvidia-smi
python - <<'PY'
import torch
import lightning
import stable_pretraining
import stable_worldmodel
print("torch", torch.__version__, "cuda", torch.version.cuda, "visible_gpus", torch.cuda.device_count())
print("lightning", lightning.__version__)
print("stable_pretraining", getattr(stable_pretraining, "__version__", "unknown"))
print("stable_worldmodel", getattr(stable_worldmodel, "__version__", "unknown"))
assert torch.cuda.is_available() and torch.cuda.device_count() == 4
PY

for f in \
  "$DATA_ROOT/datasets/pusht_expert_train.h5" \
  "$DATA_ROOT/datasets/tworoom.h5" \
  "$DATA_ROOT/datasets/reacher.h5" \
  "$DATA_ROOT/datasets/cube_single_expert.h5" \
  "$CHECKPOINT_ROOT/lewm-initialization/lewm-pusht-weights.pt" \
  "$CHECKPOINT_ROOT/lewm-initialization/lewm-reacher-weights.pt"; do
  test -s "$f" || { echo "missing_required_file=$f" >&2; exit 3; }
done

run_task() {
  local data_cfg="$1"
  local name="$2"
  shift 2
  echo "training_begin task=$name time=$(date -u +%FT%TZ)"
  srun --ntasks=4 python train.py \
    "data=$data_cfg" \
    "output_model_name=$name" \
    "subdir=$name" \
    trainer.max_epochs=20 \
    trainer.devices=4 \
    +trainer.strategy=ddp \
    loader.batch_size=32 \
    hydra.run.dir="$RUN_ROOT/hydra/$name/rank-\${oc.env:SLURM_PROCID}" \
    "$@"
  echo "training_end task=$name time=$(date -u +%FT%TZ)"
}

run_task pusht flow-jepa-pusht \
  model.predictor.causal_self_attention=true \
  model.predictor.dropout=0 \
  flow_matching.source=noisy_current \
  flow_matching.source_noise_scale=0.5 \
  loss.sigreg.weight=0.09 \
  pretrained_visual_encoder.enabled=true \
  pretrained_visual_encoder.checkpoint="$CHECKPOINT_ROOT/lewm-initialization/lewm-pusht-weights.pt" \
  pretrained_visual_encoder.freeze=true

run_task dmc flow-jepa-reacher \
  model.predictor.causal_self_attention=true \
  model.predictor.dropout=0.1 \
  flow_matching.source=noisy_current \
  flow_matching.source_noise_scale=0.5 \
  loss.sigreg.weight=0.1 \
  pretrained_visual_encoder.enabled=true \
  pretrained_visual_encoder.checkpoint="$CHECKPOINT_ROOT/lewm-initialization/lewm-reacher-weights.pt" \
  pretrained_visual_encoder.freeze=false

run_task tworoom flow-jepa-tworoom \
  model.predictor.causal_self_attention=false \
  model.predictor.dropout=0.1 \
  flow_matching.source=standard_noise \
  flow_matching.source_noise_scale=1.0 \
  loss.sigreg.weight=0.1

run_task ogb flow-jepa-cube \
  model.predictor.causal_self_attention=false \
  model.predictor.dropout=0.1 \
  flow_matching.source=standard_noise \
  flow_matching.source_noise_scale=1.0 \
  loss.sigreg.weight=0.1
