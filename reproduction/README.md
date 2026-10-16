# Reproduction setup

This repository contains the Flow-JEPA source code copied from upstream revision `ab73e7c50435e3a1fffb4d3f6d175445e57a2f4e`; see the root [README](../README.md) and [MIT license](../LICENSE). The files under `cluster/` and this folder provide the reproduction environment and four-environment Slurm workflow.

## Environment

Use Python 3.10 and a CUDA 12.8 PyTorch build:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r reproduction/requirements.txt
```

## Data and initialization

The scripts download public files from Hugging Face and verify their SHA-256 hashes before extraction. Set `DATA_ROOT` to a filesystem with enough space:

```bash
export DATA_ROOT=/path/to/data
bash cluster/prepare_data.sh
bash cluster/fetch_init.sh
```

Expected HDF5 files: `datasets/pusht_expert_train.h5`, `datasets/reacher.h5`, `datasets/tworoom.h5`, and `datasets/cube_single_expert.h5`. LeWM weights are stored under `$DATA_ROOT/checkpoints/flow-jepa/lewm-initialization`.

## Training

The Slurm example requests four GPUs on one node. Adjust scheduler resource directives for the target site. Set `PYTHON_ENV`, `DATA_ROOT`, and optionally `RUN_ROOT` / `CHECKPOINT_ROOT`, then submit:

```bash
export PYTHON_ENV=/path/to/venv
export DATA_ROOT=/path/to/data
export RUN_ROOT=/path/to/runs
sbatch cluster/train_four_envs.sbatch
```

Each task runs for 20 epochs. The launcher uses local batch size 32 on four DDP ranks, preserving global batch size 128. Tasks run sequentially: PushT, Reacher, TwoRoom, Cube.

## Paper parameter mapping

Common settings follow the paper: 224-pixel observations, ViT-Tiny/14 encoder, 192-dimensional latent/action embeddings, six predictor blocks, 16 heads, 64-dimensional heads, MLP width 2048, history size 1, prediction horizon 5, 20 epochs, global batch size 128, and SIGReg with 17 knots / 1024 projections.

| Environment | Flow source and attention | Dropout | SIGReg | Visual initialization |
| --- | --- | ---: | ---: | --- |
| PushT | noisy current state, causal future attention, noise scale 0.5 | 0.0 | 0.09 | LeWM encoder/projector, frozen |
| Reacher | noisy current state, causal future attention, noise scale 0.5 | 0.1 | 0.1 | LeWM initialization, encoder trainable |
| TwoRoom | standard normal noise, bidirectional future attention | 0.1 | 0.1 | trained from scratch |
| Cube | standard normal noise, bidirectional future attention | 0.1 | 0.1 | trained from scratch |

The paper reports downstream control metrics. Training alone does not produce those evaluation results; run the matching evaluation protocol before comparing reported scores.
