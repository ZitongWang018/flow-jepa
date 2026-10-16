# Flow-JEPA reproduction

This repository includes the Flow-JEPA training and evaluation code, together with the reproduction setup for PushT, Reacher, TwoRoom, and Cube.

## Source and attribution

The model, training, evaluation, and configuration files are copied from [HuoYanchen/Flow-JEPA](https://github.com/HuoYanchen/Flow-JEPA), revision `ab73e7c50435e3a1fffb4d3f6d175445e57a2f4e). The upstream MIT license and copyright notice are retained in [LICENSE](LICENSE). Reproduction-specific launchers, dependency pins, and paper parameter notes are in [reproduction/README.md](reproduction/README.md).

## Quick start

Use Python 3.10 with CUDA 12.8 PyTorch, install `reproduction/requirements.txt`, prepare the public datasets and LeWM initialization with the scripts in `cluster/`, then submit `cluster/train_four_envs.sbatch` through Slurm.

See [reproduction/README.md](reproduction/README.md) for full setup steps and the parameter mapping.
