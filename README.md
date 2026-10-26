# Flow-JEPA reproduction and Channelwise Cube experiment

This repository contains the original [Flow-JEPA](https://github.com/HuoYanchen/Flow-JEPA)
model and the reproducibility files for its tasks. The baseline source follows
upstream revision `ab73e7c50435e3a1fffb4d3f6d175445e57a2f4e`.

The completed **336 total / 192 active** Cube channelwise experiment is isolated
in [`channelwise/`](channelwise/). That folder contains its model, sparse
inference code, training and evaluation entry points, config, latency benchmark,
route probe, results, and load-balance heatmap. Data and model checkpoints are
not stored in this repository.

For the original four-task reproduction setup, see
[`reproduction/README.md`](reproduction/README.md). The project inherits the
source license in [`LICENSE`](LICENSE).
