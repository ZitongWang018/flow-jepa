# Flow-JEPA Cube: selected-submatrix channels

## Current experiment: `sparse336_active192`

The 192-D visual latent enters a 336-D predictor state. Each of its six blocks
has seven 48-D groups: three shared groups (144-D) and four candidate dynamic
groups. A clip selects one dynamic group in each block, so self-attention,
cross-attention, AdaLN and MLP multiply only the selected 192-D input/output
submatrices. The final readout also reads only the last block's selected 192
coordinates. Inactive coordinates retain their previous residual state and may
be selected in a later block or Euler step. The input/context projections
initialize all 336 candidate coordinates.

The router sees the current noisy state, visible context, visible history
change, action condition and flow time. Its hard choice has unit weight;
matched selected-versus-unselected swaps supervise it every eight optimizer
steps after a 100-step random-routing warmup. Usage is synchronized across
data-parallel ranks and a bounded bias corrects persistent imbalance. The
training objective and Cube settings below match the earlier reproduction.

For Cube CEM, `sparse_share_route=true` optionally chooses one dynamic group
per block and Euler step from the mean candidate features, then computes the
same 192-D submatrix for all 300 action candidates. The training and default
evaluation route each candidate independently. This shared inference option
must be compared on the same checkpoint and evaluation starts before claiming
its task quality. `cuda_graph_predictor=true` captures the fixed-shape Euler
loop for repeated CEM calls; it changes scheduling, not model weights.

Run `smoke_training.py --sparse` and `full_cube_smoke.py` before training.
`benchmark_sparse_latency.py` reports same-GPU original192 and sparse
latencies for 1, 32 and 300 candidates, eager and captured, BF16 or FP32.
For the active Cube run, use `ARM=sparse336_active192` with
`cluster/train_cube_channels.sbatch`. The job evaluates clean and Gaussian
Cube at epochs 1 and 20 using independent routes, and also evaluates the
shared-route option at epoch 20. It retains only the final checkpoint after
all evaluations succeed.

## Earlier write-only study

## Experiment

The earlier comparison is `learned384` against `dense384`, trained from scratch
on the original Cube dataset. Both use the same seed, initialization of all
shared parameters, data split, batch size and training schedule. The learned
arm includes extra router parameters; this is a comparison at the same backbone
width, not equal total parameters or equal compute.

The existing 192-dimensional Flow-JEPA result is a historical reference:
clean 28/50 (56%), Gaussian 18/50 (36%). The historical run did not seed every
training RNG; it is not the matched control for this experiment.

`learned384_active192` adds an active-write-matched arm: 128 shared coordinates
and four groups of 16 conditional coordinates, exactly 192 residual write
coordinates per clip, block and Euler evaluation. The existing original-width
checkpoint is the reference; the optional `original192` config can reproduce
it with the current training seed if a fully matched rerun becomes necessary.
This matches writable coordinates at inference, not multiply-adds: attention,
MLP and residual states in the routed arm still compute at width 384.

Parameter counts from the instantiated models: original192 23,031,236;
dense384 43,727,492; learned384 45,238,244 (including 1,510,752 router parameters).
The initialization audit confirms that every shared state tensor and the
post-initialization CPU RNG state match between dense384 and learned384.

## Channel mechanism

- Predictor residual width: 384. Encoder and predicted latent width: 192.
- Shared residual coordinates: 128, always writable.
- Conditional coordinates: 256, grouped into 16 groups of 16; select 8 per clip,
  per block and per flow evaluation.
- Apply one group selection to the existing self-attention, cross-attention and
  MLP residual writes in a block. Full states remain readable and retain their
  previous values through the residual connection.
- Router inputs: pooled current noisy state, visible context, visible history
  difference, action condition and flow time. Cube has one history frame, so
  the history-difference input is zero in this experiment.
- Select groups with `topk(score + balance_bias)`; selected write weights are
  `2 * sigmoid(score)`. The balance bias changes selection only.
- The first 100 optimizer steps choose groups uniformly at random.

All branch computations are dense in this implementation. It measures effects
on representation learning and prediction, not sparse-kernel speedups.

## Learning and balancing

Keep the original velocity MSE, target-encoder gradient path and SIGReg loss.
Every eight optimizer steps after warmup, sample one block and exchange one
selected and one unselected group per clip. Evaluate the original and swapped
routes with matched noise, time, targets and dropout; freeze all other routing
weights. If the loss difference exceeds the configured margin, its detached
sign supervises a pairwise score ranking. Router inputs are detached for this
extra loss. Credit loss weight is 0.01.

After each optimizer step, aggregate selection counts across all four ranks.
Maintain an EMA with coefficient 0.99. For each dynamic group, adjust a bounded
selection bias when its EMA falls below 0.125 or rises above 0.75. Shared channels
are excluded. Validation and MPC evaluation do not update load or bias.

## Preserved Cube settings

| Setting | Value |
|---|---|
| Training | 20 epochs; seed 3072; 90/10 original random split |
| Batch | 32 per GPU, 4 GPUs, global 128 |
| Images / encoder | 224 pixels; ViT-Tiny/14; trained from scratch |
| Temporal context | history 1; predict 5; frameskip 5 |
| Flow | standard Gaussian source; scale 1; inference Euler steps 8 |
| Predictor | 6 blocks; 16 heads; head width 64; MLP width 2048 |
| Output projection | hidden width 768; output width 192 |
| Attention / dropout | bidirectional; dropout 0.1 |
| Optimizer | AdamW; learning rate 5e-5; weight decay 1e-3 |
| SIGReg | weight 0.1; 17 knots; 1024 projections |
| Precision | bf16; gradient clipping 1.0 |
| MPC evaluation | 50 starts; seeds 42; CEM 300 samples, 30 iterations, top 30 |
| Planning | horizon 5; action block 5; receding horizon 5 |
| Goal / budget | offset 25; interaction budget 50 |
| Gaussian condition | std 100; radius 35; fixed random non-agent patch |

Configs `random384` and `nocredit384` are available for subsequent ablations.
The initial campaign runs `learned384` and `dense384` sequentially using four
GPUs. Each saves every epoch and evaluates epochs 1 and 20 under clean and
Gaussian observations after training.

## Running

Set `REPO_ROOT`, `PYTHON_ENV`, `DATA_ROOT`, and `RUN_ROOT` to local locations.
Use `cluster/prepare_cube.sh` to restore and verify the source archive; it checks
SHA256 `3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600`.

```bash
python campaign_summary.py --output campaign_summary.json
sbatch --export=ALL cluster/smoke_cube_channels.sbatch
ARM=learned384 sbatch --export=ALL cluster/train_cube_channels.sbatch
ARM=dense384 sbatch --export=ALL cluster/train_cube_channels.sbatch
```

`smoke_training.py` checks four-rank updates, gradient boundaries, replay,
globally consistent load statistics and checkpoint reload. `train.py` writes
resolved training configs, per-epoch model checkpoints and CSV metrics.

Read performance from Cube MPC success rates. Loss curves, channel loads and
credit agreement diagnose the mechanism but do not establish better control
performance by themselves.

After all four evaluations for an arm succeed, the job retains only
`weights_epoch_20.pt` and removes intermediate model and optimizer snapshots.
Metrics and task evaluation results remain available.
