"""Small synthetic integration checks for routing credit and gradient boundaries.

Run ``python smoke_training.py --device cuda`` before submitting real training.
No dataset or downloaded backbone is needed. This is not a benchmark result.
"""
import argparse
import copy
import io
import json
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from jepa import JEPA
from module import FlowTrajectoryPredictor


class SmokeStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, history, action, target, step):
        result = self.model.flow_loss(history, action, target, training_step=step,
                                      return_details=True)
        return result["pred_loss"] + 0.01 * result["route_credit_loss"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(3072)
    device = torch.device(args.device)
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    if args.ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
        if device.type == "cuda":
            # Some Slurm configurations expose one GPU per task, others all four.
            gpu_index = local_rank if torch.cuda.device_count() > 1 else 0
            torch.cuda.set_device(gpu_index)
            device = torch.device("cuda", gpu_index)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                rank=rank, world_size=world_size)
    routing_config = (
        dict(enabled=True, sparse_compute=True, shared_dim=12,
             group_size=4, top_k=1, selected_weight_mode="binary",
             router_hidden_dim=16, warmup_steps=0)
        if args.sparse else
        dict(enabled=True, shared_dim=16, group_size=4, top_k=4,
             router_hidden_dim=16, warmup_steps=0)
    )
    predictor = FlowTrajectoryPredictor(
        depth=6, heads=2, mlp_dim=64, input_dim=12,
        hidden_dim=28 if args.sparse else 48,
        action_dim=12, output_dim=12, output_proj_hidden_dim=48,
        dim_head=8, dropout=0.1, causal_self_attention=False,
        routing=routing_config,
    )
    # Original AdaLN starts at zero. Nonzero writes exercise credit immediately.
    for block in predictor.layers:
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.03)
    predictor.to(device)
    if args.sparse:
        # The sparse block must bypass the dense branch modules entirely.
        def dense_branch_called(*_args, **_kwargs):
            raise AssertionError("Sparse routing called a full-width branch")
        for block in predictor.layers:
            block.self_attn.forward = dense_branch_called
            block.cross_attn.forward = dense_branch_called
            block.mlp.forward = dense_branch_called
        block = predictor.layers[0]
        state = torch.randn(4, 3, 28, device=device)
        context = torch.randn(4, 4, 28, device=device)
        time = torch.randn(4, 1, 28, device=device)
        groups = torch.tensor([0, 1, 2, 3], device=device)
        mask = torch.zeros(4, 1, 28, device=device)
        mask[..., :12] = 1
        for row, group in enumerate(groups.tolist()):
            mask[row, :, 12 + group * 4:16 + group * 4] = 1
        block.eval()
        updated = block.forward_sparse(state, context, time, groups,
                                       predictor.sparse_group_indices, mask)
        grouped = block.forward_sparse_grouped(state, context, time, groups,
                                               predictor.sparse_group_indices, mask)
        assert torch.allclose(updated, grouped, rtol=1e-5, atol=1e-5), \
            f"Batched sparse path differs from grouped reference: {(updated-grouped).abs().max()}"
        shared_groups = torch.zeros_like(groups)
        shared_mask = torch.zeros_like(mask)
        shared_mask[..., :16] = 1
        shared = block.forward_sparse_shared(state, context, time, shared_groups,
                                             predictor.sparse_group_indices, shared_mask)
        shared_reference = block.forward_sparse_grouped(
            state, context, time, shared_groups,
            predictor.sparse_group_indices, shared_mask)
        assert torch.allclose(shared, shared_reference, rtol=1e-5, atol=1e-5)
        block.train()
        assert torch.equal(updated[mask.expand_as(updated) == 0],
                           state[mask.expand_as(state) == 0])
        assert (updated - state).abs().sum() > 0
    model = JEPA(nn.Identity(), predictor, nn.Identity(), routing_credit=dict(
        enabled=True, interval=1, warmup_steps=0, min_margin=0, relative_margin=0,
    )).to(device).train()
    history = torch.randn(4, 1, 1, 12, device=device, requires_grad=True)
    action = torch.randn(4, 3, 12, device=device, requires_grad=True)
    target = torch.randn(4, 3, 1, 12, device=device, requires_grad=True)
    result = model.flow_loss(history, action, target, training_step=0,
                             return_details=True)
    routing = predictor.routing
    assert torch.isfinite(result["pred_loss"])
    assert result["routing_metrics"]["credit_probe_rate"] == 1
    assert result["routing_metrics"]["credit_valid_rate"] > 0
    # Exactly one batch is recorded, despite two additional probe forwards.
    assert torch.equal(routing.pending_samples, torch.full_like(routing.pending_samples, 4))
    named = list(model.named_parameters())
    grads = torch.autograd.grad(
        result["route_credit_loss"], [p for _, p in named] + [history, action, target],
        retain_graph=True, allow_unused=True,
    )
    router_norm = 0.0
    for (name, _), gradient in zip(named, grads):
        if ".routing.routers." in name:
            if gradient is not None:
                router_norm += float(gradient.float().abs().sum())
        else:
            assert gradient is None, f"Credit escaped into {name}"
    assert all(g is None for g in grads[-3:]), "Credit escaped into encoder inputs"
    assert router_norm > 0
    (result["pred_loss"] + 0.01 * result["route_credit_loss"]).backward()
    assert target.grad is not None and target.grad.abs().sum() > 0
    assert history.grad is not None and torch.isfinite(history.grad).all()
    routing.after_step()
    assert routing.step == 1 and routing.pending_samples.sum() == 0

    # Fixed routes with matched dropout replay identically; probes are state-free.
    trace = routing.get_last_route_trace()
    overrides = routing.make_overrides(trace)
    noise = torch.randn_like(target)
    tau = torch.rand(4, device=device)
    pending_before = routing.pending_counts.clone()
    with torch.no_grad():
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        first = predictor(noise, history, action, tau, routing_overrides=overrides,
                          routing_record=False)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        second = predictor(noise, history, action, tau, routing_overrides=overrides,
                           routing_record=False)
    assert torch.equal(first, second), "Matched stochastic replay differs"
    assert torch.equal(pending_before, routing.pending_counts)
    assert routing.get_last_route_trace() is trace

    # Frozen inference leaves all load/bias/step buffers unchanged.
    before = {name: tensor.clone() for name, tensor in routing.named_buffers()}
    model.eval()
    with torch.no_grad():
        prediction = model.predict(history.detach(), action.detach(), num_steps=2)
    if args.sparse and device.type == "cuda":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            eager_flow = model.predict(history.detach(), action.detach(),
                                       num_steps=2, noise=noise)
            predictor.enable_cuda_graph(True)
            graphed_flow = model.predict(history.detach(), action.detach(),
                                         num_steps=2, noise=noise)
            predictor.enable_cuda_graph(False)
            model.predict(history.detach(), action.detach(),
                          num_steps=2, noise=noise)
        assert torch.allclose(graphed_flow, eager_flow, rtol=0.02, atol=0.02)
    for name, tensor in routing.named_buffers():
        assert torch.equal(before[name], tensor), name
    assert prediction.shape == target.shape and torch.isfinite(prediction).all()
    ddp_steps = 0
    if args.ddp:
        model.train()
        model.zero_grad(set_to_none=True)
        wrapped = DistributedDataParallel(
            SmokeStep(model), device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=args.sparse,
        )
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        # Rank-specific observations expose accidental rank-local load updates.
        torch.manual_seed(3072 + rank)
        for step in range(2, 4):
            optimizer.zero_grad(set_to_none=True)
            sample_hist = torch.randn_like(history)
            sample_action = torch.randn_like(action)
            sample_target = torch.randn_like(target, requires_grad=True)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = wrapped(sample_hist, sample_action, sample_target, step)
            loss.backward()
            assert torch.isfinite(loss) and sample_target.grad is not None
            missing = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad and parameter.grad is None
                       and not (args.sparse and ".routing.routers." in name)]
            assert not missing, f"Missing gradients on rank {rank}: {missing}"
            optimizer.step()
            routing.after_step()
            all_loads = [torch.empty_like(routing.load_ema) for _ in range(world_size)]
            dist.all_gather(all_loads, routing.load_ema)
            assert all(torch.equal(all_loads[0], item) for item in all_loads)
            all_bias = [torch.empty_like(routing.selection_bias) for _ in range(world_size)]
            dist.all_gather(all_bias, routing.selection_bias)
            assert all(torch.equal(all_bias[0], item) for item in all_bias)
            ddp_steps += 1

    # Model checkpoint includes route EMA, bias and step and reconstructs output.
    if args.sparse:
        model.train()
        model.eval()
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = copy.deepcopy(model)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        original_prediction = model.predict(history.detach(), action.detach(), num_steps=2, noise=noise)
        reloaded_prediction = reloaded.predict(history.detach(), action.detach(), num_steps=2, noise=noise)
    assert torch.equal(original_prediction, reloaded_prediction)
    # Each credit probe must supervise a different block, even when the CPU
    # RNG is unchanged between probes (the usual CUDA training case).
    model.train()
    probe_layers = []
    for probe_step in range(routing.depth):
        probe = model.flow_loss(
            history.detach(), action.detach(), target.detach(),
            training_step=probe_step, return_details=True,
        )
        probe_layers.append(int(probe["routing_metrics"]["credit_probe_layer"]))
    assert probe_layers == list(range(1, routing.depth + 1)), probe_layers
    print(json.dumps(dict(
        status="passed", device=str(device), rank=rank, ddp_steps=ddp_steps,
        sparse=args.sparse,
        prediction_loss=float(result["pred_loss"].detach()),
        credit_loss=float(result["route_credit_loss"].detach()),
        credit_router_gradient_l1=router_norm,
        probe_valid_rate=float(result["routing_metrics"]["credit_valid_rate"]),
        credit_probe_layers=probe_layers,
        main_target_gradient_preserved=True,
        probe_buffers_and_replay_verified=True,
        checkpoint_reload_verified=True,
    ), indent=2))
    if args.ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

