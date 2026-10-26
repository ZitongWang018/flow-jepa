"""Exercise full Cube model, real batch shape and SIGReg on four GPUs.

Synthetic tensors only: measures execution and memory, not task performance.
Run with ``srun --ntasks=4 python full_cube_smoke.py``.
"""
import json
import os
import time
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from module import SIGReg


class FullStep(nn.Module):
    def __init__(self, model, sigreg, regularizer_weight):
        super().__init__()
        self.model = model
        self.sigreg = sigreg
        self.regularizer_weight = regularizer_weight

    def forward(self, pixels, action, step):
        batch = self.model.encode(dict(pixels=pixels, action=action))
        emb, act_emb = batch["emb"], batch["act_emb"]
        losses = self.model.flow_loss(emb[:, :1], act_emb[:, :5], emb[:, 1:6],
                                      training_step=step, return_details=True)
        reg_input = emb.permute(1, 2, 0, 3).reshape(emb.size(1) * emb.size(2), emb.size(0), emb.size(3))
        sigreg_loss = self.sigreg(reg_input)
        total = losses["pred_loss"] + self.regularizer_weight * sigreg_loss
        total = total + 0.01 * losses["route_credit_loss"]
        return total, losses["pred_loss"].detach(), sigreg_loss.detach(), losses["route_credit_loss"].detach()


def main():
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    index = local_rank if torch.cuda.device_count() > 1 else 0
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(3072)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent / "config/train"), version_base=None):
        arm = os.environ.get("ARM", "sparse336_active192")
        cfg = compose(config_name="fjepa", overrides=["data=ogb", f"+routing={arm}", "model.action_encoder.input_dim=25"])
    cfg.model.predictor.routing.warmup_steps = 0
    cfg.model.routing_credit.warmup_steps = 0
    cfg.model.routing_credit.interval = 2
    model = hydra.utils.instantiate(cfg.model)
    step_module = FullStep(model, SIGReg(**cfg.loss.sigreg.kwargs), float(cfg.loss.sigreg.weight)).to(device).train()
    wrapped = DistributedDataParallel(step_module, device_ids=[index],
                                      find_unused_parameters=bool(model.predictor.sparse_compute))
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=float(cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay))
    torch.manual_seed(3072 + rank)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    losses = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        pixels = torch.randn(32, 6, 3, 224, 224, device=device)
        action = torch.randn(32, 6, 25, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, pred, reg, credit = wrapped(pixels, action, step)
        assert torch.isfinite(total), f"Nonfinite loss rank={rank}, step={step}"
        total.backward()
        missing = [name for name, param in step_module.named_parameters()
                   if param.requires_grad and param.grad is None
                   and not (model.predictor.sparse_compute and ".routing.routers." in name)]
        assert not missing, f"Missing gradients: {missing}"
        assert all(torch.isfinite(p.grad).all() for p in step_module.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), float(cfg.trainer.gradient_clip_val))
        optimizer.step()
        model.predictor.routing.after_step()
        losses.append(dict(step=step, total=float(total.detach()), pred=float(pred), sigreg=float(reg), credit=float(credit)))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print(json.dumps(dict(status="passed", rank=rank, world_size=world_size,
                          batch_per_rank=32, hidden_dim=int(cfg.model.predictor.hidden_dim), steps=3,
                          elapsed_seconds=elapsed, peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
                          losses=losses)), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
