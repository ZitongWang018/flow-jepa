"""Run an ordinary Cube evaluation and count selected channel groups per Euler step.

Set ROUTE_PROBE_OUTPUT to a JSON path. All task evaluation arguments are passed
through to eval.py. Counts include the candidate trajectories evaluated by CEM.
"""

import atexit
import json
import os
import runpy
from pathlib import Path

import torch

from channel_routing import ChannelRouting
from module import FlowTrajectoryPredictor


counts = {}
weight_sums = {}
samples = {}
current_step = None
original_forward = FlowTrajectoryPredictor.forward
original_route = ChannelRouting.route


def traced_forward(self, noisy_emb, hist_emb, act_emb, t, *args, **kwargs):
    global current_step
    previous = current_step
    current_step = int(round(float(t.reshape(-1)[0]) * 8))
    try:
        return original_forward(self, noisy_emb, hist_emb, act_emb, t, *args, **kwargs)
    finally:
        current_step = previous


def traced_route(self, block, inputs, record=False, override=None):
    weights = original_route(self, block, inputs, record=record, override=override)
    if not self.training and self.mode == "learned" and current_step is not None:
        groups = weights[:, 0, self.shared_dim:].reshape(-1, self.groups, self.group_size)
        group_weight = groups.abs().mean(-1).float()
        selected = group_weight > 0
        key = (int(block), int(current_step))
        if key not in counts:
            counts[key] = torch.zeros(self.groups, device=weights.device)
            weight_sums[key] = torch.zeros_like(counts[key])
            samples[key] = 0
        counts[key].add_(selected.float().sum(0))
        weight_sums[key].add_(group_weight.sum(0))
        samples[key] += inputs.shape[0]
    return weights


def save_report():
    output = os.environ.get("ROUTE_PROBE_OUTPUT")
    if not output or not counts:
        return
    entries = []
    for (block, step), count in sorted(counts.items()):
        n = samples[(block, step)]
        entry = {
            "block": block,
            "euler_step": step,
            "candidate_samples": n,
            "selection_frequency": (count / n).cpu().tolist(),
            "mean_group_weight": (weight_sums[(block, step)] / n).cpu().tolist(),
            "selected_groups_per_candidate": float(count.sum().item() / n),
        }
        entries.append(entry)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}, indent=2) + "\n")
    print(f"ROUTE_PROBE_SAVED {path} entries={len(entries)}", flush=True)


FlowTrajectoryPredictor.forward = traced_forward
ChannelRouting.route = traced_route
atexit.register(save_report)
runpy.run_module("eval", run_name="__main__")
