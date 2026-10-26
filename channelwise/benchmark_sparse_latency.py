"""GPU latency for the original 192-D and routed 336/192-D predictors.

Measures predictor calls and eight Euler updates with the Cube token shapes.
Random weights measure compute cost only, never task quality.
"""
import argparse
import json
import statistics
import time

import torch

import module as flow_module
from module import FlowTrajectoryPredictor


def build(sparse, share_route=False):
    predictor = FlowTrajectoryPredictor(
        depth=6, heads=16, dim_head=64, mlp_dim=2048,
        input_dim=192, hidden_dim=336 if sparse else 192,
        action_dim=192, output_dim=192, output_proj_hidden_dim=768,
        causal_self_attention=False, dropout=0.1,
        routing=(dict(enabled=True, sparse_compute=True, mode="learned",
                      shared_dim=144, group_size=48, top_k=1,
                      selected_weight_mode="binary", router_hidden_dim=128,
                      warmup_steps=0,
                      share_route_inference=share_route) if sparse else None),
    ).cuda().eval()
    for block in predictor.layers:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.03)
    return predictor


def check_equivalence(model, batch):
    noise = torch.randn(batch, 5, 1, 192, device="cuda")
    history = torch.randn(batch, 1, 1, 192, device="cuda")
    actions = torch.randn(batch, 5, 192, device="cuda")
    time_values = torch.rand(batch, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        candidate = model(noise, history, actions, time_values)
        triton_module = flow_module.sparse_triton
        flow_module.sparse_triton = None
        try:
            reference = model(noise, history, actions, time_values)
        finally:
            flow_module.sparse_triton = triton_module
    difference = float((candidate-reference).abs().max())
    assert torch.allclose(candidate, reference, rtol=0.02, atol=0.02), difference
    return round(difference, 6)


def measure(model, batch, euler_steps, trials, precision="bf16"):
    noise = torch.randn(batch, 5, 1, 192, device="cuda")
    history = torch.randn(batch, 1, 1, 192, device="cuda")
    actions = torch.randn(batch, 5, 192, device="cuda")

    def evaluate():
        state = noise
        for step in range(euler_steps):
            t = torch.full((batch,), step / euler_steps, device="cuda")
            velocity = (model.forward_inference(state, history, actions, t)
                        if model.cuda_graph_enabled else
                        model(state, history, actions, t))
            state = state + velocity / euler_steps
        return state

    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
    ):
        for _ in range(10):
            evaluate()
        torch.cuda.synchronize()
        samples = []
        for _ in range(trials):
            start = time.perf_counter()
            evaluate()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        eager = dict(median_ms=round(statistics.median(samples), 3),
                     p90_ms=round(sorted(samples)[int(0.9 * (len(samples)-1))], 3),
                     min_ms=round(min(samples), 3))
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                evaluate()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_result = evaluate()
        torch.cuda.synchronize()
        graph_samples = []
        for _ in range(max(30, trials)):
            start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            graph_samples.append((time.perf_counter() - start) * 1000)
        assert torch.isfinite(graph_result).all()
        eager["cuda_graph_median_ms"] = round(statistics.median(graph_samples), 3)
        model.enable_cuda_graph(True)
        for _ in range(3):
            evaluate()
        torch.cuda.synchronize()
        predictor_graph_samples = []
        for _ in range(trials):
            start = time.perf_counter()
            evaluate()
            torch.cuda.synchronize()
            predictor_graph_samples.append((time.perf_counter() - start) * 1000)
        eager["predictor_graph_median_ms"] = round(
            statistics.median(predictor_graph_samples), 3)
        model.enable_cuda_graph(False)
    return eager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(3072)
    results = {}
    for sparse, share_route in ((False, False), (True, False), (True, True)):
        if args.precision == "fp32" and sparse and not share_route:
            continue
        label = ("sparse336_shared_candidates" if share_route else
                 "sparse336_active192" if sparse else "original192")
        model = build(sparse, share_route)
        results[label] = {}
        if sparse and not share_route and args.precision == "bf16":
            results[label]["per_sample_vs_batched_max_abs"] = check_equivalence(model, 3)
            results[label]["grouped_vs_batched_max_abs"] = check_equivalence(model, 64)
        for batch in (1, 32, 300):
            if args.precision == "fp32" and share_route and batch == 1:
                continue
            for steps in (1, 8):
                results[label][f"batch{batch}_steps{steps}"] = measure(
                    model, batch, steps, args.trials, precision=args.precision)
        del model
        torch.cuda.empty_cache()
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(0),
                          precision=args.precision, results=results),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
