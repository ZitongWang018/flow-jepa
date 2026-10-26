"""Compare final trained sparse predictor latency with original-width compute."""

import argparse
import json
from pathlib import Path

import torch

from benchmark_sparse_latency import build, measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    predictor_state = {
        name.removeprefix("predictor."): value
        for name, value in state.items() if name.startswith("predictor.")
    }
    if not predictor_state:
        raise RuntimeError("No predictor weights found in checkpoint")
    torch.set_num_threads(2)
    results = {}
    for precision in ("fp32", "bf16"):
        results[precision] = {}
        for label, sparse, share in (
            ("original192", False, False),
            ("sparse336_shared", True, True),
            ("sparse336_independent", True, False),
        ):
            if precision == "fp32" and label == "sparse336_independent":
                continue
            model = build(sparse, share)
            if sparse:
                model.load_state_dict(predictor_state, strict=True)
            results[precision][label] = measure(
                model, batch=300, euler_steps=8, trials=args.trials,
                precision=precision,
            )
            del model
            torch.cuda.empty_cache()
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "sparse_checkpoint": args.checkpoint.name,
        "original_reference": "same-architecture original192, random weights",
        "shape": "CEM 300 candidates, 5 future latents, 8 Euler calls",
        "results": results,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

