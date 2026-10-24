"""Instantiate Cube campaign arms on CPU and audit their parameter counts.

No dataset, training run or pretrained download is required. Example:
    python campaign_summary.py --output campaign_summary.json

The optional original192 arm reports the upstream-width parameter reference;
it does not create or submit a training job.
"""

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


DEFAULT_ARMS = ["dense384", "learned384", "learned384_active192",
                "sparse336_active192", "nocredit384", "random384"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), value.dtype)).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_summary(model, cfg):
    parameters = dict(model.named_parameters())
    routing_params = {
        name: value for name, value in parameters.items()
        if name.startswith("predictor.routing.")
    }
    predictor = model.predictor
    routing = predictor.routing
    hidden_dim = int(predictor.hidden_dim)
    first_block = predictor.layers[0]
    attention = first_block.self_attn
    # These dimensions are intentionally held fixed across the 192/384
    # hidden-state variants. Hidden width is not attention head width.
    architecture = {
        "latent_dim": int(cfg.embed_dim),
        "hidden_dim": hidden_dim,
        "depth": len(predictor.layers),
        "attention_heads": attention.heads,
        "attention_dim_head": attention.dim_head,
        "attention_inner_dim": attention.heads * attention.dim_head,
        "mlp_intermediate_dim": first_block.mlp.net[0].out_features,
        "output_projection_intermediate_dim": predictor.output_proj.net[0].out_features,
        "output_dim": predictor.output_proj.net[-1].out_features,
        "action_encoder_input_dim": int(cfg.model.action_encoder.input_dim),
        "causal_self_attention": predictor.causal_self_attention,
        "candidate_writes_computed_densely": not predictor.sparse_compute,
        "sparse_selected_submatrices": predictor.sparse_compute,
        "routing_mode": routing.mode if routing is not None else "disabled",
        "shared_dim": routing.shared_dim if routing is not None else hidden_dim,
        "dynamic_group_count": routing.groups if routing is not None else 0,
        "dynamic_group_size": routing.group_size if routing is not None else 0,
        "active_dynamic_groups": (
            (routing.groups if routing.mode == "dense" else routing.top_k)
            if routing is not None else 0
        ),
    }
    architecture["active_write_dimensions"] = (
        architecture["shared_dim"]
        + architecture["active_dynamic_groups"] * architecture["dynamic_group_size"]
    )
    return {
        "parameters_total": sum(p.numel() for p in parameters.values()),
        "parameters_trainable": sum(p.numel() for p in parameters.values() if p.requires_grad),
        "parameters_router": sum(p.numel() for p in routing_params.values()),
        "parameters_excluding_router": sum(
            p.numel() for name, p in parameters.items() if name not in routing_params
        ),
        "parameters_by_component": {
            name: sum(p.numel() for p in getattr(model, name).parameters())
            for name in ("encoder", "predictor", "action_encoder", "projector", "pred_proj")
        },
        "architecture": architecture,
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
    }


def compare_state(reference, candidate):
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    mismatches = []
    compared = 0
    for name in sorted(set(reference) & set(candidate)):
        left, right = reference[name], candidate[name]
        compared += 1
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            mismatches.append(name)
    return {
        "bitwise_identical": not (missing or extra or mismatches),
        "compared_state_tensors": compared,
        "mismatched_state_tensors": mismatches,
        "missing_state_tensors": missing,
        "extra_state_tensors": extra,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                        choices=DEFAULT_ARMS + ["original192"])
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--output", type=Path, default=Path("campaign_summary.json"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    arms = list(dict.fromkeys(["dense384"] + args.arms))
    config_dir = str(Path(__file__).resolve().parent / "config" / "train")
    report = {
        "seed": args.seed,
        "device": "cpu",
        "dataset": "cube_single_expert.h5",
        "action_encoder_input_dim": 25,
        "description": "Hydra model instantiation and shared initialization audit; no training or evaluation.",
        "held_fixed_predictor_settings": {
            "attention_heads": 16,
            "attention_dim_head": 64,
            "attention_inner_dim": 1024,
            "mlp_intermediate_dim": 2048,
            "output_projection_intermediate_dim": 768,
        },
        "arms": {},
    }
    reference = None
    reference_rng = None
    failed = False
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        for arm in arms:
            overrides = ["data=ogb", "model.action_encoder.input_dim=25", f"seed={args.seed}"]
            if arm != "original192":
                overrides.append(f"+routing={arm}")
            cfg = compose(config_name="fjepa", overrides=overrides)
            seed_all(args.seed)
            model = hydra.utils.instantiate(cfg.model).cpu()
            summary = model_summary(model, cfg)
            state = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
                if not name.startswith("predictor.routing.")
            }
            summary["state_excluding_router_sha256"] = state_hash(state)
            rng = torch.get_rng_state()
            summary["post_initialization_cpu_rng_sha256"] = hashlib.sha256(rng.numpy().tobytes()).hexdigest()
            if arm == "dense384":
                # State tensors retain the CPU storage after the model is
                # released, avoiding a second full reference model in memory.
                reference, reference_rng = state, rng
                summary["shared_initialization_vs_dense384"] = compare_state(reference, state)
            elif summary["architecture"]["hidden_dim"] == 384:
                check = compare_state(reference, state)
                check["post_initialization_rng_identical"] = torch.equal(reference_rng, rng)
                summary["shared_initialization_vs_dense384"] = check
                failed |= not check["bitwise_identical"] or not check["post_initialization_rng_identical"]
            else:
                summary["shared_initialization_vs_dense384"] = {
                    "applicable": False,
                    "reason": "Different hidden width; parameter shapes and initialization stream differ.",
                }
            actual = summary["architecture"]
            fixed = report["held_fixed_predictor_settings"]
            summary["fixed_predictor_settings_match"] = all(actual[k] == v for k, v in fixed.items())
            failed |= not summary["fixed_predictor_settings_match"]
            report["arms"][arm] = summary
            print(f"{arm}: total={summary['parameters_total']:,} "
                  f"router={summary['parameters_router']:,} "
                  f"writes={actual['active_write_dimensions']}/{actual['hidden_dim']}", flush=True)
            del model, state
            gc.collect()
    report["all_same_width_initializations_match"] = not failed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Shared initialization audit: {'PASS' if not failed else 'FAIL'}; saved {args.output}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
