"""Audit untouched Flow semantics and the real checkpoint reconstruction path.

Run from the modified checkout:
  python audit_baseline.py --original-repo /path/to/original --device cuda

Small synthetic inputs exercise the original code, not benchmark performance.
The checkpoint check calls stable_worldmodel save_pretrained/load_pretrained;
the returned model is newly constructed from config.json, never deep-copied.
"""
import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from jepa import JEPA
from module import Embedder, FlowTrajectoryPredictor, MLP, SIGReg


class AuditEncoder(nn.Module):
    """Tiny trainable image encoder with the production HF-style interface."""

    def __init__(self, input_dim=48, output_dim=12):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        tokens = self.projection(pixels.flatten(1)).unsqueeze(1)
        return SimpleNamespace(last_hidden_state=tokens)


def load_original(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predictor_kwargs(causal=False):
    return dict(depth=3, heads=2, mlp_dim=64, input_dim=12, hidden_dim=48,
                action_dim=12, output_dim=12, output_proj_hidden_dim=48,
                dim_head=8, dropout=0.1, emb_dropout=0.1,
                causal_self_attention=causal)


def build_model(jepa_class, module, source, causal):
    encoder = AuditEncoder()
    predictor = module.FlowTrajectoryPredictor(**predictor_kwargs(causal))
    # The ordinary zero gates would leave most branch gradients unexercised.
    for layer in predictor.layers:
        nn.init.normal_(layer.adaLN_modulation[-1].weight, std=0.03)
    action = module.Embedder(input_dim=5, smoothed_dim=5, emb_dim=12, mlp_scale=2)
    projector = module.MLP(input_dim=12, hidden_dim=24, output_dim=12)
    return jepa_class(encoder, predictor, action, projector=projector,
                      flow_source=source,
                      flow_source_noise_scale=1.0 if source == "standard_noise" else 0.5)


def assert_equal(left, right, name):
    if left is None or right is None:
        assert left is right, f"{name}: gradient-presence mismatch"
        return 0.0
    assert left.shape == right.shape, f"{name}: shape mismatch"
    difference = float((left.detach().float() - right.detach().float()).abs().max())
    assert torch.equal(left, right), f"{name}: max_abs_diff={difference}"
    return difference


def flow_step(model, regularizer, pixels, actions, seed, device, bf16, modified):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
        output = model.encode({"pixels": pixels.clone(), "action": actions.clone()})
        emb = output["emb"]
        history, target = emb[:, :1], emb[:, 1:]
        history.retain_grad()
        target.retain_grad()
        future_action = output["act_emb"][:, :3]
        if modified:
            details = model.flow_loss(history, future_action, target,
                                      training_step=0, return_details=True)
            pred_loss = details["pred_loss"]
            assert details["route_credit_loss"].item() == 0
        else:
            pred_loss = model.flow_loss(history, future_action, target)
        sig_input = emb.permute(1, 2, 0, 3).reshape(4, pixels.size(0), 12)
        sig_loss = regularizer(sig_input)
        loss = pred_loss + 0.1 * sig_loss
    loss.backward()
    return dict(pred=pred_loss.detach(), sigreg=sig_loss.detach(), loss=loss.detach(),
                history_grad=history.grad, target_grad=target.grad,
                gradients={name: None if p.grad is None else p.grad.detach().clone()
                           for name, p in model.named_parameters()})


def baseline_audit(original_repo, device):
    original_module = load_original(original_repo / "module.py", "audit_original_module")
    original_jepa = load_original(original_repo / "jepa.py", "audit_original_jepa")
    current_module = SimpleNamespace(FlowTrajectoryPredictor=FlowTrajectoryPredictor,
                                     Embedder=Embedder, MLP=MLP)
    cases = []
    for source in ("standard_noise", "noisy_current"):
        for causal in (False, True):
            torch.manual_seed(3072)
            old = build_model(original_jepa.JEPA, original_module, source, causal).to(device)
            torch.manual_seed(3072)
            new = build_model(JEPA, current_module, source, causal).to(device)
            assert old.state_dict().keys() == new.state_dict().keys()
            for name, tensor in old.state_dict().items():
                assert_equal(tensor, new.state_dict()[name], f"initialization/{name}")
            old_reg = original_module.SIGReg(knots=17, num_proj=32).to(device)
            new_reg = SIGReg(knots=17, num_proj=32).to(device)
            torch.manual_seed(9182)
            pixels = torch.randn(4, 4, 3, 4, 4, device=device)
            actions = torch.randn(4, 4, 5, device=device)
            for bf16 in ((False, True) if device.type == "cuda" else (False,)):
                lhs = flow_step(old, old_reg, pixels, actions, 881, device, bf16, False)
                rhs = flow_step(new, new_reg, pixels, actions, 881, device, bf16, True)
                for key in ("pred", "sigreg", "loss", "history_grad", "target_grad"):
                    assert_equal(lhs[key], rhs[key], key)
                assert lhs["target_grad"].abs().sum() > 0, "Original target path is inactive"
                assert lhs["gradients"].keys() == rhs["gradients"].keys()
                for name in lhs["gradients"]:
                    assert_equal(lhs["gradients"][name], rhs["gradients"][name], name)
                assert rhs["gradients"]["encoder.projection.weight"].abs().sum() > 0
                # Compare original versus modified stochastic predictor directly.
                noisy = torch.randn(4, 3, 1, 12, device=device)
                hist = torch.randn(4, 1, 1, 12, device=device)
                act = torch.randn(4, 3, 12, device=device)
                tau = torch.rand(4, device=device)
                with torch.no_grad(), torch.autocast(device_type=device.type,
                        dtype=torch.bfloat16, enabled=bf16):
                    torch.manual_seed(719)
                    original_prediction = old.predictor(noisy, hist, act, tau)
                    torch.manual_seed(719)
                    modified_prediction = new.predictor(noisy, hist, act, tau)
                assert_equal(original_prediction, modified_prediction, "predictor/dropout")
                cases.append(dict(source=source, causal=causal,
                                  precision="bf16" if bf16 else "fp32",
                                  max_loss_and_gradient_difference=0.0,
                                  predictor_difference=0.0,
                                  target_and_encoder_gradients_preserved=True))
    return cases


def checkpoint_audit(device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from stable_worldmodel.wm.utils import load_pretrained, save_pretrained

    config = dict(
        _target_="jepa.JEPA",
        encoder=dict(_target_="audit_baseline.AuditEncoder", input_dim=48, output_dim=12),
        predictor=dict(_target_="module.FlowTrajectoryPredictor", **predictor_kwargs(),
                       routing=dict(enabled=True, shared_dim=16, group_size=4, top_k=4,
                                    router_hidden_dim=16, warmup_steps=0)),
        action_encoder=dict(_target_="module.Embedder", input_dim=5, smoothed_dim=5,
                            emb_dim=12, mlp_scale=2),
        projector=dict(_target_="module.MLP", input_dim=12, hidden_dim=24, output_dim=12),
        routing_credit=dict(enabled=True, interval=1, warmup_steps=0,
                            min_margin=0, relative_margin=0),
        num_flow_steps=2, flow_source="standard_noise", flow_source_noise_scale=1.0,
    )
    cfg = OmegaConf.create(config)
    model = instantiate(cfg).to(device).train()
    for layer in model.predictor.layers:
        nn.init.normal_(layer.adaLN_modulation[-1].weight, std=0.03)
    pixels = torch.randn(4, 4, 3, 4, 4, device=device)
    actions = torch.randn(4, 4, 5, device=device)
    output = model.encode({"pixels": pixels, "action": actions})
    result = model.flow_loss(output["emb"][:, :1], output["act_emb"][:, :3],
                             output["emb"][:, 1:], training_step=103,
                             return_details=True)
    assert result["routing_metrics"]["credit_probe_rate"] == 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    (result["pred_loss"] + 0.01 * result["route_credit_loss"]).backward()
    optimizer.step()
    model.predictor.routing.after_step()
    # Ensure a nonzero persistent control value is tested even before the
    # natural EMA crosses a load bound during this one-step synthetic run.
    with torch.no_grad():
        model.predictor.routing.selection_bias[:, 0].fill_(0.125)
    model.eval()
    with torch.no_grad():
        encoded = model.encode({"pixels": pixels, "action": actions})
        fixed_noise = torch.randn_like(encoded["emb"][:, 1:])
        prediction = model.predict(encoded["emb"][:, :1], encoded["act_emb"][:, :3],
                                   noise=fixed_noise, num_steps=2)

    with tempfile.TemporaryDirectory(prefix="flow-channel-audit-") as temporary:
        save_pretrained(model, run_name="routed-model", config=cfg,
                        filename="weights_epoch_1.pt", cache_dir=temporary)
        checkpoint = Path(temporary) / "checkpoints" / "routed-model" / "weights_epoch_1.pt"
        assert checkpoint.is_file() and (checkpoint.parent / "config.json").is_file()
        restored = load_pretrained(str(checkpoint), cache_dir=temporary).to(device).eval()
        assert restored is not model and restored.predictor.routing is not model.predictor.routing
        for name, parameter in model.named_parameters():
            assert parameter.data_ptr() != dict(restored.named_parameters())[name].data_ptr()
        assert model.state_dict().keys() == restored.state_dict().keys()
        for name, tensor in model.state_dict().items():
            assert_equal(tensor, restored.state_dict()[name], f"checkpoint/{name}")
        assert restored.predictor.routing._step_value == 104
        assert restored.routing_credit == model.routing_credit
        with torch.no_grad():
            fresh = restored.encode({"pixels": pixels, "action": actions})
            restored_prediction = restored.predict(fresh["emb"][:, :1],
                                                   fresh["act_emb"][:, :3],
                                                   noise=fixed_noise, num_steps=2)
        assert_equal(prediction, restored_prediction, "checkpoint/inference")
    return dict(api="stable_worldmodel.save_pretrained/load_pretrained",
                fresh_config_construction=True, shared_parameter_storage=False,
                full_state_exact=True, inference_exact=True, controller_step=104)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-repo", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    original_repo = args.original_repo.resolve()
    assert original_repo != Path(__file__).resolve().parent, "Original and modified checkout must differ"
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        device = torch.device("cuda", device.index or 0)
    cases = baseline_audit(original_repo, device)
    reconstruction = checkpoint_audit(device)
    print(json.dumps(dict(status="passed", device=str(device), baseline_cases=cases,
                          original_source_sha256={name: hashlib.sha256((original_repo / name).read_bytes()).hexdigest()
                                                  for name in ("module.py", "jepa.py")},
                          checkpoint_reconstruction=reconstruction), indent=2))


if __name__ == "__main__":
    main()
