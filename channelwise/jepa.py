"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        num_flow_steps=8,
        flow_source="standard_noise",
        flow_source_noise_scale=1.0,
        freeze_encoder=False,
        freeze_projector=False,
        routing_credit=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.num_flow_steps = num_flow_steps
        self.flow_source = str(flow_source).lower()
        self.flow_source_noise_scale = float(flow_source_noise_scale)
        self._validate_flow_configuration()
        self.freeze_encoder = bool(freeze_encoder)
        self.freeze_projector = bool(freeze_projector)
        self.set_visual_encoder_frozen(
            self.freeze_encoder,
            self.freeze_projector,
        )
        self._flow_seed = None
        self._flow_generators = {}
        self.routing_credit = dict(routing_credit or {})
        self._flow_graph_cache = {}

    def _validate_flow_configuration(self):
        if self.flow_source not in {
            "standard_noise",
            "noisy_current",
        }:
            raise ValueError(
                "flow_source must be either 'standard_noise' or "
                "'noisy_current'"
            )
        if self.flow_source_noise_scale < 0:
            raise ValueError(
                "flow_source_noise_scale must be non-negative"
            )

    @staticmethod
    def _set_module_frozen(module, frozen):
        module.requires_grad_(not frozen)
        if frozen:
            module.eval()

    def set_visual_encoder_frozen(
        self,
        freeze_encoder=True,
        freeze_projector=True,
    ):
        """Freeze the modules that define the visual latent coordinates."""
        self.freeze_encoder = bool(freeze_encoder)
        self.freeze_projector = bool(freeze_projector)
        self._set_module_frozen(self.encoder, self.freeze_encoder)
        self._set_module_frozen(self.projector, self.freeze_projector)
        return self

    def train(self, mode=True):
        """Keep frozen visual modules and normalization statistics in eval mode."""
        if mode:
            self._flow_graph_cache.clear()
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
        if getattr(self, "freeze_projector", False):
            self.projector.eval()
        return self

    def set_flow_seed(self, seed):
        """Reset the dedicated RNG used for flow-matching inference."""
        self._flow_seed = None if seed is None else int(seed)
        self._flow_generators = {}
        return self

    def _get_flow_generator(self, device):
        if self._flow_seed is None:
            return None
        device = torch.device(device)
        key = str(device)
        if key not in self._flow_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._flow_seed)
            self._flow_generators[key] = generator
        return self._flow_generators[key]

    def _apply_token_module(self, module, x):
        if isinstance(module, nn.Identity):
            return x
        shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        x = module(x)
        return x.reshape(*shape[:-1], x.size(-1))

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        if hasattr(output, "last_hidden_state"):
            pixels_emb = output.last_hidden_state[:, 0]
            emb = self.projector(pixels_emb)
            emb = rearrange(emb, "(b t) d -> b t 1 d", b=b)
        else:
            emb = self._apply_token_module(self.projector, output)
            emb = rearrange(emb, "(b t) k d -> b t k d", b=b)
        info["emb"] = emb

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def _repeated_current_latent(self, hist_emb, pred_len):
        """Repeat the final history latent across the prediction horizon."""
        if hist_emb.ndim != 4:
            raise ValueError(
                "history embeddings must have shape (B, H, K, D)"
            )
        pred_len = int(pred_len)
        if pred_len <= 0:
            raise ValueError("prediction length must be positive")
        return hist_emb[:, -1:].expand(-1, pred_len, -1, -1)

    def _flow_source_from_noise(self, hist_emb, noise):
        """Transform raw Gaussian noise into the configured flow source."""
        if noise.ndim != 4:
            raise ValueError(
                "flow noise must have shape (B, P, K, D)"
            )
        if hist_emb.size(0) != noise.size(0):
            raise ValueError("history and flow-noise batch sizes must match")
        if hist_emb.shape[2:] != noise.shape[2:]:
            raise ValueError(
                "history and flow-noise latent token shapes must match"
            )
        noise_scale = float(
            getattr(self, "flow_source_noise_scale", 1.0)
        )
        source = noise * noise_scale
        flow_source = getattr(
            self,
            "flow_source",
            "standard_noise",
        )
        if flow_source == "standard_noise":
            return source
        if flow_source != "noisy_current":
            raise ValueError(f"unsupported flow source: {flow_source}")
        current_latent = self._repeated_current_latent(
            hist_emb,
            noise.size(1),
        )
        return current_latent + source

    def flow_loss(self, hist_emb, future_act_emb, target_emb, *,
                  training_step=None, return_details=False):
        """
        Compute flow-matching loss for future latent prediction.
        hist_emb: (B, H, K, D)
        future_act_emb: (B, F, A_emb)
        target_emb: (B, P, K, D)
        """
        routed = bool(getattr(self.predictor, "routing_enabled", False))
        if routed and training_step is not None:
            self.predictor.set_routing_step(int(training_step))
        noise = torch.randn_like(target_emb)
        flow_source = self._flow_source_from_noise(hist_emb, noise)
        t = torch.rand(target_emb.size(0), device=target_emb.device)
        t_view = t.view(-1, 1, 1, 1)
        noisy_emb = (1 - t_view) * flow_source + t_view * target_emb
        target_velocity = target_emb - flow_source
        pred_velocity = self.predictor(noisy_emb, hist_emb, future_act_emb, t)
        pred_velocity = self._apply_token_module(self.pred_proj, pred_velocity)
        pred_loss = F.mse_loss(pred_velocity, target_velocity)
        if not return_details:
            return pred_loss

        credit_loss = pred_loss.new_zeros(())
        metrics = {}
        if routed and self.training and torch.is_grad_enabled():
            trace = self.predictor.get_last_route_trace()
            if trace:
                usage = torch.stack([item["mask"].float().mean(0) for item in trace])
                metrics.update({
                    "route_batch_load_min": usage.min(),
                    "route_batch_load_max": usage.max(),
                    "route_load_ema_min": self.predictor.routing.load_ema.min(),
                    "route_load_ema_max": self.predictor.routing.load_ema.max(),
                })
                credit_loss, credit_metrics = self._routing_credit_loss(
                    noisy_emb, hist_emb, future_act_emb, t, target_velocity,
                    trace, int(training_step or 0),
                )
                metrics.update(credit_metrics)
        return {"pred_loss": pred_loss, "route_credit_loss": credit_loss,
                "routing_metrics": metrics}

    def _routing_credit_loss(self, noisy_emb, hist_emb, future_act_emb, t,
                             target_velocity, trace, training_step):
        """Matched stochastic swap probes supervise only the detached-input router.

        Both probes freeze all other route weights and use identical dropout RNG.
        The target/noise/time and the original encoder gradient paths are unchanged.
        Forking RNG also prevents probes from changing subsequent training draws.
        """
        cfg = self.routing_credit
        zero = target_velocity.new_zeros(())
        metrics = {name: zero for name in (
            "credit_probe_rate", "credit_valid_rate", "credit_swap_win_rate",
            "credit_advantage_mean", "credit_advantage_abs", "credit_original_mse",
            "credit_probe_layer",
        )}
        routing = self.predictor.routing
        interval = max(1, int(cfg.get("interval", 8)))
        warmup = int(cfg.get("warmup_steps", 100))
        if (not cfg.get("enabled", False) or routing.mode != "learned"
                or training_step < warmup or training_step % interval != 0):
            return zero, metrics

        devices = [noisy_emb.device.index] if noisy_emb.is_cuda else []
        # A CPU randint inside fork_rng can select the same block repeatedly
        # when the normal training step only advances the CUDA RNG. Cycle the
        # probed block so all six routers receive matched-swap supervision.
        probe_block = (training_step // interval) % routing.depth
        # Includes swap sampling, so auxiliary probes consume no training RNG.
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            swap = routing.make_swap(trace, block=probe_block)
            block, removed, added = swap
            original_routes = routing.make_overrides(trace)
            swapped_routes = routing.make_overrides(trace, swap=swap)
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(noisy_emb.device) if devices else None
            args = (noisy_emb.detach(), hist_emb.detach(),
                    future_act_emb.detach(), t.detach())
            original = self.predictor(*args, routing_overrides=original_routes,
                                      routing_record=False)
            original = self._apply_token_module(self.pred_proj, original)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, noisy_emb.device)
            swapped = self.predictor(*args, routing_overrides=swapped_routes,
                                     routing_record=False)
            swapped = self._apply_token_module(self.pred_proj, swapped)
            target = target_velocity.detach().float()
            loss_original = (original.float() - target).square().flatten(1).mean(1)
            loss_swapped = (swapped.float() - target).square().flatten(1).mean(1)
            advantage = (loss_original - loss_swapped).detach()
            threshold = float(cfg.get("min_margin", 1e-5))
            threshold = threshold + float(cfg.get("relative_margin", 1e-3)) * loss_original
            valid = advantage.abs() > threshold

        scores = routing.rescore(block, trace[block]["inputs"].detach()).float()
        margin = scores.gather(1, added[:, None]).squeeze(1)
        margin = margin - scores.gather(1, removed[:, None]).squeeze(1)
        losses = F.softplus(-advantage.sign() * margin)
        credit_loss = (losses * valid).sum() / valid.sum().clamp_min(1)
        metrics.update({
            "credit_probe_rate": zero + 1,
            "credit_probe_layer": zero + block + 1,
            "credit_valid_rate": valid.float().mean(),
            "credit_swap_win_rate": (advantage > threshold).float().mean(),
            "credit_advantage_mean": advantage.mean(),
            "credit_advantage_abs": advantage.abs().mean(),
            "credit_original_mse": loss_original.mean(),
        })
        return credit_loss, metrics

    def predict(self, hist_emb, future_act_emb, horizon=None, num_steps=None, noise=None):
        """Generate a future latent trajectory by Euler integration."""
        future_len = horizon or future_act_emb.size(1)
        if future_act_emb.size(1) < future_len:
            raise ValueError("future_act_emb is shorter than requested horizon")
        if future_act_emb.size(1) > future_len:
            future_act_emb = future_act_emb[:, :future_len]
        num_steps = num_steps or self.num_flow_steps
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        b, _, query_num, dim = hist_emb.shape
        if noise is None:
            generator = self._get_flow_generator(hist_emb.device)
            noise = torch.randn(
                b,
                future_len,
                query_num,
                dim,
                device=hist_emb.device,
                dtype=hist_emb.dtype,
                generator=generator,
            )
        x = self._flow_source_from_noise(hist_emb, noise)

        if (not getattr(self.predictor, "cuda_graph_enabled", False)
                and self._flow_graph_cache):
            self._flow_graph_cache.clear()
        sparse_fp32_grouped = (
            getattr(self.predictor, "sparse_compute", False)
            and not torch.is_autocast_enabled("cuda")
            and (not getattr(self.predictor, "share_route_inference", False)
                 or x.size(0) == 1)
        )
        if (getattr(self.predictor, "cuda_graph_enabled", False)
                and not self.training and x.is_cuda and not sparse_fp32_grouped):
            return self._predict_flow_graph(x, hist_emb, future_act_emb, num_steps)

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full(
                (b,),
                step / num_steps,
                device=hist_emb.device,
                dtype=hist_emb.dtype,
            )
            if getattr(self.predictor, "cuda_graph_enabled", False):
                velocity = self.predictor.forward_inference(x, hist_emb, future_act_emb, t)
            else:
                velocity = self.predictor(x, hist_emb, future_act_emb, t)
            velocity = self._apply_token_module(self.pred_proj, velocity)
            x = x + dt * velocity
        return x

    def _predict_flow_graph(self, source, hist_emb, future_act_emb, num_steps):
        """Capture all Euler updates once for a fixed CEM candidate shape."""
        key = (tuple(source.shape), tuple(hist_emb.shape),
               tuple(future_act_emb.shape), source.dtype, hist_emb.dtype,
               future_act_emb.dtype, num_steps, torch.is_autocast_enabled("cuda"))
        cached = self._flow_graph_cache.get(key)

        def integrate(state, history, actions):
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = torch.full((state.size(0),), step / num_steps,
                               device=state.device, dtype=state.dtype)
                velocity = self.predictor(state, history, actions, t)
                velocity = self._apply_token_module(self.pred_proj, velocity)
                state = state + dt * velocity
            return state

        with torch.inference_mode():
            if cached is None:
                static = tuple(tensor.clone() for tensor in
                               (source, hist_emb, future_act_emb))
                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    for _ in range(3):
                        integrate(*static)
                torch.cuda.current_stream().wait_stream(warmup_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = integrate(*static)
                cached = (static, graph, output)
                self._flow_graph_cache[key] = cached
            static, graph, output = cached
            for target, value in zip(static, (source, hist_emb, future_act_emb)):
                target.copy_(value)
            graph.replay()
            return output.clone()

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 1):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        B, S, horizon = action_sequence.shape[:3]

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init.pop("action", None)
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1, -1)

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(action_sequence, "b s ... -> (b s) ...")
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -history_size:]
        pred_emb = self.predict(emb_trunc, act_emb, horizon=horizon)
        pred_rollout = rearrange(
            pred_emb, "(b s) ... -> b s ...", b=B, s=S
        )
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]

        pred_final = pred_emb[:, :, -1]
        goal_final = goal_emb[:, -1].unsqueeze(1).expand_as(pred_final)
        cost = F.mse_loss(
            pred_final,
            goal_final.detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_final.ndim)))

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost

