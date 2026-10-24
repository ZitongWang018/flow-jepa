import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from channel_routing import ChannelRouting
from sparse_linear import selected_linear, selected_linear_grouped, triton as sparse_triton

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings with no fixed max horizon."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class MLPBlock(nn.Module):
    """Transformer MLP without an internal norm."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with an optional keep mask."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads)
            for t in (q, k, v)
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.bool()[None, None], -torch.finfo(scores.dtype).max)
        attn = scores.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention with an optional keep mask."""

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context, mask=None):
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.bool()[None, None], -torch.finfo(scores.dtype).max)
        attn = scores.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class RMSNormMLP(nn.Module):
    """Token-local nonlinear projection without batch-dependent statistics."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        eps=1e-6,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.RMSNorm(hidden_dim, eps=eps),
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class FlowTrajectoryBlock(nn.Module):
    """AdaLN-conditioned block for noisy future latent tokens."""

    def __init__(
        self,
        dim,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.self_attn = SelfAttention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.cross_attn = CrossAttention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = MLPBlock(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        self._sparse_eval_banks = None
        self._sparse_eval_packed = None

    def train(self, mode=True):
        if mode:
            self._sparse_eval_banks = None
            self._sparse_eval_packed = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._sparse_eval_banks = None
        self._sparse_eval_packed = None
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(
        self,
        x,
        context,
        t_cond,
        self_mask=None,
        cross_mask=None,
        channel_weights=None,
    ):
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(t_cond).chunk(9, dim=-1)
        sa_output = self.self_attn(
            modulate(self.norm1(x), shift_sa, scale_sa),
            mask=self_mask,
        )
        sa_write = gate_sa * sa_output
        if channel_weights is not None:
            sa_write = sa_write * channel_weights
        x = x + sa_write
        ca_output = self.cross_attn(
            modulate(self.norm2(x), shift_ca, scale_ca),
            context,
            mask=cross_mask,
        )
        ca_write = gate_ca * ca_output
        if channel_weights is not None:
            ca_write = ca_write * channel_weights
        x = x + ca_write
        mlp_output = self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        mlp_write = gate_mlp * mlp_output
        if channel_weights is not None:
            mlp_write = mlp_write * channel_weights
        x = x + mlp_write
        return x

    @staticmethod
    def _active_attention(q, k, v, heads, scale, dropout, training, mask):
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=heads)
            for t in (q, k, v)
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if mask is not None:
            scores = scores.masked_fill(
                ~mask.bool()[None, None], -torch.finfo(scores.dtype).max
            )
        weights = F.dropout(scores.softmax(dim=-1), p=dropout, training=training)
        return rearrange(torch.matmul(weights, v), "b h n d -> b n (h d)")

    def _forward_active_group(self, h, context, t_cond, indices,
                              channel_weights, self_mask, cross_mask):
        """Only selected 192-D rows/columns enter the block's heavy GEMMs."""
        width = self.norm1.normalized_shape[0]
        active = indices.numel()
        rows = torch.cat([indices + i * width for i in range(9)])
        modulation = self.adaLN_modulation[-1]
        mod = F.linear(
            F.silu(t_cond),
            modulation.weight.index_select(0, rows).index_select(1, indices),
            modulation.bias.index_select(0, rows),
        )
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = mod.chunk(9, dim=-1)

        sa = self.self_attn
        sa_input = modulate(F.layer_norm(h, (active,), eps=self.norm1.eps),
                            shift_sa, scale_sa)
        q, k, v = F.linear(
            sa_input, sa.to_qkv.weight.index_select(1, indices)
        ).chunk(3, dim=-1)
        sa_result = self._active_attention(q, k, v, sa.heads, sa.scale,
                                            sa.dropout, self.training, self_mask)
        sa_result = sa.to_out[1](F.linear(
            sa_result, sa.to_out[0].weight.index_select(0, indices),
            sa.to_out[0].bias.index_select(0, indices),
        ))
        h = h + channel_weights * gate_sa * sa_result

        ca = self.cross_attn
        ca_input = modulate(F.layer_norm(h, (active,), eps=self.norm2.eps),
                            shift_ca, scale_ca)
        q = F.linear(ca_input, ca.to_q.weight.index_select(1, indices))
        k, v = F.linear(
            context, ca.to_kv.weight.index_select(1, indices)
        ).chunk(2, dim=-1)
        ca_result = self._active_attention(q, k, v, ca.heads, ca.scale,
                                            ca.dropout, self.training, cross_mask)
        ca_result = ca.to_out[1](F.linear(
            ca_result, ca.to_out[0].weight.index_select(0, indices),
            ca.to_out[0].bias.index_select(0, indices),
        ))
        h = h + channel_weights * gate_ca * ca_result

        mlp_input = modulate(F.layer_norm(h, (active,), eps=self.norm3.eps),
                             shift_mlp, scale_mlp)
        mlp = self.mlp.net
        intermediate = F.linear(
            mlp_input, mlp[0].weight.index_select(1, indices), mlp[0].bias
        )
        intermediate = mlp[2](mlp[1](intermediate))
        mlp_result = mlp[4](F.linear(
            intermediate, mlp[3].weight.index_select(0, indices),
            mlp[3].bias.index_select(0, indices),
        ))
        return h + channel_weights * gate_mlp * mlp_result

    def forward_sparse_grouped(self, x, context, t_cond, selected_groups,
                               active_indices, channel_weights, self_mask=None,
                               cross_mask=None):
        """Dispatch clips by selected group; scatter only active updates."""
        result = x.clone()
        for group in range(active_indices.size(0)):
            members = torch.nonzero(selected_groups == group, as_tuple=True)[0]
            if members.numel() == 0:
                continue
            indices = active_indices[group]
            original = x.index_select(0, members)
            active = original.index_select(-1, indices)
            active_context = context.index_select(0, members).index_select(-1, indices)
            active_time = t_cond.index_select(0, members).index_select(-1, indices)
            weights = channel_weights.index_select(0, members).index_select(-1, indices)
            updated = self._forward_active_group(
                active, active_context, active_time, indices, weights,
                self_mask, cross_mask,
            )
            scatter_indices = indices.view(1, 1, -1).expand_as(updated)
            result = result.index_copy(
                0, members, original.scatter(-1, scatter_indices, updated)
            )
        return result

    def forward_sparse_shared(self, x, context, t_cond, selected_groups,
                              active_indices, channel_weights, self_mask=None,
                              cross_mask=None):
        """One route for all candidate actions in the same planning context."""
        group = selected_groups[:1]
        indices = active_indices.index_select(0, group).squeeze(0)
        h = x.index_select(-1, indices)
        active_context = context.index_select(-1, indices)
        active_time = t_cond.index_select(-1, indices)
        weights = channel_weights.index_select(-1, indices)
        cache_dtype = (torch.bfloat16 if torch.is_autocast_enabled("cuda")
                       else torch.float32)
        can_cache = not torch.is_grad_enabled()
        if (can_cache and self._sparse_eval_banks is not None
                and self._sparse_eval_banks[0] == cache_dtype):
            banks = self._sparse_eval_banks[1]
        else:
            banks = self._build_sparse_banks(active_indices)
            if can_cache and cache_dtype == torch.bfloat16:
                banks = {name: (weight.to(cache_dtype),
                                bias.to(cache_dtype) if bias is not None else None)
                         for name, (weight, bias) in banks.items()}
            if can_cache:
                self._sparse_eval_banks = (cache_dtype, banks)

        if (can_cache and self._sparse_eval_packed is not None
                and self._sparse_eval_packed[0] == cache_dtype):
            packed, layout = self._sparse_eval_packed[1:]
        else:
            layout = {}
            offset = 0
            for name, (weight_bank, bias_bank) in banks.items():
                weight_size = weight_bank[0].numel()
                bias_size = bias_bank[0].numel() if bias_bank is not None else 0
                layout[name] = (offset, offset + weight_size,
                                tuple(weight_bank.shape[1:]), bias_size)
                offset += weight_size + bias_size
            packed = torch.stack([
                torch.cat([
                    part
                    for weight_bank, bias_bank in banks.values()
                    for part in ((weight_bank[index].reshape(-1),)
                                 if bias_bank is None else
                                 (weight_bank[index].reshape(-1),
                                  bias_bank[index].reshape(-1)))
                ])
                for index in range(active_indices.size(0))
            ])
            if can_cache:
                self._sparse_eval_packed = (cache_dtype, packed, layout)
        selected = packed.index_select(0, group).squeeze(0)

        def linear(tensor, name):
            start, end, shape, bias_size = layout[name]
            weight = selected[start:end].view(shape)
            bias = (selected[end:end + bias_size]
                    if bias_size else None)
            return F.linear(tensor, weight, bias)

        active = indices.numel()
        mod = linear(F.silu(active_time), "mod")
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = mod.chunk(9, dim=-1)
        sa = self.self_attn
        sa_input = modulate(F.layer_norm(h, (active,), eps=self.norm1.eps),
                            shift_sa, scale_sa)
        q, k, v = linear(sa_input, "sa_qkv").chunk(3, -1)
        sa_result = self._active_attention(q, k, v, sa.heads, sa.scale,
                                            sa.dropout, self.training, self_mask)
        h = h + weights * gate_sa * sa.to_out[1](linear(sa_result, "sa_out"))
        ca = self.cross_attn
        ca_input = modulate(F.layer_norm(h, (active,), eps=self.norm2.eps),
                            shift_ca, scale_ca)
        q = linear(ca_input, "ca_q")
        k, v = linear(active_context, "ca_kv").chunk(2, -1)
        ca_result = self._active_attention(q, k, v, ca.heads, ca.scale,
                                            ca.dropout, self.training, cross_mask)
        h = h + weights * gate_ca * ca.to_out[1](linear(ca_result, "ca_out"))
        mlp = self.mlp.net
        mlp_input = modulate(F.layer_norm(h, (active,), eps=self.norm3.eps),
                             shift_mlp, scale_mlp)
        intermediate = mlp[2](mlp[1](linear(mlp_input, "mlp_up")))
        updated = h + weights * gate_mlp * mlp[4](linear(intermediate, "mlp_down"))
        return x.scatter(-1, indices.view(1, 1, -1).expand_as(updated), updated)

    def _build_sparse_banks(self, active_indices):
        """Slice every candidate submatrix once; keep autograd in training."""
        width = self.norm1.normalized_shape[0]
        modules = {
            "mod": self.adaLN_modulation[-1],
            "sa_qkv": self.self_attn.to_qkv,
            "sa_out": self.self_attn.to_out[0],
            "ca_q": self.cross_attn.to_q,
            "ca_kv": self.cross_attn.to_kv,
            "ca_out": self.cross_attn.to_out[0],
            "mlp_up": self.mlp.net[0],
            "mlp_down": self.mlp.net[3],
        }
        banks = {}
        for name, linear in modules.items():
            weights, biases = [], []
            for indices in active_indices:
                rows = (torch.cat([indices + i * width for i in range(9)])
                        if name == "mod" else
                        indices if name in {"sa_out", "ca_out", "mlp_down"}
                        else None)
                cols = (indices if name in {"mod", "sa_qkv", "ca_q", "ca_kv", "mlp_up"}
                        else None)
                weight = linear.weight
                if rows is not None:
                    weight = weight.index_select(0, rows)
                if cols is not None:
                    weight = weight.index_select(1, cols)
                weights.append(weight)
                if linear.bias is not None:
                    bias = linear.bias if rows is None else linear.bias.index_select(0, rows)
                    biases.append(bias)
            banks[name] = (torch.stack(weights),
                           torch.stack(biases) if biases else None)
        return banks

    @staticmethod
    def _batched_linear(inputs, bank, selected_groups, use_triton=False,
                        grouped_order=None):
        if use_triton:
            if grouped_order is not None:
                return selected_linear_grouped(inputs, bank, *grouped_order)
            return selected_linear(inputs, bank, selected_groups)
        weights, biases = bank
        weights = weights.index_select(0, selected_groups)
        output = torch.bmm(inputs, weights.transpose(1, 2))
        if biases is not None:
            output = output + biases.index_select(0, selected_groups)[:, None, :].to(output.dtype)
        return output

    def forward_sparse(self, x, context, t_cond, selected_groups,
                       active_indices, channel_weights, self_mask=None,
                       cross_mask=None):
        """Batch selected 192-D submatrix products across different routes."""
        if not self.training and not torch.is_autocast_enabled("cuda"):
            return self.forward_sparse_grouped(
                x, context, t_cond, selected_groups, active_indices,
                channel_weights, self_mask=self_mask, cross_mask=cross_mask,
            )
        can_cache = not self.training and not torch.is_grad_enabled()
        cache_dtype = (torch.bfloat16 if torch.is_autocast_enabled("cuda")
                       else torch.float32)
        if (can_cache and self._sparse_eval_banks is not None
                and self._sparse_eval_banks[0] == cache_dtype):
            banks = self._sparse_eval_banks[1]
        else:
            banks = self._build_sparse_banks(active_indices)
            if can_cache:
                if cache_dtype == torch.bfloat16:
                    banks = {name: (weight.to(cache_dtype),
                                    bias.to(cache_dtype) if bias is not None else None)
                             for name, (weight, bias) in banks.items()}
                self._sparse_eval_banks = (cache_dtype, banks)
        use_triton = (can_cache and cache_dtype == torch.bfloat16
                      and sparse_triton is not None and x.is_cuda)
        grouped_order = None
        if use_triton and x.size(0) >= 64:
            order = torch.argsort(selected_groups, stable=True)
            counts = torch.stack([(selected_groups == i).sum()
                                  for i in range(active_indices.size(0))])
            offsets = counts.cumsum(0) - counts
            grouped_order = (order, offsets, counts)
        linear = lambda tensor, name: self._batched_linear(
            tensor, banks[name], selected_groups, use_triton=use_triton,
            grouped_order=grouped_order)
        indices = active_indices.index_select(0, selected_groups)
        state_indices = indices[:, None, :].expand(-1, x.size(1), -1)
        context_indices = indices[:, None, :].expand(-1, context.size(1), -1)
        h = x.gather(-1, state_indices)
        active_context = context.gather(-1, context_indices)
        active_time = t_cond.gather(-1, indices[:, None, :])
        active_weights = channel_weights.gather(-1, indices[:, None, :])
        active = indices.size(-1)
        mod = linear(F.silu(active_time), "mod")
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = mod.chunk(9, dim=-1)

        sa = self.self_attn
        sa_input = modulate(F.layer_norm(h, (active,), eps=self.norm1.eps),
                            shift_sa, scale_sa)
        q, k, v = linear(sa_input, "sa_qkv").chunk(3, -1)
        sa_result = self._active_attention(q, k, v, sa.heads, sa.scale,
                                            sa.dropout, self.training, self_mask)
        sa_result = sa.to_out[1](linear(sa_result, "sa_out"))
        h = h + active_weights * gate_sa * sa_result

        ca = self.cross_attn
        ca_input = modulate(F.layer_norm(h, (active,), eps=self.norm2.eps),
                            shift_ca, scale_ca)
        q = linear(ca_input, "ca_q")
        k, v = linear(active_context, "ca_kv").chunk(2, -1)
        ca_result = self._active_attention(q, k, v, ca.heads, ca.scale,
                                            ca.dropout, self.training, cross_mask)
        ca_result = ca.to_out[1](linear(ca_result, "ca_out"))
        h = h + active_weights * gate_ca * ca_result

        mlp = self.mlp.net
        mlp_input = modulate(F.layer_norm(h, (active,), eps=self.norm3.eps),
                             shift_mlp, scale_mlp)
        intermediate = linear(mlp_input, "mlp_up")
        intermediate = mlp[2](mlp[1](intermediate))
        mlp_result = mlp[4](linear(intermediate, "mlp_down"))
        updated = h + active_weights * gate_mlp * mlp_result
        return x.scatter(-1, state_indices, updated)


class FlowTrajectoryPredictor(nn.Module):
    """Flow-matching predictor for variable-horizon latent trajectories."""

    def __init__(
        self,
        *,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        action_dim=None,
        output_dim=None,
        output_proj_hidden_dim=None,
        causal_self_attention=True,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        routing=None,
    ):
        super().__init__()
        action_dim = action_dim or input_dim
        output_dim = output_dim or input_dim
        self.hidden_dim = hidden_dim
        self.causal_self_attention = bool(causal_self_attention)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.context_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList(
            [
                FlowTrajectoryBlock(
                    hidden_dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = RMSNormMLP(
            hidden_dim,
            output_proj_hidden_dim or 4 * hidden_dim,
            output_dim,
        )
        routing_config = dict(routing or {})
        self.routing_enabled = bool(routing_config.pop("enabled", False))
        self.sparse_compute = bool(routing_config.pop("sparse_compute", False))
        self.share_route_inference = bool(routing_config.pop("share_route_inference", False))
        self.cuda_graph_enabled = False
        self._cuda_graph_cache = {}
        self._sparse_output_bank = None
        self.routing = None
        if self.routing_enabled:
            if self.causal_self_attention:
                raise ValueError(
                    "Clip-level routing pools the prediction horizon and requires "
                    "causal_self_attention=False (the Cube protocol). Causal routing "
                    "requires separate per-prefix decisions."
                )
            # Routing initialization must not alter the RNG state used to
            # initialize other JEPA components in matched dense comparisons.
            with torch.random.fork_rng(devices=[]):
                self.routing = ChannelRouting(hidden_dim, depth, **routing_config)
        if self.sparse_compute:
            if self.routing is None or self.routing.top_k != 1:
                raise ValueError("Sparse computation requires routing with top_k=1")
            if self.routing.shared_dim != 3 * self.routing.group_size or self.routing.groups != 4:
                raise ValueError("Sparse computation requires 3 shared and 4 dynamic groups")
            shared = torch.arange(self.routing.shared_dim)
            group_indices = [
                torch.cat((shared, torch.arange(
                    self.routing.shared_dim + group * self.routing.group_size,
                    self.routing.shared_dim + (group + 1) * self.routing.group_size,
                )))
                for group in range(self.routing.groups)
            ]
            self.register_buffer("sparse_group_indices", torch.stack(group_indices),
                                 persistent=False)

    def enable_cuda_graph(self, enabled=True):
        """Reuse a captured predictor for fixed-shape CEM candidate batches."""
        self.cuda_graph_enabled = bool(enabled)
        self._cuda_graph_cache.clear()

    def train(self, mode=True):
        if mode:
            self._cuda_graph_cache.clear()
            self._sparse_output_bank = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._cuda_graph_cache.clear()
        self._sparse_output_bank = None
        return super()._load_from_state_dict(*args, **kwargs)

    def _sparse_readout(self, state, selected_groups, shared_route):
        """Read only the last block's selected 192-D coordinates."""
        batch = state.size(0)
        indices = self.sparse_group_indices.index_select(0, selected_groups)
        active = state.gather(-1, indices[:, None, :].expand(-1, state.size(1), -1))
        active = F.layer_norm(active, (indices.size(1),), eps=self.norm.eps)
        norm_weight = self.norm.weight.index_select(0, indices.reshape(-1))
        norm_bias = self.norm.bias.index_select(0, indices.reshape(-1))
        active = (active * norm_weight.reshape(batch, 1, -1)
                  + norm_bias.reshape(batch, 1, -1))
        first = self.output_proj.net[0]
        if shared_route:
            weight = first.weight.index_select(1, indices[0])
            intermediate = F.linear(active, weight, first.bias)
        elif not self.training and not torch.is_autocast_enabled("cuda"):
            intermediate = active.new_empty((batch, active.size(1), first.out_features))
            for group in range(self.routing.groups):
                members = torch.nonzero(selected_groups == group, as_tuple=True)[0]
                if members.numel():
                    weight = first.weight.index_select(1, self.sparse_group_indices[group])
                    values = F.linear(active.index_select(0, members), weight, first.bias)
                    intermediate = intermediate.index_copy(0, members, values)
        else:
            can_cache = not self.training and not torch.is_grad_enabled()
            cache_dtype = (torch.bfloat16 if torch.is_autocast_enabled("cuda")
                           else torch.float32)
            if (can_cache and self._sparse_output_bank is not None
                    and self._sparse_output_bank[0] == cache_dtype):
                bank = self._sparse_output_bank[1]
            else:
                weights = torch.stack([first.weight.index_select(1, group_indices)
                                       for group_indices in self.sparse_group_indices])
                biases = first.bias.unsqueeze(0).expand(self.routing.groups, -1)
                if can_cache and cache_dtype == torch.bfloat16:
                    weights, biases = weights.to(cache_dtype), biases.to(cache_dtype)
                bank = (weights, biases)
                if can_cache:
                    self._sparse_output_bank = (cache_dtype, bank)
            if (can_cache and torch.is_autocast_enabled("cuda")
                    and sparse_triton is not None and active.is_cuda):
                if batch >= 64:
                    order = torch.argsort(selected_groups, stable=True)
                    counts = torch.stack([(selected_groups == i).sum()
                                          for i in range(self.routing.groups)])
                    offsets = counts.cumsum(0) - counts
                    intermediate = selected_linear_grouped(
                        active, bank, order, offsets, counts)
                else:
                    intermediate = selected_linear(active, bank, selected_groups)
            else:
                weights, biases = bank
                chosen = weights.index_select(0, selected_groups)
                intermediate = torch.bmm(active, chosen.transpose(1, 2))
                intermediate = (intermediate + biases.index_select(0, selected_groups)
                                [:, None, :].to(intermediate.dtype))
        for layer in self.output_proj.net[1:]:
            intermediate = layer(intermediate)
        return intermediate

    def forward_inference(self, noisy_emb, hist_emb, act_emb, t):
        if (not self.cuda_graph_enabled or self.training or not noisy_emb.is_cuda
                or torch.is_grad_enabled()
                or (self.sparse_compute and not torch.is_autocast_enabled("cuda")
                    and (not self.share_route_inference or noisy_emb.size(0) == 1))):
            return self.forward(noisy_emb, hist_emb, act_emb, t)
        key = (tuple(noisy_emb.shape), tuple(hist_emb.shape),
               tuple(act_emb.shape), tuple(t.shape), noisy_emb.dtype,
               hist_emb.dtype, act_emb.dtype, t.dtype,
               torch.is_autocast_enabled("cuda"))
        cached = self._cuda_graph_cache.get(key)
        if cached is None:
            static = tuple(tensor.clone() for tensor in
                           (noisy_emb, hist_emb, act_emb, t))
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    self.forward(*static)
            torch.cuda.current_stream().wait_stream(warmup_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.forward(*static)
            cached = (static, graph, output)
            self._cuda_graph_cache[key] = cached
        static, graph, output = cached
        for target, source in zip(static, (noisy_emb, hist_emb, act_emb, t)):
            target.copy_(source)
        graph.replay()
        return output

    def set_routing_step(self, global_step):
        if self.routing is not None:
            self.routing.set_step(global_step)

    def get_last_route_trace(self):
        return self.routing.get_last_route_trace() if self.routing is not None else []

    def _positional_embedding(self, positions, dtype):
        return timestep_embedding(positions, self.hidden_dim).to(dtype=dtype)

    def _build_cross_mask(self, hist_len, pred_len, action_len, query_num, device):
        query_future_idx = torch.arange(pred_len, device=device).repeat_interleave(
            query_num
        )
        hist_mask = torch.ones(
            pred_len * query_num,
            hist_len * query_num,
            device=device,
            dtype=torch.bool,
        )
        action_idx = torch.arange(action_len, device=device)
        if pred_len == 1 and action_len > 1:
            action_mask = torch.ones(
                pred_len * query_num, action_len, device=device, dtype=torch.bool
            )
        else:
            action_mask = action_idx[None, :] <= query_future_idx[:, None]
        return torch.cat([hist_mask, action_mask], dim=-1)

    def _build_causal_self_mask(self, frame_count, query_num, device):
        """Build a causal mask over latent frames.

        Tokens within the same latent frame attend bidirectionally. A frame
        attends only to itself and earlier frames, preventing later predicted
        states from leaking into earlier ones.
        """
        frame_index = torch.arange(
            frame_count, device=device
        ).repeat_interleave(query_num)
        return frame_index[None, :] <= frame_index[:, None]

    def _build_future_self_mask(self, pred_len, query_num, device):
        if not self.causal_self_attention:
            return None
        return self._build_causal_self_mask(pred_len, query_num, device)

    def forward(
        self, noisy_emb, hist_emb, act_emb, t,
        routing_overrides=None, routing_record=None,
    ):
        """
        noisy_emb: (B, P, K, D)
        hist_emb: (B, H, K, D)
        act_emb: (B, F, A)
        t: (B,)
        """
        b, pred_len, query_num, _ = noisy_emb.shape
        hist_len = hist_emb.size(1)
        action_len = act_emb.size(1)
        dtype = noisy_emb.dtype
        device = noisy_emb.device

        action_pos = torch.arange(
            hist_len,
            hist_len + action_len,
            device=device,
            dtype=torch.float32,
        )
        if pred_len == action_len:
            pred_pos = action_pos
        elif pred_len == 1:
            pred_pos = action_pos[-1:].clone()
        else:
            pred_pos = action_pos[:pred_len]
        pred_pos_emb = self._positional_embedding(pred_pos, dtype)
        action_pos_emb = self._positional_embedding(action_pos, dtype)

        x = self.input_proj(noisy_emb)
        x = x + pred_pos_emb[None, :, None, :]
        x = rearrange(x, "b p k d -> b (p k) d")
        x = self.dropout(x)

        hist_ctx = self.context_proj(hist_emb)
        hist_ctx = rearrange(hist_ctx, "b h k d -> b (h k) d")

        act_ctx = self.action_proj(act_emb)
        act_ctx = act_ctx + action_pos_emb[None, :, :]
        context = torch.cat([hist_ctx, act_ctx], dim=1)

        t_emb = timestep_embedding(t, self.hidden_dim).to(dtype=dtype)
        t_cond = self.time_mlp(t_emb)[:, None, :]
        self_mask = self._build_future_self_mask(
            pred_len,
            query_num,
            device,
        )
        cross_mask = self._build_cross_mask(
            hist_len, pred_len, action_len, query_num, device=device
        )

        record_routes = False
        if self.routing is not None:
            record_routes = self.routing.begin_forward(routing_overrides, routing_record)
            hist_summary = hist_ctx.mean(dim=1)
            if hist_len > 1:
                visible_hist = hist_ctx.reshape(b, hist_len, query_num, self.hidden_dim)
                visible_delta = (visible_hist[:, -1] - visible_hist[:, 0]).mean(dim=1)
                visible_delta = visible_delta / (hist_len - 1)
            else:
                visible_delta = torch.zeros_like(hist_summary)
            action_summary = act_ctx.mean(dim=1)

        for block_index, layer in enumerate(self.layers):
            channel_weights = None
            if self.routing is not None:
                router_inputs = torch.cat(
                    [x.mean(dim=1), hist_summary, visible_delta,
                     action_summary, t_cond[:, 0]], dim=-1,
                )
                override = None if routing_overrides is None else routing_overrides[block_index]
                shared_route = (self.share_route_inference and not self.training
                                and b > 1 and routing_overrides is None)
                if shared_route:
                    channel_weights = self.routing.route(
                        block_index, router_inputs.mean(0, keepdim=True), record=False,
                    ).expand(b, -1, -1)
                else:
                    channel_weights = self.routing.route(
                        block_index, router_inputs, record=record_routes, override=override,
                    )
            if self.sparse_compute:
                chosen = channel_weights[:, 0, self.routing.shared_dim:]
                chosen = chosen.reshape(b, self.routing.groups,
                                        self.routing.group_size).abs().sum(-1).argmax(-1)
                sparse_forward = (layer.forward_sparse_shared if shared_route
                                  else layer.forward_sparse)
                x = sparse_forward(
                    x, context, t_cond, chosen, self.sparse_group_indices,
                    channel_weights, self_mask=self_mask, cross_mask=cross_mask,
                )
            else:
                x = layer(
                    x,
                    context,
                    t_cond,
                    self_mask=self_mask,
                    cross_mask=cross_mask,
                    channel_weights=channel_weights,
                )

        if self.sparse_compute:
            x = self._sparse_readout(x, chosen, shared_route)
        else:
            x = self.norm(x)
            x = self.output_proj(x)
        return rearrange(
            x,
            "b (p k) d -> b p k d",
            p=pred_len,
            k=query_num,
        )
