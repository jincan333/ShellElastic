"""Dense elastic baselines: the fully elastic model and MatFormer.

Both share one set of maximal-width weights and run each sub-network with its own forward pass
over a top-left slice of every matrix (unlike ShellElastic, whose sub-networks are prefixes of a
single pass):

  * ``mode="full"`` (fully elastic): the residual stream, the attention heads and the FFN hidden
    width all nest; sub-network ``s`` has width ``s * g`` and FFN width ``s * h``.
  * ``mode="ffn"`` (MatFormer): the residual stream, attention, embedding and readout keep the full
    model width ``g``; only the FFN hidden width ``s * h`` nests.

Every hidden matrix is stored as a frozen initialization snapshot ``Theta0`` plus a trainable
shift ``Delta`` (zero at initialization). The weight used by a sub-network whose input width is
``s_in`` base units is ``s_in**-1/2 * Theta0 + s_in**-1 * Delta`` (sliced), which realizes the
per-width initialization and learning-rate factors of mu-Elastic for a matrix shared by
sub-networks of different widths (Equation 23, Tables 5 and 6); decoupled weight decay on ``Delta``
decays toward the initialization.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from elastic.models.layers import (
    Embedding,
    Linear,
    Norm,
    Rotary,
    attention_bias,
    causal_attention,
    checkpointed_blocks,
    depth_multiplier,
    full_norm,
    grouped_norm,
    swiglu,
    trunc_normal_,
)
from elastic.models.outputs import WidthOutput, split_loss

MODES = ("full", "ffn")


@dataclass
class DenseElasticConfig:
    """Configuration of a fully elastic model (``mode="full"``) or a MatFormer (``mode="ffn"``)."""

    vocab_size: int = 50432
    seq_len: int = 2048
    n_layers: int = 32
    mode: str = "full"
    base_width: int = 64  # g: residual width per unit ("full") or the fixed model width ("ffn")
    supported: list[int] = field(default_factory=lambda: [6, 7, 8, 9, 10])  # S, width ratios
    head_dim: int = 64
    ffn_hidden: int = 256  # FFN hidden width per unit (h)
    gated_attention: bool = True
    norm: str = "rms"
    norm_eps: float = 1e-5
    residual_scale: float = 1.0
    readout_scaling: str = "init_lr"  # "init_lr" | "multiplier"

    parameterization: str = "mu_elastic"  # "mu_elastic" | "sp"
    proxy_width: int = 128
    init_std: float = 3.3e-3
    alpha: float | str | None = None
    ffn_expansion: float | None = None
    width_multiplier: float | None = None
    internal_scaling: bool | None = None

    trained: list[int] | None = None
    activation_checkpointing: str = "0/1"


class SplitLinear(nn.Module):
    """Maximal-width linear map stored as a frozen snapshot plus a trainable shift.

    ``out_groups > 1`` stores fused projections (e.g. queries, keys and values); each group nests
    independently, so a sub-network reads the first rows of every group.
    """

    def __init__(self, in_unit: int, out_unit: int, max_in: int, max_out: int, std: float,
                 internal_scaling: bool = True, out_groups: int = 1) -> None:
        super().__init__()
        if out_unit % out_groups != 0:
            raise ValueError(f"out_unit {out_unit} must be divisible by out_groups {out_groups}")
        self.in_unit = int(in_unit)
        self.out_unit = int(out_unit)
        self.out_groups = int(out_groups)
        self.group_unit = self.out_unit // self.out_groups
        self.max_in = int(max_in) * self.in_unit
        self.max_group_out = int(max_out) * self.group_unit
        self.std = float(std)
        self.internal_scaling = bool(internal_scaling)
        shape = (self.out_groups * self.max_group_out, self.max_in)
        self.weight_init = nn.Parameter(torch.empty(shape), requires_grad=False)
        self.weight_delta = nn.Parameter(torch.zeros(shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight_init, self.std)
        with torch.no_grad():
            self.weight_delta.zero_()

    def _slice(self, w: torch.Tensor, n_in: int, n_group_out: int) -> torch.Tensor:
        if self.out_groups == 1:
            return w[:n_group_out, :n_in]
        return (
            w.view(self.out_groups, self.max_group_out, self.max_in)[:, :n_group_out, :n_in]
            .reshape(self.out_groups * n_group_out, n_in)
        )

    def weight(self, s_out: int, s_in: int) -> torch.Tensor:
        """Effective weight for output ratio ``s_out`` and input ratio ``s_in``."""
        n_in = s_in * self.in_unit
        n_out = s_out * self.group_unit
        init_mult = float(s_in) ** -0.5 if self.internal_scaling else 1.0
        shift_mult = float(s_in) ** -1.0 if self.internal_scaling else 1.0
        return init_mult * self._slice(self.weight_init, n_in, n_out) + shift_mult * self._slice(
            self.weight_delta, n_in, n_out
        )

    def forward(self, x: torch.Tensor, s_out: int, s_in: int) -> torch.Tensor:
        return F.linear(x, self.weight(s_out, s_in))


class DenseAttention(nn.Module):
    def __init__(self, cfg: DenseElasticConfig, max_ratio: int, std: float, internal: bool) -> None:
        super().__init__()
        g = cfg.base_width
        if g % cfg.head_dim != 0:
            raise ValueError(f"base width {g} must be a multiple of head_dim {cfg.head_dim}")
        self.g = g
        self.head_dim = cfg.head_dim
        self.heads_per_unit = g // cfg.head_dim
        width = max_ratio * g
        self.qkv = SplitLinear(g, 3 * g, max_ratio, max_ratio, std, internal, out_groups=3)
        self.gate = SplitLinear(g, g, max_ratio, max_ratio, std, internal) if cfg.gated_attention else None
        self.out = SplitLinear(g, g, max_ratio, max_ratio, std, internal)
        self.q_norm = Norm(width, cfg.norm, cfg.norm_eps)
        self.k_norm = Norm(width, cfg.norm, cfg.norm_eps)
        self.rotary = Rotary(cfg.head_dim, cfg.seq_len)

    def forward(self, x: torch.Tensor, r: int, attn_bias: torch.Tensor | None = None,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False):
        """Returns the output and, with ``use_cache``, the keys and values of all positions so far."""
        batch, seq, _ = x.shape
        heads = r * self.heads_per_unit
        n = r * self.g
        gate = None
        if self.gate is not None:
            gate = self.gate(x, r, r).view(batch, seq, heads, self.head_dim)
        q, k, v = self.qkv(x, r, r).chunk(3, dim=-1)
        q = grouped_norm(self.q_norm, q, n, self.head_dim).view(batch, seq, heads, self.head_dim)
        k = grouped_norm(self.k_norm, k, n, self.head_dim).view(batch, seq, heads, self.head_dim)
        v = v.view(batch, seq, heads, self.head_dim)
        offset = 0 if past_kv is None else past_kv[0].shape[1]
        q, k, v = self.rotary(q, k, v, offset=offset)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        out = causal_attention(q, k, v, attn_bias)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.out(out.view(batch, seq, n), r, r), ((k, v) if use_cache else None)


class DenseMLP(nn.Module):
    def __init__(self, cfg: DenseElasticConfig, max_res: int, max_ffn: int, std: float, down_std: float,
                 expansion: float, internal: bool) -> None:
        super().__init__()
        g, h = cfg.base_width, cfg.ffn_hidden
        self.w1 = SplitLinear(g, h, max_res, max_ffn, std, internal)
        self.w3 = SplitLinear(g, h, max_res, max_ffn, std, internal)
        self.w2 = SplitLinear(h, g, max_ffn, max_res, down_std, internal)
        self.expansion = float(expansion)

    def forward(self, x: torch.Tensor, r: int, s: int) -> torch.Tensor:
        """``r``: residual ratio, ``s``: FFN hidden ratio."""
        return self.w2(swiglu(self.w1(x, s, r), self.w3(x, s, r)), r, s) / self.expansion


class DenseBlock(nn.Module):
    def __init__(self, cfg: DenseElasticConfig, max_res: int, max_ffn: int, std: float, down_std: float,
                 expansion: float, internal: bool) -> None:
        super().__init__()
        width = max_res * cfg.base_width
        self.g = cfg.base_width
        self.attn_norm = Norm(width, cfg.norm, cfg.norm_eps)
        self.attn = DenseAttention(cfg, max_res, std, internal)
        self.ffn_norm = Norm(width, cfg.norm, cfg.norm_eps)
        self.mlp = DenseMLP(cfg, max_res, max_ffn, std, down_std, expansion, internal)
        self.res_mult = depth_multiplier(cfg.n_layers, cfg.residual_scale)
        self.use_checkpoint = False

    def _forward(self, x: torch.Tensor, r: int, s: int, attn_bias=None, past_kv=None, use_cache: bool = False):
        n = r * self.g
        a, kv = self.attn(full_norm(self.attn_norm, x, n), r, attn_bias, past_kv, use_cache)
        h = x + a * self.res_mult
        return h + self.mlp(full_norm(self.ffn_norm, h, n), r, s) * self.res_mult, kv

    def forward(self, x: torch.Tensor, r: int, s: int, attn_bias=None, past_kv=None, use_cache: bool = False):
        if self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(self._forward, x, r, s, attn_bias, past_kv, use_cache, use_reentrant=False)
        return self._forward(x, r, s, attn_bias, past_kv, use_cache)


class DenseElasticLM(nn.Module):
    """Fully elastic model / MatFormer language model (see module doc)."""

    def __init__(self, cfg: DenseElasticConfig) -> None:
        super().__init__()
        if cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {cfg.mode!r}")
        if cfg.parameterization not in ("mu_elastic", "sp"):
            raise ValueError(f"unknown parameterization {cfg.parameterization!r}")
        if cfg.readout_scaling not in ("init_lr", "multiplier"):
            raise ValueError(f"unknown readout_scaling {cfg.readout_scaling!r}")
        supported = [int(s) for s in cfg.supported]
        if supported != sorted(set(supported)) or supported[0] < 1:
            raise ValueError(f"supported ratios must be strictly increasing positive integers, got {supported}")
        self.cfg = cfg
        self.supported = supported
        self.M = supported[-1]
        self.g = cfg.base_width
        self.ffn_only = cfg.mode == "ffn"
        # Residual ratio of every sub-network: follows s ("full") or stays 1 ("ffn").
        self.max_res = 1 if self.ffn_only else self.M
        self.width = self.max_res * self.g
        trained = supported if cfg.trained is None else sorted(int(s) for s in cfg.trained)
        if not trained or any(s not in supported for s in trained):
            raise ValueError(f"trained ratios {trained} must be a non-empty subset of {supported}")
        self.trained = trained

        mu = cfg.parameterization == "mu_elastic"
        self.m = float(cfg.width_multiplier) if cfg.width_multiplier is not None else (
            self.g / cfg.proxy_width if mu else 1.0
        )
        self.internal = bool(cfg.internal_scaling) if cfg.internal_scaling is not None else mu
        self.e = float(cfg.ffn_expansion) if cfg.ffn_expansion is not None else (
            cfg.ffn_hidden / self.g if mu else 1.0
        )
        alpha = cfg.alpha if cfg.alpha is not None else ("learnable" if mu else 1.0)
        if isinstance(alpha, str) and alpha != "learnable":
            raise ValueError(f"alpha must be a float or 'learnable', got {alpha!r}")
        sigma = float(cfg.init_std)
        std = sigma / math.sqrt(self.m)
        down_std = std * math.sqrt(self.e)

        self.embed = Embedding(cfg.vocab_size, self.width, sigma)
        self.blocks = nn.ModuleList(
            DenseBlock(cfg, self.max_res, self.M, std, down_std, self.e, self.internal) for _ in range(cfg.n_layers)
        )
        self.final_norm = Norm(self.width, cfg.norm, cfg.norm_eps)

        # Readout at the residual width. With several trained residual widths and width factors in
        # the initialization/learning rate, it is split like the hidden matrices.
        init_lr = cfg.readout_scaling == "init_lr"
        self.shared_split = init_lr and not self.ffn_only and trained != [self.M]
        if self.shared_split:
            readout_std = self.readout_std(1)
        else:
            readout_std = self.readout_std(self.max_res) if init_lr else sigma
        self.readout = Linear(self.width, cfg.vocab_size, readout_std)
        if self.shared_split:
            self.readout.weight.requires_grad_(False)
            self.readout_delta = nn.Parameter(torch.zeros(cfg.vocab_size, self.width))
        else:
            self.register_parameter("readout_delta", None)

        self.alpha_learnable = isinstance(alpha, str)
        n_alpha = 1 if self.ffn_only else len(supported)
        if self.alpha_learnable:
            self.alpha = nn.Parameter(torch.ones(n_alpha))
            self.alpha_value = None
        else:
            self.register_parameter("alpha", None)
            self.alpha_value = float(alpha)
        self.set_activation_checkpointing(cfg.activation_checkpointing)

    # ------------------------------------------------------------------------------------------

    def readout_std(self, r: int) -> float:
        r_int = r if self.internal else 1
        return self.cfg.init_std / math.sqrt(self.m * r_int)

    @torch.no_grad()
    def init_weights(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        if self.readout_delta is not None:
            self.readout_delta.zero_()
        if self.alpha is not None:
            self.alpha.fill_(1.0)

    def set_activation_checkpointing(self, fraction: str | float) -> None:
        for block, use in zip(self.blocks, checkpointed_blocks(fraction, len(self.blocks))):
            block.use_checkpoint = use

    def residual_ratio(self, s: int) -> int:
        return 1 if self.ffn_only else s

    def readout_weight(self, r: int) -> torch.Tensor:
        n = r * self.g
        if self.shared_split:
            r_int = float(r) if self.internal else 1.0
            return r_int**-0.5 * self.readout.weight[:, :n] + r_int**-1.0 * self.readout_delta[:, :n]
        return self.readout.weight[:, :n]

    def alpha_of(self, s: int):
        if self.alpha_learnable:
            return self.alpha[0 if self.ffn_only else self.supported.index(s)]
        return self.alpha_value

    def readout_multiplier(self, s: int):
        r = self.residual_ratio(s)
        if self.cfg.readout_scaling == "init_lr":
            return self.alpha_of(s)
        r_eff = r if self.internal else 1
        return self.alpha_of(s) / (r_eff * self.m)

    def logits(self, hidden: torch.Tensor, s: int) -> torch.Tensor:
        r = self.residual_ratio(s)
        return F.linear(hidden, self.readout_weight(r)) * self.readout_multiplier(s)

    # ------------------------------------------------------------------------------------------

    def _trunk(self, input_ids: torch.Tensor, s: int, attention_mask=None, past_key_values=None,
               use_cache: bool = False):
        if s not in self.supported:
            raise ValueError(f"ratio {s} is not a supported ratio {self.supported}")
        r = self.residual_ratio(s)
        n = r * self.g
        x = self.embed(input_ids)[..., :n]
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        bias = attention_bias(attention_mask, x, past_len)
        cache = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, kv = block(x, r, s, bias, past, use_cache)
            if use_cache:
                cache.append(kv)
        return full_norm(self.final_norm, x, n), cache

    def forward_width(self, input_ids: torch.Tensor, s: int, targets: torch.Tensor | None = None,
                      loss_fn=None) -> WidthOutput:
        x, _ = self._trunk(input_ids, s)
        r = self.residual_ratio(s)
        n = r * self.g
        loss = z_loss = None
        if loss_fn is not None:
            if targets is None:
                raise ValueError("targets are required when loss_fn is given")
            weight = self.readout_weight(r)
            if not torch.is_grad_enabled():
                weight = weight.detach()
            loss, z_loss = split_loss(loss_fn(weight, (x * self.readout_multiplier(s)).reshape(-1, n), targets.reshape(-1)))
        return WidthOutput(hidden=x, loss=loss, z_loss=z_loss)

    def forward(self, input_ids: torch.Tensor, widths: list[int] | None = None, targets: torch.Tensor | None = None,
                loss_fn=None, grad_widths: set[int] | None = None) -> dict[int, WidthOutput]:
        """One independent forward pass per requested ratio."""
        widths = list(self.supported) if widths is None else sorted(int(s) for s in widths)
        outputs: dict[int, WidthOutput] = {}
        for s in widths:
            want_grad = grad_widths is None or s in grad_widths
            with nullcontext() if want_grad else torch.no_grad():
                outputs[s] = self.forward_width(input_ids, s, targets, loss_fn)
        return outputs

    def infer(self, input_ids: torch.Tensor, width: int | None = None, attention_mask: torch.Tensor | None = None,
              past_key_values: list | None = None, use_cache: bool = False, logits_to_keep: int = 0):
        """Float32 logits of sub-network ``width`` (default: the largest) for evaluation and generation.

        ``attention_mask`` (``[batch, total_len]``, 0 at padding) covers the cached and the new
        positions; ``past_key_values`` is the cache returned by the previous call;
        ``logits_to_keep > 0`` reads out only the last positions. Returns ``(logits, cache)`` with
        ``cache=None`` unless ``use_cache``.
        """
        s = self.M if width is None else int(width)
        x, cache = self._trunk(input_ids, s, attention_mask, past_key_values, use_cache)
        if logits_to_keep:
            x = x[:, -logits_to_keep:]
        weight = self.readout_weight(self.residual_ratio(s))
        return F.linear(x, weight * self.readout_multiplier(s)).float(), cache

    # ------------------------------------------------------------------------------------------

    def optim_spec(self) -> dict[str, tuple[float, bool]]:
        """Per-parameter ``(q, decay)``: learning rate ``eta_0 / q`` and weight decay ``lambda_0 * q``.

        Shifts and split readouts carry the transfer factor m (their per-width factor is applied in
        the forward pass); an unsplit readout under ``init_lr`` scaling behaves like a hidden weight
        of its width; the embedding keeps the base values; gains and alpha are not decayed.
        """
        spec: dict[str, tuple[float, bool]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("weight_delta") or name == "readout_delta":
                spec[name] = (self.m, True)
            elif name == "readout.weight" and self.cfg.readout_scaling == "init_lr":
                spec[name] = ((float(self.max_res) if self.internal else 1.0) * self.m, True)
            else:
                spec[name] = (1.0, p.ndim > 1)
        return spec

    def num_params(self, s: int) -> int:
        """Parameters read by sub-network ``s`` (non-zero weights, as in Tables 7 and 10)."""
        r = self.residual_ratio(s)
        n = r * self.g
        hidden = s * self.cfg.ffn_hidden
        per_layer = 3 * n * n + n * n + (n * n if self.cfg.gated_attention else 0)  # qkv, out, gate
        per_layer += 3 * n * hidden  # w1, w2, w3
        per_layer += 4 * n  # attn/ffn norms and q/k norms
        return self.cfg.n_layers * per_layer + 2 * self.cfg.vocab_size * n + n
