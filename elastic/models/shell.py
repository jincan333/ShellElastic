"""ShellElastic: an elastic transformer whose sub-networks are prefixes of one trunk pass.

Hidden dimensions are organized into a dense *core* followed by *shells*. For base width ``g``
and supported widths ``S = {s_1 < ... < s_K = M}`` (in units of ``g``), group ``k`` holds
features ``[s_{k-1} g, s_k g)`` and every hidden matrix is block lower triangular: output
group ``k`` reads only the first ``s_k g`` inputs. Normalization acts within groups, attention
heads never straddle a group boundary, and every supported width has its own terminal norm and
readout, so one full-width forward pass contains every sub-network's exact hidden states
(Section 3.1, Proposition 1).

The same class also covers two special cases used in the paper:
  * StairFormer (Section 4.3, Appendix I.1): equal-width groups ``S = {1, ..., M}`` that
    normalize over cumulative prefixes and share one terminal-norm gain and one readout
    (``trunk_norm="cumulative"``, ``separate_terminal_norms=False``, ``readout="shared"``); the
    shared readout takes the split of Equation 23 (Appendix D).
  * Non-elastic models (the muP proxy and the standalone baselines): ``S = {1}`` with the
    base width equal to the model width.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from elastic.kernels.slab_linear import slab_linear, slab_partition, slab_supported
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
    head_norm,
    segment_norm,
    swiglu,
    trunc_normal_,
)
from elastic.models.outputs import WidthOutput, split_loss

TRUNK_NORMS = ("group", "block", "cumulative")
TERMINAL_NORMS = ("full", "trunk")
READOUTS = ("separate", "shared")
READOUT_SCALINGS = ("init_lr", "multiplier")
PARAMETERIZATIONS = ("mu_elastic", "sp")
MATMUL_MODES = ("pad", "groups", "slab")


@dataclass
class ShellConfig:
    """Configuration of a ShellElastic (or StairFormer / standalone) language model."""

    # Architecture.
    vocab_size: int = 50432
    seq_len: int = 2048
    n_layers: int = 32
    base_width: int = 128  # g
    supported: list[int] = field(default_factory=lambda: [4, 5, 6, 7, 8])  # S, in units of g
    head_dim: int = 128
    ffn_hidden: int = 128  # FFN width per base unit (e * g)
    gated_attention: bool = True
    norm: str = "rms"  # "rms" | "layer"
    norm_eps: float = 1e-5
    trunk_norm: str = "group"  # "group" (group-local) | "block" (every g features) | "cumulative"
    terminal_norm: str = "full"  # "full" (over all s*g features) | "trunk" (same form as the trunk)
    separate_terminal_norms: bool = True  # one terminal-norm gain per trained width
    readout: str = "separate"  # "separate" (U_s per trained width) | "shared" (sliced)
    readout_scaling: str = "init_lr"  # "init_lr" (width factor in init/LR) | "multiplier"
    residual_scale: float = 1.0  # residual-branch multiplier = residual_scale * 32 / n_layers

    # Parameterization (Table 1): mu_elastic or the standard parameterization (sp).
    parameterization: str = "mu_elastic"
    proxy_width: int = 128  # d0; the muTransfer factor is m = g / d0
    init_std: float = 3.3e-3  # sigma_0
    alpha: float | str | None = None  # readout scalar alpha_s: "learnable" or a float (default by param.)
    ffn_expansion: float | None = None  # e used by the scaling rules (default ffn_hidden / g, 1 under sp)
    width_multiplier: float | None = None  # override m (default g / d0, 1 under sp)
    internal_scaling: bool | None = None  # override the per-group factor s (default: mu_elastic)

    # Training features that live in the model.
    trained: list[int] | None = None  # widths with a non-zero loss weight (default: all of S)
    stage_grad_scale: float = 1.0  # gamma of the stage-residual gradient scaling ablation (Equation 27)
    width_router: bool = False  # learned per-token router over the widths (ensemble ablation (8))
    width_router_hidden: int = 256

    # Performance.
    matmul: str = "pad"  # "pad" | "groups" | "slab" (slab GEMMs where supported, else pad)
    fused_norm: bool = True  # Triton group-local and q/k RMSNorms; single-width models always use PyTorch norms
    activation_checkpointing: str = "0/1"  # fraction of the blocks: "0/1" (none), "1/2", "1/1" (all), ...


def _check(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


class ShellLinear(nn.Module):
    """Block-lower-triangular linear map.

    ``boundaries`` are the supported widths ``(s_1, ..., s_K)`` in base units. Output group
    ``k`` has ``(s_k - s_{k-1}) * out_unit`` rows and reads the first ``s_k * in_unit`` inputs;
    it is stored as its own parameter ``weight_{k}`` (upper-triangular blocks do not exist).
    """

    def __init__(
        self,
        in_unit: int,
        out_unit: int,
        boundaries: tuple[int, ...],
        base_std: float,
        internal_scaling: bool = True,
        mode: str = "pad",
    ) -> None:
        super().__init__()
        self.in_unit = int(in_unit)
        self.out_unit = int(out_unit)
        self.boundaries = tuple(int(b) for b in boundaries)
        self.base_std = float(base_std)
        self.internal_scaling = bool(internal_scaling)
        self.mode = _check(mode, MATMUL_MODES, "matmul")
        lo = 0
        self.group_rows: list[int] = []
        for k, hi in enumerate(self.boundaries):
            rows = (hi - lo) * self.out_unit
            self.group_rows.append(rows)
            self.register_parameter(f"weight_{k}", nn.Parameter(torch.empty(rows, hi * self.in_unit)))
            lo = hi
        self.reset_parameters()

    def group_weight(self, k: int) -> torch.Tensor:
        return getattr(self, f"weight_{k}")

    def group_std(self, k: int) -> float:
        """Initialization std of group ``k``: base_std / sqrt(s_k) under internal scaling."""
        return self.base_std / math.sqrt(self.boundaries[k]) if self.internal_scaling else self.base_std

    def reset_parameters(self) -> None:
        for k in range(len(self.boundaries)):
            trunc_normal_(self.group_weight(k), self.group_std(k))

    def num_groups(self, s: int) -> int:
        if s not in self.boundaries:
            raise ValueError(f"width {s} is not a supported width {self.boundaries}")
        return self.boundaries.index(s) + 1

    def dense_weight(self, s: int) -> torch.Tensor:
        """The (s*out_unit, s*in_unit) block-lower-triangular matrix of width ``s``."""
        k_active = self.num_groups(s)
        n_in = s * self.in_unit
        rows = []
        for k in range(k_active):
            w = self.group_weight(k)
            pad = n_in - w.shape[1]
            rows.append(F.pad(w, (0, pad)) if pad > 0 else w)
        return torch.cat(rows, dim=0)

    def forward(self, x: torch.Tensor, s: int) -> torch.Tensor:
        k_active = self.num_groups(s)
        if self.mode == "slab" and k_active > 1:
            weights = tuple(self.group_weight(k) for k in range(k_active))
            units = tuple(r // self.out_unit for r in self.group_rows[:k_active])
            unit_reads = tuple(self.boundaries[k] * self.in_unit for k in range(k_active) for _ in range(units[k]))
            if slab_supported(x, weights, unit_reads, self.out_unit):
                slabs = slab_partition(
                    tuple(self.group_rows[:k_active]), self.boundaries[:k_active], self.out_unit, units
                )
                return slab_linear(x, weights, slabs)
        if self.mode == "groups":
            outs = [
                F.linear(x[..., : self.boundaries[k] * self.in_unit], self.group_weight(k))
                for k in range(k_active)
            ]
            return torch.cat(outs, dim=-1)
        return F.linear(x, self.dense_weight(s))


class ShellAttention(nn.Module):
    """Causal self-attention whose heads nest with the supported widths."""

    def __init__(self, cfg: ShellConfig, internal_scaling: bool, base_std: float, out_std: float) -> None:
        super().__init__()
        g = cfg.base_width
        self.g = g
        self.head_dim = cfg.head_dim
        if g % cfg.head_dim != 0:
            raise ValueError(f"base width {g} must be a multiple of head_dim {cfg.head_dim}")
        self.heads_per_unit = g // cfg.head_dim
        bounds = tuple(cfg.supported)
        width = max(bounds) * g
        # Per base unit the fused projection emits [queries | keys | values] of g features each.
        self.qkv = ShellLinear(g, 3 * g, bounds, base_std, internal_scaling, cfg.matmul)
        self.gate = ShellLinear(g, g, bounds, base_std, internal_scaling, cfg.matmul) if cfg.gated_attention else None
        self.out = ShellLinear(g, g, bounds, out_std, internal_scaling, cfg.matmul)
        self.q_norm = Norm(width, cfg.norm, cfg.norm_eps)
        self.k_norm = Norm(width, cfg.norm, cfg.norm_eps)
        self.rotary = Rotary(cfg.head_dim, cfg.seq_len)
        self.elastic = len(bounds) > 1

    def _qk_norm(self, norm: Norm, x: torch.Tensor, n: int) -> torch.Tensor:
        if self.elastic:
            return head_norm(norm, x, n, self.head_dim)
        return grouped_norm(norm, x.reshape(*x.shape[:-2], n), n, self.head_dim)

    def forward(self, x: torch.Tensor, s: int, attn_bias: torch.Tensor | None = None,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False):
        """Returns the output and, with ``use_cache``, the keys and values of all positions so far."""
        batch, seq, _ = x.shape
        n = s * self.g
        heads = s * self.heads_per_unit
        gate = None
        if self.gate is not None:
            gate = self.gate(x, s).view(batch, seq, heads, self.head_dim)
        qkv = self.qkv(x, s).view(batch, seq, s, 3 * self.g)
        q, k, v = qkv.chunk(3, dim=-1)  # each [batch, seq, s, g]
        q = self._qk_norm(self.q_norm, q, n).reshape(batch, seq, heads, self.head_dim)
        k = self._qk_norm(self.k_norm, k, n).reshape(batch, seq, heads, self.head_dim)
        v = v.reshape(batch, seq, heads, self.head_dim)
        offset = 0 if past_kv is None else past_kv[0].shape[1]
        q, k, v = self.rotary(q, k, v, offset=offset)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        out = causal_attention(q, k, v, attn_bias)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.out(out.view(batch, seq, n), s), ((k, v) if use_cache else None)


class ShellMLP(nn.Module):
    """SwiGLU FFN with block-lower-triangular up, gate and down projections."""

    def __init__(self, cfg: ShellConfig, internal_scaling: bool, base_std: float, down_std: float,
                 expansion: float) -> None:
        super().__init__()
        bounds = tuple(cfg.supported)
        g, h = cfg.base_width, cfg.ffn_hidden
        self.w1 = ShellLinear(g, h, bounds, base_std, internal_scaling, cfg.matmul)
        self.w3 = ShellLinear(g, h, bounds, base_std, internal_scaling, cfg.matmul)
        self.w2 = ShellLinear(h, g, bounds, down_std, internal_scaling, cfg.matmul)
        self.expansion = float(expansion)

    def forward(self, x: torch.Tensor, s: int) -> torch.Tensor:
        return self.w2(swiglu(self.w1(x, s), self.w3(x, s)), s) / self.expansion


class ShellBlock(nn.Module):
    def __init__(self, cfg: ShellConfig, norm_scope: str, norm_bounds: tuple[int, ...] | None,
                 internal_scaling: bool, hidden_std: float, down_std: float, expansion: float) -> None:
        super().__init__()
        width = max(cfg.supported) * cfg.base_width
        self.g = cfg.base_width
        self.norm_scope = norm_scope
        self.norm_bounds = norm_bounds
        self.attn_norm = Norm(width, cfg.norm, cfg.norm_eps)
        self.attn = ShellAttention(cfg, internal_scaling, hidden_std, hidden_std)
        self.ffn_norm = Norm(width, cfg.norm, cfg.norm_eps)
        self.mlp = ShellMLP(cfg, internal_scaling, hidden_std, down_std, expansion)
        self.res_mult = depth_multiplier(cfg.n_layers, cfg.residual_scale)
        self.use_checkpoint = False

    def _norm(self, norm: Norm, x: torch.Tensor, s: int) -> torch.Tensor:
        if self.norm_scope == "full":
            return full_norm(norm, x, s * self.g)
        return segment_norm(norm, x, s * self.g, self.g, self.norm_scope, self.norm_bounds)

    def _forward(self, x: torch.Tensor, s: int, attn_bias=None, past_kv=None, use_cache: bool = False):
        a, kv = self.attn(self._norm(self.attn_norm, x, s), s, attn_bias, past_kv, use_cache)
        h = x + a * self.res_mult
        return h + self.mlp(self._norm(self.ffn_norm, h, s), s) * self.res_mult, kv

    def forward(self, x: torch.Tensor, s: int, attn_bias=None, past_kv=None, use_cache: bool = False):
        if self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(self._forward, x, s, attn_bias, past_kv, use_cache, use_reentrant=False)
        return self._forward(x, s, attn_bias, past_kv, use_cache)


class _GradScale(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by ``gamma`` in the backward pass."""

    @staticmethod
    def forward(ctx, x, gamma):  # noqa: ANN001
        ctx.gamma = gamma
        return x

    @staticmethod
    def backward(ctx, grad):  # noqa: ANN001
        return ctx.gamma * grad, None


def grad_scale(x: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma == 1.0:
        return x
    if gamma == 0.0:
        return x.detach()
    return _GradScale.apply(x, gamma)


class WidthRouter(nn.Module):
    """Per-token router over the supported widths (MLP on the widest post-norm hidden state).

    The output layer is initialized to zero, so the routing weights start uniform.
    """

    def __init__(self, in_dim: int, hidden: int, num_widths: int, init_std: float) -> None:
        super().__init__()
        self.init_std = float(init_std)
        self.fc1_weight = nn.Parameter(torch.empty(hidden, in_dim))
        self.fc1_bias = nn.Parameter(torch.empty(hidden))
        self.fc2_weight = nn.Parameter(torch.empty(num_widths, hidden))
        self.fc2_bias = nn.Parameter(torch.empty(num_widths))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.fc1_weight, self.init_std)
        with torch.no_grad():
            self.fc1_bias.zero_()
            self.fc2_weight.zero_()
            self.fc2_bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, self.fc1_weight, self.fc1_bias)), self.fc2_weight, self.fc2_bias)


class ShellElasticLM(nn.Module):
    """ShellElastic language model (also StairFormer and non-elastic models, see module doc)."""

    def __init__(self, cfg: ShellConfig) -> None:
        super().__init__()
        self.cfg = cfg
        _check(cfg.norm, ("rms", "layer"), "norm")
        _check(cfg.trunk_norm, TRUNK_NORMS, "trunk_norm")
        _check(cfg.terminal_norm, TERMINAL_NORMS, "terminal_norm")
        _check(cfg.readout, READOUTS, "readout")
        _check(cfg.readout_scaling, READOUT_SCALINGS, "readout_scaling")
        _check(cfg.parameterization, PARAMETERIZATIONS, "parameterization")
        supported = [int(s) for s in cfg.supported]
        if supported != sorted(set(supported)) or supported[0] < 1:
            raise ValueError(f"supported widths must be strictly increasing positive integers, got {supported}")
        if not 0.0 <= cfg.stage_grad_scale <= 1.0:
            raise ValueError(f"stage_grad_scale must be in [0, 1], got {cfg.stage_grad_scale}")
        self.supported = supported
        self.M = supported[-1]
        self.g = cfg.base_width
        self.width = self.M * self.g
        trained = supported if cfg.trained is None else sorted(int(s) for s in cfg.trained)
        if not trained or any(s not in supported for s in trained):
            raise ValueError(f"trained widths {trained} must be a non-empty subset of {supported}")
        self.trained = trained

        # Scaling rules (Table 1).
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
        hidden_std = sigma / math.sqrt(self.m)
        down_std = hidden_std * math.sqrt(self.e)

        # Normalization over the trunk (group-local, block-local or cumulative); a non-elastic
        # model uses the plain full-width norm.
        if len(supported) == 1:
            self._trunk_scope, self._trunk_bounds = "full", None
        elif cfg.trunk_norm == "group":
            self._trunk_scope, self._trunk_bounds = "local", tuple(supported)
        elif cfg.trunk_norm == "block":
            self._trunk_scope, self._trunk_bounds = "local", None
        else:
            self._trunk_scope, self._trunk_bounds = "cumulative", None

        self.embed = Embedding(cfg.vocab_size, self.width, sigma)
        self.blocks = nn.ModuleList(
            ShellBlock(cfg, self._trunk_scope, self._trunk_bounds, self.internal, hidden_std, down_std, self.e)
            for _ in range(cfg.n_layers)
        )
        self.final_norm = Norm(self.width, cfg.norm, cfg.norm_eps)

        # Per-width heads exist only if more than one width's head is ever trained.
        multi = len(trained) > 1
        self.separate_readouts = cfg.readout == "separate" and multi
        self.separate_norms = cfg.separate_terminal_norms and multi
        # The shared terminal norm of the trunk form is applied once inside the trunk.
        self.norm_in_head = cfg.terminal_norm == "full" or self.separate_norms

        init_lr = cfg.readout_scaling == "init_lr"
        self.shared_split = init_lr and not self.separate_readouts and trained != [self.M]
        if self.shared_split:
            readout_std = self.readout_std(1)  # frozen init snapshot at the base std
        else:
            readout_std = self.readout_std(self.M) if init_lr else sigma
        self.readout = Linear(self.width, cfg.vocab_size, readout_std)
        self.readouts = nn.ModuleDict()
        self.final_norms = nn.ModuleDict()
        for s in trained:
            if s == self.M:
                continue
            if self.separate_readouts:
                self.readouts[str(s)] = Linear(s * self.g, cfg.vocab_size, self.readout_std(s) if init_lr else sigma)
            if self.separate_norms and self.norm_in_head:
                self.final_norms[str(s)] = Norm(s * self.g, cfg.norm, cfg.norm_eps)
        if self.shared_split:
            self.readout.weight.requires_grad_(False)
            self.readout_delta = nn.Parameter(torch.zeros(cfg.vocab_size, self.width))
        else:
            self.register_parameter("readout_delta", None)

        self.alpha_learnable = isinstance(alpha, str)
        if self.alpha_learnable:
            self.alpha = nn.Parameter(torch.ones(len(supported)))
            self.alpha_value = None
        else:
            self.register_parameter("alpha", None)
            self.alpha_value = float(alpha)

        self.width_router = (
            WidthRouter(self.width, cfg.width_router_hidden, len(supported), sigma) if cfg.width_router else None
        )
        self.set_fused_norm(cfg.fused_norm)
        self.set_activation_checkpointing(cfg.activation_checkpointing)

    # ------------------------------------------------------------------------------------------
    # Initialization and configuration helpers
    # ------------------------------------------------------------------------------------------

    def readout_std(self, s: int) -> float:
        """Readout init std under ``readout_scaling="init_lr"``: sigma_0 / sqrt(m * s)."""
        s_int = s if self.internal else 1
        return self.cfg.init_std / math.sqrt(self.m * s_int)

    @torch.no_grad()
    def init_weights(self) -> None:
        """(Re-)initialize every parameter; call after materializing a meta-device model."""
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        if self.readout_delta is not None:
            self.readout_delta.zero_()
        if self.alpha is not None:
            self.alpha.fill_(1.0)

    def set_fused_norm(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, Norm):
                module.use_fused = bool(enabled)

    def set_activation_checkpointing(self, fraction: str | float) -> None:
        for block, use in zip(self.blocks, checkpointed_blocks(fraction, len(self.blocks))):
            block.use_checkpoint = use

    # ------------------------------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------------------------------

    def _trunk_norm(self, norm: Norm, x: torch.Tensor, s: int) -> torch.Tensor:
        if self._trunk_scope == "full":
            return full_norm(norm, x, s * self.g)
        return segment_norm(norm, x, s * self.g, self.g, self._trunk_scope, self._trunk_bounds)

    def terminal_norm_module(self, s: int) -> Norm:
        key = str(s)
        if s != self.M and key in self.final_norms:
            return self.final_norms[key]
        return self.final_norm

    def _head_norm(self, norm: Norm, x: torch.Tensor, s: int) -> torch.Tensor:
        if self.cfg.terminal_norm == "trunk":
            return self._trunk_norm(norm, x, s)
        return full_norm(norm, x, s * self.g)

    # ------------------------------------------------------------------------------------------
    # Readout helpers
    # ------------------------------------------------------------------------------------------

    def readout_module(self, s: int) -> nn.Module:
        key = str(s)
        if s != self.M and key in self.readouts:
            return self.readouts[key]
        return self.readout

    def readout_weight(self, s: int) -> torch.Tensor:
        """U_s: the separate readout, a slice of the shared one, or the split reconstruction."""
        n = s * self.g
        key = str(s)
        if s != self.M and key in self.readouts:
            return self.readouts[key].weight
        if self.shared_split:
            s_int = s if self.internal else 1
            return s_int**-0.5 * self.readout.weight[:, :n] + (1.0 / s_int) * self.readout_delta[:, :n]
        return self.readout.weight[:, :n]

    def alpha_of(self, s: int):
        if self.alpha_learnable:
            return self.alpha[self.supported.index(s)]
        return self.alpha_value

    def readout_multiplier(self, s: int):
        """Forward multiplier of the logits of width ``s``."""
        if self.cfg.readout_scaling == "init_lr":
            return self.alpha_of(s)
        s_eff = s if self.internal else 1
        return self.alpha_of(s) / (s_eff * self.m)

    def logits(self, hidden: torch.Tensor, s: int) -> torch.Tensor:
        """Logits of width ``s`` from its post-norm hidden states ``[..., s*g]``."""
        return F.linear(hidden, self.readout_weight(s)) * self.readout_multiplier(s)

    def _readout_private(self, s: int) -> bool:
        mine = self.readout_module(s)
        return all(self.readout_module(t) is not mine for t in self.trained if t != s)

    def _norm_private(self, s: int) -> bool:
        mine = self.terminal_norm_module(s)
        return all(self.terminal_norm_module(t) is not mine for t in self.trained if t != s)

    def _stage_boundary(self, s: int) -> int:
        """Features of the largest trained width below ``s`` (0 for the smallest)."""
        below = [t for t in self.trained if t < s]
        return max(below) * self.g if below else 0

    # ------------------------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------------------------

    def trunk(self, input_ids: torch.Tensor, s: int) -> torch.Tensor:
        """Hidden states of width ``s`` after the last block (and the in-trunk terminal norm)."""
        return self._trunk(input_ids, s)[0]

    def _trunk(self, input_ids: torch.Tensor, s: int, attention_mask=None, past_key_values=None,
               use_cache: bool = False):
        if s not in self.supported:
            raise ValueError(f"width {s} is not a supported width {self.supported}")
        x = self.embed(input_ids)[..., : s * self.g]
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        bias = attention_bias(attention_mask, x, past_len)
        cache = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, kv = block(x, s, bias, past, use_cache)
            if use_cache:
                cache.append(kv)
        if not self.norm_in_head:
            x = self._trunk_norm(self.final_norm, x, s)
        return x, cache

    def infer(self, input_ids: torch.Tensor, width: int | None = None, attention_mask: torch.Tensor | None = None,
              past_key_values: list | None = None, use_cache: bool = False, logits_to_keep: int = 0):
        """Float32 logits of sub-network ``width`` (default: the largest) for evaluation and generation.

        The trunk runs at ``width`` itself. ``attention_mask`` (``[batch, total_len]``, 0 at
        padding) covers the cached and the new positions; ``past_key_values`` is the cache
        returned by the previous call; ``logits_to_keep > 0`` reads out only the last positions.
        Returns ``(logits, cache)`` with ``cache=None`` unless ``use_cache``.
        """
        s = self.M if width is None else int(width)
        x, cache = self._trunk(input_ids, s, attention_mask, past_key_values, use_cache)
        if logits_to_keep:
            x = x[:, -logits_to_keep:]
        hidden = self.head(x, s).hidden
        return F.linear(hidden, self.readout_weight(s) * self.readout_multiplier(s)).float(), cache

    def head(self, x: torch.Tensor, s: int, targets: torch.Tensor | None = None, loss_fn=None) -> WidthOutput:
        """Terminal norm, readout and (optionally) the fused loss of width ``s``."""
        n = s * self.g
        x = x[..., :n]
        x_pre = x
        if self.norm_in_head:
            x = self._head_norm(self.terminal_norm_module(s), x, s)
        loss = z_loss = None
        if loss_fn is not None:
            if targets is None:
                raise ValueError("targets are required when loss_fn is given")
            mult = self.readout_multiplier(s)
            weight = self.readout_weight(s)
            if not torch.is_grad_enabled():
                weight = weight.detach()
            ce_hidden, ce_weight = x, weight
            gamma = float(self.cfg.stage_grad_scale)
            if gamma != 1.0 and torch.is_grad_enabled():
                n_prev = self._stage_boundary(s)
                if 0 < n_prev < n:
                    prefix = grad_scale(x[..., :n_prev], gamma)
                    if self.norm_in_head and self._norm_private(s):
                        # Restore the full gradient of this width's private terminal-norm gain.
                        y = self._head_norm(self.terminal_norm_module(s), x_pre.detach(), s)[..., :n_prev]
                        prefix = prefix + grad_scale(y - y.detach(), 1.0 - gamma)
                    ce_hidden = torch.cat([prefix, x[..., n_prev:]], dim=-1)
                    if not self._readout_private(s):
                        ce_weight = torch.cat([grad_scale(weight[:, :n_prev], gamma), weight[:, n_prev:]], dim=-1)
            loss, z_loss = split_loss(loss_fn(ce_weight, (ce_hidden * mult).reshape(-1, n), targets.reshape(-1)))
        return WidthOutput(hidden=x, loss=loss, z_loss=z_loss)

    def forward(
        self,
        input_ids: torch.Tensor,
        widths: list[int] | None = None,
        targets: torch.Tensor | None = None,
        loss_fn=None,
        grad_widths: set[int] | None = None,
        trunk_grad_widths: set[int] | None = None,
    ) -> dict[int, WidthOutput]:
        """Single trunk pass at the widest requested width, then one head per width.

        ``grad_widths``: widths whose heads build the autograd graph (``None`` = all); the others
        are evaluated without gradient. ``trunk_grad_widths``: widths whose loss reaches the
        trunk (``None`` = all); the others see a detached trunk and train only their heads.
        """
        widths = list(self.supported) if widths is None else sorted(int(s) for s in widths)
        x = self.trunk(input_ids, max(widths))
        outputs: dict[int, WidthOutput] = {}
        for s in widths:
            want_grad = grad_widths is None or s in grad_widths
            x_s = x if (trunk_grad_widths is None or s in trunk_grad_widths) else x.detach()
            with nullcontext() if want_grad else torch.no_grad():
                outputs[s] = self.head(x_s, s, targets, loss_fn)
        return outputs

    def width_router_logits(self, hidden_max: torch.Tensor) -> torch.Tensor:
        """Router logits over the supported widths from the widest post-norm hidden states."""
        if self.width_router is None:
            raise RuntimeError("the width router is disabled")
        return self.width_router(hidden_max / self.m)

    # ------------------------------------------------------------------------------------------
    # Parameterization (Table 1)
    # ------------------------------------------------------------------------------------------

    def optim_spec(self) -> dict[str, tuple[float, bool]]:
        """Per-parameter ``(q, decay)``: learning rate ``eta_0 / q`` and weight decay ``lambda_0 * q``.

        Hidden weights of an output group with boundary s have q = s m; readouts under ``init_lr``
        scaling are treated the same way with their own width; the split delta of a shared readout
        carries only m (its width power is applied in the forward pass); the embedding keeps the
        base values; gains, biases and alpha are not decayed.
        """
        spec: dict[str, tuple[float, bool]] = {}
        init_lr = self.cfg.readout_scaling == "init_lr"

        def width_q(s: int) -> float:
            return (float(s) if self.internal else 1.0) * self.m

        for mod_name, module in self.named_modules():
            prefix = f"{mod_name}." if mod_name else ""
            if isinstance(module, ShellLinear):
                for k, b in enumerate(module.boundaries):
                    spec[f"{prefix}weight_{k}"] = (width_q(b), True)
        for s in self.trained:
            key = str(s)
            if s != self.M and key in self.readouts and init_lr:
                spec[f"readouts.{key}.weight"] = (width_q(s), True)
        if init_lr and not self.shared_split:
            spec["readout.weight"] = (width_q(self.M), True)
        if self.readout_delta is not None:
            spec["readout_delta"] = (self.m, True)
        for name, p in self.named_parameters():
            if name in spec or not p.requires_grad:
                continue
            spec[name] = (1.0, p.ndim > 1)
        return spec

    def num_params(self, s: int) -> int:
        """Parameters that sub-network ``s`` reads (Appendix E.3: parameter counts)."""
        n = s * self.g
        total = self.cfg.vocab_size * n  # embedding columns
        k_active = self.supported.index(s) + 1
        for module in self.modules():
            if isinstance(module, ShellLinear):
                total += sum(module.group_weight(k).numel() for k in range(k_active))
            elif isinstance(module, Norm) and module is not self.final_norm and module.dim == self.width:
                total += n if module.weight is not None else 0
        total += n  # terminal norm gain
        total += self.cfg.vocab_size * n  # readout
        return total
