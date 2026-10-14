"""Shared building blocks: normalization, rotary embeddings, attention and SwiGLU.

Normalization is applied functionally so that every elastic model can normalize a
*prefix* of the features, either over the whole prefix or independently within
groups of features. All statistics are computed in FP32.
"""

from __future__ import annotations

import math
from fractions import Fraction

import torch
import torch.nn.functional as F
from torch import nn

from elastic.kernels.group_rmsnorm import fused_group_rms_norm

try:  # Optional fused SwiGLU kernel (CUDA only).
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
except ImportError:  # pragma: no cover
    LigerSiLUMulFunction = None


# ---------------------------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------------------------


def trunc_normal_(weight: torch.Tensor, std: float) -> None:
    """Truncated normal initialization at +-3 standard deviations (zeros for std <= 0)."""
    with torch.no_grad():
        if std > 0:
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        else:
            weight.zero_()


class Linear(nn.Linear):
    """Bias-free linear layer with a fixed truncated-normal initialization std."""

    def __init__(self, in_features: int, out_features: int, init_std: float) -> None:
        self.init_std = float(init_std)
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, self.init_std)


class Embedding(nn.Embedding):
    """Token embedding with a truncated-normal initialization."""

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float) -> None:
        self.init_std = float(init_std)
        super().__init__(num_embeddings, embedding_dim)

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, self.init_std)


# ---------------------------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------------------------

NORM_KINDS = ("rms", "layer")


class Norm(nn.Module):
    """Parameter container of an RMSNorm (``kind="rms"``) or a LayerNorm (``kind="layer"``).

    The normalization itself is applied by :func:`full_norm`, :func:`segment_norm` or
    :func:`head_norm`, which read ``weight``/``bias`` sliced to the active width.
    """

    def __init__(self, dim: int, kind: str = "rms", eps: float = 1e-5, affine: bool = True,
                 bias: bool = False) -> None:
        super().__init__()
        if kind not in NORM_KINDS:
            raise ValueError(f"norm kind must be one of {NORM_KINDS}, got {kind!r}")
        self.dim = int(dim)
        self.kind = kind
        self.eps = float(eps)
        self.use_fused = True
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.weight is not None:
                self.weight.fill_(1.0)
            if self.bias is not None:
                self.bias.zero_()

    def extra_repr(self) -> str:
        return f"{self.dim}, kind={self.kind}, eps={self.eps}"


def _affine(norm: Norm, n: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    w = norm.weight[:n] if norm.weight is not None else None
    b = norm.bias[:n] if norm.bias is not None else None
    return w, b


def _upcast(x: torch.Tensor) -> torch.Tensor:
    """At least float32 for normalization statistics (float64 stays float64)."""
    return x if x.dtype == torch.float64 else x.float()


def full_norm(norm: Norm, x: torch.Tensor, n: int) -> torch.Tensor:
    """Normalize over all ``n`` features of ``x`` (the first ``n`` gains are used)."""
    w, b = _affine(norm, n)
    if norm.kind == "rms":
        dtype = x.dtype
        xf = _upcast(x)
        out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.eps)).to(dtype)
        if w is not None:
            out = out * w
        if b is not None:
            out = out + b
        return out
    return F.layer_norm(x, (n,), w, b, norm.eps)


def grouped_norm(norm: Norm, x: torch.Tensor, n: int, group: int) -> torch.Tensor:
    """Normalize every ``group`` consecutive features of ``x[..., :n]`` (e.g. per-head q/k norm).

    This is the plain (non-elastic) formulation, used by models with a single supported width.
    """
    w, b = _affine(norm, n)
    lead = x.shape[:-1]
    if norm.kind == "rms":
        dtype = x.dtype
        xg = _upcast(x).reshape(*lead, n // group, group)
        out = (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + norm.eps)).reshape(*lead, n).to(dtype)
    else:
        out = F.layer_norm(x.reshape(*lead, n // group, group), (group,), None, None, norm.eps).reshape(*lead, n)
    if w is not None:
        out = out * w
    if b is not None:
        out = out + b
    return out


SEGMENT_SCOPES = ("local", "cumulative")


def segment_norm(
    norm: Norm,
    x: torch.Tensor,
    n: int,
    unit: int,
    scope: str = "local",
    boundaries: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Normalize the first ``n`` features of ``x`` within segments.

    Segments are given by cumulative ``boundaries`` in multiples of ``unit`` (``None`` means
    every block of ``unit`` features is a segment). With ``scope="local"`` each segment is
    normalized with its own statistics; with ``scope="cumulative"`` each segment uses the
    statistics of all features up to its right boundary. Both preserve prefixes that end at
    a segment boundary.
    """
    if scope not in SEGMENT_SCOPES:
        raise ValueError(f"segment scope must be one of {SEGMENT_SCOPES}, got {scope!r}")
    if n % unit != 0:
        raise ValueError(f"width {n} is not divisible by the segment unit {unit}")
    w, b = _affine(norm, n)
    eps = norm.eps
    units = n // unit
    if boundaries is None:
        active = list(range(1, units + 1))
    else:
        active = [int(c) for c in boundaries if int(c) <= units]
        if not active or active[-1] != units:
            raise ValueError(f"width {n} ({units} units) is not a segment boundary of {list(boundaries)}")
    if scope == "local" and norm.kind == "rms" and norm.use_fused:
        fused = fused_group_rms_norm(x, w, b, eps, tuple(c * unit for c in active))
        if fused is not None:
            return fused
    lead = list(x.shape[:-1])
    xf = _upcast(x)
    if not all(active[i] == i + 1 for i in range(len(active))):
        return _segment_norm_variable(x, xf, lead, active, unit, scope, eps, w, b, norm.kind)
    xr = xf.reshape(*lead, units, unit)
    seg_sumsq = xr.pow(2).sum(dim=-1, keepdim=True)
    seg_sum = xr.sum(dim=-1, keepdim=True)
    if scope == "cumulative":
        agg_sumsq = torch.cumsum(seg_sumsq, dim=-2)
        agg_sum = torch.cumsum(seg_sum, dim=-2)
        counts = torch.arange(1, units + 1, device=x.device, dtype=xf.dtype) * unit
    else:
        agg_sumsq = seg_sumsq
        agg_sum = seg_sum
        counts = torch.full((units,), float(unit), device=x.device, dtype=xf.dtype)
    counts = counts.reshape(*([1] * len(lead)), units, 1)
    if norm.kind == "rms":
        out = xr * torch.rsqrt(agg_sumsq / counts + eps)
    else:
        mean = agg_sum / counts
        var = agg_sumsq / counts - mean.pow(2)
        out = (xr - mean) * torch.rsqrt(var + eps)
    out = out.reshape(*lead, n).to(x.dtype)
    if w is not None:
        out = out * w
    if b is not None:
        out = out + b
    return out


def _segment_norm_variable(x, xf, lead, active, unit, scope, eps, w, b, kind):
    """Variable-width segments; the output is assembled segment by segment."""
    seg_sumsq, seg_sum, widths = [], [], []
    prev = 0
    for c in active:
        hi = c * unit
        seg = xf[..., prev:hi]
        seg_sumsq.append(seg.pow(2).sum(dim=-1))
        seg_sum.append(seg.sum(dim=-1))
        widths.append(hi - prev)
        prev = hi
    group_sumsq = torch.stack(seg_sumsq, dim=-1)
    group_sum = torch.stack(seg_sum, dim=-1)
    num = len(active)
    if scope == "cumulative":
        agg_sumsq = torch.cumsum(group_sumsq, dim=-1)
        agg_sum = torch.cumsum(group_sum, dim=-1)
        counts = torch.tensor([c * unit for c in active], device=x.device, dtype=xf.dtype)
    else:
        agg_sumsq = group_sumsq
        agg_sum = group_sum
        counts = torch.tensor([float(width) for width in widths], device=x.device, dtype=xf.dtype)
    counts = counts.reshape(*([1] * len(lead)), num)
    if kind == "rms":
        mean = None
        inv = torch.rsqrt(agg_sumsq / counts + eps)
    else:
        mean = agg_sum / counts
        inv = torch.rsqrt(agg_sumsq / counts - mean.pow(2) + eps)
    parts = []
    prev = 0
    for i, c in enumerate(active):
        hi = c * unit
        seg = xf[..., prev:hi]
        inv_i = inv[..., i : i + 1]
        parts.append(seg * inv_i if mean is None else (seg - mean[..., i : i + 1]) * inv_i)
        prev = hi
    out = torch.cat(parts, dim=-1).to(x.dtype)
    if w is not None:
        out = out * w
    if b is not None:
        out = out + b
    return out


def head_norm(norm: Norm, x: torch.Tensor, n: int, head_dim: int) -> torch.Tensor:
    """Per-head normalization of queries or keys.

    ``x`` has shape ``[..., units, unit_width]`` and may be a strided view into a fused QKV
    output; the result has the same logical shape. Each ``head_dim`` block of the ``n``
    features is normalized with its own statistics.
    """
    if norm.kind == "rms" and norm.use_fused:
        w, b = _affine(norm, n)
        fused = fused_group_rms_norm(
            x, w, b, norm.eps, tuple(range(head_dim, n + 1, head_dim)), unit_width=x.shape[-1]
        )
        if fused is not None:
            return fused
    flat = x.reshape(*x.shape[:-2], n)
    return segment_norm(norm, flat, n, head_dim, "local", None).reshape(x.shape)


# ---------------------------------------------------------------------------------------------
# Rotary position embedding
# ---------------------------------------------------------------------------------------------
# Adapted from the rotary embedding of xFormers (BSD-3-Clause), as used by OpenLM (MIT):
# https://github.com/facebookresearch/xformers/blob/748c159096d4f9fcfe3eaf22801e5aed4777210b/xformers/components/positional_embedding/rotary.py


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rotary(nn.Module):
    """Rotary embedding (base 10,000) for ``[batch, seq, heads, head_dim]`` tensors.

    The cos/sin tables are built in the dtype of the keys and the rotated queries and keys
    are cast to the dtype of the values, as in OpenLM.
    """

    def __init__(self, head_dim: int, seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.seq_len = int(seq_len)
        self.base = float(base)
        self.register_buffer("inv_freq", torch.zeros(self.head_dim // 2), persistent=False)
        self._cos = None
        self._sin = None
        self._cached_len = 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Computed on CPU in FP32 (as in the reference) and then moved next to the module.
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device="cpu").float() / self.head_dim))
        device = self.inv_freq.device
        self.inv_freq = inv_freq if device.type == "meta" else inv_freq.to(device)
        self._cos = None
        self._sin = None
        self._cached_len = 0

    def _tables(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        seq_len = max(seq_len, self._cached_len, self.seq_len)
        if (
            self._cos is None
            or seq_len > self._cached_len
            or self._cos.device != device
            or self._cos.dtype != dtype
        ):
            self._cached_len = seq_len
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device, dtype=dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            self._cos = emb.cos()[None, :, None, :].to(dtype)
            self._sin = emb.sin()[None, :, None, :].to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, offset: int = 0):
        self._tables(k.shape[1] + offset, k.device, k.dtype)
        cos = self._cos[:, offset : offset + q.shape[1]]
        sin = self._sin[:, offset : offset + q.shape[1]]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q.to(v.dtype), k.to(v.dtype), v


# ---------------------------------------------------------------------------------------------
# Attention and SwiGLU
# ---------------------------------------------------------------------------------------------


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     bias: torch.Tensor | None = None) -> torch.Tensor:
    """Causal scaled dot-product attention on ``[batch, seq, heads, head_dim]`` tensors.

    ``bias`` is the additive mask of :func:`attention_bias`; without it, queries attend causally
    (or, for a single query, to every key).
    """
    scale = q.shape[-1] ** -0.5
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if bias is None:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] > 1, scale=scale)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), scale=scale)
    return out.transpose(1, 2).contiguous()


def attention_bias(attention_mask: torch.Tensor | None, x: torch.Tensor, past_len: int = 0) -> torch.Tensor | None:
    """Additive attention mask ``[batch, 1, q_len, k_len]`` for the hidden states ``x`` of a trunk.

    It is built once per forward pass and shared by every layer. Queries are aligned with the end
    of the keys (``past_len`` cached positions come first). ``attention_mask`` (``[batch, k_len]``,
    0 at padding) masks padded keys; queries whose keys are all masked attend to everything instead
    of producing NaNs. Returns ``None`` when plain causal attention suffices.
    """
    q_len = x.shape[1]
    k_len = past_len + q_len
    if attention_mask is None and not k_len > q_len > 1:
        return None
    device = x.device
    dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else x.dtype
    batch = 1 if attention_mask is None else attention_mask.shape[0]
    bias = rectangular_causal_mask((batch, 1), q_len, k_len, device, dtype)
    if attention_mask is not None:
        apply_padding_mask_(bias, attention_mask, dtype)
    return bias


def rectangular_causal_mask(shape: tuple[int, int], q_len: int, k_len: int, device, dtype) -> torch.Tensor:
    """Additive causal mask ``[*shape, q_len, k_len]`` with the queries at the end of the keys."""
    padded = (k_len + 7) // 8 * 8
    allowed = torch.ones((q_len, k_len), device=device, dtype=torch.bool)
    allowed[:, -q_len:] = torch.tril(allowed[:, -q_len:], diagonal=0)
    mask = torch.zeros((*shape, q_len, padded), device=device, dtype=dtype)
    mask[:, :, :, :k_len].masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[:, :, :, :k_len]


def apply_padding_mask_(bias: torch.Tensor, attention_mask: torch.Tensor, dtype: torch.dtype) -> None:
    """Mask padded keys in ``bias`` in place (rows that are fully masked are unmasked)."""
    length = attention_mask.shape[-1]
    padding = bias[..., :length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
    min_value = torch.finfo(dtype).min
    bias[..., :length] = bias[..., :length].masked_fill(padding, min_value)
    bias.mul_(~torch.all(bias == min_value, dim=-1, keepdim=True))


def swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``silu(a) * b`` with the activation computed in FP32."""
    if LigerSiLUMulFunction is not None and a.is_cuda:
        return LigerSiLUMulFunction.apply(a, b)
    return (F.silu(_upcast(a)) * b).to(a.dtype)


def depth_multiplier(n_layers: int, scale: float = 1.0, reference_depth: int = 32) -> float:
    """Residual-branch multiplier ``scale * reference_depth / n_layers`` (1 at 32 layers)."""
    return float(scale) * reference_depth / n_layers


def checkpointed_blocks(fraction: str | float, n_layers: int) -> list[bool]:
    """Blocks with activation checkpointing: ``fraction`` of them, evenly spaced ("1/2": 0, 2, 4, ...)."""
    try:
        p = Fraction(str(fraction))
    except (ValueError, ZeroDivisionError):
        p = None
    if p is None or not 0 <= p <= 1:
        raise ValueError("activation_checkpointing must be the fraction of the blocks to checkpoint, from 0 to 1 "
                         f"(e.g. 0/1, 1/8, 1/2, 1/1), got {fraction!r}")
    half = Fraction(1, 2)
    return [math.floor((i + 1) * p + half) > math.floor(i * p + half) for i in range(n_layers)]


def check_divisible(a: int, b: int, what: str) -> int:
    if a % b != 0:
        raise ValueError(f"{what}: {a} is not divisible by {b}")
    return a // b


__all__ = [
    "Embedding",
    "Linear",
    "Norm",
    "Rotary",
    "attention_bias",
    "causal_attention",
    "check_divisible",
    "checkpointed_blocks",
    "depth_multiplier",
    "full_norm",
    "head_norm",
    "segment_norm",
    "swiglu",
    "trunc_normal_",
]
