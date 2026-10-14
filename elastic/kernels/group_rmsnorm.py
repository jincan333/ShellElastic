"""Fused group-local RMS normalization (Triton), including strided per-unit Q/K views.

Each segment of the feature axis (given by cumulative ``boundaries``) is normalized with its
own statistics in FP32 and cast to the input dtype *before* the affine transform, matching
:func:`elastic.models.layers.segment_norm`. Unsupported layouts or dtypes return ``None`` so
callers fall back to the PyTorch implementation. Backward supports first derivatives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch.autograd.function import once_differentiable

if TYPE_CHECKING:
    triton = cast(Any, None)
    tl = cast(Any, None)
else:
    try:
        import triton
        import triton.language as tl
    except ImportError:  # pragma: no cover
        triton = None
        tl = None


if triton is not None and tl is not None:

    @triton.jit
    def _rms_forward(
        X, W, B, Y, Inv,  # noqa: ANN001
        ROWS: tl.constexpr, WIDTH: tl.constexpr, ROW_STRIDE: tl.constexpr,
        UNIT_WIDTH: tl.constexpr, UNIT_STRIDE: tl.constexpr,
        BOUNDARIES: tl.constexpr, BLOCKS: tl.constexpr, EPS: tl.constexpr,
        HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr,
        MUL_DTYPE: tl.constexpr, BLOCK_ROWS: tl.constexpr,
    ) -> None:
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        for segment in tl.static_range(len(BOUNDARIES)):
            start = 0 if segment == 0 else BOUNDARIES[segment - 1]
            end = BOUNDARIES[segment]
            size = end - start
            cols = start + tl.arange(0, BLOCKS[segment])
            offsets = rows[:, None] * ROW_STRIDE + (cols // UNIT_WIDTH)[None, :] * UNIT_STRIDE
            offsets += (cols % UNIT_WIDTH)[None, :]
            mask = (rows[:, None] < ROWS) & (cols[None, :] < end)
            values = tl.load(X + offsets, mask, other=0).to(tl.float32)
            inv = tl.rsqrt(tl.sum(values * values, axis=1) / size + EPS)
            normalized = (values * inv[:, None]).to(X.dtype.element_ty)
            result = normalized.to(MUL_DTYPE)
            if HAS_WEIGHT:
                weight = tl.load(W + cols, cols < end, other=0).to(MUL_DTYPE)
                result = (result * weight[None, :]).to(MUL_DTYPE)
            if HAS_BIAS:
                bias = tl.load(B + cols, cols < end, other=0).to(Y.dtype.element_ty)
                result = result.to(Y.dtype.element_ty) + bias[None, :]
            tl.store(Y + rows[:, None] * WIDTH + cols[None, :], result, mask)
            tl.store(Inv + rows * len(BOUNDARIES) + segment, inv, rows < ROWS)

    @triton.jit
    def _rms_backward(
        X, W, DY, Inv, DX, DW, DB,  # noqa: ANN001
        ROWS: tl.constexpr, WIDTH: tl.constexpr, ROW_STRIDE: tl.constexpr,
        UNIT_WIDTH: tl.constexpr, UNIT_STRIDE: tl.constexpr,
        DY_ROW_STRIDE: tl.constexpr, DY_UNIT_STRIDE: tl.constexpr,
        BOUNDARIES: tl.constexpr, BLOCKS: tl.constexpr, HAS_WEIGHT: tl.constexpr,
        NEED_DW: tl.constexpr, NEED_DB: tl.constexpr,
        MUL_DTYPE: tl.constexpr, BLOCK_ROWS: tl.constexpr,
    ) -> None:
        group = tl.program_id(0)
        rows = group * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        for segment in tl.static_range(len(BOUNDARIES)):
            start = 0 if segment == 0 else BOUNDARIES[segment - 1]
            end = BOUNDARIES[segment]
            size = end - start
            cols = start + tl.arange(0, BLOCKS[segment])
            offsets = rows[:, None] * ROW_STRIDE + (cols // UNIT_WIDTH)[None, :] * UNIT_STRIDE
            offsets += (cols % UNIT_WIDTH)[None, :]
            dy_offsets = rows[:, None] * DY_ROW_STRIDE + (cols // UNIT_WIDTH)[None, :] * DY_UNIT_STRIDE
            dy_offsets += (cols % UNIT_WIDTH)[None, :]
            mask = (rows[:, None] < ROWS) & (cols[None, :] < end)
            values = tl.load(X + offsets, mask, other=0).to(tl.float32)
            inv = tl.load(Inv + rows * len(BOUNDARIES) + segment, rows < ROWS, other=0)
            dy = tl.load(DY + dy_offsets, mask, other=0)
            # The backward of the affine product first casts to the multiplication dtype.
            dy_mul = dy.to(MUL_DTYPE)
            if HAS_WEIGHT:
                weight = tl.load(W + cols, cols < end, other=0).to(MUL_DTYPE)
                dnorm = (dy_mul * weight[None, :]).to(X.dtype.element_ty).to(tl.float32)
            else:
                dnorm = dy_mul.to(X.dtype.element_ty).to(tl.float32)
            inner = tl.sum(dnorm * values, axis=1)
            dx = dnorm * inv[:, None] - values * (inner * inv * inv * inv / size)[:, None]
            tl.store(DX + rows[:, None] * WIDTH + cols[None, :], dx, mask)
            if NEED_DW:
                normalized = (values * inv[:, None]).to(X.dtype.element_ty).to(MUL_DTYPE)
                # Low-precision elementwise products round before the reduction.
                product = (dy_mul * normalized).to(MUL_DTYPE).to(tl.float32)
                partial = tl.sum(tl.where(mask, product, 0.0), axis=0)
                tl.store(DW + group * WIDTH + cols, partial, cols < end)
            if NEED_DB:
                partial_bias = tl.sum(tl.where(mask, dy.to(tl.float32), 0.0), axis=0)
                tl.store(DB + group * WIDTH + cols, partial_bias, cols < end)


def _layout(x: torch.Tensor, unit_width: int | None) -> tuple[int, int, int, int, int] | None:
    """Return (rows, width, row stride, unit width, unit stride) or ``None`` if unsupported."""
    tail = 1 if unit_width is None else 2
    if x.ndim < tail + 1 or x.stride(-1) != 1:
        return None
    if unit_width is None:
        width = x.shape[-1]
        unit_width = width
        unit_stride = width
    else:
        if x.shape[-1] != unit_width:
            return None
        width = x.shape[-2] * unit_width
        unit_stride = x.stride(-2)
    row_stride = x.stride(-tail - 1)
    for index in range(x.ndim - tail - 2, -1, -1):
        if x.shape[index] > 1 and x.stride(index) != x.stride(index + 1) * x.shape[index + 1]:
            return None
    return x.numel() // width, width, row_stride, unit_width, unit_stride


def _tl_dtype(dtype: torch.dtype) -> Any:
    return {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[dtype]


class _GroupRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps, boundaries, layout):  # noqa: ANN001
        rows, width, row_stride, unit_width, unit_stride = layout
        mul_dtype = x.dtype if weight is None else torch.promote_types(x.dtype, weight.dtype)
        out_dtype = mul_dtype if bias is None else torch.promote_types(mul_dtype, bias.dtype)
        output = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        inv = torch.empty((rows, len(boundaries)), dtype=torch.float32, device=x.device)
        blocks = tuple(
            triton.next_power_of_2(end - start) for start, end in zip((0, *boundaries[:-1]), boundaries)
        )
        block_rows = min(16, max(1, 8192 // max(blocks)))
        _rms_forward[(triton.cdiv(rows, block_rows),)](
            x, weight, bias, output, inv, rows, width, row_stride, unit_width, unit_stride,
            boundaries, blocks, eps, weight is not None, bias is not None, _tl_dtype(mul_dtype), block_rows,
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, weight, inv)
        ctx.layout = layout
        ctx.boundaries = boundaries
        ctx.mul_dtype = mul_dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.blocks = blocks
        ctx.block_rows = block_rows
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):  # noqa: ANN001
        x, weight, inv = ctx.saved_tensors
        rows, width, row_stride, unit_width, unit_stride = ctx.layout
        dy_layout = _layout(dy, None if dy.ndim == x.ndim and x.shape[-1] == width else unit_width)
        if dy_layout is None:
            dy = dy.contiguous()
            dy_layout = _layout(dy, None if x.shape[-1] == width else unit_width)
        assert dy_layout is not None
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        block_rows = ctx.block_rows
        groups = triton.cdiv(rows, block_rows)
        need_dw, need_db = ctx.needs_input_grad[1:3]
        dw_partial = torch.empty((groups, width), dtype=torch.float32, device=x.device) if need_dw else None
        db_partial = torch.empty((groups, width), dtype=torch.float32, device=x.device) if need_db else None
        _rms_backward[(groups,)](
            x, weight, dy, inv, dx, dw_partial, db_partial, rows, width, row_stride,
            unit_width, unit_stride, dy_layout[2], dy_layout[4], ctx.boundaries,
            ctx.blocks, weight is not None,
            need_dw, need_db, _tl_dtype(ctx.mul_dtype), block_rows,
            num_warps=4, enable_fp_fusion=False,
        )
        dw = dw_partial.sum(dim=0).to(weight.dtype) if need_dw else None
        db = db_partial.sum(dim=0).to(ctx.bias_dtype) if need_db else None
        return dx, dw, db, None, None, None


def fused_group_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    boundaries: tuple[int, ...] | list[int],
    *,
    unit_width: int | None = None,
) -> torch.Tensor | None:
    """Normalize independent segments of the last axis, or return ``None`` if unsupported.

    ``boundaries`` are cumulative feature offsets, e.g. ``(512, 640, 768, 896, 1024)``. With
    ``unit_width=None`` the input is ``[..., width]``; otherwise it is
    ``[..., units, unit_width]`` and its unit stride may skip over other data (e.g. the keys
    and values of a fused QKV projection). The result keeps the logical shape and is contiguous.
    """
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if (
        triton is None
        or type(x) not in (torch.Tensor, torch.nn.Parameter)
        or not x.is_cuda
        or x.dtype not in supported
        or x.numel() == 0
    ):
        return None
    layout = _layout(x, unit_width)
    if layout is None:
        return None
    boundaries = tuple(int(end) for end in boundaries)
    width = layout[1]
    if not boundaries or boundaries[-1] != width:
        return None
    starts = (0, *boundaries[:-1])
    if len(boundaries) > 32 or any(end <= start or end - start > 8192 for start, end in zip(starts, boundaries)):
        return None
    for affine in (weight, bias):
        if affine is not None and (
            type(affine) not in (torch.Tensor, torch.nn.Parameter)
            or affine.device != x.device
            or affine.dtype not in supported
            or affine.shape != (width,)
            or not affine.is_contiguous()
        ):
            return None
    return _GroupRMSNorm.apply(x, weight, bias, eps, boundaries, layout)
