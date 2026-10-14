"""Slab GEMMs for block-lower-triangular (shell) projections.

A shell projection stores one weight per output group, ``W_k`` of shape
``[rows_k, read_k]`` with non-decreasing read widths. The "pad" path assembles the full
block-lower-triangular matrix and runs one GEMM; this module instead packs consecutive groups
into at most three *slabs* (the core, then the shells in one or two slabs), zero-pads each
group only up to its slab's read width, and runs one GEMM per slab. Output tiles and
input-gradient tiles have disjoint destinations, so no scatter/add is needed in backward.

Slabs are materialized only for the duration of a forward/backward and never cached. The result
is numerically equivalent to "pad".
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


@lru_cache(maxsize=64)
def slab_partition(rows: tuple[int, ...], reads: tuple[int, ...], out_unit: int, unit_rows: tuple[int, ...]):
    """Partition the groups into slabs: the core alone, then the shells in one or two slabs.

    ``unit_rows[k]`` is the number of base units in group ``k``. The shells are split in two
    (at the group boundary closest to their unit midpoint) only when the per-unit output
    width exceeds 128 and there are at least two shell units.
    """
    k_total = len(rows)
    if k_total == 1:
        return ((0, 1),)
    tail_units = sum(unit_rows[1:])
    if out_unit <= 128 or tail_units < 2 or k_total < 3:
        return ((0, 1), (1, k_total))
    target = tail_units // 2
    best, best_err, acc = 2, None, 0
    for k in range(1, k_total - 1):
        acc += unit_rows[k]
        err = abs(acc - target)
        if best_err is None or err < best_err:
            best, best_err = k + 1, err
    return ((0, 1), (1, best), (best, k_total))


class _SlabLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, slabs: tuple[tuple[int, int], ...], *weights: torch.Tensor):
        reads = tuple(w.shape[1] for w in weights)
        rows = tuple(w.shape[0] for w in weights)
        offsets = [0]
        for r in rows:
            offsets.append(offsets[-1] + r)
        packed = []
        for lo, hi in slabs:
            width = reads[hi - 1]
            parts = [F.pad(weights[i], (0, width - reads[i])) if reads[i] != width else weights[i] for i in range(lo, hi)]
            packed.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))
        xx = x.reshape(-1, x.shape[-1])
        y = x.new_empty((xx.shape[0], offsets[-1]))
        for (lo, hi), weight in zip(slabs, packed):
            torch.mm(xx[:, : weight.shape[1]], weight.T, out=y[:, offsets[lo] : offsets[hi]])
        ctx.slabs = slabs
        ctx.reads = reads
        ctx.offsets = tuple(offsets)
        ctx.input_shape = x.shape
        ctx.save_input = any(ctx.needs_input_grad[2:])
        ctx.save_packed = ctx.needs_input_grad[0]
        ctx.save_for_backward(*([x] if ctx.save_input else []), *(packed if ctx.save_packed else []))
        return y.reshape(*x.shape[:-1], offsets[-1])

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_y: torch.Tensor):
        tensors = ctx.saved_tensors
        x = tensors[0] if ctx.save_input else None
        packed = tensors[int(ctx.save_input) :] if ctx.save_packed else ()
        offsets = ctx.offsets
        reads = ctx.reads
        dy = grad_y.reshape(-1, offsets[-1])
        dx = None
        grads: list[torch.Tensor | None] = [None] * len(reads)
        with torch.autocast(device_type=grad_y.device.type, enabled=False):
            if ctx.needs_input_grad[0]:
                dx = grad_y.new_empty((dy.shape[0], ctx.input_shape[-1]))
                previous = 0
                for j, ((lo, _hi), weight) in enumerate(zip(ctx.slabs, packed)):
                    end = weight.shape[1]
                    if end == previous:
                        continue
                    columns = [other[:, previous:end] for other in packed[j:]]
                    wt = columns[0] if len(columns) == 1 else torch.cat(columns, dim=0)
                    torch.mm(dy[:, offsets[lo] :], wt, out=dx[:, previous:end])
                    previous = end
                if previous < dx.shape[1]:
                    dx[:, previous:].zero_()
                dx = dx.reshape(ctx.input_shape)
            if x is not None:
                xx = x.reshape(-1, x.shape[-1])
                for lo, hi in ctx.slabs:
                    if not any(ctx.needs_input_grad[2 + lo : 2 + hi]):
                        continue
                    dw = torch.mm(dy[:, offsets[lo] : offsets[hi]].T, xx[:, : reads[hi - 1]])
                    for i in range(lo, hi):
                        if ctx.needs_input_grad[2 + i]:
                            r0 = offsets[i] - offsets[lo]
                            grads[i] = dw[r0 : r0 + (offsets[i + 1] - offsets[i]), : reads[i]]
        return (dx, None, *grads)


def slab_linear(x: torch.Tensor, weights: tuple[torch.Tensor, ...], slabs: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Apply a shell projection with slab GEMMs (weights are the active groups, in order)."""
    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
    else:
        dtype = x.dtype
    with torch.autocast(device_type=device_type, enabled=False):
        return _SlabLinear.apply(x.to(dtype), slabs, *(w.to(dtype) for w in weights))


# Slab GEMMs are used only on A100 with BF16 operands, 32768 token rows and the eight-unit staircase whose
# units read 512 (the four core units), 640, 768, 896 and 1024 features; every other call uses "pad".
SLAB_TOKENS = 32768
SLAB_UNIT_READS = (512, 512, 512, 512, 640, 768, 896, 1024)
SLAB_UNIT_OUT = (128, 384)


@lru_cache(maxsize=16)
def _is_a100(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (8, 0) and "A100" in torch.cuda.get_device_name(device)


def slab_supported(x: torch.Tensor, weights: tuple[torch.Tensor, ...], unit_reads: tuple[int, ...],
                   unit_out: int) -> bool:
    """Whether the slab path applies (``unit_reads``: read width of every active base unit)."""
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[-1] != SLAB_UNIT_READS[-1]
        or x.numel() // x.shape[-1] != SLAB_TOKENS
        or unit_out not in SLAB_UNIT_OUT
        or tuple(unit_reads) != SLAB_UNIT_READS
    ):
        return False
    autocast = torch.is_autocast_enabled(x.device.type)
    compute = torch.get_autocast_dtype(x.device.type) if autocast else x.dtype
    if compute != torch.bfloat16 or any(w.dtype != torch.bfloat16 for w in weights):
        return False
    return _is_a100(x.device)
