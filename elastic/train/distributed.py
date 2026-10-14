"""Process groups, device meshes and FSDP2 sharding."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn


def setup() -> tuple[int, int, torch.device]:
    """Initialize the default process group from torchrun's environment (or a single process)."""
    if "RANK" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, timeout=timedelta(minutes=30))
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world, device


def teardown() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def build_mesh(world: int, shard_size: int, device: torch.device):
    """1-D mesh (full sharding) or a ``(replicate, shard)`` mesh for hybrid sharding."""
    from torch.distributed.device_mesh import init_device_mesh

    if not dist.is_initialized():
        # FSDP2 needs a process group even on one device.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", rank=0, world_size=1)
    if shard_size <= 0 or shard_size >= world:
        return init_device_mesh(device.type, (world,), mesh_dim_names=("shard",))
    if world % shard_size != 0:
        raise ValueError(f"world size {world} is not a multiple of shard_size {shard_size}")
    return init_device_mesh(device.type, (world // shard_size, shard_size), mesh_dim_names=("replicate", "shard"))


def shard_model(model: nn.Module, mesh, param_dtype: torch.dtype = torch.bfloat16,
                reduce_dtype: torch.dtype = torch.float32) -> nn.Module:
    """FSDP2: every transformer block is its own unit, the rest belongs to the root unit."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method

    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    if hasattr(model, "infer"):
        register_fsdp_forward_method(model, "infer")
    return model


def all_reduce_mean(values: dict[str, float | torch.Tensor], device: torch.device) -> dict[str, float]:
    """Mean over ranks of every scalar metric (same keys on every rank)."""
    keys = sorted(values)
    if not keys:
        return {}
    vec = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=device)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(vec)
        vec /= dist.get_world_size()
    return dict(zip(keys, vec.tolist()))
