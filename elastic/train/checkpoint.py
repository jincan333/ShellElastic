"""Distributed checkpoints (resumable training state) and consolidated weight export."""

from __future__ import annotations

import json
import os
import shutil

import torch
import torch.distributed as dist
from torch import nn

from elastic.train.distributed import is_main

LATEST = "latest"


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def initialize_adamw_state(optimizer: torch.optim.Optimizer) -> None:
    """Allocate AdamW moments at step zero (without an optimizer step).

    Reading the state dict of an optimizer without state would otherwise take a dummy step and
    start its step count at one; this matters for models that are not updated before a save.
    """
    for group in optimizer.param_groups:
        on_device = group.get("fused") or group.get("capturable")
        for p in group["params"]:
            if p in optimizer.state and optimizer.state[p]:
                continue
            step = (torch.zeros((), dtype=torch.float32, device=p.device) if on_device
                    else torch.tensor(0.0, dtype=torch.float32))
            state = {"step": step, "exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
                     "exp_avg_sq": torch.zeros_like(p, memory_format=torch.preserve_format)}
            if group.get("amsgrad"):
                state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            optimizer.state[p] = state


class Checkpointer:
    """``<output_dir>/checkpoints/step_XXXXXXXX`` directories written with torch.distributed.checkpoint."""

    def __init__(self, output_dir: str, keep_every: int = 0) -> None:
        self.root = os.path.join(output_dir, "checkpoints")
        self.keep_every = int(keep_every)
        if is_main():
            os.makedirs(self.root, exist_ok=True)

    def path(self, step: int) -> str:
        return os.path.join(self.root, f"step_{step:08d}")

    def latest(self) -> str | None:
        marker = os.path.join(self.root, LATEST)
        if not os.path.exists(marker):
            return None
        with open(marker) as f:
            name = f.read().strip()
        path = os.path.join(self.root, name)
        return path if os.path.isdir(path) else None

    def save(self, step: int, models: list[nn.Module], optimizers: list[torch.optim.Optimizer], extra: dict) -> str:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict

        path = self.path(step)
        state = {}
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            model_sd, optim_sd = get_state_dict(model, optimizer)
            state[f"model_{i}"] = model_sd
            state[f"optim_{i}"] = optim_sd
        dcp.save(state, checkpoint_id=path)
        if is_main():
            torch.save(extra, os.path.join(path, "extra.pt"))
            with open(os.path.join(self.root, LATEST + ".tmp"), "w") as f:
                f.write(os.path.basename(path))
            os.replace(os.path.join(self.root, LATEST + ".tmp"), os.path.join(self.root, LATEST))
            self._prune(step)
        _barrier()
        return path

    def load(self, path: str, models: list[nn.Module], optimizers: list[torch.optim.Optimizer]) -> dict:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        state = {}
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            model_sd, optim_sd = get_state_dict(model, optimizer)
            state[f"model_{i}"] = model_sd
            state[f"optim_{i}"] = optim_sd
        dcp.load(state, checkpoint_id=path)
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            set_state_dict(model, optimizer, model_state_dict=state[f"model_{i}"],
                           optim_state_dict=state[f"optim_{i}"])
        return torch.load(os.path.join(path, "extra.pt"), weights_only=False)

    def _prune(self, step: int) -> None:
        for name in sorted(os.listdir(self.root)):
            if not name.startswith("step_"):
                continue
            old = int(name[5:])
            if old < step and not (self.keep_every and old % self.keep_every == 0):
                shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)


def export_weights(model: nn.Module, path: str, model_spec: dict) -> None:
    """Write the full (unsharded) weights as ``model.safetensors`` and the model spec as JSON."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if is_main():
        from safetensors.torch import save_file

        os.makedirs(path, exist_ok=True)
        save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(path, "model.safetensors"))
        with open(os.path.join(path, "model.json"), "w") as f:
            json.dump(model_spec, f, indent=2)
    _barrier()


def load_weights(path: str) -> dict[str, torch.Tensor]:
    """Weights from ``model.safetensors`` (a file or a directory holding it) or a torch state dict."""
    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)
