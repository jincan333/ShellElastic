"""AdamW parameter groups from a model's ``optim_spec`` (Table 1; Tables 5 and 6 of Appendix D for the fully
elastic model and MatFormer) and the warmup-stable-decay schedule (Table 9)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class OptimConfig:
    lr: float = 2e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.95, 0.95)
    eps: float = 1e-14
    decoupled_wd: bool = True  # weight decay lambda_0 * q (per-step shrinkage independent of q)
    fused: bool = True


def param_groups(model: nn.Module, lr: float, weight_decay: float, decoupled_wd: bool = True) -> list[dict]:
    """AdamW parameter groups: learning rate ``lr / q`` and weight decay ``weight_decay * q``.

    ``model.optim_spec()`` maps every trainable parameter name to ``(q, decay)``. Parameters with
    the same (learning rate, weight decay) share one group; each group keeps the parameter names.
    """
    spec = model.optim_spec()
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing = set(named) ^ set(spec)
    if missing:
        raise ValueError(f"optim_spec does not cover exactly the trainable parameters: {sorted(missing)}")
    groups: dict[tuple[float, float], dict] = {}
    for name, p in named.items():
        q, decay = spec[name]
        group_lr = lr / q
        group_wd = (weight_decay * q if decoupled_wd else weight_decay) if decay else 0.0
        group = groups.setdefault((group_lr, group_wd), {"params": [], "param_names": [], "lr": group_lr,
                                                         "weight_decay": group_wd})
        group["params"].append(p)
        group["param_names"].append(name)
    return list(groups.values())


def build_adamw(model: nn.Module, cfg: OptimConfig) -> torch.optim.AdamW:
    groups = param_groups(model, cfg.lr, cfg.weight_decay, cfg.decoupled_wd)
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, fused=cfg.fused)


def wsd_lambda(warmup_steps: int, decay_start: int, total_steps: int, min_ratio: float = 0.0):
    """Linear warmup from 0, constant, then linear decay to ``min_ratio`` at ``total_steps``."""
    if not 0 <= warmup_steps <= decay_start:
        raise ValueError("require 0 <= warmup_steps <= decay_start")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be in [0, 1]")
    decay_steps = max(0, total_steps - decay_start)

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if step < decay_start or decay_steps <= 0:
            return 1.0
        progress = max(0.0, min(1.0, float(step - decay_start) / float(decay_steps)))
        return 1.0 - (1.0 - min_ratio) * progress

    return scale


def wsd_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, decay_start: int, total_steps: int,
                 min_ratio: float = 0.0) -> LambdaLR:
    return LambdaLR(optimizer, wsd_lambda(warmup_steps, decay_start, total_steps, min_ratio))
