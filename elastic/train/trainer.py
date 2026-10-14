"""Training entry point.

    torchrun --nproc_per_node=8 -m elastic.train configs/paper/scale1/shellelastic.yaml [key=value ...]

Trains one elastic (or standalone) model, or -- with a ``multi`` section -- several standalone
models in one run, where each step's batch trains one of them (sampled standalone).
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn

from elastic.config import ExperimentConfig, load_config, model_config
from elastic.data.text import TextData
from elastic.eval.config import check_eval_inputs
from elastic.models import DenseElasticConfig, DenseElasticLM, ShellElasticLM
from elastic.optim import OptimConfig, build_adamw, wsd_lambda, wsd_schedule
from elastic.train.checkpoint import Checkpointer, export_weights, initialize_adamw_state, load_weights
from elastic.train.distributed import all_reduce_mean, build_mesh, is_main, setup, shard_model, teardown
from elastic.train.objective import Objective, ObjectiveConfig
from elastic.train.selection import WidthSelector, parse_step_rule


class MetricLogger:
    """Rank-0 JSON-lines log (``metrics.jsonl``) and optional Weights & Biases mirror."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.file = None
        self.wandb = None
        if not is_main():
            return
        os.makedirs(cfg.run.output_dir, exist_ok=True)
        self.file = open(os.path.join(cfg.run.output_dir, "metrics.jsonl"), "a")
        if cfg.logging.wandb:
            import wandb

            self.wandb = wandb.init(project=cfg.logging.project, entity=cfg.logging.entity, name=cfg.run.name,
                                    config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
                                    resume="allow", id=cfg.run.name)

    def log(self, values: dict, step: int) -> None:
        if self.file is None:
            return
        self.file.write(json.dumps({"step": step, **values}) + "\n")
        self.file.flush()
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
        if self.wandb is not None:
            self.wandb.finish()


def _print(*args) -> None:
    if is_main():
        print(*args, flush=True)


def build_model(spec: dict, seed: int, device: torch.device, init_from: str | None = None) -> nn.Module:
    """Full model on ``device`` initialized from ``seed`` (identical on every rank)."""
    cfg = model_config(spec)
    torch.manual_seed(seed)
    with torch.device(device):
        model = DenseElasticLM(cfg) if isinstance(cfg, DenseElasticConfig) else ShellElasticLM(cfg)
    if init_from:
        model.load_state_dict(load_weights(init_from), strict=True)
    return model


def model_spec_with_trained(spec: dict, objective: ObjectiveConfig) -> dict:
    """Fill ``trained`` from the objective's loss weights unless the spec sets it."""
    spec = dict(spec)
    if spec.get("trained") is None and objective.loss_weights is not None:
        supported = spec.get("supported") or model_config({**spec, "trained": None}).supported
        spec["trained"] = [s for s, w in zip(supported, objective.loss_weights) if w != 0]
    return spec


def clip_grad_norm(model: nn.Module, max_norm: float) -> float:
    from torch.distributed.tensor import DTensor

    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    total = torch.nn.utils.get_total_norm([p.grad for p in params], 2.0, error_if_nonfinite=False, foreach=True)
    if isinstance(total, DTensor):
        total = total.full_tensor()
    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total, foreach=True)
    return float(total)


def _floats(metrics: dict) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def _batches_in_lockstep(iterator, device: torch.device):
    """Yield batches while every rank still has one (validation shards may be uneven)."""
    while True:
        batch = next(iterator, None)
        flag = torch.tensor([0 if batch is None else 1], device=device)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if int(flag) == 0:
            return
        yield batch


VAL_PREFIXES = ("loss_width_", "loss_ensemble_", "optmix_weight_width_", "ensemble_teacher_", "loss_router_",
                "router_weight_width_")
VAL_KEYS = {"loss", "router_entropy", "loss_low", "loss_medium", "loss_high", "loss_extra", "loss_max"}


@torch.no_grad()
def validate(models: list[nn.Module], objectives: list[Objective], data: TextData, steps: int,
             device: torch.device) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    count = 0
    for model in models:
        model.eval()
    for i, (inputs, targets) in enumerate(_batches_in_lockstep(data.val_batches(), device)):
        if i >= steps:
            break
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        values: dict[str, float] = {}
        with torch.autocast(device.type, dtype=torch.bfloat16):
            for model, objective in zip(models, objectives):
                metrics = _floats(objective.validation_step(model, inputs, targets))
                keep = (lambda k: k.startswith("loss_width_") and k != "loss_width_avg") if len(models) > 1 else (
                    lambda k: k in VAL_KEYS or k.startswith(VAL_PREFIXES))
                values.update({k: v for k, v in metrics.items() if keep(k)})
        if len(models) > 1:
            widths = [v for k, v in values.items() if k.startswith("loss_width_")]
            values["loss_width_avg"] = sum(widths) / len(widths)
        for k, v in all_reduce_mean(values, device).items():
            sums[k] += v
        count += 1
    for model in models:
        if hasattr(model, "reshard"):
            model.reshard()
        model.train()
    return {f"val_{k}": v / count for k, v in sums.items()} if count else {}


def _setup_run(cfg: ExperimentConfig):
    task_files = [f for f in (cfg.eval.probe, cfg.eval.tasks if cfg.train.final_eval else None) if f]
    if task_files:
        check_eval_inputs(cfg.eval, task_files)
    rank, world, device = setup()
    mesh = build_mesh(world, cfg.train.shard_size, device)
    if is_main():
        os.makedirs(cfg.run.output_dir, exist_ok=True)
        with open(os.path.join(cfg.run.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.structured(cfg)))
    return rank, world, device, mesh


def train(cfg: ExperimentConfig) -> None:
    """Train a single elastic or standalone model."""
    if cfg.multi is not None and cfg.multi.models:
        return train_multi(cfg)
    rank, world, device, mesh = _setup_run(cfg)
    spec = model_spec_with_trained(cfg.model, cfg.objective)
    model = build_model(spec, cfg.run.seed, device, cfg.train.init_from)
    model.set_activation_checkpointing(cfg.train.activation_checkpointing)
    objective = Objective(cfg.objective, model)
    shard_model(model, mesh)
    optimizer = build_adamw(model, cfg.optim)
    scheduler = wsd_schedule(optimizer, cfg.schedule.warmup_steps, cfg.schedule.decay_start, cfg.train.steps,
                             cfg.schedule.min_ratio)
    data = TextData(cfg.data, world)
    checkpointer = Checkpointer(cfg.run.output_dir, cfg.train.keep_every)
    start = _maybe_resume(cfg, checkpointer, [model], [optimizer], [scheduler], data)
    _print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters; start step {start}")
    _loop(cfg, [model], [optimizer], [scheduler], [objective], None, data, checkpointer, start, device, spec)


def _maybe_resume(cfg, checkpointer, models, optimizers, schedulers, data) -> int:
    path = checkpointer.latest() if cfg.train.resume == "auto" else cfg.train.resume
    if not path:
        return 0
    extra = checkpointer.load(path, models, optimizers)
    for scheduler, state in zip(schedulers, extra["schedulers"]):
        scheduler.load_state_dict(state)
        # The optimizer state carries the learning rate of the saved schedule; recompute it from this
        # run's schedule so that a changed schedule (e.g. a longer run) takes effect immediately.
        for group, base, fn in zip(scheduler.optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
            group["lr"] = base * fn(scheduler.last_epoch)
    data.load_state_dict(extra["data"])
    _print(f"resumed from {path} at step {extra['step']}")
    return int(extra["step"])


def _loop(cfg, models, optimizers, schedulers, objectives, model_selector, data, checkpointer, start, device,
          spec) -> None:
    logger = MetricLogger(cfg)
    lr_scale = wsd_lambda(cfg.schedule.warmup_steps, cfg.schedule.decay_start, cfg.train.steps, cfg.schedule.min_ratio)
    step = start
    val_every = cfg.train.val_every
    last_val = start - val_every if (start > 0 and start % val_every == 0) else start
    consecutive = 0
    interval: dict[str, list[float]] = defaultdict(list)
    probe = None
    tic = time.time()
    gc.disable()
    for model in models:
        model.train()
    for inputs, targets in data.train_batches():
        if step >= cfg.train.steps:
            break
        step += 1
        idx = model_selector(step) if model_selector is not None else 0
        model, optimizer, objective = models[idx], optimizers[idx], objectives[idx]
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            metrics = _floats(objective.train_step(model, inputs, targets, step))
        grad_norm = clip_grad_norm(model, cfg.train.grad_clip)
        skip = math.isnan(grad_norm) or (grad_norm > cfg.train.grad_clip * cfg.train.skip_factor
                                         and step >= cfg.train.skip_start)
        consecutive = consecutive + 1 if skip else 0
        if consecutive > cfg.train.max_consecutive_skips:
            raise RuntimeError(f"{consecutive} consecutive skipped updates (gradient norm {grad_norm})")
        if not skip:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if model_selector is not None:
            metrics = {k: v for k, v in metrics.items() if k.startswith("loss_width_") and k != "loss_width_avg"}
        metrics.update(grad_norm=grad_norm, skipped=float(skip), lr=cfg.optim.lr * lr_scale(step - 1))
        for k, v in all_reduce_mean(metrics, device).items():
            interval[k].append(v)
        if step % cfg.train.log_every == 0:
            elapsed = time.time() - tic
            values = {f"train_{k}": sum(v) / len(v) for k, v in interval.items()}
            values["step_time"] = elapsed / len(interval["grad_norm"])  # grad_norm is logged at every step
            logger.log(values, step)
            _print(f"step {step} " + " ".join(f"{k}={v:.4f}" for k, v in values.items()
                                             if k in ("train_loss", "train_grad_norm", "step_time")))
            interval.clear()
            tic = time.time()
        if (step % cfg.train.ckpt_every == 0 or step >= cfg.train.steps) and consecutive == 0:
            extra = {"step": step, "schedulers": [s.state_dict() for s in schedulers], "data": data.state_dict()}
            checkpointer.save(step, models, optimizers, extra)
        if step // val_every > last_val // val_every or (step >= cfg.train.steps and last_val < step):
            last_val = step
            values = validate(models, objectives, data, cfg.train.val_steps, device)
            if cfg.eval.probe:
                if probe is None:
                    from elastic.eval.icl import ICLEvaluator

                    probe = ICLEvaluator(cfg.eval, cfg.eval.probe, prefix="icl_probe/")
                values.update(probe(models, device))
            logger.log(values, step)
            _print(f"step {step} validation " + " ".join(f"{k}={v:.4f}" for k, v in sorted(values.items())
                                                          if k.startswith("val_loss")))
        if step % 250 == 0:
            gc.collect()
    gc.enable()
    for i, model in enumerate(models):
        name = "final" if len(models) == 1 else f"final_{i}"
        export_weights(model, os.path.join(cfg.run.output_dir, name),
                       spec if len(models) == 1 else spec[i])
    if cfg.train.final_eval:
        from elastic.eval.icl import evaluate_models

        results = evaluate_models(cfg, models, device)
        logger.log(results, step)
    logger.close()
    teardown()


def train_multi(cfg: ExperimentConfig) -> None:
    """Sampled standalone: separate models share the data stream; one model takes each batch."""
    rank, world, device, mesh = _setup_run(cfg)
    entries = cfg.multi.models
    models, optimizers, schedulers, objectives, specs = [], [], [], [], []
    for entry in entries:
        spec = {**cfg.model, **entry.model}
        model = build_model(spec, entry.seed if entry.seed is not None else cfg.run.seed, device)
        model.set_activation_checkpointing(cfg.train.activation_checkpointing)
        objectives.append(Objective(replace(cfg.objective), model))
        shard_model(model, mesh)
        optim_cfg = OptimConfig(**{**asdict(cfg.optim), **entry.optim})
        optimizer = build_adamw(model, optim_cfg)
        initialize_adamw_state(optimizer)
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(wsd_schedule(optimizer, cfg.schedule.warmup_steps, cfg.schedule.decay_start,
                                       cfg.train.steps, cfg.schedule.min_ratio))
        specs.append(spec)
    indices = list(range(len(models)))
    if cfg.multi.selection == "schedule":
        if cfg.multi.schedule is None or len(cfg.multi.schedule) != len(models):
            raise ValueError("multi.schedule needs one rule per model")
        rules = {i: parse_step_rule(r) for i, r in zip(indices, cfg.multi.schedule)}
        period = math.lcm(*(rule[1] for rule in rules.values()))
        for t in range(1, period + 1):
            if sum(1 for rule in rules.values() if t % rule[1] == rule[0] % rule[1]) != 1:
                raise ValueError(f"multi.schedule must select exactly one model at every step (step {t})")
        selector = WidthSelector(indices, "schedule", rules)
    elif cfg.multi.selection == "random":
        selector = WidthSelector(indices, "random", seed=cfg.multi.seed)
    else:
        raise ValueError(f"multi.selection must be 'random' or 'schedule', got {cfg.multi.selection!r}")
    data = TextData(cfg.data, world)
    checkpointer = Checkpointer(cfg.run.output_dir, cfg.train.keep_every)
    start = _maybe_resume(cfg, checkpointer, models, optimizers, schedulers, data)
    _loop(cfg, models, optimizers, schedulers, objectives, lambda step: selector(step)[0], data, checkpointer,
          start, device, specs)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: torchrun ... -m elastic.train <config.yaml> [key=value ...]")
    train(load_config(argv[0], argv[1:]))


if __name__ == "__main__":
    main()
