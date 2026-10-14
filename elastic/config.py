"""Experiment configuration: YAML files with inheritance plus command-line overrides.

A YAML file may start with ``inherit: [relative/or/absolute.yaml, ...]``; the parents are merged
in order and the file's own values override them. Command-line overrides use dotted keys, for
example ``train.steps=100 objective.selection=random``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

from elastic.data.text import TextDataConfig
from elastic.eval.config import EvalConfig
from elastic.models import DenseElasticConfig, ShellConfig
from elastic.optim import OptimConfig
from elastic.train.objective import ObjectiveConfig


@dataclass
class RunConfig:
    name: str = "run"
    output_dir: str = "outputs/${run.name}"
    seed: int = 95  # model initialization seed


@dataclass
class ScheduleConfig:
    warmup_steps: int = 1000
    decay_start: int = 20000  # linear decay from here to zero at train.steps
    min_ratio: float = 0.0


@dataclass
class TrainConfig:
    steps: int = 25000
    grad_clip: float = 4.0
    skip_factor: float = 20.0  # skip the update when the gradient norm exceeds skip_factor * grad_clip ...
    skip_start: int = 5000  # ... from this step on
    max_consecutive_skips: int = 100
    shard_size: int = 0  # 0: shard over all GPUs; k: hybrid sharding in groups of k GPUs, replicated across groups
    activation_checkpointing: str = "1/2"  # fraction of the blocks: 0/1 (none), 1/8, 1/2, 1/1 (all), ...
    log_every: int = 50
    val_every: int = 2500
    val_steps: int = 300
    ckpt_every: int = 500
    keep_every: int = 5000  # older checkpoints are deleted unless their step is a multiple of this
    resume: str | None = "auto"  # "auto": latest checkpoint in output_dir; a path; or null
    init_from: str | None = None  # optional weights (.safetensors or .pt state dict) to start from
    final_eval: bool = False


@dataclass
class LoggingConfig:
    wandb: bool = False
    project: str = "elastic"
    entity: str | None = None


@dataclass
class ModelEntry:
    """One model of a sampled-standalone run: its architecture and optimizer overrides."""

    name: str = ""
    model: dict[str, Any] = field(default_factory=dict)
    optim: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None


@dataclass
class MultiConfig:
    """Sampled standalone: several separate models, one of which takes the step's batch."""

    selection: str = "random"  # random | schedule
    schedule: list[str] | None = None  # per model "M/N" rules (exactly one model must fire per step)
    seed: int = 0
    models: list[ModelEntry] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: dict[str, Any] = field(default_factory=dict)  # {type: shell | dense, **fields}
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: TextDataConfig = field(default_factory=TextDataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    multi: MultiConfig | None = None


def _load_yaml(path: str, seen: tuple[str, ...] = ()) -> DictConfig:
    path = os.path.abspath(path)
    if path in seen:
        raise ValueError(f"cyclic inherit: {' -> '.join(seen + (path,))}")
    raw = OmegaConf.load(path)
    parents = raw.pop("inherit", None) or []
    if isinstance(parents, str):
        parents = [parents]
    merged = OmegaConf.create()
    for parent in parents:
        parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
        merged = OmegaConf.merge(merged, _load_yaml(parent_path, seen + (path,)))
    return OmegaConf.merge(merged, raw)


def load_config(path: str, overrides: list[str] | None = None) -> ExperimentConfig:
    schema = OmegaConf.structured(ExperimentConfig)
    cfg = OmegaConf.merge(schema, _load_yaml(path), OmegaConf.from_dotlist(list(overrides or [])))
    OmegaConf.resolve(cfg)
    return OmegaConf.to_object(cfg)


MODEL_TYPES = {"shell": ShellConfig, "dense": DenseElasticConfig}


def model_config(spec: dict[str, Any]):
    """``{type: shell | dense, **fields}`` -> ShellConfig or DenseElasticConfig."""
    spec = dict(spec)
    kind = spec.pop("type", "shell")
    if kind not in MODEL_TYPES:
        raise ValueError(f"model type must be one of {sorted(MODEL_TYPES)}, got {kind!r}")
    return MODEL_TYPES[kind](**spec)
