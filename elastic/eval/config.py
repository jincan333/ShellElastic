"""Downstream in-context-learning evaluation settings."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass
class EvalConfig:
    tasks: str = "configs/eval/tasks.yaml"  # task list evaluated after training (LLM Foundry ICL format)
    probe: str | None = None  # smaller task list evaluated at every validation (e.g. configs/eval/probe.yaml)
    data_dir: str = "eval_data"  # root of the task files; relative dataset_uri entries resolve against it
    tokenizer: str = "EleutherAI/gpt-neox-20b"
    batch_size: int = 16  # per GPU
    max_seq_len: int = 2048
    widths: list[int] | None = None  # sub-networks to evaluate (default: the trained ones)


def check_eval_inputs(cfg: EvalConfig, task_files: list[str]) -> None:
    """Raise if the evaluation of ``task_files`` cannot run: a missing ``eval`` extra or task file.

    Training calls this at start-up so that a run fails at once rather than at its first validation.
    """
    missing_modules = [m for m in ("transformers", "datasets", "torchmetrics") if importlib.util.find_spec(m) is None]
    if missing_modules:
        raise ImportError(f"the evaluation needs the 'eval' extra ({', '.join(missing_modules)} missing): "
                          "pip install -e '.[eval]'")
    missing = []
    for task_file in task_files:
        for task in OmegaConf.to_container(OmegaConf.load(task_file).icl_tasks, resolve=True):
            uri = task["dataset_uri"]
            if not uri.startswith("hf://") and not os.path.isabs(uri):
                uri = os.path.join(cfg.data_dir, uri)
            if not uri.startswith("hf://") and not os.path.exists(uri):
                missing.append(uri)
    if missing:
        missing = sorted(set(missing))
        raise FileNotFoundError(f"{len(missing)} evaluation task files not found (e.g. {missing[0]}): run "
                                f"scripts/download_eval_data.py --out {cfg.data_dir}, or train without the "
                                "evaluation (eval.probe=null train.final_eval=false)")
