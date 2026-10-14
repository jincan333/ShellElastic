"""In-context-learning evaluation of every sub-network of an elastic model.

This is the evaluation loop that LLM Foundry runs through Composer, without Composer. The task
datasets and metrics are vendored from LLM Foundry v0.22.0 (``datasets.py``, ``metrics.py``).
Each rank evaluates its ``DistributedSampler`` shard and drops the duplicate samples that the
sampler pads onto the last batch; labels are shifted for causal language modeling; generation
is greedy with a KV cache and follows the stopping rules of Hugging Face ``generate``. A
sub-network ``s`` is evaluated through ``model.infer(..., width=s)`` under bfloat16 autocast.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from elastic.eval.config import EvalConfig
from elastic.eval.icl import metrics as icl_metrics
from elastic.eval.icl.datasets import get_icl_task_dataloader
from elastic.eval.icl.utils import ICLDataSpec

DEFAULT_METRICS = {
    "language_modeling": ["InContextLearningLMAccuracy"],
    "multiple_choice": ["InContextLearningMultipleChoiceAccuracy"],
    "schema": ["InContextLearningMultipleChoiceAccuracy"],
    "generation_task_with_answers": ["InContextLearningGenerationExactMatchAccuracy"],
}
PRIMARY_METRICS = {name for names in DEFAULT_METRICS.values() for name in names}
GENERATION_KWARGS = {"pad_token_id", "eos_token_id", "max_new_tokens", "stopping_criteria", "use_cache"}


@dataclass
class ICLTask:
    label: str  # "{task}/{k}-shot"
    spec: ICLDataSpec
    metric_names: list[str]

    @property
    def name(self) -> str:
        task, shot = self.label.rsplit("/", 1)
        return f"{task}_{shot.removesuffix('-shot')}shot"


def load_task_list(path: str) -> list[dict]:
    """The ``icl_tasks`` list of an LLM Foundry style task file."""
    return OmegaConf.to_container(OmegaConf.load(path).icl_tasks, resolve=True)


def load_tokenizer(name: str):
    """The evaluation tokenizer, without a padding token so that batches are padded with EOS.

    LLM Foundry pads with the padding token or, when there is none, with EOS (and masks every
    padding-id position). GPT-NeoX-20B has no padding token in transformers 4.x; transformers 5
    adds ``<|padding|>``, which would change the batches.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, pad_token=None)


def build_icl_tasks(task_cfgs: list[dict], tokenizer, max_seq_len: int, batch_size: int,
                    data_dir: str = "") -> list[ICLTask]:
    """One task per (dataset, number of shots) with LLM Foundry's defaults.

    Relative ``dataset_uri`` entries resolve against ``data_dir``.
    """
    pad_tok_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tasks = []
    for raw in task_cfgs:
        cfg = copy.deepcopy(dict(raw))
        for key in ("label", "dataset_uri", "icl_task_type", "num_fewshot"):
            if cfg.get(key) is None:
                raise ValueError(f"ICL task {raw} is missing {key!r}")
        if "num_beams" in cfg:
            raise ValueError("num_beams is not supported: generation is greedy")
        if cfg.pop("has_categories", False):
            raise NotImplementedError("per-category ICL tasks are not supported")
        label = cfg.pop("label")
        task_type = cfg.pop("icl_task_type")
        if task_type not in DEFAULT_METRICS:
            raise ValueError(f"unknown icl_task_type {task_type!r}; expected one of {sorted(DEFAULT_METRICS)}")
        metric_names = list(cfg.pop("metric_names", DEFAULT_METRICS[task_type]))
        task_batch = int(cfg.pop("batch_size", batch_size))
        cfg.setdefault("max_seq_len", max_seq_len)
        uri = cfg.pop("dataset_uri")
        if not uri.startswith("hf://") and not os.path.isabs(uri):
            uri = os.path.join(data_dir, uri)
        hf_loading_vars = cfg.pop("hf_loading_vars", {})
        hf_parsing_map = cfg.pop("hf_parsing_map", {})
        if "question_prelimiter" in cfg:
            if "prelimiter" in cfg:
                raise ValueError(f"ICL task {label!r} sets both question_prelimiter and prelimiter")
            cfg["prelimiter"] = cfg.pop("question_prelimiter")
        for shots in cfg.pop("num_fewshot"):
            kwargs = {**cfg, "pad_tok_id": pad_tok_id, "num_fewshot": int(shots)}
            spec = get_icl_task_dataloader(task_type, uri, tokenizer, task_batch, hf_loading_vars=hf_loading_vars,
                                           hf_parsing_map=hf_parsing_map, kwargs=kwargs)
            tasks.append(ICLTask(f"{label}/{shots}-shot", spec, metric_names))
    return tasks


def _to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        try:
            return type(batch)(_to_device(x, device) for x in batch)
        except TypeError:
            return [_to_device(x, device) for x in batch]
    return batch


def _world() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _global_sum(value: int, device: torch.device) -> int:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return int(value)
    t = torch.tensor(value, device=device)
    dist.all_reduce(t)
    return int(t.item())


@torch.no_grad()
def generate_greedy(model, width: int, input_ids: torch.Tensor, attention_mask: torch.Tensor, *,
                    max_new_tokens: int, eos_token_id=None, pad_token_id: int | None = None,
                    stopping_criteria=None, synced: bool = False) -> torch.Tensor:
    """Greedy decoding with a KV cache, as Hugging Face ``generate``; returns prompt + new tokens.

    A sequence stops at EOS; the batch stops when every sequence has stopped, a batch-level
    ``stopping_criteria`` fires, or ``max_new_tokens`` is reached. Stopped sequences are filled
    with ``pad_token_id``. With ``synced`` every rank keeps running forward passes until all
    ranks are done (required for sharded models).
    """
    batch, device = input_ids.shape[0], input_ids.device
    max_length = input_ids.shape[1] + max_new_tokens
    eos = None
    if eos_token_id is not None:
        eos = torch.tensor(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id], device=device)
        if pad_token_id is None:
            raise ValueError("pad_token_id is required with eos_token_id")
    tokens, mask, step_input, cache = input_ids, attention_mask, input_ids, None
    unfinished = torch.ones(batch, dtype=torch.long, device=device)
    finished = False
    while True:
        if synced:
            flag = torch.tensor(0.0 if finished else 1.0, device=device)
            dist.all_reduce(flag)
            if flag.item() == 0.0:
                break
        elif finished:
            break
        if finished:
            model.infer(tokens[:, -1:], width)  # keep the other ranks' collectives going
            continue
        # The readout covers every position: reading out only the last one can select another bfloat16
        # GEMM kernel for some shapes and change the last bit of the logits.
        logits, cache = model.infer(step_input, width, attention_mask=mask, past_key_values=cache, use_cache=True)
        mask = torch.cat([mask, mask.new_ones((batch, 1))], dim=-1)
        next_tokens = torch.argmax(logits[:, -1, :].float(), dim=-1)
        if eos is not None:
            next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)
        done = torch.full((batch,), tokens.shape[-1] >= max_length, dtype=torch.bool, device=device)
        if eos is not None:
            done = done | torch.isin(tokens[:, -1], eos)
        for criterion in stopping_criteria or []:
            done = done | criterion(tokens, None)
        unfinished = unfinished & ~done
        finished = bool(unfinished.max() == 0)
        step_input = next_tokens[:, None]
    return tokens


def _model_outputs(model, width: int, batch: dict, tokenizer, synced: bool):
    """Model outputs for the metrics and the (shifted) labels; pops ``labels`` from ``batch``."""
    labels = batch.pop("labels")
    if batch.get("mode") == "generate":
        kwargs = dict(batch.get("generation_kwargs", {}))
        unknown = set(kwargs) - GENERATION_KWARGS
        if unknown:
            raise ValueError(f"unsupported generation_kwargs {sorted(unknown)} (generation is greedy)")
        kwargs.pop("use_cache", None)
        generation = generate_greedy(model, width, batch["input_ids"], batch["attention_mask"], synced=synced,
                                     **kwargs)
        texts = tokenizer.batch_decode(generation[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        if len(tokenizer(" a", add_special_tokens=False)["input_ids"]) != 1:
            texts = [" " + text for text in texts]
        return texts, labels
    labels[:, :-1] = labels[:, 1:].clone()
    labels[:, -1] = -100
    # The full right-padded batch, as LLM Foundry scores it: dropping the padding would not change the
    # scored positions in exact arithmetic, but a shorter batch can select other bfloat16 GEMM kernels.
    logits, _ = model.infer(batch["input_ids"], width, attention_mask=batch.get("attention_mask"))
    return logits, labels


@torch.no_grad()
def evaluate_task(model, width: int, task: ICLTask, tokenizer, device: torch.device) -> dict[str, float]:
    """Metrics of sub-network ``width`` on one task (all ranks take part)."""
    rank, world = _world()
    metrics = {name: getattr(icl_metrics, name)().to(device) for name in task.metric_names}
    loader = task.spec.dataloader
    sampler = loader.sampler
    dataset_len = len(sampler.dataset)
    padding = sampler.total_size - dataset_len
    seen = 0
    for full_batch in loader:
        rank_samples = task.spec.get_num_samples_in_batch(full_batch)
        last_batch = seen + _global_sum(rank_samples, device) >= dataset_len
        microbatches = task.spec.split_batch(full_batch, loader.batch_size)
        for i, batch in enumerate(microbatches):
            batch = _to_device(batch, device)
            skip_update = False
            if last_batch and i == len(microbatches) - 1 and rank >= world - padding:
                # This rank's last sample repeats one from the start of the dataset.
                rank_samples -= 1
                samples = task.spec.get_num_samples_in_batch(batch)
                if samples == 1:
                    skip_update = True  # still run the forward pass: sharded models need every rank
                else:
                    batch = task.spec.split_batch(batch, samples - 1)[0]
            with torch.autocast(device.type, dtype=torch.bfloat16):
                outputs, labels = _model_outputs(model, width, batch, tokenizer, synced=world > 1)
                if not skip_update:
                    for metric in metrics.values():
                        metric.update(batch=batch, outputs=outputs, labels=labels)
            del outputs
        seen += _global_sum(rank_samples, device)
    return {name: float(metric.compute()) for name, metric in metrics.items()}


def width_label(model, s: int) -> int:
    """Width used in metric names: the model width (the FFN width for MatFormer)."""
    unit = model.cfg.ffn_hidden if getattr(model, "ffn_only", False) else model.g
    return int(s) * int(unit)


@torch.no_grad()
def evaluate_model(model, tasks: list[ICLTask], tokenizer, device: torch.device,
                   widths: list[int] | None = None) -> dict[str, float]:
    """``{task}_{k}shot_width_{n}`` for every task and sub-network, and ``mean_width_{n}`` over tasks."""
    widths = [int(s) for s in (widths if widths is not None else model.trained) if int(s) in model.supported]
    was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    for s in widths:
        label = width_label(model, s)
        primary = []
        for task in tasks:
            for metric, value in evaluate_task(model, s, task, tokenizer, device).items():
                key = task.name if metric in PRIMARY_METRICS else f"{task.name}_{metric}"
                results[f"{key}_width_{label}"] = value
                if metric in PRIMARY_METRICS:
                    primary.append(value)
        if primary:
            results[f"mean_width_{label}"] = sum(primary) / len(primary)
    model.train(was_training)
    if hasattr(model, "reshard"):
        model.reshard()
    return results


class ICLEvaluator:
    """A tokenizer and the tasks of one task file, reused across evaluations."""

    def __init__(self, cfg: EvalConfig, task_file: str, prefix: str = "icl/") -> None:
        import datasets

        datasets.disable_progress_bars()
        self.cfg = cfg
        self.prefix = prefix
        self.tokenizer = load_tokenizer(cfg.tokenizer)
        self.tasks = build_icl_tasks(load_task_list(task_file), self.tokenizer, cfg.max_seq_len, cfg.batch_size,
                                     cfg.data_dir)

    def __call__(self, models, device: torch.device) -> dict[str, float]:
        results: dict[str, float] = {}
        for model in models:
            for key, value in evaluate_model(model, self.tasks, self.tokenizer, device, self.cfg.widths).items():
                if self.prefix + key in results:
                    raise ValueError(f"two models report {key}; give them different widths")
                results[self.prefix + key] = value
        return results


def evaluate_models(cfg, models, device: torch.device) -> dict[str, float]:
    """Final evaluation of a training run (``cfg.eval.tasks``)."""
    return ICLEvaluator(cfg.eval, cfg.eval.tasks)(models, device)
