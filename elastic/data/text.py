"""Pre-tokenized WebDataset text loader (OpenLM format).

Each shard is a ``.tar`` file whose samples hold one token sequence each, either as a JSON list in
``<key>.json.gz`` or as a Python list literal in ``<key>.txt``. A source directory may contain a
``manifest.jsonl`` (one ``{"shard": name, "num_sequences": n}`` per line); otherwise its ``.tar``
files are listed. Every sample yields a random window of ``seq_len + 1`` tokens split into inputs
and next-token targets.

The shard discovery, deterministic shard shuffle, per-node/per-worker split and sample handling
are adapted from OpenLM (MIT License, https://github.com/mlfoundations/open_lm).
"""

from __future__ import annotations

import ast
import json
import logging
import math
import random
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from multiprocessing import Value
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import fsspec
    import webdataset as wds
    from webdataset.filters import _shuffle
    from webdataset.tariterators import base_plus_ext, tar_file_expander, url_opener, valid_sample
except ImportError:  # pragma: no cover
    fsspec = wds = None

_MISSING_DATA_EXTRA = "the data loader needs the 'data' extra: pip install -e '.[data]'"


@dataclass
class TextDataConfig:
    train_sources: list[str] = field(default_factory=list)
    val_sources: list[str] = field(default_factory=list)
    train_key: str = "json.gz"  # "json.gz" | "txt"
    val_key: str = "txt"
    tokens: int = 115_000_000_000  # token budget that bounds the shards read for training
    val_tokens: int = 0  # 0: every validation sample
    seq_len: int = 2048
    vocab_size: int = 50432
    batch_size: int = 16  # per GPU
    eval_batch_size: int = 16  # per GPU
    num_workers: int = 8
    seed: int = 1
    target_mask_left: int | None = 50300  # targets up to the last occurrence of this token are ignored
    target_mask_individual: int | None = 50400  # targets equal to this token are ignored
    squash_mask_left: bool = True  # remove mask-left tokens and left-pack the sequence


# ---------------------------------------------------------------------------------------------
# WebDataset plumbing (adapted from OpenLM)
# ---------------------------------------------------------------------------------------------


class SharedEpoch:
    def __init__(self, epoch: int = 0) -> None:
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch: int) -> None:
        self.shared_epoch.value = epoch

    def get_value(self) -> int:
        return self.shared_epoch.value


def _log_and_continue(exn: Exception) -> bool:
    logger.warning("Handling webdataset error (%r). Ignoring.", exn)
    return True


def _group_by_keys_nothrow(data, keys=None, lcase=True, suffixes=None, handler=None):
    keys = keys or base_plus_ext
    current = None
    for filesample in data:
        if "fname" in filesample and "data" in filesample:
            fname, value = filesample["fname"], filesample["data"]
        elif "__key__" in filesample and "data" in filesample:
            fname, value = filesample["__key__"], filesample["data"]
        else:
            continue
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current is None or prefix != current["__key__"] or suffix in current:
            if valid_sample(current):
                yield current
            current = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current[suffix] = value
    if valid_sample(current):
        yield current


def tarfile_to_samples_nothrow(src, handler=_log_and_continue):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    return _group_by_keys_nothrow(files, handler=handler)


class DetShuffle(wds.PipelineStage if wds is not None else object):
    """Shuffle whose seed depends only on (seed, epoch), identical on every rank."""

    def __init__(self, bufsize: int = 1000, initial: int = 100, seed: int = 0, epoch=-1) -> None:
        self.bufsize, self.initial, self.seed, self.epoch = bufsize, initial, seed, epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        rng.seed(self.seed + epoch)
        return _shuffle(src, self.bufsize, self.initial, rng)


class FiniteDataPipeline(wds.DataPipeline if wds is not None else object):
    def __iter__(self):
        if self.nsamples > 0:
            return islice(self.iterator(), self.nsamples)
        return self.iterator()


def _to_pipe_url(path: str) -> str:
    return f"pipe:aws s3 cp {path} -" if path.startswith("s3://") else path


def discover_shards(source: str, shuffle_seed: int | None) -> tuple[list[str], list[int]]:
    """Shard URLs and per-shard sequence counts of one source directory."""
    source = source.rstrip("/")
    try:
        with fsspec.open(source + "/manifest.jsonl", "rb") as f:
            entries = [json.loads(line) for line in f.read().decode("utf-8").strip().split("\n") if line.strip()]
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(entries)
        urls = [_to_pipe_url(source + "/" + e["shard"] + ".tar") for e in entries]
        return urls, [e.get("num_sequences", e.get("num_chunks", 0)) for e in entries]
    except FileNotFoundError:
        fs, prefix = fsspec.core.url_to_fs(source)
        tars = sorted(p for p in fs.ls(prefix.rstrip("/"), detail=False) if p.endswith(".tar"))
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
        urls = [f"{protocol}://{p}" if protocol != "file" else p for p in tars]
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(urls)
        return [_to_pipe_url(u) for u in urls], [0] * len(urls)


def select_shards(sources: list[str], num_samples: int, shuffle_seed: int | None) -> tuple[list[str], int]:
    """Shards of all sources, in order, until their sequence counts cover ``num_samples``."""
    urls: list[str] = []
    total = 0
    for source in sources:
        for url, count in zip(*discover_shards(source, shuffle_seed)):
            urls.append(url)
            total += count
            if 0 < num_samples <= total:
                return urls, min(total, num_samples)
    return urls, (min(total, num_samples) if num_samples > 0 else total)


def count_samples(urls: list[str]) -> int:
    return sum(1 for _ in wds.DataPipeline(wds.SimpleShardList(urls), tarfile_to_samples_nothrow))


# ---------------------------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------------------------


def _clamp(tokens, vocab_size: int) -> list[int]:
    return [(t % vocab_size if t < 0 else t) if isinstance(t, int) else 0 for t in tokens]


def _parse_txt(raw: bytes, vocab_size: int) -> list[int]:
    return _clamp(ast.literal_eval(raw.decode()), vocab_size)


def _example(sample: tuple, cfg: "TextDataConfig") -> tuple[torch.Tensor, torch.Tensor]:
    return make_example(sample[0], cfg.seq_len, cfg.target_mask_left, cfg.target_mask_individual,
                        cfg.squash_mask_left)


def _long_enough(seq_len: int, sample: tuple) -> bool:
    return len(sample[0]) > seq_len


def make_example(tokens: list[int], seq_len: int, mask_left: int | None = None, mask_individual: int | None = None,
                 squash_mask_left: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """A random window of ``seq_len + 1`` tokens as ``(inputs, targets)`` with target masking."""
    n = len(tokens)
    if n < seq_len + 1:
        raise ValueError(f"sequence of {n} tokens is shorter than seq_len + 1 = {seq_len + 1}")
    start = 0 if n == seq_len + 1 else random.randint(0, n - seq_len - 1)
    chunk = tokens[start : start + seq_len + 1]
    inputs = torch.LongTensor(chunk[:seq_len])
    targets = torch.LongTensor(chunk[1 : seq_len + 1])
    if mask_left is None and mask_individual is None:
        return inputs, targets
    inputs, targets = inputs.clone(), targets.clone()
    if mask_left is not None:
        at_mark = targets == mask_left
        ignore = at_mark.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]) > 0
        if squash_mask_left:
            ignore = ignore & ~at_mark
            targets[ignore] = -100
            pad = mask_individual if mask_individual is not None else 0
            kept = inputs[inputs != mask_left]
            inputs = torch.full((seq_len,), pad, dtype=inputs.dtype)
            inputs[: len(kept)] = kept
            kept = targets[targets != mask_left]
            targets = torch.full((seq_len,), pad, dtype=targets.dtype)
            targets[: len(kept)] = kept
        else:
            targets[ignore] = -100
    if mask_individual is not None:
        targets[targets == mask_individual] = -100
    return inputs, targets


def build_loader(urls: list[str], *, train: bool, cfg: TextDataConfig, batch_size: int, num_samples: int,
                 num_workers: int, world_size: int, key: str):
    """WebDataset loader yielding ``(inputs, targets)`` batches of ``[batch_size, seq_len]``."""
    if wds is None:
        raise ImportError(_MISSING_DATA_EXTRA)
    epoch = SharedEpoch(0)
    stages: list = [wds.SimpleShardList(urls)]
    if train:
        stages += [
            DetShuffle(bufsize=0, initial=0, seed=cfg.seed, epoch=epoch),
            wds.split_by_node,
            wds.split_by_worker,
            tarfile_to_samples_nothrow,
            wds.shuffle(bufsize=0, initial=0, rng=random.Random(cfg.seed)),
        ]
    else:
        stages += [wds.tarfile_to_samples(handler=wds.reraise_exception), wds.split_by_node, wds.split_by_worker]
    if key in ("json.gz", "json"):
        stages += [
            wds.decode(),
            wds.rename(json=key),
            wds.map_dict(json=partial(_clamp, vocab_size=cfg.vocab_size)),
            wds.to_tuple("json"),
        ]
    elif key == "txt":
        stages += [
            wds.map_dict(txt=partial(_parse_txt, vocab_size=cfg.vocab_size)),
            wds.to_tuple("txt"),
        ]
    else:
        raise ValueError(f"unknown data key {key!r}")
    stages += [
        wds.select(partial(_long_enough, cfg.seq_len)),
        wds.map(partial(_example, cfg=cfg)),
        wds.batched(batch_size, partial=not train),
    ]
    dataset = FiniteDataPipeline(*stages)
    if train:
        workers = max(1, num_workers)
        worker_batches = math.floor(num_samples / (batch_size * world_size * workers))
        if worker_batches <= 0:
            raise ValueError("no training batches: increase the token budget or reduce workers/GPUs")
        num_batches = worker_batches * workers
        dataset = dataset.repeat(nepochs=1, nbatches=worker_batches)
    else:
        num_batches = math.ceil(num_samples / batch_size)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                           persistent_workers=False, generator=generator)
    loader.num_batches = num_batches
    return loader


class TextData:
    """Training and validation loaders with resumable training position."""

    def __init__(self, cfg: TextDataConfig, world_size: int) -> None:
        if wds is None:
            raise ImportError(_MISSING_DATA_EXTRA)
        self.cfg = cfg
        self.world_size = world_size
        self.batches_served = 0
        self._resume_skip = 0
        self.train_samples = cfg.tokens // cfg.seq_len
        urls, _ = select_shards(cfg.train_sources, self.train_samples, cfg.seed)
        if not urls:
            raise ValueError(f"no training shards found in {cfg.train_sources}")
        self.train_urls = urls
        self.val_urls: list[str] = []
        self.val_samples = 0
        if cfg.val_sources:
            val_budget = cfg.val_tokens // cfg.seq_len if cfg.val_tokens > 0 else 0
            self.val_urls, total = select_shards(cfg.val_sources, val_budget, cfg.seed)
            self.val_samples = val_budget if val_budget > 0 else total
            if self.val_samples <= 0 and self.val_urls:
                self.val_samples = count_samples(self.val_urls)

    def train_batches(self):
        loader = build_loader(self.train_urls, train=True, cfg=self.cfg, batch_size=self.cfg.batch_size,
                              num_samples=self.train_samples, num_workers=self.cfg.num_workers,
                              world_size=self.world_size, key=self.cfg.train_key)
        iterator = iter(loader)
        skip, self._resume_skip = self._resume_skip, 0
        for _ in range(skip):
            if next(iterator, None) is None:
                return
        self.batches_served = skip
        for batch in iterator:
            self.batches_served += 1
            yield batch

    def val_batches(self):
        if not self.val_urls:
            return iter(())
        return iter(build_loader(self.val_urls, train=False, cfg=self.cfg, batch_size=self.cfg.eval_batch_size,
                                 num_samples=self.val_samples, num_workers=max(1, self.cfg.num_workers // 2),
                                 world_size=self.world_size, key=self.cfg.val_key))

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.cfg.seed, "batches_served": self.batches_served}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._resume_skip = int(state.get("batches_served", 0))
