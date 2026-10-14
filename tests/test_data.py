import gzip
import json
import os
import random

import pytest
import torch

wds = pytest.importorskip("webdataset")

from elastic.data.text import TextData, TextDataConfig, make_example  # noqa: E402


def write_shards(root, split, key, n_shards=3, per_shard=20, seq_len=32, seed=0, manifest=True):
    rng = random.Random(seed)
    d = os.path.join(root, split)
    os.makedirs(d, exist_ok=True)
    entries = []
    for i in range(n_shards):
        name = f"shard_{i:05d}"
        with wds.TarWriter(os.path.join(d, name + ".tar"), encoder=False) as w:
            for j in range(per_shard):
                toks = [rng.randint(0, 250) for _ in range(seq_len + 1 + rng.randint(0, 8))]
                payload = gzip.compress(json.dumps(toks).encode()) if key == "json.gz" else str(toks).encode()
                w.write({"__key__": f"{name}_{j:05d}", key: payload})
        entries.append({"shard": name, "num_sequences": per_shard})
    if manifest:
        with open(os.path.join(d, "manifest.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(e) for e in entries) + "\n")
    return d


def test_make_example_masks_prompt_and_individual_tokens():
    tokens = [1, 2, 3, 50300, 4, 5, 50400, 6, 7]
    random.seed(0)
    inputs, targets = make_example(tokens, 8, mask_left=50300, mask_individual=50400, squash_mask_left=False)
    assert inputs.tolist() == tokens[:8]
    assert targets.tolist() == [-100, -100, -100, 4, 5, -100, 6, 7]
    inputs, targets = make_example(tokens, 8, mask_left=50300, mask_individual=50400, squash_mask_left=True)
    assert 50300 not in inputs.tolist()
    assert targets.tolist()[:6] == [-100, -100, 4, 5, -100, 6]


def test_loader_resume_and_formats(tmp_path):
    train = write_shards(str(tmp_path), "train", "json.gz")
    val = write_shards(str(tmp_path), "val", "txt", manifest=False)
    cfg = TextDataConfig(train_sources=[train], val_sources=[val], tokens=32 * 60, seq_len=32, batch_size=4,
                         eval_batch_size=4, num_workers=1, target_mask_left=None, target_mask_individual=None)
    data = TextData(cfg, world_size=1)
    batches = list(data.train_batches())
    assert len(batches) == 15 and batches[0][0].shape == (4, 32)
    assert torch.equal(batches[0][0][:, 1:], batches[0][1][:, :-1])
    assert sum(1 for _ in data.val_batches()) == 15
    resumed = TextData(cfg, world_size=1)
    resumed.load_state_dict({"batches_served": 5})
    rest = list(resumed.train_batches())
    assert len(rest) == 10 and all(torch.equal(a[0], b[0]) for a, b in zip(rest, batches[5:]))
