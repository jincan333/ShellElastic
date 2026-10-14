import json
import os

import pytest
import torch

from elastic.config import load_config

from test_data import write_shards

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

CONFIG = """
run: {{name: t, output_dir: {out}}}
model: {{type: shell, vocab_size: 256, seq_len: 32, n_layers: 2, base_width: 32, supported: [2, 3, 4],
        head_dim: 16, ffn_hidden: 32, proxy_width: 64}}
objective: {{loss_weights: [1, 1, 4]}}
schedule: {{warmup_steps: 2, decay_start: 4}}
data: {{train_sources: [{train}], val_sources: [{val}], val_key: json.gz, tokens: 30000, seq_len: 32, vocab_size: 256,
       batch_size: 4, eval_batch_size: 4, num_workers: 1, target_mask_left: null, target_mask_individual: null}}
train: {{steps: 6, log_every: 2, val_every: 3, val_steps: 2, ckpt_every: 3, skip_start: 1000}}
"""


@cuda
def test_train_resume_and_export(tmp_path):
    from elastic.train.trainer import train

    train_dir = write_shards(str(tmp_path), "train", "json.gz", n_shards=2, per_shard=200)
    val_dir = write_shards(str(tmp_path), "val", "json.gz", n_shards=1, per_shard=20)
    out = str(tmp_path / "run")
    path = tmp_path / "cfg.yaml"
    path.write_text(CONFIG.format(out=out, train=train_dir, val=val_dir))
    train(load_config(str(path)))
    records = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert any("val_loss_width_64" in r for r in records)
    assert [r["step"] for r in records if "train_loss" in r] == [2, 4, 6]
    assert os.path.exists(os.path.join(out, "final", "model.safetensors"))
    train(load_config(str(path), ["train.steps=8"]))
    records = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert records[-1]["step"] == 8
