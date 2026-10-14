import math

import pytest
import torch

from elastic.models import DenseElasticConfig, DenseElasticLM, ShellConfig, ShellElasticLM
from elastic.train.losses import LinearCrossEntropy


def tiny(**overrides) -> DenseElasticConfig:
    cfg = dict(vocab_size=256, seq_len=32, n_layers=2, base_width=16, supported=[2, 3, 4], head_dim=16,
               ffn_hidden=64, proxy_width=32)
    cfg.update(overrides)
    return DenseElasticConfig(**cfg)


def test_effective_init_matches_standalone_scale():
    """With a zero shift, sub-network s sees weights of std sigma/sqrt(m s) (and sqrt(e) for W2)."""
    torch.manual_seed(0)
    cfg = tiny(base_width=64, head_dim=16, ffn_hidden=256, proxy_width=32, supported=[2, 4, 8], init_std=0.02,
               n_layers=1)
    model = DenseElasticLM(cfg)
    m, e = 64 / 32, 256 / 64
    mlp = model.blocks[0].mlp
    for s in cfg.supported:
        assert mlp.w1.weight(s, s).std().item() == pytest.approx(0.02 / math.sqrt(m * s), rel=0.1)
        assert mlp.w2.weight(s, s).std().item() == pytest.approx(0.02 * math.sqrt(e / (m * s)), rel=0.1)


def test_fused_qkv_nests_per_group():
    cfg = tiny()
    model = DenseElasticLM(cfg)
    qkv = model.blocks[0].attn.qkv
    with torch.no_grad():
        qkv.weight_init.copy_(torch.arange(qkv.weight_init.numel(), dtype=torch.float32).view_as(qkv.weight_init))
    full = qkv.weight_init.view(3, 4 * 16, 4 * 16)
    w = qkv.weight(2, 2) * math.sqrt(2)
    torch.testing.assert_close(w.view(3, 32, 32), full[:, :32, :32])


def test_matformer_shares_attention_and_readout():
    torch.manual_seed(0)
    cfg = tiny(mode="ffn", base_width=64, head_dim=16, ffn_hidden=16, supported=[1, 2, 4])
    model = DenseElasticLM(cfg)
    assert model.blocks[0].attn.qkv.weight_init.shape == (3 * 64, 64)
    assert model.blocks[0].mlp.w1.weight_init.shape == (4 * 16, 64)
    assert model.alpha.shape == (1,)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    outs = model(ids, targets=ids, loss_fn=LinearCrossEntropy())
    assert all(o.hidden.shape[-1] == 64 for o in outs.values())
    spec = model.optim_spec()
    assert spec["readout.weight"] == (model.m, True)
    assert "blocks.0.mlp.w1.weight_init" not in spec


def test_optim_spec_and_frozen_snapshots():
    model = DenseElasticLM(tiny())
    spec = model.optim_spec()
    m = 16 / 32
    assert spec["blocks.0.attn.qkv.weight_delta"] == (m, True)
    assert spec["readout_delta"] == (m, True)
    assert not model.readout.weight.requires_grad
    assert all(not p.requires_grad for n, p in model.named_parameters() if n.endswith("weight_init"))


def test_paper_parameter_counts():
    with torch.device("meta"):
        full = DenseElasticLM(DenseElasticConfig())
        matformer = DenseElasticLM(DenseElasticConfig(mode="ffn", base_width=512, head_dim=128, ffn_hidden=128,
                                                      supported=[4, 9, 15, 22, 30]))
        standalone = ShellElasticLM(ShellConfig(base_width=384, supported=[1], head_dim=64, ffn_hidden=1536,
                                                alpha=1.0, fused_norm=False))
    assert [round(full.num_params(s) / 1e6, 1) for s in full.supported] == [119.0, 154.4, 194.3, 238.7, 287.5]
    assert [round(matformer.num_params(s) / 1e6, 1) for s in matformer.supported] == [
        118.8, 150.3, 188.0, 232.1, 282.4]
    assert round(standalone.num_params(1) / 1e6, 1) == 119.0
