import math

import pytest
import torch

from elastic.models import ShellConfig, ShellElasticLM
from elastic.optim import OptimConfig, build_adamw, param_groups, wsd_schedule
from elastic.train.losses import LinearCrossEntropy

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def tiny(**overrides) -> ShellConfig:
    cfg = dict(
        vocab_size=256,
        seq_len=32,
        n_layers=2,
        base_width=32,
        supported=[2, 3, 4],
        head_dim=16,
        ffn_hidden=32,
        proxy_width=64,
        fused_norm=False,
    )
    cfg.update(overrides)
    return ShellConfig(**cfg)


def _ids(cfg, batch=2, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (batch, cfg.seq_len), generator=gen)


def _perturb(model, seed=1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 1 or p is model.readout_delta:
                p.add_(0.05 * torch.randn(p.shape, generator=gen).to(p))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"trunk_norm": "block"},
        {"trunk_norm": "cumulative", "terminal_norm": "trunk", "readout": "shared", "supported": [1, 2, 3, 4]},
        {"norm": "layer", "readout": "shared"},
        {"parameterization": "sp"},
    ],
)
def test_prefix_exactness(overrides):
    """A width-s sub-network run on its own equals the first s*g features of the full pass."""
    torch.manual_seed(0)
    cfg = tiny(**overrides)
    model = ShellElasticLM(cfg).double()
    _perturb(model)
    ids = _ids(cfg)
    full = model.trunk(ids, model.M)
    for s in model.supported:
        alone = model.trunk(ids, s)
        torch.testing.assert_close(full[..., : s * cfg.base_width], alone, rtol=1e-10, atol=1e-10)
    outs_joint = model(ids)
    for s in model.supported:
        alone = model(ids, widths=[s])[s]
        torch.testing.assert_close(outs_joint[s].hidden, alone.hidden, rtol=1e-10, atol=1e-10)


def test_block_lower_triangular():
    """Group k of every hidden matrix reads only the first s_k*g inputs."""
    cfg = tiny()
    model = ShellElasticLM(cfg)
    lin = model.blocks[0].mlp.w1
    dense = lin.dense_weight(model.M)
    g, h = cfg.base_width, cfg.ffn_hidden
    lo = 0
    for s in cfg.supported:
        rows = dense[lo * h : s * h]
        assert torch.count_nonzero(rows[:, s * g :]) == 0
        lo = s


def test_matmul_modes_agree():
    torch.manual_seed(0)
    cfg = tiny()
    ref = ShellElasticLM(cfg).double()
    _perturb(ref)
    ids = _ids(cfg)
    base = ref(ids)
    for mode in ("groups",):
        other = ShellElasticLM(tiny(matmul=mode)).double()
        other.load_state_dict(ref.state_dict())
        outs = other(ids)
        for s in cfg.supported:
            torch.testing.assert_close(outs[s].hidden, base[s].hidden, rtol=1e-10, atol=1e-10)


@cuda
def test_slab_kernel_matches_padded_gemm():
    """The slab GEMMs equal the padded block-lower-triangular GEMM (forward and backward)."""
    from elastic.kernels.slab_linear import slab_linear, slab_partition, slab_supported
    from elastic.models.shell import ShellLinear

    torch.manual_seed(0)
    for in_unit, out_unit in ((128, 384), (128, 128)):
        layer = ShellLinear(in_unit, out_unit, (4, 5, 6, 7, 8), 0.02, True, "pad").cuda().to(torch.bfloat16)
        x = torch.randn(4096, 8 * in_unit, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weights = tuple(layer.group_weight(k) for k in range(5))
        units = tuple(r // out_unit for r in layer.group_rows)
        slabs = slab_partition(tuple(layer.group_rows), layer.boundaries, out_unit, units)
        y = slab_linear(x, weights, slabs)
        ref = layer(x, 8)
        torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)
        grad = torch.randn_like(y)
        got = torch.autograd.grad(y, (x, *weights), grad)
        want = torch.autograd.grad(ref, (x, *weights), grad)
        for a, b in zip(got, want):
            assert float((a.float() - b.float()).norm() / b.float().norm()) < 1e-2
        # The slab path is taken only for the training shape (32768 token rows).
        reads = tuple(layer.boundaries[k] * in_unit for k in range(5) for _ in range(units[k]))
        assert not slab_supported(x.detach(), weights, reads, out_unit)


@cuda
def test_slab_and_fused_norm_match_reference():
    torch.manual_seed(0)
    cfg = tiny(base_width=128, head_dim=128, ffn_hidden=128, proxy_width=128, supported=[4, 5, 6, 7, 8])
    ref = ShellElasticLM(cfg).cuda()
    _perturb(ref)
    fast = ShellElasticLM(tiny(base_width=128, head_dim=128, ffn_hidden=128, proxy_width=128,
                               supported=[4, 5, 6, 7, 8], matmul="slab", fused_norm=True)).cuda()
    fast.load_state_dict(ref.state_dict())
    ids = _ids(cfg).cuda()
    loss_fn = LinearCrossEntropy(z_loss=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = ref(ids, targets=ids, loss_fn=loss_fn)
        b = fast(ids, targets=ids, loss_fn=loss_fn)
    for s in cfg.supported:
        assert abs(float(a[s].loss.detach()) - float(b[s].loss.detach())) < 2e-2
    sum(o.loss for o in a.values()).backward()
    sum(o.loss for o in b.values()).backward()
    for (n, p), q in zip(ref.named_parameters(), fast.parameters()):
        if p.grad is None:
            continue
        rel = (p.grad - q.grad).norm() / p.grad.norm().clamp(min=1e-12)
        assert rel < 5e-2, (n, float(rel))


def test_optimizer_groups_follow_table_1():
    cfg = tiny(supported=[2, 3, 4], base_width=32, proxy_width=16)
    model = ShellElasticLM(cfg)
    m = 32 / 16
    groups = param_groups(model, lr=1e-3, weight_decay=0.1)
    got = {n: (g["lr"], g["weight_decay"]) for g in groups for n in g["param_names"]}
    for k, s in enumerate(cfg.supported):
        assert got[f"blocks.0.attn.qkv.weight_{k}"] == (1e-3 / (s * m), 0.1 * (s * m))
    assert got["readouts.2.weight"] == (1e-3 / (2 * m), 0.1 * (2 * m))
    assert got["readout.weight"] == (1e-3 / (4 * m), 0.1 * (4 * m))
    assert got["embed.weight"] == (1e-3, 0.1)
    assert got["alpha"] == (1e-3, 0.0)
    assert got["blocks.0.attn_norm.weight"] == (1e-3, 0.0)
    # Under SP every hidden weight uses the base learning rate and weight decay.
    sp = ShellElasticLM(tiny(parameterization="sp")).optim_spec()
    assert sp["blocks.0.mlp.w2.weight_2"] == (1.0, True)


def test_wsd_schedule():
    model = ShellElasticLM(tiny())
    opt = build_adamw(model, OptimConfig(lr=1.0, fused=False))
    sched = wsd_schedule(opt, warmup_steps=10, decay_start=20, total_steps=30)
    lrs = []
    for _ in range(31):
        lrs.append(opt.param_groups[0]["lr"] / opt.param_groups[0]["initial_lr"])
        opt.step()
        sched.step()
    assert lrs[0] == 0.0 and lrs[5] == 0.5 and lrs[10] == 1.0 and lrs[19] == 1.0
    assert lrs[25] == pytest.approx(0.5) and lrs[30] == 0.0


def test_init_std_follows_table_1():
    torch.manual_seed(0)
    cfg = tiny(supported=[2, 3, 4], base_width=64, head_dim=16, ffn_hidden=256, proxy_width=16,
               init_std=0.02, n_layers=1)
    model = ShellElasticLM(cfg)
    m, e = 64 / 16, 256 / 64
    for k, s in enumerate(cfg.supported):
        w = model.blocks[0].mlp.w1.group_weight(k)
        assert w.std().item() == pytest.approx(0.02 / math.sqrt(m * s), rel=0.1)
        w2 = model.blocks[0].mlp.w2.group_weight(k)
        assert w2.std().item() == pytest.approx(0.02 * math.sqrt(e) / math.sqrt(m * s), rel=0.1)


def test_paper_parameter_counts():
    with torch.device("meta"):
        model = ShellElasticLM(ShellConfig(fused_norm=False))
    counts = {s: model.num_params(s) for s in model.supported}
    assert counts[4] / 1e6 == pytest.approx(118.8, abs=0.05)
    assert counts[8] / 1e6 == pytest.approx(279.6, abs=0.05)


def test_trunk_grad_widths_detach():
    torch.manual_seed(0)
    cfg = tiny()
    model = ShellElasticLM(cfg)
    ids = _ids(cfg)
    outs = model(ids, targets=ids, loss_fn=LinearCrossEntropy(), trunk_grad_widths={4})
    outs[2].loss.backward()
    assert model.blocks[0].attn.qkv.weight_0.grad is None
    assert model.readouts["2"].weight.grad is not None
