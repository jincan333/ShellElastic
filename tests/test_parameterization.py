"""mu-Elastic (Table 1, Appendix D): configuration consistency and a coordinate check of Theorem 1.

The coordinate check measures, for every hidden matrix and readout of small models, the RMS of its
pre-activations ``y = x W^T`` at initialization (``x``: the matrix's actual input in the forward pass)
and the RMS of ``dy = x dW^T`` after one AdamW step, with the forward multipliers included. Under
mu-Elastic both stay constant when the fan-in grows with the base width ``g = m d0``, with ``M`` or
with a group's boundary ``s`` (Theorem 1, Corollary 1); under SP they grow like ``fan-in^1/2`` and
``fan-in``. The test fits the log-log slope of each quantity against the fan-in, per matrix kind.
"""

import copy
import glob
import math
import os
from collections import defaultdict

import pytest
import torch
import torch.nn.functional as F

from elastic.config import load_config, model_config
from elastic.models import DenseElasticConfig, DenseElasticLM, ShellConfig, ShellElasticLM
from elastic.models.dense import DenseMLP, SplitLinear
from elastic.models.shell import ShellLinear, ShellMLP
from elastic.optim import OptimConfig, build_adamw
from elastic.train.losses import LinearCrossEntropy

SIGMA, ETA, LAM, D0 = 3.3e-3, 2e-3, 0.01, 32
SP = dict(internal_scaling=False, alpha=1.0, width_multiplier=1.0, ffn_expansion=1.0)
V, T, B = 256, 32, 4
BASE = dict(vocab_size=V, seq_len=T, n_layers=2, head_dim=16, proxy_width=D0, init_std=SIGMA)


def _batch():
    gen = torch.Generator().manual_seed(1)
    return torch.randint(0, V, (B, T), generator=gen), torch.randint(0, V, (B, T), generator=gen)


def _rms(t):
    return float(t.detach().pow(2).mean().sqrt())


def _capture(model, run):
    """Inputs of every shell/split projection call and (input, output) of every FFN call during ``run()``."""
    calls, ffn = [], []
    hooks = [m.register_forward_pre_hook(lambda mod, a: calls.append((mod, a[0].detach().clone(), tuple(a[1:]))))
             for m in model.modules() if isinstance(m, (ShellLinear, SplitLinear))]
    hooks += [m.register_forward_hook(lambda mod, a, out: ffn.append((mod, a[0].detach().clone(), tuple(a[1:]),
                                                                       out.detach().clone())))
              for m in model.modules() if isinstance(m, (ShellMLP, DenseMLP))]
    try:
        out = run()
    finally:
        for h in hooks:
            h.remove()
    return calls, ffn, out


def _ffn_rows(model, ffn, before):
    """FFN down projection, measured on the FFN output (forward multiplier included) per output group.

    The update keeps the up and gate projections at their initial values, so it is the change that the
    down projection's own update causes.
    """
    names = {m: n for n, m in model.named_modules()}
    rows = []
    for mod, x, args, y0 in ffn:
        local = dict(mod.named_parameters())
        current = {n: p.detach().clone() for n, p in local.items() if not n.startswith("w2.")}
        with torch.no_grad():
            for n in current:
                local[n].copy_(before[f"{names[mod]}.{n}"])
            dy = mod(x, *args) - y0
            for n, v in current.items():
                local[n].copy_(v)
        down = mod.w2
        if isinstance(down, ShellLinear):
            edges = [0, *down.boundaries]
            for k in range(down.num_groups(args[0])):
                cols = slice(edges[k] * down.out_unit, edges[k + 1] * down.out_unit)
                rows.append(("mlp.w2", edges[k + 1] * down.in_unit, _rms(y0[..., cols]), _rms(dy[..., cols])))
        else:
            rows.append(("mlp.w2", args[1] * down.in_unit, _rms(y0), _rms(dy)))
    return rows


def _adamw_step(model):
    build_adamw(model, OptimConfig(lr=ETA, weight_decay=LAM, fused=False, eps=1e-30)).step()


def _shell_rows(**kw):
    """(kind, fan-in, init RMS, update RMS) of every group and readout after one joint step of all widths."""
    torch.manual_seed(0)
    model = ShellElasticLM(ShellConfig(**BASE, **kw)).double()
    names = {m: n for n, m in model.named_modules()}
    ids, tgt = _batch()
    calls, ffn, outs = _capture(model, lambda: model(ids, targets=tgt, loss_fn=LinearCrossEntropy(use_fused=False)))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    (sum(o.loss for o in outs.values()) / len(outs)).backward()
    _adamw_step(model)
    params = dict(model.named_parameters())
    rows = _ffn_rows(model, ffn, before)
    for mod, x, (s,) in calls:
        kind = names[mod].split(".", 2)[2]
        if kind == "mlp.w2":
            continue
        for k in range(mod.num_groups(s)):
            name = f"{names[mod]}.weight_{k}"
            xin = x[..., : mod.boundaries[k] * mod.in_unit]  # a group reads the same prefix in every sub-network
            rows.append((kind, xin.shape[-1], _rms(F.linear(xin, before[name])),
                         _rms(F.linear(xin, params[name].detach() - before[name]))))
    after = {s: model.readout_weight(s).detach().clone() for s in outs}
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(before[n])
    for s, out in outs.items():
        with torch.no_grad():
            w0, h, mult = model.readout_weight(s), out.hidden, float(model.readout_multiplier(s))
        rows.append(("readout", h.shape[-1], _rms(F.linear(h, w0)) * mult, _rms(F.linear(h, after[s] - w0)) * mult))
    return rows


def _dense_rows(**kw):
    """Per sub-network t, one AdamW step on t's loss (one sub-network per step) and t's own matrices."""
    torch.manual_seed(0)
    base = DenseElasticLM(DenseElasticConfig(**BASE, **kw)).double()
    ids, tgt = _batch()
    rows = []
    for t in base.supported:
        model = copy.deepcopy(base)
        names = {m: n for n, m in model.named_modules()}
        calls, ffn, outs = _capture(model, lambda: model(ids, widths=[t], targets=tgt,
                                                         loss_fn=LinearCrossEntropy(use_fused=False)))
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        w0 = [mod.weight(*a).detach().clone() for mod, _, a in calls]
        r = model.residual_ratio(t)
        with torch.no_grad():
            h, u0, mult = outs[t].hidden.clone(), model.readout_weight(r).clone(), float(model.readout_multiplier(t))
        outs[t].loss.backward()
        _adamw_step(model)
        rows += _ffn_rows(model, ffn, before)
        for (mod, x, a), w in zip(calls, w0):
            kind = names[mod].split(".", 2)[2]
            if kind != "mlp.w2":
                rows.append((kind, x.shape[-1], _rms(F.linear(x, w)), _rms(F.linear(x, mod.weight(*a).detach() - w))))
        rows.append(("readout", h.shape[-1], _rms(F.linear(h, u0)) * mult,
                     _rms(F.linear(h, model.readout_weight(r).detach() - u0)) * mult))
    return rows


def _slopes(rows):
    """Per matrix kind, the least-squares slopes of log(init RMS) and log(update RMS) against log(fan-in)."""
    by = defaultdict(list)
    for kind, fan_in, y0, dy in rows:
        by[kind].append((math.log(fan_in), math.log(y0), math.log(dy)))
    out = {}
    for kind, pts in by.items():
        mx = sum(p[0] for p in pts) / len(pts)
        den = sum((p[0] - mx) ** 2 for p in pts)
        fit = [sum((p[0] - mx) * p[j] for p in pts) / den for j in (1, 2)]
        out[kind] = tuple(fit)
    return out


def _family_rows(family, sp):
    """Models of one family whose fan-ins grow with m, M, the group boundaries and the FFN expansion e."""
    extra = SP if sp else {}
    if family == "shellelastic":  # the proxy; g = 32, 64, 128 (m = 1, 2, 4) at e = 2; e = 1, 4; S = {4,...,8}
        rows = _shell_rows(**{**dict(base_width=D0, supported=[1], ffn_hidden=2 * D0, alpha=1.0), **extra})
        for g, e in ((32, 2), (64, 2), (128, 2), (64, 1), (64, 4)):
            rows += _shell_rows(base_width=g, supported=[2, 3, 4], ffn_hidden=e * g, **extra)
        rows += _shell_rows(base_width=32, supported=[1, 2, 3, 5], ffn_hidden=64, **extra)  # a wider outer shell
        return rows + _shell_rows(base_width=32, supported=[4, 5, 6, 7, 8], ffn_hidden=64, **extra)
    if family.startswith("stairformer"):  # equal groups, cumulative norm, one shared readout
        scaling = family.split("-")[1]
        return [r for g, e in ((32, 2), (64, 2), (128, 2), (64, 4)) for r in _shell_rows(
            base_width=g, supported=[1, 2, 3, 4], ffn_hidden=e * g, readout="shared", separate_terminal_norms=False,
            trunk_norm="cumulative", readout_scaling=scaling, **extra)]
    if family == "fully_elastic":  # g = 16, 32 (m = 1/2, 1) at e = 4; e = 2
        return [r for g, e in ((16, 4), (32, 4), (32, 2)) for r in _dense_rows(
            mode="full", base_width=g, supported=[2, 3, 4], ffn_hidden=e * g, **extra)]
    # MatFormer: D_model = 32, 64, 128 (m = 1, 2, 4) with base FFN width h0 = D_model / 4; h0 = D_model / 2
    return [r for d, h0 in ((32, 8), (64, 16), (128, 32), (64, 32)) for r in _dense_rows(
        mode="ffn", base_width=d, supported=[1, 2, 4], ffn_hidden=h0, **extra)]


FAMILIES = ["shellelastic", "stairformer-init_lr", "stairformer-multiplier", "fully_elastic", "matformer"]


@pytest.mark.parametrize("family", FAMILIES)
def test_coordinate_check(family):
    """mu-Elastic: width-independent initial pre-activations and update sizes; SP: both grow with the fan-in."""
    for kind, (init, update) in _slopes(_family_rows(family, sp=False)).items():
        if family == "stairformer-multiplier" and kind == "readout":
            # Logit multiplier alpha_s / (s m) on an unscaled readout: the update follows Table 1, but the initial
            # logits shrink like fan-in^-1/2 instead of staying constant.
            assert init == pytest.approx(-0.5, abs=0.1), (kind, init)
        else:
            assert abs(init) < 0.1, (family, kind, init)
        # With 128 tokens and fan-ins of 32-512, part of the first Adam update is not aligned with the input and
        # decays like fan-in^-1/2, so the slopes lie between -1/2 and 0 (they approach 0 with a few tokens).
        assert -0.5 < update < 0.15, (family, kind, update)
    for kind, (init, update) in _slopes(_family_rows(family, sp=True)).items():
        assert init > 0.4 and update > 0.6, (family, kind, init, update)


def test_paper_configs_follow_mu_elastic():
    """Every model of configs/paper uses m = g / d0, e = h / g, the per-group factor and the base values."""
    checked = 0
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "paper")
    for path in sorted(glob.glob(os.path.join(root, "**", "*.yaml"), recursive=True)):
        if path.endswith(("common.yaml", "sweeps.yaml")):
            continue
        cfg = load_config(path)
        specs = [dict(cfg.model)]
        if cfg.multi is not None and cfg.multi.models:
            specs = [{**cfg.model, **e.model} for e in cfg.multi.models]
        for spec in specs:
            mc = model_config(spec)
            assert mc.parameterization == "mu_elastic" and mc.internal_scaling is not False, path
            assert mc.width_multiplier is None or mc.width_multiplier == pytest.approx(mc.base_width / mc.proxy_width), path
            assert mc.ffn_expansion is None or mc.ffn_expansion == pytest.approx(mc.ffn_hidden / mc.base_width), path
            assert (mc.proxy_width, mc.init_std) == (128, 3.3e-3), path
            assert mc.alpha == ("learnable" if len(mc.supported) > 1 else 1.0), path
            if isinstance(mc, ShellConfig):
                assert mc.readout_scaling == "init_lr", path
            checked += 1
    assert checked > 100
