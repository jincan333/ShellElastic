"""Numerical verification of Proposition 1 (exact reuse at supported widths), as in Appendix B.3.

    python tools/check_prefix_exactness.py [configs/paper/scale1/shellelastic.yaml] [--layers 2] [--seq 64]

Builds a reduced copy of the configuration (fewer layers, short sequences, CPU, float64, reference
kernels) and compares one shared trunk pass, whose width-s outputs are read as prefixes, with |S|
separate evaluations of the same nested weights: the logits, the weighted cross-entropy and every
parameter gradient. The shell projections use one GEMM per output group, so that both passes reduce
over the same inputs; what remains is float64 roundoff (a strided prefix slice and a contiguous
tensor can be summed in different orders), so the differences it prints are about 1e-15 and it
passes at a tolerance of 1e-12. It also instantiates the full configuration on the meta device to
check that the module shapes build.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from elastic.config import load_config, model_config
from elastic.models import ShellConfig, ShellElasticLM


def weighted_ce(model, outs: dict, targets: torch.Tensor, weights: dict[int, float]) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    for s, out in outs.items():
        logits = model.logits(out.hidden, s)
        total = total + weights[s] * F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="configs/paper/scale1/shellelastic.yaml")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    spec = dict(cfg.model)
    full = model_config(spec)
    if not isinstance(full, ShellConfig):
        raise SystemExit("Proposition 1 concerns ShellElastic models (type: shell)")
    with torch.device("meta"):
        ShellElasticLM(full)
    print(f"full configuration builds on the meta device ({full.n_layers} layers, widths "
          f"{[s * full.base_width for s in full.supported]})")

    spec.update(n_layers=args.layers, seq_len=args.seq, matmul="groups", fused_norm=False)
    torch.manual_seed(args.seed)
    model = ShellElasticLM(model_config(spec)).double()
    gen = torch.Generator().manual_seed(args.seed + 1)
    with torch.no_grad():  # move every weight off its structured initialization (e.g. zero deltas, unit gains)
        for p in model.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    ids = torch.randint(0, spec["vocab_size"], (args.batch, args.seq + 1), generator=gen)
    inputs, targets = ids[:, :-1], ids[:, 1:]
    trained = list(model.trained)
    lw = cfg.objective.loss_weights or [1.0] * len(model.supported)
    weights = {s: float(w) for s, w in zip(model.supported, lw) if s in trained}

    shared = model(inputs, widths=trained)
    loss_shared = weighted_ce(model, shared, targets, weights)
    model.zero_grad(set_to_none=True)
    loss_shared.backward()
    grads_shared = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    model.zero_grad(set_to_none=True)
    separate = {s: model(inputs, widths=[s])[s] for s in trained}
    loss_separate = weighted_ce(model, separate, targets, weights)
    loss_separate.backward()
    grads_separate = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    logit_diff = max(float((model.logits(shared[s].hidden, s) - model.logits(separate[s].hidden, s)).abs().max())
                     for s in trained)
    grad_diff = max(float((grads_shared[n] - grads_separate[n]).abs().max()) for n in grads_shared)
    # The optional width router is trained by its own objective term, outside the proposition.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad and not n.startswith("width_router.")}
    missing = sorted(trainable - set(grads_shared))
    print(f"widths {[s * model.g for s in trained]}: max |logit difference| {logit_diff:.3e}, "
          f"|loss difference| {abs(float(loss_shared - loss_separate)):.3e}, max |gradient difference| {grad_diff:.3e}")
    print("every trainable parameter received a gradient" if not missing else f"no gradient: {missing}")
    tol = 1e-12
    if set(grads_shared) != set(grads_separate) or logit_diff > tol or grad_diff > tol or missing:
        raise SystemExit(f"prefix exactness check FAILED (tolerance {tol:g})")
    print(f"prefix exactness check passed (float64, tolerance {tol:g})")


if __name__ == "__main__":
    main()
