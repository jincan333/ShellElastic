"""Parameters of every sub-network of one or more configs (the counting rule of Appendix E.3).

    python tools/count_params.py configs/paper/scale1/shellelastic.yaml [more configs ...]

A sub-network counts the non-zero weights it reads, its embedding columns and its own terminal norm
and readout; the two tensors of a frozen-plus-delta split count once. The model is built on the meta
device, so no memory is allocated. A sampled-standalone config lists each of its models.
"""

from __future__ import annotations

import argparse

import torch

from elastic.config import load_config, model_config
from elastic.models import DenseElasticConfig, DenseElasticLM, ShellElasticLM


def build(spec: dict):
    cfg = model_config(spec)
    with torch.device("meta"):
        return DenseElasticLM(cfg) if isinstance(cfg, DenseElasticConfig) else ShellElasticLM(cfg)


def width_label(model, s: int) -> str:
    if getattr(model, "ffn_only", False):
        return f"d_ff={s * model.cfg.ffn_hidden}"
    return f"d={s * model.g}"


def report(name: str, spec: dict) -> list[tuple[str, int]]:
    model = build(spec)
    return [(f"{name} {width_label(model, s)}", model.num_params(s)) for s in model.trained]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    for path in args.configs:
        cfg = load_config(path)
        specs = [(cfg.run.name, cfg.model)]
        if cfg.multi is not None and cfg.multi.models:
            specs = [(f"{cfg.run.name}/{e.name or i}", {**cfg.model, **e.model}) for i, e in enumerate(cfg.multi.models)]
        rows = [row for name, spec in specs for row in report(name, spec)]
        for label, count in rows:
            print(f"{label:60s} {count / 1e6:9.1f}M")
        if len(rows) > 1:
            print(f"{cfg.run.name + ' (mean)':60s} {sum(c for _, c in rows) / len(rows) / 1e6:9.1f}M")


if __name__ == "__main__":
    main()
