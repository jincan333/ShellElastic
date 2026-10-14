"""Evaluate exported weights on the downstream in-context-learning tasks (Appendix F, Table 12).

    torchrun --nproc_per_node=8 -m elastic.eval outputs/<name>/final [key=value ...]

``outputs/<name>/final`` holds ``model.safetensors`` and ``model.json`` (written at the end of
training; a sampled standalone run writes one directory per model, ``final_0`` ... ``final_4``).
Keys: the fields of ``EvalConfig`` (``tasks``, ``data_dir``, ``tokenizer``, ``batch_size``,
``max_seq_len``, ``widths``) and ``out`` (a JSON file for the results). ``widths`` lists
sub-networks by their entry of the supported set (``supported`` in ``model.json``, in units of the
base width, or of the base FFN width for MatFormer); by default every trained sub-network is
evaluated. Each rank holds the full model with bfloat16 parameters, as sharded training evaluates it.
"""

from __future__ import annotations

import json
import os
import sys

import torch
from omegaconf import OmegaConf

from elastic.eval.config import EvalConfig


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: torchrun ... -m elastic.eval <weights_dir> [key=value ...]")
    from elastic.eval.icl import ICLEvaluator
    from elastic.train.distributed import is_main, setup, teardown
    from elastic.train.trainer import build_model

    path = argv[0]
    overrides = OmegaConf.to_container(OmegaConf.from_dotlist(argv[1:]), resolve=True)
    out = overrides.pop("out", None)
    cfg: EvalConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(EvalConfig), overrides))
    _, _, device = setup()
    with open(os.path.join(path, "model.json")) as f:
        spec = json.load(f)
    model = build_model(spec, 0, device, init_from=path)
    if cfg.widths is not None:
        unknown = [s for s in cfg.widths if int(s) not in model.supported]
        if unknown:
            raise SystemExit(f"widths {unknown} are not in the supported set {model.supported} of this model "
                             "(widths are entries of `supported` in model.json, not hidden sizes)")
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    results = ICLEvaluator(cfg, cfg.tasks)([model], device)
    if is_main():
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        if out:
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
    teardown()


if __name__ == "__main__":
    main()
