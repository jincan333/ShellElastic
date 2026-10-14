"""Expand a sweep file (configs/paper/sweeps.yaml) into training commands, one per line.

    python scripts/sweep.py configs/paper/sweeps.yaml [--family NAME ...] [--grid lr|init|wd ...]
                            [--sp | --no-sp] [--nproc 8]

A run is a family's base config plus one grid value; with ``--sp`` the family's sweep without
muElastic (``sp_overrides`` added). The commands are printed, not executed, so they can be sent to
any scheduler (e.g. ``python scripts/sweep.py ... | xargs -P 1 -I{} sh -c "{}"``).
"""

from __future__ import annotations

import argparse
import os

import yaml


def fmt(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return f"{value:g}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_file")
    parser.add_argument("--family", nargs="*", help="families to expand (default: all)")
    parser.add_argument("--grid", nargs="*", help="grids to expand (default: every grid of a family)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sp", dest="sp", action="store_true", default=None, help="only the sweeps without muElastic")
    group.add_argument("--no-sp", dest="sp", action="store_false", help="only the muElastic sweeps")
    parser.add_argument("--nproc", type=int, default=8, help="GPUs per run (torchrun --nproc_per_node)")
    args = parser.parse_args()

    with open(args.sweep_file) as f:
        spec = yaml.safe_load(f)
    root = os.path.dirname(os.path.abspath(args.sweep_file))
    for family, fam in spec["families"].items():
        if args.family and family not in args.family:
            continue
        base = os.path.relpath(os.path.join(root, fam["base"]))
        variants = [False, True] if fam.get("sp") else [False]
        if args.sp is not None:
            variants = [v for v in variants if v == args.sp]
        for sp in variants:
            extra = [f"{k}={fmt(v)}" for k, v in spec["sp_overrides"].items()] if sp else []
            for grid in fam["grids"]:
                if args.grid and grid not in args.grid:
                    continue
                key = spec["grids"][grid]["key"]
                values = spec["grids"][grid]["values"] + fam.get("extra_values", {}).get(grid, [])
                for value in values:
                    name = f"sweep_{family}_{'sp_' if sp else ''}{grid}{fmt(value)}"
                    overrides = " ".join(extra + [f"{key}={fmt(value)}", f"run.name={name}"])
                    print(f"torchrun --nproc_per_node={args.nproc} -m elastic.train {base} {overrides}")


if __name__ == "__main__":
    main()
