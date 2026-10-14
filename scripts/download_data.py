"""Download the tokenized pretraining data of the paper from the Hugging Face Hub.

    python scripts/download_data.py --scale 1                # the shards the 119M-285M runs read (20 GB)
    python scripts/download_data.py --scale 2                # 375M-906M (72 GB); --scale 3: 675M-1.69B (141 GB)
    python scripts/download_data.py --scale all              # every training shard of the repository
    python scripts/download_data.py --val-only               # only the C4 validation shards (1.1 GB)

The dataset repository (``Can111/ShellElastic-data``) holds the DCLM-Baseline training shards and the
held-out C4 validation shards of Section 4.1, tokenized with the GPT-NeoX-20B tokenizer and packed into
sequences of 2,049 tokens, in the layout the data loader reads: ``<out>/dclm-baseline`` (shards and the
full ``manifest.jsonl``, whose seeded shuffle fixes the reading order) and ``<out>/c4-val``, the paths of
``configs/paper/common.yaml``. ``shards.tsv`` lists the training shards in reading order with the
sequences each scale reads; with the paper's settings (8 GPUs with 8 data-loader workers each, data
seed 1) a scale reads only the shards selected here. The loader can also stream the shards without a
download: ``data.train_sources=[hf://datasets/Can111/ShellElastic-data/dclm-baseline]`` and
``data.val_sources=[hf://datasets/Can111/ShellElastic-data/c4-val]``.
"""

from __future__ import annotations

import argparse
import csv

REPO = "Can111/ShellElastic-data"
SCALES = {"1": "read_119M-285M", "2": "read_375M-906M", "3": "read_675M-1.69B"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    parser.add_argument("--scale", choices=[*SCALES, "all"], default="1",
                        help="training shards to download: those read at a model scale, or all (default: 1)")
    parser.add_argument("--val-only", action="store_true", help="download only the validation shards")
    parser.add_argument("--repo", default=REPO, help=f"dataset repository (default: {REPO})")
    parser.add_argument("--revision", default=None, help="branch, tag or commit of the repository")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    patterns = ["c4-val/*"]
    if not args.val_only:
        patterns.append("dclm-baseline/manifest.jsonl")
        if args.scale == "all":
            patterns.append("dclm-baseline/*.tar")
        else:
            table = hf_hub_download(args.repo, "shards.tsv", repo_type="dataset", revision=args.revision,
                                    local_dir=args.out)
            with open(table, newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            patterns += [f"dclm-baseline/{r['shard']}.tar" for r in rows if int(r[SCALES[args.scale]]) > 0]
    snapshot_download(args.repo, repo_type="dataset", revision=args.revision, local_dir=args.out,
                      allow_patterns=patterns, max_workers=args.workers)
    print(f"downloaded {args.repo} to {args.out}")


if __name__ == "__main__":
    main()
