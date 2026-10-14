"""Download the downstream evaluation data (the 13 in-context-learning task files of the paper, Table 12).

    python scripts/download_eval_data.py [--out eval_data]

The task files are those distributed with LLM Foundry v0.22.0 (``scripts/eval/local_data``,
Apache-2.0), except ARC-Easy, ARC-Challenge, BoolQ and MMLU, which are OpenLM's copies
(``eval/local_data``, MIT). Each file keeps the license of its original dataset. Checksums pin
the exact versions.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request

LLM_FOUNDRY = "https://raw.githubusercontent.com/mosaicml/llm-foundry/v0.22.0/scripts/eval/local_data/"
OPEN_LM_COMMIT = "9bb92ef1689333534b7057942a20d18a46d1fa52"
OPEN_LM = f"https://raw.githubusercontent.com/mlfoundations/open_lm/{OPEN_LM_COMMIT}/eval/local_data/"
FILES = {  # local path: (url, sha256)
    "commonsense_reasoning/copa.jsonl": (
        LLM_FOUNDRY + "commonsense_reasoning/copa.jsonl",
        "595dca44dec54c849490bd403a3e24a29e3b944065783222ebb0eb1a55298846"),
    "commonsense_reasoning/piqa.jsonl": (
        LLM_FOUNDRY + "commonsense_reasoning/piqa.jsonl",
        "d2870e73aec5154df3faf259b7700876e11608eebdf69e9d6667435673dbe4e0"),
    "language_understanding/hellaswag.jsonl": (
        LLM_FOUNDRY + "language_understanding/hellaswag.jsonl",
        "52a31d32b14dff7713a705be664cbeff77eb66edcf7f8b0a5e9408633929b448"),
    "language_understanding/lambada_openai.jsonl": (
        LLM_FOUNDRY + "language_understanding/lambada_openai.jsonl",
        "411bfbe5b9054346b6a19299c0242670adbefb9162efd958593cbc0290b8c9dc"),
    "language_understanding/winogrande.jsonl": (
        LLM_FOUNDRY + "language_understanding/winogrande.jsonl",
        "ea1592526f3bfdd0948ed3acb1b4dfa3f83d23698917118688ee1df7f8d4b8d7"),
    "reading_comprehension/agi_eval_lsat_lr.jsonl": (
        LLM_FOUNDRY + "reading_comprehension/agi_eval_lsat_lr.jsonl",
        "e4a273cf9d7013b48e8cabceb54a0bf3434efa6f153a884e71ba0df5dc62d29e"),
    "reading_comprehension/agi_eval_lsat_rc.jsonl": (
        LLM_FOUNDRY + "reading_comprehension/agi_eval_lsat_rc.jsonl",
        "8f67ecf0e760a8fc737ead2e63554bbf921b23989a3bc0039d9b2e993ef6295a"),
    "reading_comprehension/agi_eval_sat_en.jsonl": (
        LLM_FOUNDRY + "reading_comprehension/agi_eval_sat_en.jsonl",
        "cb78f595cd4cbd226624d58c1a48f340ea9ad4d0a2b5b32126adfc15014170ea"),
    "reading_comprehension/boolq.jsonl": (
        OPEN_LM + "boolq.jsonl",
        "9e29112df7e50ce6a35c1e2258642bfd755e36b0c5dd507744c7f329fda255b4"),
    "symbolic_problem_solving/svamp.jsonl": (
        LLM_FOUNDRY + "symbolic_problem_solving/svamp.jsonl",
        "6e9d75302c2663a8c8ce48927d784244480bc75cd0eda539677b6a92a2ff5b97"),
    "world_knowledge/arc_challenge.jsonl": (
        OPEN_LM + "arc_challenge.jsonl",
        "f44aee5cf820df030ef02b259471298fbd91ac63eaec27199e8d905c32a0a35e"),
    "world_knowledge/arc_easy.jsonl": (
        OPEN_LM + "arc_easy.jsonl",
        "8b65cc2de8bf9ef1d5a70696ea8370a1a9188e3c4b0980f3d19126371d71a01c"),
    "world_knowledge/mmlu.jsonl": (
        OPEN_LM + "mmlu.jsonl",
        "025f07c58bc35f02a3d60258b28977f18f658f7a9e8b0b445c9ce2fb9b9b610a"),
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="eval_data", help="output directory (eval.data_dir)")
    args = parser.parse_args()
    for name, (url, expected) in FILES.items():
        path = os.path.join(args.out, name)
        if os.path.exists(path) and sha256(path) == expected:
            print(f"ok       {name}")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        actual = sha256(path + ".part")
        if actual != expected:
            os.remove(path + ".part")
            raise SystemExit(f"checksum mismatch for {name}: {actual}")
        os.replace(path + ".part", path)
        print(f"fetched  {name}")


if __name__ == "__main__":
    main()
