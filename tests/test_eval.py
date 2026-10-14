import json

import pytest
import torch

from elastic.eval.icl.runner import generate_greedy
from elastic.models import DenseElasticConfig, DenseElasticLM, ShellConfig, ShellElasticLM

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _perturb(model, seed=1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 1 or p is getattr(model, "readout_delta", None):
                p.add_(0.05 * torch.randn(p.shape, generator=gen).to(p))
    return model


def shell(vocab=256):
    torch.manual_seed(0)
    cfg = ShellConfig(vocab_size=vocab, seq_len=64, n_layers=2, base_width=32, supported=[1, 2, 3], head_dim=16,
                      ffn_hidden=32, proxy_width=64, fused_norm=False)
    return _perturb(ShellElasticLM(cfg).double())


def dense(mode="full"):
    torch.manual_seed(0)
    cfg = DenseElasticConfig(vocab_size=256, seq_len=64, n_layers=2, base_width=16, supported=[2, 3, 4],
                             head_dim=16, ffn_hidden=64, proxy_width=32, mode=mode)
    return _perturb(DenseElasticLM(cfg).double())


MODELS = [shell, dense, lambda: dense("ffn")]


def _ids(batch, length, vocab=256, seed=0):
    return torch.randint(1, vocab, (batch, length), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("make", MODELS)
def test_infer_matches_training_forward(make):
    model = make()
    ids = _ids(2, 24)
    for s in model.supported:
        logits, cache = model.infer(ids, s)
        assert cache is None and logits.dtype == torch.float32
        expected = model.logits(model(ids, widths=[s])[s].hidden, s)
        torch.testing.assert_close(logits, expected.float())
        last, _ = model.infer(ids, s, logits_to_keep=3)
        torch.testing.assert_close(last, logits[:, -3:])


@pytest.mark.parametrize("make", MODELS)
def test_kv_cache_matches_full_pass(make):
    """Prefill, a multi-token chunk on top of the cache, then single tokens."""
    model = make()
    ids = _ids(2, 20)
    for s in model.supported:
        full, _ = model.infer(ids, s)
        out, cache = model.infer(ids[:, :10], s, use_cache=True)
        pieces = [out]
        out, cache = model.infer(ids[:, 10:14], s, past_key_values=cache, use_cache=True)
        pieces.append(out)
        for t in range(14, 20):
            out, cache = model.infer(ids[:, t:t + 1], s, past_key_values=cache, use_cache=True)
            pieces.append(out)
        assert cache[0][0].shape[1] == 20
        torch.testing.assert_close(torch.cat(pieces, dim=1), full)


@pytest.mark.parametrize("make", MODELS)
def test_left_padding_is_masked(make):
    model = make()
    ids = _ids(1, 16)
    pad = 5
    padded = torch.cat([torch.zeros(1, pad, dtype=ids.dtype), ids], dim=1)
    mask = torch.cat([torch.zeros(1, pad, dtype=torch.bool), torch.ones(1, 16, dtype=torch.bool)], dim=1)
    for s in model.supported:
        reference, _ = model.infer(ids, s)
        out, _ = model.infer(padded, s, attention_mask=mask)
        torch.testing.assert_close(out[:, pad:], reference)


@pytest.mark.parametrize("make", MODELS)
def test_kv_cache_with_padding(make):
    """A left-padded batch: prefill, a multi-token chunk on top of the cache, then single tokens."""
    model = make()
    ids = _ids(2, 20)
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, :4] = False
    for s in model.supported:
        full, _ = model.infer(ids, s, attention_mask=mask)
        out, cache = model.infer(ids[:, :10], s, attention_mask=mask[:, :10], use_cache=True)
        pieces = [out]
        out, cache = model.infer(ids[:, 10:14], s, attention_mask=mask[:, :14], past_key_values=cache, use_cache=True)
        pieces.append(out)
        for t in range(14, 20):
            out, cache = model.infer(ids[:, t:t + 1], s, attention_mask=mask[:, :t + 1], past_key_values=cache,
                                     use_cache=True)
            pieces.append(out)
        torch.testing.assert_close(torch.cat(pieces, dim=1)[:, 4:], full[:, 4:])


@pytest.mark.parametrize("make", MODELS)
def test_scoring_reads_the_full_padded_batch(make):
    """The runner scores the whole right-padded batch, as LLM Foundry does (the same logits, bit for bit)."""
    from elastic.eval.icl.runner import _model_outputs

    model = make()
    ids = _ids(2, 24)
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, 15:] = False
    mask[1, 19:] = False
    ids = ids.masked_fill(~mask, 0)
    cont = [torch.arange(12, 15), torch.arange(16, 19)]
    for s in model.supported:
        batch = {"input_ids": ids, "attention_mask": mask, "labels": ids.clone(), "continuation_indices": cont}
        logits, _ = _model_outputs(model, s, batch, None, synced=False)
        full, _ = model.infer(ids, s, attention_mask=mask)
        assert torch.equal(logits, full)


def _naive_greedy(model, s, ids, mask, steps):
    for _ in range(steps):
        logits, _ = model.infer(ids, s, attention_mask=mask)
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        mask = torch.cat([mask, mask.new_ones((ids.shape[0], 1))], dim=1)
    return ids


def test_generate_greedy():
    model = shell()
    ids = _ids(2, 8)
    ids[0, :3] = 0
    mask = ids != 0
    out = generate_greedy(model, 2, ids, mask, max_new_tokens=6)
    assert torch.equal(out, _naive_greedy(model, 2, ids, mask, 6))

    # A row that emits EOS stops and is filled with padding; the other row continues.
    eos = int(out[1, 8])
    stopped = generate_greedy(model, 2, ids, mask, max_new_tokens=6, eos_token_id=eos, pad_token_id=7)
    assert stopped[1, 8] == eos and torch.all(stopped[1, 9:] == 7)
    first = [i for i in range(8, 14) if out[0, i] == eos]
    end = first[0] + 1 if first else 14
    assert torch.equal(stopped[0, :end], out[0, :end])

    # A batch-level criterion ends generation for every row.
    calls = []

    def stop_after_two(tokens, scores):
        calls.append(tokens.shape[1])
        return tokens.shape[1] >= 10

    short = generate_greedy(model, 2, ids, mask, max_new_tokens=6, stopping_criteria=[stop_after_two])
    assert short.shape[1] == 10 and calls == [9, 10]


@cuda
def test_icl_runner_end_to_end(tmp_path):
    """Every task type runs through the runner on a tiny model (needs the GPT-NeoX-20B tokenizer)."""
    from elastic.eval.icl.runner import build_icl_tasks, evaluate_model, load_tokenizer

    try:
        tokenizer = load_tokenizer("EleutherAI/gpt-neox-20b")
    except Exception as exc:  # no network and no local cache
        pytest.skip(f"tokenizer unavailable: {exc}")
    rows = {
        "mc.jsonl": [{"query": f"Question {i}: is {i} even?", "choices": ["yes", "no"], "gold": i % 2}
                     for i in range(11)],
        "lm.jsonl": [{"context": f"Count: {i} {i + 1} {i + 2}", "continuation": f" {i + 3}"} for i in range(9)],
        "schema.jsonl": [{"context_options": [f"A{i} likes", f"B{i} likes"], "continuation": "tea.", "gold": i % 2}
                         for i in range(7)],
        "gen.jsonl": [{"context": f"What is {i} plus 1?", "answer": str(i + 1), "aliases": [str(i + 1)]}
                      for i in range(5)],
    }
    for name, examples in rows.items():
        with open(tmp_path / name, "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in examples)
    tasks = [
        {"label": "mc", "dataset_uri": "mc.jsonl", "num_fewshot": [1], "icl_task_type": "multiple_choice",
         "continuation_delimiter": "\nAnswer: "},
        {"label": "lm", "dataset_uri": "lm.jsonl", "num_fewshot": [0], "icl_task_type": "language_modeling"},
        {"label": "schema", "dataset_uri": "schema.jsonl", "num_fewshot": [0], "icl_task_type": "schema"},
        {"label": "gen", "dataset_uri": "gen.jsonl", "num_fewshot": [1], "icl_task_type":
         "generation_task_with_answers", "continuation_delimiter": "\nA: "},
    ]
    icl_tasks = build_icl_tasks(tasks, tokenizer, max_seq_len=64, batch_size=4, data_dir=str(tmp_path))
    model = shell(vocab=50432).float().cuda()
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    results = evaluate_model(model, icl_tasks, tokenizer, torch.device("cuda"))
    expected = {f"{t}_{k}shot_width_{32 * s}" for t, k in (("mc", 1), ("lm", 0), ("schema", 0), ("gen", 1))
                for s in (1, 2, 3)}
    assert expected <= set(results)
    assert all(0.0 <= v <= 1.0 for v in results.values())
    assert {f"mean_width_{32 * s}" for s in (1, 2, 3)} <= set(results)
