"""Token-level ensemble analysis of the sub-networks of a trained ShellElastic model (Appendix J, ablation (8)).

1) Per-token dump of every read-out (one trunk pass at the full width, every width read out as a
   prefix) on held-out text, e.g. the first 1,024 C4 validation sequences:

    python tools/analyze_ensemble.py dump outputs/<name>/final --data data/c4-val --out tokens_default.npz \
        [--n-seqs 1024] [--router-tau 1.0]

   For every target token t and width s (ascending): lp = log p_s(y_t), conf = max_v log p_s(v),
   ent = entropy of p_s, kl = KL(p_max || p_s), top1 = [argmax p_s == y_t], and the router's
   per-token weights for a model with a width router. Each sequence is the first seq_len + 1 tokens
   of a validation document, masked like the training data.

2) Summary statistics (mixtures of the read-outs, the per-token oracle, confidence gating, deciles,
   distillation teachers, the learned router) over several dumps of the same tokens:

    python tools/analyze_ensemble.py summarize --default tokens_default.npz \
        --replicas tokens_rep1.npz tokens_rep2.npz --router tokens_router.npz --out summary.json

   ``--replicas`` are further runs of the default configuration (independent ensembles of the
   full width); ``--router`` a run with the learned width router. Every mixture is a per-token
   probability mixture sum_s pi_s p_s(y); every CE is a mean over the same tokens in float64.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

N_DECILES = 10
N_CONF_BINS = 20
MIX_W = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
         0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


# ---------------------------------------------------------------------------------------------
# 1) Per-token dump
# ---------------------------------------------------------------------------------------------


def iter_sequences(source: str, seq_len: int, vocab_size: int):
    """(inputs, targets) of the first seq_len + 1 tokens of every document, in shard order."""
    import webdataset as wds

    from elastic.data.text import TextDataConfig, _parse_txt, discover_shards, make_example

    cfg = TextDataConfig()
    urls, _ = discover_shards(source, shuffle_seed=None)
    for (raw,) in wds.DataPipeline(wds.SimpleShardList(urls), wds.tarfile_to_samples(), wds.to_tuple("txt")):
        tokens = _parse_txt(raw, vocab_size)
        if len(tokens) < seq_len + 1:
            continue
        yield make_example(tokens[: seq_len + 1], seq_len, cfg.target_mask_left, cfg.target_mask_individual,
                           cfg.squash_mask_left)


def dump(args) -> None:
    import torch

    from elastic.train.losses import LinearCrossEntropy
    from elastic.train.trainer import build_model

    with open(os.path.join(args.weights, "model.json")) as f:
        spec = json.load(f)
    if spec.get("type", "shell") != "shell":
        raise SystemExit("the ensemble analysis reads the nested read-outs of a ShellElastic model")
    device = torch.device("cuda")
    model = build_model(spec, 0, device, init_from=args.weights)
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    model.eval()
    widths = list(model.trained)
    router_on = getattr(model, "width_router", None) is not None
    seq_len = int(spec.get("seq_len", 2048))
    out: dict[str, list] = {k: [] for k in ("lp", "conf", "ent", "kl", "top1", "router", "target", "pos")}
    loss_fn = LinearCrossEntropy(z_loss=0.0, reduction="none")
    fused_ce = {s: [0.0, 0] for s in widths}
    sequences = iter_sequences(args.data, seq_len, int(spec.get("vocab_size", 50432)))
    done = 0
    while done < args.n_seqs:
        batch = [next(sequences) for _ in range(min(args.batch, args.n_seqs - done))]
        inputs = torch.stack([x for x, _ in batch]).to(device)
        targets = torch.stack([y for _, y in batch]).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outs = model(inputs, widths=widths, targets=targets, loss_fn=loss_fn)
            flat = targets.reshape(-1)
            valid = (flat != -100).nonzero().squeeze(-1)
            for s in widths:
                per_token = outs[s].loss.reshape(-1)[valid].float()
                fused_ce[s][0] += float(per_token.sum())
                fused_ce[s][1] += int(per_token.numel())
            pos = torch.arange(seq_len, device=device).repeat(inputs.shape[0])[valid]
            tv = flat[valid]
            hidden = {s: outs[s].hidden.reshape(-1, s * model.g)[valid] for s in widths}
            n = tv.numel()
            lp, conf, ent, kl, top1 = (torch.empty(n, len(widths), device=device) for _ in range(5))
            router = None
            if router_on:
                router = torch.softmax(model.width_router_logits(hidden[widths[-1]]).float() / args.router_tau, -1)
            for a in range(0, n, args.chunk):
                b = min(a + args.chunk, n)
                logp = [torch.log_softmax(model.logits(hidden[s][a:b], s).float(), dim=-1) for s in widths]
                p_max = logp[-1].exp()
                for j, lq in enumerate(logp):
                    lp[a:b, j] = lq.gather(-1, tv[a:b, None]).squeeze(-1)
                    best, arg = lq.max(dim=-1)
                    conf[a:b, j] = best
                    top1[a:b, j] = (arg == tv[a:b]).float()
                    ent[a:b, j] = -(lq.exp() * lq).sum(-1)
                    kl[a:b, j] = (p_max * (logp[-1] - lq)).sum(-1)
        for key, value in (("lp", lp), ("conf", conf), ("ent", ent), ("kl", kl)):
            out[key].append(value.cpu().numpy().astype(np.float32))
        out["top1"].append(top1.cpu().numpy().astype(np.bool_))
        if router is not None:
            out["router"].append(router.cpu().numpy().astype(np.float32))
        out["target"].append(tv.cpu().numpy().astype(np.int32))
        out["pos"].append(pos.cpu().numpy().astype(np.int16))
        done += inputs.shape[0]
        print(f"{done}/{args.n_seqs} sequences", flush=True)
    arrays = {k: np.concatenate(v) for k, v in out.items() if v}
    labels = [s * model.g for s in widths]
    for s, w, ce in zip(widths, labels, -arrays["lp"].mean(0, dtype=np.float64)):
        model_ce = fused_ce[s][0] / fused_ce[s][1]
        print(f"width {w}: CE {ce:.4f} (model's fused loss {model_ce:.4f})")
        if abs(ce - model_ce) > 1e-3:
            raise SystemExit("the dump disagrees with the model's own loss")
    np.savez(args.out, widths=np.array(labels), **arrays)
    print(f"wrote {args.out}: {arrays['lp'].shape[0]} tokens")


# ---------------------------------------------------------------------------------------------
# 2) Summary statistics
# ---------------------------------------------------------------------------------------------


def mean(x: np.ndarray) -> float:
    return float(np.mean(x, dtype=np.float64))


def best_fixed_mixture(lp: np.ndarray) -> tuple[np.ndarray, float]:
    """Convex weights minimizing the mixture CE on these tokens (L-BFGS on softmax logits)."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    def f(theta: np.ndarray):
        log_pi = theta - logsumexp(theta)
        joint = lp + log_pi
        lq = logsumexp(joint, axis=1, keepdims=True)
        resp = np.exp(joint - lq).mean(axis=0)
        return -mean(lq), -(resp - np.exp(log_pi))

    res = minimize(f, np.zeros(lp.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-10})
    log_pi = res.x - logsumexp(res.x)
    return np.exp(log_pi), float(res.fun)


def combos(lp: np.ndarray, conf: np.ndarray | None) -> dict:
    """CE of every combination rule over the columns of lp, minus the CE of the last column."""
    from scipy.special import logsumexp

    n, k = lp.shape
    ref = -mean(lp[:, -1])
    pi, opt = best_fixed_mixture(lp)
    out = {"reference_ce": ref, "uniform": -mean(logsumexp(lp, axis=1) - np.log(k)) - ref,
           "best_fixed": opt - ref, "best_fixed_weights": pi.tolist(), "oracle": -mean(lp.max(axis=1)) - ref}
    if conf is not None:
        log_w = conf - logsumexp(conf, axis=1, keepdims=True)
        out["confidence"] = -mean(logsumexp(lp + log_w, axis=1)) - ref
        out["most_confident"] = -mean(lp[np.arange(n), conf.argmax(axis=1)]) - ref
    return out


def two_member_curve(lp_ref: np.ndarray, lp_other: np.ndarray) -> list[float]:
    ref = -mean(lp_ref)
    out = []
    for w in MIX_W:
        if w == 0.0:
            out.append(0.0)
            continue
        a = np.log1p(-w) + lp_ref if w < 1.0 else np.full_like(lp_ref, -np.inf)
        out.append(-mean(np.logaddexp(a, np.log(w) + lp_other)) - ref)
    return out


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties)."""
    from scipy.stats import rankdata

    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize(args) -> None:
    from scipy.special import logsumexp

    z = np.load(args.default)
    reps = [np.load(p) for p in args.replicas]
    rr = np.load(args.router) if args.router else None
    for other in reps + ([rr] if rr is not None else []):
        if not np.array_equal(other["target"], z["target"]):
            raise SystemExit("the dumps are not on the same tokens")
    widths = [int(w) for w in z["widths"]]
    k = len(widths)
    top = k - 1
    lp, conf = z["lp"].astype(np.float64), z["conf"].astype(np.float64)
    rep_lp = [r["lp"].astype(np.float64) for r in reps]
    n = lp.shape[0]
    s: dict = {"n_tokens": n, "widths": widths, "mix_w": MIX_W}
    s["per_width_ce"] = {"default": (-lp.mean(0)).tolist()}
    for i, r in enumerate(rep_lp):
        s["per_width_ce"][f"replica{i + 1}"] = (-r.mean(0)).tolist()
    s["combos"] = {"nested": combos(lp, conf),
                   f"nested_{widths[-2]}_{widths[-1]}": combos(lp[:, -2:], conf[:, -2:]),
                   f"nested_{widths[0]}_{widths[-1]}": combos(lp[:, [0, top]], conf[:, [0, top]])}
    if rep_lp:
        s["combos"]["independent2"] = combos(np.stack([rep_lp[0][:, top], lp[:, top]], 1), None)
        s["combos"]["replica1_nested"] = combos(rep_lp[0], reps[0]["conf"].astype(np.float64))
    if len(rep_lp) > 1:
        s["combos"]["independent3"] = combos(np.stack([rep_lp[0][:, top], rep_lp[1][:, top], lp[:, top]], 1), None)
    win = lp.argmax(axis=1)
    gain = lp.max(axis=1) - lp[:, top]
    s["oracle_pick"] = {
        "share": [float(np.mean(win == j)) for j in range(k)],
        "gain_contrib": [float(gain[win == j].sum() / n) for j in range(k)],
        "beats_max": [float(np.mean(lp[:, j] > lp[:, top])) for j in range(k)],
    }
    if rr is not None:
        s["oracle_pick"]["router_run_share"] = [float(np.mean(rr["lp"].argmax(axis=1) == j)) for j in range(k)]
    s["pairwise_win"] = [[float(np.mean(lp[:, a] > lp[:, b])) for b in range(k)] for a in range(k)]
    nll = -lp[:, top]
    edges = np.quantile(nll, np.linspace(0, 1, N_DECILES + 1))
    dec = np.clip(np.searchsorted(edges, nll, side="right") - 1, 0, N_DECILES - 1)
    u_nest = logsumexp(lp, axis=1) - np.log(k) - lp[:, top]
    s["deciles"] = {"nll_mean": [mean(nll[dec == i]) for i in range(N_DECILES)],
                    "oracle_gain_nested": [mean(gain[dec == i]) for i in range(N_DECILES)],
                    "uniform_delta_nested": [-mean(u_nest[dec == i]) for i in range(N_DECILES)]}
    if rep_lp:
        g_ind = np.maximum(lp[:, top], rep_lp[0][:, top]) - lp[:, top]
        u_ind = np.logaddexp(lp[:, top], rep_lp[0][:, top]) - np.log(2) - lp[:, top]
        s["deciles"]["oracle_gain_independent2"] = [mean(g_ind[dec == i]) for i in range(N_DECILES)]
        s["deciles"]["uniform_delta_independent2"] = [-mean(u_ind[dec == i]) for i in range(N_DECILES)]
    s["two_member"] = {str(w): two_member_curve(lp[:, top], lp[:, j]) for j, w in enumerate(widths[:-1])}
    if rep_lp:
        s["two_member"]["independent"] = two_member_curve(lp[:, top], rep_lp[0][:, top])
    s["confidence"] = {"auc": [], "bins": {}}
    for j, w in enumerate(widths[:-1]):
        d, dc = lp[:, j] - lp[:, top], conf[:, j] - conf[:, top]
        s["confidence"]["auc"].append(auc(d > 0, dc))
        e = np.quantile(dc, np.linspace(0, 1, N_CONF_BINS + 1))
        b = np.clip(np.searchsorted(e, dc, side="right") - 1, 0, N_CONF_BINS - 1)
        s["confidence"]["bins"][str(w)] = {
            "conf_diff_mean": [mean(dc[b == i]) for i in range(N_CONF_BINS)],
            "p_beats_max": [float(np.mean(d[b == i] > 0)) for i in range(N_CONF_BINS)],
            "gain_if_picked": [mean(d[b == i]) for i in range(N_CONF_BINS)],
        }
    jc = conf.argmax(axis=1)
    s["confidence"]["pick_share"] = [float(np.mean(jc == j)) for j in range(k)]
    s["confidence"]["pick_agrees_with_oracle"] = float(np.mean(jc == win))
    cols = np.concatenate([lp] + [r[:, top:top + 1] for r in rep_lp], axis=1)
    s["corr_labels"] = [str(w) for w in widths] + [f"{widths[-1]} (replica {i + 1})" for i in range(len(rep_lp))]
    s["corr"] = np.corrcoef(cols.T).tolist()
    ref = -mean(lp[:, top])
    loss_w = np.array(args.loss_weights if args.loss_weights else [1.0] * (k - 1) + [4.0], dtype=np.float64)
    teachers = {"ensemble": -mean(logsumexp(lp + np.log(loss_w / loss_w.sum()), axis=1)) - ref}

    def label_aware(beta: float) -> tuple[float, float]:
        lpi = beta * lp - logsumexp(beta * lp, axis=1, keepdims=True)
        ent = mean(-(np.exp(lpi) * lpi).sum(axis=1) / np.log(k))
        return -mean(logsumexp(lp + lpi, axis=1)) - ref, ent

    lo, hi = 0.0, 200.0
    for _ in range(40):  # beta whose teacher has normalized entropy 0.7
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if label_aware(mid)[1] > args.target_entropy else (lo, mid)
    teachers["label_aware"], teachers["label_aware_entropy"] = label_aware(mid)
    teachers["label_aware_beta"] = mid
    s["teachers"] = teachers
    s["top1_acc"] = z["top1"].mean(0, dtype=np.float64).tolist()
    s["top1_any"] = float(z["top1"].any(axis=1).mean())
    s["entropy"] = z["ent"].mean(0, dtype=np.float64).tolist()
    s["kl_from_max"] = z["kl"].mean(0, dtype=np.float64).tolist()
    if rr is not None and "router" in rr:
        ra, rlp = rr["router"].astype(np.float64), rr["lp"].astype(np.float64)
        r_ref = -mean(rlp[:, top])
        s["router"] = {
            "mean_weight": ra.mean(0).tolist(),
            "frac_tokens_weight_max_ge_0999": float(np.mean(ra[:, top] >= 0.999)),
            "min_weight_max": float(ra[:, top].min()),
            "soft_delta": -mean(logsumexp(rlp + np.log(np.clip(ra, 1e-30, 1)), axis=1)) - r_ref,
            "oracle_delta": -mean(rlp.max(axis=1)) - r_ref,
            "reference_ce": r_ref,
        }
    with open(args.out, "w") as f:
        json.dump(s, f, indent=1)
    print(f"wrote {args.out}")
    for key, value in s["combos"].items():
        print(key, {a: round(b, 4) for a, b in value.items() if isinstance(b, float)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump", help="per-token statistics of one model")
    d.add_argument("weights", help="directory with model.safetensors and model.json")
    d.add_argument("--data", required=True, help="validation source (shards with txt samples)")
    d.add_argument("--out", required=True)
    d.add_argument("--n-seqs", type=int, default=1024)
    d.add_argument("--batch", type=int, default=8)
    d.add_argument("--chunk", type=int, default=4096)
    d.add_argument("--router-tau", type=float, default=1.0)
    m = sub.add_parser("summarize", help="summary statistics over dumps of the same tokens")
    m.add_argument("--default", required=True)
    m.add_argument("--replicas", nargs="*", default=[])
    m.add_argument("--router", default=None)
    m.add_argument("--loss-weights", type=float, nargs="*", default=None,
                   help="weights of the ensemble teacher (default: 1,...,1,4 as in ablation (8))")
    m.add_argument("--target-entropy", type=float, default=0.7)
    m.add_argument("--out", required=True)
    args = parser.parse_args()
    dump(args) if args.command == "dump" else summarize(args)


if __name__ == "__main__":
    main()
