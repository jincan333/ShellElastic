"""Ensembles of the sub-networks (ablation (8)): diagnostics, in-place distillation (ablation (6))
and the oracle teacher.

All functions read logits through ``model.logits(hidden, s)`` from post-terminal-norm hidden
states and work in token chunks, so the full ``[tokens, vocab]`` logits of every width are never
materialized at once.
"""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

from elastic.train.losses import kl_divergence


def mixture_weights(widths: list[int], mode: str, loss_weights: dict[int, float]) -> tuple[list[int], dict[int, float]]:
    """Members and normalized weights of an ensemble over ``widths`` (``uniform`` or ``loss_weight``)."""
    widths = sorted(widths)
    if mode == "uniform":
        raw = {s: 1.0 for s in widths}
    elif mode == "loss_weight":
        raw = {s: float(loss_weights[s]) for s in widths}
    else:
        raise ValueError(f"ensemble weighting must be 'uniform' or 'loss_weight', got {mode!r}")
    members = [s for s in widths if raw[s] > 0.0]
    total = sum(raw[s] for s in members)
    return members, ({s: raw[s] / total for s in members} if total > 0 else {})


def oracle_log_weights(lp: torch.Tensor, beta, target_entropy: float) -> tuple[torch.Tensor, float, float]:
    """Per-token soft-oracle weights ``pi_s(i) = softmax_s(beta * log p_s(y_i))``.

    ``lp`` is ``[tokens, members]``. With ``beta="auto"`` the sharpness is solved by bisection so
    that the mean normalized entropy of the weights equals ``target_entropy``. Returns the log
    weights, the sharpness and the resulting normalized entropy.
    """
    n, m = lp.shape
    if m == 1:
        return lp.new_zeros((n, 1)), 0.0, 0.0
    log_m = math.log(m)

    def norm_entropy(b: float) -> float:
        logp = torch.log_softmax(b * lp, dim=1)
        h = -(logp.exp() * logp).sum(dim=1)
        return float((h / log_m).mean())

    if isinstance(beta, str):
        if beta != "auto":
            raise ValueError(f"oracle beta must be a number or 'auto', got {beta!r}")
        lo, hi = 0.0, 1.0
        for _ in range(40):
            if norm_entropy(hi) <= target_entropy:
                break
            hi *= 2.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if norm_entropy(mid) > target_entropy:
                lo = mid
            else:
                hi = mid
        b = 0.5 * (lo + hi)
    else:
        b = float(beta)
    return torch.log_softmax(b * lp, dim=1), float(b), norm_entropy(b)


@torch.no_grad()
def ensemble_diagnostics(model, hidden_by_s: dict[int, torch.Tensor], members: list[int], targets: torch.Tensor,
                         label, chunk: int = 4096, em_iters: int = 100, conf_tau: float = 1.0,
                         router_weights: torch.Tensor | None = None, oracle: dict | None = None):
    """Cross-entropies of ensembles of the read-out sub-networks (logging only).

    ``uniform``: equal-weight mixture; ``optmix``: best fixed mixture (EM warm-started at the best
    single width and clamped to it); ``conf``: per-token weights ``softmax_s(max_v log p_s(v))``;
    ``confpick``: the most confident width per token; ``oracle``: the width with the highest target
    probability per token; ``maxnet``: the widest member. With ``router_weights`` the learned router
    is evaluated as well; with ``oracle`` (``beta``, ``target_entropy``) the soft-oracle teacher.
    Returns ``(metrics, optmix_weights, oracle_beta)``.
    """
    tgt = targets.reshape(-1)
    valid_idx = (tgt != -100).nonzero(as_tuple=False).squeeze(-1)
    nv, m = int(valid_idx.numel()), len(members)
    if nv == 0 or m == 0:
        return None, None, None
    tgt_v = tgt[valid_idx]
    step = chunk if chunk and chunk > 0 else nv
    lp = torch.empty((nv, m), dtype=torch.float32, device=tgt.device)
    conf = torch.empty((nv, m), dtype=torch.float32, device=tgt.device)
    zinf = torch.empty((nv, m), dtype=torch.float32, device=tgt.device)
    for j, s in enumerate(members):
        h = hidden_by_s[s]
        h = h.reshape(-1, h.shape[-1])[valid_idx]
        for start in range(0, nv, step):
            end = min(start + step, nv)
            logits = model.logits(h[start:end], s).float()
            logp = torch.log_softmax(logits, dim=-1)
            lp[start:end, j] = logp.gather(-1, tgt_v[start:end].unsqueeze(-1)).squeeze(-1)
            conf[start:end, j] = logp.max(dim=-1).values
            zinf[start:end, j] = logits.abs().max(dim=-1).values
    log_m = math.log(m)
    metrics = {
        "loss_ensemble_uniform": -(torch.logsumexp(lp, dim=1) - log_m).mean(),
    }
    loss_oracle = -(lp.max(dim=1).values.mean())
    loss_maxnet = -(lp[:, -1].mean())
    per_net = -lp.mean(dim=0)
    j_best = int(torch.argmin(per_net))
    eps = 1e-3
    log_pi = lp.new_full((m,), math.log(eps / max(m - 1, 1)))
    log_pi[j_best] = math.log1p(-eps)
    for _ in range(int(em_iters)):
        joint = lp + log_pi.unsqueeze(0)
        log_q = torch.logsumexp(joint, dim=1, keepdim=True)
        log_pi = torch.logsumexp(joint - log_q, dim=0) - math.log(nv)
    loss_optmix = torch.minimum(-(torch.logsumexp(lp + log_pi.unsqueeze(0), dim=1).mean()), per_net[j_best])
    pi_star = log_pi.exp()
    extra: dict[str, torch.Tensor] = {}
    oracle_beta = None
    if oracle is not None:
        log_pi_tok, oracle_beta, teacher_h = oracle_log_weights(lp, oracle["beta"], oracle["target_entropy"])
        extra = {
            "loss_ensemble_teacher": -(torch.logsumexp(lp + log_pi_tok, dim=1).mean()),
            "ensemble_teacher_entropy": lp.new_tensor(float(teacher_h)),
            "ensemble_teacher_beta": lp.new_tensor(float(oracle_beta)),
        }
    log_w = torch.log_softmax(conf / conf_tau, dim=1)
    loss_conf = -(torch.logsumexp(lp + log_w, dim=1).mean())
    loss_confpick = -(lp.gather(1, conf.argmax(dim=1).unsqueeze(1)).squeeze(1).mean())
    if router_weights is not None:
        a = router_weights.reshape(-1, m)[valid_idx].to(lp.dtype)
        log_a = a.clamp_min(1e-9).log()
        extra["loss_ensemble_router"] = -(torch.logsumexp(lp + log_a, dim=1).mean())
        extra["loss_router_hard"] = -(lp.gather(1, a.argmax(dim=1).unsqueeze(1)).squeeze(1).mean())
        extra["router_entropy"] = lp.new_tensor(float((-(a.clamp_min(1e-9) * log_a).sum(dim=1).mean()) / log_m))
        mean_a = a.mean(dim=0)
        for j, s in enumerate(members):
            extra[f"router_weight_width_{label(s)}"] = lp.new_tensor(float(mean_a[j]))
    maxprob = conf.exp().mean(dim=0)
    zinf_mean = zinf.mean(dim=0)
    for j, s in enumerate(members):
        extra[f"maxprob_width_{label(s)}"] = maxprob[j]
        extra[f"logit_inf_width_{label(s)}"] = zinf_mean[j]
    if m >= 2:
        wins = (lp[:, 1:] > lp[:, :-1]).float().mean(dim=0)
        for j in range(1, m):
            extra[f"mono_rate_width_{label(members[j])}"] = wins[j - 1]
        extra["mono_rate_avg"] = wins.mean()
    metrics.update({
        "loss_ensemble_optmix": loss_optmix,
        "loss_ensemble_conf": loss_conf,
        "loss_ensemble_confpick": loss_confpick,
        "loss_ensemble_oracle": loss_oracle,
        "loss_ensemble_maxnet": loss_maxnet,
        **extra,
    })
    return metrics, {s: float(pi_star[j]) for j, s in enumerate(members)}, oracle_beta


def kd_from_max(model, teacher_hidden, student_hidden, teacher_s: int, student_s: int, valid: torch.Tensor,
                kl_mode: str = "forward", temperature: float = 1.0, chunk: int = 4096) -> torch.Tensor:
    """``T^2 * D(p_teacher, p_student)`` averaged over valid tokens (teacher detached, chunked)."""
    t = float(temperature)
    th = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
    sh = student_hidden.reshape(-1, student_hidden.shape[-1])
    mask = valid.reshape(-1)
    n = sh.shape[0]
    step = chunk if chunk and chunk > 0 else n

    def chunk_kl(sh_c, th_c, m_c):
        with torch.no_grad():
            t_logp = torch.log_softmax(model.logits(th_c, teacher_s).float() / t, dim=-1)
        s_logp = torch.log_softmax(model.logits(sh_c, student_s).float() / t, dim=-1)
        kl = kl_divergence(t_logp, s_logp, kl_mode)
        return (kl * m_c.to(kl.dtype)).sum()

    kl_sum = sh.new_zeros(())
    for start in range(0, n, step):
        end = min(start + step, n)
        m_c = mask[start:end]
        if not bool(m_c.any()):
            continue
        kl_sum = kl_sum + checkpoint(chunk_kl, sh[start:end], th[start:end], m_c, use_reentrant=False)
    n_valid = mask.sum().clamp(min=1).to(kl_sum.dtype)
    return (kl_sum / n_valid) * (t * t)


def kd_from_ensemble(model, hidden_by_s: dict[int, torch.Tensor], widths: list[int], members: list[int],
                     pi: dict[int, float], students: list[int], valid: torch.Tensor, kl_mode: str = "forward",
                     temperature: float = 1.0, chunk: int = 4096) -> dict[int, torch.Tensor]:
    """Distillation of every student toward the fixed mixture ``sum_m pi_m p_m`` of the members."""
    t = float(temperature)
    hid = {s: hidden_by_s[s].reshape(-1, hidden_by_s[s].shape[-1]) for s in widths}
    mask = valid.reshape(-1)
    n = mask.shape[0]
    step = chunk if chunk and chunk > 0 else n

    def chunk_kl(m_c, *sh_all):
        hd = {s: sh_all[i] for i, s in enumerate(widths)}
        with torch.no_grad():
            s0 = members[0]
            p_ens = pi[s0] * torch.softmax(model.logits(hd[s0].detach(), s0).float() / t, dim=-1)
            for s in members[1:]:
                p_ens = p_ens + pi[s] * torch.softmax(model.logits(hd[s].detach(), s).float() / t, dim=-1)
            t_logp = p_ens.clamp_min(1e-9).log()
        outs = []
        for s in students:
            s_logp = torch.log_softmax(model.logits(hd[s], s).float() / t, dim=-1)
            div = kl_divergence(t_logp, s_logp, kl_mode)
            outs.append((div * m_c.to(div.dtype)).sum())
        return torch.stack(outs)

    sums = None
    for start in range(0, n, step):
        end = min(start + step, n)
        m_c = mask[start:end]
        if not bool(m_c.any()):
            continue
        part = checkpoint(chunk_kl, m_c, *(hid[s][start:end] for s in widths), use_reentrant=False)
        sums = part if sums is None else sums + part
    if sums is None:
        zero = hid[students[0]].sum() * 0.0
        return {s: zero for s in students}
    sums = sums / mask.sum().clamp(min=1).to(sums.dtype) * (t * t)
    return {s: sums[i] for i, s in enumerate(students)}


def kd_from_oracle(model, hidden_by_s: dict[int, torch.Tensor], widths: list[int], students: list[int],
                   targets: torch.Tensor, valid: torch.Tensor, beta: float, kl_mode: str = "forward",
                   temperature: float = 1.0, chunk: int = 4096) -> dict[int, torch.Tensor]:
    """Distillation toward the per-token soft oracle ``sum_s softmax_s(beta log p_s(y)) p_s``."""
    t = float(temperature)
    hid = {s: hidden_by_s[s].reshape(-1, hidden_by_s[s].shape[-1]) for s in widths}
    mask = valid.reshape(-1)
    tgt = targets.reshape(-1)
    n = mask.shape[0]
    step = chunk if chunk and chunk > 0 else n

    def chunk_kl(m_c, tgt_c, *sh_all):
        hd = {s: sh_all[i] for i, s in enumerate(widths)}
        tgt_i = tgt_c.clamp_min(0).unsqueeze(-1)
        with torch.no_grad():
            cols = []
            for s in widths:
                logp1 = torch.log_softmax(model.logits(hd[s].detach(), s).float(), dim=-1)
                cols.append(logp1.gather(-1, tgt_i).squeeze(-1))
            lp_c = torch.stack(cols, dim=1)
            w = torch.softmax(beta * lp_c, dim=1)
            gm = m_c.to(lp_c.dtype)
            p_orc = None
            for i, s in enumerate(widths):
                term = w[:, i : i + 1] * torch.softmax(model.logits(hd[s].detach(), s).float() / t, dim=-1)
                p_orc = term if p_orc is None else p_orc + term
            t_logp = p_orc.clamp_min(1e-9).log()
        outs = []
        for s in students:
            s_logp = torch.log_softmax(model.logits(hd[s], s).float() / t, dim=-1)
            outs.append((kl_divergence(t_logp, s_logp, kl_mode) * gm).sum())
        outs.append(gm.sum())
        return torch.stack(outs)

    sums = None
    for start in range(0, n, step):
        end = min(start + step, n)
        m_c = mask[start:end]
        if not bool(m_c.any()):
            continue
        part = checkpoint(chunk_kl, m_c, tgt[start:end], *(hid[s][start:end] for s in widths), use_reentrant=False)
        sums = part if sums is None else sums + part
    if sums is None:
        zero = hid[students[0]].sum() * 0.0
        return {s: zero for s in students}
    kd = sums[:-1] / sums[-1].clamp(min=1) * (t * t)
    return {s: kd[i] for i, s in enumerate(students)}
