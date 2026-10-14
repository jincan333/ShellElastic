"""Training and validation objectives.

ShellElastic, StairFormer and non-elastic models (:class:`ShellElasticLM`): one trunk pass per step
at the widest selected width, and every selected width ``s`` adds its cross-entropy ``CE_s`` to

    L = sum_s w_s (CE_s + kd_s * KD_s) / sum_{s in T} w_s,

where ``T`` is the set of selected widths that train the trunk (all of them unless
``detach_trunk``); with ``kd_s = 0`` and without ``detach_trunk``, this is the joint objective of
Equation 8. Supported widths above the widest selected one are evaluated for logging in a
separate pass without gradient. With a width router (ensemble ablation (8)), the loss is instead
the router-weighted per-token cross-entropy of all widths (plus an optional all-width regularizer).

Fully elastic models and MatFormer (:class:`DenseElasticLM`): every selected width runs its own
forward pass, and its weighted loss is back-propagated before the next width runs, which bounds
the activation memory by one sub-network (the accumulated gradient equals that of the joint loss).

Validation evaluates every supported width with the pure cross-entropy averaged over tokens.
The logged training ``loss`` is ``sum_s w_s CE_s / sum_s w_s`` over the selected widths, where
``CE_s`` includes the z-loss term; the weighted z-loss itself is logged as ``z_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from elastic.models.dense import DenseElasticLM
from elastic.train.ensemble import (
    ensemble_diagnostics,
    kd_from_ensemble,
    kd_from_max,
    kd_from_oracle,
    mixture_weights,
)
from elastic.train.losses import LinearCrossEntropy
from elastic.train.selection import WidthSelector, parse_step_rule

TIER_NAMES = ("low", "medium", "high", "extra", "max")
DISTILL_TEACHERS = ("none", "max", "ensemble", "oracle")


@dataclass
class ObjectiveConfig:
    loss_weights: list[float] | None = None  # w_s per supported width (0 = untrained); default all 1
    selection: str = "all"  # all | schedule | random | sandwich | max_random | max
    schedule: list[str] | None = None  # per supported width "M/N" firing rule (selection="schedule")
    detach_trunk: bool = False  # all trained heads learn every step; only selected widths train the trunk
    seed: int = 0  # seed of the stochastic selection strategies
    z_loss: float = 1e-4

    distill: str = "none"  # none | max | ensemble | oracle
    distill_weights: list[float] | None = None  # kd_s per supported width
    distill_kl: str = "forward"  # forward | reverse | symmetric
    distill_temperature: float = 1.0
    distill_chunk: int = 4096
    ensemble_weighting: str = "uniform"  # mixture weights of the ensemble teacher: uniform | loss_weight
    oracle_beta: float | str = "auto"
    oracle_target_entropy: float = 0.7

    ensemble_diagnostics: bool = False
    router_tau: float = 1.0
    router_all_width_ce: float = 0.0


def _masked_mean(per_token: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return per_token[valid].mean() if torch.any(valid) else per_token.mean()


class Objective:
    """Joint multi-width objective of one model (see the module docstring)."""

    def __init__(self, cfg: ObjectiveConfig, model) -> None:
        self.cfg = cfg
        self.dense = isinstance(model, DenseElasticLM)
        self.supported = [int(s) for s in model.supported]
        weights = cfg.loss_weights if cfg.loss_weights is not None else [1.0] * len(self.supported)
        if len(weights) != len(self.supported):
            raise ValueError(f"loss_weights must have one entry per supported width {self.supported}")
        self.loss_weights = {s: float(w) for s, w in zip(self.supported, weights)}
        self.trained = [s for s in self.supported if self.loss_weights[s] != 0.0]
        if list(model.trained) != self.trained:
            raise ValueError(f"model trained widths {model.trained} differ from the objective's {self.trained}")
        schedule = None
        if cfg.selection == "schedule":
            if cfg.schedule is None or len(cfg.schedule) != len(self.supported):
                raise ValueError("selection 'schedule' needs one schedule entry per supported width")
            schedule = {s: parse_step_rule(e) for s, e in zip(self.supported, cfg.schedule)}
            for s, rule in schedule.items():
                if rule is not None and s not in self.trained:
                    raise ValueError(f"width {s} has a schedule but a zero loss weight")
            if not self.dense and not cfg.detach_trunk:
                untrained = [s for s in self.trained if schedule[s] is None]
                if untrained:
                    raise ValueError(f"trained widths {untrained} never fire under the schedule")
        self.selector = WidthSelector(self.trained, cfg.selection, schedule, cfg.seed)
        if cfg.distill not in DISTILL_TEACHERS:
            raise ValueError(f"distill must be one of {DISTILL_TEACHERS}, got {cfg.distill!r}")
        kd = cfg.distill_weights if cfg.distill_weights is not None else [0.0] * len(self.supported)
        if len(kd) != len(self.supported):
            raise ValueError("distill_weights must have one entry per supported width")
        self.kd_weights = {s: float(w) for s, w in zip(self.supported, kd)}
        self.router = getattr(model, "width_router", None) is not None
        if self.dense and (cfg.distill != "none" or cfg.ensemble_diagnostics or cfg.detach_trunk):
            raise NotImplementedError("distillation, ensembles and detach_trunk are implemented for ShellElastic")
        if cfg.detach_trunk and float(model.cfg.stage_grad_scale) != 1.0:
            raise ValueError("detach_trunk cannot be combined with stage_grad_scale != 1")
        if self.dense:
            unit = model.cfg.ffn_hidden if model.ffn_only else model.g
        else:
            unit = model.g
        self._unit = int(unit)

    def label(self, s: int) -> int:
        """Width used in metric names: the model width (the FFN width for MatFormer)."""
        return s * self._unit

    def trained_widths(self, step: int) -> list[int]:
        """Widths whose losses take part in the training step ``step`` (1-indexed)."""
        return self._selected(step, False)[0]

    # ------------------------------------------------------------------------------------------

    def train_step(self, model, input_ids: torch.Tensor, targets: torch.Tensor, step: int,
                   loss_scale: float = 1.0) -> dict[str, torch.Tensor]:
        """Forward and backward of one training step (1-indexed global ``step``); returns metrics."""
        if self.dense:
            return self._dense(model, input_ids, targets, step, False, loss_scale)
        return self._shell(model, input_ids, targets, step, False, loss_scale)

    @torch.no_grad()
    def validation_step(self, model, input_ids: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.dense:
            return self._dense(model, input_ids, targets, 0, True, 1.0)
        return self._shell(model, input_ids, targets, 0, True, 1.0)

    # ------------------------------------------------------------------------------------------

    def _selected(self, step: int, validation: bool) -> tuple[list[int], set[int] | None]:
        if validation:
            return list(self.trained), None
        fired = self.selector(step)
        if not self.cfg.detach_trunk:
            return sorted(fired), None
        sampled = list(self.trained)
        if len(sampled) <= 1:
            return sampled, None
        trunk = set(fired) & set(sampled)
        return sampled, (None if trunk >= set(sampled) else trunk)

    def _alphas(self, model, widths) -> dict[int, float]:
        """Readout scalars (read right after a forward pass, while FSDP keeps them unsharded)."""
        out = {}
        for s in widths:
            alpha = model.alpha_of(s)
            out[s] = float(alpha.detach()) if isinstance(alpha, torch.Tensor) else float(alpha)
        return out

    def _width_metrics(self, live: dict[int, torch.Tensor], alphas: dict[int, float], metrics: dict) -> None:
        losses = {}
        for s in sorted(live):
            n = self.label(s)
            losses[f"loss_width_{n}"] = live[s].detach()
            metrics[f"alpha_width_{n}"] = alphas[s]
            metrics[f"loss_weight_width_{n}"] = self.loss_weights[s]
            if self.cfg.distill != "none":
                metrics[f"kd_weight_width_{n}"] = self.kd_weights[s]
        if losses:
            losses["loss_width_avg"] = torch.stack(list(losses.values())).mean()
            if len(live) == len(TIER_NAMES):
                for name, s in zip(TIER_NAMES, sorted(live)):
                    losses[f"loss_{name}"] = losses[f"loss_width_{self.label(s)}"]
        metrics.update(losses)

    def _backward(self, loss: torch.Tensor, device_type: str) -> None:
        with torch.autocast(device_type=device_type, enabled=False):
            loss.backward()

    # ------------------------------------------------------------------------------------------

    def _shell(self, model, ids, tgt, step: int, validation: bool, loss_scale: float) -> dict:
        cfg = self.cfg
        sampled, trunk = self._selected(step, validation)
        weights = [self.loss_weights[s] for s in sampled]
        per_token = validation or self.router
        loss_fn = LinearCrossEntropy(z_loss=0.0 if validation else cfg.z_loss,
                                     reduction="none" if per_token else "mean", return_z_loss=not per_token)
        grad_set = set(sampled)
        if validation:
            outs = model(ids, widths=self.supported, targets=tgt, loss_fn=loss_fn, grad_widths=grad_set)
        else:
            s_run = max(sampled)
            outs = model(ids, widths=[s for s in self.supported if s <= s_run], targets=tgt, loss_fn=loss_fn,
                         grad_widths=grad_set, trunk_grad_widths=trunk)
            extra = [s for s in self.supported if s > s_run]
            if extra:
                with torch.no_grad():
                    outs.update(model(ids, widths=extra, targets=tgt, loss_fn=loss_fn, grad_widths=set()))
        valid = tgt.reshape(-1) != -100
        live = {s: (_masked_mean(outs[s].loss, valid) if per_token else outs[s].loss) for s in self.supported}
        metrics: dict = {}
        self._width_metrics(live, self._alphas(model, self.supported), metrics)

        members = sorted(self.supported)
        router_a = None
        if self.router:
            h_max = outs[max(members)].hidden
            router_a = torch.softmax(model.width_router_logits(h_max.reshape(-1, h_max.shape[-1])) / cfg.router_tau,
                                     dim=-1)
        run_diag = (
            cfg.ensemble_diagnostics
            or self.router
            or (not validation and cfg.distill in ("ensemble", "oracle") and len(sampled) > 1)
        )
        oracle_beta = None
        if run_diag:
            oracle = ({"beta": cfg.oracle_beta, "target_entropy": cfg.oracle_target_entropy}
                      if cfg.distill == "oracle" else None)
            diag, pi_star, oracle_beta = ensemble_diagnostics(
                model, {s: outs[s].hidden for s in members}, members, tgt, self.label, chunk=cfg.distill_chunk,
                router_weights=router_a.detach() if router_a is not None else None, oracle=oracle,
            )
            if diag is not None:
                metrics.update(diag)
                for s, w in pi_star.items():
                    metrics[f"optmix_weight_width_{self.label(s)}"] = w

        kd: dict[int, torch.Tensor] = {}
        if not validation and cfg.distill != "none" and len(sampled) > 1:
            students = [s for s in sorted(sampled) if self.kd_weights[s] != 0.0]
            common = dict(kl_mode=cfg.distill_kl, temperature=cfg.distill_temperature, chunk=cfg.distill_chunk)
            if cfg.distill == "max":
                teacher = max(sampled)
                t_hidden = outs[teacher].hidden.detach()
                for s in students:
                    if s != teacher:
                        kd[s] = kd_from_max(model, t_hidden, outs[s].hidden, teacher, s, valid, **common)
            elif cfg.distill == "ensemble":
                ens_members, pi = mixture_weights(sampled, cfg.ensemble_weighting, self.loss_weights)
                if ens_members and students:
                    kd = kd_from_ensemble(model, {s: outs[s].hidden for s in sampled}, sorted(sampled), ens_members,
                                          pi, students, valid, **common)
            elif len(sampled) >= 2 and students and oracle_beta is not None:
                kd = kd_from_oracle(model, {s: outs[s].hidden for s in sampled}, sorted(sampled), students, tgt,
                                    valid, float(oracle_beta), **common)
            for s, v in kd.items():
                metrics[f"kd_width_{self.label(s)}"] = v.detach()

        if self.router:
            ce_mat = torch.stack([outs[s].loss for s in members], dim=1)
            n_valid = valid.sum().clamp(min=1).to(ce_mat.dtype)
            vmask = valid.to(ce_mat.dtype)
            router_ce = ((router_a * ce_mat).sum(dim=1) * vmask).sum() / n_valid
            loss = router_ce
            if cfg.router_all_width_ce != 0.0:
                loss = loss + cfg.router_all_width_ce * ((ce_mat.mean(dim=1) * vmask).sum() / n_valid)
            reported = router_ce.detach()
        else:
            weight_sum = float(sum(weights))
            trunk_weight = weight_sum if trunk is None else float(sum(w for s, w in zip(sampled, weights) if s in trunk))
            loss = ce_total = z_total = None
            for s, w in zip(sampled, weights):
                ce_s = live[s]
                ce_total = ce_s * w if ce_total is None else ce_total + ce_s * w
                if outs[s].z_loss is not None:
                    z_s = outs[s].z_loss.detach() * w
                    z_total = z_s if z_total is None else z_total + z_s
                loss_s = ce_s + self.kd_weights[s] * kd[s] if s in kd else ce_s
                term = loss_s * (w / trunk_weight)
                loss = term if loss is None else loss + term
            reported = ce_total.detach() / weight_sum
            if z_total is not None:
                metrics["z_loss"] = z_total / weight_sum
        metrics["loss"] = reported
        if not validation:
            self._backward(loss * loss_scale, ids.device.type)
        return metrics

    def _dense(self, model, ids, tgt, step: int, validation: bool, loss_scale: float) -> dict:
        sampled, _ = self._selected(step, validation)
        weights = {s: self.loss_weights[s] for s in sampled}
        weight_sum = float(sum(weights.values()))
        metrics: dict = {}
        if validation:
            loss_fn = LinearCrossEntropy(z_loss=0.0, reduction="none")
            valid = tgt.reshape(-1) != -100
            outs = model(ids, widths=self.supported, targets=tgt, loss_fn=loss_fn)
            live = {s: _masked_mean(outs[s].loss, valid) for s in self.supported}
            ce_total = None
            for s in sampled:
                ce_total = live[s] * weights[s] if ce_total is None else ce_total + live[s] * weights[s]
            self._width_metrics(live, self._alphas(model, self.supported), metrics)
            metrics["loss"] = ce_total / weight_sum
            return metrics
        loss_fn = LinearCrossEntropy(z_loss=self.cfg.z_loss, reduction="mean", return_z_loss=True)
        live, alphas, ce_total, z_total = {}, {}, None, None
        for s in sorted(sampled):
            w = weights[s]
            out = model(ids, widths=[s], targets=tgt, loss_fn=loss_fn)[s]
            alphas.update(self._alphas(model, [s]))
            self._backward(loss_scale * (w / weight_sum) * out.loss, ids.device.type)
            ce = out.loss.detach()
            live[s] = ce
            ce_total = w * ce if ce_total is None else ce_total + w * ce
            if out.z_loss is not None:
                z = w * out.z_loss.detach()
                z_total = z if z_total is None else z_total + z
        self._width_metrics(live, alphas, metrics)
        metrics["loss"] = ce_total / weight_sum
        if z_total is not None:
            metrics["z_loss"] = z_total / weight_sum
        return metrics
