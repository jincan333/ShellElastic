"""Losses: fused linear cross-entropy with z-loss, and KL divergences for distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # Optional fused kernel (CUDA only).
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:  # pragma: no cover
    LigerFusedLinearCrossEntropyLoss = None

IGNORE_INDEX = -100


class LinearCrossEntropy:
    """Cross-entropy of ``hidden @ weight.T`` against ``target`` plus a z-loss.

    ``z_loss`` is the coefficient of ``logsumexp(logits)**2`` (averaged like the cross-entropy).
    With ``reduction="mean"`` the average runs over non-ignored targets; with ``"none"`` the
    per-token values are returned (zeros at ignored positions). With ``return_z_loss=True`` (mean
    reduction only) the result is a pair ``(loss, z_loss)`` whose second entry is the z-loss part of
    ``loss``. On CUDA the fused Liger kernel is used, which never materializes the full logits;
    elsewhere logits are built in chunks.
    """

    def __init__(self, z_loss: float = 0.0, reduction: str = "mean", chunk_size: int = 4096,
                 use_fused: bool = True, return_z_loss: bool = False) -> None:
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        if return_z_loss and reduction != "mean":
            raise ValueError("return_z_loss requires reduction='mean'")
        self.z_loss = float(z_loss)
        self.reduction = reduction
        self.chunk_size = int(chunk_size)
        self.return_z_loss = bool(return_z_loss)
        self.use_fused = use_fused and LigerFusedLinearCrossEntropyLoss is not None
        self._fused = None
        if self.use_fused:
            self._fused = LigerFusedLinearCrossEntropyLoss(
                lse_square_scale=self.z_loss,
                reduction=reduction,
                ignore_index=IGNORE_INDEX,
                accum_dtype=torch.float32,
                return_z_loss=self.return_z_loss,
            )

    def __call__(self, weight: torch.Tensor, hidden: torch.Tensor, target: torch.Tensor):
        if self._fused is not None and hidden.is_cuda:
            out = self._fused(weight, hidden, target)
            if self.return_z_loss:
                return out.loss, out.z_loss
            return out
        return self._reference(weight, hidden, target)

    def _reference(self, weight: torch.Tensor, hidden: torch.Tensor, target: torch.Tensor):
        losses, z_terms = [], []
        for start in range(0, hidden.shape[0], self.chunk_size):
            h = hidden[start : start + self.chunk_size]
            t = target[start : start + self.chunk_size]
            logits = F.linear(h, weight).float()
            ce = F.cross_entropy(logits, t, ignore_index=IGNORE_INDEX, reduction="none")
            z = torch.logsumexp(logits, dim=-1).pow(2) * self.z_loss * (t != IGNORE_INDEX)
            losses.append(ce + z if self.z_loss else ce)
            z_terms.append(z)
        per_token = torch.cat(losses)
        if self.reduction == "none":
            return per_token
        valid = (target != IGNORE_INDEX).sum().clamp(min=1)
        loss = per_token.sum() / valid
        if self.return_z_loss:
            return loss, (torch.cat(z_terms).sum() / valid).detach()
        return loss


def kl_divergence(t_logp: torch.Tensor, s_logp: torch.Tensor, mode: str = "forward") -> torch.Tensor:
    """KL divergence between a fixed teacher and a student, reduced over the vocabulary.

    ``forward``: KL(teacher || student); ``reverse``: KL(student || teacher);
    ``symmetric``: the mean of the two.
    """
    if mode == "forward":
        return (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
    if mode == "reverse":
        return (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
    if mode == "symmetric":
        fwd = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
        rev = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
        return 0.5 * (fwd + rev)
    raise ValueError(f"KL mode must be forward, reverse or symmetric, got {mode!r}")
