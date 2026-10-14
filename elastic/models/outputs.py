"""Output containers of the elastic models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WidthOutput:
    """Output of one sub-network.

    ``hidden``: post-terminal-norm hidden states ``[batch, seq, width]`` (the readout input).
    ``loss``: the value returned by the loss function (scalar or per-token), if one was given.
    ``z_loss``: the z-loss part of ``loss``, when the loss function reports it.
    """

    hidden: torch.Tensor
    loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None


def split_loss(value) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(loss, z_loss)`` from a loss-function result (a tensor or a ``(loss, z_loss)`` pair)."""
    if isinstance(value, tuple):
        return value[0], value[1]
    return value, None
