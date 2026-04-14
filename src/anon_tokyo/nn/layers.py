"""Common layers: MLP helper."""

from __future__ import annotations

import torch.nn as nn


def build_mlps(
    c_in: int,
    channels: list[int],
    *,
    ret_before_act: bool = False,
    use_norm: bool = True,
) -> nn.Sequential:
    """Stack of Linear(+BN+ReLU) layers, matching MTR's ``build_mlps``."""
    layers: list[nn.Module] = []
    for i, c_out in enumerate(channels):
        is_last = i == len(channels) - 1
        if is_last and ret_before_act:
            layers.append(nn.Linear(c_in, c_out, bias=True))
        elif use_norm:
            layers.extend([nn.Linear(c_in, c_out, bias=False), nn.BatchNorm1d(c_out), nn.ReLU()])
        else:
            layers.extend([nn.Linear(c_in, c_out, bias=True), nn.ReLU()])
        c_in = c_out
    return nn.Sequential(*layers)
