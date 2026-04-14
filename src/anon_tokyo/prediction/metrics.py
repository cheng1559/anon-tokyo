"""WOMD evaluation metrics: minADE, minFDE, MissRate."""

from __future__ import annotations

import torch
from torch import Tensor

# WOMD miss-rate thresholds by trajectory shape class (metres).
# Simplified: use lateral/longitudinal thresholds from WOMD.
# Full mAP requires 8-class motion classification — deferred.
_DEFAULT_MISS_THRESHOLD = 2.0  # metres (endpoint L2)


def compute_prediction_metrics(
    pred_trajs: Tensor,
    pred_scores: Tensor,
    gt_trajs: Tensor,
    gt_mask: Tensor,
    *,
    miss_threshold: float = _DEFAULT_MISS_THRESHOLD,
) -> dict[str, Tensor]:
    """Compute open-loop prediction metrics.

    All inputs are in the same coordinate frame (agent-local recommended).

    Args:
        pred_trajs: ``[B, K, num_modes, T, 2]`` — predicted (x, y).
        pred_scores: ``[B, K, num_modes]`` — confidence logits.
        gt_trajs: ``[B, K, T, 2]`` — ground-truth (x, y).
        gt_mask: ``[B, K, T]`` — valid mask.

    Returns:
        dict with keys ``minADE``, ``minFDE``, ``MissRate`` — each ``[B, K]``.
    """
    # Per-mode displacement errors
    err = (pred_trajs - gt_trajs[:, :, None, :, :]).norm(dim=-1)  # [B, K, M, T]
    err = err * gt_mask[:, :, None, :]  # zero out invalid

    valid_count = gt_mask.sum(dim=-1).clamp(min=1)  # [B, K]

    # ADE per mode: mean over valid frames
    ade = err.sum(dim=-1) / valid_count[:, :, None]  # [B, K, M]
    min_ade = ade.min(dim=-1).values  # [B, K]

    # FDE per mode: error at last valid frame
    # Find last valid index per agent
    last_valid = _last_valid_index(gt_mask)  # [B, K]
    b_idx = torch.arange(err.shape[0], device=err.device)[:, None].expand_as(last_valid)
    k_idx = torch.arange(err.shape[1], device=err.device)[None, :].expand_as(last_valid)
    fde = err[b_idx, k_idx, :, last_valid]  # [B, K, M]
    min_fde = fde.min(dim=-1).values  # [B, K]

    # MissRate: all modes miss if their FDE > threshold
    miss = (fde > miss_threshold).all(dim=-1).float()  # [B, K]

    return {
        "minADE": min_ade,
        "minFDE": min_fde,
        "MissRate": miss,
    }


def _last_valid_index(mask: Tensor) -> Tensor:
    """Return index of last valid (>0) frame per agent. ``mask``: ``[B, K, T]``."""
    T = mask.shape[-1]
    idx = torch.arange(T, device=mask.device).expand_as(mask)
    idx = idx * mask  # zero out invalid
    return idx.max(dim=-1).values.long().clamp(min=0)
