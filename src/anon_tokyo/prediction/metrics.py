"""WOMD evaluation metrics: minADE, minFDE, MissRate.

Aligned with MTR official ``get_ade_of_waymo``:
  - 80-frame predictions are downsampled to 2 Hz (every 5th frame starting at 4).
  - ADE is computed at three horizons (step 5 / 9 / 15) then averaged.
"""

from __future__ import annotations

import torch
from torch import Tensor

_DEFAULT_MISS_THRESHOLD = 2.0  # metres (endpoint L2)
_CALCULATE_STEPS = (5, 9, 15)  # WOMD evaluation horizons (2 Hz indices)


def compute_prediction_metrics(
    pred_trajs: Tensor,
    pred_scores: Tensor,
    gt_trajs: Tensor,
    gt_mask: Tensor,
    *,
    miss_threshold: float = _DEFAULT_MISS_THRESHOLD,
) -> dict[str, Tensor]:
    """Compute open-loop prediction metrics (aligned with MTR official).

    All inputs are in the same coordinate frame (agent-local recommended).

    Args:
        pred_trajs: ``[B, K, num_modes, T, 2]`` — predicted (x, y).
        pred_scores: ``[B, K, num_modes]`` — confidence logits.
        gt_trajs: ``[B, K, T, 2]`` — ground-truth (x, y).
        gt_mask: ``[B, K, T]`` — valid mask.

    Returns:
        dict with keys ``minADE``, ``minFDE``, ``MissRate`` — each ``[B, K]``.
    """
    T = pred_trajs.shape[3]

    # Downsample 10 Hz → 2 Hz if 80 frames (matching MTR official)
    if T == 80:
        pred_trajs = pred_trajs[:, :, :, 4::5]  # [B, K, M, 16, 2]
        gt_trajs = gt_trajs[:, :, 4::5]  # [B, K, 16, 2]
        gt_mask = gt_mask[:, :, 4::5]  # [B, K, 16]

    # Per-mode displacement errors on downsampled frames
    err = (pred_trajs - gt_trajs[:, :, None, :, :]).norm(dim=-1)  # [B, K, M, T']
    err = err * gt_mask[:, :, None, :]

    # minADE: average of truncated ADE at steps 5, 9, 15 (MTR official)
    ade_sum = torch.zeros(err.shape[0], err.shape[1], device=err.device)
    n_steps = 0
    for step in _CALCULATE_STEPS:
        if step >= err.shape[3]:
            continue
        n_steps += 1
        trunc_err = err[:, :, :, : step + 1]
        trunc_mask = gt_mask[:, :, : step + 1]
        valid_count = trunc_mask.sum(dim=-1).clamp(min=1)  # [B, K]
        trunc_ade = trunc_err.sum(dim=-1) / valid_count[:, :, None]  # [B, K, M]
        ade_sum += trunc_ade.min(dim=-1).values
    min_ade = ade_sum / max(n_steps, 1)  # [B, K]

    # minFDE: error at last valid downsampled frame
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
    idx = idx * mask
    return idx.max(dim=-1).values.long().clamp(min=0)
