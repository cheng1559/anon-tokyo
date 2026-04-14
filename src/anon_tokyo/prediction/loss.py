"""Winner-Takes-All + GMM-NLL + velocity auxiliary loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def nll_loss_gmm(
    pred_trajs: Tensor,
    gt_trajs: Tensor,
    gt_mask: Tensor,
    *,
    log_std_range: tuple[float, float] = (-1.609, 5.0),
    rho_limit: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Bivariate Gaussian NLL with Winner-Takes-All mode selection.

    Args:
        pred_trajs: ``[B, K, num_modes, T, 5]`` — (μx, μy, log_σ1, log_σ2, ρ).
        gt_trajs: ``[B, K, T, 2]`` — ground-truth (x, y).
        gt_mask: ``[B, K, T]`` — boolean/float valid mask.

    Returns:
        reg_loss: ``[B, K]`` — per-agent GMM NLL (averaged over valid frames).
        winner_idx: ``[B, K]`` — index of the best mode per agent.
    """
    B, K, M, T, _ = pred_trajs.shape

    # WTA: pick mode with smallest total L2 to gt
    pred_xy = pred_trajs[:, :, :, :, 0:2]  # [B, K, M, T, 2]
    gt_xy = gt_trajs[:, :, None, :, :]  # [B, K, 1, T, 2]
    dist = (pred_xy - gt_xy).norm(dim=-1)  # [B, K, M, T]
    dist = dist * gt_mask[:, :, None, :]  # mask invalid frames
    total_dist = dist.sum(dim=-1)  # [B, K, M]
    winner_idx = total_dist.argmin(dim=-1)  # [B, K]

    # Gather winner trajectories
    b_idx = torch.arange(B, device=pred_trajs.device)[:, None].expand(B, K)
    k_idx = torch.arange(K, device=pred_trajs.device)[None, :].expand(B, K)
    winner_trajs = pred_trajs[b_idx, k_idx, winner_idx]  # [B, K, T, 5]

    # Extract GMM parameters
    dx = gt_trajs[:, :, :, 0] - winner_trajs[:, :, :, 0]  # [B, K, T]
    dy = gt_trajs[:, :, :, 1] - winner_trajs[:, :, :, 1]
    log_std1 = winner_trajs[:, :, :, 2].clamp(*log_std_range)
    log_std2 = winner_trajs[:, :, :, 3].clamp(*log_std_range)
    rho = winner_trajs[:, :, :, 4].clamp(-rho_limit, rho_limit)

    std1 = log_std1.exp()
    std2 = log_std2.exp()
    one_minus_rho2 = (1 - rho * rho).clamp(min=1e-6)

    # Bivariate Gaussian NLL: -log N(dx, dy | 0, 0, σ1, σ2, ρ)
    log_coeff = log_std1 + log_std2 + 0.5 * one_minus_rho2.log()
    z = dx * dx / (std1 * std1) + dy * dy / (std2 * std2) - 2 * rho * dx * dy / (std1 * std2)
    nll = log_coeff + 0.5 * z / one_minus_rho2  # [B, K, T]

    # Average over valid time steps
    valid_count = gt_mask.sum(dim=-1).clamp(min=1)  # [B, K]
    reg_loss = (nll * gt_mask).sum(dim=-1) / valid_count  # [B, K]

    return reg_loss, winner_idx


def score_loss(pred_scores: Tensor, winner_idx: Tensor) -> Tensor:
    """Cross-entropy loss for mode classification.

    Args:
        pred_scores: ``[B, K, num_modes]`` — raw logits.
        winner_idx: ``[B, K]`` — target mode index.

    Returns:
        cls_loss: ``[B, K]``.
    """
    B, K, M = pred_scores.shape
    return F.cross_entropy(
        pred_scores.reshape(B * K, M),
        winner_idx.reshape(B * K),
        reduction="none",
    ).reshape(B, K)


def velocity_loss(
    pred_vel: Tensor,
    gt_vel: Tensor,
    gt_mask: Tensor,
    winner_idx: Tensor,
) -> Tensor:
    """L1 velocity loss on the winner mode.

    Args:
        pred_vel: ``[B, K, num_modes, T, 2]``.
        gt_vel: ``[B, K, T, 2]``.
        gt_mask: ``[B, K, T]``.
        winner_idx: ``[B, K]``.

    Returns:
        vel_loss: ``[B, K]``.
    """
    B, K, M, T, _ = pred_vel.shape
    b_idx = torch.arange(B, device=pred_vel.device)[:, None].expand(B, K)
    k_idx = torch.arange(K, device=pred_vel.device)[None, :].expand(B, K)
    winner_vel = pred_vel[b_idx, k_idx, winner_idx]  # [B, K, T, 2]

    l1 = (winner_vel - gt_vel).abs().sum(dim=-1)  # [B, K, T]
    valid_count = gt_mask.sum(dim=-1).clamp(min=1)
    return (l1 * gt_mask).sum(dim=-1) / valid_count  # [B, K]


def prediction_loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    loss_weights: dict[str, float],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combined prediction loss over tracks_to_predict.

    Args:
        output: Model output with keys ``pred_trajs`` and ``pred_scores``.
            - ``pred_trajs``: ``[B, K, num_modes, T, 7]``
              (μx, μy, log_σ1, log_σ2, ρ, vx, vy) in agent-local frame.
            - ``pred_scores``: ``[B, K, num_modes]`` raw logits.
        batch: Collated batch with keys from transforms:
            - ``obj_trajs_future_local``: ``[B, A, T, 4]``
            - ``obj_trajs_future_mask``: ``[B, A, T]``
            - ``tracks_to_predict``: ``[B, K_max]`` (padded with -1)
        loss_weights: ``{"reg": 1.0, "score": 1.0, "vel": 0.2}``.

    Returns:
        total_loss: scalar.
        loss_dict: per-component losses for logging.
    """
    pred_trajs = output["pred_trajs"]  # [B, K, M, T, 7]
    pred_scores = output["pred_scores"]  # [B, K, M]
    B = pred_trajs.shape[0]
    K = pred_trajs.shape[1]

    # Gather GT for tracks_to_predict
    ttp = batch["tracks_to_predict"]  # [B, K_max]
    ttp_mask = ttp >= 0  # [B, K_max]
    ttp_clamped = ttp.clamp(min=0)  # safe index

    gt_future_local = batch["obj_trajs_future_local"]  # [B, A, T, 4]
    gt_future_mask = batch["obj_trajs_future_mask"]  # [B, A, T]

    # Gather gt for the K predicted agents
    b_idx = torch.arange(B, device=gt_future_local.device)[:, None].expand(B, K)
    gt_trajs = gt_future_local[b_idx, ttp_clamped[:, :K]]  # [B, K, T, 4]
    gt_mask = gt_future_mask[b_idx, ttp_clamped[:, :K]]  # [B, K, T]

    # GMM regression loss
    pred_gmm = pred_trajs[:, :, :, :, 0:5]
    gt_xy = gt_trajs[:, :, :, 0:2]
    loss_reg, winner_idx = nll_loss_gmm(pred_gmm, gt_xy, gt_mask)

    # Score classification loss
    loss_cls = score_loss(pred_scores, winner_idx)

    # Velocity loss
    pred_vel = pred_trajs[:, :, :, :, 5:7]
    gt_vel = gt_trajs[:, :, :, 2:4]
    loss_vel = velocity_loss(pred_vel, gt_vel, gt_mask, winner_idx)

    # Mask out invalid tracks_to_predict padding
    agent_valid = ttp_mask[:, :K].float()
    valid_count = agent_valid.sum().clamp(min=1)

    loss_reg_mean = (loss_reg * agent_valid).sum() / valid_count
    loss_cls_mean = (loss_cls * agent_valid).sum() / valid_count
    loss_vel_mean = (loss_vel * agent_valid).sum() / valid_count

    total = (
        loss_weights.get("reg", 1.0) * loss_reg_mean
        + loss_weights.get("score", 1.0) * loss_cls_mean
        + loss_weights.get("vel", 0.2) * loss_vel_mean
    )

    return total, {
        "loss/reg": loss_reg_mean.detach(),
        "loss/score": loss_cls_mean.detach(),
        "loss/vel": loss_vel_mean.detach(),
        "loss/total": total.detach(),
    }
