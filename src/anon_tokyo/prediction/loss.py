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

    # Sum over valid time steps (matches MTR _nll_loss_gmm_flat)
    reg_loss = (nll * gt_mask).sum(dim=-1)  # [B, K]

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
    return (l1 * gt_mask).sum(dim=-1)  # [B, K]  (sum over T, matches MTR)


def _dense_future_loss_scene(
    pred_dense: Tensor,
    gt_future: Tensor,
    gt_mask: Tensor,
) -> Tensor:
    """Dense future prediction loss for scene-centric models.

    Same computation as ``_dense_future_loss`` but with ``[B, A, T, ...]``
    layout instead of ``[K, A, T, ...]``.  Targets are agent-local futures.
    """
    B, A, T, _ = pred_dense.shape
    pred_gmm = pred_dense[:, :, :, 0:5]
    pred_vel = pred_dense[:, :, :, 5:7]
    gt_xy = gt_future[:, :, :, 0:2]
    gt_vel = gt_future[:, :, :, 2:4]

    vel_l1 = (pred_vel - gt_vel).abs().sum(dim=-1) * gt_mask  # [B, A, T]
    vel_l1 = vel_l1.sum(dim=-1)  # [B, A]

    flat_pred = pred_gmm.reshape(B * A, 1, T, 5)
    flat_gt = gt_xy.reshape(B * A, T, 2)
    flat_mask = gt_mask.reshape(B * A, T)
    fake_idx = torch.zeros(B * A, device=pred_dense.device, dtype=torch.long)
    reg_loss, _ = _nll_loss_gmm_flat(flat_pred, flat_gt, flat_mask, pre_winner_idx=fake_idx)
    reg_loss = reg_loss.reshape(B, A)

    loss_per_agent = vel_l1 + reg_loss
    agent_valid = gt_mask.sum(dim=-1) > 0
    loss_per_sample = (loss_per_agent * agent_valid.float()).sum(dim=-1) / agent_valid.sum(dim=-1).clamp(min=1)
    return loss_per_sample.mean()


def _gather_tracks_if_all_agents(
    tensor: Tensor,
    batch: dict[str, Tensor],
    *,
    pred_is_target_agents: bool = False,
) -> Tensor:
    """Gather ``tracks_to_predict`` when a scene-centric tensor is ``[B, A, ...]``."""
    if pred_is_target_agents:
        return tensor
    if "obj_types" not in batch:
        return tensor
    A = batch["obj_types"].shape[1]
    if tensor.shape[1] != A:
        return tensor
    ttp = batch["tracks_to_predict"]
    K = ttp.shape[1]
    b_idx = torch.arange(tensor.shape[0], device=tensor.device)[:, None].expand(tensor.shape[0], K)
    return tensor[b_idx, ttp.clamp(min=0)]


def prediction_loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    loss_weights: dict[str, float],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combined prediction loss over tracks_to_predict.

    Args:
        output: Model output with keys ``pred_trajs`` and ``pred_scores``.
            - ``pred_trajs``: ``[B, A_or_K, num_modes, T, 7]``
              (μx, μy, log_σ1, log_σ2, ρ, vx, vy) in agent-local frame.
            - ``pred_scores``: ``[B, A_or_K, num_modes]`` raw logits.
        batch: Collated batch with keys from transforms:
            - ``obj_trajs_future_local``: ``[B, A, T, 4]``
            - ``obj_trajs_future_mask``: ``[B, A, T]``
            - ``tracks_to_predict``: ``[B, K_max]`` (padded with -1)
        loss_weights: ``{"reg": 1.0, "score": 1.0, "vel": 0.2}``.

    Returns:
        total_loss: scalar.
        loss_dict: per-component losses for logging.
    """
    pred_is_target_agents = bool(output.get("pred_is_target_agents", False))
    pred_trajs = _gather_tracks_if_all_agents(
        output["pred_trajs"], batch, pred_is_target_agents=pred_is_target_agents
    )  # [B, K, M, T, 7]
    pred_scores = _gather_tracks_if_all_agents(
        output["pred_scores"], batch, pred_is_target_agents=pred_is_target_agents
    )  # [B, K, M]
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

    gt_xy = gt_trajs[:, :, :, 0:2]
    gt_vel = gt_trajs[:, :, :, 2:4]

    # Mask out invalid tracks_to_predict padding
    agent_valid = ttp_mask[:, :K].float()
    valid_count = agent_valid.sum().clamp(min=1)

    w_reg = loss_weights.get("reg", 1.0)
    w_cls = loss_weights.get("score", 1.0)
    w_vel = loss_weights.get("vel", 0.2)

    # Build layer list: pred_list contains ALL layers (including final);
    # fall back to single layer from pred_trajs/pred_scores when absent.
    if "pred_list" in output:
        layers = [
            (
                _gather_tracks_if_all_agents(sc, batch, pred_is_target_agents=pred_is_target_agents),
                _gather_tracks_if_all_agents(tr, batch, pred_is_target_agents=pred_is_target_agents),
            )
            for sc, tr in output["pred_list"]
        ]
    else:
        layers = [(pred_scores, pred_trajs)]

    loss_dict: dict[str, Tensor] = {}
    num_layers = len(layers)
    total_decoder_loss = gt_xy.new_tensor(0.0)

    for i, (sc_i, tr_i) in enumerate(layers):
        gmm_i = tr_i[:, :, :, :, 0:5]
        reg_i, win_i = nll_loss_gmm(gmm_i, gt_xy, gt_mask)
        cls_i = score_loss(sc_i, win_i)
        vel_i = velocity_loss(tr_i[:, :, :, :, 5:7], gt_vel, gt_mask, win_i)

        masked_reg = reg_i * agent_valid
        masked_vel = vel_i * agent_valid
        masked_cls = cls_i * agent_valid

        per_agent = w_reg * masked_reg + w_vel * masked_vel + w_cls * masked_cls
        layer_loss = per_agent.sum() / valid_count
        total_decoder_loss = total_decoder_loss + layer_loss

        loss_dict[f"loss/layer{i}"] = layer_loss.detach()
        loss_dict[f"loss/layer{i}_reg_gmm"] = (masked_reg.sum() / valid_count * w_reg).detach()
        loss_dict[f"loss/layer{i}_reg_vel"] = (masked_vel.sum() / valid_count * w_vel).detach()
        loss_dict[f"loss/layer{i}_cls"] = (masked_cls.sum() / valid_count * w_cls).detach()

        # Per-type ADE on last decoder layer
        if i + 1 == num_layers:
            BK = B * K
            flat_valid = agent_valid.reshape(BK).bool()
            if flat_valid.any():
                center_types = batch["obj_types"][b_idx, ttp_clamped[:, :K]].reshape(BK)
                _log_ade_per_type(
                    loss_dict,
                    pred_xy=gmm_i[:, :, :, :, 0:2].reshape(BK, -1, gmm_i.shape[3], 2)[flat_valid],
                    gt_xy=gt_xy.reshape(BK, -1, 2)[flat_valid],
                    gt_mask=gt_mask.reshape(BK, -1)[flat_valid],
                    obj_types=center_types[flat_valid],
                    post_tag=f"_layer{i}",
                )

    total_decoder_loss = total_decoder_loss / num_layers
    total_loss = total_decoder_loss

    # Dense future prediction auxiliary loss
    if "pred_dense_trajs" in output:
        pred_dense = output["pred_dense_trajs"]
        if pred_dense.ndim == 5:
            pred_dense = pred_dense[:, 0]
        dense_loss = _dense_future_loss_scene(
            pred_dense,
            batch["obj_trajs_future_local"],
            batch["obj_trajs_future_mask"],
        )
        total_loss = total_loss + dense_loss
        loss_dict["loss/dense"] = dense_loss.detach()

    loss_dict["loss/decoder"] = total_decoder_loss.detach()
    loss_dict["loss/total"] = total_loss.detach()
    return total_loss, loss_dict


# ── MTR agent-centric loss ───────────────────────────────────────────────────


def _nll_loss_gmm_flat(
    pred_trajs: Tensor,
    gt_trajs: Tensor,
    gt_mask: Tensor,
    *,
    log_std_range: tuple[float, float] = (-1.609, 5.0),
    rho_limit: float = 0.5,
    pre_winner_idx: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """GMM NLL for ``[N, M, T, 5]`` format (no B×K dims).

    Args:
        pred_trajs:  ``[N, M, T, 5]`` — (μx, μy, log_σ1, log_σ2, ρ).
        gt_trajs:    ``[N, T, 2]``.
        gt_mask:     ``[N, T]``.
        pre_winner_idx: optional ``[N]`` — if given, use as initial WTA.

    Returns:
        reg_loss ``[N]``, winner_idx ``[N]``.
    """
    N, M, T, _ = pred_trajs.shape

    pred_xy = pred_trajs[:, :, :, 0:2]  # [N, M, T, 2]
    gt_xy = gt_trajs[:, None, :, :]  # [N, 1, T, 2]
    dist = (pred_xy - gt_xy).norm(dim=-1) * gt_mask[:, None, :]
    total_dist = dist.sum(dim=-1)  # [N, M]

    if pre_winner_idx is not None:
        winner_idx = pre_winner_idx
    else:
        winner_idx = total_dist.argmin(dim=-1)

    n_idx = torch.arange(N, device=pred_trajs.device)
    winner_trajs = pred_trajs[n_idx, winner_idx]  # [N, T, 5]

    dx = gt_trajs[:, :, 0] - winner_trajs[:, :, 0]
    dy = gt_trajs[:, :, 1] - winner_trajs[:, :, 1]
    log_std1 = winner_trajs[:, :, 2].clamp(*log_std_range)
    log_std2 = winner_trajs[:, :, 3].clamp(*log_std_range)
    rho = winner_trajs[:, :, 4].clamp(-rho_limit, rho_limit)

    std1 = log_std1.exp()
    std2 = log_std2.exp()
    one_minus_rho2 = (1 - rho * rho).clamp(min=1e-6)

    log_coeff = log_std1 + log_std2 + 0.5 * one_minus_rho2.log()
    z = dx * dx / (std1 * std1) + dy * dy / (std2 * std2) - 2 * rho * dx * dy / (std1 * std2)
    nll = log_coeff + 0.5 * z / one_minus_rho2

    reg_loss = (nll * gt_mask).sum(dim=-1)

    return reg_loss, winner_idx


_TYPE_NAMES = {1: "TYPE_VEHICLE", 2: "TYPE_PEDESTRIAN", 3: "TYPE_CYCLIST"}
_ADE_CALC_STEPS = (5, 9, 15)


def _log_ade_per_type(
    loss_dict: dict[str, Tensor],
    pred_xy: Tensor,
    gt_xy: Tensor,
    gt_mask: Tensor,
    obj_types: Tensor,
    post_tag: str = "",
) -> None:
    """Add per-type minADE to *loss_dict* (matching official ``get_ade_of_each_category``).

    Args:
        pred_xy:   ``[K, M, T, 2]`` — predicted positions.
        gt_xy:     ``[K, T, 2]``    — ground-truth positions.
        gt_mask:   ``[K, T]``       — valid mask.
        obj_types: ``[K]``          — integer type (1=Vehicle, 2=Ped, 3=Cyclist).
    """
    T = pred_xy.shape[2]
    p, g, m = pred_xy, gt_xy, gt_mask
    if T == 80:
        p = p[:, :, 4::5]
        g = g[:, 4::5]
        m = m[:, 4::5]

    err = (p - g[:, None]).norm(dim=-1)  # [K, M, T']
    err = err * m[:, None]

    # minADE averaged over 3 horizons (official pattern)
    ade_sum = torch.zeros(err.shape[0], device=err.device)
    n_steps = 0
    for step in _ADE_CALC_STEPS:
        if step >= err.shape[2]:
            continue
        n_steps += 1
        trunc_err = err[:, :, : step + 1]
        trunc_mask = m[:, : step + 1]
        valid_cnt = trunc_mask.sum(dim=-1).clamp(min=1)  # [K]
        trunc_ade = trunc_err.sum(dim=-1) / valid_cnt[:, None]  # [K, M]
        ade_sum += trunc_ade.min(dim=-1).values  # winner mode
    min_ade = ade_sum / max(n_steps, 1)  # [K]

    for type_id, type_name in _TYPE_NAMES.items():
        type_mask = obj_types == type_id
        if type_mask.any():
            loss_dict[f"ade/{type_name}{post_tag}"] = min_ade[type_mask].mean()


def mtr_prediction_loss(
    output: dict[str, Tensor],
    loss_weights: dict[str, float],
) -> tuple[Tensor, dict[str, Tensor]]:
    """MTR per-layer loss + dense future prediction auxiliary loss.

    Expects agent-centric output from MTRModel:
        ``pred_list``: list of ``(scores [K, Q], trajs [K, Q, T, 7])`` per layer
        ``pred_dense_trajs``: ``[K, A, T, 7]``
        ``intention_points``: ``[K, Q, 2]``
        ``center_gt_trajs``: ``[K, T, 4]`` (x, y, vx, vy)
        ``center_gt_mask``: ``[K, T]``
        ``center_obj_type``: ``[K]`` (int: 1=Vehicle, 2=Ped, 3=Cyclist)
        ``obj_trajs_future``: ``[K, A, T, 4]``
        ``obj_trajs_future_mask``: ``[K, A, T]``
    """
    pred_list = output["pred_list"]
    intention_points = output["intention_points"]  # [K, Q, 2]
    center_gt = output["center_gt_trajs"]  # [K, T, 4]
    center_mask = output["center_gt_mask"]  # [K, T]
    K = center_gt.shape[0]

    # Find last valid frame for intention-point matching
    gt_xy = center_gt[:, :, 0:2]
    idx_range = torch.arange(center_mask.shape[1], device=center_mask.device)
    last_valid = (idx_range[None, :] * center_mask).max(dim=-1).values.long().clamp(min=0)
    gt_goals = gt_xy[torch.arange(K, device=gt_xy.device), last_valid]  # [K, 2]

    # Nearest intention point as positive index
    ip_dist = (gt_goals[:, None, :] - intention_points).norm(dim=-1)  # [K, Q]
    positive_idx = ip_dist.argmin(dim=-1)  # [K]

    w_reg = loss_weights.get("reg", 1.0)
    w_cls = loss_weights.get("score", 1.0)
    w_vel = loss_weights.get("vel", 0.5)

    loss_dict: dict[str, Tensor] = {}
    total_decoder_loss = gt_xy.new_tensor(0.0)
    num_layers = len(pred_list)

    for i, (scores, trajs) in enumerate(pred_list):
        # scores: [K, Q], trajs: [K, Q, T, 7]
        pred_gmm = trajs[:, :, :, 0:5]
        pred_vel = trajs[:, :, :, 5:7]

        loss_reg, positive_idx = _nll_loss_gmm_flat(
            pred_gmm,
            gt_xy,
            center_mask,
            pre_winner_idx=positive_idx,
        )

        # Velocity loss on winner mode
        n_idx = torch.arange(K, device=trajs.device)
        winner_vel = pred_vel[n_idx, positive_idx]  # [K, T, 2]
        gt_vel = center_gt[:, :, 2:4]
        vel_l1 = (winner_vel - gt_vel).abs().sum(dim=-1)
        loss_vel = (vel_l1 * center_mask).sum(dim=-1)

        # Classification loss
        loss_cls = F.cross_entropy(scores, positive_idx, reduction="none")

        # Official MTR builds a per-center loss vector, then averages it.
        layer_loss = (w_reg * loss_reg + w_vel * loss_vel + w_cls * loss_cls).mean()
        total_decoder_loss = total_decoder_loss + layer_loss

        # Per-layer detailed logging (matches official tb_dict keys)
        loss_dict[f"loss/layer{i}"] = layer_loss.detach()
        loss_dict[f"loss/layer{i}_reg_gmm"] = loss_reg.mean().detach() * w_reg
        loss_dict[f"loss/layer{i}_reg_vel"] = loss_vel.mean().detach() * w_vel
        loss_dict[f"loss/layer{i}_cls"] = loss_cls.mean().detach() * w_cls

        # Per-type ADE on last decoder layer
        if i + 1 == num_layers:
            _log_ade_per_type(
                loss_dict,
                pred_xy=pred_gmm[:, :, :, 0:2].detach(),
                gt_xy=gt_xy,
                gt_mask=center_mask,
                obj_types=output["center_obj_type"],
                post_tag=f"_layer{i}",
            )

    total_decoder_loss = total_decoder_loss / num_layers

    # Dense future prediction loss
    dense_loss = _dense_future_loss(output)

    total = total_decoder_loss + dense_loss
    loss_dict["loss/decoder"] = total_decoder_loss.detach()
    loss_dict["loss/dense"] = dense_loss.detach()
    loss_dict["loss/total"] = total.detach()
    return total, loss_dict


def _dense_future_loss(output: dict[str, Tensor]) -> Tensor:
    """Auxiliary L1 + GMM loss on all agents' dense future predictions."""
    pred_dense = output["pred_dense_trajs"]  # [K, A, T, 7]
    gt_future = output["obj_trajs_future"]  # [K, A, T, 4]
    gt_mask = output["obj_trajs_future_mask"]  # [K, A, T]

    K, A, T, _ = pred_dense.shape
    pred_gmm = pred_dense[:, :, :, 0:5]
    pred_vel = pred_dense[:, :, :, 5:7]
    gt_xy = gt_future[:, :, :, 0:2]
    gt_vel = gt_future[:, :, :, 2:4]

    # Velocity L1
    vel_l1 = (pred_vel - gt_vel).abs().sum(dim=-1)  # [K, A, T]
    vel_l1 = (vel_l1 * gt_mask).sum(dim=-1)  # [K, A]

    # GMM NLL (single mode)
    flat_pred = pred_gmm.view(K * A, 1, T, 5)
    flat_gt = gt_xy.view(K * A, T, 2)
    flat_mask = gt_mask.view(K * A, T)
    fake_idx = torch.zeros(K * A, device=pred_dense.device, dtype=torch.long)
    reg_loss, _ = _nll_loss_gmm_flat(flat_pred, flat_gt, flat_mask, pre_winner_idx=fake_idx)
    reg_loss = reg_loss.view(K, A)

    loss_per_agent = vel_l1 + reg_loss
    agent_valid = gt_mask.sum(dim=-1) > 0  # [K, A]
    loss_per_sample = (loss_per_agent * agent_valid.float()).sum(dim=-1) / agent_valid.sum(dim=-1).clamp(min=1)

    return loss_per_sample.mean()
