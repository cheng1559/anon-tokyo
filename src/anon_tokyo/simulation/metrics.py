"""Rollout metrics shared by simulation training and visualization."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _safe_rate(count: Tensor, denom: Tensor) -> Tensor:
    return torch.where(denom > 0, count.to(torch.float32) / denom.to(torch.float32).clamp_min(1.0), torch.zeros_like(count, dtype=torch.float32))


def compute_rollout_metric_tensors(
    *,
    collision: Tensor,
    offroad: Tensor,
    goal_reached: Tensor,
    controlled_mask: Tensor,
    agent_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute per-world and aggregate rates from ``[T, B, A]`` boolean rollout events."""
    collision = collision.bool()
    offroad = offroad.bool()
    goal_reached = goal_reached.bool()
    controlled = controlled_mask.to(device=collision.device).bool()
    if agent_mask is not None:
        controlled = controlled & agent_mask.to(device=collision.device).bool()

    collision_any = collision.any(dim=0) & controlled
    offroad_any = offroad.any(dim=0) & controlled
    goal_any = goal_reached.any(dim=0) & controlled
    done_any = (collision_any | offroad_any) & controlled

    controlled_count = controlled.sum(dim=-1)
    collision_count = collision_any.sum(dim=-1)
    offroad_count = offroad_any.sum(dim=-1)
    goal_count = goal_any.sum(dim=-1)
    done_count = done_any.sum(dim=-1)

    total_controlled = controlled_count.sum()
    return {
        "controlled_count": controlled_count,
        "collision_count": collision_count,
        "offroad_count": offroad_count,
        "goal_reached_count": goal_count,
        "done_count": done_count,
        "collision_rate": _safe_rate(collision_count, controlled_count),
        "offroad_rate": _safe_rate(offroad_count, controlled_count),
        "goal_reaching_rate": _safe_rate(goal_count, controlled_count),
        "done_rate": _safe_rate(done_count, controlled_count),
        "batch_controlled_count": total_controlled,
        "batch_collision_count": collision_count.sum(),
        "batch_offroad_count": offroad_count.sum(),
        "batch_goal_reached_count": goal_count.sum(),
        "batch_done_count": done_count.sum(),
        "batch_collision_rate": _safe_rate(collision_count.sum(), total_controlled),
        "batch_offroad_rate": _safe_rate(offroad_count.sum(), total_controlled),
        "batch_goal_reaching_rate": _safe_rate(goal_count.sum(), total_controlled),
        "batch_done_rate": _safe_rate(done_count.sum(), total_controlled),
    }


def scalar_rollout_metrics(metrics: dict[str, Tensor]) -> dict[str, float]:
    """Return scalar training metrics from ``compute_rollout_metric_tensors`` output."""
    keys = (
        "batch_collision_rate",
        "batch_offroad_rate",
        "batch_goal_reaching_rate",
        "batch_done_rate",
        "batch_controlled_count",
        "batch_collision_count",
        "batch_offroad_count",
        "batch_goal_reached_count",
        "batch_done_count",
    )
    return {key.removeprefix("batch_"): float(metrics[key].detach().cpu()) for key in keys}


def serializable_batch_metrics(metrics: dict[str, Tensor]) -> dict[str, Any]:
    return {
        "controlled_count": int(metrics["batch_controlled_count"].detach().cpu().item()),
        "collision_count": int(metrics["batch_collision_count"].detach().cpu().item()),
        "offroad_count": int(metrics["batch_offroad_count"].detach().cpu().item()),
        "goal_reached_count": int(metrics["batch_goal_reached_count"].detach().cpu().item()),
        "done_count": int(metrics["batch_done_count"].detach().cpu().item()),
        "collision_rate": float(metrics["batch_collision_rate"].detach().cpu().item()),
        "offroad_rate": float(metrics["batch_offroad_rate"].detach().cpu().item()),
        "goal_reaching_rate": float(metrics["batch_goal_reaching_rate"].detach().cpu().item()),
        "done_rate": float(metrics["batch_done_rate"].detach().cpu().item()),
    }


def serializable_world_metrics(metrics: dict[str, Tensor], sample_idx: int) -> dict[str, Any]:
    return {
        "controlled_count": int(metrics["controlled_count"][sample_idx].detach().cpu().item()),
        "collision_count": int(metrics["collision_count"][sample_idx].detach().cpu().item()),
        "offroad_count": int(metrics["offroad_count"][sample_idx].detach().cpu().item()),
        "goal_reached_count": int(metrics["goal_reached_count"][sample_idx].detach().cpu().item()),
        "done_count": int(metrics["done_count"][sample_idx].detach().cpu().item()),
        "collision_rate": float(metrics["collision_rate"][sample_idx].detach().cpu().item()),
        "offroad_rate": float(metrics["offroad_rate"][sample_idx].detach().cpu().item()),
        "goal_reaching_rate": float(metrics["goal_reaching_rate"][sample_idx].detach().cpu().item()),
        "done_rate": float(metrics["done_rate"][sample_idx].detach().cpu().item()),
    }
