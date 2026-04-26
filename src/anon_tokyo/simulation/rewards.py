"""Closed-loop reward components for simplified WOMD simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


ROAD_EDGE_TYPES = (15, 16)
SOLID_LINE_TYPES = (7, 8, 11, 12)
BOUNDARY_TYPES = ROAD_EDGE_TYPES + SOLID_LINE_TYPES


@dataclass
class RewardConfig:
    num_steps: int = 80
    collision_reward_weight: float = -1.0
    offroad_reward_weight: float = -1.0
    offroad_distance_threshold: float = 0.35
    goal_reaching_threshold: float = 1.5
    goal_reaching_weight: float = 0.5
    ttc_horizon: float = 4.0
    ttc_reward_floor: float = 0.1
    ttc_radius_buffer: float = 0.25
    comfort_weight: float = 0.5
    comfort_long_accel: float = 3.0
    comfort_lat_accel: float = 4.0
    comfort_long_jerk: float = 5.0
    comfort_lat_jerk: float = 2.0


def agent_polygons(positions: Tensor, headings: Tensor, sizes: Tensor) -> Tensor:
    """Build oriented rectangle corners for agents."""
    half_l = sizes[..., 0].clamp_min(0.1) * 0.5
    half_w = sizes[..., 1].clamp_min(0.1) * 0.5
    forward = torch.stack((headings.cos(), headings.sin()), dim=-1)
    left = torch.stack((-headings.sin(), headings.cos()), dim=-1)
    corners = torch.stack(
        (
            positions + forward * half_l.unsqueeze(-1) + left * half_w.unsqueeze(-1),
            positions + forward * half_l.unsqueeze(-1) - left * half_w.unsqueeze(-1),
            positions - forward * half_l.unsqueeze(-1) - left * half_w.unsqueeze(-1),
            positions - forward * half_l.unsqueeze(-1) + left * half_w.unsqueeze(-1),
        ),
        dim=-2,
    )
    return corners


def _rectangle_overlap(polygons: Tensor, valid: Tensor) -> Tensor:
    """Pairwise oriented rectangle overlap using SAT."""
    B, A, _, _ = polygons.shape
    edges = torch.stack(
        (
            polygons[:, :, 1] - polygons[:, :, 0],
            polygons[:, :, 3] - polygons[:, :, 0],
        ),
        dim=2,
    )
    axes = edges / edges.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    axes_i = axes[:, :, None].expand(B, A, A, 2, 2)
    axes_j = axes[:, None, :].expand(B, A, A, 2, 2)
    pair_axes = torch.cat((axes_i, axes_j), dim=3)

    poly_i = polygons[:, :, None, None]
    poly_j = polygons[:, None, :, None]
    axis = pair_axes.unsqueeze(-2)
    proj_i = (poly_i * axis).sum(dim=-1)
    proj_j = (poly_j * axis).sum(dim=-1)
    min_i, max_i = proj_i.amin(dim=-1), proj_i.amax(dim=-1)
    min_j, max_j = proj_j.amin(dim=-1), proj_j.amax(dim=-1)
    overlap = (max_i >= min_j) & (max_j >= min_i)
    pair_overlap = overlap.all(dim=-1)

    pair_valid = valid[:, :, None] & valid[:, None, :]
    eye = torch.eye(A, dtype=torch.bool, device=polygons.device).unsqueeze(0)
    return pair_overlap & pair_valid & ~eye


def collision_reward(positions: Tensor, headings: Tensor, sizes: Tensor, valid: Tensor, controlled: Tensor, cfg: RewardConfig):
    polygons = agent_polygons(positions, headings, sizes)
    pair_collision = _rectangle_overlap(polygons, valid)
    collision = pair_collision.any(dim=-1) & controlled & valid
    reward = collision.float() * cfg.collision_reward_weight
    return reward, collision, pair_collision, polygons


def _point_segment_distance(points: Tensor, start: Tensor, end: Tensor) -> Tensor:
    seg = end - start
    rel = points[:, None] - start[None]
    denom = (seg * seg).sum(dim=-1).clamp_min(1e-7)
    u = ((rel * seg[None]).sum(dim=-1) / denom[None]).clamp(0.0, 1.0)
    proj = start[None] + u[..., None] * seg[None]
    return (points[:, None] - proj).norm(dim=-1)


def offroad_reward(polygons: Tensor, valid: Tensor, controlled: Tensor, map_polylines: Tensor, map_mask: Tensor, cfg: RewardConfig):
    B, A = valid.shape
    out = torch.zeros(B, A, dtype=torch.bool, device=valid.device)
    types = map_polylines[..., 6].round().long()
    boundary_type = torch.zeros_like(types, dtype=torch.bool)
    for type_code in BOUNDARY_TYPES:
        boundary_type |= types == type_code

    for b in range(B):
        agent_idx = torch.where(controlled[b] & valid[b])[0]
        if agent_idx.numel() == 0:
            continue
        point_mask = map_mask[b].bool() & boundary_type[b]
        seg_mask = point_mask[:, :-1] & point_mask[:, 1:]
        if not seg_mask.any():
            continue
        starts = map_polylines[b, :, :-1, 0:2][seg_mask]
        ends = map_polylines[b, :, 1:, 0:2][seg_mask]
        agent_points = torch.cat((polygons[b, agent_idx], polygons[b, agent_idx].mean(dim=1, keepdim=True)), dim=1)
        flat_points = agent_points.reshape(-1, 2)
        min_dist = _point_segment_distance(flat_points, starts, ends).amin(dim=1).view(agent_idx.numel(), -1).amin(dim=1)
        out[b, agent_idx] = min_dist <= cfg.offroad_distance_threshold

    reward = out.float() * cfg.offroad_reward_weight
    return reward, out


def ttc_reward(positions: Tensor, velocities: Tensor, sizes: Tensor, valid: Tensor, controlled: Tensor, cfg: RewardConfig):
    rel_pos = positions[:, None, :, :] - positions[:, :, None, :]
    rel_vel = velocities[:, None, :, :] - velocities[:, :, None, :]
    rel_speed_sq = (rel_vel * rel_vel).sum(dim=-1)
    t_star = -((rel_pos * rel_vel).sum(dim=-1)) / rel_speed_sq.clamp_min(1e-6)
    t_star = t_star.clamp(0.0, cfg.ttc_horizon)
    closest = rel_pos + rel_vel * t_star.unsqueeze(-1)
    dist = closest.norm(dim=-1)
    radius = 0.5 * sizes.norm(dim=-1).clamp_min(0.1) + cfg.ttc_radius_buffer
    radius_sum = radius[:, :, None] + radius[:, None, :]

    B, A = valid.shape
    eye = torch.eye(A, dtype=torch.bool, device=valid.device).unsqueeze(0)
    pair_mask = controlled[:, :, None] & valid[:, :, None] & valid[:, None, :] & ~eye
    alert_pair = (dist <= radius_sum) & pair_mask
    t_pair = torch.where(alert_pair, t_star, torch.full_like(t_star, cfg.ttc_horizon))
    ttc = t_pair.amin(dim=-1)
    alert = (ttc < cfg.ttc_horizon) & controlled & valid
    score = cfg.ttc_reward_floor + (1.0 - cfg.ttc_reward_floor) * (ttc / cfg.ttc_horizon).clamp(0.0, 1.0)
    score = torch.where(alert, score, torch.ones_like(score))
    return score, alert, ttc


def goal_reaching_reward(positions: Tensor, goals: Tensor, goal_reached: Tensor, controlled: Tensor, valid: Tensor, cfg: RewardConfig):
    dist = (positions - goals).norm(dim=-1)
    reached = (dist <= cfg.goal_reaching_threshold) & controlled & valid
    first_reached = reached & ~goal_reached
    reward = torch.where(first_reached, torch.full_like(dist, cfg.goal_reaching_weight), torch.zeros_like(dist))
    next_goal_reached = goal_reached | reached
    return reward, first_reached, next_goal_reached, dist


def comfort_reward(a_long: Tensor, a_lat: Tensor, jerk_long: Tensor, jerk_lat: Tensor, controlled: Tensor, valid: Tensor, cfg: RewardConfig):
    long_acc = (a_long.abs() / max(cfg.comfort_long_accel, 1e-6)).clamp(max=1.0)
    lat_acc = (a_lat.abs() / max(cfg.comfort_lat_accel, 1e-6)).clamp(max=1.0)
    long_jerk = (jerk_long.abs() / max(cfg.comfort_long_jerk, 1e-6)).clamp(max=1.0)
    lat_jerk = (jerk_lat.abs() / max(cfg.comfort_lat_jerk, 1e-6)).clamp(max=1.0)
    penalty = torch.stack((long_acc, lat_acc, long_jerk, lat_jerk), dim=-1).amax(dim=-1)
    score = 1.0 - (1.0 - cfg.comfort_weight) * penalty.square()
    return torch.where(controlled & valid, score, torch.ones_like(score))


def compute_rewards(state: dict[str, Tensor], cfg: RewardConfig) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor]:
    """Compute combined reward, done flags, component info, and next goal flags."""
    valid = state["valid_mask"].bool()
    controlled = state["controlled_mask"].bool()
    collision_r, collision, pair_collision, polygons = collision_reward(
        state["positions"], state["headings"], state["sizes"], valid, controlled, cfg
    )
    offroad_r, offroad = offroad_reward(
        polygons,
        valid,
        controlled,
        state["map_polylines"],
        state["map_polylines_mask"],
        cfg,
    )
    ttc_score, ttc_alert, ttc = ttc_reward(state["positions"], state["velocities"], state["sizes"], valid, controlled, cfg)
    goal_r, goal_first, next_goal_reached, goal_dist = goal_reaching_reward(
        state["positions"], state["goal_positions"], state["goal_reached"], controlled, valid, cfg
    )
    comfort_score = comfort_reward(
        state["a_long"], state["a_lat"], state["jerk_long"], state["jerk_lat"], controlled, valid, cfg
    )

    hard_reward = collision_r + offroad_r
    done = (collision | offroad) & controlled & valid

    soft_score = comfort_score * ttc_score
    alive = (~done).float()
    total_reward = hard_reward + alive * (goal_r + soft_score / max(cfg.num_steps, 1))
    total_reward = torch.where(controlled & valid, total_reward, torch.zeros_like(total_reward))

    info = {
        "collision": collision,
        "offroad": offroad,
        "ttc_alert": ttc_alert,
        "ttc": ttc,
        "goal_reached": goal_first,
        "goal_distance": goal_dist,
        "comfort_score": comfort_score,
        "collision_reward": collision_r,
        "offroad_reward": offroad_r,
        "goal_reward": goal_r,
        "ttc_score": ttc_score,
        "pair_collision": pair_collision,
    }
    return total_reward, done, info, next_goal_reached
