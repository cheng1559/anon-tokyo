"""Closed-loop reward components for simplified WOMD simulation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor

from anon_tokyo.simulation.dynamics import get_wheelbase_from_length
from anon_tokyo.simulation.profiling import TimingProfiler


LANE_CENTER_TYPES = (1, 2, 3)
SOLID_LINE_TYPES = (7, 8, 11, 12, 13)
ROAD_EDGE_TYPES = (15, 16)
VEHICLE_TYPE = 1
PEDESTRIAN_TYPE = 2
CYCLIST_TYPE = 3


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
    ttc_min_eval_speed: float = 0.75
    ttc_num_iters: int = 5
    tto_horizon: float = 1.5
    tto_reward_floor: float = 0.5
    tto_num_iters: int = 4
    centerline_weight: float = 0.8
    centerline_distance_limit: float = 0.5
    solid_line_weight: float = 0.4
    solid_line_distance_threshold: float = 0.25
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


def _points_in_convex_polygon(points: Tensor, polygons: Tensor, eps: float = 1e-6) -> Tensor:
    edges = torch.roll(polygons, shifts=-1, dims=1) - polygons
    rel = points.unsqueeze(1) - polygons
    cross = edges[..., 0] * rel[..., 1] - edges[..., 1] * rel[..., 0]
    return (cross >= -eps).all(dim=-1) | (cross <= eps).all(dim=-1)


def _segments_intersect(a0: Tensor, a1: Tensor, b0: Tensor, b1: Tensor, eps: float = 1e-6) -> Tensor:
    def orient(p: Tensor, q: Tensor, r: Tensor) -> Tensor:
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

    o1 = orient(a0, a1, b0)
    o2 = orient(a0, a1, b1)
    o3 = orient(b0, b1, a0)
    o4 = orient(b0, b1, a1)
    return (o1 * o2 <= eps) & (o3 * o4 <= eps)


def _segment_polygon_intersection(segments: Tensor, polygons: Tensor) -> Tensor:
    """Pairwise intersection for ``segments [N,2,2]`` and convex polygons ``[N,4,2]``."""
    if segments.numel() == 0:
        return torch.zeros(segments.shape[0], dtype=torch.bool, device=segments.device)
    s0, s1 = segments[:, 0], segments[:, 1]
    inside = _points_in_convex_polygon(s0, polygons) | _points_in_convex_polygon(s1, polygons)
    p0 = polygons
    p1 = torch.roll(polygons, shifts=-1, dims=1)
    edge_hit = _segments_intersect(s0[:, None], s1[:, None], p0, p1).any(dim=-1)
    return inside | edge_hit


def _expand_polygon_with_buffer(polygons: Tensor, width_buffer: float, length_buffer: float) -> Tensor:
    if width_buffer <= 0.0 and length_buffer <= 0.0:
        return polygons
    center = polygons.mean(dim=1, keepdim=True)
    rel = polygons - center
    length_axis = polygons[:, 0] - polygons[:, 3]
    width_axis = polygons[:, 0] - polygons[:, 1]
    length_norm = length_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    width_norm = width_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    length_scale = (length_norm + 2.0 * length_buffer) / length_norm
    width_scale = (width_norm + 2.0 * width_buffer) / width_norm
    forward = length_axis / length_norm
    left = width_axis / width_norm
    rel_f = (rel * forward[:, None]).sum(dim=-1, keepdim=True) * forward[:, None] * length_scale[:, None]
    rel_l = (rel * left[:, None]).sum(dim=-1, keepdim=True) * left[:, None] * width_scale[:, None]
    return center + rel_f + rel_l


def _motion_segments(polygon: Tensor, delta_pos: Tensor) -> Tensor:
    device = polygon.device
    v1 = polygon[:, 1] - polygon[:, 0]
    v2 = polygon[:, 2] - polygon[:, 1]
    proj1 = (delta_pos * v1).sum(dim=1)
    proj2 = (delta_pos * v2).sum(dim=1)
    mask1 = (proj1 >= 0) & (proj2 >= 0)
    mask2 = (proj1 < 0) & (proj2 >= 0)
    mask3 = (proj1 < 0) & (proj2 < 0)
    mask4 = (proj1 >= 0) & (proj2 < 0)
    shifted = polygon + delta_pos.unsqueeze(1)
    all_points = torch.cat((polygon, shifted), dim=1)
    idx1 = torch.tensor([1, 0, 3, 7, 6, 5], device=device, dtype=torch.long)
    idx2 = torch.tensor([2, 1, 0, 4, 7, 6], device=device, dtype=torch.long)
    idx3 = torch.tensor([1, 2, 3, 7, 4, 5], device=device, dtype=torch.long)
    idx4 = torch.tensor([2, 3, 0, 4, 5, 6], device=device, dtype=torch.long)
    indices = mask1.long().unsqueeze(1) * idx1 + mask2.long().unsqueeze(1) * idx2 + mask3.long().unsqueeze(1) * idx3 + mask4.long().unsqueeze(1) * idx4
    hull = torch.gather(all_points, 1, indices.unsqueeze(-1).expand(-1, -1, 2))
    return torch.stack((hull, torch.roll(hull, shifts=-1, dims=1)), dim=2)


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


def _batched_point_segment_distance(points: Tensor, start: Tensor, end: Tensor, seg_valid: Tensor) -> Tensor:
    """Distance from ``points [Q,N,2]`` to same-batch segments ``[Q,S,2]``."""
    seg = end - start
    rel = points[:, :, None] - start[:, None]
    denom = (seg * seg).sum(dim=-1).clamp_min(1e-7)
    u = ((rel * seg[:, None]).sum(dim=-1) / denom[:, None]).clamp(0.0, 1.0)
    proj = start[:, None] + u[..., None] * seg[:, None]
    dist = (points[:, :, None] - proj).norm(dim=-1)
    return dist.masked_fill(~seg_valid[:, None], float("inf"))


def _profile(profiler: TimingProfiler | None, name: str):
    return profiler.record(name) if profiler is not None else nullcontext()


def offroad_reward(
    polygons: Tensor,
    valid: Tensor,
    controlled: Tensor,
    map_polylines: Tensor,
    map_mask: Tensor,
    cfg: RewardConfig,
    profiler: TimingProfiler | None = None,
):
    B, A = valid.shape
    out = torch.zeros(B, A, dtype=torch.bool, device=valid.device)
    active = controlled & valid
    batch_idx, agent_idx = active.nonzero(as_tuple=True)
    if batch_idx.numel() == 0:
        return out.float() * cfg.offroad_reward_weight, out

    with _profile(profiler, "reward.offroad.prepare_map"):
        types = map_polylines[..., 6].round().long()
        boundary_values = torch.tensor(ROAD_EDGE_TYPES, dtype=types.dtype, device=types.device)
        boundary_type = torch.isin(types, boundary_values)
        point_mask = map_mask.bool() & boundary_type
        seg_valid = (point_mask[:, :, :-1] & point_mask[:, :, 1:]).flatten(1)
        starts = map_polylines[:, :, :-1, 0:2].reshape(B, -1, 2)
        ends = map_polylines[:, :, 1:, 0:2].reshape(B, -1, 2)

    with _profile(profiler, "reward.offroad.distance"):
        agent_polys = polygons[batch_idx, agent_idx]
        agent_points = torch.cat((agent_polys, agent_polys.mean(dim=1, keepdim=True)), dim=1)
        dist = _batched_point_segment_distance(
            agent_points,
            starts[batch_idx],
            ends[batch_idx],
            seg_valid[batch_idx],
        )
        min_dist = dist.amin(dim=2).amin(dim=1)
        out[batch_idx, agent_idx] = min_dist <= cfg.offroad_distance_threshold

    reward = out.float() * cfg.offroad_reward_weight
    return reward, out


def _eval_velocity_with_heading_fallback(velocities: Tensor, headings: Tensor, enforce_min_speed: bool, cfg: RewardConfig) -> Tensor:
    norm = velocities.norm(dim=-1, keepdim=True)
    vel_dir = velocities / norm.clamp_min(1e-6)
    heading_dir = torch.stack((headings.cos(), headings.sin()), dim=-1)
    direction = torch.where(norm < cfg.ttc_min_eval_speed, heading_dir, vel_dir)
    speed = norm.clamp_min(cfg.ttc_min_eval_speed) if enforce_min_speed else norm
    return direction * speed


def _compute_ttc(polygons: Tensor, positions: Tensor, velocities: Tensor, headings: Tensor, valid: Tensor, controlled: Tensor, cfg: RewardConfig) -> Tensor:
    B, A = valid.shape
    batch_idx, agent_idx = (controlled & valid).nonzero(as_tuple=True)
    ttc = positions.new_full((B, A), cfg.ttc_horizon)
    if batch_idx.numel() == 0 or A <= 1:
        return ttc

    other_idx = torch.arange(A, device=positions.device).expand(batch_idx.numel(), A)
    pair_agent = agent_idx[:, None].expand_as(other_idx)
    pair_batch = batch_idx[:, None].expand_as(other_idx)
    pair_valid = valid[pair_batch, other_idx] & (other_idx != pair_agent)
    pair_flat = pair_valid.reshape(-1)
    if not pair_flat.any():
        return ttc

    pair_batch = pair_batch.reshape(-1)[pair_flat]
    pair_agent = pair_agent.reshape(-1)[pair_flat]
    other_agent = other_idx.reshape(-1)[pair_flat]
    controlled_flat = torch.arange(batch_idx.numel(), device=positions.device)[:, None].expand(-1, A).reshape(-1)[pair_flat]

    ego_poly = _expand_polygon_with_buffer(polygons[pair_batch, pair_agent], cfg.ttc_radius_buffer, cfg.ttc_radius_buffer)
    other_poly = polygons[pair_batch, other_agent]
    v1 = _eval_velocity_with_heading_fallback(velocities[pair_batch, pair_agent], headings[pair_batch, pair_agent], True, cfg)
    v2 = _eval_velocity_with_heading_fallback(velocities[pair_batch, other_agent], headings[pair_batch, other_agent], False, cfg)
    rel_v = v2 - v1
    horizon = positions.new_full((pair_batch.numel(),), cfg.ttc_horizon)
    ego_for_segments = ego_poly.unsqueeze(1).expand(-1, 6, -1, -1).reshape(-1, 4, 2)

    def collision_at_time(t_vec: Tensor) -> Tensor:
        segments = _motion_segments(other_poly, rel_v * t_vec.unsqueeze(-1)).reshape(-1, 2, 2)
        return _segment_polygon_intersection(segments, ego_for_segments).view(-1, 6).any(dim=-1)

    active = collision_at_time(horizon)
    low = torch.zeros_like(horizon)
    high = horizon.clone()
    for _ in range(cfg.ttc_num_iters):
        mid = 0.5 * (low + high)
        hit = collision_at_time(mid)
        high = torch.where(active & hit, mid, high)
        low = torch.where(active & ~hit, mid, low)

    t_pair = torch.where(active, high, horizon)
    t_min = torch.full((batch_idx.numel(),), float("inf"), device=positions.device, dtype=positions.dtype)
    t_min = t_min.scatter_reduce(0, controlled_flat, t_pair, reduce="amin", include_self=True)
    ttc[batch_idx, agent_idx] = torch.minimum(t_min, positions.new_full(t_min.shape, cfg.ttc_horizon))
    return ttc


def ttc_reward(positions: Tensor, velocities: Tensor, headings: Tensor, sizes: Tensor, valid: Tensor, controlled: Tensor, cfg: RewardConfig):
    polygons = agent_polygons(positions, headings, sizes)
    ttc = _compute_ttc(polygons, positions, velocities, headings, valid, controlled, cfg)
    alert = (ttc < cfg.ttc_horizon) & controlled & valid
    score = cfg.ttc_reward_floor + (1.0 - cfg.ttc_reward_floor) * (ttc / cfg.ttc_horizon).clamp(0.0, 1.0)
    score = torch.where(alert, score, torch.ones_like(score))
    return score, alert, ttc


def tto_reward(
    polygons: Tensor,
    positions: Tensor,
    velocities: Tensor,
    headings: Tensor,
    sizes: Tensor,
    steering: Tensor,
    valid: Tensor,
    controlled: Tensor,
    map_polylines: Tensor,
    map_mask: Tensor,
    cfg: RewardConfig,
):
    B, A = valid.shape
    tto = positions.new_full((B, A), cfg.tto_horizon)
    active = controlled & valid
    batch_idx, agent_idx = active.nonzero(as_tuple=True)
    if batch_idx.numel() == 0:
        return torch.ones_like(tto), torch.zeros_like(active), tto

    types = map_polylines[..., 6].round().long()
    edge_values = torch.tensor(ROAD_EDGE_TYPES, dtype=types.dtype, device=types.device)
    point_mask = map_mask.bool() & torch.isin(types, edge_values)
    seg_valid = (point_mask[:, :, :-1] & point_mask[:, :, 1:]).flatten(1)
    starts = map_polylines[:, :, :-1, 0:2].reshape(B, -1, 2)
    ends = map_polylines[:, :, 1:, 0:2].reshape(B, -1, 2)
    if not seg_valid.any():
        return torch.ones_like(tto), torch.zeros_like(active), tto

    heading_dir = torch.stack((headings.cos(), headings.sin()), dim=-1)
    speed = (velocities * heading_dir).sum(dim=-1).clamp_min(0.0)
    poly = _expand_polygon_with_buffer(polygons[batch_idx, agent_idx], cfg.ttc_radius_buffer, cfg.ttc_radius_buffer)
    horizon = positions.new_full((batch_idx.numel(),), cfg.tto_horizon)
    ctrl_heading = headings[batch_idx, agent_idx]
    ctrl_speed = speed[batch_idx, agent_idx]
    ctrl_steering = steering[batch_idx, agent_idx]
    wheelbase = get_wheelbase_from_length(sizes[batch_idx, agent_idx, 0]).clamp_min(1e-6)
    curvature = torch.tan(ctrl_steering) / wheelbase
    straight = curvature.abs() < 1e-4
    curvature_sign = torch.where(curvature >= 0.0, 1.0, -1.0)
    curvature_safe = curvature_sign * curvature.abs().clamp_min(1e-4)
    left_normal = torch.stack((-ctrl_heading.sin(), ctrl_heading.cos()), dim=-1)
    icr = poly.mean(dim=1) + left_normal / curvature_safe.unsqueeze(-1)

    def future_polygon(t_vec: Tensor) -> Tensor:
        straight_delta = heading_dir[batch_idx, agent_idx] * (ctrl_speed * t_vec).unsqueeze(-1)
        straight_poly = poly + straight_delta.unsqueeze(1)
        delta_yaw = ctrl_speed * curvature_safe * t_vec
        rel = poly - icr.unsqueeze(1)
        cos_yaw = torch.cos(delta_yaw).unsqueeze(-1)
        sin_yaw = torch.sin(delta_yaw).unsqueeze(-1)
        rel_x = rel[..., 0]
        rel_y = rel[..., 1]
        curved_rel = torch.stack((cos_yaw * rel_x - sin_yaw * rel_y, sin_yaw * rel_x + cos_yaw * rel_y), dim=-1)
        curved_poly = icr.unsqueeze(1) + curved_rel
        return torch.where(straight[:, None, None], straight_poly, curved_poly)

    def collision_at_time(t_vec: Tensor) -> Tensor:
        future_poly = future_polygon(t_vec)
        flat_segments = torch.stack((starts[batch_idx], ends[batch_idx]), dim=2)
        num_segments = flat_segments.shape[1]
        repeated_poly = future_poly.unsqueeze(1).expand(-1, num_segments, -1, -1).reshape(-1, 4, 2)
        hits = _segment_polygon_intersection(flat_segments.reshape(-1, 2, 2), repeated_poly).view(-1, num_segments)
        return (hits & seg_valid[batch_idx]).any(dim=-1)

    active_h = collision_at_time(horizon)
    low = torch.zeros_like(horizon)
    high = horizon.clone()
    for _ in range(cfg.tto_num_iters):
        mid = 0.5 * (low + high)
        hit = collision_at_time(mid)
        high = torch.where(active_h & hit, mid, high)
        low = torch.where(active_h & ~hit, mid, low)

    tto[batch_idx, agent_idx] = torch.where(active_h, high, horizon)
    alert = (tto < cfg.tto_horizon) & controlled & valid
    score = cfg.tto_reward_floor + (1.0 - cfg.tto_reward_floor) * (tto / cfg.tto_horizon).clamp(0.0, 1.0)
    return torch.where(alert, score, torch.ones_like(score)), alert, tto


def goal_reaching_reward(positions: Tensor, goals: Tensor, goal_reached: Tensor, controlled: Tensor, valid: Tensor, cfg: RewardConfig):
    dist = (positions - goals).norm(dim=-1)
    reached = (dist <= cfg.goal_reaching_threshold) & controlled & valid
    first_reached = reached & ~goal_reached
    reward = torch.where(first_reached, torch.full_like(dist, cfg.goal_reaching_weight), torch.zeros_like(dist))
    next_goal_reached = goal_reached | reached
    return reward, first_reached, next_goal_reached, dist


def centerline_reward(
    positions: Tensor,
    valid: Tensor,
    controlled: Tensor,
    map_polylines: Tensor,
    map_mask: Tensor,
    cfg: RewardConfig,
):
    B, A = valid.shape
    min_dist = positions.new_full((B, A), float("inf"))
    active = controlled & valid
    batch_idx, agent_idx = active.nonzero(as_tuple=True)
    if batch_idx.numel() == 0:
        return torch.ones(B, A, dtype=positions.dtype, device=positions.device), min_dist

    types = map_polylines[..., 6].round().long()
    center_values = torch.tensor(LANE_CENTER_TYPES, dtype=types.dtype, device=types.device)
    point_mask = map_mask.bool() & torch.isin(types, center_values)
    seg_valid = (point_mask[:, :, :-1] & point_mask[:, :, 1:]).flatten(1)
    starts = map_polylines[:, :, :-1, 0:2].reshape(B, -1, 2)
    ends = map_polylines[:, :, 1:, 0:2].reshape(B, -1, 2)
    if not seg_valid.any():
        return torch.ones(B, A, dtype=positions.dtype, device=positions.device), min_dist

    pts = positions[batch_idx, agent_idx].unsqueeze(1)
    dist = _batched_point_segment_distance(pts, starts[batch_idx], ends[batch_idx], seg_valid[batch_idx]).squeeze(1)
    best_dist = dist.min(dim=-1).values
    min_dist[batch_idx, agent_idx] = best_dist

    has_centerline = torch.isfinite(best_dist)

    t = (best_dist / max(cfg.centerline_distance_limit, 1e-6)).clamp(0.0, 1.0)
    score_active = 1.0 - (1.0 - cfg.centerline_weight) * t
    score = torch.ones(B, A, dtype=positions.dtype, device=positions.device)
    score[batch_idx, agent_idx] = torch.where(has_centerline, score_active, torch.ones_like(score_active))
    return score, min_dist


def solid_line_reward(
    polygons: Tensor,
    valid: Tensor,
    controlled: Tensor,
    map_polylines: Tensor,
    map_mask: Tensor,
    cfg: RewardConfig,
):
    B, A = valid.shape
    crossed = torch.zeros(B, A, dtype=torch.bool, device=valid.device)
    active = controlled & valid
    batch_idx, agent_idx = active.nonzero(as_tuple=True)
    if batch_idx.numel() == 0:
        return polygons.new_ones(B, A), crossed

    types = map_polylines[..., 6].round().long()
    solid_values = torch.tensor(SOLID_LINE_TYPES, dtype=types.dtype, device=types.device)
    point_mask = map_mask.bool() & torch.isin(types, solid_values)
    seg_valid = (point_mask[:, :, :-1] & point_mask[:, :, 1:]).flatten(1)
    starts = map_polylines[:, :, :-1, 0:2].reshape(B, -1, 2)
    ends = map_polylines[:, :, 1:, 0:2].reshape(B, -1, 2)
    if not seg_valid.any():
        return polygons.new_ones(B, A), crossed

    poly = _expand_polygon_with_buffer(polygons[batch_idx, agent_idx], cfg.solid_line_distance_threshold, 0.0)
    flat_segments = torch.stack((starts[batch_idx], ends[batch_idx]), dim=2)
    num_segments = flat_segments.shape[1]
    repeated_poly = poly.unsqueeze(1).expand(-1, num_segments, -1, -1).reshape(-1, 4, 2)
    hits = _segment_polygon_intersection(flat_segments.reshape(-1, 2, 2), repeated_poly).view(-1, num_segments)
    crossed_active = (hits & seg_valid[batch_idx]).any(dim=-1)
    crossed[batch_idx, agent_idx] = crossed_active
    score = polygons.new_ones(B, A)
    score[batch_idx, agent_idx] = torch.where(crossed_active, polygons.new_full(crossed_active.shape, cfg.solid_line_weight), score[batch_idx, agent_idx])
    return score, crossed


def comfort_reward(a_long: Tensor, a_lat: Tensor, jerk_long: Tensor, jerk_lat: Tensor, controlled: Tensor, valid: Tensor, cfg: RewardConfig):
    long_acc = (a_long.abs() / max(cfg.comfort_long_accel, 1e-6)).clamp(max=1.0)
    lat_acc = (a_lat.abs() / max(cfg.comfort_lat_accel, 1e-6)).clamp(max=1.0)
    long_jerk = (jerk_long.abs() / max(cfg.comfort_long_jerk, 1e-6)).clamp(max=1.0)
    lat_jerk = (jerk_lat.abs() / max(cfg.comfort_lat_jerk, 1e-6)).clamp(max=1.0)
    penalty = torch.stack((long_acc, lat_acc, long_jerk, lat_jerk), dim=-1).amax(dim=-1)
    score = 1.0 - (1.0 - cfg.comfort_weight) * penalty.square()
    return torch.where(controlled & valid, score, torch.ones_like(score))


def compute_rewards(
    state: dict[str, Tensor],
    cfg: RewardConfig,
    profiler: TimingProfiler | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor]:
    """Compute combined reward, done flags, component info, and next goal flags."""
    valid = state["valid_mask"].bool()
    controlled = state["controlled_mask"].bool()
    obj_types = state.get("obj_types")
    if obj_types is None:
        obj_types = torch.full_like(valid, VEHICLE_TYPE, dtype=torch.long)
    else:
        obj_types = obj_types.to(device=valid.device).long()
    vehicle_mask = obj_types == VEHICLE_TYPE
    vulnerable_mask = (obj_types == PEDESTRIAN_TYPE) | (obj_types == CYCLIST_TYPE)
    with _profile(profiler, "reward.collision"):
        collision_r, collision, pair_collision, polygons = collision_reward(
            state["positions"], state["headings"], state["sizes"], valid, controlled, cfg
        )
    with _profile(profiler, "reward.offroad"):
        offroad_r, offroad = offroad_reward(
            polygons,
            valid,
            controlled & vehicle_mask,
            state["map_polylines"],
            state["map_polylines_mask"],
            cfg,
            profiler=profiler,
        )
    with _profile(profiler, "reward.ttc"):
        ttc_score, ttc_alert, ttc = ttc_reward(
            state["positions"], state["velocities"], state["headings"], state["sizes"], valid, controlled, cfg
        )
    with _profile(profiler, "reward.tto"):
        tto_score, tto_alert, tto = tto_reward(
            polygons,
            state["positions"],
            state["velocities"],
            state["headings"],
            state["sizes"],
            state.get("steering", torch.zeros_like(state["headings"])),
            valid,
            controlled,
            state["map_polylines"],
            state["map_polylines_mask"],
            cfg,
        )
    with _profile(profiler, "reward.goal"):
        goal_r, goal_first, next_goal_reached, goal_dist = goal_reaching_reward(
            state["positions"], state["goal_positions"], state["goal_reached"], controlled, valid, cfg
        )
    with _profile(profiler, "reward.centerline"):
        centerline_score, centerline_dist = centerline_reward(
            state["positions"],
            valid,
            controlled & vehicle_mask,
            state["map_polylines"],
            state["map_polylines_mask"],
            cfg,
        )
    with _profile(profiler, "reward.solid_line"):
        solid_line_score, solid_line_crossed = solid_line_reward(
            polygons,
            valid,
            controlled & vehicle_mask,
            state["map_polylines"],
            state["map_polylines_mask"],
            cfg,
        )
    with _profile(profiler, "reward.comfort"):
        comfort_score = comfort_reward(
            state["a_long"], state["a_lat"], state["jerk_long"], state["jerk_lat"], controlled, valid, cfg
        )

    with _profile(profiler, "reward.combine"):
        hard_reward = collision_r + offroad_r
        done = (collision | (offroad & vehicle_mask)) & controlled & valid

        vehicle_soft_score = comfort_score * ttc_score * tto_score * centerline_score * solid_line_score
        vulnerable_soft_score = ttc_score * tto_score
        soft_score = torch.where(vulnerable_mask, vulnerable_soft_score, vehicle_soft_score)
        alive = (~done).float()
        total_reward = hard_reward + alive * (goal_r + soft_score / max(cfg.num_steps, 1))
        total_reward = torch.where(controlled & valid, total_reward, torch.zeros_like(total_reward))

    info = {
        "collision": collision,
        "offroad": offroad,
        "ttc_alert": ttc_alert,
        "ttc": ttc,
        "tto_alert": tto_alert,
        "tto": tto,
        "goal_reached": goal_first,
        "goal_distance": goal_dist,
        "comfort_score": comfort_score,
        "centerline_score": centerline_score,
        "centerline_distance": centerline_dist,
        "solid_line_score": solid_line_score,
        "solid_line_crossed": solid_line_crossed,
        "collision_reward": collision_r,
        "offroad_reward": offroad_r,
        "goal_reward": goal_r,
        "ttc_score": ttc_score,
        "tto_score": tto_score,
        "pair_collision": pair_collision,
    }
    return total_reward, done, info, next_goal_reached
