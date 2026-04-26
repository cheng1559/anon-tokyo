"""Continuous jerk-PNC dynamics for closed-loop WOMD simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap radians to [-pi, pi]."""
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


@dataclass
class JerkPncConfig:
    dt: float = 0.1
    min_jerk_long: float = -5.0
    max_jerk_long: float = 3.0
    min_jerk_lat: float = -1.0
    max_jerk_lat: float = 1.0
    min_a_long: float = -6.5
    max_a_long: float = 1.5
    min_a_lat: float = -3.0
    max_a_lat: float = 3.0
    min_speed: float = 0.0
    max_speed: float = 40.0
    lateral_speed_floor: float = 1.0


class JerkPncModel:
    """Simplified jerk-actuated PNC dynamics.

    Actions are continuous ``[jerk_long, jerk_lat]``.  The lateral action always
    means lateral jerk; there is no low-speed switch to steering semantics.
    """

    def __init__(self, config: JerkPncConfig | None = None) -> None:
        self.config = config or JerkPncConfig()

    @property
    def action_low(self) -> Tensor:
        cfg = self.config
        return torch.tensor([cfg.min_jerk_long, cfg.min_jerk_lat], dtype=torch.float32)

    @property
    def action_high(self) -> Tensor:
        cfg = self.config
        return torch.tensor([cfg.max_jerk_long, cfg.max_jerk_lat], dtype=torch.float32)

    def step(
        self,
        positions: Tensor,
        velocities: Tensor,
        headings: Tensor,
        a_long: Tensor,
        a_lat: Tensor,
        yaw_rate: Tensor,
        actions: Tensor,
    ) -> dict[str, Tensor]:
        """Advance one time step for all agents in the tensors."""
        cfg = self.config
        dt = cfg.dt

        jerk_long = actions[..., 0].clamp(cfg.min_jerk_long, cfg.max_jerk_long)
        jerk_lat = actions[..., 1].clamp(cfg.min_jerk_lat, cfg.max_jerk_lat)

        heading_dir = torch.stack((headings.cos(), headings.sin()), dim=-1)
        v_long = (velocities * heading_dir).sum(dim=-1).clamp_min(cfg.min_speed)

        next_a_long = (a_long + jerk_long * dt).clamp(cfg.min_a_long, cfg.max_a_long)
        next_a_lat = (a_lat + jerk_lat * dt).clamp(cfg.min_a_lat, cfg.max_a_lat)

        next_v_long = (v_long + 0.5 * (a_long + next_a_long) * dt).clamp(cfg.min_speed, cfg.max_speed)
        next_yaw_rate = next_a_lat / next_v_long.clamp_min(cfg.lateral_speed_floor)
        next_headings = wrap_angle(headings + 0.5 * (yaw_rate + next_yaw_rate) * dt)

        next_heading_dir = torch.stack((next_headings.cos(), next_headings.sin()), dim=-1)
        next_velocities = next_v_long.unsqueeze(-1) * next_heading_dir
        next_positions = positions + 0.5 * (velocities + next_velocities) * dt

        return {
            "positions": next_positions,
            "velocities": next_velocities,
            "headings": next_headings,
            "a_long": next_a_long,
            "a_lat": next_a_lat,
            "yaw_rate": next_yaw_rate,
            "jerk_long": jerk_long,
            "jerk_lat": jerk_lat,
        }


def infer_log_kinematics(log_trajs: Tensor, timestamps: Tensor | None = None, default_dt: float = 0.1) -> dict[str, Tensor]:
    """Infer longitudinal/lateral acceleration and yaw rate from log states."""
    positions = log_trajs[..., 0:2]
    velocities = log_trajs[..., 7:9]
    headings = log_trajs[..., 6]
    mask = log_trajs[..., 9] > 0
    device = log_trajs.device
    dtype = log_trajs.dtype

    if timestamps is not None and timestamps.numel() >= 2:
        if timestamps.ndim == 2:
            dt_vec = (timestamps[:, 1:] - timestamps[:, :-1]).to(device=device, dtype=dtype)
            dt = dt_vec.clamp_min(1e-3)[:, None, :]
        else:
            dt_vec = (timestamps[1:] - timestamps[:-1]).to(device=device, dtype=dtype)
            dt = dt_vec.clamp_min(1e-3).view(1, 1, -1)
    else:
        dt = torch.full((*log_trajs.shape[:2], log_trajs.shape[2] - 1), default_dt, device=device, dtype=dtype)

    heading_dir = torch.stack((headings.cos(), headings.sin()), dim=-1)
    v_long = (velocities * heading_dir).sum(dim=-1)
    yaw_delta = wrap_angle(headings[..., 1:] - headings[..., :-1])

    if dt.ndim == 3 and dt.shape[0] == 1:
        dt = dt.expand(log_trajs.shape[0], log_trajs.shape[1], -1)
    elif dt.ndim == 3 and dt.shape[1] == 1:
        dt = dt.expand(-1, log_trajs.shape[1], -1)

    a_long_delta = (v_long[..., 1:] - v_long[..., :-1]) / dt
    yaw_rate_delta = yaw_delta / dt
    valid_delta = mask[..., 1:] & mask[..., :-1]
    a_long_delta = torch.where(valid_delta, a_long_delta, torch.zeros_like(a_long_delta))
    yaw_rate_delta = torch.where(valid_delta, yaw_rate_delta, torch.zeros_like(yaw_rate_delta))

    zeros = torch.zeros_like(v_long[..., :1])
    a_long = torch.cat((zeros, a_long_delta), dim=-1)
    yaw_rate = torch.cat((zeros, yaw_rate_delta), dim=-1)
    a_lat = v_long.abs() * yaw_rate

    return {
        "positions": positions,
        "velocities": velocities,
        "headings": headings,
        "a_long": a_long,
        "a_lat": a_lat,
        "yaw_rate": yaw_rate,
    }
