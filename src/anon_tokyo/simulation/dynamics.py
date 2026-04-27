"""Continuous jerk-PNC dynamics for closed-loop WOMD simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

PNC_LATERAL = (
    -221.76124992,
    -1.0,
    32.47684285,
    97.76347776,
    -9.81,
    18.16816182,
    0.0,
    -277.88361484,
    83.73152838,
    0.0,
)
MIN_DYNAMIC_SPEED = 1.5
EPS = 1e-6
MIN_STEERING_ANGLE = math.radians(-432.0) / 12.6
MAX_STEERING_ANGLE = math.radians(432.0) / 12.6


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap radians to [-pi, pi]."""
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def get_wheelbase_from_length(length: Tensor) -> Tensor:
    """Estimate wheelbase from vehicle length using the Hermes convention."""
    return (length * 3.0 / 4.995).clamp(min=1.5)


def velocity_body_frame_components(velocities: Tensor, headings: Tensor) -> tuple[Tensor, Tensor]:
    """Return longitudinal and lateral velocity in the heading-aligned body frame."""
    forward = torch.stack((headings.cos(), headings.sin()), dim=-1)
    left = torch.stack((-headings.sin(), headings.cos()), dim=-1)
    return (velocities * forward).sum(dim=-1), (velocities * left).sum(dim=-1)


def _stabilize_tensor(value: Tensor, eps: float) -> Tensor:
    abs_value = value.abs()
    sign = torch.sign(value)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return torch.where(abs_value < eps, sign * eps, value)


def _apply_jerk_and_clamp_acceleration(
    current_acceleration: Tensor,
    jerk: Tensor,
    dt: float,
    min_acceleration: float,
    max_acceleration: float,
    *,
    positive_jerk_limit_when_negative_acc: float | None = None,
    positive_jerk_limit_when_nonnegative_acc: float | None = None,
) -> Tensor:
    effective_jerk = jerk
    if positive_jerk_limit_when_negative_acc is not None and positive_jerk_limit_when_nonnegative_acc is not None:
        positive_limit = torch.where(
            current_acceleration < 0,
            torch.full_like(jerk, positive_jerk_limit_when_negative_acc),
            torch.full_like(jerk, positive_jerk_limit_when_nonnegative_acc),
        )
        effective_jerk = torch.where(jerk > 0, torch.minimum(jerk, positive_limit), jerk)
    return (current_acceleration + effective_jerk * dt).clamp(min_acceleration, max_acceleration)


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
    min_steering_angle: float = MIN_STEERING_ANGLE
    max_steering_angle: float = MAX_STEERING_ANGLE
    max_tire_angle_rate_lower_bound: float = 0.03
    max_tire_angle_rate_upper_bound: float = 0.3 * 0.7
    max_tire_angle_rate_gain: float = 1.39 * 0.7
    lateral_mode_transition_speed: float = 5.0
    lateral_mode_transition_width: float = 0.3
    curvature_speed_floor: float = 1.0
    min_dynamic_speed: float = MIN_DYNAMIC_SPEED
    positive_jerk_limit_when_negative_acc: float | None = None
    positive_jerk_limit_when_nonnegative_acc: float | None = 1.0


class JerkPncModel:
    """Hermes-style jerk-actuated PNC dynamics.

    Actions are continuous ``[jerk_long, lat_command]``.  The lateral command
    matches Hermes: low speed interprets it as normalized steering-rate command,
    high speed interprets it as lateral jerk, with a smooth speed blend.
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

    def _delta_steering_rate_bound(self, speed: Tensor) -> Tensor:
        cfg = self.config
        speed = speed.clamp_min(1e-3)
        return (cfg.max_tire_angle_rate_gain / speed).clamp(
            min=cfg.max_tire_angle_rate_lower_bound,
            max=cfg.max_tire_angle_rate_upper_bound,
        )

    def _high_speed_lateral_weight(self, speed: Tensor) -> Tensor:
        cfg = self.config
        width = max(float(cfg.lateral_mode_transition_width), 1e-3)
        return torch.sigmoid((speed - float(cfg.lateral_mode_transition_speed)) / width)

    def _lat_command_to_steering_rate(self, lat_command: Tensor, speed: Tensor) -> Tensor:
        cfg = self.config
        scale = max(abs(float(cfg.min_jerk_lat)), abs(float(cfg.max_jerk_lat)), 1e-3)
        lat_cmd_norm = (lat_command / scale).clamp(-1.0, 1.0)
        return lat_cmd_norm * self._delta_steering_rate_bound(speed)

    def _compute_lateral_target_steering(
        self,
        lat_command: Tensor,
        speed: Tensor,
        steering: Tensor,
        wheelbase: Tensor,
        v_long: Tensor,
        a_long: Tensor,
        next_a_long_control: Tensor,
        dt: float,
    ) -> Tensor:
        cfg = self.config
        current_curvature = torch.tan(steering) / wheelbase
        current_a_lat = v_long.square() * current_curvature
        target_a_lat = _apply_jerk_and_clamp_acceleration(
            current_a_lat,
            lat_command,
            dt,
            cfg.min_a_lat,
            cfg.max_a_lat,
        )
        pred_v_long = (v_long + 0.5 * (a_long + next_a_long_control) * dt).clamp(cfg.min_speed, cfg.max_speed)
        target_curvature = target_a_lat / pred_v_long.square().clamp_min(cfg.curvature_speed_floor)
        target_steering_from_jerk = torch.atan(target_curvature * wheelbase)

        steering_rate_cmd = self._lat_command_to_steering_rate(lat_command, speed)
        target_steering_from_rate = steering + steering_rate_cmd * dt

        jerk_weight = self._high_speed_lateral_weight(speed)
        return jerk_weight * target_steering_from_jerk + (1.0 - jerk_weight) * target_steering_from_rate

    @staticmethod
    def _dynamic_lateral_update(
        dt: float,
        v_long: Tensor,
        v_lat: Tensor,
        yaw_rate: Tensor,
        steering: Tensor,
    ) -> tuple[Tensor, Tensor]:
        params = torch.as_tensor(PNC_LATERAL, dtype=v_long.dtype, device=v_long.device)
        v_long_used = _stabilize_tensor(v_long, MIN_DYNAMIC_SPEED * 0.9)

        a = _stabilize_tensor(params[0] / v_long_used, EPS)
        p = (params[1] * v_long_used + params[2] / v_long_used) * yaw_rate + params[3] * steering
        exp_a = torch.exp(a * dt)
        next_v_lat = exp_a * v_lat + (exp_a - 1.0) * p / a

        b = _stabilize_tensor(params[6] * v_long_used + params[7] / v_long_used, EPS)
        q = params[5] / v_long_used * v_lat + params[8] * steering
        exp_b = torch.exp(b * dt)
        next_yaw_rate = exp_b * yaw_rate + (exp_b - 1.0) * q / b
        return next_v_lat, next_yaw_rate

    def step(
        self,
        positions: Tensor,
        velocities: Tensor,
        headings: Tensor,
        sizes: Tensor,
        a_long: Tensor,
        a_lat: Tensor,
        steering: Tensor,
        yaw_rate: Tensor,
        actions: Tensor,
    ) -> dict[str, Tensor]:
        """Advance one time step for all agents in the tensors."""
        cfg = self.config
        dt = cfg.dt

        jerk_long = actions[..., 0].clamp(cfg.min_jerk_long, cfg.max_jerk_long)
        lat_command = actions[..., 1].clamp(cfg.min_jerk_lat, cfg.max_jerk_lat)

        v_long, v_lat = velocity_body_frame_components(velocities, headings)
        speed = velocities.norm(dim=-1)
        wheelbase = get_wheelbase_from_length(sizes[..., 0])

        next_a_long = _apply_jerk_and_clamp_acceleration(
            a_long,
            jerk_long,
            dt,
            cfg.min_a_long,
            cfg.max_a_long,
            positive_jerk_limit_when_negative_acc=(
                cfg.max_jerk_long
                if cfg.positive_jerk_limit_when_negative_acc is None
                else cfg.positive_jerk_limit_when_negative_acc
            ),
            positive_jerk_limit_when_nonnegative_acc=cfg.positive_jerk_limit_when_nonnegative_acc,
        )
        next_v_long = (v_long + 0.5 * (a_long + next_a_long) * dt).clamp(cfg.min_speed, cfg.max_speed)

        target_steering = self._compute_lateral_target_steering(
            lat_command,
            speed,
            steering,
            wheelbase,
            v_long,
            a_long,
            next_a_long,
            dt,
        )

        delta_bound = self._delta_steering_rate_bound(speed) * dt
        delta_steering = (target_steering - steering).clamp(-delta_bound, delta_bound)
        next_steering = (steering + delta_steering).clamp(cfg.min_steering_angle, cfg.max_steering_angle)

        next_curvature = torch.tan(next_steering) / wheelbase
        mean_v_long = 0.5 * (v_long + next_v_long)
        kinematic_yaw_rate = mean_v_long * next_curvature
        kinematic_v_lat = kinematic_yaw_rate * wheelbase * 0.5
        dynamic_v_lat, dynamic_yaw_rate = self._dynamic_lateral_update(dt, v_long, v_lat, yaw_rate, next_steering)
        use_dynamic = v_long > cfg.min_dynamic_speed
        next_v_lat = torch.where(use_dynamic, dynamic_v_lat, kinematic_v_lat)
        next_yaw_rate = torch.where(use_dynamic, dynamic_yaw_rate, kinematic_yaw_rate)
        next_headings = wrap_angle(headings + 0.5 * (yaw_rate + next_yaw_rate) * dt)

        next_vx = next_v_long * next_headings.cos() - next_v_lat * next_headings.sin()
        next_vy = next_v_long * next_headings.sin() + next_v_lat * next_headings.cos()
        next_velocities = torch.stack((next_vx, next_vy), dim=-1)
        next_positions = positions + 0.5 * (velocities + next_velocities) * dt
        next_a_lat = next_v_long.square() * next_curvature

        return {
            "positions": next_positions,
            "velocities": next_velocities,
            "headings": next_headings,
            "a_long": next_a_long,
            "a_lat": next_a_lat,
            "steering": next_steering,
            "yaw_rate": next_yaw_rate,
            "jerk_long": jerk_long,
            "jerk_lat": lat_command,
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
    wheelbase = get_wheelbase_from_length(log_trajs[..., 3])
    curvature = torch.where(v_long.abs() > 1e-3, yaw_rate / _stabilize_tensor(v_long, 1e-3), torch.zeros_like(yaw_rate))
    steering = torch.atan(curvature * wheelbase).clamp(MIN_STEERING_ANGLE, MAX_STEERING_ANGLE)
    a_lat = v_long.square() * torch.tan(steering) / wheelbase

    return {
        "positions": positions,
        "velocities": velocities,
        "headings": headings,
        "a_long": a_long,
        "a_lat": a_lat,
        "steering": steering,
        "yaw_rate": yaw_rate,
    }
