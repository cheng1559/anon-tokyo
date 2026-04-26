"""Closed-loop simulation environment with controlled/log replay agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from anon_tokyo.simulation.dynamics import JerkPncConfig, JerkPncModel, infer_log_kinematics
from anon_tokyo.simulation.rewards import RewardConfig, compute_rewards


@dataclass
class ClosedLoopEnvConfig:
    num_steps: int = 80
    history_steps: int = 11
    device: str = "cuda"
    goal_sampling_mode: str = "last"
    goal_sampling_gamma: float = 2.0
    dynamics: JerkPncConfig = field(default_factory=JerkPncConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ClosedLoopEnvConfig":
        data = dict(data or {})
        dynamics = data.pop("dynamics", None)
        rewards = data.pop("rewards", None)
        cfg = cls(**data)
        if dynamics is not None:
            cfg.dynamics = JerkPncConfig(**dynamics)
        if rewards is not None:
            cfg.rewards = RewardConfig(**rewards)
        cfg.rewards.num_steps = cfg.num_steps
        return cfg


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def _index_tensor(value: Any, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


class ClosedLoopEnv:
    """Batched closed-loop simulator.

    Agents with ``controlled_mask=True`` are propagated by dynamics.  All other
    agents replay WOMD log states.
    """

    def __init__(self, config: ClosedLoopEnvConfig | dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, ClosedLoopEnvConfig) else ClosedLoopEnvConfig.from_dict(config)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.config.device = "cpu"
        self.device = torch.device(self.config.device)
        self.dynamics = JerkPncModel(self.config.dynamics)
        self.batch: dict[str, Any] | None = None
        self.log_trajs: Tensor | None = None
        self.log_mask: Tensor | None = None
        self.log_kinematics: dict[str, Tensor] | None = None
        self.start_index = 0
        self.step_count = 0
        self.episode_steps = 0
        self.controlled_mask: Tensor | None = None
        self.done: Tensor | None = None
        self.goal_reached: Tensor | None = None
        self.goal_positions: Tensor | None = None

        self.positions: Tensor | None = None
        self.velocities: Tensor | None = None
        self.headings: Tensor | None = None
        self.sizes: Tensor | None = None
        self.valid: Tensor | None = None
        self.a_long: Tensor | None = None
        self.a_lat: Tensor | None = None
        self.steering: Tensor | None = None
        self.yaw_rate: Tensor | None = None
        self.jerk_long: Tensor | None = None
        self.jerk_lat: Tensor | None = None

    @property
    def action_low(self) -> Tensor:
        return self.dynamics.action_low.to(self.device)

    @property
    def action_high(self) -> Tensor:
        return self.dynamics.action_high.to(self.device)

    @property
    def num_envs(self) -> int:
        assert self.log_trajs is not None
        return self.log_trajs.shape[0]

    @property
    def num_agents(self) -> int:
        assert self.log_trajs is not None
        return self.log_trajs.shape[1]

    def reset(self, batch: dict[str, Any]) -> dict[str, Tensor | list[str]]:
        self.batch = _to_device(batch, self.device)
        assert isinstance(self.batch, dict)
        self.log_trajs = self.batch["obj_trajs_full"].float()
        self.log_mask = self.batch["obj_trajs_full_mask"].bool()
        timestamps = self.batch.get("timestamps")
        self._set_dt(timestamps)
        self.log_kinematics = infer_log_kinematics(self.log_trajs, timestamps, default_dt=self.config.dynamics.dt)

        current_t = self.batch["current_time_index"]
        if isinstance(current_t, Tensor):
            self.start_index = int(current_t.flatten()[0].item())
        elif isinstance(current_t, list):
            self.start_index = int(current_t[0])
        else:
            self.start_index = int(current_t)
        total_steps = self.log_trajs.shape[2]
        self.episode_steps = min(self.config.num_steps, max(total_steps - self.start_index - 1, 0))
        self.step_count = 0

        end = self.start_index + 1
        self.positions = self.log_kinematics["positions"][:, :, :end].clone()
        self.velocities = self.log_kinematics["velocities"][:, :, :end].clone()
        self.headings = self.log_kinematics["headings"][:, :, :end].clone()
        self.sizes = self.log_trajs[:, :, :end, 3:5].clone()
        self.valid = self.log_mask[:, :, :end].clone()
        self.a_long = self.log_kinematics["a_long"][:, :, :end].clone()
        self.a_lat = self.log_kinematics["a_lat"][:, :, :end].clone()
        self.steering = self.log_kinematics["steering"][:, :, :end].clone()
        self.yaw_rate = self.log_kinematics["yaw_rate"][:, :, :end].clone()
        self.jerk_long = torch.zeros_like(self.a_long)
        self.jerk_lat = torch.zeros_like(self.a_lat)

        controlled = self.batch["controlled_mask"].bool()
        controlled = controlled & self.valid[:, :, -1]
        self.controlled_mask = controlled
        self.done = torch.zeros_like(controlled, dtype=torch.bool)
        self.goal_reached = torch.zeros_like(controlled, dtype=torch.bool)
        self.goal_positions = self._sample_goals()
        return self.get_obs()

    def _set_dt(self, timestamps: Tensor | None) -> None:
        if timestamps is not None and timestamps.numel() >= 2:
            ts = timestamps[0] if timestamps.ndim == 2 else timestamps
            dt = float((ts[1:] - ts[:-1]).median().item())
            if dt > 0:
                self.config.dynamics.dt = dt
                self.dynamics.config.dt = dt

    def _sample_goals(self) -> Tensor:
        assert self.log_kinematics is not None and self.log_mask is not None
        future_pos = self.log_kinematics["positions"][:, :, self.start_index + 1 :]
        future_mask = self.log_mask[:, :, self.start_index + 1 :]
        current_pos = self.log_kinematics["positions"][:, :, self.start_index]
        if future_pos.shape[2] == 0:
            return current_pos

        if self.config.goal_sampling_mode == "last":
            idx_base = torch.arange(future_pos.shape[2], device=self.device).view(1, 1, -1)
            last_idx = torch.where(future_mask, idx_base, torch.zeros_like(idx_base)).amax(dim=-1)
        elif self.config.goal_sampling_mode == "linear":
            t = torch.arange(future_pos.shape[2], device=self.device, dtype=future_pos.dtype)
            weights = ((t + 1.0) ** self.config.goal_sampling_gamma).view(1, 1, -1) * future_mask.float()
            flat_weights = weights.reshape(-1, future_pos.shape[2])
            row_sum = flat_weights.sum(dim=-1, keepdim=True)
            fallback = torch.zeros_like(flat_weights)
            fallback[:, -1] = 1.0
            probs = torch.where(row_sum > 0, flat_weights / row_sum.clamp_min(1e-6), fallback)
            last_idx = torch.multinomial(probs, 1).view(future_pos.shape[:2])
        else:
            raise ValueError(f"Unsupported goal_sampling_mode: {self.config.goal_sampling_mode}")

        b = torch.arange(future_pos.shape[0], device=self.device)[:, None]
        a = torch.arange(future_pos.shape[1], device=self.device)[None, :]
        sampled = future_pos[b, a, last_idx]
        has_future = future_mask.any(dim=-1)
        return torch.where(has_future.unsqueeze(-1), sampled, current_pos)

    def step(self, actions: Tensor) -> tuple[dict[str, Tensor | list[str]], Tensor, Tensor, dict[str, Tensor]]:
        self._assert_ready()
        assert self.positions is not None and self.velocities is not None and self.headings is not None
        assert self.sizes is not None and self.valid is not None and self.a_long is not None
        assert self.a_lat is not None and self.steering is not None and self.yaw_rate is not None and self.done is not None
        assert self.controlled_mask is not None and self.log_kinematics is not None and self.log_mask is not None

        actions = actions.to(self.device, dtype=self.positions.dtype)
        next_index = self.start_index + self.step_count + 1
        latest_valid = self.valid[:, :, -1]
        active = self.controlled_mask & latest_valid

        dyn = self.dynamics.step(
            self.positions[:, :, -1],
            self.velocities[:, :, -1],
            self.headings[:, :, -1],
            self.sizes[:, :, -1],
            self.a_long[:, :, -1],
            self.a_lat[:, :, -1],
            self.steering[:, :, -1],
            self.yaw_rate[:, :, -1],
            actions,
        )

        log_pos = self.log_kinematics["positions"][:, :, next_index]
        log_vel = self.log_kinematics["velocities"][:, :, next_index]
        log_heading = self.log_kinematics["headings"][:, :, next_index]
        log_size = self.log_trajs[:, :, next_index, 3:5]
        log_valid = self.log_mask[:, :, next_index]
        log_a_long = self.log_kinematics["a_long"][:, :, next_index]
        log_a_lat = self.log_kinematics["a_lat"][:, :, next_index]
        log_steering = self.log_kinematics["steering"][:, :, next_index]
        log_yaw_rate = self.log_kinematics["yaw_rate"][:, :, next_index]

        active_e = active.unsqueeze(-1)
        next_pos = torch.where(active_e, dyn["positions"], log_pos)
        next_vel = torch.where(active_e, dyn["velocities"], log_vel)
        next_heading = torch.where(active, dyn["headings"], log_heading)
        next_size = torch.where(active_e, self.sizes[:, :, -1], log_size)
        next_valid = torch.where(self.controlled_mask, active, log_valid)
        next_a_long = torch.where(active, dyn["a_long"], log_a_long)
        next_a_lat = torch.where(active, dyn["a_lat"], log_a_lat)
        next_steering = torch.where(active, dyn["steering"], log_steering)
        next_yaw_rate = torch.where(active, dyn["yaw_rate"], log_yaw_rate)
        next_jerk_long = torch.where(active, dyn["jerk_long"], torch.zeros_like(log_a_long))
        next_jerk_lat = torch.where(active, dyn["jerk_lat"], torch.zeros_like(log_a_lat))

        self.positions = torch.cat((self.positions, next_pos.unsqueeze(2)), dim=2)
        self.velocities = torch.cat((self.velocities, next_vel.unsqueeze(2)), dim=2)
        self.headings = torch.cat((self.headings, next_heading.unsqueeze(2)), dim=2)
        self.sizes = torch.cat((self.sizes, next_size.unsqueeze(2)), dim=2)
        self.valid = torch.cat((self.valid, next_valid.unsqueeze(2)), dim=2)
        self.a_long = torch.cat((self.a_long, next_a_long.unsqueeze(2)), dim=2)
        self.a_lat = torch.cat((self.a_lat, next_a_lat.unsqueeze(2)), dim=2)
        self.steering = torch.cat((self.steering, next_steering.unsqueeze(2)), dim=2)
        self.yaw_rate = torch.cat((self.yaw_rate, next_yaw_rate.unsqueeze(2)), dim=2)
        self.jerk_long = torch.cat((self.jerk_long, next_jerk_long.unsqueeze(2)), dim=2)
        self.jerk_lat = torch.cat((self.jerk_lat, next_jerk_lat.unsqueeze(2)), dim=2)

        reward_state = self._reward_state()
        reward, done_step, info, next_goal_reached = compute_rewards(reward_state, self.config.rewards)
        self.goal_reached = next_goal_reached
        self.done = self.done | done_step
        self.step_count += 1
        if self.step_count >= self.episode_steps:
            self.done = self.done | (self.controlled_mask & self.valid[:, :, -1])

        return self.get_obs(), reward, self.done.clone(), info

    def get_obs(self) -> dict[str, Tensor | list[str]]:
        self._assert_ready()
        assert self.batch is not None and self.positions is not None and self.velocities is not None
        assert self.headings is not None and self.sizes is not None and self.valid is not None
        assert self.steering is not None and self.yaw_rate is not None
        H = self.config.history_steps
        B, A, T = self.valid.shape
        start = max(T - H, 0)
        used = T - start
        obj_trajs = self.positions.new_zeros((B, A, H, 10))
        obj_mask = torch.zeros((B, A, H), dtype=torch.bool, device=self.device)
        dst = slice(H - used, H)
        obj_trajs[:, :, dst, 0:2] = self.positions[:, :, start:T]
        obj_trajs[:, :, dst, 3:5] = self.sizes[:, :, start:T]
        obj_trajs[:, :, dst, 6] = self.headings[:, :, start:T].sin()
        obj_trajs[:, :, dst, 7] = self.headings[:, :, start:T].cos()
        obj_trajs[:, :, dst, 8:10] = self.velocities[:, :, start:T]
        obj_mask[:, :, dst] = self.valid[:, :, start:T]
        obj_trajs = obj_trajs.masked_fill(~obj_mask.unsqueeze(-1), 0.0)

        obs: dict[str, Tensor | list[str]] = {
            "obj_trajs": obj_trajs,
            "obj_trajs_mask": obj_mask.float(),
            "obj_positions": self.positions[:, :, -1],
            "obj_headings": self.headings[:, :, -1],
            "obj_types": self.batch["obj_types"],
            "agent_mask": self.valid[:, :, -1].float(),
            "tracks_to_predict": self.batch["tracks_to_predict"],
            "sdc_track_index": _index_tensor(self.batch["sdc_track_index"], self.device),
            "map_polylines": self.batch["map_polylines"],
            "map_polylines_mask": self.batch["map_polylines_mask"],
            "map_polylines_center": self.batch["map_polylines_center"],
            "map_headings": self.batch["map_headings"],
            "map_mask": self.batch["map_mask"],
            "obj_trajs_future": self.batch["obj_trajs_future"],
            "obj_trajs_future_mask": self.batch["obj_trajs_future_mask"],
            "controlled_mask": self.controlled_mask.float(),
            "goal_positions": self.goal_positions,
            "timestep": torch.full((B,), self.step_count, dtype=torch.long, device=self.device),
            "steering": self.steering[:, :, -1],
            "yaw_rate": self.yaw_rate[:, :, -1],
        }
        if "scenario_id" in self.batch:
            obs["scenario_id"] = self.batch["scenario_id"]
        if "map_token_features" in self.batch:
            obs["map_token_features"] = self.batch["map_token_features"]
        return obs

    def _reward_state(self) -> dict[str, Tensor]:
        assert self.batch is not None and self.goal_positions is not None and self.goal_reached is not None
        assert self.positions is not None and self.velocities is not None and self.headings is not None
        assert self.sizes is not None and self.valid is not None and self.a_long is not None
        assert self.a_lat is not None and self.jerk_long is not None and self.jerk_lat is not None
        assert self.controlled_mask is not None
        return {
            "positions": self.positions[:, :, -1],
            "prev_positions": self.positions[:, :, -2] if self.positions.shape[2] >= 2 else self.positions[:, :, -1],
            "velocities": self.velocities[:, :, -1],
            "headings": self.headings[:, :, -1],
            "sizes": self.sizes[:, :, -1],
            "valid_mask": self.valid[:, :, -1],
            "controlled_mask": self.controlled_mask,
            "map_polylines": self.batch["map_polylines"],
            "map_polylines_mask": self.batch["map_polylines_mask"],
            "goal_positions": self.goal_positions,
            "goal_reached": self.goal_reached,
            "a_long": self.a_long[:, :, -1],
            "a_lat": self.a_lat[:, :, -1],
            "jerk_long": self.jerk_long[:, :, -1],
            "jerk_lat": self.jerk_lat[:, :, -1],
        }

    def _assert_ready(self) -> None:
        if self.batch is None:
            raise RuntimeError("ClosedLoopEnv.reset(batch) must be called before using the environment.")
