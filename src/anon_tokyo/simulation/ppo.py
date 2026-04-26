"""Model-agnostic PPO for closed-loop simulation."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import time
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from anon_tokyo.simulation.env import ClosedLoopEnv


_DROP_FROM_POLICY_BUFFER_WHEN_TOKENIZED = {
    "map_polylines",
    "map_polylines_mask",
    "map_headings",
    "obj_trajs_future",
    "obj_trajs_future_mask",
    "tracks_to_predict",
    "scenario_id",
}


@dataclass
class PPOConfig:
    num_steps: int = 80
    num_updates: int | None = 1500
    total_epochs: int | None = None
    learning_rate: float = 3.0e-4
    minibatch_size: int | None = None
    optimization_epochs: int = 4
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    vf_clip_epsilon: float = 0.2
    target_kl: float | None = None
    use_bf16: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PPOConfig":
        return cls(**dict(data or {}))


def masked_mean(x: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    mask_f = mask.to(dtype=x.dtype)
    return (x * mask_f).sum() / mask_f.sum().clamp_min(eps)


def masked_std(x: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    mean = masked_mean(x, mask, eps)
    return torch.sqrt(masked_mean((x - mean).square(), mask, eps).clamp_min(0.0))


def _same_storage(values: list[Tensor]) -> bool:
    first = values[0]
    first_ptr = first.untyped_storage().data_ptr()
    return all(value.shape == first.shape and value.untyped_storage().data_ptr() == first_ptr for value in values[1:])


def stack_obs_steps(obs_steps: list[dict[str, Any]]) -> tuple[dict[str, Tensor], set[str]]:
    """Stack dynamic rollout observations and keep static tensors unstacked."""
    out: dict[str, Tensor] = {}
    static_keys: set[str] = set()
    for key in obs_steps[0]:
        values = [obs[key] for obs in obs_steps]
        if not isinstance(values[0], Tensor):
            raise TypeError(f"Rollout observation {key!r} is not a tensor")
        if _same_storage(values):
            out[key] = values[0]
            static_keys.add(key)
        else:
            out[key] = torch.stack(values, dim=0)
    return out, static_keys


def gather_obs(
    stacked_obs: dict[str, Tensor],
    static_keys: set[str],
    env_indices: Tensor,
    step_indices: Tensor,
) -> dict[str, Tensor]:
    """Gather a PPO minibatch from stacked ``[T, B, ...]`` observations."""
    return {
        key: value[env_indices] if key in static_keys else value[step_indices, env_indices]
        for key, value in stacked_obs.items()
    }


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, num_agents: int, action_dim: int, device: torch.device) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.device = device
        self.obs: list[dict[str, Any]] = []
        self.actions = torch.zeros(num_steps, num_envs, num_agents, action_dim, device=device)
        self.logprobs = torch.zeros(num_steps, num_envs, num_agents, device=device)
        self.values = torch.zeros(num_steps, num_envs, num_agents, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, num_agents, device=device)
        self.dones = torch.zeros(num_steps, num_envs, num_agents, dtype=torch.bool, device=device)
        self.train_masks = torch.zeros(num_steps, num_envs, num_agents, dtype=torch.bool, device=device)
        self.size = 0

    def store(
        self,
        obs: dict[str, Any],
        action: Tensor,
        logprob: Tensor,
        value: Tensor,
        reward: Tensor,
        done: Tensor,
        train_mask: Tensor,
    ) -> None:
        if self.size >= self.num_steps:
            raise RuntimeError("RolloutBuffer is full")
        has_map_tokens = "map_token_features" in obs
        stored_obs = {}
        for key, obs_value in obs.items():
            if has_map_tokens and key in _DROP_FROM_POLICY_BUFFER_WHEN_TOKENIZED:
                continue
            stored_obs[key] = obs_value.detach() if isinstance(obs_value, Tensor) else obs_value
        self.obs.append(stored_obs)
        self.actions[self.size] = action.detach()
        self.logprobs[self.size] = logprob.detach()
        self.values[self.size] = value.detach()
        self.rewards[self.size] = reward.detach()
        self.dones[self.size] = done.detach()
        self.train_masks[self.size] = train_mask.detach()
        self.size += 1


class PPOTrainer:
    """PPO trainer for policies with ``policy(obs, action=None)`` API."""

    def __init__(
        self,
        env: ClosedLoopEnv,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        config: PPOConfig | dict[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.policy = policy.to(env.device)
        self.config = config if isinstance(config, PPOConfig) else PPOConfig.from_dict(config)
        self.optimizer = optimizer or torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5)

    def _autocast_enabled(self) -> bool:
        return self.config.use_bf16 and self.env.device.type == "cuda"

    def _sync_timing(self) -> None:
        if self.env.device.type == "cuda":
            torch.cuda.synchronize(self.env.device)

    def _policy_forward(
        self,
        obs: dict[str, Any],
        action: Tensor | None = None,
        sampling_method: str | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sig = inspect.signature(self.policy.forward)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        kwargs: dict[str, Any] = {}
        if action is not None and ("action" in sig.parameters or accepts_kwargs):
            kwargs["action"] = action
        if sampling_method is not None and ("sampling_method" in sig.parameters or accepts_kwargs):
            kwargs["sampling_method"] = sampling_method
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._autocast_enabled()):
            action, logprob, entropy, value = self.policy(obs, **kwargs)
        return action.float(), logprob.float(), entropy.float(), value.float()

    @torch.no_grad()
    def collect_rollout(self, batch: dict[str, Any], sampling_method: str = "sample") -> tuple[RolloutBuffer, dict[str, Any], Tensor]:
        obs = self.env.reset(batch)
        steps = min(self.config.num_steps, self.env.episode_steps)
        if steps <= 0:
            raise RuntimeError("Batch has no future frames for closed-loop rollout")
        buffer = RolloutBuffer(steps, self.env.num_envs, self.env.num_agents, action_dim=2, device=self.env.device)
        prev_done = torch.zeros(self.env.num_envs, self.env.num_agents, dtype=torch.bool, device=self.env.device)
        for _ in range(steps):
            action, logprob, _, value = self._policy_forward(obs, sampling_method=sampling_method)
            next_obs, reward, done, _ = self.env.step(action)
            train_mask = obs["controlled_mask"].bool() & obs["agent_mask"].bool() & ~prev_done
            buffer.store(obs, action.float(), logprob.float(), value.float(), reward.float(), done, train_mask)
            obs = next_obs
            prev_done = done
        return buffer, obs, prev_done

    @torch.no_grad()
    def estimate_returns_and_advantages(self, buffer: RolloutBuffer, next_obs: dict[str, Any], next_done: Tensor) -> tuple[Tensor, Tensor]:
        _, _, _, next_value = self._policy_forward(next_obs, sampling_method="sample")
        next_value = next_value.float()
        advantages = torch.zeros_like(buffer.rewards)
        last_gae = torch.zeros_like(next_value)
        for t in reversed(range(buffer.size)):
            next_nonterminal = 1.0 - (next_done.float() if t == buffer.size - 1 else buffer.dones[t].float())
            next_values = next_value if t == buffer.size - 1 else buffer.values[t + 1]
            delta = buffer.rewards[t] + self.config.gamma * next_values * next_nonterminal - buffer.values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + buffer.values
        return returns, advantages

    def update(self, buffer: RolloutBuffer, returns: Tensor, advantages: Tensor) -> dict[str, float]:
        total_samples = buffer.size * buffer.num_envs
        minibatch_size = self.config.minibatch_size or total_samples
        flat_indices = torch.arange(total_samples, device=self.env.device)
        stacked_obs, static_obs_keys = stack_obs_steps(buffer.obs)
        metrics: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}

        for _ in range(self.config.optimization_epochs):
            perm = flat_indices[torch.randperm(total_samples, device=self.env.device)]
            approx_kl_epoch = torch.tensor(0.0, device=self.env.device)
            for start in range(0, total_samples, minibatch_size):
                mb = perm[start : start + minibatch_size]
                step_idx = mb // buffer.num_envs
                env_idx = mb % buffer.num_envs
                obs_mb = gather_obs(stacked_obs, static_obs_keys, env_idx, step_idx)
                actions_mb = buffer.actions[step_idx, env_idx]
                _, new_logprob, entropy, new_value = self._policy_forward(obs_mb, action=actions_mb)

                old_logprob = buffer.logprobs[step_idx, env_idx]
                logratio = new_logprob.float() - old_logprob
                ratio = logratio.exp()
                mask = buffer.train_masks[step_idx, env_idx]
                if not mask.any():
                    continue

                mb_adv = advantages[step_idx, env_idx]
                if self.config.norm_adv:
                    mb_adv = (mb_adv - masked_mean(mb_adv, mask)) / (masked_std(mb_adv, mask) + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
                policy_loss = masked_mean(torch.maximum(pg_loss1, pg_loss2), mask)

                old_value = buffer.values[step_idx, env_idx]
                target = returns[step_idx, env_idx]
                new_value = new_value.float()
                if self.config.clip_vloss:
                    unclipped = (new_value - target).square()
                    clipped_value = old_value + (new_value - old_value).clamp(-self.config.vf_clip_epsilon, self.config.vf_clip_epsilon)
                    clipped = (clipped_value - target).square()
                    value_loss = 0.5 * masked_mean(torch.maximum(unclipped, clipped), mask)
                else:
                    value_loss = 0.5 * masked_mean((new_value - target).square(), mask)

                entropy_loss = masked_mean(entropy.float(), mask)
                loss = policy_loss - self.config.entropy_coeff * entropy_loss + self.config.value_coeff * value_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    sample_kl = (ratio - 1.0) - logratio
                    approx_kl_epoch = masked_mean(sample_kl, mask)
                    clipfrac = masked_mean(((ratio - 1.0).abs() > self.config.clip_epsilon).float(), mask)

                metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
                metrics["value_loss"].append(float(value_loss.detach().cpu()))
                metrics["entropy"].append(float(entropy_loss.detach().cpu()))
                metrics["approx_kl"].append(float(approx_kl_epoch.detach().cpu()))
                metrics["clipfrac"].append(float(clipfrac.detach().cpu()))

            if self.config.target_kl is not None and float(approx_kl_epoch.detach().cpu()) > self.config.target_kl:
                break

        return {k: float(sum(v) / max(len(v), 1)) for k, v in metrics.items()}

    def train_one_update(self, batch: dict[str, Any], sampling_method: str = "sample") -> dict[str, float]:
        self._sync_timing()
        t0 = time.perf_counter()
        buffer, next_obs, next_done = self.collect_rollout(batch, sampling_method=sampling_method)
        self._sync_timing()
        t1 = time.perf_counter()
        returns, advantages = self.estimate_returns_and_advantages(buffer, next_obs, next_done)
        self._sync_timing()
        t2 = time.perf_counter()
        metrics = self.update(buffer, returns, advantages)
        self._sync_timing()
        t3 = time.perf_counter()
        valid_mask = buffer.train_masks
        metrics["mean_reward"] = float(masked_mean(buffer.rewards, valid_mask).detach().cpu()) if valid_mask.any() else 0.0
        metrics["rollout_seconds"] = t1 - t0
        metrics["gae_seconds"] = t2 - t1
        metrics["update_seconds"] = t3 - t2
        return metrics
