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
from anon_tokyo.simulation.metrics import compute_rollout_metric_tensors, scalar_rollout_metrics
from anon_tokyo.simulation.profiling import TimingProfiler


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
    profile: bool = False
    profile_cuda_sync: bool = True

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
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_agents: int,
        action_dim: int,
        device: torch.device,
        *,
        drop_raw_map_when_tokenized: bool = True,
    ) -> None:
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
        self.drop_raw_map_when_tokenized = drop_raw_map_when_tokenized

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
        has_map_tokens = self.drop_raw_map_when_tokenized and "map_token_features" in obs
        stored_obs = {}
        for key, obs_value in obs.items():
            if not isinstance(obs_value, Tensor):
                continue
            if has_map_tokens and key in _DROP_FROM_POLICY_BUFFER_WHEN_TOKENIZED:
                continue
            stored_obs[key] = obs_value.detach()
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
        self.profiler = TimingProfiler(
            enabled=self.config.profile,
            device=self.env.device,
            sync_cuda=self.config.profile_cuda_sync,
        )
        self.env.profiler = self.profiler if self.config.profile else None

    def _drop_raw_map_when_tokenized(self) -> bool:
        policy = self.policy
        while hasattr(policy, "module"):
            policy = policy.module
        architecture = getattr(policy, "architecture", "legacy")
        return architecture != "agentcentric"

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

    def _set_policy_profiler(self) -> None:
        profiler = self.profiler if self.config.profile else None
        for module in self.policy.modules():
            setattr(module, "_profiler", profiler)

    @torch.no_grad()
    def collect_rollout(self, batch: dict[str, Any], sampling_method: str = "sample") -> tuple[RolloutBuffer, dict[str, Any], Tensor, dict[str, float]]:
        self._set_policy_profiler()
        self.env.profiler = self.profiler if self.config.profile else None
        with self.profiler.record("rollout.env_reset"):
            obs = self.env.reset(batch)
        steps = min(self.config.num_steps, self.env.episode_steps)
        if steps <= 0:
            raise RuntimeError("Batch has no future frames for closed-loop rollout")
        buffer = RolloutBuffer(
            steps,
            self.env.num_envs,
            self.env.num_agents,
            action_dim=2,
            device=self.env.device,
            drop_raw_map_when_tokenized=self._drop_raw_map_when_tokenized(),
        )
        prev_done = torch.zeros(self.env.num_envs, self.env.num_agents, dtype=torch.bool, device=self.env.device)
        collision_steps = []
        offroad_steps = []
        goal_reached_steps = []
        for _ in range(steps):
            with self.profiler.record("rollout.policy_forward"):
                action, logprob, _, value = self._policy_forward(obs, sampling_method=sampling_method)
            with self.profiler.record("rollout.env_step"):
                next_obs, reward, done, info = self.env.step(action)
            assert self.env.goal_reached is not None
            with self.profiler.record("rollout.metrics_collect"):
                collision_steps.append(info["collision"].detach())
                offroad_steps.append(info["offroad"].detach())
                goal_reached_steps.append(self.env.goal_reached.detach())
            train_mask = obs["controlled_mask"].bool() & obs["agent_mask"].bool() & ~prev_done
            with self.profiler.record("rollout.buffer_store"):
                buffer.store(obs, action.float(), logprob.float(), value.float(), reward.float(), done, train_mask)
            obs = next_obs
            prev_done = done
        with self.profiler.record("rollout.metric_reduce"):
            rollout_metrics = scalar_rollout_metrics(
                compute_rollout_metric_tensors(
                    collision=torch.stack(collision_steps, dim=0),
                    offroad=torch.stack(offroad_steps, dim=0),
                    goal_reached=torch.stack(goal_reached_steps, dim=0),
                    controlled_mask=buffer.obs[0]["controlled_mask"].bool(),
                    agent_mask=buffer.obs[0]["agent_mask"].bool(),
                )
            )
        return buffer, obs, prev_done, rollout_metrics

    @torch.no_grad()
    def estimate_returns_and_advantages(self, buffer: RolloutBuffer, next_obs: dict[str, Any], next_done: Tensor) -> tuple[Tensor, Tensor]:
        self._set_policy_profiler()
        with self.profiler.record("gae.bootstrap_policy_forward"):
            _, _, _, next_value = self._policy_forward(next_obs, sampling_method="sample")
        next_value = next_value.float()
        with self.profiler.record("gae.compute"):
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
        self._set_policy_profiler()
        total_samples = buffer.size * buffer.num_envs
        minibatch_size = self.config.minibatch_size or total_samples
        flat_indices = torch.arange(total_samples, device=self.env.device)
        with self.profiler.record("update.stack_obs"):
            stacked_obs, static_obs_keys = stack_obs_steps(buffer.obs)
        metrics: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}

        for _ in range(self.config.optimization_epochs):
            perm = flat_indices[torch.randperm(total_samples, device=self.env.device)]
            approx_kl_epoch = torch.tensor(0.0, device=self.env.device)
            for start in range(0, total_samples, minibatch_size):
                mb = perm[start : start + minibatch_size]
                step_idx = mb // buffer.num_envs
                env_idx = mb % buffer.num_envs
                with self.profiler.record("update.gather_obs"):
                    obs_mb = gather_obs(stacked_obs, static_obs_keys, env_idx, step_idx)
                actions_mb = buffer.actions[step_idx, env_idx]
                with self.profiler.record("update.policy_forward"):
                    _, new_logprob, entropy, new_value = self._policy_forward(obs_mb, action=actions_mb)

                with self.profiler.record("update.loss"):
                    old_logprob = buffer.logprobs[step_idx, env_idx]
                    logratio = new_logprob.float() - old_logprob
                    ratio = logratio.exp()
                    mask = buffer.train_masks[step_idx, env_idx]
                    has_valid = mask.any()
                    if has_valid:
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
                            clipped_value = old_value + (new_value - old_value).clamp(
                                -self.config.vf_clip_epsilon,
                                self.config.vf_clip_epsilon,
                            )
                            clipped = (clipped_value - target).square()
                            value_loss = 0.5 * masked_mean(torch.maximum(unclipped, clipped), mask)
                        else:
                            value_loss = 0.5 * masked_mean((new_value - target).square(), mask)

                        entropy_loss = masked_mean(entropy.float(), mask)
                        loss = policy_loss - self.config.entropy_coeff * entropy_loss + self.config.value_coeff * value_loss
                    else:
                        zero_terms = [param.sum() * 0.0 for param in self.policy.parameters() if param.requires_grad]
                        loss = sum(zero_terms) if zero_terms else new_logprob.float().sum() * 0.0

                self.optimizer.zero_grad(set_to_none=True)
                with self.profiler.record("update.backward"):
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                with self.profiler.record("update.optimizer_step"):
                    self.optimizer.step()

                if has_valid:
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
        self.profiler.reset()
        self._sync_timing()
        t0 = time.perf_counter()
        buffer, next_obs, next_done, rollout_metrics = self.collect_rollout(batch, sampling_method=sampling_method)
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
        metrics.update(rollout_metrics)
        metrics["rollout_seconds"] = t1 - t0
        metrics["gae_seconds"] = t2 - t1
        metrics["update_seconds"] = t3 - t2
        if self.config.profile:
            metrics.update(self.profiler.metrics())
        return metrics
