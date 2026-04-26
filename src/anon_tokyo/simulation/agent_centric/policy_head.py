"""Agent-centric simulation policy and value heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta


class AgentCentricPolicyHead(nn.Module):
    """Bounded Beta policy head for jerk actions plus a scalar value head."""

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int | None = None,
        action_low: tuple[float, float] | list[float] = (-5.0, -1.0),
        action_high: tuple[float, float] | list[float] = (3.0, 1.0),
        min_concentration: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.actor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.critic = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.min_concentration = min_concentration
        self.eps = eps
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32), persistent=False)
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32), persistent=False)

    def _distribution(self, features: Tensor) -> Beta:
        params = self.actor(features)
        alpha_raw, beta_raw = params.chunk(2, dim=-1)
        alpha = F.softplus(alpha_raw) + self.min_concentration
        beta = F.softplus(beta_raw) + self.min_concentration
        return Beta(alpha, beta)

    def _unit_action(self, dist: Beta, sampling_method: str | None) -> Tensor:
        method = sampling_method or "sample"
        if method in {"mean", "deterministic"}:
            return dist.mean
        if method == "mode":
            alpha = dist.concentration1
            beta = dist.concentration0
            mode = (alpha - 1.0) / (alpha + beta - 2.0).clamp_min(self.eps)
            return torch.where((alpha > 1.0) & (beta > 1.0), mode, dist.mean).clamp(self.eps, 1.0 - self.eps)
        if method == "rsample":
            return dist.rsample()
        if method != "sample":
            raise ValueError(f"Unsupported sampling_method: {sampling_method}")
        return dist.sample()

    def forward(
        self,
        features: Tensor,
        mask: Tensor,
        action: Tensor | None = None,
        sampling_method: str | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dist = self._distribution(features)
        low = self.action_low.to(device=features.device, dtype=features.dtype)
        high = self.action_high.to(device=features.device, dtype=features.dtype)
        scale = (high - low).clamp_min(self.eps)

        if action is None:
            unit_action = self._unit_action(dist, sampling_method).clamp(self.eps, 1.0 - self.eps)
            action = low + unit_action * scale
        else:
            action = action.to(device=features.device, dtype=features.dtype)
            unit_action = ((action - low) / scale).clamp(self.eps, 1.0 - self.eps)

        log_scale = scale.log().sum()
        logprob = dist.log_prob(unit_action).sum(dim=-1) - log_scale
        entropy = dist.entropy().sum(dim=-1) + log_scale
        value = self.critic(features).squeeze(-1)

        mask = mask.bool()
        action = action.masked_fill(~mask[..., None], 0.0)
        logprob = logprob.masked_fill(~mask, 0.0)
        entropy = entropy.masked_fill(~mask, 0.0)
        value = value.masked_fill(~mask, 0.0)
        return action, logprob, entropy, value
