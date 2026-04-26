"""Agent-centric simulation policy model."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from anon_tokyo.simulation.agent_centric.encoder import AgentCentricEncoder
from anon_tokyo.simulation.agent_centric.policy_head import AgentCentricPolicyHead


class AgentCentricModel(nn.Module):
    """Full PPO policy using explicit ego-centric context per agent."""

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        sparse_k: int = 16,
        dropout: float = 0.1,
        max_context_agents: int = 32,
        max_context_maps: int = 128,
        use_map_self_attention: bool = True,
        action_low: tuple[float, float] | list[float] = (-5.0, -1.0),
        action_high: tuple[float, float] | list[float] = (3.0, 1.0),
        policy_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = AgentCentricEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            sparse_k=sparse_k,
            dropout=dropout,
            max_context_agents=max_context_agents,
            max_context_maps=max_context_maps,
            use_map_self_attention=use_map_self_attention,
        )
        self.policy_head = AgentCentricPolicyHead(
            d_model=d_model,
            hidden_dim=policy_hidden_dim,
            action_low=action_low,
            action_high=action_high,
        )

    def forward(
        self,
        obs: dict[str, Tensor],
        action: Tensor | None = None,
        sampling_method: str | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        encoded = self.encoder(obs)
        return self.policy_head(encoded["ego_feature"], encoded["ego_mask"], action=action, sampling_method=sampling_method)


AgentCentricPolicy = AgentCentricModel
