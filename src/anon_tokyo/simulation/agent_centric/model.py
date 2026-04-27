"""Agent-centric simulation policy model."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from anon_tokyo.simulation.agent_centric.agentcentric import AgentCentricBackbone


class AgentCentricModel(nn.Module):
    """Agent-centric PPO policy."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        max_context_agents: int = 32,
        action_low: tuple[float, float] | list[float] = (-5.0, -1.0),
        action_high: tuple[float, float] | list[float] = (3.0, 1.0),
        architecture: str = "agentcentric",
        max_lanes: int = 96,
        history_steps: int = 5,
        enable_occupancy_grid: bool = True,
        no_goal_allowed: bool = True,
        agent_filter_radius: float = 200.0,
        topk_front_weight: float = 10.0,
        topk_rear_weight: float = 2.0,
        use_layer_norm_layout: bool = False,
    ) -> None:
        super().__init__()
        if architecture != "agentcentric":
            raise ValueError(f"Unsupported AgentCentricModel architecture: {architecture}")
        self.architecture = "agentcentric"
        self.model = AgentCentricBackbone(
            embed_dim=d_model,
            num_heads=num_heads,
            max_agents=max_context_agents,
            max_lanes=max_lanes,
            history_steps=history_steps,
            enable_occupancy_grid=enable_occupancy_grid,
            no_goal_allowed=no_goal_allowed,
            agent_filter_radius=agent_filter_radius,
            topk_front_weight=topk_front_weight,
            topk_rear_weight=topk_rear_weight,
            use_layer_norm_layout=use_layer_norm_layout,
            action_low=action_low,
            action_high=action_high,
        )

    def forward(
        self,
        obs: dict[str, Tensor],
        action: Tensor | None = None,
        sampling_method: str | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.model(obs, action=action, sampling_method=sampling_method)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # type: ignore[override]
        remapped = {}
        for key, value in state_dict.items():
            clean_key = key
            while clean_key.startswith("module."):
                clean_key = clean_key[len("module.") :]
            legacy_policy_prefix = "hermes" + "_policy."
            if clean_key.startswith(legacy_policy_prefix):
                clean_key = "model." + clean_key[len(legacy_policy_prefix) :]
            elif clean_key.startswith("model.") and not clean_key.startswith(
                ("model.model.", "model.action_low", "model.action_high")
            ):
                clean_key = "model." + clean_key
            elif not clean_key.startswith("model."):
                clean_key = "model.model." + clean_key
            remapped[clean_key] = value
        state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


AgentCentricPolicy = AgentCentricModel
