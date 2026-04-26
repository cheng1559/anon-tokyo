"""Query-centric simulation policy model."""

from __future__ import annotations

from contextlib import nullcontext

import torch.nn as nn
from torch import Tensor

from anon_tokyo.simulation.query_centric.encoder import QueryCentricEncoder
from anon_tokyo.simulation.query_centric.policy_head import QueryCentricPolicyHead


class QueryCentricModel(nn.Module):
    """Full PPO policy that encodes each scene once before controlled-agent readout."""

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        sparse_k: int = 16,
        dropout: float = 0.1,
        use_rope: bool = False,
        use_drope: bool = False,
        position_encoding: str | None = "rpe",
        include_goal: bool = True,
        action_low: tuple[float, float] | list[float] = (-5.0, -1.0),
        action_high: tuple[float, float] | list[float] = (3.0, 1.0),
        policy_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = QueryCentricEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            sparse_k=sparse_k,
            dropout=dropout,
            use_rope=use_rope,
            use_drope=use_drope,
            position_encoding=position_encoding,
            include_goal=include_goal,
        )
        self.policy_head = QueryCentricPolicyHead(
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
        profiler = getattr(self, "_profiler", None)
        encoder_timer = profiler.record("policy.encoder") if profiler is not None else nullcontext()
        head_timer = profiler.record("policy.head") if profiler is not None else nullcontext()
        with encoder_timer:
            encoded = self.encoder(obs)
        with head_timer:
            return self.policy_head(
                encoded["ego_feature"],
                encoded["ego_mask"],
                action=action,
                sampling_method=sampling_method,
            )


QueryCentricPolicy = QueryCentricModel

__all__ = ["QueryCentricModel", "QueryCentricPolicy"]
