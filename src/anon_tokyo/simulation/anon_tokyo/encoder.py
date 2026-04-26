"""AnonTokyo simulation encoder with RoPE/DRoPE scene interaction."""

from __future__ import annotations

from torch import Tensor

from anon_tokyo.simulation.query_centric.encoder import QueryCentricEncoder


class AnonTokyoEncoder(QueryCentricEncoder):
    """Query-centric simulation encoder using AnonTokyo RoPE/DRoPE attention."""

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        sparse_k: int = 16,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_drope: bool = True,
        position_encoding: str | None = "rope_drope",
        agent_in_channels: int = 19,
        map_in_channels: int = 9,
        include_goal: bool = True,
    ) -> None:
        super().__init__(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            sparse_k=sparse_k,
            dropout=dropout,
            use_rope=use_rope,
            use_drope=use_drope,
            position_encoding=position_encoding,
            agent_in_channels=agent_in_channels,
            map_in_channels=map_in_channels,
            include_goal=include_goal,
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return super().forward(batch)


__all__ = ["AnonTokyoEncoder"]
