"""AnonTokyo: scene-centric encoder with RoPE/DRoPE + sparse top-k attention.

Dual-stream architecture (Map Stream + Agent Stream) with 3 attention
blocks per layer: Map-Map → Agent-Agent → Agent-Map, all using sparse
top-k neighbours with RoPE (even heads) and DRoPE (odd heads).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from anon_tokyo.nn.attention import SparseTopKAttention, select_topk
from anon_tokyo.nn.polyline_encoder import PointNetPolylineEncoder


class _EncoderLayer(nn.Module):
    """Single encoder layer: Map-Map → Agent-Agent → Agent-Map."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sparse_k: int,
        dropout: float,
        use_rope: bool,
        use_drope: bool,
        position_encoding: str | None = None,
    ) -> None:
        super().__init__()
        # Map-Map self-attention
        self.mm_attn = SparseTopKAttention(d_model, num_heads, sparse_k, dropout, use_rope, use_drope, position_encoding)
        self.mm_norm = nn.LayerNorm(d_model)
        self.mm_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.mm_ffn_norm = nn.LayerNorm(d_model)
        self.mm_drop = nn.Dropout(dropout)
        self.mm_ffn_drop = nn.Dropout(dropout)

        # Agent-Agent self-attention
        self.aa_attn = SparseTopKAttention(d_model, num_heads, sparse_k, dropout, use_rope, use_drope, position_encoding)
        self.aa_norm = nn.LayerNorm(d_model)
        self.aa_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.aa_ffn_norm = nn.LayerNorm(d_model)
        self.aa_drop = nn.Dropout(dropout)
        self.aa_ffn_drop = nn.Dropout(dropout)

        # Agent-Map cross-attention
        self.am_attn = SparseTopKAttention(d_model, num_heads, sparse_k, dropout, use_rope, use_drope, position_encoding)
        self.am_norm = nn.LayerNorm(d_model)
        self.am_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.am_ffn_norm = nn.LayerNorm(d_model)
        self.am_drop = nn.Dropout(dropout)
        self.am_ffn_drop = nn.Dropout(dropout)

    def forward(
        self,
        agent_feat: Tensor,
        map_feat: Tensor,
        agent_pos: Tensor,
        map_pos: Tensor,
        agent_heading: Tensor,
        map_heading: Tensor,
        agent_mask: Tensor,
        map_mask: Tensor,
        mm_topk_idx: Tensor,
        aa_topk_idx: Tensor,
        am_topk_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Map-Map self-attention
        mm_out = self.mm_attn(map_feat, map_feat, map_pos, map_pos, map_heading, map_heading, map_mask, mm_topk_idx)
        map_feat = self.mm_norm(map_feat + self.mm_drop(mm_out))
        map_feat = self.mm_ffn_norm(map_feat + self.mm_ffn_drop(self.mm_ffn(map_feat)))

        # Agent-Agent self-attention
        aa_out = self.aa_attn(
            agent_feat, agent_feat, agent_pos, agent_pos, agent_heading, agent_heading, agent_mask, aa_topk_idx
        )
        agent_feat = self.aa_norm(agent_feat + self.aa_drop(aa_out))
        agent_feat = self.aa_ffn_norm(agent_feat + self.aa_ffn_drop(self.aa_ffn(agent_feat)))

        # Agent-Map cross-attention
        am_out = self.am_attn(
            agent_feat, map_feat, agent_pos, map_pos, agent_heading, map_heading, map_mask, am_topk_idx
        )
        agent_feat = self.am_norm(agent_feat + self.am_drop(am_out))
        agent_feat = self.am_ffn_norm(agent_feat + self.am_ffn_drop(self.am_ffn(agent_feat)))

        return agent_feat, map_feat


class AnonTokyoEncoder(nn.Module):
    """Scene-centric encoder with dual-stream sparse attention + RoPE/DRoPE."""

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        sparse_k: int = 32,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_drope: bool = True,
        position_encoding: str | None = None,
        agent_in_channels: int = 10,
        map_in_channels: int = 7,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.sparse_k = sparse_k

        # PointNet for agent trajectories (+1 for mask channel)
        self.agent_encoder = PointNetPolylineEncoder(
            in_channels=agent_in_channels + 1,
            hidden_dim=d_model,
            num_layers=3,
            num_pre_layers=1,
            out_channels=d_model,
        )
        # PointNet for map polylines
        self.map_encoder = PointNetPolylineEncoder(
            in_channels=map_in_channels,
            hidden_dim=64,
            num_layers=5,
            num_pre_layers=3,
            out_channels=d_model,
        )

        self.layers = nn.ModuleList(
            [
                _EncoderLayer(
                    d_model,
                    num_heads,
                    sparse_k,
                    dropout,
                    use_rope,
                    use_drope,
                    position_encoding,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Expected batch keys (scene-centric, from dataloader collation):
            obj_trajs:            [B, A, T_hist, 10]
            obj_trajs_mask:       [B, A, T_hist]
            obj_positions:        [B, A, 2]
            obj_headings:         [B, A]
            agent_mask:           [B, A]
            map_polylines:        [B, M, P, 7]
            map_polylines_mask:   [B, M, P]
            map_polylines_center: [B, M, 2]
            map_headings:         [B, M]
            map_mask:             [B, M]

        Returns:
            obj_feature:  [B, A, D]
            map_feature:  [B, M, D]
            obj_mask:     [B, A]
            map_mask_out: [B, M]
            obj_pos:      [B, A, 2]
            map_pos:      [B, M, 2]
        """
        obj_trajs = batch["obj_trajs"]  # [B, A, T, 10]
        obj_mask_t = batch["obj_trajs_mask"]  # [B, A, T]
        agent_mask = batch["agent_mask"].bool()  # [B, A]
        agent_pos = batch["obj_positions"]  # [B, A, 2]
        agent_heading = batch["obj_headings"]  # [B, A]

        map_polys = batch["map_polylines"]  # [B, M, P, 7]
        map_poly_mask = batch["map_polylines_mask"]  # [B, M, P]
        map_center = batch["map_polylines_center"]  # [B, M, 2]
        map_heading = batch["map_headings"]  # [B, M]
        map_mask = batch["map_mask"].bool()  # [B, M]

        # Append mask channel to agent trajectories
        agent_input = torch.cat([obj_trajs, obj_mask_t.unsqueeze(-1)], dim=-1)

        # PointNet embedding
        agent_feat = self.agent_encoder(agent_input, obj_mask_t.bool())  # [B, A, D]
        map_feat = self.map_encoder(map_polys, map_poly_mask.bool())  # [B, M, D]

        # Precompute top-k indices (static, reused across layers)
        mm_topk = select_topk(map_center, map_center, map_mask, self.sparse_k)
        aa_topk = select_topk(agent_pos, agent_pos, agent_mask, self.sparse_k)
        am_topk = select_topk(agent_pos, map_center, map_mask, self.sparse_k)

        for layer in self.layers:
            agent_feat, map_feat = layer(
                agent_feat,
                map_feat,
                agent_pos,
                map_center,
                agent_heading,
                map_heading,
                agent_mask,
                map_mask,
                mm_topk,
                aa_topk,
                am_topk,
            )

        return {
            "obj_feature": agent_feat,
            "map_feature": map_feat,
            "obj_mask": agent_mask,
            "map_mask": map_mask,
            "obj_pos": agent_pos,
            "map_pos": map_center,
            "obj_headings": agent_heading,
        }
