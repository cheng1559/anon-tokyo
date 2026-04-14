"""MTR encoder: PointNet embedding → global self-attention with sinusoidal PE.

Follows the official MTR implementation but uses ``nn.TransformerEncoderLayer``
and our scene-centric data pipeline (single ego coordinate frame for all agents).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from anon_tokyo.nn.polyline_encoder import PointNetPolylineEncoder


def _sinusoidal_pe(pos: Tensor, d_model: int) -> Tensor:
    """Sinusoidal positional encoding for 2-D coordinates.

    Args:
        pos: ``[..., 2]`` — (x, y) coordinates.
        d_model: embedding dimension (must be divisible by 4).

    Returns:
        ``[..., d_model]``
    """
    half = d_model // 2
    dim_t = torch.arange(half, dtype=torch.float32, device=pos.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half)
    x_emb = pos[..., 0:1] / dim_t  # [..., half]
    y_emb = pos[..., 1:2] / dim_t  # [..., half]
    pe = torch.cat([x_emb.sin(), x_emb.cos(), y_emb.sin(), y_emb.cos()], dim=-1)
    return pe[..., :d_model]


class MTREncoder(nn.Module):
    """Scene-level encoder that jointly operates on agents + map polylines."""

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        agent_in_channels: int = 10,
        agent_hist_steps: int = 11,
        map_in_channels: int = 7,
        map_num_points: int = 20,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # PointNet for agent trajectories: input = hist features + mask channel
        self.agent_encoder = PointNetPolylineEncoder(
            in_channels=agent_in_channels + 1,  # +1 for validity mask channel
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

        # Transformer layers (post-norm, like MTR)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=False,  # (S, B, D) convention
            norm_first=False,
        )
        self.attn_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Expected batch keys (from collated WOMDDataset):
            obj_trajs: [B, A, T_hist, 10]
            obj_trajs_mask: [B, A, T_hist]
            obj_positions: [B, A, 2]
            agent_mask: [B, A]
            map_polylines: [B, M, P, 7]
            map_polylines_mask: [B, M, P]
            map_polylines_center: [B, M, 2]
            map_mask: [B, M]
            tracks_to_predict: [B, K_max]

        Returns dict with:
            obj_feature: [B, A, D]
            map_feature: [B, M, D]
            center_objects_feature: [B, K, D]  (K = tracks_to_predict count)
        """
        obj_trajs = batch["obj_trajs"]  # [B, A, T, 10]
        obj_mask_t = batch["obj_trajs_mask"]  # [B, A, T]
        agent_mask = batch["agent_mask"].bool()  # [B, A]
        obj_pos = batch["obj_positions"]  # [B, A, 2]

        map_polys = batch["map_polylines"]  # [B, M, P, 7]
        map_poly_mask = batch["map_polylines_mask"]  # [B, M, P]
        map_center = batch["map_polylines_center"]  # [B, M, 2]
        map_mask = batch["map_mask"].bool()  # [B, M]

        B, A, T, C_agent = obj_trajs.shape
        M = map_polys.shape[1]

        # Append mask channel to agent trajs
        agent_input = torch.cat([obj_trajs, obj_mask_t.unsqueeze(-1)], dim=-1)  # [B, A, T, 11]
        obj_feat = self.agent_encoder(agent_input, obj_mask_t.bool())  # [B, A, D]
        map_feat = self.map_encoder(map_polys, map_poly_mask.bool())  # [B, M, D]

        # Concat agent + map tokens
        tokens = torch.cat([obj_feat, map_feat], dim=1)  # [B, N, D]  N = A + M
        token_mask = torch.cat([agent_mask, map_mask], dim=1)  # [B, N]
        token_pos = torch.cat([obj_pos, map_center], dim=1)  # [B, N, 2]

        # Sinusoidal PE added to tokens
        pe = _sinusoidal_pe(token_pos, self.d_model)  # [B, N, D]
        tokens = tokens + pe

        # Transformer (S, B, D) convention; key_padding_mask: True = ignore
        tokens_t = tokens.transpose(0, 1)  # [N, B, D]
        pad_mask = ~token_mask  # True = padded
        tokens_t = self.attn_layers(tokens_t, src_key_padding_mask=pad_mask)
        tokens = tokens_t.transpose(0, 1)  # [B, N, D]

        obj_feature = tokens[:, :A]
        map_feature = tokens[:, A:]

        # Extract center object features for tracks_to_predict
        ttp = batch["tracks_to_predict"]  # [B, K_max]
        ttp_clamped = ttp.clamp(min=0)
        b_idx = torch.arange(B, device=ttp.device)[:, None].expand_as(ttp_clamped)
        center_feat = obj_feature[b_idx, ttp_clamped]  # [B, K_max, D]

        return {
            "obj_feature": obj_feature,
            "map_feature": map_feature,
            "center_objects_feature": center_feat,
            "obj_mask": agent_mask,
            "map_mask": map_mask,
            "obj_pos": obj_pos,
            "map_pos": map_center,
        }
