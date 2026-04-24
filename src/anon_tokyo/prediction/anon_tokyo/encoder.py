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
        agent_in_channels: int = 29,
        map_in_channels: int = 9,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.sparse_k = sparse_k
        self.agent_in_channels = agent_in_channels
        self.map_in_channels = map_in_channels

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

    @staticmethod
    def _augment_agent_features(batch: dict[str, Tensor]) -> Tensor:
        obj_trajs = batch["obj_trajs"]
        obj_mask = batch["obj_trajs_mask"].bool()
        B, A, T, _ = obj_trajs.shape
        dtype = obj_trajs.dtype
        device = obj_trajs.device

        pos_size = obj_trajs[..., 0:6]

        obj_types = batch.get("obj_types")
        if obj_types is None:
            obj_types = torch.zeros(B, A, dtype=torch.long, device=device)
        obj_types = obj_types.to(device=device)
        type_onehot = obj_trajs.new_zeros(B, A, T, 5)
        type_onehot[..., 0] = (obj_types == 1).to(dtype=dtype)[:, :, None]
        type_onehot[..., 1] = (obj_types == 2).to(dtype=dtype)[:, :, None]
        type_onehot[..., 2] = (obj_types == 3).to(dtype=dtype)[:, :, None]

        ttp = batch.get("tracks_to_predict")
        if ttp is not None:
            ttp = ttp.to(device=device).long()
            valid = (ttp >= 0) & (ttp < A)
            if valid.any():
                bi = torch.arange(B, device=device)[:, None].expand_as(ttp)
                type_onehot[bi[valid], ttp.clamp(min=0)[valid], :, 3] = 1

        sdc_idx = batch.get("sdc_track_index")
        if sdc_idx is not None:
            sdc_idx = sdc_idx.to(device=device).long()
            if sdc_idx.ndim == 0:
                sdc_idx = sdc_idx[None].expand(B)
            valid_sdc = (sdc_idx >= 0) & (sdc_idx < A)
            if valid_sdc.any():
                bi = torch.arange(B, device=device)[valid_sdc]
                type_onehot[bi, sdc_idx[valid_sdc], :, 4] = 1

        time_embedding = obj_trajs.new_zeros(B, A, T, T + 1)
        time_idx = torch.arange(T, device=device)
        time_embedding[:, :, time_idx, time_idx] = 1
        if "timestamps" in batch:
            ts = batch["timestamps"].to(dtype=dtype, device=device)
            if ts.ndim == 1:
                ts = ts[None].expand(B, -1)
            time_embedding[..., -1] = ts[:, None, :T]
        else:
            time_embedding[..., -1] = torch.linspace(0, 1, T, device=device, dtype=dtype)[None, None]

        heading = obj_trajs[..., 6:8]
        vel = obj_trajs[..., 8:10]
        vel_pre = torch.roll(vel, shifts=1, dims=2)
        acce = (vel - vel_pre) / 0.1
        if T > 1:
            acce[:, :, 0] = acce[:, :, 1]

        out = torch.cat((pos_size, type_onehot, time_embedding, heading, vel, acce), dim=-1)
        out = out.masked_fill(~obj_mask[..., None], 0)
        return out

    @staticmethod
    def _augment_map_features(batch: dict[str, Tensor]) -> Tensor:
        map_polys = batch["map_polylines"]
        if map_polys.shape[-1] == 9:
            return map_polys
        xy = map_polys[..., 0:2]
        pre_xy = torch.roll(xy, shifts=1, dims=2)
        if xy.shape[2] > 1:
            pre_xy[:, :, 0] = pre_xy[:, :, 1]
        out = torch.cat((map_polys, pre_xy), dim=-1)
        out = out.masked_fill(~batch["map_polylines_mask"].bool()[..., None], 0)
        return out

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
        obj_mask_t = batch["obj_trajs_mask"]  # [B, A, T]
        agent_mask = batch["agent_mask"].bool()  # [B, A]
        agent_pos = batch["obj_positions"]  # [B, A, 2]
        agent_heading = batch["obj_headings"]  # [B, A]

        map_poly_mask = batch["map_polylines_mask"]  # [B, M, P]
        map_center = batch["map_polylines_center"]  # [B, M, 2]
        map_heading = batch["map_headings"]  # [B, M]
        map_mask = batch["map_mask"].bool()  # [B, M]

        obj_trajs = self._augment_agent_features(batch)
        map_polys = self._augment_map_features(batch)
        if obj_trajs.shape[-1] != self.agent_in_channels:
            raise ValueError(f"Expected {self.agent_in_channels} agent features, got {obj_trajs.shape[-1]}")
        if map_polys.shape[-1] != self.map_in_channels:
            raise ValueError(f"Expected {self.map_in_channels} map features, got {map_polys.shape[-1]}")

        agent_input = torch.cat([obj_trajs, obj_mask_t.unsqueeze(-1).to(dtype=obj_trajs.dtype)], dim=-1)

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
