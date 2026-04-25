"""AnonTokyo: query-centric encoder with DRoPE/RoPE spatial interaction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from anon_tokyo.nn.attention import SparseTopKAttention, select_topk
from anon_tokyo.nn.polyline_encoder import PointNetPolylineEncoder


class _EncoderLayer(nn.Module):
    """Single scene-interaction layer over agent and map tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sparse_k: int,
        dropout: float,
        use_rope: bool,
        use_drope: bool,
        position_encoding: str | None = None,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.mm_attn = SparseTopKAttention(d_model, num_heads, sparse_k, dropout, use_rope, use_drope, position_encoding)
        self.mm_norm = nn.LayerNorm(d_model)
        self.mm_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.mm_ffn_norm = nn.LayerNorm(d_model)
        self.mm_drop = nn.Dropout(dropout)
        self.mm_ffn_drop = nn.Dropout(dropout)

        self.aa_attn = SparseTopKAttention(d_model, num_heads, sparse_k, dropout, use_rope, use_drope, position_encoding)
        self.aa_norm = nn.LayerNorm(d_model)
        self.aa_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.aa_ffn_norm = nn.LayerNorm(d_model)
        self.aa_drop = nn.Dropout(dropout)
        self.aa_ffn_drop = nn.Dropout(dropout)

        self.am_attn = SparseTopKAttention(d_model, num_heads, sparse_k, dropout, use_rope, use_drope, position_encoding)
        self.am_norm = nn.LayerNorm(d_model)
        self.am_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.am_ffn_norm = nn.LayerNorm(d_model)
        self.am_drop = nn.Dropout(dropout)
        self.am_ffn_drop = nn.Dropout(dropout)

        if use_temporal:
            self.temporal_attn = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )

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
        aa_topk_idx: Tensor | None = None,
        am_topk_idx: Tensor | None = None,
        agent_token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        squeeze_time = False
        if agent_feat.ndim == 3:
            squeeze_time = True
            agent_feat = agent_feat.unsqueeze(2)
            agent_pos = agent_pos.unsqueeze(2)
            agent_heading = agent_heading.unsqueeze(2)
            agent_token_mask = agent_mask.unsqueeze(2)
        elif agent_token_mask is None:
            agent_token_mask = agent_mask[:, :, None].expand(agent_feat.shape[:3])

        B, A, T, D = agent_feat.shape
        M = map_feat.shape[1]

        mm_out = self.mm_attn(map_feat, map_feat, map_pos, map_pos, map_heading, map_heading, map_mask, mm_topk_idx)
        map_feat = self.mm_norm(map_feat + self.mm_drop(mm_out))
        map_feat = self.mm_ffn_norm(map_feat + self.mm_ffn_drop(self.mm_ffn(map_feat)))

        agent_bt = agent_feat.permute(0, 2, 1, 3).reshape(B * T, A, D)
        agent_pos_bt = agent_pos.permute(0, 2, 1, 3).reshape(B * T, A, 2)
        agent_heading_bt = agent_heading.permute(0, 2, 1).reshape(B * T, A)
        agent_mask_bt = agent_token_mask.permute(0, 2, 1).reshape(B * T, A)
        if aa_topk_idx is None:
            aa_topk_idx = select_topk(agent_pos_bt, agent_pos_bt, agent_mask_bt, self.aa_attn.sparse_k)
        aa_out = self.aa_attn(
            agent_bt,
            agent_bt,
            agent_pos_bt,
            agent_pos_bt,
            agent_heading_bt,
            agent_heading_bt,
            agent_mask_bt,
            aa_topk_idx,
        )
        agent_bt = self.aa_norm(agent_bt + self.aa_drop(aa_out))
        agent_bt = self.aa_ffn_norm(agent_bt + self.aa_ffn_drop(self.aa_ffn(agent_bt)))

        map_bt = map_feat[:, None].expand(B, T, M, D).reshape(B * T, M, D)
        map_pos_bt = map_pos[:, None].expand(B, T, M, 2).reshape(B * T, M, 2)
        map_heading_bt = map_heading[:, None].expand(B, T, M).reshape(B * T, M)
        map_mask_bt = map_mask[:, None].expand(B, T, M).reshape(B * T, M)
        if am_topk_idx is None:
            am_topk_idx = select_topk(agent_pos_bt, map_pos_bt, map_mask_bt, self.am_attn.sparse_k)
        am_out = self.am_attn(
            agent_bt,
            map_bt,
            agent_pos_bt,
            map_pos_bt,
            agent_heading_bt,
            map_heading_bt,
            map_mask_bt,
            am_topk_idx,
        )
        agent_bt = self.am_norm(agent_bt + self.am_drop(am_out))
        agent_bt = self.am_ffn_norm(agent_bt + self.am_ffn_drop(self.am_ffn(agent_bt)))

        agent_feat = agent_bt.reshape(B, T, A, D).permute(0, 2, 1, 3).contiguous()

        if T > 1 and self.use_temporal:
            temporal_in = agent_feat.reshape(B * A, T, D)
            temporal_pad = ~agent_token_mask.reshape(B * A, T)
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=agent_feat.device),
                diagonal=1,
            )
            temporal_out = self.temporal_attn(temporal_in, src_mask=causal_mask, src_key_padding_mask=temporal_pad)
            agent_feat = temporal_out.reshape(B, A, T, D)
        agent_feat = agent_feat.masked_fill(~agent_token_mask[..., None], 0)
        if squeeze_time:
            agent_feat = agent_feat.squeeze(2)

        return agent_feat, map_feat


class AnonTokyoEncoder(nn.Module):
    """Query-centric encoder with local history polylines and RoPE/DRoPE scene interaction."""

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        sparse_k: int = 16,
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

        self.agent_polyline_encoder = PointNetPolylineEncoder(
            in_channels=agent_in_channels + 1,
            hidden_dim=d_model,
            num_layers=3,
            num_pre_layers=1,
            out_channels=d_model,
        )
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
                    use_temporal=False,
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

        agent_pos = batch.get("obj_positions")
        if agent_pos is None:
            valid_count = obj_mask.long().sum(dim=-1)
            last_idx = valid_count.sub(1).clamp(min=0)
            b_idx = torch.arange(B, device=device)[:, None].expand(B, A)
            a_idx = torch.arange(A, device=device)[None, :].expand(B, A)
            agent_pos = obj_trajs[b_idx, a_idx, last_idx, 0:2]
        agent_pos = agent_pos.to(device=device, dtype=dtype)

        agent_heading = batch.get("obj_headings")
        if agent_heading is None:
            valid_count = obj_mask.long().sum(dim=-1)
            last_idx = valid_count.sub(1).clamp(min=0)
            b_idx = torch.arange(B, device=device)[:, None].expand(B, A)
            a_idx = torch.arange(A, device=device)[None, :].expand(B, A)
            agent_heading = torch.atan2(obj_trajs[b_idx, a_idx, last_idx, 6], obj_trajs[b_idx, a_idx, last_idx, 7])
        agent_heading = agent_heading.to(device=device, dtype=dtype)

        c = agent_heading.cos()[:, :, None, None]
        s = agent_heading.sin()[:, :, None, None]

        rel_xy = obj_trajs[..., 0:2] - agent_pos[:, :, None, :]
        rel_x, rel_y = rel_xy[..., 0:1], rel_xy[..., 1:2]
        local_xy = torch.cat((rel_x * c + rel_y * s, -rel_x * s + rel_y * c), dim=-1)

        heading_t = torch.atan2(obj_trajs[..., 6], obj_trajs[..., 7])
        rel_heading = heading_t - agent_heading[:, :, None]
        heading_feature = torch.stack((rel_heading.sin(), rel_heading.cos()), dim=-1)

        vel = obj_trajs[..., 8:10]
        vx, vy = vel[..., 0:1], vel[..., 1:2]
        local_vel = torch.cat((vx * c + vy * s, -vx * s + vy * c), dim=-1)

        pos_size = torch.cat((local_xy, obj_trajs[..., 2:6]), dim=-1)

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

        vel_pre = torch.roll(local_vel, shifts=1, dims=2)
        acce = (local_vel - vel_pre) / 0.1
        if T > 1:
            acce[:, :, 0] = acce[:, :, 1]

        out = torch.cat((pos_size, type_onehot, time_embedding, heading_feature, local_vel, acce), dim=-1)
        out = out.masked_fill(~obj_mask[..., None], 0)
        return out

    @staticmethod
    def _augment_map_features(batch: dict[str, Tensor]) -> Tensor:
        map_polys = batch["map_polylines"]
        center = batch["map_polylines_center"][:, :, None, :]
        heading = batch["map_headings"][:, :, None, None]

        c = heading.cos()
        s = heading.sin()

        xy = map_polys[..., 0:2] - center
        x, y = xy[..., 0:1], xy[..., 1:2]
        local_xy = torch.cat((x * c + y * s, -x * s + y * c), dim=-1)

        direction = map_polys[..., 3:5]
        dx, dy = direction[..., 0:1], direction[..., 1:2]
        local_dir = torch.cat((dx * c + dy * s, -dx * s + dy * c), dim=-1)

        pre_xy = torch.roll(local_xy, shifts=1, dims=2)
        if local_xy.shape[2] > 1:
            first_pre = pre_xy.clone()
            first_pre[:, :, 0] = first_pre[:, :, 1]
            pre_xy = first_pre

        out = torch.cat((local_xy, map_polys[..., 2:3], local_dir, map_polys[..., 5:7], pre_xy), dim=-1)
        out = out.masked_fill(~batch["map_polylines_mask"].bool()[..., None], 0)
        return out

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Expected batch keys (query-centric, from dataloader collation):
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

        B, A, _ = obj_mask_t.shape
        obj_trajs = self._augment_agent_features(batch)
        map_polys = self._augment_map_features(batch)
        if obj_trajs.shape[-1] != self.agent_in_channels:
            raise ValueError(f"Expected {self.agent_in_channels} agent features, got {obj_trajs.shape[-1]}")
        if map_polys.shape[-1] != self.map_in_channels:
            raise ValueError(f"Expected {self.map_in_channels} map features, got {map_polys.shape[-1]}")

        agent_token_mask = obj_mask_t.bool()
        agent_input = torch.cat([obj_trajs, agent_token_mask.unsqueeze(-1).to(dtype=obj_trajs.dtype)], dim=-1)

        agent_feat = self.agent_polyline_encoder(agent_input, agent_token_mask)  # [B, A, D]
        agent_feat = agent_feat.masked_fill(~agent_mask[..., None], 0)
        map_feat = self.map_encoder(map_polys, map_poly_mask.bool())  # [B, M, D]

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
                aa_topk_idx=aa_topk,
                am_topk_idx=am_topk,
            )

        agent_feat = agent_feat.masked_fill(~agent_mask[..., None], 0)

        return {
            "obj_feature": agent_feat,
            "map_feature": map_feat,
            "obj_mask": agent_mask,
            "map_mask": map_mask,
            "obj_pos": agent_pos,
            "map_pos": map_center,
            "obj_headings": agent_heading,
        }
