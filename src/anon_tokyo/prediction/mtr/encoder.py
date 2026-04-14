"""MTR encoder: feature augmentation → PointNet embedding → global self-attention.

Matches the official MTR implementation:
- Sinusoidal PE with 2π scale, Y-first order, interleaved sin/cos
- Per-layer PE injection (Q = K = src + pos, V = src)
- Agent feature augmentation inside encoder (type one-hot, time embed, heading, acceleration)
- Map pre_xy augmentation inside encoder (roll + concat)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from anon_tokyo.nn.polyline_encoder import PointNetPolylineEncoder


def gen_sineembed_for_position(pos_tensor: Tensor, hidden_dim: int = 256) -> Tensor:
    """Sinusoidal PE matching official MTR / DAB-DETR.

    Args:
        pos_tensor: ``[..., 2]`` or ``[..., 4]`` — coordinates.
        hidden_dim: output dim (must be divisible by 2).

    Returns:
        ``[..., hidden_dim]``
    """
    half = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half)

    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed.unsqueeze(-1) / dim_t  # [..., half]
    pos_y = y_embed.unsqueeze(-1) / dim_t

    # Interleaved sin/cos: (sin, cos, sin, cos, ...)
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

    # Y-first, then X (matching official)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[..., 2] * scale
        h_embed = pos_tensor[..., 3] * scale
        pos_w = w_embed.unsqueeze(-1) / dim_t
        pos_h = h_embed.unsqueeze(-1) / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1): {pos_tensor.size(-1)}")
    return pos


# ── Custom encoder layer with per-layer PE injection ─────────────────────────


class _EncoderLayerWithPE(nn.Module):
    """Post-norm transformer encoder layer with positional encoding
    added to Q and K only (not V), matching official MTR."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, pos: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        """
        Args:
            src: [N, B, D]
            pos: [N, B, D]  — positional encoding
            src_key_padding_mask: [B, N]  — True = ignore
        """
        q = k = src + pos  # PE added to Q and K
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MTREncoder(nn.Module):
    """Scene-level encoder that jointly operates on agents + map polylines."""

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        agent_hist_steps: int = 11,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.agent_hist_steps = agent_hist_steps

        # Agent feature dim: 6(pos/size) + 5(type) + (T+1)(time) + 2(heading) + 2(vel) + 2(acce) + 1(mask)
        agent_total_channels = 6 + 5 + (agent_hist_steps + 1) + 2 + 2 + 2 + 1

        # PointNet for agent trajectories
        self.agent_encoder = PointNetPolylineEncoder(
            in_channels=agent_total_channels,
            hidden_dim=d_model,
            num_layers=3,
            num_pre_layers=1,
            out_channels=d_model,
        )
        # PointNet for map polylines (9 = 7 original + 2 pre_xy)
        self.map_encoder = PointNetPolylineEncoder(
            in_channels=9,  # x,y,z,dir_x,dir_y,dir_z,type,pre_x,pre_y
            hidden_dim=64,
            num_layers=5,
            num_pre_layers=3,
            out_channels=d_model,
        )

        # Custom encoder layers with per-layer PE injection
        self.self_attn_layers = nn.ModuleList(
            [_EncoderLayerWithPE(d_model, num_heads, d_model * 4, dropout) for _ in range(num_layers)]
        )

    def _augment_agent_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Augment raw agent trajectories with type, time, heading, acceleration.

        Official MTR feature order:
            [x, y, z, dx, dy, dz] (6) +
            [is_vehicle, is_ped, is_cyclist, is_center, is_sdc] (5) +
            [time_one_hot(T), global_time] (T+1) +
            [sin(heading), cos(heading)] (2) +
            [vx, vy] (2) +
            [ax, ay] (2) +
            [mask] (1) = total: 6 + 5 + (T+1) + 2 + 2 + 2 + 1

        Our raw input: [x, y, z, dx, dy, dz, sin_h, cos_h, vx, vy] (10)
        """
        obj_trajs = batch["obj_trajs"]  # [B, A, T, 10]
        obj_mask = batch["obj_trajs_mask"]  # [B, A, T]
        obj_types = batch["obj_types"]  # [B, A]
        B, A, T, _ = obj_trajs.shape
        device = obj_trajs.device
        dtype = obj_trajs.dtype

        # Position + size: [x, y, z, dx, dy, dz]
        pos_size = obj_trajs[..., 0:6]  # [B, A, T, 6]

        # Type one-hot: [vehicle, pedestrian, cyclist, is_center, is_sdc]
        type_onehot = obj_trajs.new_zeros(B, A, T, 5)
        type_onehot[:, :, :, 0] = (obj_types == 1).unsqueeze(-1).to(dtype)  # vehicle
        type_onehot[:, :, :, 1] = (obj_types == 2).unsqueeze(-1).to(dtype)  # pedestrian
        type_onehot[:, :, :, 2] = (obj_types == 3).unsqueeze(-1).to(dtype)  # cyclist
        # is_center: the track_index_to_predict agent
        track_idx = batch["track_index_to_predict"]  # [B]
        center_mask = torch.zeros(B, A, device=device, dtype=dtype)
        b_idx = torch.arange(B, device=device)
        center_mask[b_idx, track_idx] = 1.0
        type_onehot[:, :, :, 3] = center_mask.unsqueeze(-1)
        # is_sdc: agent index 0 is SDC in our agent_centric_preprocess (inherited from scene-centric)
        type_onehot[:, 0, :, 4] = 1.0

        # Time embedding: one-hot + global timestamp
        time_onehot = obj_trajs.new_zeros(B, A, T, T + 1)
        time_onehot[:, :, torch.arange(T), torch.arange(T)] = 1.0
        # Global timestamp: linear from 0 to 1 (we don't have raw timestamps,
        # use normalized step index matching official behavior)
        timestamps = torch.linspace(0, 1, T, device=device, dtype=dtype)
        time_onehot[:, :, :, -1] = timestamps[None, None, :]

        # Heading embedding: sin(heading), cos(heading) — already in raw features
        heading = obj_trajs[..., 6:8]  # [B, A, T, 2]  (sin_h, cos_h)

        # Velocity
        vel = obj_trajs[..., 8:10]  # [B, A, T, 2]

        # Acceleration from velocity difference (dt = 0.1s)
        vel_pre = torch.roll(vel, shifts=1, dims=2)
        acce = (vel - vel_pre) / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]  # first step: copy from second

        # Concatenate all features (mask appended later by forward)
        augmented = torch.cat(
            [
                pos_size,  # 6
                type_onehot,  # 5
                time_onehot,  # T + 1
                heading,  # 2
                vel,  # 2
                acce,  # 2
            ],
            dim=-1,
        )  # [B, A, T, 6+5+(T+1)+2+2+2 = 18+T]

        return augmented

    def _augment_map_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Add pre_x, pre_y to map polylines (roll along points dim).

        Input:  [B, M, P, 7]  — [x, y, z, dir_x, dir_y, dir_z, type]
        Output: [B, M, P, 9]  — [x, y, z, dir_x, dir_y, dir_z, type, pre_x, pre_y]
        """
        map_polys = batch["map_polylines"]  # [B, M, P, 7]
        map_mask = batch["map_polylines_mask"]  # [B, M, P]

        xy = map_polys[..., 0:2]  # [B, M, P, 2]
        pre_xy = torch.roll(xy, shifts=1, dims=2)  # shift along points dim
        pre_xy[:, :, 0, :] = pre_xy[:, :, 1, :]  # first point: use second point

        augmented = torch.cat([map_polys, pre_xy], dim=-1)  # [B, M, P, 9]
        augmented[~map_mask.bool()] = 0
        return augmented

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Expected batch keys (from agent_centric_preprocess):
            obj_trajs: [B, A, T_hist, 10]
            obj_trajs_mask: [B, A, T_hist]
            obj_positions: [B, A, 2]
            obj_types: [B, A]
            agent_mask: [B, A]
            map_polylines: [B, M, P, 7]
            map_polylines_mask: [B, M, P]
            map_polylines_center: [B, M, 2]
            map_mask: [B, M]
            tracks_to_predict: [B, K_max]
            track_index_to_predict: [B]

        Returns dict with:
            obj_feature: [B, A, D]
            map_feature: [B, M, D]
            center_objects_feature: [B, K, D]
        """
        agent_mask = batch["agent_mask"].bool()  # [B, A]
        obj_pos = batch["obj_positions"]  # [B, A, 2]
        map_poly_mask = batch["map_polylines_mask"]  # [B, M, P]
        map_center = batch["map_polylines_center"]  # [B, M, 2]
        map_mask = batch["map_mask"].bool()  # [B, M]

        B, A = agent_mask.shape
        M = map_poly_mask.shape[1]

        # Feature augmentation (inside model, not dataset)
        agent_augmented = self._augment_agent_features(batch)  # [B, A, T, 18+T]
        map_input = self._augment_map_features(batch)  # [B, M, P, 9]

        obj_mask_t = batch["obj_trajs_mask"].bool()  # [B, A, T]

        # Append mask channel (matching official: cat(features, mask) then encode)
        agent_input = torch.cat(
            [
                agent_augmented,
                obj_mask_t.unsqueeze(-1).to(agent_augmented.dtype),
            ],
            dim=-1,
        )  # [B, A, T, 19+T]

        obj_feat = self.agent_encoder(agent_input, obj_mask_t)  # [B, A, D]
        map_feat = self.map_encoder(map_input, map_poly_mask.bool())  # [B, M, D]

        # Concat agent + map tokens
        tokens = torch.cat([obj_feat, map_feat], dim=1)  # [B, N, D]  N = A + M
        token_mask = torch.cat([agent_mask, map_mask], dim=1)  # [B, N]
        token_pos = torch.cat([obj_pos, map_center], dim=1)  # [B, N, 2]

        # Sinusoidal PE (computed once, reused for all layers)
        tokens_t = tokens.permute(1, 0, 2)  # [N, B, D]
        pos_emb = gen_sineembed_for_position(token_pos.permute(1, 0, 2), hidden_dim=self.d_model)  # [N, B, D]
        pad_mask = ~token_mask  # True = padded

        # Per-layer PE injection: Q = K = src + pos, V = src
        for layer in self.self_attn_layers:
            tokens_t = layer(tokens_t, pos_emb, src_key_padding_mask=pad_mask)

        tokens = tokens_t.permute(1, 0, 2)  # [B, N, D]

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
