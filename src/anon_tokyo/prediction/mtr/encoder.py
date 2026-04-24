"""Official-compatible MTR context encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from anon_tokyo.nn.polyline_encoder import PointNetPolylineEncoder
from anon_tokyo.prediction.mtr.attention import MultiheadAttention, MultiheadAttentionLocal, knn_batch_mlogk


def gen_sineembed_for_position(pos_tensor: Tensor, hidden_dim: int = 256) -> Tensor:
    """Sinusoidal position encoding matching official MTR/DAB-DETR."""
    half = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half)

    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    if pos_tensor.shape[-1] == 2:
        return torch.cat((pos_y, pos_x), dim=-1)
    if pos_tensor.shape[-1] == 4:
        w_embed = pos_tensor[..., 2] * scale
        h_embed = pos_tensor[..., 3] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_h = h_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)
        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    raise ValueError(f"Unsupported position dimension: {pos_tensor.shape[-1]}")


class TransformerEncoderLayer(nn.Module):
    """Official MTR transformer encoder layer with optional local attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        normalize_before: bool = False,
        use_local_attn: bool = False,
    ) -> None:
        super().__init__()
        self.use_local_attn = use_local_attn
        self.self_attn = (
            MultiheadAttentionLocal(d_model, nhead, dropout=dropout)
            if use_local_attn
            else MultiheadAttention(d_model, nhead, dropout=dropout)
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        index_pair: Tensor | None = None,
        query_batch_cnt: Tensor | None = None,
        key_batch_cnt: Tensor | None = None,
        index_pair_batch: Tensor | None = None,
    ) -> Tensor:
        if self.normalize_before:
            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(
                q,
                k,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                index_pair=index_pair,
                query_batch_cnt=query_batch_cnt,
                key_batch_cnt=key_batch_cnt,
                index_pair_batch=index_pair_batch,
                local_indices=True,
            )[0]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
            return src + self.dropout2(src2)

        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            index_pair=index_pair,
            query_batch_cnt=query_batch_cnt,
            key_batch_cnt=key_batch_cnt,
            index_pair_batch=index_pair_batch,
            local_indices=True,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)


class MTREncoder(nn.Module):
    """Context encoder matching official ``mtr.models.context_encoder.MTREncoder``."""

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        agent_hist_steps: int = 11,
        num_attn_neighbors: int = 16,
        use_local_attn: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.agent_hist_steps = agent_hist_steps
        self.use_local_attn = use_local_attn
        self.num_attn_neighbors = num_attn_neighbors
        self.num_out_channels = d_model

        self.agent_polyline_encoder = PointNetPolylineEncoder(
            in_channels=29 + 1,
            hidden_dim=256,
            num_layers=3,
            num_pre_layers=1,
            out_channels=d_model,
        )
        self.map_polyline_encoder = PointNetPolylineEncoder(
            in_channels=9,
            hidden_dim=64,
            num_layers=5,
            num_pre_layers=3,
            out_channels=d_model,
        )
        self.self_attn_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    normalize_before=False,
                    use_local_attn=use_local_attn,
                )
                for _ in range(num_layers)
            ]
        )

    def _augment_agent_features(self, batch: dict[str, Tensor]) -> Tensor:
        if batch["obj_trajs"].shape[-1] == 29:
            return batch["obj_trajs"]

        obj_trajs = batch["obj_trajs"]
        obj_mask = batch["obj_trajs_mask"]
        obj_types = batch["obj_types"]
        B, A, T, _ = obj_trajs.shape
        device, dtype = obj_trajs.device, obj_trajs.dtype

        pos_size = obj_trajs[..., 0:6]
        type_onehot = obj_trajs.new_zeros(B, A, T, 5)
        type_onehot[..., 0] = (obj_types == 1).unsqueeze(-1).to(dtype)
        type_onehot[..., 1] = (obj_types == 2).unsqueeze(-1).to(dtype)
        type_onehot[..., 2] = (obj_types == 3).unsqueeze(-1).to(dtype)
        track_idx = batch["track_index_to_predict"]
        center_mask = torch.zeros(B, A, dtype=dtype, device=device)
        center_mask[torch.arange(B, device=device), track_idx.clamp(min=0)] = 1.0
        type_onehot[..., 3] = center_mask.unsqueeze(-1)
        if "sdc_track_index" in batch:
            sdc_idx = batch["sdc_track_index"].clamp(min=0)
            type_onehot[torch.arange(B, device=device), sdc_idx, :, 4] = 1.0
        else:
            type_onehot[:, 0, :, 4] = 1.0

        time_embedding = obj_trajs.new_zeros(B, A, T, T + 1)
        time_embedding[:, :, torch.arange(T, device=device), torch.arange(T, device=device)] = 1.0
        if "timestamps" in batch:
            ts = batch["timestamps"].to(dtype=dtype, device=device)
            time_embedding[..., -1] = ts[:, None, :]
        else:
            time_embedding[..., -1] = torch.linspace(0, 1, T, device=device, dtype=dtype)[None, None]

        heading = obj_trajs[..., 6:8]
        vel = obj_trajs[..., 8:10]
        vel_pre = torch.roll(vel, shifts=1, dims=2)
        acce = (vel - vel_pre) / 0.1
        if T > 1:
            acce[:, :, 0] = acce[:, :, 1]
        out = torch.cat((pos_size, type_onehot, time_embedding, heading, vel, acce), dim=-1)
        out[obj_mask == 0] = 0
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
        out[~batch["map_polylines_mask"].bool()] = 0
        return out

    def apply_local_attn(self, x: Tensor, x_mask: Tensor, x_pos: Tensor, num_of_neighbors: int) -> Tensor:
        batch_size, n_token, d_model = x.shape
        x_full = x.reshape(-1, d_model)
        mask_full = x_mask.reshape(-1)
        pos_full = x_pos.reshape(-1, x_pos.shape[-1])
        batch_idxs_full = torch.arange(batch_size, device=x.device)[:, None].expand(batch_size, n_token).reshape(-1).int()

        x_stack = x_full[mask_full]
        pos_stack = pos_full[mask_full]
        batch_idxs = batch_idxs_full[mask_full]
        if x_stack.numel() == 0:
            return torch.zeros_like(x)

        batch_cnt = torch.stack([(batch_idxs == b).sum() for b in range(batch_size)]).int().to(x.device)
        batch_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
        batch_offsets[1:] = batch_cnt.cumsum(0)
        index_pair = knn_batch_mlogk(pos_stack, pos_stack, batch_idxs, batch_offsets, num_of_neighbors)
        pos_embedding = gen_sineembed_for_position(pos_stack[None, :, 0:2], hidden_dim=d_model)[0]

        output = x_stack
        for layer in self.self_attn_layers:
            output = layer(
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs,
            )

        ret = torch.zeros_like(x_full)
        ret[mask_full] = output.to(dtype=ret.dtype)
        return ret.reshape(batch_size, n_token, d_model)

    def apply_global_attn(self, x: Tensor, x_mask: Tensor, x_pos: Tensor) -> Tensor:
        x_t = x.permute(1, 0, 2)
        pos_embedding = gen_sineembed_for_position(x_pos.permute(1, 0, 2)[..., 0:2], hidden_dim=x.shape[-1])
        for layer in self.self_attn_layers:
            x_t = layer(src=x_t, src_key_padding_mask=~x_mask, pos=pos_embedding)
        return x_t.permute(1, 0, 2)

    def forward(self, batch_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        input_dict = batch_dict.get("input_dict", batch_dict)
        obj_trajs = self._augment_agent_features(input_dict)
        obj_trajs_mask = input_dict["obj_trajs_mask"].bool()
        map_polylines = self._augment_map_features(input_dict)
        map_polylines_mask = input_dict["map_polylines_mask"].bool()

        obj_last_pos = input_dict.get("obj_trajs_last_pos", input_dict.get("obj_positions"))
        map_center = input_dict["map_polylines_center"]
        obj_pos_for_pe = obj_last_pos[..., 0:2]
        map_pos_for_pe = map_center[..., 0:2]

        obj_in = torch.cat((obj_trajs, obj_trajs_mask[..., None].to(obj_trajs.dtype)), dim=-1)
        obj_feature = self.agent_polyline_encoder(obj_in, obj_trajs_mask)
        map_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)

        obj_valid_mask = obj_trajs_mask.sum(dim=-1) > 0
        map_valid_mask = map_polylines_mask.sum(dim=-1) > 0
        token_feature = torch.cat((obj_feature, map_feature), dim=1)
        token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1)
        token_pos = torch.cat((obj_pos_for_pe, map_pos_for_pe), dim=1)

        if self.use_local_attn:
            token_feature = self.apply_local_attn(token_feature, token_mask, token_pos, self.num_attn_neighbors)
        else:
            token_feature = self.apply_global_attn(token_feature, token_mask, token_pos)

        num_objects = obj_feature.shape[1]
        obj_feature = token_feature[:, :num_objects]
        map_feature = token_feature[:, num_objects:]
        track_idx = input_dict["track_index_to_predict"].long()
        if track_idx.ndim == 2:
            center_feature = obj_feature[torch.arange(obj_feature.shape[0], device=obj_feature.device)[:, None], track_idx.clamp(min=0)]
        else:
            center_feature = obj_feature[torch.arange(obj_feature.shape[0], device=obj_feature.device), track_idx.clamp(min=0)]

        batch_dict["center_objects_feature"] = center_feature
        batch_dict["obj_feature"] = obj_feature
        batch_dict["map_feature"] = map_feature
        batch_dict["obj_mask"] = obj_valid_mask
        batch_dict["map_mask"] = map_valid_mask
        batch_dict["obj_pos"] = obj_last_pos
        batch_dict["map_pos"] = map_center
        return batch_dict
