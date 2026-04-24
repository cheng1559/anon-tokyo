"""Official-compatible MTR motion decoder."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from anon_tokyo.nn.layers import build_mlps
from anon_tokyo.prediction.mtr.attention import MultiheadAttention, MultiheadAttentionLocal
from anon_tokyo.prediction.mtr.encoder import gen_sineembed_for_position

_TYPE_MAP = {1: "TYPE_VEHICLE", 2: "TYPE_PEDESTRIAN", 3: "TYPE_CYCLIST"}
_TYPE_STRING_TO_ID = {"TYPE_VEHICLE": 1, "TYPE_PEDESTRIAN": 2, "TYPE_CYCLIST": 3}


class TransformerDecoderLayer(nn.Module):
    """MTR decoder layer with official parameter names and PE concatenation."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        keep_query_pos: bool = False,
        rm_self_attn_decoder: bool = False,
        use_local_attn: bool = False,
    ) -> None:
        super().__init__()
        del activation
        self.use_local_attn = use_local_attn
        self.rm_self_attn_decoder = rm_self_attn_decoder
        self.keep_query_pos = keep_query_pos
        self.normalize_before = normalize_before
        self.nhead = nhead

        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model, without_weight=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = (
            MultiheadAttentionLocal(d_model * 2, nhead, dropout=dropout, vdim=d_model, without_weight=True)
            if use_local_attn
            else MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model, without_weight=True)
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
        query_sine_embed: Tensor | None = None,
        is_first: bool = False,
        memory_key_padding_mask: Tensor | None = None,
        key_batch_cnt: Tensor | None = None,
        index_pair: Tensor | None = None,
        index_pair_batch: Tensor | None = None,
        memory_valid_mask: Tensor | None = None,
    ) -> Tensor:
        num_queries, bs, n_model = tgt.shape
        if not self.rm_self_attn_decoder:
            assert query_pos is not None
            q_content = self.sa_qcontent_proj(tgt)
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)
            tgt2 = self.self_attn(q_content + q_pos, k_content + k_pos, value=v, attn_mask=tgt_mask)[0]
            tgt = self.norm1(tgt + self.dropout1(tgt2))

        assert query_pos is not None and query_sine_embed is not None and pos is not None
        if self.use_local_attn:
            assert key_batch_cnt is not None and index_pair is not None and index_pair_batch is not None
            query_batch_cnt = torch.zeros_like(key_batch_cnt)
            query_batch_cnt.fill_(num_queries)
            query_pos = query_pos.permute(1, 0, 2).contiguous().view(-1, n_model)
            query_sine_embed = query_sine_embed.permute(1, 0, 2).contiguous().view(-1, n_model)
            tgt = tgt.permute(1, 0, 2).contiguous().view(-1, n_model)

        q_content = self.ca_qcontent_proj(tgt)
        if self.use_local_attn and memory_valid_mask is not None:
            valid_memory = memory[memory_valid_mask]
            k_content_valid = self.ca_kcontent_proj(valid_memory)
            k_content = memory.new_zeros(memory.shape[0], k_content_valid.shape[-1])
            k_content[memory_valid_mask] = k_content_valid.to(dtype=k_content.dtype)
            v_valid = self.ca_v_proj(valid_memory)
            v = memory.new_zeros(memory.shape[0], v_valid.shape[-1])
            v[memory_valid_mask] = v_valid.to(dtype=v.dtype)
            valid_pos = pos[memory_valid_mask]
            k_pos_valid = self.ca_kpos_proj(valid_pos)
            k_pos = pos.new_zeros(memory.shape[0], k_pos_valid.shape[-1])
            k_pos[memory_valid_mask] = k_pos_valid.to(dtype=k_pos.dtype)
        else:
            k_content = self.ca_kcontent_proj(memory)
            v = self.ca_v_proj(memory)
            k_pos = self.ca_kpos_proj(pos)

        if is_first or self.keep_query_pos:
            q = q_content + self.ca_qpos_proj(query_pos)
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        if self.use_local_attn:
            q = q.view(q.shape[0], self.nhead, n_model // self.nhead)
            query_sine_embed = query_sine_embed.view(q.shape[0], self.nhead, n_model // self.nhead)
            q = torch.cat([q, query_sine_embed], dim=-1).view(q.shape[0], n_model * 2)
            k = k.view(k.shape[0], self.nhead, n_model // self.nhead)
            k_pos = k_pos.view(k.shape[0], self.nhead, n_model // self.nhead)
            k = torch.cat([k, k_pos], dim=-1).view(k.shape[0], n_model * 2)
            tgt2 = self.cross_attn(
                query=q,
                key=k,
                value=v,
                index_pair=index_pair,
                query_batch_cnt=query_batch_cnt,
                key_batch_cnt=key_batch_cnt,
                index_pair_batch=index_pair_batch,
                vdim=n_model,
                local_indices=True,
            )[0]
        else:
            q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
            query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
            q = torch.cat([q, query_sine_embed], dim=-1).view(num_queries, bs, n_model * 2)
            k = k.view(k.shape[0], bs, self.nhead, n_model // self.nhead)
            k_pos = k_pos.view(k.shape[0], bs, self.nhead, n_model // self.nhead)
            k = torch.cat([k, k_pos], dim=-1).view(k.shape[0], bs, n_model * 2)
            tgt2 = self.cross_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]

        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class MTRDecoder(nn.Module):
    """Motion decoder matching official MTR names, with the local CUDA op replaced."""

    def __init__(
        self,
        in_channels: int = 256,
        d_model: int = 512,
        map_d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_future_frames: int = 80,
        num_motion_modes: int = 6,
        num_intention_queries: int = 64,
        intention_points_file: str = "data/intention_points.pkl",
        nms_dist_thresh: float = 2.5,
        center_offset_of_map: tuple[float, float] = (30.0, 0.0),
        num_base_map_polylines: int = 256,
        num_waypoint_map_polylines: int = 128,
        keep_query_pos_all: bool = False,
    ) -> None:
        super().__init__()
        self.object_type = ["TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"]
        self.num_future_frames = num_future_frames
        self.num_motion_modes = num_motion_modes
        self.use_place_holder = False
        self.d_model = d_model
        self.num_decoder_layers = num_layers
        self.num_queries = num_intention_queries
        self.nms_dist_thresh = nms_dist_thresh
        self.center_offset_of_map = center_offset_of_map
        self.num_base_map_polylines = num_base_map_polylines
        self.num_waypoint_map_polylines = num_waypoint_map_polylines
        self.keep_query_pos_all = keep_query_pos_all

        self.in_proj_center_obj = nn.Sequential(nn.Linear(in_channels, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.in_proj_obj, self.obj_decoder_layers = self.build_transformer_decoder(
            in_channels, d_model, num_heads, dropout, num_layers, use_local_attn=False, keep_query_pos=keep_query_pos_all
        )
        self.in_proj_map, self.map_decoder_layers = self.build_transformer_decoder(
            in_channels, map_d_model, num_heads, dropout, num_layers, use_local_attn=True, keep_query_pos=keep_query_pos_all
        )
        if map_d_model != d_model:
            temp_layer = nn.Linear(d_model, map_d_model)
            self.map_query_content_mlps = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(num_layers)])
            self.map_query_embed_mlps = nn.Linear(d_model, map_d_model)
        else:
            self.map_query_content_mlps = None
            self.map_query_embed_mlps = None

        self.build_dense_future_prediction_layers(d_model, num_future_frames)
        self.intention_points, self.intention_query, self.intention_query_mlps = self.build_motion_query(
            d_model, intention_points_file
        )
        fuse = build_mlps(d_model * 2 + map_d_model, [d_model, d_model], ret_before_act=True)
        self.query_feature_fusion_layers = nn.ModuleList([copy.deepcopy(fuse) for _ in range(num_layers)])
        self.motion_reg_heads, self.motion_cls_heads, self.motion_vel_heads = self.build_motion_head(
            d_model, d_model, num_layers
        )
        self._freeze_unused_query_pos_projections()
        self.forward_ret_dict: dict[str, Tensor | list] = {}

    def _freeze_unused_query_pos_projections(self) -> None:
        """Avoid DDP unused-parameter errors for official inactive layers.

        Official MTR constructs ``ca_qpos_proj`` in every decoder layer, but
        with ``keep_query_pos=False`` only layer 0 uses it. Later-layer weights
        remain in checkpoints for key compatibility, while their forward path is
        intentionally inactive.
        """
        if self.keep_query_pos_all:
            return
        for layers in (self.obj_decoder_layers, self.map_decoder_layers):
            for layer in layers[1:]:
                for param in layer.ca_qpos_proj.parameters():
                    param.requires_grad_(False)

    def build_dense_future_prediction_layers(self, hidden_dim: int, num_future_frames: int) -> None:
        self.obj_pos_encoding_layer = build_mlps(2, [hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, use_norm=False)
        self.dense_future_head = build_mlps(hidden_dim * 2, [hidden_dim, hidden_dim, num_future_frames * 7], ret_before_act=True)
        self.future_traj_mlps = build_mlps(4 * num_future_frames, [hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, use_norm=False)
        self.traj_fusion_mlps = build_mlps(hidden_dim * 2, [hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, use_norm=False)

    @staticmethod
    def build_transformer_decoder(
        in_channels: int,
        d_model: int,
        nhead: int,
        dropout: float,
        num_decoder_layers: int,
        use_local_attn: bool,
        keep_query_pos: bool = False,
    ) -> tuple[nn.Sequential, nn.ModuleList]:
        in_proj = nn.Sequential(nn.Linear(in_channels, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            normalize_before=False,
            keep_query_pos=keep_query_pos,
            rm_self_attn_decoder=False,
            use_local_attn=use_local_attn,
        )
        return in_proj, nn.ModuleList([copy.deepcopy(layer) for _ in range(num_decoder_layers)])

    def build_motion_query(self, d_model: int, intention_points_file: str) -> tuple[dict[str, Tensor], None, nn.Sequential]:
        with open(intention_points_file, "rb") as f:
            pts_dict = pickle.load(f)
        points = {}
        for cur_type in self.object_type:
            points[cur_type] = torch.from_numpy(pts_dict[cur_type]).float().view(-1, 2)
        return points, None, build_mlps(d_model, [d_model, d_model], ret_before_act=True)

    def build_motion_head(
        self,
        in_channels: int,
        hidden_size: int,
        num_decoder_layers: int,
    ) -> tuple[nn.ModuleList, nn.ModuleList, None]:
        reg_head = build_mlps(in_channels, [hidden_size, hidden_size, self.num_future_frames * 7], ret_before_act=True)
        cls_head = build_mlps(in_channels, [hidden_size, hidden_size, 1], ret_before_act=True)
        return (
            nn.ModuleList([copy.deepcopy(reg_head) for _ in range(num_decoder_layers)]),
            nn.ModuleList([copy.deepcopy(cls_head) for _ in range(num_decoder_layers)]),
            None,
        )

    def get_motion_query(self, center_objects_type: Tensor | list[str]) -> tuple[Tensor, Tensor]:
        if isinstance(center_objects_type, Tensor):
            type_names = [_TYPE_MAP.get(int(t.item()), "TYPE_VEHICLE") for t in center_objects_type]
        else:
            type_names = [
                _TYPE_MAP.get(int(t), "TYPE_VEHICLE") if not isinstance(t, str) else t
                for t in center_objects_type
            ]
        intention_points = torch.stack(
            [self.intention_points[t].to(next(self.parameters()).device)[: self.num_queries] for t in type_names],
            dim=0,
        ).permute(1, 0, 2)
        intention_query = gen_sineembed_for_position(intention_points, hidden_dim=self.d_model)
        intention_query = self.intention_query_mlps(intention_query.reshape(-1, self.d_model)).view(
            -1, len(type_names), self.d_model
        )
        return intention_query, intention_points

    def apply_dense_future_prediction(self, obj_feature: Tensor, obj_mask: Tensor, obj_pos: Tensor) -> tuple[Tensor, Tensor]:
        num_center_objects, num_objects, _ = obj_feature.shape
        valid = obj_mask.bool()
        obj_pos_valid = obj_pos[valid][..., 0:2]
        obj_feature_valid = obj_feature[valid]
        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)
        pred_dense = self.dense_future_head(torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1))
        pred_dense = pred_dense.view(pred_dense.shape[0], self.num_future_frames, 7)
        pred_dense = torch.cat((pred_dense[:, :, 0:2] + obj_pos_valid[:, None, 0:2], pred_dense[:, :, 2:]), dim=-1)
        future_input = pred_dense[:, :, [0, 1, -2, -1]].flatten(1, 2)
        future_feature = self.future_traj_mlps(future_input)
        obj_feature_valid = self.traj_fusion_mlps(torch.cat((obj_feature_valid, future_feature), dim=-1))
        ret_feature = torch.zeros_like(obj_feature)
        ret_feature[valid] = obj_feature_valid.to(dtype=ret_feature.dtype)
        ret_dense = obj_feature.new_zeros(num_center_objects, num_objects, self.num_future_frames, 7)
        ret_dense[valid] = pred_dense.to(dtype=ret_dense.dtype)
        self.forward_ret_dict["pred_dense_trajs"] = ret_dense
        return ret_feature, ret_dense

    def apply_cross_attention(
        self,
        kv_feature: Tensor,
        kv_mask: Tensor,
        kv_pos: Tensor,
        query_content: Tensor,
        query_embed: Tensor,
        attention_layer: TransformerDecoderLayer,
        dynamic_query_center: Tensor,
        layer_idx: int = 0,
        use_local_attn: bool = False,
        query_index_pair: Tensor | None = None,
        query_content_pre_mlp: nn.Module | None = None,
        query_embed_pre_mlp: nn.Module | None = None,
    ) -> Tensor:
        if query_content_pre_mlp is not None:
            query_content = query_content_pre_mlp(query_content)
        if query_embed_pre_mlp is not None:
            query_embed = query_embed_pre_mlp(query_embed)
        num_q, batch_size, d_model = query_content.shape
        searching_query = gen_sineembed_for_position(dynamic_query_center, hidden_dim=d_model)
        kv_pos_embed = gen_sineembed_for_position(kv_pos.permute(1, 0, 2)[..., 0:2], hidden_dim=d_model)
        if not use_local_attn:
            return attention_layer(
                tgt=query_content,
                query_pos=query_embed,
                query_sine_embed=searching_query,
                memory=kv_feature.permute(1, 0, 2),
                memory_key_padding_mask=~kv_mask,
                pos=kv_pos_embed,
                is_first=(layer_idx == 0),
            )

        assert query_index_pair is not None
        batch_size, num_kv, _ = kv_feature.shape
        kv_feature_stack = kv_feature.flatten(0, 1)
        kv_pos_embed_stack = kv_pos_embed.permute(1, 0, 2).contiguous().flatten(0, 1)
        kv_mask_stack = kv_mask.reshape(-1)
        key_batch_cnt = num_kv * torch.ones(batch_size, dtype=torch.int32, device=kv_feature.device)
        query_index_pair = query_index_pair.view(batch_size * num_q, -1).long()
        index_pair_batch = torch.arange(batch_size, dtype=torch.int32, device=kv_feature.device)[:, None].repeat(1, num_q).view(-1)
        query_feature = attention_layer(
            tgt=query_content,
            query_pos=query_embed,
            query_sine_embed=searching_query,
            memory=kv_feature_stack,
            memory_valid_mask=kv_mask_stack,
            pos=kv_pos_embed_stack,
            is_first=(layer_idx == 0),
            key_batch_cnt=key_batch_cnt,
            index_pair=query_index_pair,
            index_pair_batch=index_pair_batch,
        )
        return query_feature.view(batch_size, num_q, d_model).permute(1, 0, 2)

    def apply_dynamic_map_collection(
        self,
        map_pos: Tensor,
        map_mask: Tensor,
        pred_waypoints: Tensor,
        base_region_offset: tuple[float, float],
        num_query: int,
        num_waypoint_polylines: int = 128,
        num_base_polylines: int = 256,
        base_map_idxs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        map_pos = map_pos.clone()
        map_pos[~map_mask] = 10000000.0
        num_polylines = map_pos.shape[1]
        if base_map_idxs is None:
            base_points = torch.tensor(base_region_offset, dtype=map_pos.dtype, device=map_pos.device)
            base_dist = (map_pos[:, :, 0:2] - base_points[None, None]).norm(dim=-1)
            base_topk_dist, base_map_idxs = base_dist.topk(k=min(num_polylines, num_base_polylines), dim=-1, largest=False)
            base_map_idxs[base_topk_dist > 10000000] = -1
            base_map_idxs = base_map_idxs[:, None, :].repeat(1, num_query, 1)
            if base_map_idxs.shape[-1] < num_base_polylines:
                base_map_idxs = F.pad(base_map_idxs, (0, num_base_polylines - base_map_idxs.shape[-1]), value=-1)

        dynamic_dist = (pred_waypoints[:, :, None, :, 0:2] - map_pos[:, None, :, None, 0:2]).norm(dim=-1).min(dim=-1)[0]
        dynamic_topk_dist, dynamic_map_idxs = dynamic_dist.topk(k=min(num_polylines, num_waypoint_polylines), dim=-1, largest=False)
        dynamic_map_idxs[dynamic_topk_dist > 10000000] = -1
        if dynamic_map_idxs.shape[-1] < num_waypoint_polylines:
            dynamic_map_idxs = F.pad(dynamic_map_idxs, (0, num_waypoint_polylines - dynamic_map_idxs.shape[-1]), value=-1)
        collected_idxs = torch.cat((base_map_idxs, dynamic_map_idxs), dim=-1)

        sorted_idxs = collected_idxs.sort(dim=-1)[0]
        duplicate_slice = sorted_idxs[..., 1:] - sorted_idxs[..., :-1] != 0
        duplicate_mask = torch.ones_like(collected_idxs, dtype=torch.bool)
        duplicate_mask[..., 1:] = duplicate_slice
        sorted_idxs[~duplicate_mask] = -1
        return sorted_idxs.int(), base_map_idxs

    def apply_transformer_decoder(
        self,
        center_objects_feature: Tensor,
        center_objects_type: Tensor | list[str],
        obj_feature: Tensor,
        obj_mask: Tensor,
        obj_pos: Tensor,
        map_feature: Tensor,
        map_mask: Tensor,
        map_pos: Tensor,
    ) -> list[list[Tensor]]:
        intention_query, intention_points = self.get_motion_query(center_objects_type)
        query_content = torch.zeros_like(intention_query)
        self.forward_ret_dict["intention_points"] = intention_points.permute(1, 0, 2)
        num_center_objects = query_content.shape[1]
        num_query = query_content.shape[0]
        center_objects_feature = center_objects_feature[None].repeat(num_query, 1, 1)

        base_map_idxs = None
        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]
        dynamic_query_center = intention_points
        pred_list: list[list[Tensor]] = []
        for layer_idx in range(self.num_decoder_layers):
            obj_query_feature = self.apply_cross_attention(
                obj_feature,
                obj_mask,
                obj_pos,
                query_content,
                intention_query,
                self.obj_decoder_layers[layer_idx],
                dynamic_query_center,
                layer_idx=layer_idx,
            )
            collected_idxs, base_map_idxs = self.apply_dynamic_map_collection(
                map_pos,
                map_mask,
                pred_waypoints,
                base_region_offset=self.center_offset_of_map,
                num_waypoint_polylines=self.num_waypoint_map_polylines,
                num_base_polylines=self.num_base_map_polylines,
                base_map_idxs=base_map_idxs,
                num_query=num_query,
            )
            map_query_feature = self.apply_cross_attention(
                map_feature,
                map_mask,
                map_pos,
                query_content,
                intention_query,
                self.map_decoder_layers[layer_idx],
                dynamic_query_center,
                layer_idx=layer_idx,
                use_local_attn=True,
                query_index_pair=collected_idxs,
                query_content_pre_mlp=self.map_query_content_mlps[layer_idx] if self.map_query_content_mlps is not None else None,
                query_embed_pre_mlp=self.map_query_embed_mlps,
            )
            query_feature = torch.cat([center_objects_feature, obj_query_feature, map_query_feature], dim=-1)
            query_content = self.query_feature_fusion_layers[layer_idx](query_feature.flatten(0, 1)).view(
                num_query, num_center_objects, -1
            )
            query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
            pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(
                num_center_objects, num_query, self.num_future_frames, 7
            )
            pred_list.append([pred_scores, pred_trajs])
            pred_waypoints = pred_trajs[:, :, :, 0:2]
            dynamic_query_center = pred_trajs[:, :, -1, 0:2].contiguous().permute(1, 0, 2)
        return pred_list

    @torch.no_grad()
    def batch_nms(self, pred_trajs: Tensor, pred_scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_modes, num_timestamps, feat_dim = pred_trajs.shape
        sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
        bs_full = torch.arange(batch_size, device=pred_trajs.device)[:, None].repeat(1, num_modes)
        sorted_scores = pred_scores[bs_full, sorted_idxs]
        sorted_trajs = pred_trajs[bs_full, sorted_idxs]
        goals = sorted_trajs[:, :, -1]
        dist = (goals[:, :, None, 0:2] - goals[:, None, :, 0:2]).norm(dim=-1)
        cover = dist < self.nms_dist_thresh
        point_val = sorted_scores.clone()
        selected_val = torch.zeros_like(point_val)
        ret_idxs = sorted_idxs.new_zeros(batch_size, self.num_motion_modes).long()
        ret_trajs = pred_trajs.new_zeros(batch_size, self.num_motion_modes, num_timestamps, feat_dim)
        ret_scores = pred_scores.new_zeros(batch_size, self.num_motion_modes)
        bs_idx = torch.arange(batch_size, device=pred_trajs.device)
        for k in range(self.num_motion_modes):
            cur_idx = point_val.argmax(dim=-1)
            ret_idxs[:, k] = cur_idx
            point_val = point_val * (~cover[bs_idx, cur_idx]).float()
            selected_val[bs_idx, cur_idx] = -1
            point_val += selected_val
            ret_trajs[:, k] = sorted_trajs[bs_idx, cur_idx]
            ret_scores[:, k] = sorted_scores[bs_idx, cur_idx]
        ret_idxs = sorted_idxs[bs_idx[:, None].repeat(1, self.num_motion_modes), ret_idxs]
        return ret_trajs, ret_scores, ret_idxs

    def generate_final_prediction(self, pred_list: list[list[Tensor]]) -> tuple[Tensor, Tensor]:
        pred_scores, pred_trajs = pred_list[-1]
        pred_scores = torch.softmax(pred_scores, dim=-1)
        if self.num_motion_modes != pred_scores.shape[1]:
            pred_trajs, pred_scores, _ = self.batch_nms(pred_trajs, pred_scores)
        return pred_scores, pred_trajs

    def forward(self, batch_or_enc: dict[str, Tensor], batch: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        self.forward_ret_dict = {}
        if batch is None:
            batch_dict = batch_or_enc
            input_dict = batch_dict.get("input_dict", batch_dict)
        else:
            input_dict = batch
            batch_dict = {"input_dict": input_dict, **batch_or_enc}

        obj_feature = batch_dict["obj_feature"]
        obj_mask = batch_dict["obj_mask"].bool()
        obj_pos = batch_dict["obj_pos"]
        map_feature = batch_dict["map_feature"]
        map_mask = batch_dict["map_mask"].bool()
        map_pos = batch_dict["map_pos"]
        center_objects_feature = batch_dict["center_objects_feature"]
        if center_objects_feature.ndim == 3 and center_objects_feature.shape[1] == 1:
            center_objects_feature = center_objects_feature.squeeze(1)
        num_center_objects, num_objects, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]

        center_objects_feature = self.in_proj_center_obj(center_objects_feature)
        obj_proj_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_proj = obj_feature.new_zeros(num_center_objects, num_objects, obj_proj_valid.shape[-1])
        obj_proj[obj_mask] = obj_proj_valid

        map_proj_valid = self.in_proj_map(map_feature[map_mask])
        map_proj = map_feature.new_zeros(num_center_objects, num_polylines, map_proj_valid.shape[-1])
        map_proj[map_mask] = map_proj_valid

        obj_proj, pred_dense = self.apply_dense_future_prediction(obj_proj, obj_mask, obj_pos)
        center_type = input_dict.get("center_objects_type", input_dict.get("center_obj_type"))
        pred_list = self.apply_transformer_decoder(
            center_objects_feature,
            center_type,
            obj_proj,
            obj_mask,
            obj_pos,
            map_proj,
            map_mask,
            map_pos,
        )
        self.forward_ret_dict["pred_list"] = pred_list
        pred_scores, pred_trajs = self.generate_final_prediction(pred_list)
        return {
            "pred_scores": pred_scores if not self.training else pred_list[-1][0],
            "pred_trajs": pred_trajs if not self.training else pred_list[-1][1],
            "pred_list": [(s, t) for s, t in pred_list],
            "pred_dense_trajs": pred_dense,
            "intention_points": self.forward_ret_dict["intention_points"],
        }
