"""MTR decoder: intention-query based multi-modal trajectory prediction.

Matches the official MTR architecture:
- Concatenative PE cross-attention (Q/K doubled to 2*d_model)
- Separate obj and map cross-attention with separate query results
- Dynamic map collection with local cross-attention
- Per-layer prediction heads with waypoint update
"""

from __future__ import annotations

import copy
import math
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from anon_tokyo.nn.layers import build_mlps
from anon_tokyo.prediction.mtr.encoder import gen_sineembed_for_position


# ── Single decoder layer (cross-attn only, matching official TransformerDecoderLayer) ────


class MTRDecoderLayer(nn.Module):
    """Cross-attention decoder layer with additive PE.

    Q = content + sine_embed (+ pos on first layer)
    K = content + k_pos
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm_sa = nn.LayerNorm(d_model)
        self.dropout_sa = nn.Dropout(dropout)

        # Cross-attention projection layers
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        # Standard MHA cross-attention (d_model, no dim doubling)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=False,
        )
        self.norm_ca = nn.LayerNorm(d_model)
        self.dropout_ca = nn.Dropout(dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,  # [Q, K_total, D]
        query_pos: Tensor,  # [Q, K_total, D]
        query_sine_embed: Tensor,  # [Q, K_total, D] — sine embed from dynamic center
        memory: Tensor,  # [S, K_total, D]
        pos: Tensor,  # [S, K_total, D] — memory positional encoding
        memory_key_padding_mask: Tensor | None = None,  # [K_total, S]
        query_pos_projected: Tensor | None = None,  # [Q, K_total, D] pre-projected (first layer only)
    ) -> Tensor:
        # ── Self-attention ──
        q_sa = k_sa = tgt + query_pos
        sa_out = self.self_attn(q_sa, k_sa, tgt)[0]
        tgt = tgt + self.dropout_sa(sa_out)
        tgt = self.norm_sa(tgt)

        # ── Cross-attention with additive PE ──
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        k_pos = self.ca_kpos_proj(pos)

        if query_pos_projected is not None:
            q = q_content + query_pos_projected
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        q = q + query_sine_embed
        k = k + k_pos

        ca_out = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=memory_key_padding_mask,
        )[0]

        tgt = tgt + self.dropout_ca(ca_out)
        tgt = self.norm_ca(tgt)

        # ── FFN ──
        ffn_out = self.linear2(self.dropout1(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(ffn_out)
        tgt = self.norm_ffn(tgt)
        return tgt


# ── Full decoder ──────────────────────────────────────────────────────────────

# obj_type int → pkl key / buffer suffix
_TYPE_MAP = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
_PKL_KEY = {1: "TYPE_VEHICLE", 2: "TYPE_PEDESTRIAN", 3: "TYPE_CYCLIST"}


class MTRDecoder(nn.Module):
    """Intention-query decoder for multi-modal prediction (official MTR layout)."""

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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.map_d_model = map_d_model
        self.num_layers = num_layers
        self.num_future_frames = num_future_frames
        self.num_motion_modes = num_motion_modes
        self.num_queries = num_intention_queries
        self.nms_dist_thresh = nms_dist_thresh

        # Input projections (encoder d_model → decoder d_model / map_d_model)
        self.in_proj_center = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.in_proj_obj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.in_proj_map = nn.Sequential(
            nn.Linear(in_channels, map_d_model),
            nn.ReLU(),
            nn.Linear(map_d_model, map_d_model),
        )

        # Separate obj and map decoder layers (matching official)
        self.obj_decoder_layers = nn.ModuleList(
            [MTRDecoderLayer(d_model, num_heads, d_model * 4, dropout) for _ in range(num_layers)]
        )
        self.map_decoder_layers = nn.ModuleList(
            [MTRDecoderLayer(map_d_model, num_heads, map_d_model * 4, dropout) for _ in range(num_layers)]
        )

        # First-layer query position projections (only used at layer 0)
        self.obj_ca_qpos_proj = nn.Linear(d_model, d_model)
        self.map_ca_qpos_proj = nn.Linear(map_d_model, map_d_model)

        # Map query projection MLPs (d_model → map_d_model) when dimensions differ
        if map_d_model != d_model:
            self.map_query_content_mlps = nn.ModuleList([nn.Linear(d_model, map_d_model) for _ in range(num_layers)])
            self.map_query_embed_mlps: nn.Linear | None = nn.Linear(d_model, map_d_model)
        else:
            self.map_query_content_mlps = None
            self.map_query_embed_mlps = None

        # Intention query (loaded from k-means cluster centres)
        self.intention_query_mlp = build_mlps(d_model, [d_model, d_model], ret_before_act=True)
        self._register_intention_points(intention_points_file)

        # Dense future prediction (auxiliary task on all agents)
        self.obj_pos_enc = build_mlps(2, [d_model, d_model, d_model], ret_before_act=True, use_norm=False)
        self.dense_future_head = build_mlps(d_model * 2, [d_model, d_model, num_future_frames * 7], ret_before_act=True)
        self.future_traj_mlps = build_mlps(
            4 * num_future_frames, [d_model, d_model, d_model], ret_before_act=True, use_norm=False
        )
        self.traj_fusion_mlps = build_mlps(
            d_model * 2, [d_model, d_model, d_model], ret_before_act=True, use_norm=False
        )

        # Feature fusion per decoder layer: center(D) + obj_query(D) + map_query(map_D)
        layer_fuse = build_mlps(d_model * 2 + map_d_model, [d_model, d_model], ret_before_act=True)
        self.query_feature_fusion = nn.ModuleList([copy.deepcopy(layer_fuse) for _ in range(num_layers)])

        # Per-layer prediction heads
        reg_head = build_mlps(d_model, [d_model, d_model, num_future_frames * 7], ret_before_act=True)
        cls_head = build_mlps(d_model, [d_model, d_model, 1], ret_before_act=True)
        self.reg_heads = nn.ModuleList([copy.deepcopy(reg_head) for _ in range(num_layers)])
        self.cls_heads = nn.ModuleList([copy.deepcopy(cls_head) for _ in range(num_layers)])

    def _register_intention_points(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            pts_dict = pickle.load(f)
        for type_id, buf_name in _TYPE_MAP.items():
            pkl_key = _PKL_KEY[type_id]
            arr = torch.from_numpy(pts_dict[pkl_key]).float().view(-1, 2)
            self.register_buffer(f"intention_pts_{buf_name}", arr, persistent=False)

    def _get_intention_points(self, center_obj_type: Tensor) -> Tensor:
        """Gather intention points per agent type.

        Args:
            center_obj_type: ``[K_total]`` int — 1=vehicle, 2=ped, 3=cyclist.

        Returns:
            ``[K_total, Q, 2]``
        """
        bufs = {tid: getattr(self, f"intention_pts_{name}") for tid, name in _TYPE_MAP.items()}
        K = center_obj_type.shape[0]
        Q = self.num_queries
        out = center_obj_type.new_zeros(K, Q, 2, dtype=torch.float32)
        for type_id, buf in bufs.items():
            mask = center_obj_type == type_id
            if mask.any():
                out[mask] = buf[:Q].unsqueeze(0).expand(mask.sum(), -1, -1)
        return out

    # ── Dense future prediction ──────────────────────────────────────────

    def apply_dense_future_prediction(
        self,
        obj_feat: Tensor,
        obj_mask: Tensor,
        obj_pos: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict future for all agents, fuse back into obj features.

        Args:
            obj_feat: ``[K_total, A, D]``
            obj_mask: ``[K_total, A]``  bool
            obj_pos:  ``[K_total, A, 2]``

        Returns:
            updated obj_feat, dense_pred ``[K_total, A, T_f, 7]``
        """
        K, A, D = obj_feat.shape
        valid = obj_mask.bool()

        pos_valid = obj_pos[valid]  # [V, 2]
        feat_valid = obj_feat[valid]  # [V, D]
        pos_enc = self.obj_pos_enc(pos_valid)
        fused = torch.cat([pos_enc, feat_valid], dim=-1)
        dense_pred_valid = self.dense_future_head(fused).view(-1, self.num_future_frames, 7)

        # offset → absolute position
        dense_pred_valid[:, :, 0:2] = dense_pred_valid[:, :, 0:2] + pos_valid[:, None, :]

        # future trajectory feature → fuse with obj
        fut_input = dense_pred_valid[:, :, [0, 1, -2, -1]].flatten(1, 2)
        fut_feat = self.future_traj_mlps(fut_input)
        fused_feat = self.traj_fusion_mlps(torch.cat([feat_valid, fut_feat], dim=-1))

        out_feat = torch.zeros_like(obj_feat)
        out_feat[valid] = fused_feat

        dense_pred = obj_feat.new_zeros(K, A, self.num_future_frames, 7, dtype=dense_pred_valid.dtype)
        dense_pred[valid] = dense_pred_valid

        return out_feat, dense_pred

    # ── Dynamic map collection ───────────────────────────────────────────

    @torch.no_grad()
    def _collect_map_indices(
        self,
        map_pos: Tensor,
        map_mask: Tensor,
        pred_waypoints: Tensor,
        num_base: int = 256,
        num_dynamic: int = 128,
    ) -> Tensor:
        """Collect nearest map polyline indices per query.

        Args:
            map_pos:  ``[K_total, M, 2]``
            map_mask: ``[K_total, M]`` bool
            pred_waypoints: ``[K_total, Q, num_wp, 2]``

        Returns:
            ``[K_total, Q, num_base + num_dynamic]`` int indices (−1 = invalid)
        """
        mp = map_pos.clone()
        mp[~map_mask] = 1e7
        M = mp.shape[1]

        # Base polylines: closest to origin
        base_dist = mp.norm(dim=-1)  # [K, M]
        n_base = min(M, num_base)
        _, base_idx = base_dist.topk(n_base, dim=-1, largest=False)  # [K, n_base]
        base_idx = base_idx.unsqueeze(1).expand(-1, self.num_queries, -1)  # [K, Q, n_base]

        # Dynamic polylines: closest to predicted waypoints
        # pred_waypoints: [K, Q, nw, 2], mp: [K, M, 2]
        dist = (pred_waypoints[:, :, None, :, :] - mp[:, None, :, None, :]).norm(dim=-1)  # [K, Q, M, nw]
        dist = dist.min(dim=-1)[0]  # [K, Q, M]
        n_dyn = min(M, num_dynamic)
        _, dyn_idx = dist.topk(n_dyn, dim=-1, largest=False)  # [K, Q, n_dyn]

        # Pad to fixed sizes
        if n_base < num_base:
            base_idx = F.pad(base_idx, (0, num_base - n_base), value=-1)
        if n_dyn < num_dynamic:
            dyn_idx = F.pad(dyn_idx, (0, num_dynamic - n_dyn), value=-1)

        return torch.cat([base_idx, dyn_idx], dim=-1)

    def _gather_map_kv(
        self,
        map_feat: Tensor,
        map_pos: Tensor,
        map_mask: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gather map features/positions by collected indices.

        All queries share the union of collected indices (max-pool over Q dim).

        Returns:
            map_kv ``[K, M', D]``, map_kv_pos ``[K, M', 2]``, pad_mask ``[K, M']``
        """
        # Union over queries: take unique indices per sample
        K, Q, N_coll = indices.shape
        # flatten query dim → take union
        flat = indices.view(K, -1)  # [K, Q*N_coll]
        # deduplicate & get top-N unique per sample (keep all, just sort and mask dups)
        sorted_idx, _ = flat.sort(dim=-1)
        unique_mask = torch.ones_like(sorted_idx, dtype=torch.bool)
        unique_mask[:, 1:] = sorted_idx[:, 1:] != sorted_idx[:, :-1]
        sorted_idx[~unique_mask] = -1
        # keep top M' valid
        valid_count = unique_mask.sum(dim=-1)  # [K]
        M_prime = min(int(valid_count.max().item()), flat.shape[1])

        # re-sort to push -1 to end
        sorted_idx[sorted_idx < 0] = map_feat.shape[1]  # sentinel
        top_idx, _ = sorted_idx.sort(dim=-1)
        top_idx = top_idx[:, :M_prime]
        pad_mask = top_idx >= map_feat.shape[1]
        top_idx = top_idx.clamp(max=map_feat.shape[1] - 1)

        b_idx = torch.arange(K, device=map_feat.device)[:, None].expand(K, M_prime)
        gathered_feat = map_feat[b_idx, top_idx]
        gathered_pos = map_pos[b_idx, top_idx]
        return gathered_feat, gathered_pos, pad_mask

    # ── NMS ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _batch_nms(
        self,
        pred_trajs: Tensor,
        pred_scores: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Distance-based NMS: 64 modes → num_motion_modes.

        Args:
            pred_trajs:  ``[K_total, Q, T, 7]``
            pred_scores: ``[K_total, Q]``

        Returns:
            trajs ``[K_total, num_motion_modes, T, 7]``,
            scores ``[K_total, num_motion_modes]``
        """
        K, Q, T, C = pred_trajs.shape
        N = self.num_motion_modes
        scores = pred_scores.softmax(dim=-1)

        out_trajs = pred_trajs.new_zeros(K, N, T, C)
        out_scores = scores.new_zeros(K, N)

        for k in range(K):
            order = scores[k].argsort(descending=True)
            selected: list[int] = []
            for idx in order.tolist():
                if len(selected) >= N:
                    break
                endpoint = pred_trajs[k, idx, -1, 0:2]
                too_close = False
                for s in selected:
                    if (pred_trajs[k, s, -1, 0:2] - endpoint).norm() < self.nms_dist_thresh:
                        too_close = True
                        break
                if not too_close:
                    selected.append(idx)
            # pad if fewer selected
            while len(selected) < N:
                for idx in order.tolist():
                    if idx not in selected:
                        selected.append(idx)
                        break
                else:
                    break
            sel = torch.tensor(selected[:N], device=pred_trajs.device)
            out_trajs[k] = pred_trajs[k, sel]
            out_scores[k] = scores[k, sel]

        # renormalise
        out_scores = out_scores / out_scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return out_trajs, out_scores

    # ── Forward ───────────────────────────────────────────────────────────

    def _apply_cross_attention(
        self,
        kv_feature: Tensor,
        kv_mask: Tensor,
        kv_pos: Tensor,
        query_content: Tensor,
        query_embed: Tensor,
        attention_layer: MTRDecoderLayer,
        dynamic_query_center: Tensor,
        layer_idx: int,
        query_content_pre_mlp: nn.Module | None = None,
        query_embed_pre_mlp: nn.Module | None = None,
        qpos_proj: nn.Module | None = None,
    ) -> Tensor:
        """Apply a single cross-attention step (matching official apply_cross_attention).

        Args:
            kv_feature: [K, S, D_kv]
            kv_mask: [K, S]  bool (True = valid)
            kv_pos: [K, S, 2]
            query_content: [Q, K, D]
            query_embed: [Q, K, D]
            dynamic_query_center: [Q, K, 2]
            layer_idx: current layer index
            query_content_pre_mlp: optional projection to map_d_model
            query_embed_pre_mlp: optional projection to map_d_model

        Returns:
            query_feature: [Q, K, D_kv]
        """
        if query_content_pre_mlp is not None:
            query_content = query_content_pre_mlp(query_content)
        if query_embed_pre_mlp is not None:
            query_embed = query_embed_pre_mlp(query_embed)

        num_q, batch_size, d_model = query_content.shape

        # Sine embedding from dynamic query center
        searching_query = gen_sineembed_for_position(dynamic_query_center, hidden_dim=d_model)  # [Q, K, D]

        # KV positional encoding
        kv_pos_2d = kv_pos[..., 0:2]  # [K, S, 2]
        kv_pos_embed = gen_sineembed_for_position(kv_pos_2d.permute(1, 0, 2), hidden_dim=d_model)  # [S, K, D]

        query_pos_projected = None
        if layer_idx == 0 and qpos_proj is not None:
            query_pos_projected = qpos_proj(query_embed)

        query_feature = attention_layer(
            tgt=query_content,
            query_pos=query_embed,
            query_sine_embed=searching_query,
            memory=kv_feature.permute(1, 0, 2),  # [S, K, D_kv]
            memory_key_padding_mask=~kv_mask,  # [K, S]
            pos=kv_pos_embed,
            query_pos_projected=query_pos_projected,
        )  # [Q, K, D_kv]

        return query_feature

    def forward(self, enc_out: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Args:
            enc_out: From MTREncoder — keys:
                ``obj_feature``  [K_total, A, D_enc]
                ``map_feature``  [K_total, M, D_enc]
                ``center_objects_feature``  [K_total, D_enc]
                ``obj_mask``  [K_total, A]
                ``map_mask``  [K_total, M]
                ``obj_pos``   [K_total, A, 2]
                ``map_pos``   [K_total, M, 2]
            batch: Agent-centric batch — keys:
                ``center_obj_type``  [K_total]

        Returns:
            ``pred_trajs``:  ``[K_total, Q or N, T, 7]``
            ``pred_scores``: ``[K_total, Q or N]``
            ``pred_list``:   list of per-layer ``(scores, trajs)`` for training
            ``pred_dense_trajs``: ``[K_total, A, T, 7]`` for aux loss
        """
        obj_feat_raw = enc_out["obj_feature"]  # [K, A, D_enc]
        map_feat_raw = enc_out["map_feature"]  # [K, M, D_enc]
        center_feat = enc_out["center_objects_feature"]  # [K, D_enc]
        obj_mask = enc_out["obj_mask"]  # [K, A]
        map_mask = enc_out["map_mask"]  # [K, M]
        obj_pos = enc_out["obj_pos"]  # [K, A, 2]
        map_pos = enc_out["map_pos"]  # [K, M, 2]
        center_obj_type = batch["center_obj_type"]  # [K]
        K = center_feat.shape[0]

        # Input projection
        center_feat = self.in_proj_center(center_feat)  # [K, D]

        # Object projection (only on valid)
        obj_proj = self.in_proj_obj(obj_feat_raw[obj_mask])
        obj_feat = obj_feat_raw.new_zeros(K, obj_feat_raw.shape[1], self.d_model, dtype=obj_proj.dtype)
        obj_feat[obj_mask] = obj_proj

        # Map projection (only on valid) → map_d_model
        map_proj = self.in_proj_map(map_feat_raw[map_mask])
        map_feat = map_feat_raw.new_zeros(K, map_feat_raw.shape[1], self.map_d_model, dtype=map_proj.dtype)
        map_feat[map_mask] = map_proj

        # Dense future prediction (auxiliary)
        obj_feat, dense_pred = self.apply_dense_future_prediction(obj_feat, obj_mask, obj_pos)

        # Intention queries
        intention_pts = self._get_intention_points(center_obj_type)  # [K, Q, 2]
        Q = intention_pts.shape[1]
        intention_pe = gen_sineembed_for_position(intention_pts, self.d_model)  # [K, Q, D]
        intention_embed = self.intention_query_mlp(intention_pe.view(-1, self.d_model)).view(K, Q, self.d_model)

        # Transpose to (S, B, D) for nn.MultiheadAttention
        query_content = torch.zeros(Q, K, self.d_model, device=center_feat.device, dtype=center_feat.dtype)
        query_embed = intention_embed.permute(1, 0, 2)  # [Q, K, D]
        center_feat_exp = center_feat[None].expand(Q, -1, -1)  # [Q, K, D]

        # Initial waypoints for map collection → dynamic_query_center
        pred_waypoints = intention_pts.unsqueeze(2)  # [K, Q, 1, 2]
        dynamic_query_center = intention_pts.permute(1, 0, 2)  # [Q, K, 2]

        pred_list: list[tuple[Tensor, Tensor]] = []

        for i in range(self.num_layers):
            # ── Object cross-attention ──
            obj_query_feature = self._apply_cross_attention(
                kv_feature=obj_feat,
                kv_mask=obj_mask,
                kv_pos=obj_pos,
                query_content=query_content,
                query_embed=query_embed,
                attention_layer=self.obj_decoder_layers[i],
                dynamic_query_center=dynamic_query_center,
                layer_idx=i,
                qpos_proj=self.obj_ca_qpos_proj,
            )  # [Q, K, D]

            # ── Dynamic map collection ──
            coll_idx = self._collect_map_indices(map_pos, map_mask, pred_waypoints)
            map_kv, map_kv_p, map_pad = self._gather_map_kv(map_feat, map_pos, map_mask, coll_idx)
            # map_kv: [K, M', map_D], map_kv_p: [K, M', 2], map_pad: [K, M']

            # ── Map cross-attention ──
            map_query_feature = self._apply_cross_attention(
                kv_feature=map_kv,
                kv_mask=~map_pad,  # map_pad is True=padding, we need True=valid
                kv_pos=map_kv_p,
                query_content=query_content,
                query_embed=query_embed,
                attention_layer=self.map_decoder_layers[i],
                dynamic_query_center=dynamic_query_center,
                layer_idx=i,
                query_content_pre_mlp=self.map_query_content_mlps[i]
                if self.map_query_content_mlps is not None
                else None,
                query_embed_pre_mlp=self.map_query_embed_mlps,
                qpos_proj=self.map_ca_qpos_proj,
            )  # [Q, K, map_D]

            # ── Feature fusion: [center, obj_query, map_query] → D ──
            fused = torch.cat([center_feat_exp, obj_query_feature, map_query_feature], dim=-1)
            # [Q, K, D + D + map_D]
            query_content = self.query_feature_fusion[i](fused.reshape(Q * K, -1)).view(Q, K, self.d_model)

            # Prediction heads
            flat = query_content.permute(1, 0, 2).reshape(K * Q, -1)  # [K*Q, D]
            pred_scores = self.cls_heads[i](flat).view(K, Q)
            pred_trajs = self.reg_heads[i](flat).view(K, Q, self.num_future_frames, 7)

            pred_list.append((pred_scores, pred_trajs))

            # Update waypoints and dynamic query center for next layer
            pred_waypoints = pred_trajs[:, :, :, 0:2].detach()
            dynamic_query_center = pred_trajs[:, :, -1, 0:2].detach().permute(1, 0, 2)  # [Q, K, 2]

        # Final output
        result: dict[str, Tensor] = {
            "pred_list": pred_list,
            "pred_dense_trajs": dense_pred,
            "intention_points": intention_pts,
        }

        if not self.training:
            final_scores, final_trajs = pred_list[-1]
            if Q > self.num_motion_modes:
                final_trajs, final_scores = self._batch_nms(final_trajs, final_scores)
            else:
                final_scores = final_scores.softmax(dim=-1)
            result["pred_trajs"] = final_trajs
            result["pred_scores"] = final_scores
        else:
            result["pred_trajs"] = pred_list[-1][1]
            result["pred_scores"] = pred_list[-1][0]

        return result
