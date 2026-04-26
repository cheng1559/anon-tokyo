"""Agent-centric simulation encoder with explicit ego-frame context."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import Tensor

from anon_tokyo.nn.attention import SparseTopKAttention, select_topk
from anon_tokyo.nn.polyline_encoder import PointNetPolylineEncoder
from anon_tokyo.simulation.dynamics import wrap_angle


def _rotate_to_ego(xy: Tensor, heading: Tensor) -> Tensor:
    """Rotate scene-frame vectors into an ego frame with x aligned to heading."""
    c = heading.cos().unsqueeze(-1)
    s = heading.sin().unsqueeze(-1)
    x = xy[..., 0:1]
    y = xy[..., 1:2]
    return torch.cat((x * c + y * s, -x * s + y * c), dim=-1)


def _gather_context(values: Tensor, batch_idx: Tensor, indices: Tensor) -> Tensor:
    """Gather ``values [B, N, ...]`` for query rows ``indices [Q, K]``."""
    return values[batch_idx[:, None], indices]


def _pad_topk(idx: Tensor, valid: Tensor, k: int) -> tuple[Tensor, Tensor]:
    if idx.shape[-1] == k:
        return idx, valid
    pad_shape = (*idx.shape[:-1], k - idx.shape[-1])
    idx = torch.cat((idx, idx.new_zeros(pad_shape)), dim=-1)
    valid = torch.cat((valid, torch.zeros(pad_shape, dtype=torch.bool, device=valid.device)), dim=-1)
    return idx, valid


def _nearest_agent_indices_for_queries(
    query_batch_idx: Tensor,
    query_agent_idx: Tensor,
    agent_pos: Tensor,
    agent_mask: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    query_pos = agent_pos[query_batch_idx, query_agent_idx]
    key_pos = agent_pos[query_batch_idx]
    key_mask = agent_mask[query_batch_idx]
    dist = (query_pos[:, None] - key_pos).norm(dim=-1)
    dist = dist.masked_fill(~key_mask, float("inf"))
    dist[torch.arange(query_agent_idx.numel(), device=agent_pos.device), query_agent_idx] = -1.0
    actual_k = min(k, agent_pos.shape[1])
    top_dist, idx = dist.topk(actual_k, dim=-1, largest=False)
    valid = torch.isfinite(top_dist)
    return _pad_topk(idx, valid, k)


def _nearest_map_indices_for_queries(
    query_batch_idx: Tensor,
    query_agent_idx: Tensor,
    agent_pos: Tensor,
    map_center: Tensor,
    map_mask: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    query_pos = agent_pos[query_batch_idx, query_agent_idx]
    key_pos = map_center[query_batch_idx]
    key_mask = map_mask[query_batch_idx]
    dist = (query_pos[:, None] - key_pos).norm(dim=-1)
    dist = dist.masked_fill(~key_mask, float("inf"))
    actual_k = min(k, map_center.shape[1])
    top_dist, idx = dist.topk(actual_k, dim=-1, largest=False)
    valid = torch.isfinite(top_dist)
    return _pad_topk(idx, valid, k)


def _ensure_points(mask: Tensor) -> Tensor:
    """Make every polyline have at least one point for the PointNet encoder."""
    safe = mask.clone()
    empty = ~safe.any(dim=-1)
    if empty.any():
        safe[..., 0] = torch.where(empty, torch.ones_like(safe[..., 0]), safe[..., 0])
    return safe


def _ensure_tokens(mask: Tensor) -> Tensor:
    """Make every attention row have at least one token to avoid all -inf softmax."""
    safe = mask.clone()
    empty = ~safe.any(dim=-1)
    if empty.any():
        safe[empty, 0] = True
    return safe


def _profile(profiler, name: str):
    return profiler.record(name) if profiler is not None else nullcontext()


class _InteractionLayer(nn.Module):
    """AnonTokyo-style map-map, agent-agent, and agent-map sparse attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sparse_k: int,
        dropout: float,
        use_map_self_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_map_self_attention = use_map_self_attention
        if use_map_self_attention:
            self.mm_attn = SparseTopKAttention(
                d_model=d_model,
                num_heads=num_heads,
                sparse_k=sparse_k,
                dropout=dropout,
                position_encoding="rope",
            )
            self.mm_norm = nn.LayerNorm(d_model)
            self.mm_ffn_norm = nn.LayerNorm(d_model)
            self.mm_ffn = self._ffn(d_model, dropout)
        self.aa_attn = SparseTopKAttention(
            d_model=d_model,
            num_heads=num_heads,
            sparse_k=sparse_k,
            dropout=dropout,
            position_encoding="rope",
        )
        self.am_attn = SparseTopKAttention(
            d_model=d_model,
            num_heads=num_heads,
            sparse_k=sparse_k,
            dropout=dropout,
            position_encoding="rope",
        )

        self.aa_norm = nn.LayerNorm(d_model)
        self.am_norm = nn.LayerNorm(d_model)
        self.aa_ffn_norm = nn.LayerNorm(d_model)
        self.am_ffn_norm = nn.LayerNorm(d_model)

        self.aa_ffn = self._ffn(d_model, dropout)
        self.am_ffn = self._ffn(d_model, dropout)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _ffn(d_model: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        agent_feat: Tensor,
        map_feat: Tensor,
        agent_pos: Tensor,
        map_pos: Tensor,
        agent_mask: Tensor,
        map_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        zero_agent_heading = agent_pos.new_zeros(agent_pos.shape[:-1])
        zero_map_heading = map_pos.new_zeros(map_pos.shape[:-1])

        if self.use_map_self_attention:
            mm_topk = select_topk(map_pos, map_pos, map_mask, self.mm_attn.sparse_k)
            mm = self.mm_attn(map_feat, map_feat, map_pos, map_pos, zero_map_heading, zero_map_heading, map_mask, mm_topk)
            map_feat = self.mm_norm(map_feat + self.drop(mm))
            map_feat = self.mm_ffn_norm(map_feat + self.drop(self.mm_ffn(map_feat)))
            map_feat = map_feat.masked_fill(~map_mask[..., None], 0.0)

        if self.use_map_self_attention:
            aa_topk = select_topk(agent_pos, agent_pos, agent_mask, self.aa_attn.sparse_k)
            aa = self.aa_attn(
                agent_feat,
                agent_feat,
                agent_pos,
                agent_pos,
                zero_agent_heading,
                zero_agent_heading,
                agent_mask,
                aa_topk,
            )
            agent_feat = self.aa_norm(agent_feat + self.drop(aa))
            agent_feat = self.aa_ffn_norm(agent_feat + self.drop(self.aa_ffn(agent_feat)))
            agent_feat = agent_feat.masked_fill(~agent_mask[..., None], 0.0)

            am_topk = select_topk(agent_pos, map_pos, map_mask, self.am_attn.sparse_k)
            am = self.am_attn(
                agent_feat,
                map_feat,
                agent_pos,
                map_pos,
                zero_agent_heading,
                zero_map_heading,
                map_mask,
                am_topk,
            )
            agent_feat = self.am_norm(agent_feat + self.drop(am))
            agent_feat = self.am_ffn_norm(agent_feat + self.drop(self.am_ffn(agent_feat)))
            agent_feat = agent_feat.masked_fill(~agent_mask[..., None], 0.0)
            return agent_feat, map_feat

        ego_feat = agent_feat[:, :1]
        ego_pos = agent_pos[:, :1]
        ego_heading = zero_agent_heading[:, :1]
        aa_topk = select_topk(ego_pos, agent_pos, agent_mask, self.aa_attn.sparse_k)
        aa = self.aa_attn(
            ego_feat,
            agent_feat,
            ego_pos,
            agent_pos,
            ego_heading,
            zero_agent_heading,
            agent_mask,
            aa_topk,
        )
        ego_feat = self.aa_norm(ego_feat + self.drop(aa))
        ego_feat = self.aa_ffn_norm(ego_feat + self.drop(self.aa_ffn(ego_feat)))

        am_topk = select_topk(ego_pos, map_pos, map_mask, self.am_attn.sparse_k)
        am = self.am_attn(
            ego_feat,
            map_feat,
            ego_pos,
            map_pos,
            ego_heading,
            zero_map_heading,
            map_mask,
            am_topk,
        )
        ego_feat = self.am_norm(ego_feat + self.drop(am))
        ego_feat = self.am_ffn_norm(ego_feat + self.drop(self.am_ffn(ego_feat)))
        agent_feat = agent_feat.clone()
        agent_feat[:, :1] = ego_feat.masked_fill(~agent_mask[:, :1, None], 0.0)
        return agent_feat, map_feat


class AgentCentricEncoder(nn.Module):
    """Encode each agent from explicitly ego-centric neighbouring agents and map tokens."""

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        sparse_k: int = 16,
        dropout: float = 0.1,
        max_context_agents: int = 32,
        max_context_maps: int = 128,
        agent_in_channels: int = 20,
        map_in_channels: int = 11,
        use_map_self_attention: bool = True,
    ) -> None:
        super().__init__()
        if max_context_agents < 1:
            raise ValueError("max_context_agents must be >= 1")
        if max_context_maps < 1:
            raise ValueError("max_context_maps must be >= 1")
        if d_model % num_heads != 0 or (d_model // num_heads) % 4 != 0:
            raise ValueError("d_model / num_heads must be divisible by 4 for 2-D RoPE")

        self.d_model = d_model
        self.sparse_k = sparse_k
        self.max_context_agents = max_context_agents
        self.max_context_maps = max_context_maps
        self.agent_in_channels = agent_in_channels
        self.map_in_channels = map_in_channels
        self.use_map_self_attention = use_map_self_attention

        self.agent_polyline_encoder = PointNetPolylineEncoder(
            in_channels=agent_in_channels + 1,
            hidden_dim=d_model,
            num_layers=3,
            num_pre_layers=1,
            out_channels=d_model,
        )
        self.map_token_encoder = nn.Sequential(
            nn.Linear(map_in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.layers = nn.ModuleList(
            [
                _InteractionLayer(
                    d_model,
                    num_heads,
                    sparse_k,
                    dropout,
                    use_map_self_attention=use_map_self_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(5, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def _agent_features(
        self,
        batch: dict[str, Tensor],
        query_batch_idx: Tensor,
        query_agent_idx: Tensor,
        agent_idx: Tensor,
        selected_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        obj_trajs = batch["obj_trajs"]
        obj_mask = batch["obj_trajs_mask"].bool()
        agent_pos = batch["obj_positions"].to(dtype=obj_trajs.dtype)
        agent_heading = batch["obj_headings"].to(dtype=obj_trajs.dtype)
        agent_mask = batch["agent_mask"].bool()
        Q, K = agent_idx.shape
        B, A = agent_mask.shape
        T = obj_trajs.shape[2]
        device = obj_trajs.device
        dtype = obj_trajs.dtype

        ctx = _gather_context(obj_trajs, query_batch_idx, agent_idx)
        ctx_mask = _gather_context(obj_mask, query_batch_idx, agent_idx).bool()
        ctx_agent_mask = _gather_context(agent_mask, query_batch_idx, agent_idx).bool() & selected_valid

        ego_pos = agent_pos[query_batch_idx, query_agent_idx].view(Q, 1, 1, 2)
        ego_heading = agent_heading[query_batch_idx, query_agent_idx].view(Q, 1, 1)
        local_xy = _rotate_to_ego(ctx[..., 0:2] - ego_pos, ego_heading)
        local_vel = _rotate_to_ego(ctx[..., 8:10], ego_heading)

        heading_t = torch.atan2(ctx[..., 6], ctx[..., 7])
        rel_heading = wrap_angle(heading_t - ego_heading)
        heading_feature = torch.stack((rel_heading.sin(), rel_heading.cos()), dim=-1)

        vel_prev = local_vel.roll(shifts=1, dims=3).clone()
        vel_prev[..., 0, :] = local_vel[..., 0, :]
        accel = (local_vel - vel_prev) / 0.1

        obj_types = batch.get("obj_types")
        if obj_types is None:
            obj_types = torch.zeros(B, A, dtype=torch.long, device=device)
        ctx_types = _gather_context(obj_types.to(device=device).long(), query_batch_idx, agent_idx)
        type_onehot = obj_trajs.new_zeros(Q, K, T, 4)
        type_onehot[..., 0] = (ctx_types == 1).to(dtype=dtype)[..., None]
        type_onehot[..., 1] = (ctx_types == 2).to(dtype=dtype)[..., None]
        type_onehot[..., 2] = (ctx_types == 3).to(dtype=dtype)[..., None]
        type_onehot[..., 3] = ((ctx_types <= 0) | (ctx_types > 3)).to(dtype=dtype)[..., None]

        is_ego = ((agent_idx == query_agent_idx[:, None]) & selected_valid).to(dtype=dtype)
        is_ego = is_ego[..., None, None].expand(-1, -1, T, -1)

        controlled = batch.get("controlled_mask")
        if controlled is None:
            controlled = torch.zeros(B, A, dtype=torch.bool, device=device)
        ctx_controlled = _gather_context(controlled.to(device=device).bool(), query_batch_idx, agent_idx)
        is_controlled = (ctx_controlled & selected_valid).to(dtype=dtype)
        is_controlled = is_controlled[..., None, None].expand(-1, -1, T, -1)

        sdc_idx = batch.get("sdc_track_index")
        if sdc_idx is None:
            is_sdc_token = torch.zeros(Q, K, dtype=torch.bool, device=device)
        else:
            sdc_idx = torch.as_tensor(sdc_idx, device=device).long()
            if sdc_idx.ndim == 0:
                sdc_idx = sdc_idx[None].expand(B)
            is_sdc_token = (agent_idx == sdc_idx[query_batch_idx, None]) & selected_valid
        is_sdc = is_sdc_token.to(dtype=dtype)[..., None, None].expand(-1, -1, T, -1)

        time = torch.linspace(-1.0, 0.0, T, device=device, dtype=dtype).view(1, 1, T, 1)
        time = time.expand(Q, K, T, 1)

        point_features = torch.cat(
            (
                local_xy,
                ctx[..., 2:6],
                heading_feature,
                local_vel,
                accel,
                type_onehot,
                is_ego,
                is_controlled,
                is_sdc,
                time,
            ),
            dim=-1,
        )
        if point_features.shape[-1] != self.agent_in_channels:
            raise ValueError(f"Expected {self.agent_in_channels} agent channels, got {point_features.shape[-1]}")

        point_mask = ctx_mask & ctx_agent_mask[..., None]
        point_features = point_features.masked_fill(~point_mask[..., None], 0.0)
        token_pos = local_xy[..., -1, :]
        return point_features, point_mask, token_pos, ctx_agent_mask

    def _map_features(
        self,
        batch: dict[str, Tensor],
        query_batch_idx: Tensor,
        query_agent_idx: Tensor,
        map_idx: Tensor,
        selected_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        map_token_features = batch.get("map_token_features")
        if map_token_features is not None:
            map_tokens = map_token_features
            map_center = batch["map_polylines_center"].to(dtype=map_tokens.dtype)
            map_mask = batch["map_mask"].bool()
            agent_pos = batch["obj_positions"].to(dtype=map_tokens.dtype)
            agent_heading = batch["obj_headings"].to(dtype=map_tokens.dtype)

            selected_tokens = _gather_context(map_tokens, query_batch_idx, map_idx)
            selected_center = _gather_context(map_center, query_batch_idx, map_idx)
            token_mask = _gather_context(map_mask, query_batch_idx, map_idx).bool() & selected_valid

            ego_pos = agent_pos[query_batch_idx, query_agent_idx]
            ego_heading = agent_heading[query_batch_idx, query_agent_idx]
            ego_heading_tokens = ego_heading[:, None]
            center_xy = _rotate_to_ego(selected_center - ego_pos[:, None, :], ego_heading_tokens)
            first_xy = _rotate_to_ego(selected_tokens[..., 2:4] - ego_pos[:, None, :], ego_heading_tokens)
            last_xy = _rotate_to_ego(selected_tokens[..., 4:6] - ego_pos[:, None, :], ego_heading_tokens)
            dir_xy = _rotate_to_ego(selected_tokens[..., 6:8], ego_heading_tokens)
            token_features = torch.cat(
                (
                    center_xy,
                    first_xy,
                    last_xy,
                    dir_xy,
                    selected_tokens[..., 8:11],
                ),
                dim=-1,
            )
            if token_features.shape[-1] != self.map_in_channels:
                raise ValueError(f"Expected {self.map_in_channels} map channels, got {token_features.shape[-1]}")
            token_features = token_features.masked_fill(~token_mask[..., None], 0.0)
            return token_features, center_xy, token_mask

        map_polys = batch["map_polylines"]
        map_point_mask = batch["map_polylines_mask"].bool()
        map_center = batch["map_polylines_center"].to(dtype=map_polys.dtype)
        agent_pos = batch["obj_positions"].to(dtype=map_polys.dtype)
        agent_heading = batch["obj_headings"].to(dtype=map_polys.dtype)

        selected_polys = _gather_context(map_polys, query_batch_idx, map_idx)
        selected_point_mask = _gather_context(map_point_mask, query_batch_idx, map_idx).bool() & selected_valid[..., None]
        selected_center = _gather_context(map_center, query_batch_idx, map_idx)

        ego_pos = agent_pos[query_batch_idx, query_agent_idx]
        ego_heading = agent_heading[query_batch_idx, query_agent_idx]
        ego_pos_points = ego_pos[:, None, None, :]
        ego_heading_points = ego_heading[:, None, None]
        local_xy = _rotate_to_ego(selected_polys[..., 0:2] - ego_pos_points, ego_heading_points)
        local_dir = _rotate_to_ego(selected_polys[..., 3:5], ego_heading_points)

        mask_f = selected_point_mask.to(dtype=map_polys.dtype)
        count = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        center_xy = (local_xy * mask_f[..., None]).sum(dim=2) / count
        z_mean = (selected_polys[..., 2] * mask_f).sum(dim=2, keepdim=True) / count

        point_order = torch.arange(selected_point_mask.shape[-1], device=selected_point_mask.device)
        first_idx = torch.where(
            selected_point_mask,
            point_order.view(1, 1, -1),
            point_order.new_full(selected_point_mask.shape, selected_point_mask.shape[-1]),
        ).amin(dim=-1)
        last_idx = torch.where(
            selected_point_mask,
            point_order.view(1, 1, -1),
            point_order.new_zeros(selected_point_mask.shape),
        ).amax(dim=-1)
        first_idx = first_idx.clamp_max(selected_point_mask.shape[-1] - 1)
        gather_first = first_idx[..., None, None].expand(-1, -1, 1, 2)
        gather_last = last_idx[..., None, None].expand(-1, -1, 1, 2)
        first_xy = local_xy.gather(2, gather_first).squeeze(2)
        last_xy = local_xy.gather(2, gather_last).squeeze(2)

        segment = last_xy - first_xy
        dir_xy = torch.nn.functional.normalize(segment, p=2, dim=-1, eps=1e-6)
        mean_dir = (local_dir * mask_f[..., None]).sum(dim=2)
        mean_dir = torch.nn.functional.normalize(mean_dir, p=2, dim=-1, eps=1e-6)
        dir_xy = torch.where(segment.norm(dim=-1, keepdim=True) > 1e-4, dir_xy, mean_dir)

        edge_len = (local_xy[:, :, 1:] - local_xy[:, :, :-1]).norm(dim=-1)
        edge_mask = selected_point_mask[:, :, 1:] & selected_point_mask[:, :, :-1]
        length = (edge_len * edge_mask.to(dtype=map_polys.dtype)).sum(dim=-1, keepdim=True)
        type_code = (selected_polys[..., 6] * mask_f).sum(dim=2, keepdim=True) / count

        token_features = torch.cat(
            (
                center_xy,
                first_xy,
                last_xy,
                dir_xy,
                length,
                z_mean,
                type_code,
            ),
            dim=-1,
        )
        if token_features.shape[-1] != self.map_in_channels:
            raise ValueError(f"Expected {self.map_in_channels} map channels, got {token_features.shape[-1]}")

        token_mask = selected_point_mask.any(dim=-1)
        token_features = token_features.masked_fill(~token_mask[..., None], 0.0)
        ego_heading_tokens = ego_heading[:, None]
        token_pos = _rotate_to_ego(selected_center - ego_pos[:, None, :], ego_heading_tokens)
        return token_features, token_pos, token_mask

    def _goal_features(self, batch: dict[str, Tensor], query_batch_idx: Tensor, query_agent_idx: Tensor) -> Tensor:
        agent_pos = batch["obj_positions"]
        agent_heading = batch["obj_headings"]
        goal_pos = batch.get("goal_positions")
        if goal_pos is None:
            goal_pos = agent_pos
        ego_pos = agent_pos[query_batch_idx, query_agent_idx]
        ego_heading = agent_heading[query_batch_idx, query_agent_idx]
        ego_goal = goal_pos.to(dtype=agent_pos.dtype)[query_batch_idx, query_agent_idx]
        local_goal = _rotate_to_ego(ego_goal - ego_pos, ego_heading)
        dist = local_goal.norm(dim=-1, keepdim=True)
        direction = local_goal / dist.clamp_min(1e-3)
        return torch.cat((local_goal, dist, direction), dim=-1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        profiler = getattr(self, "_profiler", None)
        with _profile(profiler, "encoder.prepare"):
            agent_pos = batch["obj_positions"]
            agent_mask = batch["agent_mask"].bool()
            map_center = batch["map_polylines_center"]
            map_mask = batch["map_mask"].bool()
            B, A, _ = agent_pos.shape

            controlled_mask = batch.get("controlled_mask")
            if controlled_mask is None:
                ego_mask = agent_mask
            else:
                ego_mask = controlled_mask.to(device=agent_pos.device).bool() & agent_mask

            query_batch_idx, query_agent_idx = ego_mask.nonzero(as_tuple=True)
            context_agent_indices = torch.zeros(B, A, self.max_context_agents, dtype=torch.long, device=agent_pos.device)
            context_map_indices = torch.zeros(B, A, self.max_context_maps, dtype=torch.long, device=agent_pos.device)
            ego_feature = agent_pos.new_zeros(B, A, self.d_model)
        if query_batch_idx.numel() == 0:
            return {
                "ego_feature": ego_feature,
                "ego_mask": ego_mask,
                "context_agent_indices": context_agent_indices,
                "context_map_indices": context_map_indices,
            }

        with _profile(profiler, "encoder.nearest_agents"):
            agent_idx, agent_selected = _nearest_agent_indices_for_queries(
                query_batch_idx,
                query_agent_idx,
                agent_pos,
                agent_mask,
                self.max_context_agents,
            )
        with _profile(profiler, "encoder.nearest_maps"):
            map_idx, map_selected = _nearest_map_indices_for_queries(
                query_batch_idx,
                query_agent_idx,
                agent_pos,
                map_center,
                map_mask,
                self.max_context_maps,
            )

        with _profile(profiler, "encoder.agent_features"):
            agent_points, agent_point_mask, agent_token_pos, agent_token_mask = self._agent_features(
                batch, query_batch_idx, query_agent_idx, agent_idx, agent_selected
            )
        with _profile(profiler, "encoder.map_features"):
            map_token_features, map_token_pos, map_token_mask = self._map_features(
                batch, query_batch_idx, query_agent_idx, map_idx, map_selected
            )

        K = agent_points.shape[1]
        L = map_token_features.shape[1]

        with _profile(profiler, "encoder.agent_pointnet"):
            agent_input = torch.cat((agent_points, agent_point_mask.unsqueeze(-1).to(dtype=agent_points.dtype)), dim=-1)
            agent_feat = self.agent_polyline_encoder(agent_input, _ensure_points(agent_point_mask))
            agent_feat = agent_feat.masked_fill(~agent_token_mask[..., None], 0.0)

        with _profile(profiler, "encoder.map_token_mlp"):
            map_feat = self.map_token_encoder(map_token_features)
            map_feat = map_feat.masked_fill(~map_token_mask[..., None], 0.0)

        with _profile(profiler, "encoder.interaction_layers"):
            safe_agent_token_mask = _ensure_tokens(agent_token_mask)
            safe_map_token_mask = _ensure_tokens(map_token_mask)
            for layer in self.layers:
                agent_feat, map_feat = layer(
                    agent_feat,
                    map_feat,
                    agent_token_pos,
                    map_token_pos,
                    safe_agent_token_mask,
                    safe_map_token_mask,
                )
                agent_feat = agent_feat.masked_fill(~agent_token_mask[..., None], 0.0)
                map_feat = map_feat.masked_fill(~map_token_mask[..., None], 0.0)

        with _profile(profiler, "encoder.output"):
            ego_feature_query = agent_feat[:, 0]
            goal_feature = self.goal_encoder(self._goal_features(batch, query_batch_idx, query_agent_idx))
            ego_feature_query = self.out_norm(ego_feature_query + goal_feature)
            ego_feature[query_batch_idx, query_agent_idx] = ego_feature_query
            context_agent_indices[query_batch_idx, query_agent_idx, :K] = agent_idx[:, :K]
            context_map_indices[query_batch_idx, query_agent_idx, :L] = map_idx[:, :L]

        return {
            "ego_feature": ego_feature,
            "ego_mask": ego_mask,
            "context_agent_indices": context_agent_indices,
            "context_map_indices": context_map_indices,
        }
