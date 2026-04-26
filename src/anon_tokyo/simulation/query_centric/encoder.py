"""Query-centric simulation encoder.

The scene is encoded once in the same query-centric layout used by prediction,
then controlled agents are read out for PPO policy/value heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from anon_tokyo.nn.attention import select_topk
from anon_tokyo.prediction.anon_tokyo.encoder import AnonTokyoEncoder as PredictionAnonTokyoEncoder
from anon_tokyo.simulation.agent_centric.encoder import _rotate_to_ego


class _SimulationSceneEncoder(PredictionAnonTokyoEncoder):
    """Prediction scene encoder with fixed-width simulation agent history features."""

    def __init__(self, *args, map_token_in_channels: int = 11, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.map_token_encoder = nn.Sequential(
            nn.Linear(map_token_in_channels, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    @staticmethod
    def _augment_agent_features(batch: dict[str, Tensor]) -> Tensor:
        obj_trajs = batch["obj_trajs"]
        obj_mask = batch["obj_trajs_mask"].bool()
        B, A, T, _ = obj_trajs.shape
        dtype = obj_trajs.dtype
        device = obj_trajs.device

        agent_pos = batch["obj_positions"].to(device=device, dtype=dtype)
        agent_heading = batch["obj_headings"].to(device=device, dtype=dtype)
        local_xy = _rotate_to_ego(obj_trajs[..., 0:2] - agent_pos[:, :, None, :], agent_heading[:, :, None])
        local_vel = _rotate_to_ego(obj_trajs[..., 8:10], agent_heading[:, :, None])

        heading_t = torch.atan2(obj_trajs[..., 6], obj_trajs[..., 7])
        rel_heading = heading_t - agent_heading[:, :, None]
        heading_feature = torch.stack((rel_heading.sin(), rel_heading.cos()), dim=-1)

        vel_prev = local_vel.roll(shifts=1, dims=2).clone()
        vel_prev[:, :, 0] = local_vel[:, :, 0]
        accel = (local_vel - vel_prev) / 0.1

        obj_types = batch.get("obj_types")
        if obj_types is None:
            obj_types = torch.zeros(B, A, dtype=torch.long, device=device)
        obj_types = obj_types.to(device=device).long()
        type_onehot = obj_trajs.new_zeros(B, A, T, 4)
        type_onehot[..., 0] = (obj_types == 1).to(dtype=dtype)[:, :, None]
        type_onehot[..., 1] = (obj_types == 2).to(dtype=dtype)[:, :, None]
        type_onehot[..., 2] = (obj_types == 3).to(dtype=dtype)[:, :, None]
        type_onehot[..., 3] = ((obj_types <= 0) | (obj_types > 3)).to(dtype=dtype)[:, :, None]

        controlled = batch.get("controlled_mask")
        if controlled is None:
            controlled = torch.zeros(B, A, dtype=torch.bool, device=device)
        is_controlled = controlled.to(device=device).bool().to(dtype=dtype)[:, :, None, None].expand(-1, -1, T, -1)

        sdc_idx = batch.get("sdc_track_index")
        is_sdc_token = torch.zeros(B, A, dtype=torch.bool, device=device)
        if sdc_idx is not None:
            sdc_idx = torch.as_tensor(sdc_idx, device=device).long()
            if sdc_idx.ndim == 0:
                sdc_idx = sdc_idx[None].expand(B)
            valid_sdc = (sdc_idx >= 0) & (sdc_idx < A)
            if valid_sdc.any():
                bi = torch.arange(B, device=device)[valid_sdc]
                is_sdc_token[bi, sdc_idx[valid_sdc]] = True
        is_sdc = is_sdc_token.to(dtype=dtype)[:, :, None, None].expand(-1, -1, T, -1)

        time = torch.linspace(-1.0, 0.0, T, device=device, dtype=dtype).view(1, 1, T, 1)
        time = time.expand(B, A, T, 1)

        out = torch.cat(
            (
                local_xy,
                obj_trajs[..., 2:6],
                heading_feature,
                local_vel,
                accel,
                type_onehot,
                is_controlled,
                is_sdc,
                time,
            ),
            dim=-1,
        )
        return out.masked_fill(~obj_mask[..., None], 0.0)

    def _encode_map_features(
        self,
        batch: dict[str, Tensor],
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        map_center = batch["map_polylines_center"].to(dtype=dtype)
        map_mask = batch["map_mask"].bool()
        map_heading = batch.get("map_headings")
        if map_heading is None:
            map_heading = map_center.new_zeros(map_center.shape[:2])
        else:
            map_heading = map_heading.to(device=map_center.device, dtype=dtype)

        map_token_features = batch.get("map_token_features")
        if map_token_features is not None:
            tokens = map_token_features.to(device=map_center.device, dtype=dtype)
            center = tokens[..., 0:2]
            token_features = torch.cat(
                (
                    torch.zeros_like(center),
                    tokens[..., 2:4] - center,
                    tokens[..., 4:6] - center,
                    tokens[..., 6:8],
                    tokens[..., 8:11],
                ),
                dim=-1,
            )
            map_feat = self.map_token_encoder(token_features)
            map_feat = map_feat.masked_fill(~map_mask[..., None], 0.0)
            return map_feat, map_center, map_heading, map_mask

        map_poly_mask = batch["map_polylines_mask"]
        map_polys = self._augment_map_features(batch)
        if map_polys.shape[-1] != self.map_in_channels:
            raise ValueError(f"Expected {self.map_in_channels} map features, got {map_polys.shape[-1]}")
        map_feat = self.map_encoder(map_polys, map_poly_mask.bool())
        map_feat = map_feat.masked_fill(~map_mask[..., None], 0.0)
        return map_feat, map_center, map_heading, map_mask

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        obj_mask_t = batch["obj_trajs_mask"]
        agent_mask = batch["agent_mask"].bool()
        agent_pos = batch["obj_positions"]
        agent_heading = batch["obj_headings"]

        obj_trajs = self._augment_agent_features(batch)
        if obj_trajs.shape[-1] != self.agent_in_channels:
            raise ValueError(f"Expected {self.agent_in_channels} agent features, got {obj_trajs.shape[-1]}")

        agent_token_mask = obj_mask_t.bool()
        agent_input = torch.cat([obj_trajs, agent_token_mask.unsqueeze(-1).to(dtype=obj_trajs.dtype)], dim=-1)

        agent_feat = self.agent_polyline_encoder(agent_input, agent_token_mask)
        agent_feat = agent_feat.masked_fill(~agent_mask[..., None], 0.0)
        map_feat, map_center, map_heading, map_mask = self._encode_map_features(batch, dtype=obj_trajs.dtype)

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

        agent_feat = agent_feat.masked_fill(~agent_mask[..., None], 0.0)
        return {
            "obj_feature": agent_feat,
            "map_feature": map_feat,
            "obj_mask": agent_mask,
            "map_mask": map_mask,
            "obj_pos": agent_pos,
            "map_pos": map_center,
            "obj_headings": agent_heading,
        }


class QueryCentricEncoder(nn.Module):
    """Encode a full scene once and return features for controlled agents."""

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        sparse_k: int = 16,
        dropout: float = 0.1,
        use_rope: bool = False,
        use_drope: bool = False,
        position_encoding: str | None = "rpe",
        agent_in_channels: int = 19,
        map_in_channels: int = 9,
        include_goal: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.include_goal = include_goal
        self.scene_encoder = _SimulationSceneEncoder(
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
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(5, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _ego_mask(batch: dict[str, Tensor]) -> Tensor:
        agent_mask = batch["agent_mask"].bool()
        controlled_mask = batch.get("controlled_mask")
        if controlled_mask is None:
            return agent_mask
        return controlled_mask.to(device=agent_mask.device).bool() & agent_mask

    @staticmethod
    def _goal_features(batch: dict[str, Tensor]) -> Tensor:
        agent_pos = batch["obj_positions"]
        agent_heading = batch["obj_headings"]
        goal_pos = batch.get("goal_positions")
        if goal_pos is None:
            goal_pos = agent_pos
        local_goal = _rotate_to_ego(goal_pos.to(dtype=agent_pos.dtype) - agent_pos, agent_heading)
        dist = local_goal.norm(dim=-1, keepdim=True)
        direction = local_goal / dist.clamp_min(1e-3)
        return torch.cat((local_goal, dist, direction), dim=-1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        enc_out = self.scene_encoder(batch)
        ego_mask = self._ego_mask(batch)
        ego_feature = enc_out["obj_feature"]
        if self.include_goal:
            goal_feature = self.goal_encoder(self._goal_features(batch))
            ego_feature = self.out_norm(ego_feature + goal_feature)
        ego_feature = ego_feature.masked_fill(~ego_mask[..., None], 0.0)
        return {
            "ego_feature": ego_feature,
            "ego_mask": ego_mask,
            "obj_feature": enc_out["obj_feature"],
            "map_feature": enc_out["map_feature"],
            "obj_mask": enc_out["obj_mask"],
            "map_mask": enc_out["map_mask"],
            "obj_pos": enc_out["obj_pos"],
            "map_pos": enc_out["map_pos"],
            "obj_headings": enc_out["obj_headings"],
        }


__all__ = ["QueryCentricEncoder"]
