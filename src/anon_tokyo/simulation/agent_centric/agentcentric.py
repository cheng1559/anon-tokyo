"""Agent-centric multi-agent simulation policy."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta

from anon_tokyo.simulation.dynamics import wrap_angle


def _layer_init(layer: nn.Linear | nn.Conv1d, std: float = math.sqrt(2.0), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _sine_embed_for_position(pos: Tensor, hidden_dim: int) -> Tensor:
    half_hidden_dim = hidden_dim // 2
    scale = 2.0 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos[..., 0] * scale
    y_embed = pos[..., 1] * scale
    pos_x = x_embed.unsqueeze(-1) / dim_t
    pos_y = y_embed.unsqueeze(-1) / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


def _rotate_to_local(xy: Tensor, heading: Tensor) -> Tensor:
    c = heading.cos().unsqueeze(-1)
    s = heading.sin().unsqueeze(-1)
    x = xy[..., 0:1]
    y = xy[..., 1:2]
    return torch.cat((x * c + y * s, -x * s + y * c), dim=-1)


def _vehicle_to_polygon(length: Tensor, width: Tensor, pos: Tensor, heading: Tensor) -> Tensor:
    half_length = length / 2.0
    half_width = width / 2.0
    length_x = half_length * heading.cos()
    length_y = half_length * heading.sin()
    width_x = half_width * heading.sin()
    width_y = half_width * heading.cos()
    posx = pos[..., 0]
    posy = pos[..., 1]
    front_left = torch.stack((posx + length_x - width_x, posy + length_y + width_y), dim=-1)
    front_right = torch.stack((posx + length_x + width_x, posy + length_y - width_y), dim=-1)
    rear_right = torch.stack((posx - length_x + width_x, posy - length_y - width_y), dim=-1)
    rear_left = torch.stack((posx - length_x - width_x, posy - length_y + width_y), dim=-1)
    return torch.stack((front_left, front_right, rear_right, rear_left), dim=-2)


def _masked_amax(x: Tensor, mask: Tensor, dim: int, keepdim: bool) -> Tensor:
    fill = torch.finfo(x.dtype).min
    return x.masked_fill(~mask, fill).amax(dim=dim, keepdim=keepdim)


def _swap_ego_index(tensor: Tensor, persp_idx: Tensor) -> Tensor:
    n, a = tensor.shape[:2]
    perm = torch.arange(a, device=tensor.device).expand(n, a).clone()
    n_idx = torch.arange(n, device=tensor.device)
    perm[n_idx, 0] = persp_idx
    perm[n_idx, persp_idx] = 0
    idx_shape = [n, a] + [1] * (tensor.ndim - 2)
    return tensor.gather(1, perm.view(idx_shape).expand_as(tensor))


class AgentCentricNet(nn.Module):
    """Exact module layout for agent-centric checkpoints."""

    def __init__(
        self,
        *,
        agent_feature_size: int = 15,
        goal_feature_size: int = 2,
        kinematics_feature_size: int = 8,
        lane_bound_feature_size: int = 9,
        output_size: int = 2,
        embed_dim: int = 256,
        num_heads: int = 4,
        no_goal_allowed: bool = True,
        max_agents: int | None = 16,
        history_steps: int = 5,
        is_continuous: bool = True,
        use_layer_norm_layout: bool = False,
        lane_bezier_degree: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.no_goal_allowed = no_goal_allowed
        self.max_agents = max_agents
        self.history_steps = history_steps
        self.is_continuous = is_continuous
        self.lane_bezier_degree = lane_bezier_degree

        if history_steps > 1 and use_layer_norm_layout:
            self.agent_mlp = nn.Sequential(
                _layer_init(nn.Linear(agent_feature_size + 1, embed_dim // 2)),
                nn.LayerNorm(embed_dim // 2),
                nn.ReLU(),
                _layer_init(nn.Linear(embed_dim // 2, embed_dim)),
                nn.LayerNorm(embed_dim),
            )
        elif history_steps > 1:
            self.agent_mlp = nn.Sequential(
                _layer_init(nn.Linear(agent_feature_size + 1, embed_dim // 2)),
                nn.ReLU(),
                _layer_init(nn.Linear(embed_dim // 2, embed_dim)),
            )
        elif use_layer_norm_layout:
            self.agent_mlp = nn.Sequential(
                _layer_init(nn.Linear(agent_feature_size, embed_dim)),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                _layer_init(nn.Linear(embed_dim, embed_dim)),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.agent_mlp = nn.Sequential(
                _layer_init(nn.Linear(agent_feature_size, embed_dim)),
                nn.ReLU(),
                _layer_init(nn.Linear(embed_dim, embed_dim)),
            )
        self.kinematics_mlp = nn.Sequential(
            _layer_init(nn.Linear(kinematics_feature_size, embed_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(embed_dim, embed_dim)),
        )
        if use_layer_norm_layout:
            self.lane_bound_mlp = nn.Sequential(
                _layer_init(nn.Linear(lane_bound_feature_size, embed_dim)),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                _layer_init(nn.Linear(embed_dim, embed_dim)),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.lane_bound_mlp = nn.Sequential(
                _layer_init(nn.Linear(lane_bound_feature_size, embed_dim)),
                nn.ReLU(),
                _layer_init(nn.Linear(embed_dim, embed_dim)),
            )
        self.agent_ln1 = nn.LayerNorm(embed_dim)
        self.agent_self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.agent_ln2 = nn.LayerNorm(embed_dim)
        self.agent_ffn = nn.Sequential(
            _layer_init(nn.Linear(embed_dim, embed_dim * 4)),
            nn.ReLU(),
            _layer_init(nn.Linear(embed_dim * 4, embed_dim)),
        )
        self.default_lane_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        no_goal_label_dim = 1 if no_goal_allowed else 0
        self.combiner_mlp = nn.Sequential(
            _layer_init(nn.Linear(embed_dim * 4 + no_goal_label_dim, embed_dim * 4)),
            nn.ReLU(),
        )
        if is_continuous:
            self.final_mlp = nn.Sequential(
                _layer_init(nn.Linear(embed_dim * 4, embed_dim * 4)),
                nn.ReLU(),
                _layer_init(nn.Linear(embed_dim * 4, output_size * 2), std=0.01, bias_const=0.54),
            )
        else:
            self.final_mlp = nn.Sequential(
                _layer_init(nn.Linear(embed_dim * 4, embed_dim * 4)),
                nn.ReLU(),
                _layer_init(nn.Linear(embed_dim * 4, output_size), std=0.01),
            )
        self.value_head = nn.Sequential(
            _layer_init(nn.Linear(embed_dim * 4, embed_dim * 4)),
            nn.ReLU(),
            _layer_init(nn.Linear(embed_dim * 4, 1), std=1.0),
        )

    def _polyline_features(self, lane_features: Tensor, lane_mask: Tensor) -> Tensor:
        p = lane_features.shape[2]
        dtype = lane_features.dtype
        device = lane_features.device
        weight = lane_mask.to(dtype)
        valid = lane_mask.bool()

        first_idx = valid.float().argmax(dim=-1)
        last_idx = p - 1 - valid.flip(dims=[-1]).float().argmax(dim=-1)
        gather_idx = first_idx[..., None, None].expand(-1, -1, 1, 2)
        start_xy = lane_features[..., 0:2].gather(2, gather_idx).squeeze(2)
        gather_idx = last_idx[..., None, None].expand(-1, -1, 1, 2)
        end_xy = lane_features[..., 0:2].gather(2, gather_idx).squeeze(2)

        degree = max(int(self.lane_bezier_degree), 1)
        t = torch.linspace(0.0, 1.0, p, dtype=dtype, device=device)
        basis = torch.stack(
            [
                math.comb(degree, idx) * (1.0 - t).pow(degree - idx) * t.pow(idx)
                for idx in range(degree + 1)
            ],
            dim=-1,
        )
        endpoint_curve = basis[:, 0].view(1, 1, p, 1) * start_xy[..., None, :] + basis[:, -1].view(1, 1, p, 1) * end_xy[..., None, :]
        if degree > 1:
            inner_basis = basis[:, 1:-1]
            residual = lane_features[..., 0:2] - endpoint_curve
            normal = torch.einsum("...p,pc,pd->...cd", weight, inner_basis, inner_basis)
            eye = torch.eye(degree - 1, dtype=dtype, device=device)
            normal = normal + eye.view(*((1,) * (normal.ndim - 2)), *eye.shape) * 1e-4
            rhs = torch.einsum("...p,pc,...pk->...ck", weight, inner_basis, residual)
            controls = torch.linalg.solve(normal.float(), rhs.float()).to(dtype)
            bezier_params = controls.transpose(-1, -2).flatten(start_dim=-2)
        else:
            bezier_params = lane_features.new_zeros(*lane_features.shape[:2], 0)

        count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        line_type = (lane_features[..., 4:5] * weight.unsqueeze(-1)).sum(dim=-2) / count
        return torch.cat((start_xy, end_xy, bezier_params, line_type), dim=-1)

    def forward(
        self,
        agent_features: Tensor,
        goal_features: Tensor,
        kinematics_features: Tensor,
        agent_mask: Tensor,
        lane_bound_features: Tensor,
        lane_bound_mask: Tensor,
        visible_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        b, a, _, _ = agent_features.shape
        agent_padding_mask = ~agent_mask.any(dim=-1).bool()
        if self.history_steps > 1:
            time_info = torch.linspace(0, 1, self.history_steps, device=agent_features.device, dtype=agent_features.dtype)
            time_info = time_info.view(1, 1, self.history_steps, 1).expand(b, a, self.history_steps, 1)
            agent_features = torch.cat((agent_features, time_info * agent_mask.unsqueeze(-1).to(agent_features.dtype)), dim=-1)
            frame_embeddings = self.agent_mlp(agent_features)
            agent_embeddings = frame_embeddings.max(dim=2).values
        else:
            agent_embeddings = self.agent_mlp(agent_features.squeeze(2))

        goal_positions = goal_features[:, :1, :2]
        goal_reached = goal_features[:, :1, 2:]
        goal_embeddings = _sine_embed_for_position(goal_positions, self.embed_dim)
        kinematics_embeddings = self.kinematics_mlp(kinematics_features[:, :1])
        if lane_bound_features.shape[1] == 0 or lane_bound_features.shape[2] == 0:
            global_lane_embedding = self.default_lane_embedding.expand(b, 1, -1)
        else:
            polyline_mask = lane_bound_mask.bool().any(dim=-1, keepdim=True)
            polyline_features = self._polyline_features(lane_bound_features, lane_bound_mask)
            polyline_embeddings = self.lane_bound_mlp(polyline_features)
            pooled_lane = _masked_amax(polyline_embeddings, polyline_mask, dim=1, keepdim=True)
            has_lanes = polyline_mask.any(dim=1, keepdim=True)
            global_lane_embedding = torch.where(has_lanes, pooled_lane, self.default_lane_embedding.expand(b, 1, -1))

        agent_norm = self.agent_ln1(agent_embeddings)
        attn_mask = None
        if visible_mask is not None:
            attn_mask = (~visible_mask[:, :1, :].bool()).repeat_interleave(self.num_heads, dim=0)
            agent_padding_mask = None
        ego_attn_output = self.agent_self_attention(
            query=agent_norm[:, :1],
            key=agent_norm,
            value=agent_norm,
            key_padding_mask=agent_padding_mask,
            attn_mask=attn_mask,
        )[0]
        ego_embedding = agent_embeddings[:, :1] + ego_attn_output
        ego_embedding = ego_embedding + self.agent_ffn(self.agent_ln2(ego_embedding))
        combined = torch.cat((ego_embedding, goal_embeddings, kinematics_embeddings, global_lane_embedding), dim=-1)
        if self.no_goal_allowed:
            combined = torch.cat((combined, goal_reached), dim=-1)
        context = self.combiner_mlp(combined)
        if self.is_continuous:
            raw_alpha, raw_beta = self.final_mlp(context).chunk(2, dim=-1)
            action_params = torch.cat((F.softplus(raw_alpha) + 1.0, F.softplus(raw_beta) + 1.0), dim=-1)
        else:
            action_params = self.final_mlp(context)
        values = self.value_head(context).squeeze(-1)
        return action_params, values


class AgentCentricBackbone(nn.Module):
    """Adapter matching agent-centric preprocessing."""

    def __init__(
        self,
        *,
        embed_dim: int = 256,
        num_heads: int = 4,
        max_agents: int = 16,
        max_lanes: int = 96,
        history_steps: int = 5,
        no_goal_allowed: bool = True,
        agent_filter_radius: float = 200.0,
        topk_front_weight: float = 10.0,
        topk_rear_weight: float = 2.0,
        use_layer_norm_layout: bool = False,
        lane_bezier_degree: int = 3,
        action_low: tuple[float, float] | list[float] = (-5.0, -1.0),
        action_high: tuple[float, float] | list[float] = (3.0, 1.0),
    ) -> None:
        super().__init__()
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.history_steps = history_steps
        self.no_goal_allowed = no_goal_allowed
        self.agent_filter_radius = agent_filter_radius
        self.topk_front_weight = topk_front_weight
        self.topk_rear_weight = topk_rear_weight
        self.model = AgentCentricNet(
            embed_dim=embed_dim,
            num_heads=num_heads,
            no_goal_allowed=no_goal_allowed,
            max_agents=max_agents,
            history_steps=history_steps,
            is_continuous=True,
            use_layer_norm_layout=use_layer_norm_layout,
            lane_bezier_degree=lane_bezier_degree,
            lane_bound_feature_size=(lane_bezier_degree + 1) * 2 + 1,
        )
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32), persistent=False)
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32), persistent=False)

    def _select_queries(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        agent_mask = obs["agent_mask"].bool()
        controlled = obs.get("controlled_mask")
        training_mask = agent_mask if controlled is None else controlled.bool() & agent_mask
        b_idx, a_idx = training_mask.nonzero(as_tuple=True)
        return training_mask, b_idx, a_idx

    def _select_nearby_agents(self, positions: Tensor, valid: Tensor, b_idx: Tensor, a_idx: Tensor, base_heading: Tensor) -> Tensor:
        n, a = b_idx.numel(), positions.shape[1]
        if self.max_agents <= 0 or self.max_agents >= a:
            return torch.arange(a, device=positions.device).expand(n, a)
        last_pos = positions[b_idx, :, -1]
        ego_pos = positions[b_idx, a_idx, -1].unsqueeze(1)
        diff = last_pos - ego_pos
        cos_h = base_heading.cos().unsqueeze(1)
        sin_h = base_heading.sin().unsqueeze(1)
        longitudinal = diff[..., 0] * cos_h + diff[..., 1] * sin_h
        lateral = -diff[..., 0] * sin_h + diff[..., 1] * cos_h
        w = torch.where(longitudinal >= 0, self.topk_front_weight, self.topk_rear_weight)
        dists = torch.sqrt((longitudinal / w.clamp_min(1e-6)).square() + lateral.square())
        dists = dists.masked_fill(~valid[b_idx, :, -1].bool(), torch.inf)
        dists[torch.arange(n, device=positions.device), a_idx] = 0.0
        return torch.topk(dists, k=self.max_agents, dim=1, largest=False).indices

    def _select_lanes(self, lane_features: Tensor, lane_mask: Tensor) -> tuple[Tensor, Tensor]:
        m = lane_features.shape[1]
        if m == 0:
            return lane_features, lane_mask
        k = self.max_lanes if self.max_lanes > 0 else max(m // 8, 1)
        k = min(k, m)
        w_front = max(self.topk_front_weight, 1e-6)
        w_rear = max(self.topk_rear_weight, 1e-6)
        point_valid = lane_mask.bool()
        denom = point_valid.sum(dim=-1).clamp_min(1).to(lane_features.dtype)
        center = (lane_features[..., 0:2] * point_valid.unsqueeze(-1).to(lane_features.dtype)).sum(dim=-2) / denom.unsqueeze(-1)
        lane_cx = center[..., 0]
        lane_cy = center[..., 1]
        w = torch.where(lane_cx >= 0, w_front, w_rear)
        dists = torch.sqrt((lane_cx / w).square() + lane_cy.square())
        dists = dists.masked_fill(~point_valid.any(dim=-1), torch.inf)
        idx = torch.topk(dists, k=k, dim=1, largest=False).indices
        selected = lane_features.gather(1, idx[..., None, None].expand(-1, -1, lane_features.shape[2], lane_features.shape[-1]))
        selected_mask = lane_mask.gather(1, idx[..., None].expand(-1, -1, lane_mask.shape[-1]))
        return selected, selected_mask

    def _features(self, obs: dict[str, Tensor]) -> tuple[tuple[Any, ...], Tensor, tuple[int, int]]:
        training_mask, b_idx, a_idx = self._select_queries(obs)
        b, a, h, _ = obs["obj_trajs"].shape
        if b_idx.numel() == 0:
            empty = obs["obj_trajs"].new_zeros(0)
            return (empty, empty, empty, empty, empty, empty, None), training_mask, (b, a)

        obj = obs["obj_trajs"].float()
        valid = obs["obj_trajs_mask"].bool()
        positions = obj[..., 0:2]
        velocities = obj[..., 8:10]
        headings = torch.atan2(obj[..., 6], obj[..., 7])
        sizes = obj[..., 3:5]

        base_vel = velocities[b_idx, a_idx, -1]
        base_ori = obs["obj_headings"].float()[b_idx, a_idx]
        vel_yaw = torch.atan2(base_vel[:, 1], base_vel[:, 0])
        base_heading = torch.where(base_vel.norm(dim=-1) < 0.5, base_ori, vel_yaw)
        base_pos = obs["obj_positions"].float()[b_idx, a_idx]

        topk_idx = self._select_nearby_agents(positions, valid, b_idx, a_idx, base_heading)
        n, k = topk_idx.shape
        gather_idx_4 = topk_idx.view(n, k, 1, 1).expand(-1, -1, h, 2)
        pos_n = positions[b_idx].gather(1, gather_idx_4)
        vel_n = velocities[b_idx].gather(1, gather_idx_4)
        size_n = sizes[b_idx].gather(1, gather_idx_4)
        heading_n = headings[b_idx].gather(1, topk_idx.view(n, k, 1).expand(-1, -1, h))
        mask_n = valid[b_idx].gather(1, topk_idx.view(n, k, 1).expand(-1, -1, h))

        base_pos_h = base_pos[:, None, None]
        base_heading_h = base_heading[:, None, None]
        pos_n = _rotate_to_local(pos_n - base_pos_h, base_heading_h)
        vel_n = _rotate_to_local(vel_n, base_heading_h)
        heading_n = wrap_angle(heading_n - base_heading_h)
        persp_idx = (topk_idx == a_idx[:, None]).int().argmax(dim=1)
        pos_n = _swap_ego_index(pos_n, persp_idx)
        vel_n = _swap_ego_index(vel_n, persp_idx)
        size_n = _swap_ego_index(size_n, persp_idx)
        heading_n = _swap_ego_index(heading_n, persp_idx)
        mask_n = _swap_ego_index(mask_n, persp_idx)

        dist = pos_n.norm(dim=-1)
        mask_n = mask_n & (dist <= self.agent_filter_radius)

        polygons = _vehicle_to_polygon(size_n[..., 0], size_n[..., 1], pos_n, heading_n)
        agent_features = torch.cat((pos_n, vel_n, heading_n.unsqueeze(-1), size_n, polygons.flatten(-2)), dim=-1)

        goal = obs.get("goal_positions", obs["obj_positions"]).float()
        goal_n = goal[b_idx].gather(1, topk_idx[..., None].expand(-1, -1, 2))
        goal_n = _rotate_to_local(goal_n - base_pos[:, None], base_heading[:, None])
        goal_n = _swap_ego_index(goal_n, persp_idx)
        if self.no_goal_allowed:
            goal_reached = torch.zeros(n, k, 1, dtype=torch.bool, device=obj.device)
            goal_features = torch.cat((goal_n, goal_reached.to(goal_n.dtype)), dim=-1)
        else:
            goal_features = goal_n

        steering = obs.get("steering", obj.new_zeros(b, a)).float()[b_idx].gather(1, topk_idx)
        accel = obs.get("acceleration", obj.new_zeros(b, a)).float()[b_idx].gather(1, topk_idx)
        yaw_rate = obs.get("yaw_rate", obj.new_zeros(b, a)).float()[b_idx].gather(1, topk_idx)
        steering = _swap_ego_index(steering[..., None], persp_idx).squeeze(-1)
        accel = _swap_ego_index(accel[..., None], persp_idx).squeeze(-1)
        yaw_rate = _swap_ego_index(yaw_rate[..., None], persp_idx).squeeze(-1)
        kin = torch.cat((vel_n[:, :, -1], heading_n[:, :, -1:], size_n[:, :, -1], steering[..., None], accel[..., None], yaw_rate[..., None]), dim=-1)

        lane_features, lane_mask = self._lane_features(obs, b_idx, base_pos, base_heading)
        lane_features, lane_mask = self._select_lanes(lane_features, lane_mask)

        agent_features = agent_features * mask_n.unsqueeze(-1).to(agent_features.dtype)
        goal_features = goal_features * mask_n[:, :, -1:].to(goal_features.dtype)
        return (
            agent_features,
            goal_features,
            kin,
            mask_n,
            lane_features,
            lane_mask,
            None,
        ), training_mask, (b, a)

    def _lane_features(self, obs: dict[str, Tensor], b_idx: Tensor, base_pos: Tensor, base_heading: Tensor) -> tuple[Tensor, Tensor]:
        polys = obs["map_polylines"].float()[b_idx]
        point_mask = obs["map_polylines_mask"].bool()[b_idx]
        poly_mask = obs["map_mask"].bool()[b_idx]

        n, m, p, _ = polys.shape
        if p == 0:
            features = polys.new_zeros(n, m, 0, 6)
            mask = torch.zeros(n, m, 0, dtype=torch.bool, device=polys.device)
            return features, mask

        valid_count = point_mask.sum(dim=-1)
        lane_mask = (valid_count > 0) & poly_mask

        local_xy = _rotate_to_local(polys[..., 0:2] - base_pos[:, None, None], base_heading[:, None, None])
        local_dir = _rotate_to_local(polys[..., 3:5], base_heading[:, None, None])
        dist = local_xy.norm(dim=-1).masked_fill(~point_mask, torch.inf)
        min_dist = dist.amin(dim=-1)
        lane_mask = lane_mask & (min_dist <= self.agent_filter_radius)

        raw_type = polys[..., 6:7]
        speed = torch.zeros_like(raw_type)
        features = torch.cat((local_xy, local_dir, raw_type, speed), dim=-1)
        point_mask = point_mask & lane_mask.unsqueeze(-1)
        features = features * point_mask.unsqueeze(-1).to(features.dtype)
        return features, point_mask

    def forward(
        self,
        obs: dict[str, Tensor],
        action: Tensor | None = None,
        sampling_method: str | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        features, training_mask, (b, a) = self._features(obs)
        out_action = obs["obj_trajs"].new_zeros(b, a, 2)
        out_logprob = obs["obj_trajs"].new_zeros(b, a)
        out_entropy = obs["obj_trajs"].new_zeros(b, a)
        out_value = obs["obj_trajs"].new_zeros(b, a)
        if not training_mask.any():
            return out_action, out_logprob, out_entropy, out_value

        params, values = self.model(*features)
        params = params.squeeze(1)
        values = values.squeeze(1)
        alpha, beta = params.chunk(2, dim=-1)
        dist = Beta(alpha.clamp_min(1e-2), beta.clamp_min(1e-2))
        low = self.action_low.to(device=params.device, dtype=params.dtype)
        high = self.action_high.to(device=params.device, dtype=params.dtype)
        scale = high - low

        if action is None:
            method = sampling_method or "sample"
            if method in {"mean", "deterministic", "expectation"}:
                unit_action = alpha / (alpha + beta)
            elif method in {"mode", "argmax"}:
                unit_action = (alpha - 1.0) / (alpha + beta - 2.0).clamp_min(1e-6)
            else:
                unit_action = dist.sample()
            unit_action = unit_action.clamp(1e-6, 1.0 - 1e-6)
            packed_action = low + unit_action * scale
        else:
            packed_action = action.to(device=params.device, dtype=params.dtype)[training_mask]
            unit_action = ((packed_action - low) / scale).clamp(1e-6, 1.0 - 1e-6)

        logprob = dist.log_prob(unit_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        out_action[training_mask] = packed_action.to(out_action.dtype)
        out_logprob[training_mask] = logprob.to(out_logprob.dtype)
        out_entropy[training_mask] = entropy.to(out_entropy.dtype)
        out_value[training_mask] = values.to(out_value.dtype)
        return out_action, out_logprob, out_entropy, out_value
