"""Scene-centric AnonTokyo decoder with MTR-style motion heads.

The encoder sees one scene batch ``[B, A/M, D]``.  The motion decoder only
creates intention queries for ``tracks_to_predict`` agents to avoid the
``128 agents * 64 queries`` OOM path, while still attending to the full scene.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from anon_tokyo.prediction.mtr.decoder import MTRDecoder


def _local_xy_to_scene(local_xy: Tensor, agent_pos: Tensor, agent_heading: Tensor) -> Tensor:
    """Transform agent-local xy to the scene frame.

    Args:
        local_xy: ``[B, A, ..., 2]``.
        agent_pos: ``[B, A, 2]`` scene-frame current positions.
        agent_heading: ``[B, A]`` scene-frame headings.
    """
    extra_dims = local_xy.ndim - agent_heading.ndim - 1
    view_shape = (*agent_heading.shape, *([1] * extra_dims))
    c = agent_heading.cos().view(view_shape)
    s = agent_heading.sin().view(view_shape)
    pos = agent_pos.view(*agent_pos.shape[:2], *([1] * extra_dims), 2)
    x = local_xy[..., 0]
    y = local_xy[..., 1]
    scene_x = x * c - y * s + pos[..., 0]
    scene_y = x * s + y * c + pos[..., 1]
    return torch.stack((scene_x, scene_y), dim=-1)


class AnonTokyoDecoder(MTRDecoder):
    """Target-agent scene decoder that reuses MTR-compatible decoder blocks."""

    def _get_target_motion_query(self, obj_types: Tensor) -> tuple[Tensor, Tensor]:
        B, K = obj_types.shape
        query_embed, intention_points = self.get_motion_query(obj_types.reshape(-1))
        query_embed = query_embed.permute(1, 0, 2).contiguous().view(B, K, self.num_queries, self.d_model)
        query_embed = query_embed.reshape(B, K * self.num_queries, self.d_model).permute(1, 0, 2).contiguous()
        intention_points = intention_points.permute(1, 0, 2).contiguous().view(B, K, self.num_queries, 2)
        return query_embed, intention_points

    def _collect_map_idxs_scene(
        self,
        map_pos: Tensor,
        map_mask: Tensor,
        pred_waypoints_scene: Tensor,
        base_points_scene: Tensor,
    ) -> Tensor:
        """Collect base and dynamic map polylines per query in scene frame."""
        B, num_query, _ = base_points_scene.shape
        num_polylines = map_pos.shape[1]
        masked_map_pos = map_pos.masked_fill(~map_mask[..., None], 10000000.0)

        base_dist = (base_points_scene[:, :, None, :] - masked_map_pos[:, None, :, 0:2]).norm(dim=-1)
        base_k = min(num_polylines, self.num_base_map_polylines)
        base_topk_dist, base_idxs = base_dist.topk(k=base_k, dim=-1, largest=False)
        base_idxs[base_topk_dist > 10000000] = -1
        if base_k < self.num_base_map_polylines:
            base_idxs = F.pad(base_idxs, (0, self.num_base_map_polylines - base_k), value=-1)

        dynamic_dist = (
            pred_waypoints_scene[:, :, None, :, 0:2] - masked_map_pos[:, None, :, None, 0:2]
        ).norm(dim=-1).min(dim=-1).values
        dyn_k = min(num_polylines, self.num_waypoint_map_polylines)
        dynamic_topk_dist, dynamic_idxs = dynamic_dist.topk(k=dyn_k, dim=-1, largest=False)
        dynamic_idxs[dynamic_topk_dist > 10000000] = -1
        if dyn_k < self.num_waypoint_map_polylines:
            dynamic_idxs = F.pad(dynamic_idxs, (0, self.num_waypoint_map_polylines - dyn_k), value=-1)

        collected = torch.cat((base_idxs, dynamic_idxs), dim=-1)
        sorted_idxs = collected.sort(dim=-1).values
        unique_mask = torch.ones_like(sorted_idxs, dtype=torch.bool)
        unique_mask[..., 1:] = sorted_idxs[..., 1:] != sorted_idxs[..., :-1]
        sorted_idxs = sorted_idxs.masked_fill(~unique_mask, -1)
        return sorted_idxs.int()

    def apply_dense_future_prediction(self, obj_feature: Tensor, obj_mask: Tensor, obj_pos: Tensor) -> tuple[Tensor, Tensor]:
        B, A, _ = obj_feature.shape
        valid = obj_mask.bool()
        obj_pos_valid = obj_pos[valid][..., 0:2]
        obj_feature_valid = obj_feature[valid]
        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)
        pred_dense = self.dense_future_head(torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1))
        pred_dense = pred_dense.view(pred_dense.shape[0], self.num_future_frames, 7)
        future_input = pred_dense[:, :, [0, 1, -2, -1]].flatten(1, 2)
        future_feature = self.future_traj_mlps(future_input)
        obj_feature_valid = self.traj_fusion_mlps(torch.cat((obj_feature_valid, future_feature), dim=-1))

        ret_feature = torch.zeros_like(obj_feature)
        ret_feature[valid] = obj_feature_valid.to(dtype=ret_feature.dtype)
        ret_dense = obj_feature.new_zeros(B, A, self.num_future_frames, 7)
        ret_dense[valid] = pred_dense.to(dtype=ret_dense.dtype)
        self.forward_ret_dict["pred_dense_trajs"] = ret_dense
        return ret_feature, ret_dense

    def apply_transformer_decoder(
        self,
        center_objects_feature: Tensor,
        center_objects_type: Tensor,
        center_objects_pos: Tensor,
        center_objects_heading: Tensor,
        obj_feature: Tensor,
        obj_mask: Tensor,
        obj_pos: Tensor,
        map_feature: Tensor,
        map_mask: Tensor,
        map_pos: Tensor,
    ) -> list[list[Tensor]]:
        B, K, _ = center_objects_feature.shape
        num_query_total = K * self.num_queries

        intention_query, intention_points_local = self._get_target_motion_query(center_objects_type)
        query_content = torch.zeros_like(intention_query)
        self.forward_ret_dict["intention_points"] = intention_points_local

        center_feature = center_objects_feature[:, :, None, :].expand(B, K, self.num_queries, -1)
        center_feature = center_feature.reshape(B, num_query_total, -1).permute(1, 0, 2).contiguous()

        intention_points_scene = _local_xy_to_scene(intention_points_local, center_objects_pos, center_objects_heading)
        base_forward = torch.tensor(self.center_offset_of_map, device=obj_feature.device, dtype=obj_feature.dtype)
        base_points_scene = _local_xy_to_scene(
            base_forward.view(1, 1, 1, 2).expand(B, K, 1, 2),
            center_objects_pos,
            center_objects_heading,
        ).squeeze(2)
        base_points_scene = base_points_scene[:, :, None, :].expand(B, K, self.num_queries, 2).reshape(B, num_query_total, 2)

        pred_waypoints_scene = intention_points_scene.reshape(B, num_query_total, 1, 2)
        dynamic_query_center = intention_points_scene.reshape(B, num_query_total, 2).permute(1, 0, 2).contiguous()

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
            collected_idxs = self._collect_map_idxs_scene(
                map_pos,
                map_mask,
                pred_waypoints_scene,
                base_points_scene,
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
            query_feature = torch.cat([center_feature, obj_query_feature, map_query_feature], dim=-1)
            query_content = self.query_feature_fusion_layers[layer_idx](query_feature.flatten(0, 1)).view(
                num_query_total, B, -1
            )

            query_content_t = query_content.permute(1, 0, 2).contiguous().view(B * K * self.num_queries, -1)
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(B, K, self.num_queries)
            pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(
                B, K, self.num_queries, self.num_future_frames, 7
            )
            pred_list.append([pred_scores, pred_trajs])

            pred_waypoints_scene = _local_xy_to_scene(
                pred_trajs[..., 0:2].flatten(2, 3),
                center_objects_pos,
                center_objects_heading,
            ).view(B, K * self.num_queries, self.num_future_frames, 2)
            dynamic_query_center = pred_waypoints_scene[:, :, -1].permute(1, 0, 2).contiguous()

        return pred_list

    def generate_final_prediction(self, pred_list: list[list[Tensor]]) -> tuple[Tensor, Tensor]:
        pred_scores, pred_trajs = pred_list[-1]
        B, K, Q, T, C = pred_trajs.shape
        pred_scores = torch.softmax(pred_scores, dim=-1)
        if self.num_motion_modes != Q:
            pred_trajs_flat, pred_scores_flat, _ = self.batch_nms(
                pred_trajs.reshape(B * K, Q, T, C),
                pred_scores.reshape(B * K, Q),
            )
            pred_trajs = pred_trajs_flat.view(B, K, self.num_motion_modes, T, C)
            pred_scores = pred_scores_flat.view(B, K, self.num_motion_modes)
        return pred_scores, pred_trajs

    @staticmethod
    def _scatter_target_predictions(
        pred_list_valid: list[list[Tensor]],
        pred_scores_valid: Tensor,
        pred_trajs_valid: Tensor,
        intention_points_valid: Tensor,
        valid_b: Tensor,
        valid_k: Tensor,
        batch_size: int,
        num_targets: int,
    ) -> tuple[list[tuple[Tensor, Tensor]], Tensor, Tensor, Tensor]:
        pred_list: list[tuple[Tensor, Tensor]] = []
        for scores_i, trajs_i in pred_list_valid:
            q = scores_i.shape[2]
            t = trajs_i.shape[3]
            scores = scores_i.new_zeros(batch_size, num_targets, q)
            trajs = trajs_i.new_zeros(batch_size, num_targets, q, t, trajs_i.shape[-1])
            scores[valid_b, valid_k] = scores_i[:, 0]
            trajs[valid_b, valid_k] = trajs_i[:, 0]
            pred_list.append((scores, trajs))

        m = pred_scores_valid.shape[2]
        t = pred_trajs_valid.shape[3]
        pred_scores = pred_scores_valid.new_zeros(batch_size, num_targets, m)
        pred_trajs = pred_trajs_valid.new_zeros(batch_size, num_targets, m, t, pred_trajs_valid.shape[-1])
        pred_scores[valid_b, valid_k] = pred_scores_valid[:, 0]
        pred_trajs[valid_b, valid_k] = pred_trajs_valid[:, 0]

        intention_points = intention_points_valid.new_zeros(batch_size, num_targets, intention_points_valid.shape[2], 2)
        intention_points[valid_b, valid_k] = intention_points_valid[:, 0]
        return pred_list, pred_scores, pred_trajs, intention_points

    def forward(self, enc_out: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        self.forward_ret_dict = {}

        obj_feature = enc_out["obj_feature"]
        obj_mask = enc_out["obj_mask"].bool()
        obj_pos = enc_out["obj_pos"]
        obj_heading = enc_out.get("obj_headings", batch.get("obj_headings"))
        if obj_heading is None:
            obj_heading = obj_pos.new_zeros(obj_pos.shape[:2])
        map_feature = enc_out["map_feature"]
        map_mask = enc_out["map_mask"].bool()
        map_pos = enc_out["map_pos"]

        B, A, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]

        center_feature = self.in_proj_center_obj(obj_feature.flatten(0, 1)).view(B, A, -1)
        obj_proj = self.in_proj_obj(obj_feature.flatten(0, 1)).view(B, A, -1)
        obj_proj = obj_proj * obj_mask.unsqueeze(-1).type_as(obj_proj)
        map_proj = self.in_proj_map(map_feature.flatten(0, 1)).view(B, num_polylines, -1)
        map_proj = map_proj * map_mask.unsqueeze(-1).type_as(map_proj)

        obj_proj, pred_dense = self.apply_dense_future_prediction(obj_proj, obj_mask, obj_pos)

        ttp = batch["tracks_to_predict"].long()
        K = ttp.shape[1]
        valid_b, valid_k = torch.nonzero(ttp >= 0, as_tuple=True)
        if valid_b.numel() == 0:
            valid_b = torch.zeros(1, dtype=torch.long, device=obj_feature.device)
            valid_k = torch.zeros(1, dtype=torch.long, device=obj_feature.device)
        valid_tidx = ttp[valid_b, valid_k].clamp(min=0)
        center_feature_valid = center_feature[valid_b, valid_tidx].unsqueeze(1)
        center_pos_valid = obj_pos[valid_b, valid_tidx].unsqueeze(1)
        center_heading_valid = obj_heading[valid_b, valid_tidx].unsqueeze(1)
        center_type_valid = batch["obj_types"].long()[valid_b, valid_tidx].unsqueeze(1)

        pred_list_valid = self.apply_transformer_decoder(
            center_feature_valid,
            center_type_valid,
            center_pos_valid,
            center_heading_valid,
            obj_proj[valid_b],
            obj_mask[valid_b],
            obj_pos[valid_b],
            map_proj[valid_b],
            map_mask[valid_b],
            map_pos[valid_b],
        )
        pred_scores_valid, pred_trajs_valid = self.generate_final_prediction(pred_list_valid)
        pred_list, pred_scores, pred_trajs, intention_points = self._scatter_target_predictions(
            pred_list_valid,
            pred_scores_valid,
            pred_trajs_valid,
            self.forward_ret_dict["intention_points"],
            valid_b,
            valid_k,
            B,
            K,
        )
        self.forward_ret_dict["pred_list"] = pred_list
        return {
            "pred_scores": pred_scores if not self.training else pred_list[-1][0],
            "pred_trajs": pred_trajs if not self.training else pred_list[-1][1],
            "pred_list": pred_list,
            "pred_dense_trajs": pred_dense,
            "intention_points": intention_points,
            "track_index_to_predict": ttp,
            "pred_is_target_agents": True,
        }
