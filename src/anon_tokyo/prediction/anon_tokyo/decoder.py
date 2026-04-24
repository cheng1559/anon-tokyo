"""AnonTokyo: scene-centric decoder wrapping MTR's intention-query decoder.

Converts scene-centric encoder output ``[B, A, D]`` to agent-centric
format ``[B*K, ...]`` expected by the MTR decoder, runs the decoder,
and reshapes output back to ``[B, K, ...]``.

The core decoder architecture is identical to MTR for fair ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from anon_tokyo.prediction.mtr.decoder import MTRDecoder


class AnonTokyoDecoder(nn.Module):
    """Scene-centric → MTR decoder → scene-centric output adapter.

    Handles the gathering of tracked-agent features and expansion of
    scene-level context so the MTR decoder can run in its native format.
    """

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
        self.mtr_decoder = MTRDecoder(
            in_channels=in_channels,
            d_model=d_model,
            map_d_model=map_d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_future_frames=num_future_frames,
            num_motion_modes=num_motion_modes,
            num_intention_queries=num_intention_queries,
            intention_points_file=intention_points_file,
            nms_dist_thresh=nms_dist_thresh,
            keep_query_pos_all=True,
        )

    def forward(
        self,
        enc_out: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """
        Args:
            enc_out: Scene-centric encoder output:
                ``obj_feature``  [B, A, D_enc]
                ``map_feature``  [B, M, D_enc]
                ``obj_mask``     [B, A]  bool
                ``map_mask``     [B, M]  bool
                ``obj_pos``      [B, A, 2]
                ``map_pos``      [B, M, 2]
            batch: Collated batch with:
                ``tracks_to_predict``  [B, K_max]  int, -1 padded
                ``obj_types``          [B, A]      int

        Returns:
            ``pred_trajs``:  [B, K, num_modes, T, 7]  or  [B, K, M_out, T, 7]
            ``pred_scores``: [B, K, num_modes]         or  [B, K, M_out]
        """
        obj_feat = enc_out["obj_feature"]  # [B, A, D]
        map_feat = enc_out["map_feature"]  # [B, M, D]
        obj_mask = enc_out["obj_mask"]  # [B, A]
        map_mask = enc_out["map_mask"]  # [B, M]
        obj_pos = enc_out["obj_pos"]  # [B, A, 2]
        map_pos = enc_out["map_pos"]  # [B, M, 2]

        B, A, D = obj_feat.shape
        M = map_feat.shape[1]
        device = obj_feat.device

        ttp = batch["tracks_to_predict"]  # [B, K_max]
        K = ttp.shape[1]
        ttp_clamped = ttp.clamp(min=0)

        # Gather center object features: [B, K, D]
        b_idx = torch.arange(B, device=device)[:, None].expand(B, K)
        center_feat = obj_feat[b_idx, ttp_clamped]  # [B, K, D]

        # Gather center object types: [B, K]
        center_obj_type = batch["obj_types"][b_idx, ttp_clamped]  # [B, K]

        # Expand scene-level features per tracked agent → [B*K, ...]
        # Each tracked agent in the same scene shares the same context
        obj_feat_exp = obj_feat[:, None, :, :].expand(B, K, A, D).reshape(B * K, A, D)
        map_feat_exp = map_feat[:, None, :, :].expand(B, K, M, map_feat.shape[-1]).reshape(B * K, M, map_feat.shape[-1])
        obj_mask_exp = obj_mask[:, None, :].expand(B, K, A).reshape(B * K, A)
        map_mask_exp = map_mask[:, None, :].expand(B, K, M).reshape(B * K, M)
        obj_pos_exp = obj_pos[:, None, :, :].expand(B, K, A, 2).reshape(B * K, A, 2)
        map_pos_exp = map_pos[:, None, :, :].expand(B, K, M, 2).reshape(B * K, M, 2)
        center_feat_flat = center_feat.reshape(B * K, D)
        center_type_flat = center_obj_type.reshape(B * K)

        # Build agent-centric enc_out for MTR decoder
        ac_enc_out = {
            "obj_feature": obj_feat_exp,
            "map_feature": map_feat_exp,
            "center_objects_feature": center_feat_flat,
            "obj_mask": obj_mask_exp,
            "map_mask": map_mask_exp,
            "obj_pos": obj_pos_exp,
            "map_pos": map_pos_exp,
        }
        ac_batch = {"center_obj_type": center_type_flat}

        # Run MTR decoder
        dec_out = self.mtr_decoder(ac_enc_out, ac_batch)

        # Reshape outputs back to [B, K, ...]
        result: dict[str, Tensor] = {}

        pred_trajs = dec_out["pred_trajs"]  # [B*K, modes, T, 7]
        pred_scores = dec_out["pred_scores"]  # [B*K, modes]
        modes = pred_trajs.shape[1]
        T = pred_trajs.shape[2]

        result["pred_trajs"] = pred_trajs.reshape(B, K, modes, T, 7)
        result["pred_scores"] = pred_scores.reshape(B, K, modes)

        # Pass through per-layer predictions for per-layer loss (avoids DDP unused params)
        if "pred_list" in dec_out:
            reshaped_list: list[tuple[Tensor, Tensor]] = []
            for scores_i, trajs_i in dec_out["pred_list"]:
                Q_i = scores_i.shape[1]
                T_i = trajs_i.shape[2]
                reshaped_list.append(
                    (
                        scores_i.reshape(B, K, Q_i),
                        trajs_i.reshape(B, K, Q_i, T_i, 7),
                    )
                )
            result["pred_list"] = reshaped_list

        # Pass through dense future prediction for auxiliary loss
        if "pred_dense_trajs" in dec_out:
            dense = dec_out["pred_dense_trajs"]  # [B*K, A, T, 7]
            result["pred_dense_trajs"] = dense.reshape(B, K, A, dense.shape[2], 7)

        return result
