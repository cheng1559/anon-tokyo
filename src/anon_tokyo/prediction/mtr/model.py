"""MTR: full model assembly (encoder + decoder).

Scene-centric batch → per-agent-centric preprocessing → encoder → decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from anon_tokyo.prediction.mtr.preprocessing import agent_centric_preprocess
from anon_tokyo.prediction.mtr.encoder import MTREncoder
from anon_tokyo.prediction.mtr.decoder import MTRDecoder


class MTRModel(nn.Module):
    """Full MTR prediction model."""

    def __init__(
        self,
        d_model: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_decoder: int = 512,
        num_heads: int = 8,
        num_modes: int = 6,
        num_intention_queries: int = 64,
        num_future_frames: int = 80,
        dropout: float = 0.1,
        intention_points_file: str = "data/intention_points.pkl",
        nms_dist_thresh: float = 2.5,
    ) -> None:
        super().__init__()
        self.encoder = MTREncoder(
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = MTRDecoder(
            in_channels=d_model,
            d_model=d_decoder,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_future_frames=num_future_frames,
            num_motion_modes=num_modes,
            num_intention_queries=num_intention_queries,
            intention_points_file=intention_points_file,
            nms_dist_thresh=nms_dist_thresh,
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Input: scene-centric batch ``[B, ...]`` from dataloader.

        Returns dict matching ``PredictionModel`` protocol:
            ``pred_trajs``  ``[K_total, N, T, 7]``
            ``pred_scores`` ``[K_total, N]``
        plus training-only extras (pred_list, pred_dense_trajs, etc.)
        and metadata (batch_idx, batch_sample_count, track_index_to_predict).
        """
        # Per-agent ego-centric transform
        ac_batch = agent_centric_preprocess(batch)

        # Encode
        enc_out = self.encoder(ac_batch)

        # For decoder: center_objects_feature should be [K_total, D]
        # Encoder returns [K_total, 1, D] because tracks_to_predict is [K_total, 1]
        center_feat = enc_out["center_objects_feature"].squeeze(1)
        enc_out["center_objects_feature"] = center_feat

        # Decode
        dec_out = self.decoder(enc_out, ac_batch)

        # Pass through metadata
        dec_out["batch_idx"] = ac_batch["batch_idx"]
        dec_out["batch_sample_count"] = ac_batch["batch_sample_count"]
        dec_out["track_index_to_predict"] = ac_batch["track_index_to_predict"]
        dec_out["center_obj_type"] = ac_batch["center_obj_type"]

        # Pass through GT for loss computation
        tidx = ac_batch["track_index_to_predict"]  # [K_total]
        k_idx = torch.arange(tidx.shape[0], device=tidx.device)
        dec_out["center_gt_trajs"] = ac_batch["obj_trajs_future"][k_idx, tidx]  # [K, T, 4]
        dec_out["center_gt_mask"] = ac_batch["obj_trajs_future_mask"][k_idx, tidx]  # [K, T]
        dec_out["obj_trajs_future"] = ac_batch["obj_trajs_future"]
        dec_out["obj_trajs_future_mask"] = ac_batch["obj_trajs_future_mask"]

        return dec_out
