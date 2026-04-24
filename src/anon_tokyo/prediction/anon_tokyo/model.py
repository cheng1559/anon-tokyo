"""AnonTokyo: scene-centric prediction model (encoder + decoder).

No per-agent coordinate transform — the encoder uses RoPE/DRoPE
for spatial and rotational encoding directly in the global frame.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from anon_tokyo.prediction.anon_tokyo.encoder import AnonTokyoEncoder
from anon_tokyo.prediction.anon_tokyo.decoder import AnonTokyoDecoder


class AnonTokyoModel(nn.Module):
    """Full AnonTokyo scene-centric prediction model."""

    def __init__(
        self,
        d_model: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_decoder: int = 512,
        map_d_model: int | None = None,
        num_heads: int = 8,
        sparse_k: int = 32,
        use_local_attn: bool = True,
        num_attn_neighbors: int | None = None,
        num_modes: int = 6,
        num_intention_queries: int = 64,
        num_future_frames: int = 80,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_drope: bool = True,
        position_encoding: str | None = None,
        intention_points_file: str = "data/intention_points.pkl",
        nms_dist_thresh: float = 2.5,
        center_offset_of_map: tuple[float, float] = (30.0, 0.0),
        num_base_map_polylines: int = 256,
        num_waypoint_map_polylines: int = 128,
    ) -> None:
        super().__init__()
        if map_d_model is None:
            map_d_model = d_model
        if num_attn_neighbors is not None:
            sparse_k = num_attn_neighbors
        if not use_local_attn:
            raise ValueError("AnonTokyo encoder/decoder currently requires sparse local attention.")

        self.encoder = AnonTokyoEncoder(
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            sparse_k=sparse_k,
            dropout=dropout,
            use_rope=use_rope,
            use_drope=use_drope,
            position_encoding=position_encoding,
        )
        self.decoder = AnonTokyoDecoder(
            in_channels=d_model,
            d_model=d_decoder,
            map_d_model=map_d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_future_frames=num_future_frames,
            num_motion_modes=num_modes,
            num_intention_queries=num_intention_queries,
            intention_points_file=intention_points_file,
            nms_dist_thresh=nms_dist_thresh,
            center_offset_of_map=center_offset_of_map,
            num_base_map_polylines=num_base_map_polylines,
            num_waypoint_map_polylines=num_waypoint_map_polylines,
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Input: scene-centric batch ``[B, ...]`` from dataloader.

        Returns:
            ``pred_trajs``  ``[B, K, num_modes_or_queries, T, 7]``
            ``pred_scores`` ``[B, K, num_modes_or_queries]``
        """
        enc_out = self.encoder(batch)
        dec_out = self.decoder(enc_out, batch)
        return dec_out
