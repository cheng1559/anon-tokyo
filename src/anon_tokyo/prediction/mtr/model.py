"""Official-compatible MTR model assembly."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from anon_tokyo.prediction.mtr.preprocessing import agent_centric_preprocess
from anon_tokyo.prediction.mtr.encoder import MTREncoder
from anon_tokyo.prediction.mtr.decoder import MTRDecoder


class MTRModel(nn.Module):
    """Full MTR prediction model with official checkpoint-compatible keys."""

    def __init__(
        self,
        d_model: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_decoder: int = 512,
        map_d_model: int = 256,
        num_heads: int = 8,
        num_modes: int = 6,
        num_intention_queries: int = 64,
        num_future_frames: int = 80,
        dropout: float = 0.1,
        intention_points_file: str = "data/intention_points.pkl",
        nms_dist_thresh: float = 2.5,
        use_local_attn: bool = True,
        num_attn_neighbors: int = 16,
        center_offset_of_map: tuple[float, float] = (30.0, 0.0),
        num_base_map_polylines: int = 256,
        num_waypoint_map_polylines: int = 128,
    ) -> None:
        super().__init__()
        self.context_encoder = MTREncoder(
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_local_attn=use_local_attn,
            num_attn_neighbors=num_attn_neighbors,
        )
        self.motion_decoder = MTRDecoder(
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

    @property
    def encoder(self) -> MTREncoder:
        """Backward-compatible alias for older project code."""
        return self.context_encoder

    @property
    def decoder(self) -> MTRDecoder:
        """Backward-compatible alias for older project code."""
        return self.motion_decoder

    def load_official_checkpoint(self, path: str, strict: bool = True, map_location: str | torch.device = "cpu") -> None:
        """Load an official MTR ``.pth`` checkpoint directly.

        Official checkpoints store weights under ``checkpoint["model_state"]``.
        This model intentionally uses the same top-level module names
        (``context_encoder`` and ``motion_decoder``), so strict loading should
        work when hyperparameters match the official config.
        """
        checkpoint = torch.load(path, map_location=map_location)
        state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        self.load_state_dict(state, strict=strict)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Input: scene-centric batch ``[B, ...]`` from dataloader.

        Returns dict matching ``PredictionModel`` protocol:
            ``pred_trajs``  ``[K_total, N, T, 7]``
            ``pred_scores`` ``[K_total, N]``
        plus training-only extras (pred_list, pred_dense_trajs, etc.)
        and metadata (batch_idx, batch_sample_count, track_index_to_predict).
        """
        if "input_dict" in batch:
            input_dict = batch["input_dict"]
            center_types = input_dict["center_objects_type"]
            if isinstance(center_types, Tensor):
                center_type_tensor = center_types.long()
            else:
                center_type_tensor = torch.tensor(
                    [
                        {"TYPE_VEHICLE": 1, "TYPE_PEDESTRIAN": 2, "TYPE_CYCLIST": 3}.get(str(x), int(x) if str(x).isdigit() else 1)
                        for x in center_types
                    ],
                    device=input_dict["obj_trajs"].device,
                    dtype=torch.long,
                )
            ac_batch = {
                "obj_trajs": input_dict["obj_trajs"],
                "obj_trajs_mask": input_dict["obj_trajs_mask"],
                "map_polylines": input_dict["map_polylines"],
                "map_polylines_mask": input_dict["map_polylines_mask"],
                "map_polylines_center": input_dict["map_polylines_center"],
                "track_index_to_predict": input_dict["track_index_to_predict"].long(),
                "center_objects_type": center_type_tensor,
                "center_obj_type": center_type_tensor,
                "obj_types": input_dict["obj_types"].long(),
                "obj_trajs_last_pos": input_dict["obj_trajs_last_pos"],
                "sdc_track_index": input_dict.get("sdc_track_index", torch.zeros_like(input_dict["track_index_to_predict"])),
                "obj_trajs_future": input_dict["obj_trajs_future_state"],
                "obj_trajs_future_mask": input_dict["obj_trajs_future_mask"],
                "batch_sample_count": batch["batch_sample_count"].to(input_dict["obj_trajs"].device),
                "batch_idx": torch.repeat_interleave(
                    torch.arange(batch["batch_size"], device=input_dict["obj_trajs"].device),
                    batch["batch_sample_count"].to(input_dict["obj_trajs"].device),
                ),
                "center_gt_trajs": input_dict["center_gt_trajs"],
                "center_gt_mask": input_dict["center_gt_trajs_mask"],
            }
        else:
            # Backward-compatible path for scene-centric batches and tests.
            ac_batch = agent_centric_preprocess(batch)

        if ac_batch["track_index_to_predict"].numel() == 0:
            raise ValueError("MTRModel received a batch with no tracks_to_predict")

        enc_out = self.context_encoder(ac_batch)
        if enc_out["center_objects_feature"].ndim == 3:
            enc_out["center_objects_feature"] = enc_out["center_objects_feature"].squeeze(1)
        dec_out = self.motion_decoder(enc_out, ac_batch)

        # Pass through metadata
        dec_out["batch_idx"] = ac_batch["batch_idx"]
        dec_out["batch_sample_count"] = ac_batch["batch_sample_count"]
        dec_out["track_index_to_predict"] = ac_batch["track_index_to_predict"]
        dec_out["center_obj_type"] = ac_batch["center_obj_type"]
        dec_out["center_objects_type"] = ac_batch["center_obj_type"]

        # Pass through GT for loss computation
        tidx = ac_batch["track_index_to_predict"]  # [K_total]
        k_idx = torch.arange(tidx.shape[0], device=tidx.device)
        if "center_gt_trajs" in ac_batch:
            dec_out["center_gt_trajs"] = ac_batch["center_gt_trajs"]
            dec_out["center_gt_mask"] = ac_batch["center_gt_mask"]
        else:
            dec_out["center_gt_trajs"] = ac_batch["obj_trajs_future"][k_idx, tidx]
            dec_out["center_gt_mask"] = ac_batch["obj_trajs_future_mask"][k_idx, tidx]
        dec_out["obj_trajs_future"] = ac_batch["obj_trajs_future"]
        dec_out["obj_trajs_future_mask"] = ac_batch["obj_trajs_future_mask"]

        return dec_out
