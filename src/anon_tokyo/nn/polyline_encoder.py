"""PointNet-style polyline encoder (MLP + MaxPool).

Faithfully follows MTR's ``PointNetPolylineEncoder``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anon_tokyo.nn.layers import build_mlps


class PointNetPolylineEncoder(nn.Module):
    """Encode variable-length polylines into fixed-size feature vectors.

    Architecture: pre_mlps → maxpool → concat → mlps → maxpool → out_mlps.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_pre_layers: int = 1,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.pre_mlps = build_mlps(in_channels, [hidden_dim] * num_pre_layers)
        self.mlps = build_mlps(hidden_dim * 2, [hidden_dim] * (num_layers - num_pre_layers))
        self.out_mlps = (
            build_mlps(hidden_dim, [hidden_dim, out_channels], ret_before_act=True, use_norm=False)
            if out_channels is not None
            else None
        )

    def forward(self, polylines: torch.Tensor, polylines_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            polylines: ``[B, N, P, C]`` — polyline point features.
            polylines_mask: ``[B, N, P]`` — bool mask (True = valid).

        Returns:
            ``[B, N, D]`` — per-polyline feature.
        """
        B, N, P, C = polylines.shape
        mask = polylines_mask.bool()

        # Pre-MLP (only on valid points to avoid BN on zeros)
        feat_valid = self.pre_mlps(polylines[mask])  # (num_valid, hidden)
        feat = polylines.new_zeros(B, N, P, feat_valid.shape[-1], dtype=feat_valid.dtype)
        feat[mask] = feat_valid

        # Global pooling per polyline → concat with point features
        pooled = feat.max(dim=2).values  # [B, N, hidden]
        feat = torch.cat([feat, pooled[:, :, None, :].expand(-1, -1, P, -1)], dim=-1)

        # Second MLP stage
        feat_valid = self.mlps(feat[mask])
        buf = feat.new_zeros(B, N, P, feat_valid.shape[-1], dtype=feat_valid.dtype)
        buf[mask] = feat_valid

        # Final max pool
        out = buf.max(dim=2).values  # [B, N, hidden]

        if self.out_mlps is not None:
            valid_poly = mask.any(dim=-1)  # [B, N]
            out_valid = self.out_mlps(out[valid_poly])
            result = out.new_zeros(B, N, out_valid.shape[-1], dtype=out_valid.dtype)
            result[valid_poly] = out_valid
            return result

        return out
