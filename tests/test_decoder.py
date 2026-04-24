"""Unit tests for AnonTokyo query-centric decoder."""

from __future__ import annotations

import pickle
import tempfile

import pytest
import torch
import numpy as np

from anon_tokyo.prediction.anon_tokyo.decoder import AnonTokyoDecoder


@pytest.fixture
def intention_points_file(tmp_path):
    """Create a temporary intention points pickle file."""
    Q = 64
    pts_dict = {
        "TYPE_VEHICLE": np.random.randn(Q, 2).astype(np.float32),
        "TYPE_PEDESTRIAN": np.random.randn(Q, 2).astype(np.float32),
        "TYPE_CYCLIST": np.random.randn(Q, 2).astype(np.float32),
    }
    path = tmp_path / "intention_points.pkl"
    with open(path, "wb") as f:
        pickle.dump(pts_dict, f)
    return str(path)


def _make_enc_out(B: int = 2, A: int = 16, M: int = 32, D: int = 64) -> dict[str, torch.Tensor]:
    return {
        "obj_feature": torch.randn(B, A, D),
        "map_feature": torch.randn(B, M, D),
        "obj_mask": torch.ones(B, A, dtype=torch.bool),
        "map_mask": torch.ones(B, M, dtype=torch.bool),
        "obj_pos": torch.randn(B, A, 2) * 50,
        "map_pos": torch.randn(B, M, 2) * 100,
    }


def _make_batch(B: int = 2, A: int = 16, K: int = 4) -> dict[str, torch.Tensor]:
    ttp = torch.full((B, K), -1, dtype=torch.long)
    for b in range(B):
        n = min(K, A)
        ttp[b, :n] = torch.randperm(A)[:n]
    return {
        "tracks_to_predict": ttp,
        "obj_types": torch.randint(1, 4, (B, A)),
    }


@pytest.fixture
def decoder(intention_points_file) -> AnonTokyoDecoder:
    return AnonTokyoDecoder(
        in_channels=64,
        d_model=64,
        map_d_model=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        num_future_frames=20,
        num_motion_modes=6,
        num_intention_queries=64,
        intention_points_file=intention_points_file,
        nms_dist_thresh=2.5,
    )


class TestAnonTokyoDecoder:
    def test_output_keys(self, decoder: AnonTokyoDecoder):
        enc_out = _make_enc_out(B=2, A=8, M=16, D=64)
        batch = _make_batch(B=2, A=8, K=4)
        decoder.eval()
        out = decoder(enc_out, batch)
        assert "pred_trajs" in out
        assert "pred_scores" in out

    def test_output_shapes_eval(self, decoder: AnonTokyoDecoder):
        B, A, M, D, K = 2, 8, 16, 64, 4
        enc_out = _make_enc_out(B=B, A=A, M=M, D=D)
        batch = _make_batch(B=B, A=A, K=K)
        decoder.eval()
        out = decoder(enc_out, batch)
        # In eval mode, NMS reduces to num_motion_modes for target agents only.
        assert out["pred_trajs"].shape == (B, K, 6, 20, 7)
        assert out["pred_scores"].shape == (B, K, 6)

    def test_output_shapes_train(self, decoder: AnonTokyoDecoder):
        B, A, M, D, K = 2, 8, 16, 64, 4
        enc_out = _make_enc_out(B=B, A=A, M=M, D=D)
        batch = _make_batch(B=B, A=A, K=K)
        decoder.train()
        out = decoder(enc_out, batch)
        # In train mode, all Q=64 queries returned for target agents only.
        assert out["pred_trajs"].shape == (B, K, 64, 20, 7)
        assert out["pred_scores"].shape == (B, K, 64)

    def test_gradient_flow(self, decoder: AnonTokyoDecoder):
        B, A, M, D, K = 1, 4, 8, 64, 2
        enc_out = _make_enc_out(B=B, A=A, M=M, D=D)
        enc_out["obj_feature"].requires_grad_(True)
        batch = _make_batch(B=B, A=A, K=K)
        decoder.train()
        out = decoder(enc_out, batch)
        loss = out["pred_trajs"].sum() + out["pred_scores"].sum()
        loss.backward()
        assert enc_out["obj_feature"].grad is not None

    def test_single_tracked_agent(self, decoder: AnonTokyoDecoder):
        """Works with K=1 tracked agent."""
        B, A, M, D = 1, 4, 8, 64
        enc_out = _make_enc_out(B=B, A=A, M=M, D=D)
        batch = {"tracks_to_predict": torch.tensor([[0]]), "obj_types": torch.ones(B, A, dtype=torch.long)}
        decoder.eval()
        out = decoder(enc_out, batch)
        assert out["pred_trajs"].shape[0] == 1
        assert out["pred_trajs"].shape[1] == 1

    def test_padded_tracks(self, decoder: AnonTokyoDecoder):
        """Handles -1 padded tracks_to_predict."""
        B, A, M, D, K = 1, 8, 16, 64, 4
        enc_out = _make_enc_out(B=B, A=A, M=M, D=D)
        ttp = torch.tensor([[0, 2, -1, -1]])  # only 2 valid tracks, rest -1
        batch = {"tracks_to_predict": ttp, "obj_types": torch.ones(B, A, dtype=torch.long)}
        decoder.eval()
        out = decoder(enc_out, batch)
        assert out["pred_trajs"].shape == (B, K, 6, 20, 7)

    def test_deterministic_eval(self, decoder: AnonTokyoDecoder):
        """Same input produces same output in eval mode."""
        decoder.eval()
        enc_out = _make_enc_out(B=1, A=4, M=8, D=64)
        batch = _make_batch(B=1, A=4, K=2)
        out1 = decoder(enc_out, batch)
        out2 = decoder(enc_out, batch)
        torch.testing.assert_close(out1["pred_trajs"], out2["pred_trajs"])
        torch.testing.assert_close(out1["pred_scores"], out2["pred_scores"])
