"""Unit tests for AnonTokyo full model assembly."""

from __future__ import annotations

import pickle

import pytest
import torch
import numpy as np

from anon_tokyo.prediction.anon_tokyo.model import AnonTokyoModel


@pytest.fixture
def intention_points_file(tmp_path):
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


def _make_batch(B: int = 2, A: int = 16, M: int = 32, T: int = 11, P: int = 20, K: int = 4):
    ttp = torch.full((B, K), -1, dtype=torch.long)
    for b in range(B):
        n = min(K, A)
        ttp[b, :n] = torch.randperm(A)[:n]
    return {
        "obj_trajs": torch.randn(B, A, T, 10),
        "obj_trajs_mask": torch.ones(B, A, T),
        "obj_positions": torch.randn(B, A, 2) * 50,
        "obj_headings": torch.randn(B, A),
        "obj_types": torch.randint(1, 4, (B, A)),
        "agent_mask": torch.ones(B, A),
        "map_polylines": torch.randn(B, M, P, 7),
        "map_polylines_mask": torch.ones(B, M, P),
        "map_polylines_center": torch.randn(B, M, 2) * 100,
        "map_headings": torch.randn(B, M),
        "map_mask": torch.ones(B, M),
        "tracks_to_predict": ttp,
        # GT for loss (not needed by model, but needed for loss tests)
        "obj_trajs_future_local": torch.randn(B, A, 80, 4),
        "obj_trajs_future_mask": torch.ones(B, A, 80),
    }


@pytest.fixture
def model(intention_points_file) -> AnonTokyoModel:
    return AnonTokyoModel(
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_decoder=64,
        num_heads=4,
        sparse_k=8,
        num_modes=6,
        num_intention_queries=64,
        num_future_frames=20,
        dropout=0.0,
        use_rope=True,
        use_drope=True,
        intention_points_file=intention_points_file,
    )


class TestAnonTokyoModel:
    def test_output_keys(self, model: AnonTokyoModel):
        batch = _make_batch(B=1, A=8, M=16, K=2, T=11, P=10)
        model.eval()
        out = model(batch)
        assert "pred_trajs" in out
        assert "pred_scores" in out

    def test_output_shapes_eval(self, model: AnonTokyoModel):
        B, A, K = 2, 8, 4
        batch = _make_batch(B=B, A=A, M=16, K=K, T=11, P=10)
        model.eval()
        out = model(batch)
        assert out["pred_trajs"].shape == (B, K, 6, 20, 7)
        assert out["pred_scores"].shape == (B, K, 6)

    def test_output_shapes_train(self, model: AnonTokyoModel):
        B, A, K = 2, 8, 4
        batch = _make_batch(B=B, A=A, M=16, K=K, T=11, P=10)
        model.train()
        out = model(batch)
        assert out["pred_trajs"].shape == (B, K, 64, 20, 7)
        assert out["pred_scores"].shape == (B, K, 64)

    def test_gradient_flow(self, model: AnonTokyoModel):
        batch = _make_batch(B=1, A=4, M=8, K=2, T=11, P=10)
        batch["obj_trajs"].requires_grad_(True)
        model.train()
        out = model(batch)
        loss = out["pred_trajs"].sum() + out["pred_scores"].sum()
        loss.backward()
        assert batch["obj_trajs"].grad is not None

    def test_no_rope_no_drope(self, intention_points_file):
        model = AnonTokyoModel(
            d_model=64,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_decoder=64,
            num_heads=4,
            sparse_k=8,
            num_modes=6,
            num_intention_queries=64,
            num_future_frames=20,
            use_rope=False,
            use_drope=False,
            intention_points_file=intention_points_file,
        )
        batch = _make_batch(B=1, A=4, M=8, K=2, T=11, P=10)
        model.eval()
        out = model(batch)
        assert out["pred_trajs"].shape[0] == 1

    def test_deterministic_eval(self, model: AnonTokyoModel):
        model.eval()
        batch = _make_batch(B=1, A=4, M=8, K=2, T=11, P=10)
        out1 = model(batch)
        out2 = model(batch)
        torch.testing.assert_close(out1["pred_trajs"], out2["pred_trajs"])

    def test_loss_compatible(self, model: AnonTokyoModel):
        """Model output is compatible with prediction_loss()."""
        from anon_tokyo.prediction.loss import prediction_loss

        batch = _make_batch(B=2, A=8, M=16, K=4, T=11, P=10)
        # Adjust future frames to match model
        batch["obj_trajs_future_local"] = torch.randn(2, 8, 20, 4)
        batch["obj_trajs_future_mask"] = torch.ones(2, 8, 20)
        model.train()
        out = model(batch)
        loss, loss_dict = prediction_loss(out, batch, {"reg": 1.0, "score": 1.0, "vel": 0.2})
        assert loss.isfinite()
        assert loss.requires_grad
        loss.backward()

    def test_pred_list_returned_in_train(self, model: AnonTokyoModel):
        """Training mode returns pred_list for per-layer loss."""
        batch = _make_batch(B=1, A=4, M=8, K=2, T=11, P=10)
        model.train()
        out = model(batch)
        assert "pred_list" in out
        assert len(out["pred_list"]) == 2  # num_decoder_layers=2

    def test_all_params_have_grad(self, model: AnonTokyoModel):
        """All parameters participate in the loss via per-layer predictions."""
        from anon_tokyo.prediction.loss import prediction_loss

        batch = _make_batch(B=1, A=4, M=8, K=2, T=11, P=10)
        batch["obj_trajs_future_local"] = torch.randn(1, 4, 20, 4)
        batch["obj_trajs_future_mask"] = torch.ones(1, 4, 20)
        model.train()
        out = model(batch)
        loss, _ = prediction_loss(out, batch, {"reg": 1.0, "score": 1.0, "vel": 0.2})
        loss.backward()
        no_grad_params = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert no_grad_params == [], f"Parameters without grad: {no_grad_params}"
