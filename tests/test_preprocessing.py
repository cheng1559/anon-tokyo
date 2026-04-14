"""Tests for MTR agent-centric preprocessing."""

from __future__ import annotations

import math

import pytest
import torch

from anon_tokyo.prediction.mtr.preprocessing import (
    _rotate_2d,
    agent_centric_preprocess,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_batch(
    B: int = 2,
    A: int = 4,
    T_hist: int = 3,
    M: int = 8,
    P: int = 5,
    T_fut: int = 4,
    K_max: int = 2,
) -> dict[str, torch.Tensor]:
    """Build a minimal scene-centric batch with known values."""
    torch.manual_seed(42)
    obj_trajs = torch.randn(B, A, T_hist, 10)
    # Make heading channel [6:8] valid sin/cos from random angles
    angles = torch.randn(B, A, T_hist)
    obj_trajs[..., 6] = torch.sin(angles)
    obj_trajs[..., 7] = torch.cos(angles)
    batch: dict[str, torch.Tensor] = {
        "obj_trajs": obj_trajs,
        "obj_trajs_mask": torch.ones(B, A, T_hist),
        "obj_positions": torch.randn(B, A, 2),
        "obj_headings": torch.randn(B, A),
        "obj_types": torch.randint(0, 3, (B, A)),
        "agent_mask": torch.ones(B, A),
        "tracks_to_predict": torch.zeros(B, K_max, dtype=torch.long),
        "map_polylines": torch.randn(B, M, P, 7),
        "map_polylines_mask": torch.ones(B, M, P),
        "map_polylines_center": torch.randn(B, M, 2),
        "map_headings": torch.randn(B, M),
        "map_mask": torch.ones(B, M),
        "obj_trajs_future": torch.randn(B, A, T_fut, 4),
        "obj_trajs_future_mask": torch.ones(B, A, T_fut),
    }
    # first scene predicts agents 0,1; second predicts agent 2 only
    batch["tracks_to_predict"][0, 0] = 0
    batch["tracks_to_predict"][0, 1] = 1
    batch["tracks_to_predict"][1, 0] = 2
    batch["tracks_to_predict"][1, 1] = -1  # padding
    return batch


# ── rotate_2d ────────────────────────────────────────────────────────────────


def test_rotate_2d_identity():
    xy = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = _rotate_2d(xy, torch.tensor(0.0))
    torch.testing.assert_close(out, xy)


def test_rotate_2d_90deg():
    xy = torch.tensor([[1.0, 0.0]])
    out = _rotate_2d(xy, torch.tensor(math.pi / 2))
    torch.testing.assert_close(out, torch.tensor([[0.0, 1.0]]), atol=1e-6, rtol=1e-5)


def test_rotate_2d_broadcast():
    xy = torch.randn(3, 4, 2)
    angle = torch.randn(3, 4)
    out = _rotate_2d(xy, angle)
    # magnitude preserved
    torch.testing.assert_close(out.norm(dim=-1), xy.norm(dim=-1), atol=1e-5, rtol=1e-5)


# ── agent_centric_preprocess ─────────────────────────────────────────────────


def test_output_shapes():
    batch = _make_batch()
    out = agent_centric_preprocess(batch)
    K_total = 3  # 2 + 1
    assert out["obj_trajs"].shape == (K_total, 4, 3, 10)
    assert out["track_index_to_predict"].shape == (K_total,)
    assert out["center_obj_type"].shape == (K_total,)
    assert out["batch_sample_count"].tolist() == [2, 1]
    assert out["batch_idx"].tolist() == [0, 0, 1]
    assert out["tracks_to_predict"].shape == (K_total, 1)


def test_center_agent_at_origin():
    """Centre agent should be at (0, 0) with heading 0 after transform."""
    batch = _make_batch()
    out = agent_centric_preprocess(batch)
    for k in range(3):
        aidx = out["track_index_to_predict"][k].item()
        pos = out["obj_positions"][k, aidx]
        torch.testing.assert_close(pos, torch.zeros(2), atol=1e-5, rtol=1e-5)
        hd = out["obj_headings"][k, aidx]
        assert abs(hd.item()) < 1e-5


def test_distances_preserved():
    """Pairwise distances between agents should be unchanged."""
    batch = _make_batch()
    out = agent_centric_preprocess(batch)
    # Check scene 0, agent-to-predict 0 (first in K_total)
    orig_pos = batch["obj_positions"][0]  # [A, 2]
    new_pos = out["obj_positions"][0]  # [A, 2]
    orig_dist = torch.cdist(orig_pos.unsqueeze(0), orig_pos.unsqueeze(0)).squeeze()
    new_dist = torch.cdist(new_pos.unsqueeze(0), new_pos.unsqueeze(0)).squeeze()
    torch.testing.assert_close(orig_dist, new_dist, atol=1e-4, rtol=1e-4)


def test_velocity_magnitude_preserved():
    """Velocity magnitudes should be unchanged by rotation."""
    batch = _make_batch()
    out = agent_centric_preprocess(batch)
    orig_vel = batch["obj_trajs"][0, :, :, 8:10]  # [A, T, 2]
    new_vel = out["obj_trajs"][0, :, :, 8:10]
    torch.testing.assert_close(orig_vel.norm(dim=-1), new_vel.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_heading_sin_cos_unit():
    """sin^2 + cos^2 should equal 1 after heading adjustment."""
    batch = _make_batch()
    out = agent_centric_preprocess(batch)
    sin_h = out["obj_trajs"][..., 6]
    cos_h = out["obj_trajs"][..., 7]
    norm = sin_h**2 + cos_h**2
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=1e-5, rtol=1e-5)


def test_map_direction_magnitude_preserved():
    """Map direction vectors should preserve magnitude."""
    batch = _make_batch()
    out = agent_centric_preprocess(batch)
    orig_dir = batch["map_polylines"][0, :, :, 3:5]
    new_dir = out["map_polylines"][0, :, :, 3:5]
    torch.testing.assert_close(orig_dir.norm(dim=-1), new_dir.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_map_type_unchanged():
    """Map type feature should not be modified."""
    batch = _make_batch()
    out = agent_centric_preprocess(batch)
    for k in range(3):
        b = out["batch_idx"][k].item()
        torch.testing.assert_close(
            out["map_polylines"][k, :, :, 6],
            batch["map_polylines"][b, :, :, 6],
        )


def test_agent_size_unchanged():
    """Agent size (dx dy dz) should not be modified."""
    batch = _make_batch()
    out = agent_centric_preprocess(batch)
    for k in range(3):
        b = out["batch_idx"][k].item()
        torch.testing.assert_close(
            out["obj_trajs"][k, :, :, 3:6],
            batch["obj_trajs"][b, :, :, 3:6],
        )


def test_future_local_passthrough():
    """obj_trajs_future_local should be passed through unchanged."""
    batch = _make_batch()
    batch["obj_trajs_future_local"] = torch.randn(2, 4, 4, 4)
    out = agent_centric_preprocess(batch)
    for k in range(3):
        b = out["batch_idx"][k].item()
        torch.testing.assert_close(
            out["obj_trajs_future_local"][k],
            batch["obj_trajs_future_local"][b],
        )


def test_all_padding_returns_empty():
    """If all tracks_to_predict are -1, K_total = 0."""
    batch = _make_batch()
    batch["tracks_to_predict"][:] = -1
    out = agent_centric_preprocess(batch)
    assert out["obj_trajs"].shape[0] == 0
    assert out["batch_sample_count"].tolist() == [0, 0]
