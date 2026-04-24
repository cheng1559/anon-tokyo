"""Tests for ego-frame coordinate transform reversibility."""

from __future__ import annotations

import numpy as np
import pytest

from anon_tokyo.data.transforms import break_polylines, rotate_2d, scene_centric_transform


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_scenario(num_agents: int = 8, num_map_points: int = 200) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    trajs = rng.standard_normal((num_agents, 91, 10)).astype(np.float32)
    trajs[:, :, 9] = (rng.random((num_agents, 91)) > 0.1).astype(np.float32)
    trajs[:, :, 3:6] = np.abs(trajs[:, :, 3:6])
    # Ensure SDC valid at current_time
    trajs[0, 10, 9] = 1.0
    return {
        "scenario_id": np.array("test_scenario", dtype="U"),
        "timestamps": np.linspace(0, 9, 91, dtype=np.float64),
        "current_time_index": np.int32(10),
        "sdc_track_index": np.int32(0),
        "object_id": np.arange(num_agents, dtype=np.int64),
        "object_type": rng.integers(0, 4, size=num_agents).astype(np.int8),
        "trajs": trajs,
        "map_polylines": rng.standard_normal((num_map_points, 7)).astype(np.float32),
        "traffic_lights": rng.standard_normal((91, 5, 5)).astype(np.float32),
        "tracks_to_predict": np.array([0, 1], dtype=np.int32),
        "predict_difficulty": np.array([1, 2], dtype=np.int32),
    }


# ── rotate_2d ─────────────────────────────────────────────────────────────────


class TestRotate2D:
    def test_identity(self) -> None:
        pts = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(rotate_2d(pts, 0.0), pts, atol=1e-6)

    def test_90deg(self) -> None:
        pts = np.array([[1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(rotate_2d(pts, np.pi / 2), [[0.0, 1.0]], atol=1e-6)

    def test_roundtrip(self) -> None:
        rng = np.random.default_rng(123)
        pts = rng.standard_normal((50, 2)).astype(np.float32)
        angle = 1.234
        np.testing.assert_allclose(rotate_2d(rotate_2d(pts, angle), -angle), pts, atol=1e-5)


# ── break_polylines ───────────────────────────────────────────────────────────


class TestBreakPolylines:
    def test_empty(self) -> None:
        polys, mask = break_polylines(np.zeros((0, 7), dtype=np.float32), 20, 1.0)
        assert polys.shape == (1, 20, 7)
        assert mask.sum() == 0

    def test_single_segment(self) -> None:
        pts = np.zeros((10, 7), dtype=np.float32)
        pts[:, 0] = np.arange(10) * 0.5
        polys, mask = break_polylines(pts, 20, 1.0)
        assert polys.shape == (1, 20, 7)
        assert mask[0, :10].sum() == 10

    def test_break_at_gap(self) -> None:
        pts = np.zeros((6, 7), dtype=np.float32)
        pts[:3, 0] = [0, 0.5, 1.0]
        pts[3:, 0] = [10, 10.5, 11.0]  # big gap
        polys, mask = break_polylines(pts, 20, 2.0)
        assert polys.shape[0] == 2  # should split into 2 polylines


# ── scene_centric_transform ───────────────────────────────────────────────────


class TestSceneCentricTransform:
    def test_output_shapes(self) -> None:
        out = scene_centric_transform(_make_scenario(), max_agents=32, max_polylines=64)
        assert out["obj_trajs"].shape == (32, 11, 10)
        assert out["obj_trajs_mask"].shape == (32, 11)
        assert out["obj_positions"].shape == (32, 2)
        assert out["obj_headings"].shape == (32,)
        assert out["obj_types"].shape == (32,)
        assert out["agent_mask"].shape == (32,)
        assert out["tracks_to_predict"].shape == (8,)
        assert out["obj_trajs_future"].shape == (32, 80, 4)
        assert out["obj_trajs_future_mask"].shape == (32, 80)
        assert out["obj_trajs_future_local"].shape == (32, 80, 4)
        assert out["map_polylines"].shape[0] == 64
        assert out["map_polylines"].shape[2] == 7
        assert out["map_polylines_mask"].shape[0] == 64
        assert out["map_polylines_center"].shape == (64, 2)
        assert out["map_headings"].shape == (64,)
        assert out["map_mask"].shape == (64,)

    def test_sdc_at_origin(self) -> None:
        out = scene_centric_transform(_make_scenario(), max_agents=32, max_polylines=64)
        sdc = int(out["sdc_track_index"])
        np.testing.assert_allclose(out["obj_positions"][sdc], [0, 0], atol=1e-5)

    def test_tracks_to_predict_valid(self) -> None:
        out = scene_centric_transform(_make_scenario(), max_agents=32, max_polylines=64)
        ttp = out["tracks_to_predict"]
        valid = ttp[ttp >= 0]
        assert len(valid) == 2
        assert np.all(valid < 32)

    def test_eval_meta_optional(self) -> None:
        data = _make_scenario()
        out = scene_centric_transform(data, max_agents=32, max_polylines=64, include_eval_meta=True)
        assert out["eval_object_id"].shape == (8,)
        assert out["eval_object_type"].shape == (8,)
        assert out["eval_gt_trajs"].shape == (8, 91, 10)
        np.testing.assert_array_equal(out["eval_object_id"][:2], data["object_id"][data["tracks_to_predict"]])
        np.testing.assert_array_equal(out["eval_object_type"][:2], data["object_type"][data["tracks_to_predict"]])
        np.testing.assert_allclose(out["eval_gt_trajs"][:2], data["trajs"][data["tracks_to_predict"]])

    def test_masked_features_zero(self) -> None:
        out = scene_centric_transform(_make_scenario(), max_agents=32, max_polylines=64)
        invalid = out["obj_trajs_mask"] == 0
        assert np.all(out["obj_trajs"][invalid] == 0)

    def test_many_agents_truncation(self) -> None:
        data = _make_scenario(num_agents=200)
        data["trajs"][:, :, 9] = 1.0  # all valid
        out = scene_centric_transform(data, max_agents=32, max_polylines=64)
        assert out["obj_trajs"].shape[0] == 32
        # SDC and tracks_to_predict must survive
        assert int(out["sdc_track_index"]) >= 0
        ttp = out["tracks_to_predict"]
        assert np.all(ttp[ttp >= 0] < 32)

    def test_future_local_invertible(self) -> None:
        """Agent-local future offset can be rotated back to ego-frame offset."""
        out = scene_centric_transform(_make_scenario(), max_agents=32, max_polylines=64)
        mask = out["obj_trajs_future_mask"]  # (A, T_fut)
        headings = out["obj_headings"]  # (A,)
        positions = out["obj_positions"]  # (A, 2)
        fut_ego = out["obj_trajs_future"]  # (A, T_fut, 4) [x, y, vx, vy] in ego frame
        fut_local = out["obj_trajs_future_local"]  # (A, T_fut, 4) in agent-local

        for a in range(min(8, 32)):
            if mask[a].sum() == 0:
                continue
            h = headings[a]
            # Rotate local back to ego frame
            xy_ego = rotate_2d(fut_local[a, :, 0:2], h) + positions[a]
            vxy_ego = rotate_2d(fut_local[a, :, 2:4], h)
            valid = mask[a] > 0
            np.testing.assert_allclose(xy_ego[valid], fut_ego[a, valid, 0:2], atol=1e-4)
            np.testing.assert_allclose(vxy_ego[valid], fut_ego[a, valid, 2:4], atol=1e-4)

    def test_sdc_heading_zero(self) -> None:
        """SDC heading should be ~0 after ego-frame transform."""
        out = scene_centric_transform(_make_scenario(), max_agents=32, max_polylines=64)
        sdc = int(out["sdc_track_index"])
        np.testing.assert_allclose(out["obj_headings"][sdc], 0.0, atol=1e-5)
