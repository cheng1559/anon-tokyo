"""Ego-frame coordinate transform and feature engineering.

Scene-centric: centres on SDC at ``current_time_index``, rotates to heading-up,
pads / truncates agents and polylines to fixed sizes.
"""

from __future__ import annotations

import numpy as np


# ── Geometry helpers ──────────────────────────────────────────────────────────


def rotate_2d(xy: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2-D points by *angle* (radians).  ``xy``: ``(..., 2)``."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    shape = xy.shape
    return (xy.reshape(-1, 2) @ rot.T).reshape(shape)


# ── Polyline segmentation ────────────────────────────────────────────────────


def break_polylines(
    points: np.ndarray,
    num_points_per_polyline: int = 20,
    break_dist: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Break flat map points ``(P, 7)`` into fixed-size polyline segments.

    Returns ``(num_segs, num_points_per_polyline, 7)`` and matching mask.
    """
    P = num_points_per_polyline
    if len(points) == 0:
        return np.zeros((1, P, 7), dtype=np.float32), np.zeros((1, P), dtype=np.float32)

    shifted = np.roll(points[:, 0:2], 1, axis=0)
    shifted[0] = points[0, 0:2]
    dists = np.linalg.norm(points[:, 0:2] - shifted, axis=-1)
    break_idxs = np.where(dists > break_dist)[0]
    segments = np.split(points, break_idxs, axis=0)

    polys: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for seg in segments:
        if len(seg) == 0:
            continue
        for start in range(0, len(seg), P):
            chunk = seg[start : start + P]
            padded = np.zeros((P, 7), dtype=np.float32)
            m = np.zeros(P, dtype=np.float32)
            padded[: len(chunk)] = chunk
            m[: len(chunk)] = 1.0
            polys.append(padded)
            masks.append(m)

    if not polys:
        return np.zeros((1, P, 7), dtype=np.float32), np.zeros((1, P), dtype=np.float32)
    return np.stack(polys), np.stack(masks)


# ── Main transform ────────────────────────────────────────────────────────────

MAX_TRACKS_TO_PREDICT = 8


def _select_scene_agents(
    *,
    valid: np.ndarray,
    tracks_to_predict: np.ndarray,
    sdc_idx: int,
    max_agents: int,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select and reindex scene agents with the same priority as scene transform."""
    num_agents = valid.shape[0]
    hist_mask = valid[:, :history_len]
    has_history = hist_mask.any(axis=1)
    valid_ttp = tracks_to_predict[(tracks_to_predict >= 0) & (tracks_to_predict < num_agents)]

    keep = has_history.copy().astype(bool)
    keep[valid_ttp] = True
    if 0 <= sdc_idx < num_agents:
        keep[sdc_idx] = True
    keep_idx = np.where(keep)[0]

    if len(keep_idx) > max_agents:
        priority = np.zeros(num_agents, dtype=np.float32)
        priority[keep_idx] = 1
        priority[valid_ttp] = 2
        if 0 <= sdc_idx < num_agents:
            priority[sdc_idx] = 3
        keep_idx = np.sort(np.argsort(-priority)[:max_agents])

    old_to_new = np.full(num_agents, -1, dtype=np.int32)
    old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int32)
    return keep_idx, old_to_new


def _build_map_token_features(polys: np.ndarray, poly_mask: np.ndarray, poly_centers: np.ndarray) -> np.ndarray:
    """Aggregate each fixed-size map polyline into one scene-frame token."""
    M, P, _ = polys.shape
    token_features = np.zeros((M, 11), dtype=np.float32)
    valid_poly = poly_mask.any(axis=1)
    if not valid_poly.any():
        return token_features

    token_features[:, 0:2] = poly_centers.astype(np.float32)
    point_order = np.arange(P, dtype=np.int32)[None, :]
    first_idx = np.where(poly_mask > 0, point_order, P).min(axis=1).clip(max=P - 1)
    last_idx = np.where(poly_mask > 0, point_order, 0).max(axis=1)
    rows = np.arange(M)
    first_xy = polys[rows, first_idx, 0:2]
    last_xy = polys[rows, last_idx, 0:2]
    token_features[:, 2:4] = first_xy
    token_features[:, 4:6] = last_xy

    segment = last_xy - first_xy
    seg_norm = np.linalg.norm(segment, axis=-1, keepdims=True)
    dir_xy = segment / np.clip(seg_norm, 1e-6, None)
    mask_f = poly_mask.astype(np.float32)
    count = np.clip(mask_f.sum(axis=1, keepdims=True), 1.0, None)
    mean_dir = (polys[:, :, 3:5] * mask_f[:, :, None]).sum(axis=1)
    mean_norm = np.linalg.norm(mean_dir, axis=-1, keepdims=True)
    mean_dir = mean_dir / np.clip(mean_norm, 1e-6, None)
    dir_xy = np.where(seg_norm > 1e-4, dir_xy, mean_dir)
    token_features[:, 6:8] = dir_xy

    edge_len = np.linalg.norm(polys[:, 1:, 0:2] - polys[:, :-1, 0:2], axis=-1)
    edge_mask = (poly_mask[:, 1:] > 0) & (poly_mask[:, :-1] > 0)
    token_features[:, 8:9] = (edge_len * edge_mask.astype(np.float32)).sum(axis=1, keepdims=True)
    token_features[:, 9:10] = (polys[:, :, 2] * mask_f).sum(axis=1, keepdims=True) / count
    token_features[:, 10:11] = (polys[:, :, 6] * mask_f).sum(axis=1, keepdims=True) / count
    token_features[~valid_poly] = 0.0
    return token_features


def _transform_scene_trajs(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float, int, int, int]:
    """Return full trajectories in the SDC frame used by scene-centric models."""
    current_t = int(data["current_time_index"])
    sdc_idx = int(data["sdc_track_index"])
    trajs = data["trajs"]  # (A, T, 10)
    num_agents, num_timestamps, _ = trajs.shape

    sdc_state = trajs[sdc_idx, current_t]
    center_xy = sdc_state[0:2].copy()
    center_heading = float(sdc_state[6])

    t = trajs.copy()
    t[:, :, 0:2] -= center_xy[None, None, :]
    t[:, :, 0:2] = rotate_2d(t[:, :, 0:2].reshape(-1, 2), -center_heading).reshape(num_agents, num_timestamps, 2)
    t[:, :, 7:9] = rotate_2d(t[:, :, 7:9].reshape(-1, 2), -center_heading).reshape(num_agents, num_timestamps, 2)
    t[:, :, 6] -= center_heading
    return t, center_xy, center_heading, current_t, sdc_idx, num_timestamps


def scene_centric_transform(
    data: dict[str, np.ndarray],
    *,
    max_agents: int = 128,
    max_polylines: int = 4096,
    num_points_per_polyline: int = 20,
    vector_break_dist: float = 1.0,
    center_offset_of_map: tuple[float, float] = (30.0, 0.0),
    include_eval_meta: bool = False,
) -> dict[str, np.ndarray]:
    """Transform a raw scenario dict into scene-centric padded features."""

    trajs = data["trajs"]  # (A, 91, 10)
    num_agents, num_timestamps, _ = trajs.shape
    t, center_xy, center_heading, current_t, sdc_idx, _ = _transform_scene_trajs(data)
    history_len = current_t + 1
    future_len = num_timestamps - history_len

    valid = t[:, :, 9]  # (A, T)

    # History features: [x, y, z, dx, dy, dz, sin_h, cos_h, vx, vy]  (10-d)
    hist = t[:, :history_len]
    hist_feat = np.concatenate(
        [
            hist[:, :, 0:6],
            np.sin(hist[:, :, 6:7]),
            np.cos(hist[:, :, 6:7]),
            hist[:, :, 7:9],
        ],
        axis=-1,
    ).astype(np.float32)
    hist_mask = valid[:, :history_len].astype(np.float32)
    hist_feat[hist_mask == 0] = 0

    # Future labels: [x, y, vx, vy]
    fut = t[:, history_len:]
    fut_feat = np.concatenate([fut[:, :, 0:2], fut[:, :, 7:9]], axis=-1).astype(np.float32)
    fut_mask = valid[:, history_len:].astype(np.float32)
    fut_feat[fut_mask == 0] = 0

    # Last valid observed position / heading per agent (vectorised).
    # Agents unobserved at current_t have raw coords [0,0] which become
    # -center_xy after SDC centering (thousands of metres off), so we must
    # use the latest actually-observed timestep instead.
    hist_valid = valid[:, :history_len] > 0  # (A, T_hist)
    last_t = np.where(hist_valid, np.arange(history_len), -1).max(1)  # (A,)
    has_any = last_t >= 0
    last_t = last_t.clip(min=0)
    ai = np.arange(num_agents)
    positions = (t[ai, last_t, 0:2] * has_any[:, None]).astype(np.float32)
    headings = (t[ai, last_t, 6] * has_any).astype(np.float32)

    # Agent-local future offset (regression target)
    # Δp_local = R(-ψ'_i)^T (p'_future - p'_last_obs)  — paper Eq. gt_offset
    agent_pos_ct = positions.copy()  # last valid observed position (A, 2)
    fut_xy = fut[:, :, 0:2] - agent_pos_ct[:, None, :]  # offset from current_t pos
    fut_vxy = fut[:, :, 7:9].copy()
    fut_local = np.zeros((num_agents, future_len, 4), dtype=np.float32)
    for a in range(num_agents):
        h = -headings[a]
        fut_local[a, :, 0:2] = rotate_2d(fut_xy[a], h)
        fut_local[a, :, 2:4] = rotate_2d(fut_vxy[a], h)
    fut_local[fut_mask == 0] = 0

    obj_types = data["object_type"].astype(np.int32)
    tracks_to_predict = data["tracks_to_predict"].astype(np.int32)

    # ── Agent selection / reindex ────────────────────────────────────────
    keep_idx, old_to_new = _select_scene_agents(
        valid=valid,
        tracks_to_predict=tracks_to_predict,
        sdc_idx=sdc_idx,
        max_agents=max_agents,
        history_len=history_len,
    )
    n = len(keep_idx)

    A = max_agents
    p_hist = np.zeros((A, history_len, 10), dtype=np.float32)
    p_hist_mask = np.zeros((A, history_len), dtype=np.float32)
    p_fut = np.zeros((A, future_len, 4), dtype=np.float32)
    p_fut_mask = np.zeros((A, future_len), dtype=np.float32)
    p_fut_local = np.zeros((A, future_len, 4), dtype=np.float32)
    p_pos = np.zeros((A, 2), dtype=np.float32)
    p_headings = np.zeros(A, dtype=np.float32)
    p_types = np.zeros(A, dtype=np.int32)
    agent_mask = np.zeros(A, dtype=np.float32)

    p_hist[:n] = hist_feat[keep_idx]
    p_hist_mask[:n] = hist_mask[keep_idx]
    p_fut[:n] = fut_feat[keep_idx]
    p_fut_mask[:n] = fut_mask[keep_idx]
    p_fut_local[:n] = fut_local[keep_idx]
    p_pos[:n] = positions[keep_idx]
    p_headings[:n] = headings[keep_idx]
    p_types[:n] = obj_types[keep_idx]
    agent_mask[:n] = 1.0

    new_ttp = old_to_new[tracks_to_predict]
    # Pad tracks_to_predict to fixed length
    K = MAX_TRACKS_TO_PREDICT
    padded_ttp = np.full(K, -1, dtype=np.int32)
    nk = min(len(new_ttp), K)
    padded_ttp[:nk] = new_ttp[:nk]

    new_sdc = old_to_new[sdc_idx]

    # ── Map polylines ────────────────────────────────────────────────────
    map_pts = data["map_polylines"].copy()  # (P, 7)
    map_pts[:, 0:2] -= center_xy[None, :]
    map_pts[:, 0:2] = rotate_2d(map_pts[:, 0:2], -center_heading)
    map_pts[:, 3:5] = rotate_2d(map_pts[:, 3:5], -center_heading)

    polys, poly_mask = break_polylines(map_pts, num_points_per_polyline, vector_break_dist)

    M = max_polylines
    P_pts = num_points_per_polyline
    num_polys = len(polys)

    if num_polys > M:
        # Keep map polylines close to the agents that will be decoded.  This
        # preserves target-local lane context even when a target is far from
        # the SDC origin, while still keeping an SDC-origin query for scene
        # context and backwards-compatible behavior on simple cases.
        csum = (polys[:, :, 0:2] * poly_mask[:, :, None]).sum(axis=1)
        ccnt = np.clip(poly_mask.sum(axis=1, keepdims=True), 1, None)
        centers = csum / ccnt
        query_points = [np.zeros(2, dtype=np.float32)]
        offset = np.asarray(center_offset_of_map, dtype=np.float32)
        for track_idx in tracks_to_predict:
            if track_idx < 0 or track_idx >= num_agents or not has_any[track_idx]:
                continue
            query_points.append(positions[track_idx] + rotate_2d(offset[None], headings[track_idx])[0])
        queries = np.stack(query_points, axis=0)
        dists = np.linalg.norm(centers[:, None, :] - queries[None, :, :], axis=-1).min(axis=1)
        top_k = np.argpartition(dists, M)[:M]
        top_k = top_k[np.argsort(dists[top_k])]
        polys, poly_mask = polys[top_k], poly_mask[top_k]
        num_polys = M

    p_polys = np.zeros((M, P_pts, 7), dtype=np.float32)
    p_poly_mask = np.zeros((M, P_pts), dtype=np.float32)
    p_polys[:num_polys] = polys[:num_polys]
    p_poly_mask[:num_polys] = poly_mask[:num_polys]

    # Polyline centres (for RoPE)
    csum = (p_polys[:, :, 0:2] * p_poly_mask[:, :, None]).sum(axis=1)
    ccnt = np.clip(p_poly_mask.sum(axis=1, keepdims=True), 1, None)
    poly_centers = (csum / ccnt).astype(np.float32)
    map_token_features = _build_map_token_features(p_polys, p_poly_mask, poly_centers)

    # Polyline headings (for DRoPE): direction from first to last valid point
    map_headings = np.zeros(M, dtype=np.float32)
    for m_idx in range(min(num_polys, M)):
        valid_pts = np.where(p_poly_mask[m_idx] > 0)[0]
        if len(valid_pts) >= 2:
            first, last = valid_pts[0], valid_pts[-1]
            dx = p_polys[m_idx, last, 0] - p_polys[m_idx, first, 0]
            dy = p_polys[m_idx, last, 1] - p_polys[m_idx, first, 1]
            map_headings[m_idx] = np.arctan2(dy, dx)

    map_element_mask = np.zeros(M, dtype=np.float32)
    map_element_mask[:num_polys] = 1.0

    out = {
        # Agents
        "obj_trajs": p_hist,  # (A, T_hist, 10)
        "obj_trajs_mask": p_hist_mask,  # (A, T_hist)
        "obj_positions": p_pos,  # (A, 2)
        "obj_headings": p_headings,  # (A,)
        "obj_types": p_types,  # (A,)
        "agent_mask": agent_mask,  # (A,)
        "tracks_to_predict": padded_ttp,  # (K,)
        "sdc_track_index": np.int32(new_sdc),
        # Map
        "map_polylines": p_polys,  # (M, P_pts, 7)
        "map_polylines_mask": p_poly_mask,  # (M, P_pts)
        "map_polylines_center": poly_centers,  # (M, 2)
        "map_token_features": map_token_features,  # (M, 11)
        "map_headings": map_headings,  # (M,)
        "map_mask": map_element_mask,  # (M,)
        # Future (ego frame)
        "obj_trajs_future": p_fut,  # (A, T_fut, 4)
        "obj_trajs_future_mask": p_fut_mask,  # (A, T_fut)
        # Future (agent-local frame, regression target)
        "obj_trajs_future_local": p_fut_local,  # (A, T_fut, 4)
        # Meta (for inverse transform / eval)
        "scenario_id": str(data["scenario_id"]),
        "center_xy": center_xy.astype(np.float32),
        "center_heading": np.float32(center_heading),
    }

    if include_eval_meta:
        eval_object_id = np.zeros(K, dtype=np.int64)
        eval_object_type = np.zeros(K, dtype=np.int32)
        eval_gt_trajs = np.zeros((K, num_timestamps, trajs.shape[-1]), dtype=np.float32)
        eval_raw_track_index = np.full(K, -1, dtype=np.int32)
        raw_eval_tracks = tracks_to_predict[:nk]
        eval_raw_track_index[:nk] = raw_eval_tracks
        eval_object_id[:nk] = data["object_id"][raw_eval_tracks].astype(np.int64)
        eval_object_type[:nk] = obj_types[raw_eval_tracks].astype(np.int32)
        eval_gt_trajs[:nk] = trajs[raw_eval_tracks].astype(np.float32)
        out.update(
            {
                "eval_object_id": eval_object_id,
                "eval_object_type": eval_object_type,
                "eval_gt_trajs": eval_gt_trajs,
                "eval_raw_track_index": eval_raw_track_index,
            }
        )

    return out


def simulation_transform(
    data: dict[str, np.ndarray],
    *,
    max_agents: int = 128,
    max_polylines: int = 4096,
    num_points_per_polyline: int = 20,
    vector_break_dist: float = 1.0,
    center_offset_of_map: tuple[float, float] = (30.0, 0.0),
    control_mode: str = "tracks_to_predict",
) -> dict[str, np.ndarray]:
    """Scene-centric transform with extra closed-loop simulation tensors.

    The returned sample keeps the prediction field names and adds only the
    tensors needed by the simulator: full log trajectories, timestamps and a
    boolean controlled mask derived from valid ``tracks_to_predict`` entries.
    """
    out = scene_centric_transform(
        data,
        max_agents=max_agents,
        max_polylines=max_polylines,
        num_points_per_polyline=num_points_per_polyline,
        vector_break_dist=vector_break_dist,
        center_offset_of_map=center_offset_of_map,
        include_eval_meta=False,
    )

    t, _, _, current_t, sdc_idx, num_timestamps = _transform_scene_trajs(data)
    valid = t[:, :, 9]
    tracks_to_predict = data["tracks_to_predict"].astype(np.int32)
    keep_idx, old_to_new = _select_scene_agents(
        valid=valid,
        tracks_to_predict=tracks_to_predict,
        sdc_idx=sdc_idx,
        max_agents=max_agents,
        history_len=current_t + 1,
    )

    A = max_agents
    p_full = np.zeros((A, num_timestamps, t.shape[-1]), dtype=np.float32)
    p_full_mask = np.zeros((A, num_timestamps), dtype=np.float32)
    n = len(keep_idx)
    p_full[:n] = t[keep_idx].astype(np.float32)
    p_full_mask[:n] = valid[keep_idx].astype(np.float32)
    p_full[p_full_mask == 0] = 0

    new_ttp = old_to_new[tracks_to_predict[(tracks_to_predict >= 0) & (tracks_to_predict < t.shape[0])]]
    controlled_mask = np.zeros(A, dtype=np.bool_)
    if control_mode in {"non_reactive", "sdc", "ego"}:
        if 0 <= int(out["sdc_track_index"]) < A:
            controlled_mask[int(out["sdc_track_index"])] = True
    elif control_mode == "tracks_to_predict":
        valid_new_ttp = new_ttp[(new_ttp >= 0) & (new_ttp < A)]
        controlled_mask[valid_new_ttp] = True
    else:
        raise ValueError(f"Unsupported simulation control_mode: {control_mode}")
    controlled_mask &= out["agent_mask"].astype(bool)
    controlled_mask &= p_full_mask[:, current_t].astype(bool)

    timestamps = data.get("timestamps")
    if timestamps is None:
        timestamps = np.arange(num_timestamps, dtype=np.float32) * 0.1

    out.update(
        {
            "obj_trajs_full": p_full,
            "obj_trajs_full_mask": p_full_mask,
            "controlled_mask": controlled_mask,
            "current_time_index": np.int32(current_t),
            "timestamps": np.asarray(timestamps, dtype=np.float32),
        }
    )
    return out
