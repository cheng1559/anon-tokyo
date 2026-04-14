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


def scene_centric_transform(
    data: dict[str, np.ndarray],
    *,
    max_agents: int = 128,
    max_polylines: int = 4096,
    num_points_per_polyline: int = 20,
    vector_break_dist: float = 1.0,
) -> dict[str, np.ndarray]:
    """Transform a raw scenario dict into scene-centric padded features."""

    current_t = int(data["current_time_index"])
    sdc_idx = int(data["sdc_track_index"])
    trajs = data["trajs"]  # (A, 91, 10)
    num_agents, num_timestamps, _ = trajs.shape
    history_len = current_t + 1
    future_len = num_timestamps - history_len

    # ── SDC reference frame ──────────────────────────────────────────────
    sdc_state = trajs[sdc_idx, current_t]
    center_xy = sdc_state[0:2].copy()
    center_heading = float(sdc_state[6])

    # ── Transform trajectories ───────────────────────────────────────────
    t = trajs.copy()
    t[:, :, 0:2] -= center_xy[None, None, :]
    t[:, :, 0:2] = rotate_2d(t[:, :, 0:2].reshape(-1, 2), -center_heading).reshape(num_agents, num_timestamps, 2)
    t[:, :, 7:9] = rotate_2d(t[:, :, 7:9].reshape(-1, 2), -center_heading).reshape(num_agents, num_timestamps, 2)
    t[:, :, 6] -= center_heading

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

    # Position at current_t (for RoPE)
    positions = t[:, current_t, 0:2].astype(np.float32)

    # Heading at current_t (for DRoPE)
    headings = t[:, current_t, 6].astype(np.float32)

    # Agent-local future offset (regression target)
    # Δp_local = R(-ψ'_i)^T (p'_future - p'_last_obs)  — paper Eq. gt_offset
    agent_pos_ct = t[:, current_t, 0:2]  # (A, 2)
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
    has_history = hist_mask.any(axis=1)
    keep = has_history.copy().astype(bool)
    keep[tracks_to_predict] = True
    keep[sdc_idx] = True
    keep_idx = np.where(keep)[0]

    if len(keep_idx) > max_agents:
        # Priority: sdc=3, tracks_to_predict=2, valid=1
        priority = np.zeros(num_agents, dtype=np.float32)
        priority[keep_idx] = 1
        priority[tracks_to_predict] = 2
        priority[sdc_idx] = 3
        keep_idx = np.sort(np.argsort(-priority)[:max_agents])

    old_to_new = np.full(num_agents, -1, dtype=np.int32)
    old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int32)
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
        # Keep closest to origin
        csum = (polys[:, :, 0:2] * poly_mask[:, :, None]).sum(axis=1)
        ccnt = np.clip(poly_mask.sum(axis=1, keepdims=True), 1, None)
        centers = csum / ccnt
        dists = np.linalg.norm(centers, axis=1)
        top_k = np.sort(np.argpartition(dists, M)[:M])
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

    return {
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
