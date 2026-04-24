"""Official-style MTR preprocessing from raw WOMD npz arrays."""

from __future__ import annotations

import numpy as np
import torch

from anon_tokyo.data.transforms import break_polylines


def _rotate_np(xy: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    x, y = xy[..., 0], xy[..., 1]
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((x * c - y * s, x * s + y * c), axis=-1).astype(np.float32)


def _transform_trajs(obj_trajs: np.ndarray, center: np.ndarray, heading_index: int, rot_vel_index: tuple[int, int]) -> np.ndarray:
    num_center = center.shape[0]
    out = np.repeat(obj_trajs[None], num_center, axis=0).astype(np.float32)
    out[..., 0:3] -= center[:, None, None, 0:3]
    out[..., 0:2] = _rotate_np(out[..., 0:2], -center[:, None, None, 6])
    out[..., heading_index] -= center[:, None, None, 6]
    out[..., rot_vel_index[0] : rot_vel_index[1] + 1] = _rotate_np(
        out[..., rot_vel_index[0] : rot_vel_index[1] + 1],
        -center[:, None, None, 6],
    )
    return out


def _map_for_centers(
    map_points: np.ndarray,
    centers: np.ndarray,
    *,
    max_polylines: int,
    num_points_per_polyline: int,
    vector_break_dist: float,
    center_offset: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(map_points) == 0:
        map_points = np.zeros((2, 7), dtype=np.float32)
    polys, poly_mask = break_polylines(map_points, num_points_per_polyline, vector_break_dist)
    polyline_center = (polys[:, :, 0:2] * poly_mask[:, :, None]).sum(1) / np.clip(poly_mask.sum(1, keepdims=True), 1, None)
    num_center = centers.shape[0]
    if len(polys) > max_polylines:
        offset = np.array(center_offset, dtype=np.float32)[None].repeat(num_center, axis=0)
        offset = _rotate_np(offset[:, None], centers[:, 6])[:, 0]
        query = centers[:, 0:2] + offset
        dist = np.linalg.norm(query[:, None] - polyline_center[None], axis=-1)
        topk = np.argpartition(dist, max_polylines, axis=-1)[:, :max_polylines]
        topk = np.take_along_axis(topk, np.argsort(np.take_along_axis(dist, topk, axis=-1), axis=-1), axis=-1)
        map_polys = polys[topk]
        map_masks = poly_mask[topk]
    else:
        map_polys = np.repeat(polys[None], num_center, axis=0)
        map_masks = np.repeat(poly_mask[None], num_center, axis=0)

    map_polys = map_polys.astype(np.float32)
    map_masks = map_masks.astype(np.float32)
    map_polys[..., 0:3] -= centers[:, None, None, 0:3]
    map_polys[..., 0:2] = _rotate_np(map_polys[..., 0:2], -centers[:, None, None, 6])
    map_polys[..., 3:5] = _rotate_np(map_polys[..., 3:5], -centers[:, None, None, 6])
    pre_xy = np.roll(map_polys[..., 0:2], shift=1, axis=-2)
    if pre_xy.shape[-2] > 1:
        pre_xy[:, :, 0] = pre_xy[:, :, 1]
    map_polys = np.concatenate((map_polys, pre_xy), axis=-1)
    map_polys[map_masks == 0] = 0
    map_centers = (map_polys[..., 0:3] * map_masks[..., None]).sum(-2) / np.clip(map_masks.sum(-1, keepdims=True), 1, None)
    return map_polys.astype(np.float32), map_masks.astype(bool), map_centers.astype(np.float32)


def official_mtr_transform(
    data: dict[str, np.ndarray],
    *,
    max_polylines: int = 768,
    num_points_per_polyline: int = 20,
    vector_break_dist: float = 1.0,
    center_offset: tuple[float, float] = (30.0, 0.0),
) -> dict[str, np.ndarray]:
    """Build one official MTR scene sample with first dimension = target agents."""
    cur_t = int(data["current_time_index"])
    trajs = data["trajs"].astype(np.float32)
    obj_types_all = data["object_type"].astype(np.int32)
    tracks = data["tracks_to_predict"].astype(np.int64)
    tracks = np.array([i for i in tracks if obj_types_all[i] in (1, 2, 3) and trajs[i, cur_t, -1] > 0], dtype=np.int64)
    if len(tracks) == 0:
        tracks = data["tracks_to_predict"][:0].astype(np.int64)

    center_objects = trajs[tracks, cur_t] if len(tracks) else np.zeros((0, 10), dtype=np.float32)
    past = trajs[:, : cur_t + 1]
    future = trajs[:, cur_t + 1 :]
    timestamps = data.get("timestamps", np.arange(cur_t + 1, dtype=np.float32))[: cur_t + 1].astype(np.float32)

    centered = _transform_trajs(past, center_objects, 6, (7, 8)) if len(tracks) else np.zeros((0, trajs.shape[0], cur_t + 1, 10), dtype=np.float32)
    num_center, num_objects, hist_len, _ = centered.shape

    type_onehot = np.zeros((num_center, num_objects, hist_len, 5), dtype=np.float32)
    type_onehot[:, obj_types_all == 1, :, 0] = 1
    # Official MTR has a typo ("TYPE_PEDESTRAIN") in the pedestrian branch, so
    # this channel stays zero for WOMD pedestrians. Preserve that behavior for
    # official checkpoint input parity.
    type_onehot[:, obj_types_all == 3, :, 2] = 1
    if num_center:
        type_onehot[np.arange(num_center), tracks, :, 3] = 1
        type_onehot[:, int(data["sdc_track_index"]), :, 4] = 1
    time_emb = np.zeros((num_center, num_objects, hist_len, hist_len + 1), dtype=np.float32)
    time_emb[:, :, np.arange(hist_len), np.arange(hist_len)] = 1
    time_emb[:, :, np.arange(hist_len), -1] = timestamps
    heading_emb = np.stack((np.sin(centered[..., 6]), np.cos(centered[..., 6])), axis=-1).astype(np.float32)
    vel = centered[..., 7:9]
    vel_pre = np.roll(vel, shift=1, axis=2)
    acce = (vel - vel_pre) / 0.1
    if hist_len > 1:
        acce[:, :, 0] = acce[:, :, 1]
    obj_trajs = np.concatenate((centered[..., 0:6], type_onehot, time_emb, heading_emb, vel, acce), axis=-1)
    obj_mask = centered[..., -1] > 0
    obj_trajs[obj_mask == 0] = 0

    centered_future = _transform_trajs(future, center_objects, 6, (7, 8)) if num_center else np.zeros((0, num_objects, future.shape[1], 10), dtype=np.float32)
    future_state = centered_future[..., [0, 1, 7, 8]].astype(np.float32)
    future_mask = centered_future[..., -1] > 0
    future_state[future_mask == 0] = 0

    valid_past = ~(past[:, :, -1].sum(axis=-1) == 0)
    obj_trajs = obj_trajs[:, valid_past]
    obj_mask = obj_mask[:, valid_past]
    future_state = future_state[:, valid_past]
    future_mask = future_mask[:, valid_past]
    obj_types = obj_types_all[valid_past]
    obj_ids = data["object_id"][valid_past]
    valid_count = valid_past.cumsum()
    track_new = valid_count[tracks] - 1 if len(tracks) else np.zeros((0,), dtype=np.int64)
    sdc_new = np.array([valid_count[int(data["sdc_track_index"])] - 1], dtype=np.int64)

    obj_pos = obj_trajs[..., 0:3]
    last_pos = np.zeros((num_center, obj_trajs.shape[1], 3), dtype=np.float32)
    for t in range(hist_len):
        cur = obj_mask[:, :, t] > 0
        last_pos[cur] = obj_pos[:, :, t][cur]

    center_gt = future_state[np.arange(num_center), track_new] if num_center else np.zeros((0, future.shape[1], 4), dtype=np.float32)
    center_gt_mask = future_mask[np.arange(num_center), track_new] if num_center else np.zeros((0, future.shape[1]), dtype=bool)
    center_gt[center_gt_mask == 0] = 0
    final_idx = np.zeros(num_center, dtype=np.float32)
    for t in range(center_gt_mask.shape[1]):
        final_idx[center_gt_mask[:, t] > 0] = t

    map_polys, map_masks, map_centers = _map_for_centers(
        data["map_polylines"].astype(np.float32),
        center_objects,
        max_polylines=max_polylines,
        num_points_per_polyline=num_points_per_polyline,
        vector_break_dist=vector_break_dist,
        center_offset=center_offset,
    )

    return {
        "scenario_id": np.array([str(data["scenario_id"])] * num_center),
        "obj_trajs": obj_trajs.astype(np.float32),
        "obj_trajs_mask": obj_mask,
        "track_index_to_predict": track_new.astype(np.int64),
        "obj_trajs_pos": obj_pos.astype(np.float32),
        "obj_trajs_last_pos": last_pos,
        "obj_types": np.repeat(obj_types[None], num_center, axis=0).astype(np.int32),
        "obj_ids": np.repeat(obj_ids[None], num_center, axis=0).astype(np.int64),
        "sdc_track_index": np.repeat(sdc_new, num_center).astype(np.int64),
        "center_objects_world": center_objects.astype(np.float32),
        "center_objects_id": data["object_id"][tracks].astype(np.int64),
        "center_objects_type": obj_types_all[tracks].astype(np.int32),
        "obj_trajs_future_state": future_state.astype(np.float32),
        "obj_trajs_future_mask": future_mask,
        "center_gt_trajs": center_gt.astype(np.float32),
        "center_gt_trajs_mask": center_gt_mask,
        "center_gt_final_valid_idx": final_idx.astype(np.float32),
        "center_gt_trajs_src": trajs[tracks].astype(np.float32),
        "map_polylines": map_polys,
        "map_polylines_mask": map_masks,
        "map_polylines_center": map_centers,
    }


def collate_official_mtr(batch: list[dict]) -> dict:
    """Collate official MTR samples like the reference DatasetTemplate."""
    input_dict: dict[str, object] = {}
    keys = batch[0].keys()
    pad_second = {
        "obj_trajs",
        "obj_trajs_mask",
        "obj_trajs_pos",
        "obj_trajs_last_pos",
        "obj_trajs_future_state",
        "obj_trajs_future_mask",
        "map_polylines",
        "map_polylines_mask",
        "map_polylines_center",
    }
    concat_np = {"scenario_id", "center_objects_type", "center_objects_id"}

    for key in keys:
        vals = [b[key] for b in batch]
        if key in pad_second:
            tensors = [torch.from_numpy(v) for v in vals]
            max_second = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                if t.shape[1] < max_second:
                    pad_shape = list(t.shape)
                    pad_shape[1] = max_second - t.shape[1]
                    t = torch.cat((t, t.new_zeros(pad_shape)), dim=1)
                padded.append(t)
            input_dict[key] = torch.cat(padded, dim=0)
        elif key in concat_np:
            input_dict[key] = np.concatenate(vals, axis=0)
        elif key in {"obj_types", "obj_ids"}:
            tensors = [torch.from_numpy(v) for v in vals]
            max_second = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                if t.shape[1] < max_second:
                    pad_shape = list(t.shape)
                    pad_shape[1] = max_second - t.shape[1]
                    t = torch.cat((t, t.new_zeros(pad_shape)), dim=1)
                padded.append(t)
            input_dict[key] = torch.cat(padded, dim=0)
        else:
            input_dict[key] = torch.cat([torch.from_numpy(v) for v in vals], dim=0)
    return {
        "batch_size": len(batch),
        "input_dict": input_dict,
        "batch_sample_count": torch.tensor([len(b["track_index_to_predict"]) for b in batch], dtype=torch.long),
    }
