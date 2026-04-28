"""Serialize prediction/simulation batches into a small web-viewer format."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from anon_tokyo.simulation.metrics import (
    compute_rollout_metric_tensors,
    serializable_batch_metrics,
    serializable_world_metrics,
)


def _cpu(value: Tensor) -> Tensor:
    return value.detach().cpu()


def _as_list(value: Tensor, digits: int = 3) -> list:
    arr = _cpu(value.float()).numpy()
    return arr.round(digits).tolist()


def _pad_event_frames(events: Tensor, num_frames: int) -> Tensor:
    """Pad ``[T, A]`` event tensors to match rollout frames that include history."""
    if events.shape[0] >= num_frames:
        return events[-num_frames:]
    pad = torch.zeros((num_frames - events.shape[0], events.shape[1]), dtype=events.dtype, device=events.device)
    return torch.cat((pad, events), dim=0)


def _pad_series_frames(values: Tensor, num_frames: int) -> Tensor:
    """Pad ``[T, A]`` scalar series with NaNs for non-simulated history frames."""
    if values.shape[0] >= num_frames:
        return values[-num_frames:]
    pad = torch.full((num_frames - values.shape[0], values.shape[1]), float("nan"), dtype=values.dtype, device=values.device)
    return torch.cat((pad, values), dim=0)


def _pad_current_frame_series(values: Tensor, num_frames: int) -> Tensor:
    """Pad ``[T, A]`` state series so the first value lands on the current frame."""
    if values.shape[0] >= num_frames:
        return values[-num_frames:]
    pad_before = max(num_frames - values.shape[0] - 1, 0)
    pad_after = num_frames - values.shape[0] - pad_before
    before = torch.full((pad_before, values.shape[1]), float("nan"), dtype=values.dtype, device=values.device)
    after = torch.full((pad_after, values.shape[1]), float("nan"), dtype=values.dtype, device=values.device)
    return torch.cat((before, values, after), dim=0)


def _as_optional_list(value: Tensor, digits: int = 4) -> list[float | None]:
    flat = _cpu(value.float()).reshape(-1)
    return [round(float(item.item()), digits) if bool(torch.isfinite(item)) else None for item in flat]


def _sample_int(batch: dict[str, Any], key: str, sample_idx: int) -> int | None:
    value = batch.get(key)
    if value is None:
        return None
    if isinstance(value, Tensor):
        item = value[sample_idx] if value.ndim > 0 else value
        return int(_cpu(item).item())
    if isinstance(value, list):
        return int(value[sample_idx])
    return int(value)


def _valid_lines(polylines: Tensor, mask: Tensor, max_lines: int | None = None) -> list[dict[str, Any]]:
    lines = []
    valid = mask.bool()
    count = int(valid.any(dim=-1).sum().item())
    limit = count if max_lines is None else min(count, max_lines)
    added = 0
    for idx in range(polylines.shape[0]):
        m = valid[idx]
        if not bool(m.any()):
            continue
        points = polylines[idx, m, 0:2]
        line_type = int(polylines[idx, m, 6].float().median().item()) if polylines.shape[-1] > 6 else 0
        lines.append({"type": line_type, "points": _as_list(points)})
        added += 1
        if added >= limit:
            break
    return lines


def _preprocessed_map_records(batch: dict[str, Any], sample_idx: int) -> list[dict[str, Any]]:
    required = (
        "preprocessed_map_polylines",
        "preprocessed_map_mask",
        "preprocessed_map_types",
        "preprocessed_map_batch_idx",
        "preprocessed_map_agent_idx",
        "preprocessed_map_frame",
    )
    if not all(key in batch for key in required):
        return []

    polylines = _cpu(batch["preprocessed_map_polylines"]).float()
    mask = _cpu(batch["preprocessed_map_mask"]).bool()
    types = _cpu(batch["preprocessed_map_types"]).float()
    batch_idx = _cpu(batch["preprocessed_map_batch_idx"]).long()
    agent_idx = _cpu(batch["preprocessed_map_agent_idx"]).long()
    frame_idx = _cpu(batch["preprocessed_map_frame"]).long()
    records = []
    for row in torch.where(batch_idx == sample_idx)[0].tolist():
        valid = mask[row].any(dim=-1)
        lines = [
            {
                "type": int(round(float(line_type[point_mask].median().item()))),
                "points": _as_list(line[point_mask]),
            }
            for line, point_mask, line_type in zip(polylines[row, valid], mask[row, valid], types[row, valid], strict=False)
        ]
        records.append({"frame": int(frame_idx[row].item()), "agent_id": int(agent_idx[row].item()), "polylines": lines})
    return records


def _agent_type_name(obj_type: int) -> str:
    return {1: "vehicle", 2: "pedestrian", 3: "cyclist"}.get(int(obj_type), "unknown")


def _agent_records(batch: dict[str, Any], sample_idx: int) -> list[dict[str, Any]]:
    obj_types = _cpu(batch["obj_types"][sample_idx]).long()
    agent_mask = _cpu(batch.get("agent_mask", torch.ones_like(obj_types, dtype=torch.float32))[sample_idx]).bool()
    hist_xy = _cpu(batch["obj_trajs"][sample_idx, :, :, 0:2])
    hist_mask = _cpu(batch["obj_trajs_mask"][sample_idx]).bool()
    fut_xy = _cpu(batch.get("obj_trajs_future", torch.zeros((*hist_xy.shape[:1], 0, 4)))[sample_idx, :, :, 0:2])
    fut_mask = _cpu(batch.get("obj_trajs_future_mask", torch.zeros((*hist_xy.shape[:1], 0), dtype=torch.bool))[sample_idx]).bool()
    positions = _cpu(batch.get("obj_positions", hist_xy[:, -1])[sample_idx])
    headings = _cpu(batch.get("obj_headings", torch.zeros_like(obj_types, dtype=torch.float32))[sample_idx]).float()
    sizes = _cpu(batch["obj_trajs"][sample_idx, :, :, 3:5])
    tracks_to_predict = set()
    if "tracks_to_predict" in batch:
        tracks_to_predict = {int(x) for x in _cpu(batch["tracks_to_predict"][sample_idx]).long().tolist() if int(x) >= 0}
    sdc_idx = None
    if "sdc_track_index" in batch:
        sdc_tensor = batch["sdc_track_index"]
        sdc_idx = int(_cpu(sdc_tensor[sample_idx] if sdc_tensor.ndim > 0 else sdc_tensor).item())
    controlled = None
    if "controlled_mask" in batch:
        controlled = _cpu(batch["controlled_mask"][sample_idx]).bool()

    agents = []
    for agent_idx in range(obj_types.shape[0]):
        if not bool(agent_mask[agent_idx]):
            continue
        history = hist_xy[agent_idx, hist_mask[agent_idx]]
        future = fut_xy[agent_idx, fut_mask[agent_idx]] if fut_xy.numel() else fut_xy.new_zeros((0, 2))
        valid_sizes = sizes[agent_idx, hist_mask[agent_idx]]
        size = valid_sizes[-1] if valid_sizes.numel() else torch.tensor([4.5, 2.0])
        agents.append(
            {
                "id": agent_idx,
                "type": _agent_type_name(int(obj_types[agent_idx].item())),
                "history": _as_list(history),
                "future": _as_list(future),
                "position": _as_list(positions[agent_idx]),
                "size": _as_list(size),
                "heading": round(float(headings[agent_idx].item()), 4),
                "target": agent_idx in tracks_to_predict,
                "sdc": sdc_idx == agent_idx,
                "controlled": bool(controlled[agent_idx].item()) if controlled is not None else False,
            }
        )
    return agents


def _scenario_base(batch: dict[str, Any], sample_idx: int, *, max_map_lines: int | None = None) -> dict[str, Any]:
    scenario_ids = batch.get("scenario_id")
    scenario_id = scenario_ids[sample_idx] if isinstance(scenario_ids, list) else f"sample_{sample_idx}"
    scenario = {
        "id": str(scenario_id),
        "map": _valid_lines(
            _cpu(batch["map_polylines"][sample_idx]),
            _cpu(batch["map_polylines_mask"][sample_idx]),
            max_lines=max_map_lines,
        ),
        "agents": _agent_records(batch, sample_idx),
    }
    if "goal_positions" in batch:
        goals = []
        goal_pos = _cpu(batch["goal_positions"][sample_idx])
        controlled = _cpu(batch.get("controlled_mask", torch.zeros(goal_pos.shape[:-1]))[sample_idx]).bool()
        for agent_idx in torch.where(controlled)[0].tolist():
            goals.append({"agent_id": int(agent_idx), "point": _as_list(goal_pos[agent_idx])})
        scenario["goals"] = goals
    return scenario


def local_prediction_to_scene(
    pred_trajs: Tensor,
    pred_scores: Tensor,
    batch: dict[str, Tensor],
    *,
    max_modes: int = 6,
) -> list[list[dict[str, Any]]]:
    """Convert query-centric local predictions to scene-frame JSON records."""
    pred_trajs = _cpu(pred_trajs)
    pred_scores = _cpu(pred_scores)
    tracks = _cpu(batch["tracks_to_predict"]).long()
    positions = _cpu(batch["obj_positions"]).float()
    headings = _cpu(batch["obj_headings"]).float()
    B, K = pred_trajs.shape[:2]
    output: list[list[dict[str, Any]]] = []
    for b in range(B):
        sample_preds = []
        for k in range(K):
            agent_idx = int(tracks[b, k].item()) if k < tracks.shape[1] else -1
            if agent_idx < 0 or agent_idx >= positions.shape[1]:
                continue
            theta = headings[b, agent_idx]
            c, s = torch.cos(theta), torch.sin(theta)
            rot = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))
            modes = pred_trajs[b, k, :max_modes, :, 0:2].float()
            scene_xy = modes @ rot.T + positions[b, agent_idx]
            scores = pred_scores[b, k, : scene_xy.shape[0]].float()
            for mode_idx in range(scene_xy.shape[0]):
                sample_preds.append(
                    {
                        "agent_id": agent_idx,
                        "mode": int(mode_idx),
                        "score": round(float(scores[mode_idx].item()), 4),
                        "points": _as_list(scene_xy[mode_idx]),
                    }
                )
        output.append(sample_preds)
    return output


def agent_centric_prediction_to_scene(
    pred_trajs: Tensor,
    pred_scores: Tensor,
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    max_modes: int = 6,
) -> list[list[dict[str, Any]]]:
    """Convert MTR-style agent-centric predictions to scene-frame JSON records."""
    pred_trajs = _cpu(pred_trajs)
    pred_scores = _cpu(pred_scores)
    batch_idx = _cpu(output["batch_idx"]).long()
    track_idx = _cpu(output["track_index_to_predict"]).long()
    positions = _cpu(batch["obj_positions"]).float()
    headings = _cpu(batch["obj_headings"]).float()
    scenarios: list[list[dict[str, Any]]] = [[] for _ in range(positions.shape[0])]
    for row in range(pred_trajs.shape[0]):
        b = int(batch_idx[row].item())
        agent_idx = int(track_idx[row].item())
        if b < 0 or b >= len(scenarios) or agent_idx < 0 or agent_idx >= positions.shape[1]:
            continue
        theta = headings[b, agent_idx]
        c, s = torch.cos(theta), torch.sin(theta)
        rot = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))
        modes = pred_trajs[row, :max_modes, :, 0:2].float()
        scene_xy = modes @ rot.T + positions[b, agent_idx]
        scores = pred_scores[row, : scene_xy.shape[0]].float()
        for mode_idx in range(scene_xy.shape[0]):
            scenarios[b].append(
                {
                    "agent_id": agent_idx,
                    "mode": int(mode_idx),
                    "score": round(float(scores[mode_idx].item()), 4),
                    "points": _as_list(scene_xy[mode_idx]),
                }
            )
    return scenarios


def official_mtr_prediction_to_scene(
    pred_trajs: Tensor,
    pred_scores: Tensor,
    output: dict[str, Any],
    inference_batch: dict[str, Any],
    display_batch: dict[str, Tensor],
    *,
    max_modes: int = 6,
) -> list[list[dict[str, Any]]]:
    """Convert official-MTR local predictions to the display scene frame."""
    pred_trajs = _cpu(pred_trajs)
    pred_scores = _cpu(pred_scores)
    batch_idx = _cpu(output["batch_idx"]).long()

    input_dict = inference_batch["input_dict"]
    center_world = _cpu(input_dict["center_objects_world"]).float()
    center_object_id = input_dict["center_objects_id"]
    if isinstance(center_object_id, Tensor):
        center_object_id = _cpu(center_object_id).long().tolist()
    else:
        center_object_id = [int(x) for x in center_object_id.tolist()]

    display_center_xy = _cpu(display_batch["center_xy"]).float()
    display_center_heading = _cpu(display_batch["center_heading"]).float()
    display_tracks = _cpu(display_batch["tracks_to_predict"]).long()
    display_eval_ids = _cpu(display_batch.get("eval_object_id", torch.empty(0, dtype=torch.long))).long()
    display_eval_raw = _cpu(display_batch.get("eval_raw_track_index", torch.empty(0, dtype=torch.long))).long()

    id_to_agent: list[dict[int, int]] = []
    for b in range(display_tracks.shape[0]):
        mapping: dict[int, int] = {}
        if display_eval_ids.numel():
            for k in range(display_eval_ids.shape[1]):
                if int(display_eval_raw[b, k].item()) < 0:
                    continue
                agent_idx = int(display_tracks[b, k].item())
                if agent_idx >= 0:
                    mapping[int(display_eval_ids[b, k].item())] = agent_idx
        id_to_agent.append(mapping)

    scenarios: list[list[dict[str, Any]]] = [[] for _ in range(display_center_xy.shape[0])]
    for row in range(pred_trajs.shape[0]):
        b = int(batch_idx[row].item())
        if b < 0 or b >= len(scenarios):
            continue
        agent_idx = id_to_agent[b].get(int(center_object_id[row]), -1)
        if agent_idx < 0:
            continue

        center = center_world[row, 0:2]
        heading = center_world[row, 6]
        c, s = torch.cos(heading), torch.sin(heading)
        local_to_world = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))

        scene_heading = -display_center_heading[b]
        c2, s2 = torch.cos(scene_heading), torch.sin(scene_heading)
        world_to_scene = torch.stack((torch.stack((c2, -s2)), torch.stack((s2, c2))))

        modes = pred_trajs[row, :max_modes, :, 0:2].float()
        world_xy = modes @ local_to_world.T + center
        scene_xy = (world_xy - display_center_xy[b]) @ world_to_scene.T
        scores = pred_scores[row, : scene_xy.shape[0]].float()
        for mode_idx in range(scene_xy.shape[0]):
            scenarios[b].append(
                {
                    "agent_id": agent_idx,
                    "mode": int(mode_idx),
                    "score": round(float(scores[mode_idx].item()), 4),
                    "points": _as_list(scene_xy[mode_idx]),
                }
            )
    return scenarios


def serialize_prediction_batch(
    batch: dict[str, Any],
    *,
    predictions: list[list[dict[str, Any]]] | None = None,
    max_map_lines: int | None = None,
) -> dict[str, Any]:
    scenarios = []
    batch_size = int(batch["obj_trajs"].shape[0])
    for sample_idx in range(batch_size):
        scenario = _scenario_base(batch, sample_idx, max_map_lines=max_map_lines)
        scenario["predictions"] = predictions[sample_idx] if predictions is not None else []
        scenarios.append(scenario)
    return {"task": "prediction", "scenarios": scenarios}


def serialize_simulation_batch(
    batch: dict[str, Any],
    *,
    rollout_positions: Tensor | None = None,
    rollout_headings: Tensor | None = None,
    rollout_valid: Tensor | None = None,
    rollout_events: dict[str, Tensor] | None = None,
    rollout_series: dict[str, Tensor] | None = None,
    goal_reaching_threshold: float = 1.5,
    max_map_lines: int | None = None,
) -> dict[str, Any]:
    scenarios = []
    batch_size = int(batch["obj_trajs"].shape[0])
    rollout_metric_tensors = None
    if rollout_events is not None:
        rollout_metric_tensors = compute_rollout_metric_tensors(
            collision=rollout_events["collision"],
            offroad=rollout_events["offroad"],
            goal_reached=rollout_events["goal_reached"],
            controlled_mask=batch["controlled_mask"].bool(),
            agent_mask=batch.get("agent_mask").bool() if "agent_mask" in batch else None,
        )
    for sample_idx in range(batch_size):
        scenario = _scenario_base(batch, sample_idx, max_map_lines=max_map_lines)
        for agent in scenario["agents"]:
            agent["history"] = []
        current_frame = _sample_int(batch, "current_time_index", sample_idx)
        if current_frame is not None:
            scenario["sim_start_frame"] = current_frame + 1
        preprocessed_map = _preprocessed_map_records(batch, sample_idx)
        if preprocessed_map:
            scenario["preprocessed_map"] = preprocessed_map
        if rollout_metric_tensors is not None:
            scenario["metrics"] = serializable_world_metrics(rollout_metric_tensors, sample_idx)
        if rollout_positions is not None:
            positions = _cpu(rollout_positions[sample_idx])
            headings = _cpu(rollout_headings[sample_idx]).float() if rollout_headings is not None else None
            valid = _cpu(rollout_valid[sample_idx]).bool() if rollout_valid is not None else torch.ones(positions.shape[:-1], dtype=torch.bool)
            num_frames = positions.shape[1]
            collision = (
                _pad_event_frames(_cpu(rollout_events["collision"][:, sample_idx]).bool(), num_frames).permute(1, 0)
                if rollout_events is not None
                else torch.zeros_like(valid)
            )
            offroad = (
                _pad_event_frames(_cpu(rollout_events["offroad"][:, sample_idx]).bool(), num_frames).permute(1, 0)
                if rollout_events is not None
                else torch.zeros_like(valid)
            )
            goal_reached_events = (
                _pad_event_frames(_cpu(rollout_events["goal_reached"][:, sample_idx]).bool(), num_frames).permute(1, 0).cumsum(dim=-1) > 0
                if rollout_events is not None
                else None
            )
            reward_series = (
                _pad_series_frames(_cpu(rollout_series["reward"][:, sample_idx]).float(), num_frames).permute(1, 0)
                if rollout_series is not None and "reward" in rollout_series
                else None
            )
            value_series = (
                _pad_current_frame_series(_cpu(rollout_series["value"][:, sample_idx]).float(), num_frames).permute(1, 0)
                if rollout_series is not None and "value" in rollout_series
                else None
            )
            tracks = []
            agent_mask = _cpu(batch.get("agent_mask", torch.ones(positions.shape[0]))[sample_idx]).bool()
            controlled = _cpu(batch.get("controlled_mask", torch.zeros(positions.shape[0]))[sample_idx]).bool()
            goal_pos = _cpu(batch["goal_positions"][sample_idx]).float() if "goal_positions" in batch else None
            for agent_idx in torch.where(agent_mask)[0].tolist():
                track_valid = valid[agent_idx]
                if not bool(track_valid.any()) and not bool(controlled[agent_idx].item()):
                    continue
                keep = torch.ones_like(track_valid, dtype=torch.bool)
                pts = positions[agent_idx, keep]
                record = {
                    "agent_id": int(agent_idx),
                    "points": _as_list(pts),
                    "controlled": bool(controlled[agent_idx].item()),
                    "valid": _cpu(track_valid[keep]).int().tolist(),
                    "collision": _cpu(collision[agent_idx, keep]).int().tolist(),
                    "offroad": _cpu(offroad[agent_idx, keep]).int().tolist(),
                }
                if headings is not None:
                    record["headings"] = _as_list(headings[agent_idx, keep], digits=4)
                if reward_series is not None:
                    record["reward"] = _as_optional_list(reward_series[agent_idx, keep])
                if value_series is not None:
                    record["value"] = _as_optional_list(value_series[agent_idx, keep])
                if goal_pos is not None and bool(controlled[agent_idx].item()):
                    goal = goal_pos[agent_idx]
                    reached = (
                        goal_reached_events[agent_idx, keep]
                        if goal_reached_events is not None
                        else (pts - goal).norm(dim=-1) <= goal_reaching_threshold
                    )
                    record["goal"] = _as_list(goal)
                    record["goal_reached"] = _cpu(reached).int().tolist()
                    reached_idx = torch.where(reached)[0]
                    record["goal_reached_frame"] = int(reached_idx[0].item()) if reached_idx.numel() else None
                tracks.append(record)
            scenario["rollout"] = tracks
        else:
            scenario["rollout"] = []
        scenarios.append(scenario)
    payload = {"task": "simulation", "scenarios": scenarios}
    if rollout_metric_tensors is not None:
        payload["metrics"] = serializable_batch_metrics(rollout_metric_tensors)
    elif rollout_positions is not None:
        total = {"controlled_count": 0, "collision_count": 0, "offroad_count": 0, "goal_reached_count": 0, "done_count": 0}
        for scenario in scenarios:
            controlled_tracks = [track for track in scenario.get("rollout", []) if track.get("controlled")]
            controlled_count = len(controlled_tracks)
            collision_count = sum(1 for track in controlled_tracks if any(track.get("collision", [])))
            offroad_count = sum(1 for track in controlled_tracks if any(track.get("offroad", [])))
            goal_count = sum(1 for track in controlled_tracks if any(track.get("goal_reached", [])))
            done_count = sum(1 for track in controlled_tracks if any(track.get("collision", [])) or any(track.get("offroad", [])))
            denom = max(controlled_count, 1)
            scenario["metrics"] = {
                "controlled_count": controlled_count,
                "collision_count": collision_count,
                "offroad_count": offroad_count,
                "goal_reached_count": goal_count,
                "done_count": done_count,
                "collision_rate": collision_count / denom,
                "offroad_rate": offroad_count / denom,
                "goal_reaching_rate": goal_count / denom,
                "done_rate": done_count / denom,
            }
            total["controlled_count"] += controlled_count
            total["collision_count"] += collision_count
            total["offroad_count"] += offroad_count
            total["goal_reached_count"] += goal_count
            total["done_count"] += done_count
        denom = max(total["controlled_count"], 1)
        payload["metrics"] = {
            **total,
            "collision_rate": total["collision_count"] / denom,
            "offroad_rate": total["offroad_count"] / denom,
            "goal_reaching_rate": total["goal_reached_count"] / denom,
            "done_rate": total["done_count"] / denom,
        }
    return payload
