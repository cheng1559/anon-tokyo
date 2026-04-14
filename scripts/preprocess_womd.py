"""Parse WOMD TFRecords into compressed npz files (one per scenario).

Usage (run in .venv-scripts which has tensorflow + waymo SDK + ray):
    source .venv-scripts/bin/activate
    python scripts/preprocess_womd.py \
        --raw_dir data/raw \
        --out_dir data/processed \
        --splits training validation \
        --num_cpus 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import ray
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

# ── type mappings ──────────────────────────────────────────────────────────────

OBJECT_TYPE: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}
# 0=unset/other, 1=vehicle, 2=pedestrian, 3=cyclist

POLYLINE_TYPE: dict[str, int] = {
    "TYPE_UNDEFINED": -1,
    "TYPE_FREEWAY": 1,
    "TYPE_SURFACE_STREET": 2,
    "TYPE_BIKE_LANE": 3,
    "TYPE_BROKEN_SINGLE_WHITE": 6,
    "TYPE_SOLID_SINGLE_WHITE": 7,
    "TYPE_SOLID_DOUBLE_WHITE": 8,
    "TYPE_BROKEN_SINGLE_YELLOW": 9,
    "TYPE_BROKEN_DOUBLE_YELLOW": 10,
    "TYPE_SOLID_SINGLE_YELLOW": 11,
    "TYPE_SOLID_DOUBLE_YELLOW": 12,
    "TYPE_PASSING_DOUBLE_YELLOW": 13,
    "TYPE_ROAD_EDGE_BOUNDARY": 15,
    "TYPE_ROAD_EDGE_MEDIAN": 16,
    "TYPE_STOP_SIGN": 17,
    "TYPE_CROSSWALK": 18,
    "TYPE_SPEED_BUMP": 19,
}

LANE_TYPE: dict[int, str] = {0: "TYPE_UNDEFINED", 1: "TYPE_FREEWAY", 2: "TYPE_SURFACE_STREET", 3: "TYPE_BIKE_LANE"}
ROAD_LINE_TYPE: dict[int, str] = {
    0: "TYPE_UNKNOWN",
    1: "TYPE_BROKEN_SINGLE_WHITE",
    2: "TYPE_SOLID_SINGLE_WHITE",
    3: "TYPE_SOLID_DOUBLE_WHITE",
    4: "TYPE_BROKEN_SINGLE_YELLOW",
    5: "TYPE_BROKEN_DOUBLE_YELLOW",
    6: "TYPE_SOLID_SINGLE_YELLOW",
    7: "TYPE_SOLID_DOUBLE_YELLOW",
    8: "TYPE_PASSING_DOUBLE_YELLOW",
}
ROAD_EDGE_TYPE: dict[int, str] = {0: "TYPE_UNKNOWN", 1: "TYPE_ROAD_EDGE_BOUNDARY", 2: "TYPE_ROAD_EDGE_MEDIAN"}
SIGNAL_STATE: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}

# ── helpers ────────────────────────────────────────────────────────────────────


def _polyline_dir(pts: np.ndarray) -> np.ndarray:
    """Compute unit direction vectors along a polyline.  pts: (N, 3)."""
    prev = np.roll(pts, 1, axis=0)
    prev[0] = pts[0]
    diff = pts - prev
    norm = np.clip(np.linalg.norm(diff, axis=-1, keepdims=True), 1e-6, None)
    return diff / norm


def _decode_tracks(scenario: Any) -> dict[str, np.ndarray]:
    """Extract object tracks.

    Returns:
        object_id:   (A,)    int64   unique agent ids
        object_type: (A,)    int8    0=other,1=vehicle,2=ped,3=cyclist
        trajs:       (A,T,10) float32  [cx,cy,cz, dx,dy,dz, heading, vx,vy, valid]
    """
    ids, types, trajs = [], [], []
    for trk in scenario.tracks:
        states = np.array(
            [
                [
                    s.center_x,
                    s.center_y,
                    s.center_z,
                    s.length,
                    s.width,
                    s.height,
                    s.heading,
                    s.velocity_x,
                    s.velocity_y,
                    s.valid,
                ]
                for s in trk.states
            ],
            dtype=np.float32,
        )
        ids.append(trk.id)
        types.append(OBJECT_TYPE.get(trk.object_type, 0))
        trajs.append(states)
    return {
        "object_id": np.array(ids, dtype=np.int64),
        "object_type": np.array(types, dtype=np.int8),
        "trajs": np.stack(trajs, axis=0),  # (A, T, 10)
    }


def _decode_map(scenario: Any) -> np.ndarray:
    """Extract all map polylines into a single array.

    Each point: [x, y, z, dir_x, dir_y, dir_z, global_type].
    Returns (P, 7) float32.
    """
    polylines: list[np.ndarray] = []
    for feat in scenario.map_features:
        if feat.lane.ByteSize() > 0:
            gtype = POLYLINE_TYPE.get(LANE_TYPE.get(feat.lane.type, "TYPE_UNDEFINED"), -1)
            pts = np.array([[p.x, p.y, p.z] for p in feat.lane.polyline], dtype=np.float32)
        elif feat.road_line.ByteSize() > 0:
            gtype = POLYLINE_TYPE.get(ROAD_LINE_TYPE.get(feat.road_line.type, "TYPE_UNKNOWN"), -1)
            pts = np.array([[p.x, p.y, p.z] for p in feat.road_line.polyline], dtype=np.float32)
        elif feat.road_edge.ByteSize() > 0:
            gtype = POLYLINE_TYPE.get(ROAD_EDGE_TYPE.get(feat.road_edge.type, "TYPE_UNKNOWN"), -1)
            pts = np.array([[p.x, p.y, p.z] for p in feat.road_edge.polyline], dtype=np.float32)
        elif feat.stop_sign.ByteSize() > 0:
            gtype = POLYLINE_TYPE["TYPE_STOP_SIGN"]
            p = feat.stop_sign.position
            pts = np.array([[p.x, p.y, p.z]], dtype=np.float32)
        elif feat.crosswalk.ByteSize() > 0:
            gtype = POLYLINE_TYPE["TYPE_CROSSWALK"]
            pts = np.array([[p.x, p.y, p.z] for p in feat.crosswalk.polygon], dtype=np.float32)
        elif feat.speed_bump.ByteSize() > 0:
            gtype = POLYLINE_TYPE["TYPE_SPEED_BUMP"]
            pts = np.array([[p.x, p.y, p.z] for p in feat.speed_bump.polygon], dtype=np.float32)
        else:
            continue

        dirs = _polyline_dir(pts) if len(pts) > 1 else np.zeros_like(pts)
        typed = np.full((len(pts), 1), gtype, dtype=np.float32)
        polylines.append(np.concatenate([pts, dirs, typed], axis=-1))  # (n, 7)

    if polylines:
        return np.concatenate(polylines, axis=0).astype(np.float32)
    return np.zeros((0, 7), dtype=np.float32)


def _decode_traffic_lights(scenario: Any) -> np.ndarray:
    """Extract per-timestep traffic light signals.

    Returns (T, max_signals, 5) float32 with [lane_id, state, x, y, z], zero-padded.
    """
    all_timesteps: list[np.ndarray] = []
    for ts_data in scenario.dynamic_map_states:
        signals = []
        for sig in ts_data.lane_states:
            signals.append(
                [sig.lane, SIGNAL_STATE.get(sig.state, 0), sig.stop_point.x, sig.stop_point.y, sig.stop_point.z]
            )
        if signals:
            all_timesteps.append(np.array(signals, dtype=np.float32))
        else:
            all_timesteps.append(np.zeros((0, 5), dtype=np.float32))

    if not all_timesteps:
        return np.zeros((0, 0, 5), dtype=np.float32)

    max_signals = max(a.shape[0] for a in all_timesteps)
    T = len(all_timesteps)
    padded = np.zeros((T, max_signals, 5), dtype=np.float32)
    for t, arr in enumerate(all_timesteps):
        if arr.shape[0] > 0:
            padded[t, : arr.shape[0]] = arr
    return padded


def _process_scenario(scenario: Any) -> dict[str, np.ndarray]:
    """Convert a single Scenario proto into a flat dict of numpy arrays."""
    tracks = _decode_tracks(scenario)
    map_polylines = _decode_map(scenario)
    traffic_lights = _decode_traffic_lights(scenario)

    predict_indices = np.array([t.track_index for t in scenario.tracks_to_predict], dtype=np.int32)
    predict_difficulties = np.array([t.difficulty for t in scenario.tracks_to_predict], dtype=np.int32)

    return {
        # metadata
        "scenario_id": np.array(scenario.scenario_id, dtype="U"),
        "timestamps": np.array(scenario.timestamps_seconds, dtype=np.float64),
        "current_time_index": np.array(scenario.current_time_index, dtype=np.int32),
        "sdc_track_index": np.array(scenario.sdc_track_index, dtype=np.int32),
        # tracks
        "object_id": tracks["object_id"],  # (A,)
        "object_type": tracks["object_type"],  # (A,)
        "trajs": tracks["trajs"],  # (A, T, 10)
        # map
        "map_polylines": map_polylines,  # (P, 7)
        # traffic lights
        "traffic_lights": traffic_lights,  # (T, S, 5)
        # prediction targets
        "tracks_to_predict": predict_indices,  # (K,)
        "predict_difficulty": predict_difficulties,  # (K,)
    }


# ── ray worker ─────────────────────────────────────────────────────────────────


@ray.remote
def process_tfrecord(tfrecord_path: str, out_dir: str, skip_existing: bool = False) -> tuple[int, int]:
    """Process one TFRecord file and write per-scenario .npz files.

    Returns (written, skipped) counts.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    written = 0
    skipped = 0
    for data in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))  # type: ignore[union-attr]
        out_path = out / f"{scenario.scenario_id}.npz"  # type: ignore[attr-defined]
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        arrays = _process_scenario(scenario)
        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(out_path)
        written += 1
    return written, skipped


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse WOMD TFRecords → per-scenario .npz")
    parser.add_argument(
        "--raw_dir", type=str, required=True, help="Root of raw data (contains training/, validation/, etc.)"
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output root (will create <split>/ subdirs)")
    parser.add_argument("--splits", nargs="+", default=["training", "validation"], help="Which splits to process")
    parser.add_argument("--num_cpus", type=int, default=32, help="Number of Ray workers")
    parser.add_argument("--skip_existing", action="store_true", help="Skip scenarios whose .npz already exists")
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True)

    for split in args.splits:
        raw_split = Path(args.raw_dir) / split
        out_split = Path(args.out_dir) / split
        out_split.mkdir(parents=True, exist_ok=True)

        tfrecords = sorted(raw_split.glob("*.tfrecord*"))
        if not tfrecords:
            print(f"[WARN] No tfrecord files found in {raw_split}, skipping.")
            continue

        print(f"[{split}] Found {len(tfrecords)} tfrecord files → {out_split}")

        futures = [process_tfrecord.remote(str(tf_path), str(out_split), args.skip_existing) for tf_path in tfrecords]  # type: ignore[call-arg]

        total_written = 0
        total_skipped = 0
        remaining = set(futures)
        with tqdm(total=len(futures), desc=f"Processing {split}", unit="file") as pbar:
            while remaining:
                done, remaining = ray.wait(list(remaining), num_returns=1)
                for ref in done:
                    w, s = ray.get(ref)  # type: ignore[misc]
                    total_written += w
                    total_skipped += s
                    pbar.update(1)

        print(
            f"[{split}] Done: {total_written} written, {total_skipped} skipped, from {len(tfrecords)} files → {out_split}"
        )

    ray.shutdown()


if __name__ == "__main__":
    main()
