"""Evaluate predictions using official WOMD metrics (minADE, minFDE, MissRate, mAP).

Step 2 of the evaluation pipeline. Runs in .venv-scripts (TF + waymo SDK).

Usage:
    .venv-scripts/bin/python scripts/eval_womd.py \
        --predictions predictions.npz \
        --eval_second 8 \
        --num_modes 6
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# TF + Waymo SDK imports (available only in .venv-scripts)
try:
    import tensorflow as tf
    from google.protobuf import text_format
    from waymo_open_dataset.metrics.ops import py_metrics_ops
    from waymo_open_dataset.metrics.python import config_util_py as config_util
    from waymo_open_dataset.protos import motion_metrics_pb2
except ImportError:
    print(
        "ERROR: tensorflow / waymo-open-dataset not found.\n"
        "Run this script in .venv-scripts:\n"
        "  .venv-scripts/bin/python scripts/eval_womd.py --predictions predictions.npz",
        file=sys.stderr,
    )
    sys.exit(1)

# Limit TF GPU memory growth
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


OBJECT_TYPE_TO_ID = {
    "TYPE_UNSET": 0,
    "TYPE_VEHICLE": 1,
    "TYPE_PEDESTRIAN": 2,
    "TYPE_CYCLIST": 3,
    "TYPE_OTHER": 4,
}
ID_TO_OBJECT_TYPE = {v: k for k, v in OBJECT_TYPE_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WOMD evaluation")
    p.add_argument("--predictions", required=True, help="Path to predictions.npz")
    p.add_argument("--eval_second", type=int, default=8, choices=[3, 5, 8])
    p.add_argument("--num_modes", type=int, default=6)
    return p.parse_args()


def _default_metrics_config(eval_second: int, num_modes: int = 6) -> motion_metrics_pb2.MotionMetricsConfig:
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = f"""
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 10
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    step_configurations {{
      measurement_step: 5
      lateral_miss_threshold: 1.0
      longitudinal_miss_threshold: 2.0
    }}
    max_predictions: {num_modes}
    """
    if eval_second == 3:
        config_text += "track_future_samples: 30\n"
    elif eval_second == 5:
        config_text += """
        track_future_samples: 50
        step_configurations {
          measurement_step: 9
          lateral_miss_threshold: 1.8
          longitudinal_miss_threshold: 3.6
        }
        """
    else:
        config_text += """
        track_future_samples: 80
        step_configurations {
          measurement_step: 9
          lateral_miss_threshold: 1.8
          longitudinal_miss_threshold: 3.6
        }
        step_configurations {
          measurement_step: 15
          lateral_miss_threshold: 3.0
          longitudinal_miss_threshold: 6.0
        }
        """
    text_format.Parse(config_text, config)
    return config


def transform_npz_to_waymo_format(
    pred_npz: np.lib.npyio.NpzFile, top_k: int = -1, eval_second: int = 8
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Convert exported NPZ arrays to arrays expected by WOMD metrics op."""
    scenario_ids_flat = pred_npz["scenario_id"]
    pred_trajs_flat = pred_npz["pred_trajs"]
    pred_scores_flat = pred_npz["pred_scores"]
    gt_trajectory_flat = pred_npz["gt_trajectory"]
    gt_is_valid_flat = pred_npz["gt_is_valid"]
    object_type_flat = pred_npz["object_type"]
    object_id_flat = pred_npz["object_id"]

    print(f"Total predictions: {len(pred_trajs_flat)}")

    unique_sids, scene_inverse = np.unique(scenario_ids_flat, return_inverse=True)
    num_scenario = len(unique_sids)
    scene_counts = np.bincount(scene_inverse, minlength=num_scenario)
    max_objs = int(scene_counts.max())
    topK, _, _ = pred_trajs_flat.shape[1:]

    if top_k > 0:
        topK = min(top_k, topK)

    sampled_interval = 5
    if eval_second == 3:
        num_frames_total = 41
        num_frame_eval = 6
    elif eval_second == 5:
        num_frames_total = 61
        num_frame_eval = 10
    else:
        num_frames_total = 91
        num_frame_eval = 16

    batch_pred_trajs = np.zeros((num_scenario, max_objs, topK, 1, num_frame_eval, 2), dtype=np.float32)
    batch_pred_scores = np.zeros((num_scenario, max_objs, topK), dtype=np.float32)
    gt_trajs = np.zeros((num_scenario, max_objs, num_frames_total, 7), dtype=np.float32)
    gt_is_valid = np.zeros((num_scenario, max_objs, num_frames_total), dtype=int)
    pred_gt_idxs = np.zeros((num_scenario, max_objs, 1))
    pred_gt_idx_mask = np.zeros((num_scenario, max_objs, 1), dtype=int)
    object_type = np.zeros((num_scenario, max_objs), dtype=np.int64)
    object_id = np.zeros((num_scenario, max_objs), dtype=int)

    type_cnt: dict[str, int] = {k: 0 for k in OBJECT_TYPE_TO_ID}
    obj_offsets = np.zeros(num_scenario, dtype=np.int64)

    for i in range(len(pred_trajs_flat)):
        scene_idx = scene_inverse[i]
        obj_idx = obj_offsets[scene_idx]
        obj_offsets[scene_idx] += 1

        sort_idx = pred_scores_flat[i].argsort()[::-1]
        scores = pred_scores_flat[i, sort_idx]
        trajs = pred_trajs_flat[i, sort_idx]
        scores = scores / scores.sum()

        sampled = trajs[:topK, np.newaxis, 4::sampled_interval, :][:, :, :num_frame_eval, :]
        batch_pred_trajs[scene_idx, obj_idx] = sampled
        batch_pred_scores[scene_idx, obj_idx] = scores[:topK]

        gt_trajs[scene_idx, obj_idx] = gt_trajectory_flat[i, :num_frames_total]
        gt_is_valid[scene_idx, obj_idx] = gt_is_valid_flat[i, :num_frames_total].astype(int)

        pred_gt_idxs[scene_idx, obj_idx, 0] = obj_idx
        pred_gt_idx_mask[scene_idx, obj_idx, 0] = 1
        obj_type_int = int(object_type_flat[i])
        object_type[scene_idx, obj_idx] = obj_type_int
        object_id[scene_idx, obj_idx] = int(object_id_flat[i])

        obj_type_str = ID_TO_OBJECT_TYPE.get(obj_type_int, "TYPE_UNSET")
        if obj_type_str in type_cnt:
            type_cnt[obj_type_str] += 1

    gt_infos = {
        "scenario_id": unique_sids.tolist(),
        "object_id": object_id.tolist(),
        "object_type": object_type.tolist(),
        "gt_is_valid": gt_is_valid,
        "gt_trajectory": gt_trajs,
        "pred_gt_indices": pred_gt_idxs,
        "pred_gt_indices_mask": pred_gt_idx_mask,
    }
    return batch_pred_scores, batch_pred_trajs, gt_infos, type_cnt


def waymo_evaluation(
    pred_npz: np.lib.npyio.NpzFile,
    top_k: int = -1,
    eval_second: int = 8,
    num_modes: int = 6,
) -> tuple[dict, str]:
    pred_score, pred_traj, gt_infos, type_cnt = transform_npz_to_waymo_format(
        pred_npz, top_k=top_k, eval_second=eval_second
    )
    eval_config = _default_metrics_config(eval_second, num_modes)

    metric_results = py_metrics_ops.motion_metrics(
        config=eval_config.SerializeToString(),
        prediction_trajectory=tf.convert_to_tensor(pred_traj, tf.float32),
        prediction_score=tf.convert_to_tensor(pred_score, tf.float32),
        ground_truth_trajectory=tf.convert_to_tensor(gt_infos["gt_trajectory"], tf.float32),
        ground_truth_is_valid=tf.convert_to_tensor(gt_infos["gt_is_valid"], tf.bool),
        prediction_ground_truth_indices=tf.convert_to_tensor(gt_infos["pred_gt_indices"], tf.int64),
        prediction_ground_truth_indices_mask=tf.convert_to_tensor(gt_infos["pred_gt_indices_mask"], tf.bool),
        object_type=tf.convert_to_tensor(gt_infos["object_type"], tf.int64),
    )

    metric_names = config_util.get_breakdown_names_from_motion_config(eval_config)
    result_dict: dict[str, float] = {}
    avg_accum: dict[str, list[float]] = {}
    metric_labels = ["minADE", "minFDE", "MissRate", "OverlapRate", "mAP"]
    type_labels = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]

    for m in metric_labels:
        for t in type_labels:
            avg_accum[f"{m} - {t}"] = [0.0, 0.0]

    for i, m in enumerate(metric_labels):
        for j, n in enumerate(metric_names):
            cur_type = n.split("_")[1]
            val = float(metric_results[i][j])
            avg_accum[f"{m} - {cur_type}"][0] += val
            avg_accum[f"{m} - {cur_type}"][1] += 1
            result_dict[f"{m} - {n}"] = val

    avg_results: dict[str, float] = {}
    for key, (total, cnt) in avg_accum.items():
        avg_results[key] = total / cnt if cnt > 0 else 0.0

    result_dict.update(avg_results)

    # Summary table
    header = f"{'Waymo':>12} {'mAP':>12} {'minADE':>12} {'minFDE':>12} {'MissRate':>12}"
    rows = [header]
    final_avg: dict[str, float] = {}
    for m in ["mAP", "minADE", "minFDE", "MissRate"]:
        final_avg[m] = 0.0
        for t in type_labels:
            final_avg[m] += avg_results.get(f"{m} - {t}", 0.0)
        final_avg[m] /= len(type_labels)

    for t in type_labels:
        rows.append(
            f"{t:>12} "
            f"{avg_results.get(f'mAP - {t}', 0.0):>12.4f} "
            f"{avg_results.get(f'minADE - {t}', 0.0):>12.4f} "
            f"{avg_results.get(f'minFDE - {t}', 0.0):>12.4f} "
            f"{avg_results.get(f'MissRate - {t}', 0.0):>12.4f}"
        )
    rows.append(
        f"{'Avg':>12} "
        f"{final_avg['mAP']:>12.4f} "
        f"{final_avg['minADE']:>12.4f} "
        f"{final_avg['minFDE']:>12.4f} "
        f"{final_avg['MissRate']:>12.4f}"
    )
    result_str = "\n".join(rows)

    result_dict.update(final_avg)
    result_dict.update(type_cnt)
    return result_dict, result_str


def main() -> None:
    args = parse_args()

    pred_npz = np.load(args.predictions, allow_pickle=False)
    print(f"Loaded predictions from {args.predictions}")

    result_dict, result_str = waymo_evaluation(
        pred_npz,
        eval_second=args.eval_second,
        num_modes=args.num_modes,
    )

    print("\n========== WOMD Evaluation Results ==========")
    print(result_str)
    print("==============================================\n")

    for k, v in result_dict.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
