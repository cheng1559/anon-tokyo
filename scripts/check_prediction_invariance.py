"""Check prediction invariance under raw-world rigid perturbations.

All models are evaluated through the same path:
1. Load raw WOMD scenarios.
2. Apply a global translation or rotation in raw world coordinates.
3. Re-run the configured preprocessing transform.
4. Compare local predicted trajectories and scores against the unperturbed run.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from tqdm import tqdm

from anon_tokyo.data.datamodule import WOMDDataModule, collate_fn
from anon_tokyo.data.mtr_transform import collate_official_mtr, official_mtr_transform
from anon_tokyo.data.transforms import scene_centric_transform
from anon_tokyo.prediction.lit_module import PredictionModule


DEFAULT_CKPT = (
    "/high_perf_store3/l3_data/huchengcheng/anon-tokyo-v2/"
    "tb_logs/prediction_anon_tokyo/anon_tokyo_v3/checkpoints/last.ckpt"
)
DEFAULT_CONFIG = (
    "/high_perf_store3/l3_data/huchengcheng/anon-tokyo-v2/"
    "tb_logs/prediction_anon_tokyo/anon_tokyo_v3/config.yaml"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check prediction translation/rotation invariance.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--split", default="validation", choices=["validation", "testing"])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--translation_range", type=float, nargs=2, default=[-50.0, 50.0])
    p.add_argument("--rotation_range", type=float, nargs=2, default=[0.0, 2.0 * math.pi])
    return p.parse_args()


def instantiate_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]["model"]
    module_path, _, cls_name = model_cfg["class_path"].rpartition(".")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**model_cfg.get("init_args", {}))


def to_device(x: Any, device: str) -> Any:
    if isinstance(x, Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def clone_raw(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in data.items()}


def rotate_np_xy(xy: np.ndarray, angle: float) -> np.ndarray:
    c = np.float32(math.cos(angle))
    s = np.float32(math.sin(angle))
    x = xy[..., 0].copy()
    y = xy[..., 1].copy()
    return np.stack((x * c - y * s, x * s + y * c), axis=-1).astype(np.float32)


def translate_raw_world(data: dict[str, np.ndarray], delta_xy: np.ndarray) -> dict[str, np.ndarray]:
    perturbed = clone_raw(data)
    delta_xy = delta_xy.astype(np.float32, copy=False)
    perturbed["trajs"][..., 0:2] += delta_xy
    perturbed["map_polylines"][..., 0:2] += delta_xy
    return perturbed


def rotate_raw_world(data: dict[str, np.ndarray], angle: float) -> dict[str, np.ndarray]:
    perturbed = clone_raw(data)
    perturbed["trajs"][..., 0:2] = rotate_np_xy(perturbed["trajs"][..., 0:2], angle)
    perturbed["trajs"][..., 7:9] = rotate_np_xy(perturbed["trajs"][..., 7:9], angle)
    perturbed["trajs"][..., 6] += np.float32(angle)
    perturbed["map_polylines"][..., 0:2] = rotate_np_xy(perturbed["map_polylines"][..., 0:2], angle)
    perturbed["map_polylines"][..., 3:5] = rotate_np_xy(perturbed["map_polylines"][..., 3:5], angle)
    return perturbed


def valid_prediction_mask(batch: dict[str, Any], output: dict[str, Tensor]) -> Tensor:
    scores = output["pred_scores"]
    if "input_dict" in batch:
        return torch.ones_like(scores, dtype=torch.bool)
    ttp = batch["tracks_to_predict"].to(device=scores.device)
    if bool(output.get("pred_is_target_agents", False)):
        return (ttp[:, : scores.shape[1]] >= 0).unsqueeze(-1).expand_as(scores)
    return batch["agent_mask"].to(device=scores.device).bool().unsqueeze(-1).expand_as(scores)


def compare_outputs(base: dict[str, Tensor], perturbed: dict[str, Tensor], mask: Tensor) -> dict[str, float]:
    traj_mask = mask[..., None, None].expand_as(base["pred_trajs"][..., 0:2])
    score_abs = (base["pred_scores"] - perturbed["pred_scores"]).abs()[mask]
    traj_abs = (base["pred_trajs"][..., 0:2] - perturbed["pred_trajs"][..., 0:2]).abs()[traj_mask]
    return {
        "traj_mae_m": float(traj_abs.mean().item()) if traj_abs.numel() else 0.0,
        "traj_max_m": float(traj_abs.max().item()) if traj_abs.numel() else 0.0,
        "score_mae": float(score_abs.mean().item()) if score_abs.numel() else 0.0,
        "score_max": float(score_abs.max().item()) if score_abs.numel() else 0.0,
    }


def update_sum(dst: dict[str, float], src: dict[str, float], count: int) -> None:
    for key, value in src.items():
        if key.endswith("_max_m") or key.endswith("_max"):
            dst[key] = max(dst.get(key, 0.0), value)
        else:
            dst[key] = dst.get(key, 0.0) + value * count


def print_results(args: argparse.Namespace, totals: dict[str, dict[str, float]], counts: dict[str, int], seen: int) -> None:
    print(f"checkpoint: {Path(args.ckpt)}")
    print(f"split={args.split}, batches={args.max_batches}, valid target/mode scores={seen}")
    print("perturbation,traj_mae_m,traj_max_m,score_mae,score_max")
    for name in sorted(totals):
        denom = max(counts[name], 1)
        stats = totals[name]
        print(
            f"{name},"
            f"{stats.get('traj_mae_m', 0.0) / denom:.9g},"
            f"{stats.get('traj_max_m', 0.0):.9g},"
            f"{stats.get('score_mae', 0.0) / denom:.9g},"
            f"{stats.get('score_max', 0.0):.9g}"
        )


def numpy_to_torch_sample(sample: dict[str, Any]) -> dict[str, Tensor | str]:
    out: dict[str, Tensor | str] = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            if value.dtype.kind in ("U", "S"):
                out[key] = str(value)
            else:
                out[key] = torch.from_numpy(value)
        elif isinstance(value, str):
            out[key] = value
        else:
            out[key] = torch.tensor(value)
    return out


def make_batch_from_raw(raw_items: list[dict[str, np.ndarray]], data_cfg: dict[str, Any]) -> dict[str, Any]:
    transform = data_cfg.get("transform", "scene")
    if transform == "mtr_official":
        samples = [
            official_mtr_transform(
                raw,
                max_polylines=data_cfg.get("max_polylines", 768),
                num_points_per_polyline=data_cfg.get("num_points_per_polyline", 20),
            )
            for raw in raw_items
        ]
        return collate_official_mtr(samples)
    if transform != "scene":
        raise ValueError(f"Unsupported prediction transform for invariance check: {transform}")

    samples = [
        numpy_to_torch_sample(
            scene_centric_transform(
                raw,
                max_agents=data_cfg.get("max_agents", 128),
                max_polylines=data_cfg.get("max_polylines", 768),
                num_points_per_polyline=data_cfg.get("num_points_per_polyline", 20),
                include_eval_meta=False,
            )
        )
        for raw in raw_items
    ]
    return collate_fn(samples)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = instantiate_model(config)
    lit_module = PredictionModule.load_from_checkpoint(args.ckpt, model=model, strict=True)
    lit_module.eval().to(args.device)

    data_cfg = config["data"]
    dm = WOMDDataModule(
        data_root=data_cfg.get("data_root", "data/shards"),
        batch_size=args.batch_size,
        num_workers=0,
        max_agents=data_cfg.get("max_agents", 128),
        max_polylines=data_cfg.get("max_polylines", 768),
        num_points_per_polyline=data_cfg.get("num_points_per_polyline", 20),
        use_npz=data_cfg.get("use_npz", False),
        npz_root=data_cfg.get("npz_root"),
        transform=data_cfg.get("transform", "scene"),
        include_eval_meta=False,
    )
    dm.setup("validate" if args.split == "validation" else "test")
    dataset = dm.val_dataset if args.split == "validation" else dm.test_dataset
    rng = np.random.default_rng(args.seed)

    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    seen = 0
    with torch.inference_mode():
        for batch_idx in tqdm(range(args.max_batches), desc="Checking"):
            start = batch_idx * args.batch_size
            raw_items = [dataset._load_raw(i) for i in range(start, min(start + args.batch_size, len(dataset)))]
            if not raw_items:
                break

            base_batch = to_device(make_batch_from_raw(raw_items, data_cfg), args.device)
            base = lit_module.model(base_batch)
            mask = valid_prediction_mask(base_batch, base)
            count = int(mask.sum().item())
            if count == 0:
                continue

            lo_t, hi_t = args.translation_range
            delta_xy = rng.uniform(lo_t, hi_t, size=2).astype(np.float32)
            name = f"raw_world_translation_random_{lo_t:g}_{hi_t:g}m_xy"
            perturbed_batch = to_device(
                make_batch_from_raw([translate_raw_world(raw, delta_xy) for raw in raw_items], data_cfg),
                args.device,
            )
            out = lit_module.model(perturbed_batch)
            stats = compare_outputs(base, out, mask)
            update_sum(totals.setdefault(name, {}), stats, count)
            counts[name] = counts.get(name, 0) + count

            lo_r, hi_r = args.rotation_range
            angle = float(rng.uniform(lo_r, hi_r))
            name = f"raw_world_rotation_random_{lo_r:g}_{hi_r:g}rad"
            perturbed_batch = to_device(
                make_batch_from_raw([rotate_raw_world(raw, angle) for raw in raw_items], data_cfg),
                args.device,
            )
            out = lit_module.model(perturbed_batch)
            stats = compare_outputs(base, out, mask)
            update_sum(totals.setdefault(name, {}), stats, count)
            counts[name] = counts.get(name, 0) + count
            seen += count

    print_results(args, totals, counts, seen)


if __name__ == "__main__":
    main()
