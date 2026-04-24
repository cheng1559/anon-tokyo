"""Export model predictions to pickle for WOMD evaluation.

Step 1 of the evaluation pipeline. Runs in the main venv (with torch).
Loads a checkpoint, runs inference on validation/testing data, transforms
predictions to world coordinates, and saves them alongside raw GT metadata.

Usage:
    uv run python scripts/export_predictions.py \
        --config configs/prediction/mtr_baseline.yaml \
        --ckpt checkpoints/last.ckpt \
        --split validation \
        --output predictions.pkl
"""

from __future__ import annotations

import argparse
import importlib
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor
from tqdm import tqdm

from anon_tokyo.data.datamodule import WOMDDataModule
from anon_tokyo.prediction.lit_module import PredictionModule

INT_TO_WOMD_TYPE: dict[int, str] = {
    0: "TYPE_UNSET",
    1: "TYPE_VEHICLE",
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_CYCLIST",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export predictions for WOMD eval")
    p.add_argument("--config", required=True, help="Training config YAML")
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--split", default="validation", choices=["validation", "testing"])
    p.add_argument("--output", default="predictions.pkl")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--npz_root", default="data/processed")
    return p.parse_args()


def _instantiate_model(config: dict) -> torch.nn.Module:
    model_cfg = config["model"]["model"]
    class_path = model_cfg["class_path"]
    init_args = model_cfg.get("init_args", {})
    module_path, _, cls_name = class_path.rpartition(".")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**init_args)


def _rotate_to_world(
    pred_xy: Tensor,
    agent_heading: Tensor,
    agent_pos: Tensor,
    center_heading: Tensor,
    center_xy: Tensor,
) -> Tensor:
    """Agent-local frame → world frame.  pred_xy: ``[..., 2]``."""
    cos_a, sin_a = torch.cos(agent_heading), torch.sin(agent_heading)
    x, y = pred_xy[..., 0], pred_xy[..., 1]
    sdc_x = x * cos_a - y * sin_a + agent_pos[..., 0]
    sdc_y = x * sin_a + y * cos_a + agent_pos[..., 1]

    cos_c, sin_c = torch.cos(center_heading), torch.sin(center_heading)
    world_x = sdc_x * cos_c - sdc_y * sin_c + center_xy[..., 0]
    world_y = sdc_x * sin_c + sdc_y * cos_c + center_xy[..., 1]
    return torch.stack([world_x, world_y], dim=-1)


def _process_agent_centric(
    output: dict[str, Tensor],
    batch: dict[str, Tensor | list[str]],
    npz_root: Path,
    device: str,
) -> list[dict]:
    """Collect predictions from MTR (agent-centric) model output."""
    pred_trajs = output["pred_trajs"]  # [K_total, M, T, 7]
    # MTR decoder returns softmaxed/NMS scores in eval mode, matching official.
    pred_scores = output["pred_scores"]  # [K_total, M]
    if "input_dict" in batch:
        input_dict = batch["input_dict"]
        center_world = input_dict["center_objects_world"].to(device)
        pred_world = pred_trajs.clone()
        h = center_world[:, 6]
        c, s = torch.cos(h), torch.sin(h)
        x, y = pred_world[..., 0].clone(), pred_world[..., 1].clone()
        pred_world[..., 0] = x * c[:, None, None] - y * s[:, None, None] + center_world[:, None, None, 0]
        pred_world[..., 1] = x * s[:, None, None] + y * c[:, None, None] + center_world[:, None, None, 1]

        results: list[dict] = []
        scenario_ids = input_dict["scenario_id"]
        center_ids = input_dict["center_objects_id"]
        center_types = input_dict["center_objects_type"]
        gt_src = input_dict["center_gt_trajs_src"]
        for k in range(pred_world.shape[0]):
            ctype = int(center_types[k]) if not isinstance(center_types[k], str) else center_types[k]
            results.append(
                {
                    "scenario_id": str(scenario_ids[k]),
                    "pred_trajs": pred_world[k, :, :, 0:2].cpu().numpy(),
                    "pred_scores": pred_scores[k].cpu().numpy(),
                    "object_id": int(center_ids[k]),
                    "object_type": INT_TO_WOMD_TYPE.get(ctype, str(ctype)),
                    "gt_trajs": gt_src[k].cpu().numpy(),
                    "track_index_to_predict": k,
                }
            )
        return results

    track_idx = output["track_index_to_predict"]  # [K_total]
    batch_sample_count = output["batch_sample_count"]  # [B]

    pred_xy = pred_trajs[:, :, :, 0:2]
    B = len(batch["scenario_id"])
    center_xy = batch["center_xy"]
    center_heading = batch["center_heading"]
    results: list[dict] = []

    start_k = 0
    for b in range(B):
        scenario_id = batch["scenario_id"][b]
        n_agents = int(batch_sample_count[b].item())
        raw = np.load(str(npz_root / f"{scenario_id}.npz"), allow_pickle=False)
        raw_ttp = raw["tracks_to_predict"]

        for ki in range(n_agents):
            k = start_k + ki
            ti = int(track_idx[k].item())

            a_pos = batch["obj_positions"][b, ti].to(device)
            a_head = batch["obj_headings"][b, ti].to(device)
            c_xy = center_xy[b].to(device)
            c_head = center_heading[b].to(device)

            world_xy = _rotate_to_world(pred_xy[k], a_head, a_pos, c_head, c_xy)

            raw_idx = int(raw_ttp[ki])
            results.append(
                {
                    "scenario_id": scenario_id,
                    "pred_trajs": world_xy.cpu().numpy(),
                    "pred_scores": pred_scores[k].cpu().numpy(),
                    "object_id": int(raw["object_id"][raw_idx]),
                    "object_type": INT_TO_WOMD_TYPE.get(int(raw["object_type"][raw_idx]), "TYPE_UNSET"),
                    "gt_trajs": raw["trajs"][raw_idx],
                    "track_index_to_predict": ki,
                }
            )
        start_k += n_agents

    return results


def _process_scene_centric(
    output: dict[str, Tensor],
    batch: dict[str, Tensor | list[str]],
    npz_root: Path,
    device: str,
) -> list[dict]:
    """Collect predictions from scene-centric model output."""
    pred_trajs_all = output["pred_trajs"]  # [B, A, M, T, 7] or legacy [B, K, M, T, 7]
    pred_scores_all = output["pred_scores"]  # eval-mode scores are already softmaxed/NMSed

    ttp = batch["tracks_to_predict"]  # [B, K_max]
    B = pred_trajs_all.shape[0]
    K = ttp.shape[1]
    center_xy = batch["center_xy"]
    center_heading = batch["center_heading"]
    results: list[dict] = []

    for b in range(B):
        scenario_id = batch["scenario_id"][b]
        raw = np.load(str(npz_root / f"{scenario_id}.npz"), allow_pickle=False)
        raw_ttp = raw["tracks_to_predict"]

        for ki in range(K):
            if ttp[b, ki].item() < 0:
                continue
            ti = int(ttp[b, ki].item())

            a_pos = batch["obj_positions"][b, ti].to(device)
            a_head = batch["obj_headings"][b, ti].to(device)
            c_xy = center_xy[b].to(device)
            c_head = center_heading[b].to(device)

            pred_agent_idx = ki if bool(output.get("pred_is_target_agents", False)) else ti
            pred_xy = pred_trajs_all[b, pred_agent_idx, :, :, 0:2]
            pred_scores = pred_scores_all[b, pred_agent_idx]
            world_xy = _rotate_to_world(pred_xy, a_head, a_pos, c_head, c_xy)

            raw_idx = int(raw_ttp[ki])
            results.append(
                {
                    "scenario_id": scenario_id,
                    "pred_trajs": world_xy.cpu().numpy(),
                    "pred_scores": pred_scores.cpu().numpy(),
                    "object_id": int(raw["object_id"][raw_idx]),
                    "object_type": INT_TO_WOMD_TYPE.get(int(raw["object_type"][raw_idx]), "TYPE_UNSET"),
                    "gt_trajs": raw["trajs"][raw_idx],
                    "track_index_to_predict": ki,
                }
            )

    return results


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = _instantiate_model(config)
    lit_module = PredictionModule.load_from_checkpoint(
        args.ckpt,
        model=model,
        strict=True,
    )
    lit_module.eval()
    lit_module.to(args.device)

    data_cfg = config["data"]
    dm = WOMDDataModule(
        data_root=data_cfg.get("data_root", "data/shards"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_agents=data_cfg.get("max_agents", 128),
        max_polylines=data_cfg.get("max_polylines", 768),
        num_points_per_polyline=data_cfg.get("num_points_per_polyline", 20),
        use_npz=data_cfg.get("use_npz", False),
        npz_root=data_cfg.get("npz_root"),
        transform=data_cfg.get("transform", "scene"),
    )
    stage = "validate" if args.split == "validation" else "test"
    dm.setup(stage)
    dataloader = dm.val_dataloader() if args.split == "validation" else dm.test_dataloader()
    npz_root = Path(args.npz_root) / args.split

    all_preds: list[dict] = []
    def to_device(x):
        if isinstance(x, Tensor):
            return x.to(args.device)
        if isinstance(x, dict):
            return {k: to_device(v) for k, v in x.items()}
        return x

    for batch in tqdm(dataloader, desc="Predicting"):
        batch_dev = to_device(batch)
        with torch.no_grad():
            output = lit_module.model(batch_dev)

        is_ac = "batch_idx" in output or "input_dict" in output
        fn = _process_agent_centric if is_ac else _process_scene_centric
        all_preds.extend(fn(output, batch, npz_root, args.device))

    # Convert numpy arrays to plain lists so the pickle is readable
    # by the older numpy (1.x) in .venv-scripts.
    for d in all_preds:
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_preds, f)
    print(f"Saved {len(all_preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()
