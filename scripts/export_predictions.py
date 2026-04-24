"""Export model predictions to NPZ for WOMD evaluation.

Step 1 of the evaluation pipeline. Runs in the main venv (with torch).
Loads a checkpoint, runs inference on validation/testing data, transforms
predictions to world coordinates, and saves them alongside raw GT metadata.

Usage:
    uv run python scripts/export_predictions.py \
        --config configs/prediction/mtr_baseline.yaml \
        --ckpt checkpoints/last.ckpt \
        --split validation \
        --output predictions.npz
"""

from __future__ import annotations

import argparse
import importlib
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
WOMD_TYPE_TO_INT: dict[str, int] = {v: k for k, v in INT_TO_WOMD_TYPE.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export predictions for WOMD eval")
    p.add_argument("--config", required=True, help="Training config YAML")
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--split", default="validation", choices=["validation", "testing"])
    p.add_argument("--output", default="predictions.npz")
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

    valid_b, valid_k = torch.nonzero(ttp >= 0, as_tuple=True)
    if valid_b.numel() == 0:
        return results

    valid_tidx = ttp[valid_b, valid_k].long()
    pred_agent_idx = valid_k if bool(output.get("pred_is_target_agents", False)) else valid_tidx

    out_device = pred_trajs_all.device
    valid_b_dev = valid_b.to(out_device)
    pred_agent_idx_dev = pred_agent_idx.to(out_device)

    pred_xy = pred_trajs_all[valid_b_dev, pred_agent_idx_dev, :, :, 0:2]
    pred_scores = pred_scores_all[valid_b_dev, pred_agent_idx_dev]

    a_pos = batch["obj_positions"][valid_b, valid_tidx].to(device)
    a_head = batch["obj_headings"][valid_b, valid_tidx].to(device)
    c_xy = center_xy[valid_b].to(device)
    c_head = center_heading[valid_b].to(device)

    world_xy = _rotate_to_world(
        pred_xy,
        a_head[:, None, None],
        a_pos[:, None, None, :],
        c_head[:, None, None],
        c_xy[:, None, None, :],
    )
    world_xy_np = world_xy.cpu().numpy()
    pred_scores_np = pred_scores.cpu().numpy()

    valid_b_np = valid_b.numpy()
    valid_k_np = valid_k.numpy()
    has_eval_meta = all(k in batch for k in ("eval_object_id", "eval_object_type", "eval_gt_trajs"))
    raw_by_batch: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    if not has_eval_meta:
        for b in range(B):
            scenario_id = batch["scenario_id"][b]
            raw = np.load(str(npz_root / f"{scenario_id}.npz"), allow_pickle=False)
            raw_by_batch[b] = (raw["tracks_to_predict"], raw["object_id"], raw["object_type"], raw["trajs"])

    for i, (b, ki) in enumerate(zip(valid_b_np, valid_k_np, strict=True)):
        scenario_id = batch["scenario_id"][int(b)]
        if has_eval_meta:
            object_id = int(batch["eval_object_id"][b, ki])
            object_type = int(batch["eval_object_type"][b, ki])
            gt_trajs = batch["eval_gt_trajs"][b, ki].numpy()
        else:
            raw_ttp, raw_object_id, raw_object_type, raw_trajs = raw_by_batch[int(b)]
            raw_idx = int(raw_ttp[int(ki)])
            object_id = int(raw_object_id[raw_idx])
            object_type = int(raw_object_type[raw_idx])
            gt_trajs = raw_trajs[raw_idx]
        results.append(
            {
                "scenario_id": scenario_id,
                "pred_trajs": world_xy_np[i],
                "pred_scores": pred_scores_np[i],
                "object_id": object_id,
                "object_type": INT_TO_WOMD_TYPE.get(object_type, "TYPE_UNSET"),
                "gt_trajs": gt_trajs,
                "track_index_to_predict": int(ki),
            }
        )

    return results


def _save_predictions_npz(preds: list[dict], out_path: Path) -> None:
    if out_path.suffix != ".npz":
        raise ValueError(f"Prediction output must be a .npz file, got: {out_path}")
    if not preds:
        raise ValueError("No predictions to save")

    pred_trajs = np.stack([p["pred_trajs"] for p in preds]).astype(np.float32, copy=False)
    pred_scores = np.stack([p["pred_scores"] for p in preds]).astype(np.float32, copy=False)

    first_gt = np.asarray(preds[0]["gt_trajs"])
    num_gt_frames = first_gt.shape[0]
    gt_trajectory = np.empty((len(preds), num_gt_frames, 7), dtype=np.float32)
    gt_is_valid = np.empty((len(preds), num_gt_frames), dtype=np.bool_)
    object_id = np.empty(len(preds), dtype=np.int64)
    object_type = np.empty(len(preds), dtype=np.int64)
    track_index_to_predict = np.empty(len(preds), dtype=np.int32)
    scenario_id = np.empty(len(preds), dtype=f"U{max(len(str(p['scenario_id'])) for p in preds)}")

    for i, pred in enumerate(preds):
        raw_gt = np.asarray(pred["gt_trajs"])
        gt_trajectory[i] = raw_gt[:, [0, 1, 3, 4, 6, 7, 8]]
        gt_is_valid[i] = raw_gt[:, -1].astype(bool)
        object_id[i] = int(pred["object_id"])
        object_type[i] = WOMD_TYPE_TO_INT.get(str(pred["object_type"]), 0)
        track_index_to_predict[i] = int(pred["track_index_to_predict"])
        scenario_id[i] = str(pred["scenario_id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        scenario_id=scenario_id,
        object_id=object_id,
        object_type=object_type,
        track_index_to_predict=track_index_to_predict,
        pred_trajs=pred_trajs,
        pred_scores=pred_scores,
        gt_trajectory=gt_trajectory,
        gt_is_valid=gt_is_valid,
    )


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
        include_eval_meta=data_cfg.get("transform", "scene") != "mtr_official",
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
            return {k: v if k.startswith("eval_") else to_device(v) for k, v in x.items()}
        return x

    for batch in tqdm(dataloader, desc="Predicting"):
        batch_dev = to_device(batch)
        with torch.no_grad():
            output = lit_module.model(batch_dev)

        is_ac = "batch_idx" in output or "input_dict" in output
        fn = _process_agent_centric if is_ac else _process_scene_centric
        all_preds.extend(fn(output, batch, npz_root, args.device))

    out_path = Path(args.output)
    _save_predictions_npz(all_preds, out_path)
    print(f"Saved {len(all_preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()
