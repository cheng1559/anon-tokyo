from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from anon_tokyo.data.datamodule import collate_fn
from anon_tokyo.data.mtr_transform import collate_official_mtr
from anon_tokyo.data.womd_dataset import WOMDDataset
from anon_tokyo.simulation.env import ClosedLoopEnv, ClosedLoopEnvConfig
from anon_tokyo.visualize.serialize import (
    agent_centric_prediction_to_scene,
    local_prediction_to_scene,
    official_mtr_prediction_to_scene,
    serialize_prediction_batch,
    serialize_simulation_batch,
)

SPLITS = ("training", "validation", "testing")
SIMULATION_CONTROL_MODES = ("tracks_to_predict", "sdc", "ego", "non_reactive", "all_agents", "all")


def _import_class(path: str):
    module_name, _, class_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected fully-qualified class path, got {path!r}")
    return getattr(importlib.import_module(module_name), class_name)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


class WebVisualizerService:
    """Small stateful service matching the Hermes web visualizer shape."""

    def __init__(
        self,
        *,
        task: str,
        config_path: str,
        checkpoint_path: str | None = None,
        split: str | None = None,
        batch_size: int = 4,
        simulation_control_mode: str | None = None,
    ) -> None:
        self.task = task
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.split = split
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = _load_yaml(config_path)
        self.simulation_control_mode = self._validate_simulation_control_mode(
            simulation_control_mode or self._default_simulation_control_mode()
        )
        self.dataset = self._build_display_dataset()
        self.inference_dataset = self._build_inference_dataset()
        self.inference_transform = self._data_transform(self.inference_dataset)
        self.model: torch.nn.Module | None = None
        self._batch_cache: dict[int, dict[str, Any]] = {}
        self._inference_batch_cache: dict[int, dict[str, Any]] = {}
        self._payload_cache: dict[int, dict[str, Any]] = {}

    @classmethod
    def from_env(cls) -> "WebVisualizerService":
        return cls(
            task=os.environ.get("ANON_TOKYO_VIS_TASK", "prediction"),
            config_path=os.environ.get("ANON_TOKYO_VIS_CONFIG", "configs/prediction/anon_tokyo.yaml"),
            checkpoint_path=os.environ.get("ANON_TOKYO_VIS_CHECKPOINT") or None,
            split=os.environ.get("ANON_TOKYO_VIS_SPLIT") or None,
            batch_size=int(os.environ.get("ANON_TOKYO_VIS_BATCH_SIZE", "4")),
            simulation_control_mode=os.environ.get("ANON_TOKYO_VIS_SIMULATION_CONTROL_MODE") or None,
        )

    def initialize_env(self, **payload: Any) -> dict[str, Any]:
        self.task = payload.get("task", self.task)
        self.config_path = payload.get("config_path", payload.get("config", self.config_path))
        self.checkpoint_path = payload.get("checkpoint_path", payload.get("checkpoint", self.checkpoint_path))
        self.split = payload.get("split", self.split)
        if self.split is not None and self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        self.batch_size = int(payload.get("batch_size", self.batch_size))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = _load_yaml(self.config_path)
        control_mode = payload.get("simulation_control_mode", payload.get("control_mode"))
        self.simulation_control_mode = self._validate_simulation_control_mode(
            control_mode or self._default_simulation_control_mode()
        )
        self.dataset = self._build_display_dataset()
        self.inference_dataset = self._build_inference_dataset()
        self.inference_transform = self._data_transform(self.inference_dataset)
        self.model = None
        self._batch_cache.clear()
        self._inference_batch_cache.clear()
        self._payload_cache.clear()
        return self.fetch_env()

    def fetch_env(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "split": self.split or self._default_split(),
            "batch_size": self.batch_size,
            "dataset_size": len(self.dataset),
            "simulation_control_mode": self.simulation_control_mode,
        }

    def fetch_files(self) -> dict[str, list[str]]:
        return {
            "prediction_configs": self._list_files("configs/prediction", {".yaml", ".yml"}),
            "simulation_configs": self._list_files("configs/simulation", {".yaml", ".yml"}),
            "checkpoints": self._list_files("checkpoints", {".ckpt", ".pt", ".pth"})
            + self._list_files("tb_logs", {".ckpt", ".pt", ".pth"}, limit=1000),
            "splits": list(SPLITS),
        }

    def fetch_batch(self, batch_idx: int, batch_size: int | None = None) -> dict[str, Any]:
        if batch_size is not None and int(batch_size) != self.batch_size:
            self.batch_size = int(batch_size)
            self._batch_cache.clear()
            self._payload_cache.clear()
        if batch_idx not in self._payload_cache:
            if self.task == "prediction":
                batch, inference_batch = self._load_prediction_batch(batch_idx)
                payload = self._serialize_batch(batch, inference_batch=inference_batch)
                self._inference_batch_cache[batch_idx] = inference_batch
            else:
                batch = self._load_batch(self.dataset, batch_idx, collate_fn)
                payload = self._serialize_batch(batch)
            self._batch_cache[batch_idx] = batch
            self._payload_cache[batch_idx] = payload
        return self._payload_cache[batch_idx]

    def fetch_world(self, batch_idx: int, world_idx: int) -> dict[str, Any]:
        payload = self.fetch_batch(batch_idx)
        return payload["scenarios"][world_idx]

    def rollout_world(self, batch_idx: int, world_idx: int, count: int | None = None) -> dict[str, Any]:
        payload = self.fetch_batch(batch_idx)
        world = payload["scenarios"][world_idx]
        if count is not None and "rollout" in world:
            for track in world["rollout"]:
                track["points"] = track["points"][: int(count)]
        return world

    def _default_split(self) -> str:
        if self.task == "simulation":
            return self.cfg.get("data", {}).get("split", "training")
        return "validation"

    def _default_simulation_control_mode(self) -> str:
        return self.cfg.get("data", {}).get("simulation_control_mode", "tracks_to_predict")

    def _validate_simulation_control_mode(self, value: Any) -> str:
        mode = str(value)
        if mode not in SIMULATION_CONTROL_MODES:
            raise ValueError(f"simulation_control_mode must be one of {SIMULATION_CONTROL_MODES}, got {mode!r}")
        return mode

    def _list_files(self, root: str, suffixes: set[str], limit: int = 300) -> list[str]:
        root_path = Path(root)
        if not root_path.exists():
            return []
        files = [str(path) for path in root_path.rglob("*") if path.is_file() and path.suffix in suffixes]
        files.sort(key=lambda item: Path(item).stat().st_mtime, reverse=True)
        return files[:limit]

    def _base_data_cfg(self) -> tuple[dict[str, Any], str]:
        data_cfg = dict(self.cfg.get("data") or {})
        config_split = data_cfg.pop("split", self._default_split())
        split = self.split or config_split
        data_cfg.pop("batch_size", None)
        data_cfg.pop("num_workers", None)
        return data_cfg, split

    def _build_display_dataset(self) -> WOMDDataset:
        data_cfg, split = self._base_data_cfg()
        if self.task == "prediction":
            data_cfg["transform"] = "scene"
            data_cfg["include_eval_meta"] = True
        elif self.task == "simulation":
            data_cfg["transform"] = "simulation"
            data_cfg["simulation_control_mode"] = self.simulation_control_mode
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        return WOMDDataset(split=split, **data_cfg)

    def _build_inference_dataset(self) -> WOMDDataset:
        if self.task != "prediction":
            return self.dataset
        data_cfg, split = self._base_data_cfg()
        return WOMDDataset(split=split, **data_cfg)

    def _data_transform(self, dataset: WOMDDataset) -> str:
        return getattr(dataset, "transform", "scene")

    def _collate_prediction(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.inference_transform == "mtr_official":
            return collate_official_mtr(samples)
        return collate_fn(samples)

    def _load_batch(self, dataset: WOMDDataset, batch_idx: int, collate: Any) -> dict[str, Any]:
        start = batch_idx * self.batch_size
        if start >= len(dataset):
            raise IndexError(f"batch_idx {batch_idx} is outside dataset size {len(dataset)}")
        samples = [dataset[(start + offset) % len(dataset)] for offset in range(self.batch_size)]
        return collate(samples)

    def _load_prediction_batch(self, batch_idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
        display = self._load_batch(self.dataset, batch_idx, collate_fn)
        inference = self._load_batch(self.inference_dataset, batch_idx, self._collate_prediction)
        return display, inference

    def _build_model(self) -> torch.nn.Module | None:
        if self.checkpoint_path is None:
            return None
        if self.model is not None:
            return self.model
        if self.task == "prediction":
            model_cfg = self.cfg["model"]["model"]
        else:
            model_cfg = self.cfg["model"]
        cls = _import_class(model_cfg["class_path"])
        model = cls(**dict(model_cfg.get("init_args") or {})).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state = checkpoint.get("state_dict") or checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint
        clean_state = {}
        for key, value in state.items():
            while key.startswith(("model.", "module.")):
                if key.startswith("model."):
                    key = key[len("model.") :]
                elif key.startswith("module."):
                    key = key[len("module.") :]
            clean_state[key] = value
        model.load_state_dict(clean_state, strict=False)
        model.eval()
        self.model = model
        return model

    def _serialize_batch(self, batch: dict[str, Any], *, inference_batch: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.task == "prediction":
            predictions = self._predict(batch, inference_batch or batch)
            return serialize_prediction_batch(batch, predictions=predictions, max_map_lines=512)
        rollout_positions, rollout_headings, rollout_valid, enriched_batch, rollout_events, rollout_series = self._simulate(batch)
        return serialize_simulation_batch(
            enriched_batch,
            rollout_positions=rollout_positions,
            rollout_headings=rollout_headings,
            rollout_valid=rollout_valid,
            rollout_events=rollout_events,
            rollout_series=rollout_series,
            goal_reaching_threshold=float(self.cfg.get("env", {}).get("rewards", {}).get("goal_reaching_threshold", 1.5)),
            max_map_lines=512,
        )

    @torch.no_grad()
    def _predict(self, display_batch: dict[str, Any], inference_batch: dict[str, Any]) -> list[list[dict[str, Any]]] | None:
        model = self._build_model()
        if model is None:
            return None
        inference_dev = _to_device(inference_batch, self.device)
        output = model(inference_dev)
        if "input_dict" in inference_batch and self.inference_transform == "mtr_official":
            return official_mtr_prediction_to_scene(
                output["pred_trajs"],
                output["pred_scores"],
                output,
                inference_dev,
                display_batch,
            )
        if "batch_idx" in output and "track_index_to_predict" in output:
            return agent_centric_prediction_to_scene(output["pred_trajs"], output["pred_scores"], output, display_batch)
        return local_prediction_to_scene(output["pred_trajs"], output["pred_scores"], display_batch)

    @torch.no_grad()
    def _simulate(
        self,
        batch: dict[str, Any],
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, Any],
        dict[str, torch.Tensor] | None,
        dict[str, torch.Tensor] | None,
    ]:
        env_cfg = dict(self.cfg.get("env") or {})
        env_cfg["device"] = str(self.device)
        env = ClosedLoopEnv(ClosedLoopEnvConfig.from_dict(env_cfg))
        obs = env.reset(batch)
        model = self._build_model()
        if model is None:
            assert env.log_kinematics is not None and env.log_mask is not None and env.batch is not None and env.goal_positions is not None
            enriched = dict(env.batch)
            enriched["goal_positions"] = env.goal_positions
            return env.log_kinematics["positions"], env.log_kinematics["headings"], env.log_mask, enriched, None, None
        collision_steps = []
        offroad_steps = []
        goal_reached_steps = []
        reward_steps = []
        value_steps = []
        for _ in range(env.episode_steps):
            action, _, _, value = model(obs)
            obs, reward, _, info = env.step(action)
            assert env.goal_reached is not None
            collision_steps.append(info["collision"].detach())
            offroad_steps.append(info["offroad"].detach())
            goal_reached_steps.append(env.goal_reached.detach())
            reward_steps.append(reward.detach())
            value_steps.append(value.detach())
        assert env.positions is not None and env.headings is not None and env.valid is not None and env.batch is not None and env.goal_positions is not None
        enriched = dict(env.batch)
        enriched["goal_positions"] = env.goal_positions
        rollout_events = {
            "collision": torch.stack(collision_steps, dim=0),
            "offroad": torch.stack(offroad_steps, dim=0),
            "goal_reached": torch.stack(goal_reached_steps, dim=0),
        }
        rollout_series = {
            "reward": torch.stack(reward_steps, dim=0),
            "value": torch.stack(value_steps, dim=0),
        }
        return env.positions, env.headings, env.valid, enriched, rollout_events, rollout_series
