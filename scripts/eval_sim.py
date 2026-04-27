"""Evaluate closed-loop simulation PPO checkpoints.

Runs deterministic rollouts on a WOMD split and reports aggregate rollout
metrics. This script uses the main PyTorch environment.

Usage:
    uv run python scripts/eval_sim.py \
        --config configs/simulation/anon_tokyo_ppo.yaml \
        --ckpt tb_logs/simulation_anon_tokyo/checkpoint_100.pt \
        --split validation \
        --output results/simulation_eval.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from anon_tokyo.data.datamodule import collate_fn
from anon_tokyo.data.womd_dataset import WOMDDataset
from anon_tokyo.simulation.env import ClosedLoopEnv, ClosedLoopEnvConfig
from anon_tokyo.simulation.ppo import PPOConfig, PPOTrainer, masked_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop simulation evaluation")
    parser.add_argument("--config", required=True, help="Simulation config YAML")
    parser.add_argument("--ckpt", required=True, help="PPO checkpoint path")
    parser.add_argument("--split", default="validation", choices=["training", "validation", "testing"])
    parser.add_argument("--output", default=None, help="Optional JSON metrics output path")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="Limit evaluation batches for smoke checks")
    parser.add_argument("--rollout_steps", type=int, default=None, help="Override env.num_steps and ppo.num_steps")
    parser.add_argument("--sampling_method", default="mean", choices=["mean", "deterministic", "mode", "sample"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--policy_class", default=None, help="Override model.class_path")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def import_class(path: str):
    module_name, _, class_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected fully-qualified class path, got {path!r}")
    return getattr(importlib.import_module(module_name), class_name)


def build_policy(cfg: dict[str, Any], override_class: str | None) -> torch.nn.Module:
    model_cfg = dict(cfg.get("model") or {})
    class_path = override_class or model_cfg.get("class_path")
    if not class_path:
        raise ValueError("No simulation policy model configured. Set model.class_path or pass --policy_class.")
    cls = import_class(class_path)
    return cls(**dict(model_cfg.get("init_args") or {}))


def clean_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    clean_state = {}
    for key, value in state.items():
        while key.startswith(("model.", "module.")):
            if key.startswith("model."):
                key = key[len("model.") :]
            elif key.startswith("module."):
                key = key[len("module.") :]
        clean_state[key] = value
    return clean_state


def load_checkpoint(policy: torch.nn.Module, path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict") or checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint
    missing, unexpected = policy.load_state_dict(clean_state_dict(state), strict=False)
    if missing:
        print(f"Warning: missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected checkpoint keys: {len(unexpected)}")


def build_dataloader(cfg: dict[str, Any], split: str) -> DataLoader:
    data_cfg = dict(cfg["data"])
    batch_size = data_cfg.pop("batch_size", 16)
    num_workers = data_cfg.pop("num_workers", 4)
    data_cfg.pop("split", None)
    dataset = WOMDDataset(split=split, **data_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    cfg.setdefault("data", {})["split"] = args.split
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.rollout_steps is not None:
        cfg.setdefault("env", {})["num_steps"] = args.rollout_steps
        cfg.setdefault("ppo", {})["num_steps"] = args.rollout_steps
    if args.device is not None:
        cfg.setdefault("env", {})["device"] = args.device


def mean_metrics(metric_sums: dict[str, float], num_batches: int) -> dict[str, float]:
    return {key: value / max(num_batches, 1) for key, value in sorted(metric_sums.items())}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args)

    seed = int(cfg.get("seed_everything", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loader = build_dataloader(cfg, args.split)
    env = ClosedLoopEnv(ClosedLoopEnvConfig.from_dict(cfg.get("env")))
    policy = build_policy(cfg, args.policy_class).to(env.device)
    load_checkpoint(policy, args.ckpt, env.device)
    policy.eval()

    ppo_cfg = PPOConfig.from_dict(cfg.get("ppo"))
    trainer = PPOTrainer(env=env, policy=policy, config=ppo_cfg)

    metric_sums: dict[str, float] = {}
    num_batches = 0
    num_scenarios = 0
    num_rollout_steps = 0
    progress = tqdm(loader, desc="Evaluating simulation", dynamic_ncols=True)
    for batch in progress:
        buffer, _, _, rollout_metrics = trainer.collect_rollout(batch, sampling_method=args.sampling_method)
        valid_mask = buffer.train_masks
        metrics = dict(rollout_metrics)
        metrics["mean_reward"] = float(masked_mean(buffer.rewards, valid_mask).detach().cpu()) if valid_mask.any() else 0.0
        metrics["rollout_steps"] = float(buffer.size)

        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        num_batches += 1
        num_scenarios += int(buffer.num_envs)
        num_rollout_steps += int(buffer.size)
        progress.set_postfix(
            reward=f"{metrics.get('mean_reward', 0.0):.4f}",
            coll=f"{metrics.get('collision_rate', 0.0):.3f}",
            goal=f"{metrics.get('goal_reaching_rate', 0.0):.3f}",
            off=f"{metrics.get('offroad_rate', 0.0):.3f}",
        )
        if args.max_batches is not None and num_batches >= args.max_batches:
            break

    if num_batches == 0:
        raise RuntimeError("No batches evaluated")

    results = mean_metrics(metric_sums, num_batches)
    results.update(
        {
            "config": args.config,
            "checkpoint": args.ckpt,
            "split": args.split,
            "sampling_method": args.sampling_method,
            "num_batches": num_batches,
            "num_scenarios": num_scenarios,
            "avg_rollout_steps": num_rollout_steps / num_batches,
        }
    )

    print("\n========== Simulation Evaluation Results ==========")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print("===================================================\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved simulation metrics to {out_path}")


if __name__ == "__main__":
    main()
