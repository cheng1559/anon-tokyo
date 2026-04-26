"""Closed-loop simulation PPO entrypoint."""

from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from anon_tokyo.data.datamodule import collate_fn
from anon_tokyo.data.womd_dataset import WOMDDataset
from anon_tokyo.simulation.env import ClosedLoopEnv, ClosedLoopEnvConfig
from anon_tokyo.simulation.ppo import PPOConfig, PPOTrainer
from anon_tokyo.simulation.profiling import TimingProfiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop PPO training")
    parser.add_argument("--config", type=str, default="configs/simulation/agent_centric_ppo.yaml")
    parser.add_argument("--policy_class", type=str, default=None, help="Override model.class_path, e.g. package.module.Class")
    parser.add_argument("--num_updates", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--npz_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_agents", type=int, default=None)
    parser.add_argument("--max_polylines", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=None, help="Override both env.num_steps and ppo.num_steps")
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--optimization_epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke_env", action="store_true", help="Run reset + one zero-action env step without a policy")
    parser.add_argument("--profile", action="store_true", help="Emit fine-grained timing metrics for one or more updates")
    parser.add_argument(
        "--profile_cuda_sync",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Synchronize CUDA around profile regions for more accurate GPU timings",
    )
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA in this entrypoint")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return is_distributed, rank, local_rank, world_size, device


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_rank_zero(rank: int) -> bool:
    return rank == 0


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int, rank: int = 0) -> None:
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def import_class(path: str):
    module_name, _, class_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected fully-qualified class path, got {path!r}")
    return getattr(importlib.import_module(module_name), class_name)


def build_dataloader(cfg: dict[str, Any], *, is_distributed: bool, rank: int, world_size: int, seed: int) -> DataLoader:
    data_cfg = dict(cfg["data"])
    split = data_cfg.pop("split", "training")
    batch_size = data_cfg.pop("batch_size", 16)
    num_workers = data_cfg.pop("num_workers", 4)
    dataset = WOMDDataset(split=split, **data_cfg)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        if is_distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def build_policy(cfg: dict[str, Any], override_class: str | None) -> torch.nn.Module:
    model_cfg = dict(cfg.get("model") or {})
    class_path = override_class or model_cfg.get("class_path")
    if not class_path:
        raise ValueError(
            "No simulation policy model configured. Set model.class_path in the YAML "
            "or pass --policy_class. Use --smoke_env to validate data/env without a policy."
        )
    cls = import_class(class_path)
    return cls(**dict(model_cfg.get("init_args") or {}))


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.num_updates is not None:
        cfg.setdefault("ppo", {})["num_updates"] = args.num_updates
    if args.data_root is not None:
        cfg.setdefault("data", {})["data_root"] = args.data_root
    if args.npz_root is not None:
        cfg.setdefault("data", {})["npz_root"] = args.npz_root
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg.setdefault("data", {})["num_workers"] = args.num_workers
    if args.max_agents is not None:
        cfg.setdefault("data", {})["max_agents"] = args.max_agents
    if args.max_polylines is not None:
        cfg.setdefault("data", {})["max_polylines"] = args.max_polylines
    if args.rollout_steps is not None:
        cfg.setdefault("env", {})["num_steps"] = args.rollout_steps
        cfg.setdefault("ppo", {})["num_steps"] = args.rollout_steps
    if args.minibatch_size is not None:
        cfg.setdefault("ppo", {})["minibatch_size"] = args.minibatch_size
    if args.optimization_epochs is not None:
        cfg.setdefault("ppo", {})["optimization_epochs"] = args.optimization_epochs
    if args.device is not None:
        cfg.setdefault("env", {})["device"] = args.device
    if args.profile:
        cfg.setdefault("ppo", {})["profile"] = True
    if args.profile_cuda_sync is not None:
        cfg.setdefault("ppo", {})["profile_cuda_sync"] = args.profile_cuda_sync


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_metrics(metrics: dict[str, float], is_distributed: bool, device: torch.device) -> dict[str, float]:
    if not is_distributed:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.AVG)
    return {key: float(value.detach().cpu()) for key, value in zip(keys, values)}


def top_profile_seconds(metrics: dict[str, float], limit: int = 16) -> dict[str, float]:
    items = [
        (key.removeprefix("profile_").removesuffix("_seconds"), value)
        for key, value in metrics.items()
        if key.startswith("profile_") and key.endswith("_seconds")
    ]
    items.sort(key=lambda item: item[1], reverse=True)
    return {key: round(value, 4) for key, value in items[:limit]}


def warn_single_process_heavy_config(cfg: dict[str, Any], world_size: int) -> None:
    if world_size > 1:
        return
    data_cfg = cfg.get("data", {})
    ppo_cfg = cfg.get("ppo", {})
    batch_size = int(data_cfg.get("batch_size", 16))
    minibatch_size = ppo_cfg.get("minibatch_size")
    if batch_size >= 512 or (minibatch_size is not None and int(minibatch_size) >= 10000):
        print(
            {
                "warning": "single-process run with large per-GPU PPO config; use torchrun --nproc_per_node=8 or lower batch/minibatch",
                "batch_size_per_gpu": batch_size,
                "minibatch_size": minibatch_size,
            }
        )


def main() -> None:
    args = parse_args()
    is_distributed, rank, local_rank, world_size, device = setup_distributed()
    writer: SummaryWriter | None = None
    try:
        cfg = load_config(args.config)
        apply_overrides(cfg, args)
        seed = int(cfg.get("seed_everything", 42))
        seed_everything(seed, rank)

        if is_distributed:
            cfg.setdefault("env", {})["device"] = str(device)
        loader = build_dataloader(cfg, is_distributed=is_distributed, rank=rank, world_size=world_size, seed=seed)
        env = ClosedLoopEnv(ClosedLoopEnvConfig.from_dict(cfg.get("env")))
        device = env.device
        batch = next(iter(loader))

        if args.smoke_env:
            if is_rank_zero(rank):
                per_gpu_batch = int(cfg["data"].get("batch_size", 16))
                print(
                    {
                        "ddp": is_distributed,
                        "rank": rank,
                        "world_size": world_size,
                        "local_rank": local_rank,
                        "device": str(device),
                        "batch_size_per_gpu": per_gpu_batch,
                        "global_batch_size": per_gpu_batch * world_size,
                    }
                )
            profiler = TimingProfiler(
                enabled=bool(cfg.get("ppo", {}).get("profile", False)),
                device=env.device,
                sync_cuda=bool(cfg.get("ppo", {}).get("profile_cuda_sync", True)),
            )
            env.profiler = profiler if profiler.enabled else None
            with profiler.record("smoke.env_reset"):
                obs = env.reset(batch)
            actions = torch.zeros(obs["controlled_mask"].shape[0], obs["controlled_mask"].shape[1], 2, device=env.device)
            with profiler.record("smoke.env_step"):
                _, reward, done, info = env.step(actions)
            metrics = {
                "reward_mean": float(reward.mean().detach().cpu()),
                "done_count": int(done.sum().detach().cpu()),
                "controlled_count": int(obs["controlled_mask"].sum().detach().cpu()),
                "collision_count": int(info["collision"].sum().detach().cpu()),
            }
            metrics.update(profiler.metrics())
            if is_rank_zero(rank):
                metrics = reduce_metrics(metrics, is_distributed, device)
                print(metrics)
                if profiler.enabled:
                    print({"profile_top_seconds": top_profile_seconds(metrics)})
            elif is_distributed:
                reduce_metrics(metrics, is_distributed, device)
            return

        policy = build_policy(cfg, args.policy_class).to(device)
        if is_distributed:
            policy = DDP(policy, device_ids=[local_rank], output_device=local_rank)
        ppo_cfg = PPOConfig.from_dict(cfg.get("ppo"))
        if ppo_cfg.num_updates is None:
            if ppo_cfg.total_epochs is None:
                raise ValueError("Set either ppo.num_updates or ppo.total_epochs")
            ppo_cfg.num_updates = ppo_cfg.total_epochs * len(loader)
        trainer = PPOTrainer(env=env, policy=policy, config=ppo_cfg)
        if is_rank_zero(rank):
            warn_single_process_heavy_config(cfg, world_size)
            per_gpu_batch = int(cfg["data"].get("batch_size", 16))
            print(
                {
                    "ddp": is_distributed,
                    "rank": rank,
                    "world_size": world_size,
                    "local_rank": local_rank,
                    "device": str(device),
                    "batch_size_per_gpu": per_gpu_batch,
                    "global_batch_size": per_gpu_batch * world_size,
                    "batches_per_epoch": len(loader),
                    "total_epochs": ppo_cfg.total_epochs,
                    "num_updates": ppo_cfg.num_updates,
                }
            )
        log_dir = Path(cfg.get("trainer", {}).get("log_dir", "tb_logs/simulation"))
        if is_rank_zero(rank):
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(log_dir))
        save_interval = int(cfg.get("trainer", {}).get("save_interval", 100))
        log_interval = int(cfg.get("trainer", {}).get("log_interval", 20))

        sampler = loader.sampler if isinstance(loader.sampler, DistributedSampler) else None
        if sampler is not None:
            sampler.set_epoch(0)
        data_iter = iter(loader)
        updates = range(1, ppo_cfg.num_updates + 1)
        progress = tqdm(updates, desc="ppo", dynamic_ncols=True, disable=not is_rank_zero(rank))
        for update in progress:
            data_t0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(update)
                data_iter = iter(loader)
                batch = next(data_iter)
            data_seconds = time.perf_counter() - data_t0
            metrics = trainer.train_one_update(batch)
            metrics["data_seconds"] = data_seconds
            metrics = reduce_metrics(metrics, is_distributed, device)
            if is_rank_zero(rank):
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, update)
                writer.add_scalar("train/learning_rate", trainer.optimizer.param_groups[0]["lr"], update)
                progress.set_postfix(
                    reward=f"{metrics.get('mean_reward', 0.0):.4f}",
                    policy=f"{metrics.get('policy_loss', 0.0):.4f}",
                    value=f"{metrics.get('value_loss', 0.0):.4f}",
                    kl=f"{metrics.get('approx_kl', 0.0):.5f}",
                    data=f"{metrics.get('data_seconds', 0.0):.1f}s",
                    rollout=f"{metrics.get('rollout_seconds', 0.0):.1f}s",
                    upd=f"{metrics.get('update_seconds', 0.0):.1f}s",
                    coll=f"{metrics.get('collision_rate', 0.0):.3f}",
                    goal=f"{metrics.get('goal_reaching_rate', 0.0):.3f}",
                    off=f"{metrics.get('offroad_rate', 0.0):.3f}",
                )
                if ppo_cfg.profile:
                    progress.write(str({"update": update, "profile_top_seconds": top_profile_seconds(metrics)}))
            if is_rank_zero(rank) and update % log_interval == 0:
                progress.write(str({"update": update, **metrics}))
                writer.flush()
            if is_rank_zero(rank) and update % save_interval == 0:
                torch.save(
                    {
                        "update": update,
                        "model_state_dict": unwrap_model(policy).state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "config": cfg,
                    },
                    log_dir / f"checkpoint_{update}.pt",
                )
                writer.flush()
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
