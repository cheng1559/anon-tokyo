"""WOMD binary shard Dataset."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from anon_tokyo.data.shard_io import ShardIndex, read_item
from anon_tokyo.data.transforms import scene_centric_transform


class WOMDDataset(Dataset):
    """Random-access dataset backed by binary shards or raw .npz files.

    When *use_npz=True*, reads directly from per-scenario ``.npz`` files
    (faster on NVMe).  Otherwise reads from packed binary shards.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        max_agents: int = 128,
        max_polylines: int = 4096,
        num_points_per_polyline: int = 20,
        use_npz: bool = False,
        npz_root: str | Path | None = None,
    ) -> None:
        self.max_agents = max_agents
        self.max_polylines = max_polylines
        self.num_points_per_polyline = num_points_per_polyline
        self.use_npz = use_npz

        self.split_dir = Path(data_root) / split
        self.index = ShardIndex.load(self.split_dir / "index.json")

        if use_npz:
            root = Path(npz_root) / split if npz_root else self.split_dir
            self._npz_paths = [root / f"{sid}.npz" for sid in self.index.scenario_ids]
        else:
            self._shard_paths = [str(self.split_dir / s) for s in self.index.shards]

        # Lazily opened per-worker file descriptors (populated on first read)
        self._shard_fds: dict[int, int] | None = None

    def _get_fd(self, shard_idx: int) -> int:
        if self._shard_fds is None:
            self._shard_fds = {}
        if shard_idx not in self._shard_fds:
            self._shard_fds[shard_idx] = os.open(self._shard_paths[shard_idx], os.O_RDONLY)
        return self._shard_fds[shard_idx]

    def __del__(self) -> None:
        if self._shard_fds:
            for fd in self._shard_fds.values():
                os.close(fd)
            self._shard_fds.clear()

    def __len__(self) -> int:
        return len(self.index)

    def _load_raw(self, idx: int) -> dict[str, np.ndarray]:
        if self.use_npz:
            return dict(np.load(self._npz_paths[idx], allow_pickle=False))
        shard_idx, offset, size = self.index.items[idx]
        fd = self._get_fd(shard_idx)
        return read_item(self._shard_paths[shard_idx], offset, size, fd=fd)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        data = self._load_raw(idx)

        sample = scene_centric_transform(
            data,
            max_agents=self.max_agents,
            max_polylines=self.max_polylines,
            num_points_per_polyline=self.num_points_per_polyline,
        )

        out: dict[str, torch.Tensor | str] = {}
        for key, val in sample.items():
            if isinstance(val, np.ndarray):
                if val.dtype.kind in ("U", "S"):
                    out[key] = str(val)
                else:
                    out[key] = torch.from_numpy(val)
            elif isinstance(val, str):
                out[key] = val
            else:
                out[key] = torch.tensor(val)
        return out
