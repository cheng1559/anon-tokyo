"""LightningDataModule for WOMD."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from anon_tokyo.data.womd_dataset import WOMDDataset


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Stack tensors, collect strings into lists."""
    out: dict[str, Any] = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


class WOMDDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str = "data/shards",
        batch_size: int = 16,
        num_workers: int = 8,
        max_agents: int = 128,
        max_polylines: int = 4096,
        num_points_per_polyline: int = 20,
        use_npz: bool = False,
        npz_root: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._ds_kwargs = dict(
            data_root=self.data_root,
            max_agents=max_agents,
            max_polylines=max_polylines,
            num_points_per_polyline=num_points_per_polyline,
            use_npz=use_npz,
            npz_root=npz_root,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = WOMDDataset(split="training", **self._ds_kwargs)
            self.val_dataset = WOMDDataset(split="validation", **self._ds_kwargs)
        if stage == "validate":
            self.val_dataset = WOMDDataset(split="validation", **self._ds_kwargs)
        if stage == "test":
            self.test_dataset = WOMDDataset(split="testing", **self._ds_kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
