"""Lightweight timing helpers for simulation profiling."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import time
from typing import Iterator

import torch


class TimingProfiler:
    """Accumulate wall-clock timings with optional CUDA synchronization."""

    def __init__(self, enabled: bool = False, device: torch.device | str | None = None, sync_cuda: bool = True) -> None:
        self.enabled = enabled
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.sync_cuda = sync_cuda
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self.seconds.clear()
        self.calls.clear()

    def _sync(self) -> None:
        if self.sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1

    def metrics(self, prefix: str = "profile_") -> dict[str, float]:
        out: dict[str, float] = {}
        for name, seconds in self.seconds.items():
            key = name.replace(".", "_")
            calls = self.calls[name]
            out[f"{prefix}{key}_seconds"] = float(seconds)
            out[f"{prefix}{key}_calls"] = float(calls)
            out[f"{prefix}{key}_ms_per_call"] = float(seconds * 1000.0 / max(calls, 1))
        return out
