"""Binary shard serialization / deserialization.

Shard format
============
Each ``.bin`` shard is a flat concatenation of raw ``.npz`` byte blobs.
A companion ``index.json`` per split records per-item (shard, offset, size)
so that any scenario can be random-accessed with a single ``pread``.

index.json schema::

    {
        "shards": ["shard_000000.bin", ...],
        "scenario_ids": ["abc123", ...],
        "items": [[shard_idx, byte_offset, byte_size], ...]
    }
"""

from __future__ import annotations

import io
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

@dataclass
class ShardIndex:
    shards: list[str] = field(default_factory=list)
    scenario_ids: list[str] = field(default_factory=list)
    items: list[tuple[int, int, int]] = field(default_factory=list)  # (shard_idx, offset, size)

    def __len__(self) -> int:
        return len(self.items)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "shards": self.shards,
            "scenario_ids": self.scenario_ids,
            "items": self.items,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> ShardIndex:
        data = json.loads(path.read_text())
        return cls(
            shards=data["shards"],
            scenario_ids=data["scenario_ids"],
            items=[tuple(x) for x in data["items"]],
        )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_shard(
    out_path: Path,
    npz_paths: list[Path],
) -> list[tuple[int, int, str]]:
    """Pack *npz_paths* into a single shard binary file.

    Returns a list of ``(byte_offset, byte_size, scenario_id)`` per item.
    """
    entries: list[tuple[int, int, str]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".bin.tmp")
    with open(tmp, "wb") as fp:
        for npz_path in npz_paths:
            blob = npz_path.read_bytes()
            offset = fp.tell()
            fp.write(blob)
            scenario_id = npz_path.stem
            entries.append((offset, len(blob), scenario_id))
    tmp.replace(out_path)
    return entries


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_item(shard_path: Path | str, offset: int, size: int, *, fd: int | None = None) -> dict[str, np.ndarray]:
    """Read a single scenario from a shard file by byte offset and size.

    When *fd* is provided the caller-managed file descriptor is used with
    ``os.pread`` (single syscall, no seek, safe under concurrent access).
    """
    blob = _pread_blob(shard_path, offset, size, fd=fd)
    return dict(np.load(io.BytesIO(blob), allow_pickle=False))


def read_item_raw(shard_path: Path | str, offset: int, size: int, *, fd: int | None = None) -> bytes:
    """Read raw npz bytes from a shard file (for testing / re-packing)."""
    return _pread_blob(shard_path, offset, size, fd=fd)


def _pread_blob(shard_path: Path | str, offset: int, size: int, *, fd: int | None = None) -> bytes:
    """Read *size* bytes at *offset* using a single ``os.pread`` call."""
    own_fd = fd is None
    if own_fd:
        fd = os.open(str(shard_path), os.O_RDONLY)
    try:
        blob = os.pread(fd, size, offset)
    finally:
        if own_fd:
            os.close(fd)
    if len(blob) != size:
        raise ValueError(
            f"Short read from {shard_path}: expected {size} bytes at "
            f"offset {offset}, got {len(blob)}"
        )
    return blob
