"""Tests for binary shard read/write consistency."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from anon_tokyo.data.shard_io import ShardIndex, read_item, write_shard


def _make_fake_npz(tmp_path: Path, name: str) -> Path:
    """Create a small fake .npz that mirrors WOMD schema."""
    out = tmp_path / f"{name}.npz"
    np.savez_compressed(
        out,
        scenario_id=np.array(name, dtype="U"),
        trajs=np.random.randn(4, 91, 10).astype(np.float32),
        object_id=np.arange(4, dtype=np.int64),
        object_type=np.array([1, 1, 2, 3], dtype=np.int8),
        map_polylines=np.random.randn(100, 7).astype(np.float32),
        traffic_lights=np.random.randn(91, 3, 5).astype(np.float32),
        timestamps=np.linspace(0, 9, 91, dtype=np.float64),
        current_time_index=np.int32(10),
        sdc_track_index=np.int32(0),
        tracks_to_predict=np.array([0, 1], dtype=np.int32),
        predict_difficulty=np.array([1, 2], dtype=np.int32),
    )
    return out


class TestWriteAndRead:
    def test_roundtrip_single(self, tmp_path: Path) -> None:
        npz = _make_fake_npz(tmp_path, "scene_a")
        shard_path = tmp_path / "shard.bin"
        entries = write_shard(shard_path, [npz])

        assert len(entries) == 1
        offset, size, sid = entries[0]
        assert sid == "scene_a"
        assert offset == 0
        assert size > 0

        data = read_item(shard_path, offset, size)
        orig = dict(np.load(npz, allow_pickle=True))
        for key in orig:
            np.testing.assert_array_equal(data[key], orig[key])

    def test_roundtrip_multiple(self, tmp_path: Path) -> None:
        names = [f"scene_{i:04d}" for i in range(10)]
        npz_paths = [_make_fake_npz(tmp_path, n) for n in names]
        shard_path = tmp_path / "shard.bin"
        entries = write_shard(shard_path, npz_paths)

        assert len(entries) == 10
        for i, (offset, size, sid) in enumerate(entries):
            assert sid == names[i]
            data = read_item(shard_path, offset, size)
            assert str(data["scenario_id"]) == names[i]
            assert data["trajs"].shape == (4, 91, 10)


class TestShardIndex:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        idx = ShardIndex(
            shards=["s0.bin", "s1.bin"],
            scenario_ids=["a", "b", "c"],
            items=[(0, 0, 100), (0, 100, 200), (1, 0, 150)],
        )
        path = tmp_path / "index.json"
        idx.save(path)

        loaded = ShardIndex.load(path)
        assert loaded.shards == idx.shards
        assert loaded.scenario_ids == idx.scenario_ids
        assert loaded.items == idx.items
        assert len(loaded) == 3

    def test_empty_index(self, tmp_path: Path) -> None:
        idx = ShardIndex()
        assert len(idx) == 0
        path = tmp_path / "index.json"
        idx.save(path)
        loaded = ShardIndex.load(path)
        assert len(loaded) == 0


class TestEndToEnd:
    """Simulate pack_shards workflow."""

    def test_multi_shard_pack(self, tmp_path: Path) -> None:
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        names = [f"sc_{i:04d}" for i in range(5)]
        npz_paths = [_make_fake_npz(npz_dir, n) for n in names]

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()

        index = ShardIndex()
        scenes_per_shard = 2
        shard_idx = 0

        for start in range(0, len(npz_paths), scenes_per_shard):
            chunk = npz_paths[start : start + scenes_per_shard]
            shard_name = f"shard_{shard_idx:06d}.bin"
            entries = write_shard(shard_dir / shard_name, chunk)
            index.shards.append(shard_name)
            for offset, size, sid in entries:
                index.items.append((shard_idx, offset, size))
                index.scenario_ids.append(sid)
            shard_idx += 1

        index.save(shard_dir / "index.json")

        # Verify: load index, read every item, check data
        loaded = ShardIndex.load(shard_dir / "index.json")
        assert len(loaded) == 5
        assert len(loaded.shards) == 3  # ceil(5/2)

        for i in range(len(loaded)):
            si, off, sz = loaded.items[i]
            data = read_item(shard_dir / loaded.shards[si], off, sz)
            assert str(data["scenario_id"]) == loaded.scenario_ids[i]
