"""Benchmark: direct .npz reads vs shard-based reads (with shard size sweep).

Usage:
    uv run python scripts/bench_read.py \
        --npz_dir data/processed/validation \
        --shard_dir data/shards/validation \
        --num_samples 500

    # Sweep different shard sizes
    uv run python scripts/bench_read.py \
        --npz_dir data/processed/validation \
        --shard_dir data/shards/validation \
        --num_samples 500 \
        --sweep 32,64,128,256,512,1024,2048
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from pathlib import Path

import numpy as np

from anon_tokyo.data.shard_io import ShardIndex, read_item, write_shard


def bench_direct(files: list[Path]) -> float:
    t0 = time.perf_counter()
    for f in files:
        _ = dict(np.load(f, allow_pickle=False))
    return time.perf_counter() - t0


def bench_shard(index: ShardIndex, shard_dir: Path, indices: list[int]) -> float:
    paths = {s: shard_dir / s for s in index.shards}
    t0 = time.perf_counter()
    for i in indices:
        si, off, sz = index.items[i]
        _ = read_item(paths[index.shards[si]], off, sz)
    return time.perf_counter() - t0


def build_temp_shards(npz_files: list[Path], scenes_per_shard: int, tmp_dir: Path) -> tuple[ShardIndex, Path]:
    out_dir = tmp_dir / f"sps_{scenes_per_shard}"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = ShardIndex()
    shard_idx = 0
    for start in range(0, len(npz_files), scenes_per_shard):
        chunk = npz_files[start : start + scenes_per_shard]
        shard_name = f"shard_{shard_idx:06d}.bin"
        entries = write_shard(out_dir / shard_name, chunk)
        index.shards.append(shard_name)
        for offset, size, sid in entries:
            index.items.append((shard_idx, offset, size))
            index.scenario_ids.append(sid)
        shard_idx += 1
    return index, out_dir


def run_single(
    npz_files: list[Path], sample_indices: list[int], index: ShardIndex, shard_dir: Path, label: str
) -> None:
    n = len(sample_indices)
    t_direct = bench_direct(npz_files)
    t_shard = bench_shard(index, shard_dir, sample_indices)
    ratio = t_direct / t_shard if t_shard > 0 else float("inf")
    print(f"  Direct .npz:  {t_direct:.3f}s  ({n / t_direct:.0f} samples/s, {t_direct / n * 1000:.2f} ms/sample)")
    print(f"  Shard read:   {t_shard:.3f}s  ({n / t_shard:.0f} samples/s, {t_shard / n * 1000:.2f} ms/sample)")
    print(f"  Speedup:      {ratio:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark npz vs shard read speed")
    parser.add_argument("--npz_dir", type=str, required=True)
    parser.add_argument("--shard_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sweep", type=str, default="", help="Comma-separated scenes_per_shard values to test, e.g. 32,128,512,2048"
    )
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    shard_dir = Path(args.shard_dir)
    index = ShardIndex.load(shard_dir / "index.json")

    rng = random.Random(args.seed)
    n = min(args.num_samples, len(index))
    sample_indices = rng.sample(range(len(index)), n)
    sample_ids = [index.scenario_ids[i] for i in sample_indices]
    npz_files = [npz_dir / f"{sid}.npz" for sid in sample_ids]

    missing = [f for f in npz_files if not f.exists()]
    if missing:
        print(f"WARNING: {len(missing)} npz files not found, reducing sample")
        valid = [(i, f) for i, f in zip(sample_indices, npz_files) if f.exists()]
        sample_indices = [x[0] for x in valid]
        npz_files = [x[1] for x in valid]
        n = len(npz_files)

    # Warmup
    _ = dict(np.load(npz_files[0], allow_pickle=False))
    _ = read_item(shard_dir / index.shards[index.items[sample_indices[0]][0]], *index.items[sample_indices[0]][1:])

    if not args.sweep:
        print(f"Benchmarking {n} samples (default shards)...\n")
        print("--- Sequential ---")
        run_single(npz_files, sample_indices, index, shard_dir, "default")
        rng2 = random.Random(args.seed + 1)
        shuffled_npz = npz_files.copy()
        shuffled_idx = sample_indices.copy()
        rng2.shuffle(shuffled_npz)
        rng2.shuffle(shuffled_idx)
        print("\n--- Random access ---")
        run_single(shuffled_npz, shuffled_idx, index, shard_dir, "random")
    else:
        sps_list = [int(x) for x in args.sweep.split(",")]
        print(f"Sweeping shard sizes: {sps_list}")
        print(f"Samples: {n}\n")

        # Direct npz baseline
        t_direct = bench_direct(npz_files)
        print(
            f"Direct .npz baseline: {t_direct:.3f}s ({n / t_direct:.0f} samples/s, {t_direct / n * 1000:.2f} ms/sample)\n"
        )

        # Only repack the sampled files (not all 44K+)
        all_npz = sorted(npz_files)

        with tempfile.TemporaryDirectory(dir=shard_dir.parent) as tmp:
            tmp_path = Path(tmp)
            print(
                f"{'SPS':>6}  {'Shards':>6}  {'~MB/shard':>10}  {'Time(s)':>8}  {'samples/s':>10}  {'ms/sample':>10}  {'Speedup':>8}"
            )
            print("-" * 78)

            for sps in sps_list:
                tmp_index, tmp_dir = build_temp_shards(all_npz, sps, tmp_path)

                # Map sample scenario_ids to new index
                id_to_idx = {sid: i for i, sid in enumerate(tmp_index.scenario_ids)}
                new_indices = [id_to_idx[sid] for sid in sample_ids if sid in id_to_idx]
                new_npz = [npz_dir / f"{tmp_index.scenario_ids[i]}.npz" for i in new_indices]

                # Warmup
                si0, off0, sz0 = tmp_index.items[new_indices[0]]
                _ = read_item(tmp_dir / tmp_index.shards[si0], off0, sz0)

                t = bench_shard(tmp_index, tmp_dir, new_indices)
                ns = len(new_indices)
                avg_shard_mb = sum((tmp_dir / s).stat().st_size for s in tmp_index.shards) / len(tmp_index.shards) / 1e6
                speedup = t_direct / t if t > 0 else float("inf")
                print(
                    f"{sps:>6}  {len(tmp_index.shards):>6}  {avg_shard_mb:>10.1f}  {t:>8.3f}  {ns / t:>10.0f}  {t / ns * 1000:>10.2f}  {speedup:>8.2f}x"
                )


if __name__ == "__main__":
    main()
