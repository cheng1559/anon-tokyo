"""Stage 2: Pack per-scene npz files into binary shards with index.

Usage (run in main .venv with numpy):
    python scripts/pack_shards.py \
        --src_dir data/processed \
        --dst_dir data/shards \
        --splits training validation \
        --scenes_per_shard 512 \
        --num_cpus 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ray
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anon_tokyo.data.shard_io import ShardIndex


@ray.remote
def pack_one_shard(
    shard_idx: int,
    shard_path: str,
    npz_paths: list[str],
) -> tuple[int, str, list[tuple[int, int, str]]]:
    """Write one shard and return (shard_idx, shard_name, entries)."""
    from anon_tokyo.data.shard_io import write_shard

    entries = write_shard(Path(shard_path), [Path(p) for p in npz_paths])
    return shard_idx, Path(shard_path).name, entries


def pack_split(
    src_dir: Path,
    dst_dir: Path,
    split: str,
    scenes_per_shard: int,
) -> None:
    npz_dir = src_dir / split
    out_dir = dst_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(npz_dir.glob("*.npz"))
    if not npz_paths:
        print(f"[WARN] No .npz files in {npz_dir}, skipping.")
        return

    num_shards = (len(npz_paths) + scenes_per_shard - 1) // scenes_per_shard
    print(f"[{split}] {len(npz_paths)} scenarios → {out_dir}  ({scenes_per_shard}/shard, {num_shards} shards)")

    # Submit all shards in parallel
    futures = []
    for shard_idx, start in enumerate(range(0, len(npz_paths), scenes_per_shard)):
        chunk = npz_paths[start : start + scenes_per_shard]
        shard_name = f"shard_{shard_idx:06d}.bin"
        shard_path = out_dir / shard_name
        futures.append(
            pack_one_shard.remote(  # type: ignore[call-arg]
                shard_idx,
                str(shard_path),
                [str(p) for p in chunk],
            )
        )

    # Collect results with progress
    results: list[tuple[int, str, list[tuple[int, int, str]]]] = []
    remaining = set(futures)
    with tqdm(total=len(futures), desc=split, unit="shard") as pbar:
        while remaining:
            done, remaining = ray.wait(list(remaining), num_returns=1)
            for ref in done:
                results.append(ray.get(ref))  # type: ignore[arg-type]
                pbar.update(1)

    # Build index (sort by shard_idx to keep deterministic order)
    results.sort(key=lambda x: x[0])
    index = ShardIndex()
    for shard_idx, shard_name, entries in results:
        index.shards.append(shard_name)
        for offset, size, scenario_id in entries:
            index.items.append((shard_idx, offset, size))
            index.scenario_ids.append(scenario_id)

    index.save(out_dir / "index.json")
    print(f"[{split}] Done: {len(index)} scenarios in {len(results)} shards")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack per-scene .npz into binary shards")
    parser.add_argument("--src_dir", type=str, required=True, help="Directory with processed npz files")
    parser.add_argument("--dst_dir", type=str, required=True, help="Output directory for shards")
    parser.add_argument("--splits", nargs="+", default=["training", "validation"])
    parser.add_argument("--scenes_per_shard", type=int, default=512, help="Scenarios per shard file")
    parser.add_argument("--num_cpus", type=int, default=32, help="Number of Ray workers")
    args = parser.parse_args()

    ray.init(
        num_cpus=args.num_cpus,
        runtime_env={"env_vars": {"PYTHONPATH": str(PROJECT_ROOT / "src")}},
        ignore_reinit_error=True,
    )
    for split in args.splits:
        pack_split(Path(args.src_dir), Path(args.dst_dir), split, args.scenes_per_shard)
    ray.shutdown()


if __name__ == "__main__":
    main()
