"""Verify integrity of processed .npz scenario files.

Usage:
    python scripts/verify_npz.py --data_dir data/processed --splits training validation --num_cpus 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

import numpy as np
import ray
from tqdm import tqdm

EXPECTED_KEYS = {
    "scenario_id",
    "timestamps",
    "current_time_index",
    "sdc_track_index",
    "object_id",
    "object_type",
    "trajs",
    "map_polylines",
    "traffic_lights",
    "tracks_to_predict",
    "predict_difficulty",
}

EXPECTED_SHAPES = {
    "trajs": (None, None, 10),  # (A, T, 10) — T=91 for train/val, T=11 for test
    "map_polylines": (None, 7),  # (P, 7)
    "traffic_lights": (None, None, 5),  # (T, S, 5)
}


@ray.remote
def check_batch(file_paths: list[str], fix: bool = False) -> list[str]:
    """Check a batch of .npz files. Returns list of error messages (prefixed with path if fix)."""
    errors = []
    for path in file_paths:
        p = Path(path)
        fname = p.name
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as e:
            errors.append(f"{fname}: failed to load: {e}")
            if fix:
                p.unlink(missing_ok=True)
            continue

        keys = set(data.files)
        missing = EXPECTED_KEYS - keys
        if missing:
            errors.append(f"{fname}: missing keys: {missing}")
            data.close()
            if fix:
                p.unlink(missing_ok=True)
            continue

        bad = False
        for key, expected in EXPECTED_SHAPES.items():
            shape = data[key].shape
            if len(shape) != len(expected):
                errors.append(f"{fname}: {key} ndim={len(shape)}, expected {len(expected)}")
                bad = True
                continue
            for i, (actual, exp) in enumerate(zip(shape, expected)):
                if exp is not None and actual != exp:
                    errors.append(f"{fname}: {key}.shape[{i}]={actual}, expected {exp}")
                    bad = True

        if data["trajs"].shape[0] == 0:
            errors.append(f"{fname}: zero agents")
            bad = True

        data.close()
        if fix and bad:
            p.unlink(missing_ok=True)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify processed .npz files")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["training", "validation"])
    parser.add_argument("--num_cpus", type=int, default=32)
    parser.add_argument("--fix", action="store_true", help="Delete bad files so they can be re-processed")
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True)

    has_errors = False
    for split in args.splits:
        split_dir = Path(args.data_dir) / split
        files = sorted(split_dir.glob("*.npz"))
        if not files:
            print(f"[{split}] No files found, skipping.")
            continue

        # also detect leftover .tmp files
        tmp_files = list(split_dir.glob("*.tmp")) + list(split_dir.glob("*.tmp.npz"))
        if tmp_files:
            print(f"[{split}] WARNING: {len(tmp_files)} leftover .tmp files (incomplete writes)")

        str_files = [str(f) for f in files]
        batch_size = max(1, len(str_files) // args.num_cpus)
        batches = [str_files[i : i + batch_size] for i in range(0, len(str_files), batch_size)]
        futures = [check_batch.remote(batch, fix=args.fix) for batch in batches]

        all_errors: list[str] = []
        remaining = set(futures)
        with tqdm(total=len(futures), desc=f"Verifying {split}", unit="batch") as pbar:
            while remaining:
                done, remaining = ray.wait(list(remaining), num_returns=1)
                for ref in done:
                    all_errors.extend(ray.get(ref))
                    pbar.update(1)

        if all_errors:
            has_errors = True
            action = "deleted" if args.fix else "found"
            print(f"[{split}] {len(all_errors)} errors {action} in {len(files)} files:")
            for err in all_errors[:20]:
                print(f"  {err}")
            if len(all_errors) > 20:
                print(f"  ... and {len(all_errors) - 20} more")
        else:
            print(f"[{split}] OK: {len(files)} files verified.")

    ray.shutdown()
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
