# Project Memory

Last updated: 2026-04-24

## Project Snapshot

- Project: `anon-tokyo`
- Location: `/high_perf_store3/l3_data/huchengcheng/anon-tokyo-v2`
- Python package: `anon_tokyo`
- Runtime stack: Python 3.12, PyTorch, Lightning, uv
- Purpose: WOMD trajectory prediction and multi-agent interaction modeling with RoPE/DRoPE.

## Current Status

- Open-loop prediction is the main implemented path.
- `train_prediction.py` is the LightningCLI entry point.
- Prediction models are implemented under `src/anon_tokyo/prediction`:
  - `prediction/anon_tokyo`: scene-centric AnonTokyo model with RoPE/DRoPE.
  - `prediction/mtr`: agent-centric MTR baseline.
- Core shared modules are implemented under:
  - `src/anon_tokyo/data`: WOMD dataset, shard/npz loading, transforms, datamodule.
  - `src/anon_tokyo/nn`: attention, RoPE, layers, polyline encoder.
  - `src/anon_tokyo/utils`: geometry helpers.
- Closed-loop simulation / PPO is not implemented yet.
  - `train_sim.py` currently raises `NotImplementedError`.
  - `src/anon_tokyo/simulation` is mostly placeholder files.

## Data And Artifacts

- `data/shards` exists for `training`, `validation`, and `testing`.
  - Approximate shard counts from the last scan:
    - training: 952
    - validation: 87
    - testing: 88
- `data/processed` exists with per-scenario `.npz` files.
  - Approximate npz counts from the last scan:
    - training: 486996
    - validation: 44097
    - testing: 44920
- Runtime artifacts present:
  - `checkpoints`: about 4.4G at last scan.
  - `tb_logs`: about 583G at last scan.
  - top-level `predictions.pkl`: about 3.7G at last scan.

## Configs And Scripts

- Main prediction configs live in `configs/prediction`:
  - `anon_tokyo.yaml`
  - `mtr_baseline.yaml`
  - `rope_ablation.yaml`
  - `drope_ablation.yaml`
- Training wrapper:
  - `scripts/train.sh`
  - Defaults to `configs/prediction/mtr_baseline.yaml`.
  - Uses `torchrun` and auto-detects Volcengine ML Platform environment variables.
  - Passes `--ckpt_path last` for automatic resume.
- Evaluation wrapper:
  - `scripts/eval.sh`
  - Exports predictions, then computes WOMD metrics through `.venv-scripts`.

## Verification

- Last verified command:
  - `uv run pytest -q`
- Last result:
  - `94 passed in 17.17s`
- Git state at that time:
  - Branch: `master`
  - Working tree was clean before this memory file was created.

## Notes For Future Work

- Treat open-loop prediction as the mature part of the repository.
- Treat simulation / PPO as planned but not yet implemented.
- Be careful with large data and log directories when scanning the repository.
- Avoid expensive full `du -sh data` scans unless necessary; the data tree is large.
