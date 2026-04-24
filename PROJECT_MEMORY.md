# Project Memory

Last updated: 2026-04-24

## Project Snapshot

- Project: `anon-tokyo`
- Location: `/high_perf_store3/l3_data/huchengcheng/anon-tokyo-v2`
- Python package: `anon_tokyo`
- Runtime stack: Python 3.12, PyTorch, Lightning, uv
- Purpose: WOMD trajectory prediction and multi-agent interaction modeling with RoPE/DRoPE.

## Current Status

- Open-loop prediction is the mature implemented path.
- `train_prediction.py` is the LightningCLI entry point.
- Main prediction models:
  - `src/anon_tokyo/prediction/mtr`: agent-centric MTR baseline.
  - `src/anon_tokyo/prediction/anon_tokyo`: scene-centric AnonTokyo model.
- Closed-loop simulation / PPO is still not implemented.
  - `train_sim.py` raises `NotImplementedError`.
  - `src/anon_tokyo/simulation` is still mostly placeholder files.

## MTR Baseline

- `configs/prediction/mtr_baseline.yaml` targets official MTR baseline behavior.
- Model class remains `anon_tokyo.prediction.mtr.model.MTRModel`.
- Internal module names are official-compatible:
  - `context_encoder`
  - `motion_decoder`
- `MTRModel.load_official_checkpoint(path, strict=True)` supports loading official `.pth` checkpoints through `checkpoint["model_state"]`.
- Current MTR parameter count:
  - total: about `65.781M`
  - trainable: about `64.139M`
- Important MTR config values:
  - `num_encoder_layers: 6`
  - `num_decoder_layers: 6`
  - `d_model: 256`
  - `d_decoder: 512`
  - `map_d_model: 256`
  - `num_heads: 8`
  - `num_modes: 6`
  - `num_intention_queries: 64`
  - `num_future_frames: 80`
  - `use_local_attn: true`
  - `num_attn_neighbors: 16`
  - `center_offset_of_map: [30.0, 0.0]`
  - `num_base_map_polylines: 256`
  - `num_waypoint_map_polylines: 128`
- Training hyperparameters are aligned to official MTR:
  - AdamW, `lr: 1e-4`, `weight_decay: 0.01`
  - Lambda step decay epochs `[22, 24, 26, 28]`
  - `lr_decay: 0.5`, `lr_clip: 1e-6`
  - gradient clip `1000.0`
  - loss weights: `reg: 1.0`, `score: 1.0`, `vel: 0.5`

## CUDA Ops

- Runtime must not import from `reference-code`.
- Project-owned Python wrappers live under:
  - `src/anon_tokyo/prediction/mtr/ops/attention/attention_utils_v2.py`
  - `src/anon_tokyo/prediction/mtr/ops/knn/knn_utils.py`
- Tracked CUDA/C++ source lives under:
  - `third_party/mtr_ops_src`
- Build helper:
  - `scripts/build_mtr_ops.sh`
- Build command:
  - `bash scripts/build_mtr_ops.sh`
- The script directly compiles `third_party/mtr_ops_src` and copies generated `.so` files into:
  - `src/anon_tokyo/prediction/mtr/ops/attention/`
  - `src/anon_tokyo/prediction/mtr/ops/knn/`
- Generated `.so` files are ignored by git through the existing `*.so` rule.
- `attention.py` auto-enables bundled CUDA ops when import succeeds and tensors are CUDA.
- Disable CUDA ops with:
  - `ANON_TOKYO_DISABLE_MTR_CUDA_OPS=1`
- Check availability with:
  - `uv run python -c "from anon_tokyo.prediction.mtr.attention import cuda_ops_available; print(cuda_ops_available())"`
- Important implementation details:
  - Official KNN expects 3D coords, so 2D positions are padded to `[x, y, 0]`.
  - Official KNN returns batch-local indices.
  - MTR encoder local attention passes `local_indices=True`.
  - Local attention masks out-of-range local indices before calling the CUDA kernel.

## AnonTokyo

- `AnonTokyoModel` is scene-centric at input time.
- It predicts only `tracks_to_predict` targets in the decoder to avoid decoding all 128 agents.
- Regression targets remain agent-local, not scene-frame.
- Decoder, loss style, NMS behavior, and training pipeline follow the MTR baseline.
- Current parameter count is intentionally aligned with MTR:
  - `configs/prediction/anon_tokyo.yaml`: about `65.776M`
  - `configs/prediction/rope_ablation.yaml`: about `65.776M`
  - `configs/prediction/drope_ablation.yaml`: about `65.776M`
- AnonTokyo config now uses:
  - `num_encoder_layers: 2`
  - `num_decoder_layers: 6`
  - `d_model: 256`
  - `d_decoder: 512`
  - `map_d_model: 256`
  - `sparse_k: 16`
  - `num_attn_neighbors: 16`
- Reason for `num_encoder_layers: 2`:
  - Each AnonTokyo encoder layer contains Map-Map, Agent-Agent, and Agent-Map attention/FFN blocks.
  - Six AnonTokyo encoder layers made the model about `75.253M`.
  - Two layers brings it to about `65.776M`, matching MTR's `65.781M`.
- `src/anon_tokyo/nn/attention.py` sine fallback is parameter-free now.

## Data And Artifacts

- `data/shards` exists for `training`, `validation`, and `testing`.
- `data/processed` exists with per-scenario `.npz` files.
- `assets/mtr/cluster_64_center_dict.pkl` is the intended local intention-points file.
  - It is ignored by git because `*.pkl` is ignored.
  - Current local copy was restored from `/high_perf_store3/l3_data/huchengcheng/anon-tokyo/assets/mtr/cluster_64_center_dict.pkl`.
- Configs point to:
  - `assets/mtr/cluster_64_center_dict.pkl`
- Runtime outputs and large artifacts remain ignored:
  - `outputs/`
  - `lightning_logs/`
  - `tb_logs/`
  - `checkpoints/`
  - `*.ckpt`

## Configs And Scripts

- Main prediction configs:
  - `configs/prediction/mtr_baseline.yaml`
  - `configs/prediction/anon_tokyo.yaml`
  - `configs/prediction/rope_ablation.yaml`
  - `configs/prediction/drope_ablation.yaml`
- Training wrapper:
  - `scripts/train.sh`
  - Defaults to `configs/prediction/mtr_baseline.yaml`.
  - Uses `torchrun`.
  - Auto-detects Volcengine ML Platform environment variables.
  - Passes `--ckpt_path last` for automatic resume.
- CUDA op build wrapper:
  - `scripts/build_mtr_ops.sh`

## Verification

- Recent parameter-count check:
  - MTR: `65.781M`
  - AnonTokyo: `65.776M`
  - RoPE ablation: `65.776M`
  - DRoPE/sine ablation: `65.776M`
- Recent targeted test command:
  - `uv run pytest -q tests/test_attention.py tests/test_rope.py tests/test_encoder.py tests/test_model.py`
- Recent targeted result:
  - `44 passed`
- Recent CUDA-op related test command:
  - `uv run pytest -q tests/test_mtr_attention_ops.py tests/test_encoder.py tests/test_model.py`
- Recent CUDA-op related result:
  - `24 passed`
- Earlier full suite after CUDA-op fixes:
  - `uv run pytest -q`
  - `96 passed`
- `CUDA_LAUNCH_BLOCKING=1` MTR `batch_size=10` fast-dev run passed after KNN 2D-to-3D fix.

## Notes For Future Work

- Do not treat `reference-code` as a runtime dependency.
- Keep `.so` files generated locally and ignored by git.
- If Python, PyTorch, CUDA, or GPU architecture changes, rerun `scripts/build_mtr_ops.sh`.
- Keep `assets/mtr/cluster_64_center_dict.pkl` local-only unless project policy changes.
- Be careful with large data and log directories when scanning the repository.
- Avoid expensive full `du -sh data` scans unless necessary.
