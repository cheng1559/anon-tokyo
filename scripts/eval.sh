#!/usr/bin/env bash
# Checkpoint evaluation pipeline:
#   Prediction:
#     Step 1: Export predictions (main venv, torch)
#     Step 2: Compute WOMD metrics (.venv-scripts, TF + waymo SDK)
#   Simulation:
#     Run deterministic closed-loop rollout metrics (main venv, torch)
#
# Usage:
#   bash scripts/eval.sh <config> <ckpt> [split] [output]
#
# Examples:
#   bash scripts/eval.sh configs/prediction/mtr_baseline.yaml checkpoints/last.ckpt
#   bash scripts/eval.sh configs/prediction/mtr_baseline.yaml checkpoints/last.ckpt validation results/pred.npz
#   bash scripts/eval.sh configs/simulation/anon_tokyo_ppo.yaml tb_logs/simulation_anon_tokyo/checkpoint_100.pt validation results/sim_eval.json

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:?Usage: eval.sh <config> <ckpt> [split] [output]}"
CKPT="${2:?Usage: eval.sh <config> <ckpt> [split] [output]}"
SPLIT="${3:-validation}"
OUTPUT="${4:-}"

if [[ "${CONFIG}" == configs/simulation/* ]]; then
    OUTPUT="${OUTPUT:-results/simulation_eval.json}"
    echo "=== Simulation evaluation (main venv) ==="
    uv run python scripts/eval_sim.py \
        --config "$CONFIG" \
        --ckpt "$CKPT" \
        --split "$SPLIT" \
        --output "$OUTPUT"
    exit 0
fi

OUTPUT="${OUTPUT:-predictions.npz}"

echo "=== Step 1: Export predictions (main venv) ==="
uv run python scripts/export_predictions.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --split "$SPLIT" \
    --output "$OUTPUT"

echo ""
echo "=== Step 2: WOMD evaluation (.venv-scripts) ==="
.venv-scripts/bin/python scripts/eval_womd.py \
    --predictions "$OUTPUT" \
    --eval_second 8 \
    --num_modes 6
