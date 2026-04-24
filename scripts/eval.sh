#!/usr/bin/env bash
# Checkpoint evaluation pipeline:
#   Step 1: Export predictions (main venv, torch)
#   Step 2: Compute WOMD metrics (.venv-scripts, TF + waymo SDK)
#
# Usage:
#   bash scripts/eval.sh <config> <ckpt> [split] [output_npz]
#
# Examples:
#   bash scripts/eval.sh configs/prediction/mtr_baseline.yaml checkpoints/last.ckpt
#   bash scripts/eval.sh configs/prediction/mtr_baseline.yaml checkpoints/last.ckpt validation results/pred.npz

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:?Usage: eval.sh <config> <ckpt> [split] [output_npz]}"
CKPT="${2:?Usage: eval.sh <config> <ckpt> [split] [output_npz]}"
SPLIT="${3:-validation}"
OUTPUT="${4:-predictions.npz}"

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
