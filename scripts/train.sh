#!/usr/bin/env bash
# Multi-node / multi-GPU training via torchrun
#
# Auto-detects Volcengine ML Platform environment variables:
#   MLP_WORKER_NUM, MLP_ROLE_INDEX, MLP_WORKER_0_HOST,
#   MLP_WORKER_0_PORT, MLP_WORKER_GPU
#
# Usage:
#   bash scripts/train.sh                                  # 训练（自动续训 latest ckpt）
#   bash scripts/train.sh --name exp01                     # 指定实验名
#   bash scripts/train.sh --config other.yaml              # 自定义 config

set -euo pipefail

# --- Volcengine ML Platform env vars (auto-injected on cluster) ---
NPROC_PER_NODE="${MLP_WORKER_GPU:-8}"
NNODES="${MLP_WORKER_NUM:-1}"
NODE_RANK="${MLP_ROLE_INDEX:-0}"
MASTER_ADDR="${MLP_WORKER_0_HOST:-127.0.0.1}"
MASTER_PORT="${MLP_WORKER_0_PORT:-29500}"

# --- Parse named arguments ---
CONFIG="configs/prediction/mtr_baseline.yaml"
EXP_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  CONFIG="$2";    shift 2 ;;
        --name)    EXP_NAME="$2";  shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export UV_CACHE_DIR="${UV_CACHE_DIR:-/high_perf_store3/l3_data/huchengcheng/.cache/uv}"

echo "=== Training Config ==="
echo "  nproc_per_node: ${NPROC_PER_NODE}"
echo "  nnodes:         ${NNODES}"
echo "  node_rank:      ${NODE_RANK}"
echo "  master_addr:    ${MASTER_ADDR}"
echo "  master_port:    ${MASTER_PORT}"
echo "  config:         ${CONFIG}"
echo "  exp_name:       ${EXP_NAME:-<default>}"
echo "======================="

# Build extra CLI args
# ckpt_path=last: auto-resume from latest checkpoint if exists, otherwise train from scratch
EXTRA_ARGS=(--ckpt_path last)
[[ -n "${EXP_NAME}" ]] && EXTRA_ARGS+=(--trainer.logger.init_args.name "${EXP_NAME}")

exec uv run torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train_prediction.py fit \
    --config "${CONFIG}" \
    --trainer.num_nodes="${NNODES}" \
    "${EXTRA_ARGS[@]}"
