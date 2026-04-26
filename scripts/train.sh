#!/usr/bin/env bash
# Multi-node / multi-GPU training via torchrun
#
# Auto-detects Volcengine ML Platform environment variables:
#   MLP_WORKER_NUM, MLP_ROLE_INDEX, MLP_WORKER_0_HOST,
#   MLP_WORKER_0_PORT, MLP_WORKER_GPU
#
# Usage:
#   bash scripts/train.sh                                  # 训练（自动续训 latest ckpt）
#   bash scripts/train.sh --version baseline_v1            # 指定 version 名（续训同一目录）
#   bash scripts/train.sh --config other.yaml              # 自定义 config
#   bash scripts/train.sh --task simulation                # simulation PPO（默认 agent-centric）
#   bash scripts/train.sh --task simulation --version exp1 # simulation logs: tb_logs/<name>/exp1
#   bash scripts/train.sh --config configs/simulation/anon_tokyo_ppo.yaml
#   bash scripts/train.sh --task simulation -- --smoke_env # 透传 train_sim.py 参数

set -euo pipefail

detect_nproc_per_node() {
    if [[ -n "${MLP_WORKER_GPU:-}" ]]; then
        echo "${MLP_WORKER_GPU}"
        return
    fi
    if [[ -n "${NPROC_PER_NODE:-}" ]]; then
        echo "${NPROC_PER_NODE}"
        return
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" && "${CUDA_VISIBLE_DEVICES}" != "void" ]]; then
        local visible_count
        visible_count="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
        if [[ "${visible_count}" -gt 0 ]]; then
            echo "${visible_count}"
            return
        fi
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_count
        gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "${gpu_count}" -gt 0 ]]; then
            echo "${gpu_count}"
            return
        fi
    fi
    echo 1
}

# --- Volcengine ML Platform env vars (auto-injected on cluster) ---
NPROC_PER_NODE="$(detect_nproc_per_node)"
NNODES="${MLP_WORKER_NUM:-1}"
NODE_RANK="${MLP_ROLE_INDEX:-0}"
MASTER_ADDR="${MLP_WORKER_0_HOST:-127.0.0.1}"
MASTER_PORT="${MLP_WORKER_0_PORT:-29500}"

# --- Parse named arguments ---
TASK="prediction"
CONFIG=""
VERSION=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)     TASK="$2";     shift 2 ;;
        --config)   CONFIG="$2";   shift 2 ;;
        --version)  VERSION="$2";  shift 2 ;;
        --)         shift; PASSTHROUGH_ARGS+=("$@"); break ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "${CONFIG}" ]]; then
    case "${TASK}" in
        prediction) CONFIG="configs/prediction/mtr_baseline.yaml" ;;
        simulation) CONFIG="configs/simulation/agent_centric_ppo.yaml" ;;
        *)          echo "Unknown task: ${TASK}"; exit 1 ;;
    esac
elif [[ "${CONFIG}" == configs/simulation/* ]]; then
    TASK="simulation"
elif [[ "${CONFIG}" == configs/prediction/* ]]; then
    TASK="prediction"
fi

case "${TASK}" in
    prediction|simulation) ;;
    *) echo "Unknown task: ${TASK}"; exit 1 ;;
esac

export UV_CACHE_DIR="${UV_CACHE_DIR:-/high_perf_store3/l3_data/huchengcheng/.cache/uv}"

echo "=== Training Config ==="
echo "  task:           ${TASK}"
echo "  nproc_per_node: ${NPROC_PER_NODE}"
echo "  nnodes:         ${NNODES}"
echo "  node_rank:      ${NODE_RANK}"
echo "  master_addr:    ${MASTER_ADDR}"
echo "  master_port:    ${MASTER_PORT}"
echo "  config:         ${CONFIG}"
echo "  version:        ${VERSION:-<auto>}"
echo "  extra_args:     ${PASSTHROUGH_ARGS[*]:-<none>}"
echo "======================="

TORCHRUN_ARGS=(
    --nproc_per_node="${NPROC_PER_NODE}"
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
)

if [[ "${TASK}" == "simulation" ]]; then
    EXTRA_ARGS=()
    [[ -n "${VERSION}" ]] && EXTRA_ARGS+=(--version "${VERSION}")
    exec uv run torchrun \
        "${TORCHRUN_ARGS[@]}" \
        train_sim.py \
        --config "${CONFIG}" \
        "${EXTRA_ARGS[@]}" \
        "${PASSTHROUGH_ARGS[@]}"
else
    # ckpt_path=last: auto-resume from latest checkpoint if exists, otherwise train from scratch
    EXTRA_ARGS=(--ckpt_path last)
    [[ -n "${VERSION}" ]] && EXTRA_ARGS+=(--trainer.logger.init_args.version "${VERSION}")

    exec uv run torchrun \
        "${TORCHRUN_ARGS[@]}" \
        train_prediction.py fit \
        --config "${CONFIG}" \
        --trainer.num_nodes="${NNODES}" \
        "${EXTRA_ARGS[@]}" \
        "${PASSTHROUGH_ARGS[@]}"
fi
