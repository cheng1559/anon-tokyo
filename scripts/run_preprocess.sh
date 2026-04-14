#!/usr/bin/env bash
# ── WOMD Preprocessing & Packing Runner ───────────────────────────────────────
#   bash scripts/run_preprocess.sh --raw_dir data/raw --out_dir data/processed --shard_dir data/shards
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR%/scripts}"
VENV_DIR="${PROJECT_DIR}/.venv-scripts"
PYTHON="${VENV_DIR}/bin/python"

if [ ! -x "${PYTHON}" ]; then
    echo "[ERROR] ${PYTHON} not found. Make sure .venv-scripts is in the project root." >&2
    exit 1
fi

# Parse args
RAW_DIR=""
OUT_DIR=""
SHARD_DIR=""
SPLITS="training validation testing"
NUM_CPUS="$(nproc)"
SKIP_EXISTING=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw_dir)  RAW_DIR="$2";  shift 2 ;;
        --out_dir)  OUT_DIR="$2";  shift 2 ;;
        --shard_dir) SHARD_DIR="$2"; shift 2 ;;
        --splits)   SPLITS="$2";   shift 2 ;;
        --num_cpus) NUM_CPUS="$2"; shift 2 ;;
        --skip_existing) SKIP_EXISTING="--skip_existing"; shift 1 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "${RAW_DIR}" ] || [ -z "${OUT_DIR}" ] || [ -z "${SHARD_DIR}" ]; then
    echo "Usage: bash scripts/run_preprocess.sh --raw_dir <path> --out_dir <path> --shard_dir <path> [--splits 'training validation'] [--num_cpus N] [--skip_existing]" >&2
    exit 1
fi

echo ""
echo "============================================"
echo "  WOMD Preprocessing & Packing"
echo "  RAW_DIR:   ${RAW_DIR}"
echo "  OUT_DIR:   ${OUT_DIR}"
echo "  SHARD_DIR: ${SHARD_DIR}"
echo "  SPLITS:    ${SPLITS}"
echo "  NUM_CPUS:  ${NUM_CPUS}"
echo "  PYTHON:    ${PYTHON}"
echo "============================================"
echo ""

for split in ${SPLITS}
do
    echo "[${split}] Processing..."
    TF_CPP_MIN_LOG_LEVEL=2 "${PYTHON}" "${SCRIPT_DIR}/preprocess_womd.py" \
        --raw_dir "${RAW_DIR}" \
        --out_dir "${OUT_DIR}" \
        --splits "${split}" \
        --num_cpus "${NUM_CPUS}" \
        ${SKIP_EXISTING}

    echo "[${split}] Verifying (--fix: delete bad files)..."
    if ! "${PYTHON}" "${SCRIPT_DIR}/verify_npz.py" \
        --data_dir "${OUT_DIR}" \
        --splits "${split}" \
        --num_cpus "${NUM_CPUS}" \
        --fix; then
        echo "[${split}] Re-processing deleted files..."
        TF_CPP_MIN_LOG_LEVEL=2 "${PYTHON}" "${SCRIPT_DIR}/preprocess_womd.py" \
            --raw_dir "${RAW_DIR}" \
            --out_dir "${OUT_DIR}" \
            --splits "${split}" \
            --num_cpus "${NUM_CPUS}" \
            --skip_existing
    fi

    count=$(find "${OUT_DIR}/${split}" -maxdepth 1 -name '*.npz' | wc -l)
    echo "[${split}] ${count} scenarios"

    # ── Pack shards ──────────────────────────────────────────────────────────
    if [ -n "${SKIP_EXISTING}" ] && [ -f "${SHARD_DIR}/${split}/index.json" ]; then
        echo "[${split}] Shards already exist, skipping pack."
    else
        echo "[${split}] Packing shards..."
        "${PYTHON}" "${SCRIPT_DIR}/pack_shards.py" \
            --src_dir "${OUT_DIR}" \
            --dst_dir "${SHARD_DIR}" \
            --splits "${split}" \
            --num_cpus "${NUM_CPUS}"
    fi
    echo ""
done

echo "[DONE] All splits complete. Output: ${OUT_DIR}"
