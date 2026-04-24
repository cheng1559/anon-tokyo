#!/usr/bin/env bash
# Build MTR CUDA extensions and copy the generated .so files into this project.
#
# Usage:
#   bash scripts/build_mtr_ops.sh
#   MTR_OPS_SOURCE_DIR=/path/to/mtr_ops_src bash scripts/build_mtr_ops.sh
#
# Notes:
# - This is a build-time helper only. The tracked source tree lives at
#   third_party/mtr_ops_src by default.
# - Runtime imports use files under src/anon_tokyo/prediction/mtr/ops and never
#   import from reference-code.
# - The generated .so files are ignored by git. Re-run this script after
#   changing Python, PyTorch, CUDA, or GPU architecture.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

MTR_OPS_SOURCE_DIR="${MTR_OPS_SOURCE_DIR:-${PROJECT_ROOT}/third_party/mtr_ops_src}"
if [[ ! -f "${MTR_OPS_SOURCE_DIR}/setup.py" ]]; then
  echo "MTR ops source not found: ${MTR_OPS_SOURCE_DIR}" >&2
  echo "Set MTR_OPS_SOURCE_DIR=/path/to/mtr_ops_src and rerun." >&2
  exit 1
fi

ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
MAX_JOBS="${MAX_JOBS:-4}"

echo "=== Build MTR CUDA ops ==="
echo "  project:         ${PROJECT_ROOT}"
echo "  ops source:      ${MTR_OPS_SOURCE_DIR}"
echo "  python:          ${PYTHON_BIN}"
echo "  TORCH_CUDA_ARCH_LIST: ${ARCH_LIST}"
echo "  MAX_JOBS:        ${MAX_JOBS}"
echo "=========================="

(
  cd "${MTR_OPS_SOURCE_DIR}"
  MAX_JOBS="${MAX_JOBS}" TORCH_CUDA_ARCH_LIST="${ARCH_LIST}" "${PYTHON_BIN}" setup.py build_ext --inplace
)

PY_TAG="$("${PYTHON_BIN}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"

ATTN_SRC="${MTR_OPS_SOURCE_DIR}/mtr/ops/attention/attention_cuda${PY_TAG}"
KNN_SRC="${MTR_OPS_SOURCE_DIR}/mtr/ops/knn/knn_cuda${PY_TAG}"
ATTN_DST="${PROJECT_ROOT}/src/anon_tokyo/prediction/mtr/ops/attention"
KNN_DST="${PROJECT_ROOT}/src/anon_tokyo/prediction/mtr/ops/knn"

if [[ ! -f "${ATTN_SRC}" || ! -f "${KNN_SRC}" ]]; then
  echo "Expected built extensions were not found:" >&2
  echo "  ${ATTN_SRC}" >&2
  echo "  ${KNN_SRC}" >&2
  exit 1
fi

mkdir -p "${ATTN_DST}" "${KNN_DST}"
cp "${ATTN_SRC}" "${ATTN_DST}/"
cp "${KNN_SRC}" "${KNN_DST}/"
rm -f "${MTR_OPS_SOURCE_DIR}/mtr/version.py"

"${PYTHON_BIN}" - <<'PY'
from anon_tokyo.prediction.mtr.attention import cuda_ops_available

if not cuda_ops_available():
    raise SystemExit("Built .so files were copied, but CUDA ops are still unavailable.")
print("CUDA ops available: True")
PY
