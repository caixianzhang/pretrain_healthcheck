#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JOB_NAME="${JOB_NAME:-}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/hccl_official_multi_node}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-1}"
EXEC_TIMEOUT_SECONDS="${EXEC_TIMEOUT_SECONDS:-3600}"

ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
MPI_BIN="${MPI_BIN:-/usr/local/mpich-3.2.1/bin/mpirun}"
MPI_LIB_DIR="${MPI_LIB_DIR:-/usr/local/mpich-3.2.1/lib}"
HCCL_TEST_BIN_DIR="${HCCL_TEST_BIN_DIR:-/usr/local/Ascend/ascend-toolkit/8.1.RC1/tools/hccl_test/bin}"
NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
DTYPE="${DTYPE:-bfp16}"
MIN_MESSAGE_SIZE="${MIN_MESSAGE_SIZE:-1K}"
MAX_MESSAGE_SIZE="${MAX_MESSAGE_SIZE:-8G}"
STEP_FACTOR="${STEP_FACTOR:-2}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-30}"
HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
COLLECTIVE_OPS="${COLLECTIVE_OPS:-all_reduce,reduce_scatter,all_gather,broadcast,all_to_all,all_to_allv}"

if [[ -z "${JOB_NAME}" ]]; then
  echo "[hccl-multi-node] JOB_NAME is required" >&2
  exit 2
fi

echo "[hccl-multi-node] project      : ${PROJECT_DIR}"
echo "[hccl-multi-node] job          : ${JOB_NAME}"
echo "[hccl-multi-node] namespace    : ${NAMESPACE}"
echo "[hccl-multi-node] run id       : ${RUN_ID}"
echo "[hccl-multi-node] output       : ${RESULT_ROOT}/${RUN_ID}"
echo "[hccl-multi-node] range        : ${MIN_MESSAGE_SIZE}..${MAX_MESSAGE_SIZE}, factor=${STEP_FACTOR}"
echo "[hccl-multi-node] ops          : ${COLLECTIVE_OPS}"
echo "[hccl-multi-node] warmup/iters : ${WARMUP}/${ITERS}"
echo "[hccl-multi-node] dry run      : ${DRY_RUN}"

exec python3 "${PROJECT_DIR}/tools/vcctl_hccl_multi_node_collective_sweep.py" \
  --job-name "${JOB_NAME}" \
  --namespace "${NAMESPACE}" \
  --vcctl-bin "${VCCTL_BIN}" \
  --container-name "${CONTAINER_NAME}" \
  --result-root "${RESULT_ROOT}" \
  --run-id "${RUN_ID}" \
  --ascend-env-script "${ASCEND_ENV_SCRIPT}" \
  --mpi-bin "${MPI_BIN}" \
  --mpi-lib-dir "${MPI_LIB_DIR}" \
  --test-bin-dir "${HCCL_TEST_BIN_DIR}" \
  --npus-per-node "${NPUS_PER_NODE}" \
  --dtype "${DTYPE}" \
  --min-message-size "${MIN_MESSAGE_SIZE}" \
  --max-message-size "${MAX_MESSAGE_SIZE}" \
  --step-factor "${STEP_FACTOR}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --socket-ifname "${HCCL_SOCKET_IFNAME}" \
  --ops "${COLLECTIVE_OPS}" \
  --exec-timeout-seconds "${EXEC_TIMEOUT_SECONDS}" \
  --dry-run "${DRY_RUN}"
