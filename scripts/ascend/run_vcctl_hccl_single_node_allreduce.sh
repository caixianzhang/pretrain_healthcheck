#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JOB_NAME="${JOB_NAME:-}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/hccl_official_single_node}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
EXEC_TIMEOUT_SECONDS="${EXEC_TIMEOUT_SECONDS:-300}"

ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
MPI_BIN="${MPI_BIN:-/usr/local/mpich-3.2.1/bin/mpirun}"
MPI_LIB_DIR="${MPI_LIB_DIR:-/usr/local/mpich-3.2.1/lib}"
HCCL_TEST_BIN="${HCCL_TEST_BIN:-/usr/local/Ascend/ascend-toolkit/8.1.RC1/tools/hccl_test/bin/all_reduce_test}"
NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
DTYPE="${DTYPE:-bfp16}"
MESSAGE_SIZE="${MESSAGE_SIZE:-1G}"
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-3}"
HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"

if [[ -z "${JOB_NAME}" ]]; then
  echo "[hccl-single-node] JOB_NAME is required" >&2
  exit 2
fi

echo "[hccl-single-node] project      : ${PROJECT_DIR}"
echo "[hccl-single-node] job          : ${JOB_NAME}"
echo "[hccl-single-node] namespace    : ${NAMESPACE}"
echo "[hccl-single-node] run id       : ${RUN_ID}"
echo "[hccl-single-node] output       : ${RESULT_ROOT}/${RUN_ID}"
echo "[hccl-single-node] workload     : ${NPUS_PER_NODE} ranks, all_reduce, ${MESSAGE_SIZE}, ${DTYPE}, warmup=${WARMUP}, iters=${ITERS}"
echo "[hccl-single-node] variant      : aligned_baseline"
echo "[hccl-single-node] dry run      : ${DRY_RUN}"

exec python3 "${PROJECT_DIR}/tools/vcctl_hccl_single_node_allreduce.py" \
  --job-name "${JOB_NAME}" \
  --namespace "${NAMESPACE}" \
  --vcctl-bin "${VCCTL_BIN}" \
  --container-name "${CONTAINER_NAME}" \
  --result-root "${RESULT_ROOT}" \
  --run-id "${RUN_ID}" \
  --ascend-env-script "${ASCEND_ENV_SCRIPT}" \
  --mpi-bin "${MPI_BIN}" \
  --mpi-lib-dir "${MPI_LIB_DIR}" \
  --test-bin "${HCCL_TEST_BIN}" \
  --npus-per-node "${NPUS_PER_NODE}" \
  --dtype "${DTYPE}" \
  --message-size "${MESSAGE_SIZE}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --socket-ifname "${HCCL_SOCKET_IFNAME}" \
  --max-parallel "${MAX_PARALLEL}" \
  --exec-timeout-seconds "${EXEC_TIMEOUT_SECONDS}" \
  --dry-run "${DRY_RUN}"
