#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_DIR}/scripts/common/driver_python.sh"

JOB_NAME="${JOB_NAME:-}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/mccl_official_multi_node}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-1}"
OP_TIMEOUT_SECONDS="${OP_TIMEOUT_SECONDS:-600}"

MACA_PATH="${MACA_PATH:-/opt/maca}"
MPI_BIN="${MPI_BIN:-${MACA_PATH}/ompi/bin/mpirun}"
MCCL_TEST_BIN_DIR="${MCCL_TEST_BIN_DIR:-${MACA_PATH}/samples/mccl_tests/perf/mccl_perf}"
GPUS_PER_NODE="${GPUS_PER_NODE:-0}"
DTYPE="${DTYPE:-float}"
MIN_MESSAGE_SIZE="${MIN_MESSAGE_SIZE:-1K}"
MAX_MESSAGE_SIZE="${MAX_MESSAGE_SIZE:-2G}"
STEP_FACTOR="${STEP_FACTOR:-2}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-10}"
MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-eth0}"
MCCL_IB_HCA="${MCCL_IB_HCA:-xscale_0,xscale_1,xscale_2,xscale_3}"
MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-5}"
MCCL_IB_TC="${MCCL_IB_TC:-128}"
MCCL_ENABLE_VSWITCH="${MCCL_ENABLE_VSWITCH:-1}"
MCCL_PCIE_BUFFER_MODE="${MCCL_PCIE_BUFFER_MODE:-0}"
MCCL_CROSS_NIC="${MCCL_CROSS_NIC:-1}"
FORCE_ACTIVE_WAIT="${FORCE_ACTIVE_WAIT:-2}"
COLLECTIVE_OPS="${COLLECTIVE_OPS:-all_reduce,reduce_scatter,all_gather,broadcast,all_to_all,all_to_allv}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
METAX_KILL_ALL_PROCESS_BEFORE_OP="${METAX_KILL_ALL_PROCESS_BEFORE_OP:-0}"
ALLOW_KILL_ALL_PROCESS="${ALLOW_KILL_ALL_PROCESS:-0}"
KILL_ALL_PROCESS_WAIT_SECONDS="${KILL_ALL_PROCESS_WAIT_SECONDS:-5}"

if [[ -z "${JOB_NAME}" ]]; then
  echo "[mccl-multi-node] JOB_NAME is required" >&2
  exit 2
fi
if [[ "${METAX_KILL_ALL_PROCESS_BEFORE_OP}" == "1" && "${ALLOW_KILL_ALL_PROCESS}" != "1" ]]; then
  echo "[mccl-multi-node] METAX_KILL_ALL_PROCESS_BEFORE_OP=1 requires ALLOW_KILL_ALL_PROCESS=1" >&2
  exit 2
fi

resolve_driver_python
print_driver_python

echo "[mccl-multi-node] project      : ${PROJECT_DIR}"
echo "[mccl-multi-node] job          : ${JOB_NAME}"
echo "[mccl-multi-node] namespace    : ${NAMESPACE}"
echo "[mccl-multi-node] run id       : ${RUN_ID}"
echo "[mccl-multi-node] output       : ${RESULT_ROOT}/${RUN_ID}"
echo "[mccl-multi-node] range        : ${MIN_MESSAGE_SIZE}..${MAX_MESSAGE_SIZE}, factor=${STEP_FACTOR}"
echo "[mccl-multi-node] ops          : ${COLLECTIVE_OPS}"
echo "[mccl-multi-node] warmup/iters : ${WARMUP}/${ITERS}"
echo "[mccl-multi-node] op timeout   : ${OP_TIMEOUT_SECONDS}s"
echo "[mccl-multi-node] dry run      : ${DRY_RUN}"

exec "${DRIVER_PYTHON}" "${PROJECT_DIR}/tools/vcctl_mccl_multi_node_collective_sweep.py" \
  --job-name "${JOB_NAME}" \
  --namespace "${NAMESPACE}" \
  --vcctl-bin "${VCCTL_BIN}" \
  --container-name "${CONTAINER_NAME}" \
  --result-root "${RESULT_ROOT}" \
  --run-id "${RUN_ID}" \
  --maca-path "${MACA_PATH}" \
  --mpi-bin "${MPI_BIN}" \
  --test-bin-dir "${MCCL_TEST_BIN_DIR}" \
  --gpus-per-node "${GPUS_PER_NODE}" \
  --dtype "${DTYPE}" \
  --min-message-size "${MIN_MESSAGE_SIZE}" \
  --max-message-size "${MAX_MESSAGE_SIZE}" \
  --step-factor "${STEP_FACTOR}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --socket-ifname "${MCCL_SOCKET_IFNAME}" \
  --ib-hca "${MCCL_IB_HCA}" \
  --ib-gid-index "${MCCL_IB_GID_INDEX}" \
  --ib-tc "${MCCL_IB_TC}" \
  --enable-vswitch "${MCCL_ENABLE_VSWITCH}" \
  --pcie-buffer-mode "${MCCL_PCIE_BUFFER_MODE}" \
  --cross-nic "${MCCL_CROSS_NIC}" \
  --force-active-wait "${FORCE_ACTIVE_WAIT}" \
  --ops "${COLLECTIVE_OPS}" \
  --op-timeout-seconds "${OP_TIMEOUT_SECONDS}" \
  --continue-on-failure "${CONTINUE_ON_FAILURE}" \
  --metax-kill-all-process-before-op "${METAX_KILL_ALL_PROCESS_BEFORE_OP}" \
  --allow-kill-all-process "${ALLOW_KILL_ALL_PROCESS}" \
  --kill-all-process-wait-seconds "${KILL_ALL_PROCESS_WAIT_SECONDS}" \
  --dry-run "${DRY_RUN}"
