#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_DIR}/scripts/common/driver_python.sh"

JOB_NAME="${JOB_NAME:-}"
NAMESPACE="${NAMESPACE:-default}"
TARGET_NODES="${TARGET_NODES:-}"
EXCLUDED_NODES="${EXCLUDED_NODES:-}"
CONFIRM_NODE_LOSS_REPRO="${CONFIRM_NODE_LOSS_REPRO:-}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
REPRO_RUN_ID="${REPRO_RUN_ID:-}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/vcctl}"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-/tmp/pretrain_healthcheck_group_outputs/vcctl}"
MEGATRON_PATH="${MEGATRON_PATH:-/afs-a3-241ceshi-shared/geruijun/muxi-Megatron/Megatron-LM}"
POD_PROJECT_DIR="${POD_PROJECT_DIR:-${PROJECT_DIR}}"
POD_PYTHON="${POD_PYTHON:-/opt/conda/bin/python3}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
DRIVER_PYTHON="${DRIVER_PYTHON:-/opt/conda/bin/python3}"
MACA_HOME="${MACA_HOME:-/opt/maca}"
MACA_PATH="${MACA_PATH:-${MACA_HOME}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
IDLE_BASELINE_SECONDS="${IDLE_BASELINE_SECONDS:-120}"
FINAL_COOLDOWN_SECONDS="${FINAL_COOLDOWN_SECONDS:-120}"
POST_FAILURE_OBSERVE_SECONDS="${POST_FAILURE_OBSERVE_SECONDS:-120}"
LIVENESS_POLL_SECONDS="${LIVENESS_POLL_SECONDS:-5}"
EXEC_TIMEOUT_SECONDS="${EXEC_TIMEOUT_SECONDS:-300}"
CONTROLLER_TIMEOUT_SECONDS="${CONTROLLER_TIMEOUT_SECONDS:-420}"
HEALTHCHECK_MASTER_PORT="${HEALTHCHECK_MASTER_PORT:-29741}"

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<job> TARGET_NODES=96|128 [env ...] \
    bash scripts/metax/run_vcctl_node_loss_repro.sh

Required for a formal communication run:
  CONFIRM_NODE_LOSS_REPRO=YES

Important env:
  EXCLUDED_NODES            Comma, space, or newline separated exact hostnames.
  PREFLIGHT_ONLY            1 validates and writes the plan without communication.
  REPRO_RUN_ID              Optional stable run id; default is timestamped.
  MEGATRON_PATH             Megatron-LM path used to export rank groups.
  POD_PROJECT_DIR           Project path visible inside target Pods.
  IDLE_BASELINE_SECONDS     Pre-communication liveness baseline. Default: 120
  FINAL_COOLDOWN_SECONDS    Observation after a successful workload. Default: 120
  POST_FAILURE_OBSERVE_SECONDS
                            Observation after workload/node failure. Default: 120

Exit codes:
  0   completed without node loss, or preflight passed
  10  selected communication node lost health/readiness
  11  non-selected Job node lost health/readiness
  20  preflight failed or eligible nodes were insufficient
  30  communication failed without platform node loss
  40  controller error
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ -z "${JOB_NAME}" ]]; then
  echo "[node-loss-repro] JOB_NAME is required" >&2
  usage >&2
  exit 2
fi
if [[ "${TARGET_NODES}" != "96" && "${TARGET_NODES}" != "128" ]]; then
  echo "[node-loss-repro] TARGET_NODES must be 96 or 128" >&2
  exit 2
fi

unset DRIVER_PYTHON_RESOLVED DRIVER_PYTHON_VERSION
export MACA_HOME MACA_PATH
resolve_driver_python
print_driver_python

args=(
  --project-dir "${PROJECT_DIR}"
  --pod-project-dir "${POD_PROJECT_DIR}"
  --job-name "${JOB_NAME}"
  --namespace "${NAMESPACE}"
  --vcctl-bin "${VCCTL_BIN}"
  --target-nodes "${TARGET_NODES}"
  --gpus-per-node "${GPUS_PER_NODE}"
  --excluded-nodes "${EXCLUDED_NODES}"
  --result-root "${RESULT_ROOT}"
  --local-output-root "${LOCAL_OUTPUT_ROOT}"
  --run-id "${REPRO_RUN_ID}"
  --megatron-path "${MEGATRON_PATH}"
  --driver-python "${DRIVER_PYTHON}"
  --pod-python "${POD_PYTHON}"
  --idle-seconds "${IDLE_BASELINE_SECONDS}"
  --cooldown-seconds "${FINAL_COOLDOWN_SECONDS}"
  --post-failure-observe-seconds "${POST_FAILURE_OBSERVE_SECONDS}"
  --poll-seconds "${LIVENESS_POLL_SECONDS}"
  --exec-timeout-seconds "${EXEC_TIMEOUT_SECONDS}"
  --controller-timeout-seconds "${CONTROLLER_TIMEOUT_SECONDS}"
  --master-port "${HEALTHCHECK_MASTER_PORT}"
  --confirmation "${CONFIRM_NODE_LOSS_REPRO}"
)
if [[ "${PREFLIGHT_ONLY}" == "1" || "${PREFLIGHT_ONLY}" == "true" ]]; then
  args+=(--preflight-only)
fi

exec "${DRIVER_PYTHON}" "${PROJECT_DIR}/tools/vcctl_node_loss_repro.py" "${args[@]}"
