#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JOB_NAME="${JOB_NAME:-}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
POD_JSON_FILE="${POD_JSON_FILE:-}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/vcctl}"
BATCH_RUN_ID="${BATCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TARGET_SCALE="${TARGET_SCALE:-0}"
PHASES="${PHASES:-}"
GROUP_SEED="${GROUP_SEED:-20260706}"
GROUP_TIMEOUT_SECONDS="${GROUP_TIMEOUT_SECONDS:-180}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-10}"
PHASE_GROUP_CONCURRENCY="${PHASE_GROUP_CONCURRENCY:-0}"
DRY_RUN="${DRY_RUN:-1}"
PRE_CLEAN="${PRE_CLEAN:-1}"
DYNAMIC_COMPARE="${DYNAMIC_COMPARE:-1}"
KEEP_GROUP_OUTPUTS="${KEEP_GROUP_OUTPUTS:-0}"
POD_PROJECT_DIR="${POD_PROJECT_DIR:-}"
GROUP_OUTPUT_ROOT="${GROUP_OUTPUT_ROOT:-/tmp/pretrain_healthcheck_group_outputs/vcctl}"
FAILED_GROUP_OUTPUT_MODE="${FAILED_GROUP_OUTPUT_MODE:-local-link}"
DRYRUN_RUN_ID="${DRYRUN_RUN_ID:-${BATCH_RUN_ID}_dryrun_tmp}"
KEEP_DRYRUN_TMP="${KEEP_DRYRUN_TMP:-0}"

DEFAULT_INJECT_ENV='export MCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3 MCCL_IB_GID_INDEX=5 MCCL_IB_TC=128 MCCL_ENABLE_VSWITCH=1 MCCL_PCIE_BUFFER_MODE=0 MCCL_SOCKET_IFNAME=eth0'
INJECT_ENV="${INJECT_ENV:-${DEFAULT_INJECT_ENV}}"
INJECT_ANCHOR="${INJECT_ANCHOR:-DIST_BACKEND=}"

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<vcjob-name> [env ...] bash scripts/metax/run_vcctl_multi_node_batch_healthcheck_with_env_injection.sh

This wrapper is for platforms that do not inject communication runtime envs into
pods. It first generates a dry-run command template, archives the original and
injected commands into the formal batch result directory, then runs the normal
multi-node batch healthcheck with the injected MULTI_NODE_CMD.

Common env:
  JOB_NAME                   Existing vcjob name to inspect and test.
  RESULT_ROOT                Shared result root. Default: <project>/results/vcctl
  BATCH_RUN_ID               Batch id. Default: current timestamp
  DRYRUN_RUN_ID              Temporary dry-run id. Default: <BATCH_RUN_ID>_dryrun_tmp
  INJECT_ENV                 Shell snippet injected before DIST_BACKEND=.
                             Default: MetaX MCCL envs.
  INJECT_ANCHOR              Anchor in MULTI_NODE_CMD. Default: DIST_BACKEND=
  KEEP_DRYRUN_TMP            1 keeps temporary dry-run directory. Default: 0

All normal run_vcctl_multi_node_batch_healthcheck.sh envs are also supported,
including TARGET_SCALE, PHASES, GROUP_TIMEOUT_SECONDS, PHASE_GROUP_CONCURRENCY,
PRE_CLEAN, DYNAMIC_COMPARE, POD_PROJECT_DIR, GROUP_OUTPUT_ROOT,
FAILED_GROUP_OUTPUT_MODE, and POD_JSON_FILE.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${JOB_NAME}" ]]; then
  echo "[batch-env-inject] JOB_NAME is required" >&2
  usage >&2
  exit 2
fi

RUN_DIR="${RESULT_ROOT}/${BATCH_RUN_ID}"
DRYRUN_DIR="${RESULT_ROOT}/${DRYRUN_RUN_ID}"
DRYRUN_STAGE_DIR="${DRYRUN_DIR}/multi_node"
ORIGINAL_DIR="${RUN_DIR}/dryrun_original"
INJECTED_DIR="${RUN_DIR}/dryrun_injected"

mkdir -p "${RUN_DIR}" "${ORIGINAL_DIR}" "${INJECTED_DIR}"
rm -rf "${DRYRUN_DIR}"

echo "[batch-env-inject] project        : ${PROJECT_DIR}"
echo "[batch-env-inject] job            : ${JOB_NAME}"
echo "[batch-env-inject] namespace      : ${NAMESPACE}"
echo "[batch-env-inject] batch_run_id   : ${BATCH_RUN_ID}"
echo "[batch-env-inject] dryrun_run_id  : ${DRYRUN_RUN_ID}"
echo "[batch-env-inject] result_dir     : ${RUN_DIR}"
echo "[batch-env-inject] inject_anchor  : ${INJECT_ANCHOR}"

dryrun_env=(
  "JOB_NAME=${JOB_NAME}"
  "NAMESPACE=${NAMESPACE}"
  "VCCTL_BIN=${VCCTL_BIN}"
  "MODE=multi-node"
  "PROFILE=dynamic-suite"
  "DRY_RUN=1"
  "PRE_CLEAN=${PRE_CLEAN}"
  "DYNAMIC_COMPARE=${DYNAMIC_COMPARE}"
  "RESULT_ROOT=${RESULT_ROOT}"
  "RUN_ID=${DRYRUN_RUN_ID}"
)

if [[ -n "${POD_JSON_FILE}" ]]; then
  dryrun_env+=("POD_JSON_FILE=${POD_JSON_FILE}")
fi
if [[ -n "${POD_PROJECT_DIR}" ]]; then
  dryrun_env+=("PROJECT_REMOTE_DIR=${POD_PROJECT_DIR}")
fi

echo "[batch-env-inject] generating dry-run command template"
env "${dryrun_env[@]}" bash "${SCRIPT_DIR}/run_vcctl_healthcheck.sh"

COMMANDS_ENV="${DRYRUN_STAGE_DIR}/commands.env"
if [[ ! -f "${COMMANDS_ENV}" ]]; then
  echo "[batch-env-inject] missing dry-run commands file: ${COMMANDS_ENV}" >&2
  exit 1
fi

cp "${COMMANDS_ENV}" "${ORIGINAL_DIR}/commands.env"

original_cmd="$(grep '^MULTI_NODE_CMD=' "${COMMANDS_ENV}" | sed 's/^MULTI_NODE_CMD=//')"
if [[ -z "${original_cmd}" ]]; then
  echo "[batch-env-inject] MULTI_NODE_CMD not found in ${COMMANDS_ENV}" >&2
  exit 1
fi

template_cmd="$(printf '%s\n' "${original_cmd}" | sed -E 's/--master-port="?([0-9]+)"?/--master-port="__HC_MASTER_PORT__"/g')"

if [[ "${template_cmd}" != *"${INJECT_ANCHOR}"* ]]; then
  echo "[batch-env-inject] inject anchor not found in MULTI_NODE_CMD: ${INJECT_ANCHOR}" >&2
  exit 1
fi

injected_cmd="${template_cmd/${INJECT_ANCHOR}/${INJECT_ENV} \&\& ${INJECT_ANCHOR}}"

printf '%s\n' "${original_cmd}" > "${ORIGINAL_DIR}/MULTI_NODE_CMD.original.sh"
printf '%s\n' "${template_cmd}" > "${ORIGINAL_DIR}/MULTI_NODE_CMD.original.template.sh"
printf '%s\n' "${INJECT_ENV}" > "${INJECTED_DIR}/inject_env.sh"
printf '%s\n' "${injected_cmd}" > "${INJECTED_DIR}/MULTI_NODE_CMD.injected.sh"
chmod 600 "${ORIGINAL_DIR}/commands.env" "${ORIGINAL_DIR}/MULTI_NODE_CMD.original.sh" \
  "${ORIGINAL_DIR}/MULTI_NODE_CMD.original.template.sh" "${INJECTED_DIR}/inject_env.sh" \
  "${INJECTED_DIR}/MULTI_NODE_CMD.injected.sh" 2>/dev/null || true

echo "[batch-env-inject] archived original command: ${ORIGINAL_DIR}/MULTI_NODE_CMD.original.sh"
echo "[batch-env-inject] archived injected command: ${INJECTED_DIR}/MULTI_NODE_CMD.injected.sh"

if [[ "${KEEP_DRYRUN_TMP}" != "1" && "${KEEP_DRYRUN_TMP}" != "true" ]]; then
  rm -rf "${DRYRUN_DIR}"
fi

export MULTI_NODE_CMD="${injected_cmd}"
export JOB_NAME NAMESPACE VCCTL_BIN POD_JSON_FILE RESULT_ROOT BATCH_RUN_ID TARGET_SCALE PHASES
export GROUP_SEED GROUP_TIMEOUT_SECONDS PROGRESS_INTERVAL_SECONDS PHASE_GROUP_CONCURRENCY
export DRY_RUN PRE_CLEAN DYNAMIC_COMPARE KEEP_GROUP_OUTPUTS POD_PROJECT_DIR
export GROUP_OUTPUT_ROOT FAILED_GROUP_OUTPUT_MODE

echo "[batch-env-inject] starting formal batch healthcheck"
exec bash "${SCRIPT_DIR}/run_vcctl_multi_node_batch_healthcheck.sh" "$@"
