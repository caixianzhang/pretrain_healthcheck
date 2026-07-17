#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_DIR}/scripts/common/driver_python.sh"

JOB_NAME="${JOB_NAME:-}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
GROUP_IDS="${GROUP_IDS:-scale64_r1_group_0000,scale64_r1_group_0001,final_all_group_0000_split_0,final_all_group_0000_split_1}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/vcctl}"
DIAG_RUN_ID="${DIAG_RUN_ID:-}"
PHASES="${PHASES:-pairwise,ep8,scale16,scale32,scale64}"
MESSAGE_SIZES="${MESSAGE_SIZES:-1M,128M,1G}"
GROUP_SEED="${GROUP_SEED:-20260706}"
GROUP_TIMEOUT_SECONDS="${GROUP_TIMEOUT_SECONDS:-180}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-10}"
BATCH_RUNTIME_WARN_SECONDS="${BATCH_RUNTIME_WARN_SECONDS:-900}"
DRY_RUN="${DRY_RUN:-1}"
PRE_CLEAN="${PRE_CLEAN:-1}"
DYNAMIC_COMPARE="${DYNAMIC_COMPARE:-1}"
COMM_PATH_DEBUG="${COMM_PATH_DEBUG:-1}"
POD_PROJECT_DIR="${POD_PROJECT_DIR:-}"
GROUP_OUTPUT_ROOT="${GROUP_OUTPUT_ROOT:-/tmp/pretrain_healthcheck_fixed_group_outputs/vcctl}"

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<vcjob> SOURCE_MANIFEST=<failed-group-manifest.json> DRY_RUN=0 \
    bash scripts/metax/run_vcctl_fixed_group_diagnosis.sh

Runs layered pairwise, EP8, scale16, scale32, and scale64 diagnosis inside fixed
64-node groups, followed by an exact manifest-order final-all reproduction.
Overlapping source groups run in separate rounds; disjoint groups run concurrently.

Important env:
  GROUP_IDS                  Comma-separated manifest group IDs.
  DIAG_RUN_ID                Result ID. Default: fixed64_diagnosis_<timestamp>
  PHASES                     Default: pairwise,ep8,scale16,scale32,scale64
  MESSAGE_SIZES              Localization matrix. Default: 1M,128M,1G
  GROUP_TIMEOUT_SECONDS      Timeout for each generated group. Default: 180
  COMM_PATH_DEBUG            Record communication-path diagnostics. Default: 1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ -z "${JOB_NAME}" || -z "${SOURCE_MANIFEST}" ]]; then
  echo "[fixed-group-diagnosis] JOB_NAME and SOURCE_MANIFEST are required" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  echo "[fixed-group-diagnosis] source manifest not found: ${SOURCE_MANIFEST}" >&2
  exit 2
fi

resolve_driver_python
print_driver_python
exec "${DRIVER_PYTHON}" "${PROJECT_DIR}/tools/vcctl_fixed_group_diagnosis.py" \
  --project-dir "${PROJECT_DIR}" \
  --job-name "${JOB_NAME}" \
  --namespace "${NAMESPACE}" \
  --vcctl-bin "${VCCTL_BIN}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --group-ids "${GROUP_IDS}" \
  --result-root "${RESULT_ROOT}" \
  --diag-run-id "${DIAG_RUN_ID}" \
  --phases "${PHASES}" \
  --message-sizes "${MESSAGE_SIZES}" \
  --group-seed "${GROUP_SEED}" \
  --group-timeout-seconds "${GROUP_TIMEOUT_SECONDS}" \
  --progress-interval-seconds "${PROGRESS_INTERVAL_SECONDS}" \
  --runtime-warn-seconds "${BATCH_RUNTIME_WARN_SECONDS}" \
  --dry-run "${DRY_RUN}" \
  --pre-clean "${PRE_CLEAN}" \
  --dynamic-compare "${DYNAMIC_COMPARE}" \
  --comm-path-debug "${COMM_PATH_DEBUG}" \
  --pod-project-dir "${POD_PROJECT_DIR}" \
  --group-output-root "${GROUP_OUTPUT_ROOT}"
