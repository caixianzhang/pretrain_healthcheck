#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JOB_NAME="${JOB_NAME:-}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
POD_JSON_FILE="${POD_JSON_FILE:-}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/vcctl}"
BATCH_RUN_ID="${BATCH_RUN_ID:-}"
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
BATCH_FAULT_TYPE="${BATCH_FAULT_TYPE:-}"
BATCH_FAULT_NODE="${BATCH_FAULT_NODE:-}"
BATCH_FAULT_POD="${BATCH_FAULT_POD:-}"
BATCH_FAULT_NODES="${BATCH_FAULT_NODES:-}"
BATCH_FAULT_PODS="${BATCH_FAULT_PODS:-}"
BATCH_FAULT_PHASE="${BATCH_FAULT_PHASE:-all}"
BATCH_FAULT_MAX_HITS="${BATCH_FAULT_MAX_HITS:-0}"
BATCH_FAULT_SLEEP_SECONDS="${BATCH_FAULT_SLEEP_SECONDS:-300}"
BATCH_FAULT_DELAY_MS="${BATCH_FAULT_DELAY_MS:-200}"
COMM_PATH_DEBUG="${COMM_PATH_DEBUG:-0}"

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<vcjob-name> [env ...] bash scripts/ascend/run_vcctl_multi_node_batch_healthcheck.sh [--resume]

This script only runs multi-node grouped dynamic checks. It does not run static
or single-node dynamic-suite checks.

Common env:
  JOB_NAME                   Existing vcjob name to inspect and test.
  NAMESPACE                  Kubernetes namespace. Default: default
  POD_JSON_FILE              Optional saved vcctl pod JSON for dry-run/local tests.
  RESULT_ROOT                Shared result root. Default: <project>/results/vcctl
  BATCH_RUN_ID               Batch id. Default: current timestamp
  TARGET_SCALE               Max phase scale, for example 128 or 256. Default: auto
  PHASES                     Comma-separated phases. Default: auto by node count
  GROUP_SEED                 Deterministic grouping seed. Default: 20260706
  GROUP_TIMEOUT_SECONDS      Per-group timeout passed to run_vcctl_healthcheck. Default: 180
  PROGRESS_INTERVAL_SECONDS  Progress print interval while one group is running. Default: 10
  PHASE_GROUP_CONCURRENCY    Max groups to run concurrently within one round. 0 means all. Default: 0
  DRY_RUN                    1 previews commands. Default: 1
  PRE_CLEAN                  1 cleans residual healthcheck processes before each group. Default: 1
  DYNAMIC_COMPARE            1 runs dynamic compact compare per group. Default: 1
  KEEP_GROUP_OUTPUTS         1 keeps normal per-group shared outputs. Default: 0
  POD_PROJECT_DIR            Project path inside target pods. Set explicitly for with_sync/copied project jobs.
  GROUP_OUTPUT_ROOT          Dev-machine local root for per-group outputs. Default: /tmp/pretrain_healthcheck_group_outputs/vcctl
  FAILED_GROUP_OUTPUT_MODE   local-link keeps failed group details under GROUP_OUTPUT_ROOT and links them from shared results; shared restores old behavior. Default: local-link
  BATCH_FAULT_TYPE           Optional fault injection: nan|corrupt|sleep|backend|join_timeout|comm_env_bad|eth_fallback|net_slow|rank_exit.
  BATCH_FAULT_NODE           Node name to inject into. Optional for backend/global tests.
  BATCH_FAULT_POD            Pod name to inject into. Optional for backend/global tests.
  BATCH_FAULT_NODES          Comma-separated node names. Combined with BATCH_FAULT_NODE.
  BATCH_FAULT_PODS           Comma-separated pod names. Combined with BATCH_FAULT_POD.
  BATCH_FAULT_PHASE          pairwise|ep8|scale64|scale128|scale256|final_all|all. Default: all
  BATCH_FAULT_MAX_HITS       0 means all matching groups; N limits injected groups. Default: 0
  BATCH_FAULT_SLEEP_SECONDS  Sleep/join-timeout seconds. Default: 300
  BATCH_FAULT_DELAY_MS       net_slow delay in milliseconds. Default: 200
  COMM_PATH_DEBUG            1 records compact communication-path env summaries. Default: 0

Examples:
  JOB_NAME=grj-megatron-128-235b-moe0708 DRY_RUN=0 TARGET_SCALE=8 bash scripts/ascend/run_vcctl_multi_node_batch_healthcheck.sh
  JOB_NAME=grj-megatron-128-235b-moe0708 BATCH_RUN_ID=20260706_153012 bash scripts/ascend/run_vcctl_multi_node_batch_healthcheck.sh --resume
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${JOB_NAME}" ]]; then
  echo "[batch-healthcheck] JOB_NAME is required" >&2
  usage >&2
  exit 2
fi

exec python3 "${PROJECT_DIR}/tools/vcctl_multi_node_batch.py" "$@" \
  --project-dir "${PROJECT_DIR}" \
  --healthcheck-script "${SCRIPT_DIR}/run_vcctl_healthcheck.sh" \
  --job-name "${JOB_NAME}" \
  --namespace "${NAMESPACE}" \
  --vcctl-bin "${VCCTL_BIN}" \
  --pod-json-file "${POD_JSON_FILE}" \
  --result-root "${RESULT_ROOT}" \
  --batch-run-id "${BATCH_RUN_ID}" \
  --target-scale "${TARGET_SCALE}" \
  --phases "${PHASES}" \
  --group-seed "${GROUP_SEED}" \
  --group-timeout-seconds "${GROUP_TIMEOUT_SECONDS}" \
  --progress-interval-seconds "${PROGRESS_INTERVAL_SECONDS}" \
  --phase-group-concurrency "${PHASE_GROUP_CONCURRENCY}" \
  --dry-run "${DRY_RUN}" \
  --pre-clean "${PRE_CLEAN}" \
  --dynamic-compare "${DYNAMIC_COMPARE}" \
  --keep-group-outputs "${KEEP_GROUP_OUTPUTS}" \
  --pod-project-dir "${POD_PROJECT_DIR}" \
  --group-output-root "${GROUP_OUTPUT_ROOT}" \
  --failed-group-output-mode "${FAILED_GROUP_OUTPUT_MODE}" \
  --batch-fault-type "${BATCH_FAULT_TYPE}" \
  --batch-fault-node "${BATCH_FAULT_NODE}" \
  --batch-fault-pod "${BATCH_FAULT_POD}" \
  --batch-fault-nodes "${BATCH_FAULT_NODES}" \
  --batch-fault-pods "${BATCH_FAULT_PODS}" \
  --batch-fault-phase "${BATCH_FAULT_PHASE}" \
  --batch-fault-max-hits "${BATCH_FAULT_MAX_HITS}" \
  --batch-fault-sleep-seconds "${BATCH_FAULT_SLEEP_SECONDS}" \
  --batch-fault-delay-ms "${BATCH_FAULT_DELAY_MS}" \
  --comm-path-debug "${COMM_PATH_DEBUG}"
