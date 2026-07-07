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

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<vcjob-name> [env ...] bash scripts/metax/run_vcctl_multi_node_batch_healthcheck.sh [--resume]

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

Examples:
  JOB_NAME=muxi-1024node DRY_RUN=0 TARGET_SCALE=128 bash scripts/metax/run_vcctl_multi_node_batch_healthcheck.sh
  JOB_NAME=muxi-1024node BATCH_RUN_ID=20260706_153012 bash scripts/metax/run_vcctl_multi_node_batch_healthcheck.sh --resume
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
  --keep-group-outputs "${KEEP_GROUP_OUTPUTS}"
