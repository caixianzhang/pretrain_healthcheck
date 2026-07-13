#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JOB_NAME="${JOB_NAME:-grj-megatron-muxi-0630-moe-30ba3b}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-1}"

TRAIN_DIR="${TRAIN_DIR:-}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_qwen3_30B_A3B.sh}"
TRAIN_RESULT_ROOT="${TRAIN_RESULT_ROOT:-${PROJECT_DIR}/results/training_repro}"
POD_TRAIN_RESULT_ROOT="${POD_TRAIN_RESULT_ROOT:-${PROJECT_DIR}/results/training_repro}"

PRE_CLEAN="${PRE_CLEAN:-1}"
PRE_CLEAN_CMD="${PRE_CLEAN_CMD:-ps -eo pid=,args= | awk '/[p]retrain_gpt.py|[t]orchrun|[t]rain_qwen3_30B_A3B.sh/ {print \$1}' | xargs -r kill -TERM || true}"
EXEC_TIMEOUT_SECONDS="${EXEC_TIMEOUT_SECONDS:-0}"
VCCTL_TIMEOUT_SECONDS="${VCCTL_TIMEOUT_SECONDS:-120}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<vcjob-name> DRY_RUN=0 bash scripts/ascend/run_vcctl_training_repro.sh

This wrapper starts the target Megatron training script in every pod of a vcctl
job concurrently. It is intended for reproducing training-time MCCL/MCR errors
while keeping per-pod logs and a vcctl-style summary.

Common env:
  JOB_NAME                vcjob name. Default: grj-megatron-muxi-0630-moe-30ba3b
  NAMESPACE               Kubernetes namespace. Default: default
  TRAIN_DIR               Training workdir inside pods. Required.
  TRAIN_SCRIPT            Training script name. Default: train_qwen3_30B_A3B.sh
  TRAIN_RESULT_ROOT       Dev-machine result root. Default: pretrain_healthcheck/results/training_repro
  POD_TRAIN_RESULT_ROOT   Pod-visible result root. Default: PROJECT_DIR/results/training_repro
  RUN_ID                  Run id. Default: current timestamp
  PRE_CLEAN               1 kills stale training processes before launch. Default: 1
  EXEC_TIMEOUT_SECONDS    Per-pod vcctl exec timeout. 0 means no timeout. Default: 0
  DRY_RUN                 1 prints generated commands without running. Default: 1

Example:
  JOB_NAME=grj-megatron-muxi-0630-moe-30ba3b \
  DRY_RUN=0 \
  bash scripts/ascend/run_vcctl_training_repro.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${TRAIN_DIR}" ]]; then
  echo "[training-repro] TRAIN_DIR is required; set it to the training workdir inside pods" >&2
  usage >&2
  exit 2
fi

if [[ -z "${JOB_NAME}" ]]; then
  echo "[training-repro] JOB_NAME is required" >&2
  usage >&2
  exit 2
fi

mkdir -p "${TRAIN_RESULT_ROOT}/${RUN_ID}"

TRAIN_CMD="mkdir -p \"\$HC_POD_RESULT_DIR\" && export PATH=/opt/conda/bin:\$PATH && cd ${TRAIN_DIR} && echo \"[training-repro] pod=\$HC_POD_NAME node=\$HC_NODE_NAME rank=\$RANK world=\$WORLD_SIZE start=\$(date '+%F %T')\" | tee \"\$HC_POD_RESULT_DIR/launch.log\" && bash ${TRAIN_SCRIPT}; rc=\$?; echo \"[training-repro] pod=\$HC_POD_NAME rc=\$rc end=\$(date '+%F %T')\" | tee -a \"\$HC_POD_RESULT_DIR/launch.log\"; exit \$rc"

echo "[training-repro] project       : ${PROJECT_DIR}"
echo "[training-repro] job           : ${JOB_NAME}"
echo "[training-repro] namespace     : ${NAMESPACE}"
echo "[training-repro] train dir     : ${TRAIN_DIR}"
echo "[training-repro] train script  : ${TRAIN_SCRIPT}"
echo "[training-repro] result root   : ${TRAIN_RESULT_ROOT}"
echo "[training-repro] pod result    : ${POD_TRAIN_RESULT_ROOT}"
echo "[training-repro] run id        : ${RUN_ID}"
echo "[training-repro] dry run       : ${DRY_RUN}"

args=(
  "${PROJECT_DIR}/tools/vcctl_healthcheck_driver.py"
  --job-name "${JOB_NAME}"
  --namespace "${NAMESPACE}"
  --mode multi-node
  --device-type ascend-training
  --result-root "${TRAIN_RESULT_ROOT}"
  --pod-result-root "${POD_TRAIN_RESULT_ROOT}"
  --run-id "${RUN_ID}"
  --vcctl-bin "${VCCTL_BIN}"
  --healthcheck-master-port ""
  --pre-clean-cmd "$([[ "${PRE_CLEAN}" == "1" || "${PRE_CLEAN}" == "true" ]] && printf '%s' "${PRE_CLEAN_CMD}" || true)"
  --multi-node-cmd "${TRAIN_CMD}"
  --exec-timeout-seconds "${EXEC_TIMEOUT_SECONDS}"
  --vcctl-timeout-seconds "${VCCTL_TIMEOUT_SECONDS}"
  --max-parallel "${MAX_PARALLEL}"
)

if [[ -n "${CONTAINER_NAME}" ]]; then
  args+=(--container-name "${CONTAINER_NAME}")
fi

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  args+=(--dry-run)
fi

python3 "${args[@]}"
