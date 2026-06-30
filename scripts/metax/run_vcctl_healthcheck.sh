#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JOB_NAME="${JOB_NAME:-muxi-2node}"
NAMESPACE="${NAMESPACE:-default}"
MODE="${MODE:-all}"
DEVICE_TYPE="${DEVICE_TYPE:-metax}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
PROJECT_REMOTE_DIR="${PROJECT_REMOTE_DIR:-/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck}"
PROFILE="${PROFILE:-quick}"
PRE_CLEAN="${PRE_CLEAN:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
HEALTHCHECK_MASTER_PORT="${HEALTHCHECK_MASTER_PORT:-auto}"
DEVICE_VENDOR="${DEVICE_VENDOR:-metax}"
COMM_RUNTIME="${COMM_RUNTIME:-mccl}"
DTYPE="${DTYPE:-bf16}"
MESSAGE_SIZES="${MESSAGE_SIZES:-1M,16M,64M}"
MOE_PATTERNS="${MOE_PATTERNS:-uniform,skewed,hot_expert,random,empty_expert}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-5}"
SEED="${SEED:-20260623}"
FAULT_BACKEND="${FAULT_BACKEND:-}"
FAULT_SLEEP_RANK="${FAULT_SLEEP_RANK:-}"
FAULT_SLEEP_SECONDS="${FAULT_SLEEP_SECONDS:-30}"
FAULT_NAN_RANK="${FAULT_NAN_RANK:-}"
FAULT_CORRUPT_RANK="${FAULT_CORRUPT_RANK:-}"
PRE_CLEAN_CMD="${PRE_CLEAN_CMD:-ps -eo pid=,args= | awk '/[p]retrain_healthcheck.cli|[t]orchrun/ {print \$1}' | xargs -r kill -TERM || true}"
STATIC_CMD="${STATIC_CMD:-cd ${PROJECT_REMOTE_DIR} && OUT_DIR=\"\$HC_POD_RESULT_DIR\" bash scripts/metax/probe_pod_capabilities.sh}"
FAULT_ENV="FAULT_BACKEND=\"${FAULT_BACKEND}\" FAULT_SLEEP_RANK=\"${FAULT_SLEEP_RANK}\" FAULT_SLEEP_SECONDS=\"${FAULT_SLEEP_SECONDS}\" FAULT_NAN_RANK=\"${FAULT_NAN_RANK}\" FAULT_CORRUPT_RANK=\"${FAULT_CORRUPT_RANK}\""
SINGLE_NODE_CMD="${SINGLE_NODE_CMD:-cd ${PROJECT_REMOTE_DIR} && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} torchrun --standalone --nproc-per-node=\"${GPUS_PER_NODE}\" -m pretrain_healthcheck.cli run-single-node --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${MESSAGE_SIZES}\" --moe-patterns \"${MOE_PATTERNS}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --seed \"${SEED}\"; rc=\$?; python3 -m pretrain_healthcheck.cli analyze --input-dir \"\$HC_POD_RESULT_DIR\" --output \"\$HC_POD_RESULT_DIR/report.md\"; exit \$rc}"
if [[ -z "${MULTI_NODE_CMD:-}" ]]; then
  if [[ "${PROFILE}" == "smoke" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"1\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli ping-group --output-dir \"\$HC_POD_RESULT_DIR\" --test-round smoke --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\""
  else
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-group --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${MESSAGE_SIZES}\" --moe-patterns \"${MOE_PATTERNS}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --seed \"${SEED}\" --test-round current_vcjob --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; rc=\$?; if [ \"\$RANK\" = \"0\" ]; then python3 -m pretrain_healthcheck.cli analyze --input-dir \"\$HC_POD_RESULT_DIR\" --output \"\$HC_POD_RESULT_DIR/report.md\"; fi; exit \$rc"
  fi
fi
EXEC_TIMEOUT_SECONDS="${EXEC_TIMEOUT_SECONDS:-3600}"
VCCTL_TIMEOUT_SECONDS="${VCCTL_TIMEOUT_SECONDS:-120}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
RESULT_ROOT="${RESULT_ROOT:-/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-1}"
POD_JSON_FILE="${POD_JSON_FILE:-}"

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<vcjob-name> [env ...] bash scripts/metax/run_vcctl_healthcheck.sh

Required:
  JOB_NAME                 vcjob name used by "vcctl pod get --job". Default: muxi-2node

Common env:
  NAMESPACE                Kubernetes namespace. Default: default
  MODE                     static|single-node|multi-node|all. Default: all
  DEVICE_TYPE              Metadata only, for example gpu, npu, metax. Default: metax
  PROJECT_REMOTE_DIR       Project path inside target pods.
  PROFILE                  quick|smoke. smoke only checks torchrun connectivity. Default: quick
  PRE_CLEAN                1 runs cleanup before checks; 0 disables it. Default: 1
  PRE_CLEAN_CMD            Command used for cleanup. Default: pkill healthcheck torchrun/python.
  GPUS_PER_NODE            Local device count per pod. Default: 8
  DIST_BACKEND             PyTorch distributed backend name. Default: nccl
  HEALTHCHECK_MASTER_PORT  auto or explicit rendezvous port. Default: auto
  DEVICE_VENDOR            Result metadata. Default: metax
  COMM_RUNTIME             Result metadata. Default: mccl
  FAULT_*                  Optional fault injection envs for backend/sleep/nan/corrupt tests.
  STATIC_CMD               Command executed in each pod for static checks. Default: MetaX static probe
  SINGLE_NODE_CMD          Command executed in each pod for single-node checks. Default: MetaX 8-card torchrun
  MULTI_NODE_CMD           Command executed in every pod concurrently for multi-node checks. Default: MetaX current vcjob torchrun
  RESULT_ROOT              Shared result root. Default: /afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl
  RUN_ID                   Run id. Default: current timestamp
  EXEC_TIMEOUT_SECONDS     Per-pod exec timeout. Default: 3600
  MAX_PARALLEL             0 means all pods concurrently. Default: 0
  DRY_RUN                  1 prints generated vcctl exec commands without executing them. Default: 1
  POD_JSON_FILE            Optional fixture file instead of calling vcctl pod get.

Example:
  bash scripts/metax/run_vcctl_healthcheck.sh

  DRY_RUN=0 bash scripts/metax/run_vcctl_healthcheck.sh

  MODE=multi-node \
  DRY_RUN=1 \
  MULTI_NODE_CMD='cd /path/to/healthcheck && torchrun --nnodes=${WORLD_SIZE} --nproc-per-node=8 --node-rank=${RANK} --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} -m pretrain_healthcheck.cli run-group --output-dir ${HC_POD_RESULT_DIR}' \
  bash scripts/metax/run_vcctl_healthcheck.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${JOB_NAME}" && -z "${POD_JSON_FILE}" ]]; then
  echo "[vcctl-healthcheck] JOB_NAME is required unless POD_JSON_FILE is set" >&2
  usage >&2
  exit 2
fi

case "${MODE}" in
  static|single-node|multi-node|all)
    ;;
  *)
    echo "[vcctl-healthcheck] invalid MODE=${MODE}; expected static|single-node|multi-node|all" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "static" || "${MODE}" == "all" ]]; then
  if [[ -z "${STATIC_CMD}" ]]; then
    echo "[vcctl-healthcheck] STATIC_CMD is required for MODE=${MODE}" >&2
    exit 2
  fi
fi

if [[ "${MODE}" == "single-node" || "${MODE}" == "all" ]]; then
  if [[ -z "${SINGLE_NODE_CMD}" ]]; then
    echo "[vcctl-healthcheck] SINGLE_NODE_CMD is required for MODE=${MODE}" >&2
    exit 2
  fi
fi

if [[ "${MODE}" == "multi-node" || "${MODE}" == "all" ]]; then
  if [[ -z "${MULTI_NODE_CMD}" ]]; then
    echo "[vcctl-healthcheck] MULTI_NODE_CMD is required for MODE=${MODE}" >&2
    exit 2
  fi
fi

OUT_DIR="${RESULT_ROOT}/${RUN_ID}"
mkdir -p "${OUT_DIR}"

echo "[vcctl-healthcheck] project     : ${PROJECT_DIR}"
echo "[vcctl-healthcheck] job         : ${JOB_NAME:-<fixture>}"
echo "[vcctl-healthcheck] namespace   : ${NAMESPACE}"
echo "[vcctl-healthcheck] mode        : ${MODE}"
echo "[vcctl-healthcheck] device      : ${DEVICE_TYPE}"
echo "[vcctl-healthcheck] result root : ${RESULT_ROOT}"
echo "[vcctl-healthcheck] run id      : ${RUN_ID}"
echo "[vcctl-healthcheck] output      : ${OUT_DIR}"
echo "[vcctl-healthcheck] dry run     : ${DRY_RUN}"

args=(
  "${PROJECT_DIR}/tools/vcctl_healthcheck_driver.py"
  --job-name "${JOB_NAME:-fixture}"
  --namespace "${NAMESPACE}"
  --mode "${MODE}"
  --device-type "${DEVICE_TYPE}"
  --result-root "${RESULT_ROOT}"
  --run-id "${RUN_ID}"
  --vcctl-bin "${VCCTL_BIN}"
  --healthcheck-master-port "${HEALTHCHECK_MASTER_PORT}"
  --pre-clean-cmd "$([[ "${PRE_CLEAN}" == "1" || "${PRE_CLEAN}" == "true" ]] && printf '%s' "${PRE_CLEAN_CMD}" || true)"
  --static-cmd "${STATIC_CMD}"
  --single-node-cmd "${SINGLE_NODE_CMD}"
  --multi-node-cmd "${MULTI_NODE_CMD}"
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

if [[ -n "${POD_JSON_FILE}" ]]; then
  args+=(--pod-json-file "${POD_JSON_FILE}")
fi

python3 "${args[@]}"
