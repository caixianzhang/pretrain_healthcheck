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
BANDWIDTH_MESSAGE_SIZES="${BANDWIDTH_MESSAGE_SIZES:-1G,4G,8G,16G}"
BANDWIDTH_WARMUP="${BANDWIDTH_WARMUP:-5}"
BANDWIDTH_ITERS="${BANDWIDTH_ITERS:-100}"
BANDWIDTH_MIN_BUSBW="${BANDWIDTH_MIN_BUSBW:-270}"
BANDWIDTH_AVG_BUSBW="${BANDWIDTH_AVG_BUSBW:-290}"
COLLECTIVE_BANDWIDTH_OPS="${COLLECTIVE_BANDWIDTH_OPS:-all_reduce,reduce_scatter,all_gather,all_to_all,all_to_allv}"
COLLECTIVE_BANDWIDTH_MESSAGE_SIZES="${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES:-1G}"
COLLECTIVE_BANDWIDTH_WARMUP="${COLLECTIVE_BANDWIDTH_WARMUP:-5}"
COLLECTIVE_BANDWIDTH_ITERS="${COLLECTIVE_BANDWIDTH_ITERS:-30}"
COLLECTIVE_BANDWIDTH_MIN_BUSBW="${COLLECTIVE_BANDWIDTH_MIN_BUSBW:-0}"
COLLECTIVE_BANDWIDTH_AVG_BUSBW="${COLLECTIVE_BANDWIDTH_AVG_BUSBW:-0}"
COLLECTIVE_BANDWIDTH_EP_SIZE="${COLLECTIVE_BANDWIDTH_EP_SIZE:-8}"
COLLECTIVE_BANDWIDTH_MOE_PATTERNS="${COLLECTIVE_BANDWIDTH_MOE_PATTERNS:-${MOE_PATTERNS}}"
SEED="${SEED:-20260623}"
FAULT_BACKEND="${FAULT_BACKEND:-}"
FAULT_SLEEP_RANK="${FAULT_SLEEP_RANK:-}"
FAULT_SLEEP_SECONDS="${FAULT_SLEEP_SECONDS:-30}"
FAULT_NAN_RANK="${FAULT_NAN_RANK:-}"
FAULT_CORRUPT_RANK="${FAULT_CORRUPT_RANK:-}"
PRE_CLEAN_CMD="${PRE_CLEAN_CMD:-ps -eo pid=,args= | awk '/[p]retrain_healthcheck.cli|[t]orchrun/ {print \$1}' | xargs -r kill -TERM || true}"
STATIC_CMD="${STATIC_CMD:-cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && OUT_DIR=\"\$HC_POD_RESULT_DIR\" bash scripts/metax/probe_pod_capabilities.sh}"
FAULT_ENV="FAULT_BACKEND=\"${FAULT_BACKEND}\" FAULT_SLEEP_RANK=\"${FAULT_SLEEP_RANK}\" FAULT_SLEEP_SECONDS=\"${FAULT_SLEEP_SECONDS}\" FAULT_NAN_RANK=\"${FAULT_NAN_RANK}\" FAULT_CORRUPT_RANK=\"${FAULT_CORRUPT_RANK}\""
SINGLE_NODE_CMD="${SINGLE_NODE_CMD:-cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} torchrun --standalone --nproc-per-node=\"${GPUS_PER_NODE}\" -m pretrain_healthcheck.cli run-single-node --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${MESSAGE_SIZES}\" --moe-patterns \"${MOE_PATTERNS}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --seed \"${SEED}\"; rc=\$?; python3 -m pretrain_healthcheck.cli analyze --input-dir \"\$HC_POD_RESULT_DIR\" --output \"\$HC_POD_RESULT_DIR/report.md\"; exit \$rc}"
if [[ -z "${MULTI_NODE_CMD:-}" ]]; then
  if [[ "${PROFILE}" == "smoke" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"1\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli ping-group --output-dir \"\$HC_POD_RESULT_DIR\" --test-round smoke --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\""
  elif [[ "${PROFILE}" == "bandwidth" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-bandwidth --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${BANDWIDTH_MESSAGE_SIZES}\" --warmup \"${BANDWIDTH_WARMUP}\" --iters \"${BANDWIDTH_ITERS}\" --seed \"${SEED}\" --min-busbw \"${BANDWIDTH_MIN_BUSBW}\" --avg-busbw \"${BANDWIDTH_AVG_BUSBW}\" --test-round bandwidth --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; rc=\$?; exit \$rc"
  elif [[ "${PROFILE}" == "collective-bandwidth" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-collective-bandwidth --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES}\" --ops \"${COLLECTIVE_BANDWIDTH_OPS}\" --moe-patterns \"${COLLECTIVE_BANDWIDTH_MOE_PATTERNS}\" --ep-size \"${COLLECTIVE_BANDWIDTH_EP_SIZE}\" --warmup \"${COLLECTIVE_BANDWIDTH_WARMUP}\" --iters \"${COLLECTIVE_BANDWIDTH_ITERS}\" --seed \"${SEED}\" --min-busbw \"${COLLECTIVE_BANDWIDTH_MIN_BUSBW}\" --avg-busbw \"${COLLECTIVE_BANDWIDTH_AVG_BUSBW}\" --test-round collective_bandwidth --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; rc=\$?; exit \$rc"
  else
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-group --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${MESSAGE_SIZES}\" --moe-patterns \"${MOE_PATTERNS}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --seed \"${SEED}\" --test-round current_vcjob --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; rc=\$?; if [ \"\$RANK\" = \"0\" ]; then python3 -m pretrain_healthcheck.cli analyze --input-dir \"\$HC_POD_RESULT_DIR\" --output \"\$HC_POD_RESULT_DIR/report.md\"; fi; exit \$rc"
  fi
fi
EXEC_TIMEOUT_SECONDS="${EXEC_TIMEOUT_SECONDS:-3600}"
HANG_TIMEOUT_SECONDS="${HANG_TIMEOUT_SECONDS:-0}"
HANG_KILL_GRACE_SECONDS="${HANG_KILL_GRACE_SECONDS:-10}"
HANG_CLEANUP_CMD="${HANG_CLEANUP_CMD:-${PRE_CLEAN_CMD}}"
VCCTL_TIMEOUT_SECONDS="${VCCTL_TIMEOUT_SECONDS:-120}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
RESULT_ROOT="${RESULT_ROOT:-/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl}"
POD_RESULT_ROOT="${POD_RESULT_ROOT:-${RESULT_ROOT}}"
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
  PROFILE                  quick|smoke|bandwidth|collective-bandwidth. smoke only checks torchrun connectivity. Default: quick
  PRE_CLEAN                1 runs cleanup before checks; 0 disables it. Default: 1
  PRE_CLEAN_CMD            Command used for cleanup. Default: pkill healthcheck torchrun/python.
  GPUS_PER_NODE            Local device count per pod. Default: 8
  DIST_BACKEND             PyTorch distributed backend name. Default: nccl
  HEALTHCHECK_MASTER_PORT  auto or explicit rendezvous port. Default: auto
  DEVICE_VENDOR            Result metadata. Default: metax
  COMM_RUNTIME             Result metadata. Default: mccl
  BANDWIDTH_MESSAGE_SIZES  Payload sizes for PROFILE=bandwidth. Default: 1G,4G,8G,16G
  BANDWIDTH_WARMUP         Warmup rounds for PROFILE=bandwidth. Default: 5
  BANDWIDTH_ITERS          Timed rounds for PROFILE=bandwidth. Default: 100
  BANDWIDTH_MIN_BUSBW      Second-lowest BusBW gate in GB/s. Default: 270
  BANDWIDTH_AVG_BUSBW      Average BusBW gate in GB/s. Default: 290
  COLLECTIVE_BANDWIDTH_OPS Ops for PROFILE=collective-bandwidth. Default: all_reduce,reduce_scatter,all_gather,all_to_all,all_to_allv
  COLLECTIVE_BANDWIDTH_MESSAGE_SIZES
                           Payload sizes for PROFILE=collective-bandwidth. Default: 1G
  COLLECTIVE_BANDWIDTH_WARMUP
                           Warmup rounds for PROFILE=collective-bandwidth. Default: 5
  COLLECTIVE_BANDWIDTH_ITERS
                           Timed rounds for PROFILE=collective-bandwidth. Default: 30
  COLLECTIVE_BANDWIDTH_MIN_BUSBW
                           Second-lowest BusBW gate in GB/s. Default: 0
  COLLECTIVE_BANDWIDTH_AVG_BUSBW
                           Average BusBW gate in GB/s. Default: 0
  COLLECTIVE_BANDWIDTH_EP_SIZE
                           EP group size for all_to_allv. Default: 8
  FAULT_*                  Optional fault injection envs for backend/sleep/nan/corrupt tests.
  STATIC_CMD               Command executed in each pod for static checks. Default: MetaX static probe
  SINGLE_NODE_CMD          Command executed in each pod for single-node checks. Default: MetaX 8-card torchrun
  MULTI_NODE_CMD           Command executed in every pod concurrently for multi-node checks. Default: MetaX current vcjob torchrun
  RESULT_ROOT              Shared result root. Default: /afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl
  POD_RESULT_ROOT          Result root visible inside target pods. Default: RESULT_ROOT
  RUN_ID                   Run id. Default: current timestamp
  EXEC_TIMEOUT_SECONDS     Per-pod exec timeout. Default: 3600
  HANG_TIMEOUT_SECONDS     Multi-node wall-clock hang timeout. 0 disables it. Default: 0
  HANG_KILL_GRACE_SECONDS  Seconds to wait after cleanup before local vcctl exec kill. Default: 10
  HANG_CLEANUP_CMD         Command executed in pods after hang diagnostics. Default: PRE_CLEAN_CMD
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
echo "[vcctl-healthcheck] pod result  : ${POD_RESULT_ROOT}"
echo "[vcctl-healthcheck] run id      : ${RUN_ID}"
echo "[vcctl-healthcheck] output      : ${OUT_DIR}"
echo "[vcctl-healthcheck] dry run     : ${DRY_RUN}"
if [[ "${HANG_TIMEOUT_SECONDS}" != "0" ]]; then
  echo "[vcctl-healthcheck] hang timeout: ${HANG_TIMEOUT_SECONDS}s"
fi

args=(
  "${PROJECT_DIR}/tools/vcctl_healthcheck_driver.py"
  --job-name "${JOB_NAME:-fixture}"
  --namespace "${NAMESPACE}"
  --mode "${MODE}"
  --device-type "${DEVICE_TYPE}"
  --result-root "${RESULT_ROOT}"
  --pod-result-root "${POD_RESULT_ROOT}"
  --run-id "${RUN_ID}"
  --vcctl-bin "${VCCTL_BIN}"
  --healthcheck-master-port "${HEALTHCHECK_MASTER_PORT}"
  --pre-clean-cmd "$([[ "${PRE_CLEAN}" == "1" || "${PRE_CLEAN}" == "true" ]] && printf '%s' "${PRE_CLEAN_CMD}" || true)"
  --static-cmd "${STATIC_CMD}"
  --single-node-cmd "${SINGLE_NODE_CMD}"
  --multi-node-cmd "${MULTI_NODE_CMD}"
  --exec-timeout-seconds "${EXEC_TIMEOUT_SECONDS}"
  --hang-timeout-seconds "${HANG_TIMEOUT_SECONDS}"
  --hang-kill-grace-seconds "${HANG_KILL_GRACE_SECONDS}"
  --hang-cleanup-cmd "${HANG_CLEANUP_CMD}"
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
