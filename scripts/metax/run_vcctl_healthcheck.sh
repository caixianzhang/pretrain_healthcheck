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
MOE_PATTERNS="${MOE_PATTERNS:-uniform,skewed,hot_expert,random,empty_expert}"
COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES="1K,2K,4K,8K,16K,32K,64K,128K,256K,512K,1M,2M,4M,8M,16M,32M,64M,128M,256M,512M,1G,2G"
COLLECTIVE_ACCEPTANCE_OPS="all_reduce,reduce_scatter,all_gather,broadcast,all_to_all,all_to_allv"
COLLECTIVE_BANDWIDTH_OPS="${COLLECTIVE_BANDWIDTH_OPS:-${COLLECTIVE_ACCEPTANCE_OPS}}"
if [[ "${PROFILE}" == "dynamic-suite" ]]; then
  MESSAGE_SIZES="${MESSAGE_SIZES:-1M}"
  WARMUP="${WARMUP:-1}"
  ITERS="${ITERS:-1}"
  BANDWIDTH_MESSAGE_SIZES="${BANDWIDTH_MESSAGE_SIZES:-1G}"
  BANDWIDTH_WARMUP="${BANDWIDTH_WARMUP:-1}"
  BANDWIDTH_ITERS="${BANDWIDTH_ITERS:-3}"
  BANDWIDTH_MIN_BUSBW="${BANDWIDTH_MIN_BUSBW:-0}"
  BANDWIDTH_AVG_BUSBW="${BANDWIDTH_AVG_BUSBW:-0}"
  COLLECTIVE_BANDWIDTH_MESSAGE_SIZES="${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES:-${COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES}}"
  COLLECTIVE_BANDWIDTH_WARMUP="${COLLECTIVE_BANDWIDTH_WARMUP:-1}"
  COLLECTIVE_BANDWIDTH_ITERS="${COLLECTIVE_BANDWIDTH_ITERS:-3}"
else
  MESSAGE_SIZES="${MESSAGE_SIZES:-1M,16M,64M}"
  WARMUP="${WARMUP:-2}"
  ITERS="${ITERS:-5}"
  BANDWIDTH_MESSAGE_SIZES="${BANDWIDTH_MESSAGE_SIZES:-1G,4G,8G,16G}"
  BANDWIDTH_WARMUP="${BANDWIDTH_WARMUP:-5}"
  BANDWIDTH_ITERS="${BANDWIDTH_ITERS:-100}"
  BANDWIDTH_MIN_BUSBW="${BANDWIDTH_MIN_BUSBW:-270}"
  BANDWIDTH_AVG_BUSBW="${BANDWIDTH_AVG_BUSBW:-290}"
  COLLECTIVE_BANDWIDTH_MESSAGE_SIZES="${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES:-${COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES}}"
  COLLECTIVE_BANDWIDTH_WARMUP="${COLLECTIVE_BANDWIDTH_WARMUP:-5}"
  COLLECTIVE_BANDWIDTH_ITERS="${COLLECTIVE_BANDWIDTH_ITERS:-30}"
fi
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
STATIC_OUTPUT_MODE="${STATIC_OUTPUT_MODE:-compact}"
STATIC_TMP_ROOT="${STATIC_TMP_ROOT:-/tmp}"
STATIC_KEEP_LOCAL_TMP="${STATIC_KEEP_LOCAL_TMP:-1}"
STATIC_COPY_RAW_OUTPUT="${STATIC_COPY_RAW_OUTPUT:-0}"
STATIC_STDOUT_MAX_BYTES="${STATIC_STDOUT_MAX_BYTES:-1048576}"
STATIC_CMD="${STATIC_CMD:-cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && OUT_DIR=\"\$HC_POD_RESULT_DIR\" STATIC_OUTPUT_MODE=\"${STATIC_OUTPUT_MODE}\" STATIC_TMP_ROOT=\"${STATIC_TMP_ROOT}\" STATIC_KEEP_LOCAL_TMP=\"${STATIC_KEEP_LOCAL_TMP}\" STATIC_COPY_RAW_OUTPUT=\"${STATIC_COPY_RAW_OUTPUT}\" STATIC_STDOUT_MAX_BYTES=\"${STATIC_STDOUT_MAX_BYTES}\" bash scripts/metax/probe_pod_capabilities.sh}"
FAULT_ENV="FAULT_BACKEND=\"${FAULT_BACKEND}\" FAULT_SLEEP_RANK=\"${FAULT_SLEEP_RANK}\" FAULT_SLEEP_SECONDS=\"${FAULT_SLEEP_SECONDS}\" FAULT_NAN_RANK=\"${FAULT_NAN_RANK}\" FAULT_CORRUPT_RANK=\"${FAULT_CORRUPT_RANK}\""
if [[ -z "${SINGLE_NODE_CMD:-}" ]]; then
  if [[ "${PROFILE}" == "dynamic-suite" ]]; then
    SINGLE_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PROJECT_REMOTE_DIR=\"${PROJECT_REMOTE_DIR}\" PYTHONPATH=\"${PROJECT_REMOTE_DIR}:\${PYTHONPATH:-}\" && STAGE_KIND=\"dynamic-suite\" GPUS_PER_NODE=\"${GPUS_PER_NODE}\" DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" DTYPE=\"${DTYPE}\" MESSAGE_SIZES=\"${MESSAGE_SIZES}\" MOE_PATTERNS=\"${MOE_PATTERNS}\" WARMUP=\"${WARMUP}\" ITERS=\"${ITERS}\" BANDWIDTH_MESSAGE_SIZES=\"${BANDWIDTH_MESSAGE_SIZES}\" BANDWIDTH_WARMUP=\"${BANDWIDTH_WARMUP}\" BANDWIDTH_ITERS=\"${BANDWIDTH_ITERS}\" BANDWIDTH_MIN_BUSBW=\"${BANDWIDTH_MIN_BUSBW}\" BANDWIDTH_AVG_BUSBW=\"${BANDWIDTH_AVG_BUSBW}\" COLLECTIVE_BANDWIDTH_OPS=\"${COLLECTIVE_BANDWIDTH_OPS}\" COLLECTIVE_BANDWIDTH_MESSAGE_SIZES=\"${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES}\" COLLECTIVE_BANDWIDTH_MOE_PATTERNS=\"${COLLECTIVE_BANDWIDTH_MOE_PATTERNS}\" COLLECTIVE_BANDWIDTH_EP_SIZE=\"${COLLECTIVE_BANDWIDTH_EP_SIZE}\" COLLECTIVE_BANDWIDTH_WARMUP=\"${COLLECTIVE_BANDWIDTH_WARMUP}\" COLLECTIVE_BANDWIDTH_ITERS=\"${COLLECTIVE_BANDWIDTH_ITERS}\" COLLECTIVE_BANDWIDTH_MIN_BUSBW=\"${COLLECTIVE_BANDWIDTH_MIN_BUSBW}\" COLLECTIVE_BANDWIDTH_AVG_BUSBW=\"${COLLECTIVE_BANDWIDTH_AVG_BUSBW}\" ${FAULT_ENV} bash scripts/metax/run_single_node_dynamic_stage.sh"
  else
    SINGLE_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} torchrun --standalone --nproc-per-node=\"${GPUS_PER_NODE}\" -m pretrain_healthcheck.cli run-single-node --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${MESSAGE_SIZES}\" --moe-patterns \"${MOE_PATTERNS}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --seed \"${SEED}\"; rc=\$?; python3 -m pretrain_healthcheck.cli analyze --input-dir \"\$HC_POD_RESULT_DIR\" --output \"\$HC_POD_RESULT_DIR/report.md\"; exit \$rc"
  fi
fi
if [[ -z "${MULTI_NODE_CMD:-}" ]]; then
  if [[ "${PROFILE}" == "smoke" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"1\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli ping-group --output-dir \"\$HC_POD_RESULT_DIR\" --test-round smoke --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\""
  elif [[ "${PROFILE}" == "bandwidth" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-bandwidth --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${BANDWIDTH_MESSAGE_SIZES}\" --warmup \"${BANDWIDTH_WARMUP}\" --iters \"${BANDWIDTH_ITERS}\" --seed \"${SEED}\" --min-busbw \"${BANDWIDTH_MIN_BUSBW}\" --avg-busbw \"${BANDWIDTH_AVG_BUSBW}\" --test-round bandwidth --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; rc=\$?; exit \$rc"
  elif [[ "${PROFILE}" == "collective-bandwidth" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-collective-bandwidth --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES}\" --ops \"${COLLECTIVE_BANDWIDTH_OPS}\" --moe-patterns \"${COLLECTIVE_BANDWIDTH_MOE_PATTERNS}\" --ep-size \"${COLLECTIVE_BANDWIDTH_EP_SIZE}\" --warmup \"${COLLECTIVE_BANDWIDTH_WARMUP}\" --iters \"${COLLECTIVE_BANDWIDTH_ITERS}\" --seed \"${SEED}\" --min-busbw \"${COLLECTIVE_BANDWIDTH_MIN_BUSBW}\" --avg-busbw \"${COLLECTIVE_BANDWIDTH_AVG_BUSBW}\" --test-round collective_bandwidth --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; rc=\$?; exit \$rc"
  elif [[ "${PROFILE}" == "dynamic-suite" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-dynamic-suite --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${MESSAGE_SIZES}\" --moe-patterns \"${MOE_PATTERNS}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --bandwidth-message-sizes \"${BANDWIDTH_MESSAGE_SIZES}\" --bandwidth-warmup \"${BANDWIDTH_WARMUP}\" --bandwidth-iters \"${BANDWIDTH_ITERS}\" --bandwidth-min-busbw \"${BANDWIDTH_MIN_BUSBW}\" --bandwidth-avg-busbw \"${BANDWIDTH_AVG_BUSBW}\" --collective-bandwidth-message-sizes \"${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES}\" --collective-bandwidth-ops \"${COLLECTIVE_BANDWIDTH_OPS}\" --collective-bandwidth-moe-patterns \"${COLLECTIVE_BANDWIDTH_MOE_PATTERNS}\" --collective-bandwidth-ep-size \"${COLLECTIVE_BANDWIDTH_EP_SIZE}\" --collective-bandwidth-warmup \"${COLLECTIVE_BANDWIDTH_WARMUP}\" --collective-bandwidth-iters \"${COLLECTIVE_BANDWIDTH_ITERS}\" --collective-bandwidth-min-busbw \"${COLLECTIVE_BANDWIDTH_MIN_BUSBW}\" --collective-bandwidth-avg-busbw \"${COLLECTIVE_BANDWIDTH_AVG_BUSBW}\" --seed \"${SEED}\" --test-round dynamic_suite --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; rc=\$?; exit \$rc"
  else
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-group --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${MESSAGE_SIZES}\" --moe-patterns \"${MOE_PATTERNS}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --seed \"${SEED}\" --test-round current_vcjob --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; rc=\$?; if [ \"\$RANK\" = \"0\" ]; then python3 -m pretrain_healthcheck.cli analyze --input-dir \"\$HC_POD_RESULT_DIR\" --output \"\$HC_POD_RESULT_DIR/report.md\"; fi; exit \$rc"
  fi
fi
if [[ -z "${EXEC_TIMEOUT_SECONDS:-}" && ( "${PROFILE}" == "dynamic-suite" || "${PROFILE}" == "collective-bandwidth" ) && "${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES}" == "${COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES}" ]]; then
  EXEC_TIMEOUT_SECONDS="1800"
fi
EXEC_TIMEOUT_SECONDS="${EXEC_TIMEOUT_SECONDS:-180}"
STATIC_EXEC_TIMEOUT_SECONDS="${STATIC_EXEC_TIMEOUT_SECONDS:-60}"
STATIC_DRIVER_TMP_ROOT="${STATIC_DRIVER_TMP_ROOT:-/tmp}"
VCCTL_TIMEOUT_SECONDS="${VCCTL_TIMEOUT_SECONDS:-120}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
STATIC_COMPARE="${STATIC_COMPARE:-1}"
STATIC_COMPARE_WORKERS="${STATIC_COMPARE_WORKERS:-0}"
STATIC_COMPARE_STRICT="${STATIC_COMPARE_STRICT:-1}"
STATIC_EXPECTED_GPUS="${STATIC_EXPECTED_GPUS:-${GPUS_PER_NODE}}"
STATIC_EXPECTED_XSCALE_PORTS="${STATIC_EXPECTED_XSCALE_PORTS:-0}"
STATIC_KEEP_POD_FILES="${STATIC_KEEP_POD_FILES:-0}"
STATIC_KEEP_EXEC_LOGS="${STATIC_KEEP_EXEC_LOGS:-0}"
DYNAMIC_COMPARE="${DYNAMIC_COMPARE:-0}"
DYNAMIC_COMPARE_STRICT="${DYNAMIC_COMPARE_STRICT:-1}"
DYNAMIC_COMPARE_RATIO_THRESHOLD="${DYNAMIC_COMPARE_RATIO_THRESHOLD:-0.7}"
DYNAMIC_KEEP_EXEC_LOGS="${DYNAMIC_KEEP_EXEC_LOGS:-0}"
RESULT_ROOT="${RESULT_ROOT:-/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
POD_RESULT_ROOT="${POD_RESULT_ROOT:-/tmp/pretrain_healthcheck_driver_${RUN_ID}}"
RUN_STAGE="${RUN_STAGE:-}"
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
  PROFILE                  quick|smoke|bandwidth|collective-bandwidth|dynamic-suite. smoke only checks torchrun connectivity. Default: quick
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
  COLLECTIVE_BANDWIDTH_OPS Ops for PROFILE=collective-bandwidth/dynamic-suite. Default: all_reduce,reduce_scatter,all_gather,broadcast,all_to_all,all_to_allv
  COLLECTIVE_BANDWIDTH_MESSAGE_SIZES
                           Payload sizes for PROFILE=collective-bandwidth/dynamic-suite. Default: 1K,2K,...,2G
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
  STATIC_OUTPUT_MODE       compact keeps only compact pod facts before run-level aggregation. Default: compact
  STATIC_TMP_ROOT          Pod-local temp root for static raw work files. Default: /tmp
  STATIC_KEEP_LOCAL_TMP    1 keeps pod-local static temp files. Default: 1
  STATIC_COPY_RAW_OUTPUT   1 copies raw static temp files to shared result dir. Default: 0
  STATIC_STDOUT_MAX_BYTES  Max bytes for one static stdout result frame. Default: 1048576
  SINGLE_NODE_CMD          Command executed in each pod for single-node checks. Default: MetaX 8-card torchrun
  MULTI_NODE_CMD           Command executed in every pod concurrently for multi-node checks. Default: MetaX current vcjob torchrun
  RESULT_ROOT              Shared result root. Default: /afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl
  POD_RESULT_ROOT          Temporary pod result root. Default: /tmp/pretrain_healthcheck_driver_<RUN_ID>
  RUN_ID                   Run id. Default: current timestamp
  RUN_STAGE                Override result stage directory. Default: derived from MODE
  EXEC_TIMEOUT_SECONDS     Per-pod exec timeout. Default: 180
  STATIC_EXEC_TIMEOUT_SECONDS
                           Per-pod static exec timeout. Default: 120
  STATIC_DRIVER_TMP_ROOT   Dev-machine local temp root for static exec stdout/stderr. Default: /tmp
  MAX_PARALLEL             0 means all pods concurrently. Default: 0
  STATIC_COMPARE           1 compares static outputs across pods. Default: 1
  STATIC_COMPARE_WORKERS   Parallel static parser workers; 0 means auto. Default: 0
  STATIC_COMPARE_STRICT    1 lets static outliers affect overall status. Default: 1
  STATIC_EXPECTED_GPUS     Expected visible GPU count per pod. Default: GPUS_PER_NODE
  STATIC_EXPECTED_XSCALE_PORTS
                           Expected xscale/HCA port count per pod; 0 disables this gate. Default: 0
  STATIC_KEEP_POD_FILES    1 keeps pod_results/<pod>/static after aggregation. Default: 0
  STATIC_KEEP_EXEC_LOGS    1 keeps logs/<pod>.static.stdout/stderr after aggregation. Default: 0
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

RUN_DIR="${RESULT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "[vcctl-healthcheck] project     : ${PROJECT_DIR}"
echo "[vcctl-healthcheck] job         : ${JOB_NAME:-<fixture>}"
echo "[vcctl-healthcheck] namespace   : ${NAMESPACE}"
echo "[vcctl-healthcheck] mode        : ${MODE}"
echo "[vcctl-healthcheck] device      : ${DEVICE_TYPE}"
echo "[vcctl-healthcheck] result root : ${RESULT_ROOT}"
echo "[vcctl-healthcheck] pod result  : ${POD_RESULT_ROOT}"
echo "[vcctl-healthcheck] run id      : ${RUN_ID}"
echo "[vcctl-healthcheck] output      : ${RUN_DIR}"
echo "[vcctl-healthcheck] dry run     : ${DRY_RUN}"
stage_for_mode() {
  case "$1" in
    static)
      printf '%s\n' "static"
      ;;
    single-node)
      if [[ "${PROFILE}" == "dynamic-suite" ]]; then
        printf '%s\n' "dynamic_suite"
      else
        printf '%s\n' "single_node"
      fi
      ;;
    multi-node)
      printf '%s\n' "multi_node"
      ;;
    *)
      echo "[vcctl-healthcheck] unsupported stage mode: $1" >&2
      return 2
      ;;
  esac
}

run_driver_for_mode() {
  local stage_mode="$1"
  local run_stage="$2"
  if [[ -n "${RUN_STAGE}" ]]; then
    run_stage="${RUN_STAGE}"
  fi
  local stage_out="${RUN_DIR}/${run_stage}"

  echo "[vcctl-healthcheck] stage       : ${run_stage}"
  echo "[vcctl-healthcheck] stage mode  : ${stage_mode}"
  echo "[vcctl-healthcheck] stage output: ${stage_out}"

  local args=(
    "${PROJECT_DIR}/tools/vcctl_healthcheck_driver.py"
    --job-name "${JOB_NAME:-fixture}"
    --namespace "${NAMESPACE}"
    --mode "${stage_mode}"
    --run-stage "${run_stage}"
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
    --static-exec-timeout-seconds "${STATIC_EXEC_TIMEOUT_SECONDS}"
    --static-driver-tmp-root "${STATIC_DRIVER_TMP_ROOT}"
    --vcctl-timeout-seconds "${VCCTL_TIMEOUT_SECONDS}"
    --max-parallel "${MAX_PARALLEL}"
    --static-compare-workers "${STATIC_COMPARE_WORKERS}"
    --static-expected-gpus "${STATIC_EXPECTED_GPUS}"
    --static-expected-xscale-ports "${STATIC_EXPECTED_XSCALE_PORTS}"
    --dynamic-compare-ratio-threshold "${DYNAMIC_COMPARE_RATIO_THRESHOLD}"
  )

  if [[ "${STATIC_KEEP_POD_FILES}" == "1" || "${STATIC_KEEP_POD_FILES}" == "true" ]]; then
    args+=(--static-keep-pod-files)
  else
    args+=(--no-static-keep-pod-files)
  fi

  if [[ "${STATIC_KEEP_EXEC_LOGS}" == "1" || "${STATIC_KEEP_EXEC_LOGS}" == "true" ]]; then
    args+=(--static-keep-exec-logs)
  else
    args+=(--no-static-keep-exec-logs)
  fi

  if [[ "${STATIC_COMPARE}" == "0" || "${STATIC_COMPARE}" == "false" ]]; then
    args+=(--no-static-compare)
  else
    args+=(--static-compare)
  fi

  if [[ "${STATIC_COMPARE_STRICT}" == "0" || "${STATIC_COMPARE_STRICT}" == "false" ]]; then
    args+=(--no-static-compare-strict)
  else
    args+=(--static-compare-strict)
  fi

  if [[ "${DYNAMIC_COMPARE}" == "0" || "${DYNAMIC_COMPARE}" == "false" ]]; then
    args+=(--no-dynamic-compare)
  else
    args+=(--dynamic-compare)
  fi

  if [[ "${DYNAMIC_COMPARE_STRICT}" == "0" || "${DYNAMIC_COMPARE_STRICT}" == "false" ]]; then
    args+=(--no-dynamic-compare-strict)
  else
    args+=(--dynamic-compare-strict)
  fi

  if [[ "${DYNAMIC_KEEP_EXEC_LOGS}" == "1" || "${DYNAMIC_KEEP_EXEC_LOGS}" == "true" ]]; then
    args+=(--dynamic-keep-exec-logs)
  else
    args+=(--no-dynamic-keep-exec-logs)
  fi

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
}

if [[ "${MODE}" == "all" ]]; then
  overall_rc=0
  for stage_mode in static single-node multi-node; do
    run_stage="$(stage_for_mode "${stage_mode}")"
    if ! run_driver_for_mode "${stage_mode}" "${run_stage}"; then
      overall_rc=1
    fi
  done
  exit "${overall_rc}"
fi

run_stage="$(stage_for_mode "${MODE}")"
run_driver_for_mode "${MODE}" "${run_stage}"
