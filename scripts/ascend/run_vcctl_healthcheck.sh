#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_DIR}/scripts/common/driver_python.sh"

JOB_NAME="${JOB_NAME:-grj-megatron-128-235b-moe0708}"
NAMESPACE="${NAMESPACE:-default}"
MODE="${MODE:-all}"
DEVICE_TYPE="${DEVICE_TYPE:-ascend}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
PROJECT_REMOTE_DIR="${PROJECT_REMOTE_DIR:-${PROJECT_DIR}}"
PROFILE="${PROFILE:-quick}"
PRE_CLEAN="${PRE_CLEAN:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-16}"
DIST_BACKEND="${DIST_BACKEND:-hccl}"
HEALTHCHECK_MASTER_PORT="${HEALTHCHECK_MASTER_PORT:-auto}"
DEVICE_VENDOR="${DEVICE_VENDOR:-ascend}"
COMM_RUNTIME="${COMM_RUNTIME:-hccl}"
DTYPE="${DTYPE:-bf16}"
ASCEND_LD_LIBRARY_PATH="${ASCEND_LD_LIBRARY_PATH:-/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64}"
ASCEND_ENV_CMD="${ASCEND_ENV_CMD:-}"
if [[ -z "${ASCEND_ENV_CMD}" ]]; then
  ASCEND_ENV_CMD="export LD_LIBRARY_PATH=\"${ASCEND_LD_LIBRARY_PATH}:\${LD_LIBRARY_PATH:-}\""
fi
MOE_PATTERNS="${MOE_PATTERNS:-uniform,skewed,hot_expert,random,empty_expert}"
COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES="1K,2K,4K,8K,16K,32K,64K,128K,256K,512K,1M,2M,4M,8M,16M,32M,64M,128M,256M,512M,1G,2G"
COLLECTIVE_ACCEPTANCE_OPS="all_reduce,reduce_scatter,all_gather,broadcast,all_to_all,all_to_allv"
SINGLE_NODE_GATE_MESSAGE_SIZES="${COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES}"
SINGLE_NODE_GATE_MOE_PATTERNS="uniform,hot_expert,empty_expert"
COLLECTIVE_BANDWIDTH_OPS="${COLLECTIVE_BANDWIDTH_OPS:-${COLLECTIVE_ACCEPTANCE_OPS}}"
TRAINING_TOPOLOGY_MANIFEST="${TRAINING_TOPOLOGY_MANIFEST:-}"
POD_TRAINING_TOPOLOGY_MANIFEST="${POD_TRAINING_TOPOLOGY_MANIFEST:-${TRAINING_TOPOLOGY_MANIFEST}}"
TOPOLOGY_MANIFEST_SHA256="${TOPOLOGY_MANIFEST_SHA256:-}"
TOPOLOGY_WARMUP="${TOPOLOGY_WARMUP:-1}"
TOPOLOGY_ITERS="${TOPOLOGY_ITERS:-1}"
TOPOLOGY_OVERLAP_CANARY="${TOPOLOGY_OVERLAP_CANARY:-0}"
if [[ "${PROFILE}" == "dynamic-suite" || "${PROFILE}" == "training-topology" ]]; then
  MESSAGE_SIZES="${MESSAGE_SIZES:-1M}"
  WARMUP="${WARMUP:-1}"
  ITERS="${ITERS:-1}"
  BANDWIDTH_MESSAGE_SIZES="${BANDWIDTH_MESSAGE_SIZES:-1G}"
  BANDWIDTH_WARMUP="${BANDWIDTH_WARMUP:-1}"
  BANDWIDTH_ITERS="${BANDWIDTH_ITERS:-3}"
  BANDWIDTH_MIN_BUSBW="${BANDWIDTH_MIN_BUSBW:-0}"
  BANDWIDTH_AVG_BUSBW="${BANDWIDTH_AVG_BUSBW:-0}"
  if [[ "${MODE}" == "single-node" ]]; then
    COLLECTIVE_BANDWIDTH_MESSAGE_SIZES="${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES:-${SINGLE_NODE_GATE_MESSAGE_SIZES}}"
    COLLECTIVE_BANDWIDTH_MOE_PATTERNS="${COLLECTIVE_BANDWIDTH_MOE_PATTERNS:-${SINGLE_NODE_GATE_MOE_PATTERNS}}"
  else
    COLLECTIVE_BANDWIDTH_MESSAGE_SIZES="${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES:-${COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES}}"
  fi
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
DYNAMIC_COMPARE_MEASUREMENT_BATCHES="${DYNAMIC_COMPARE_MEASUREMENT_BATCHES:-1}"
DYNAMIC_COMPARE_RETEST_MEASUREMENT_BATCHES="${DYNAMIC_COMPARE_RETEST_MEASUREMENT_BATCHES:-3}"
for batch_var in DYNAMIC_COMPARE_MEASUREMENT_BATCHES DYNAMIC_COMPARE_RETEST_MEASUREMENT_BATCHES; do
  batch_value="${!batch_var}"
  if [[ ! "${batch_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[vcctl-healthcheck] ${batch_var} must be an integer >= 1, got: ${batch_value}" >&2
    exit 2
  fi
done
DYNAMIC_COMPARE_EFFECTIVE_RETEST_MEASUREMENT_BATCHES="${DYNAMIC_COMPARE_RETEST_MEASUREMENT_BATCHES}"
if (( DYNAMIC_COMPARE_EFFECTIVE_RETEST_MEASUREMENT_BATCHES < DYNAMIC_COMPARE_MEASUREMENT_BATCHES )); then
  DYNAMIC_COMPARE_EFFECTIVE_RETEST_MEASUREMENT_BATCHES="${DYNAMIC_COMPARE_MEASUREMENT_BATCHES}"
fi
DYNAMIC_COMPARE_BUSBW_RATIO_THRESHOLD="${DYNAMIC_COMPARE_BUSBW_RATIO_THRESHOLD:-${DYNAMIC_COMPARE_RATIO_THRESHOLD:-0.7}}"
DYNAMIC_COMPARE_LATENCY_RATIO_THRESHOLD="${DYNAMIC_COMPARE_LATENCY_RATIO_THRESHOLD:-1.5}"
DYNAMIC_COMPARE_SMALL_MAX_SIZE="${DYNAMIC_COMPARE_SMALL_MAX_SIZE:-1M}"
DYNAMIC_COMPARE_LARGE_MIN_SIZE="${DYNAMIC_COMPARE_LARGE_MIN_SIZE:-1G}"
DYNAMIC_COMPARE_SMALL_LATENCY_WARN="${DYNAMIC_COMPARE_SMALL_LATENCY_WARN:-0}"
DYNAMIC_COMPARE_SMALL_LATENCY_ABS_DELTA_MS="${DYNAMIC_COMPARE_SMALL_LATENCY_ABS_DELTA_MS:-0.2}"
DYNAMIC_COMPARE_SMALL_LATENCY_MAD_MULTIPLIER="${DYNAMIC_COMPARE_SMALL_LATENCY_MAD_MULTIPLIER:-6}"
DYNAMIC_COMPARE_MIN_COHORT="${DYNAMIC_COMPARE_MIN_COHORT:-3}"
DYNAMIC_COMPARE_AUTO_RETEST="${DYNAMIC_COMPARE_AUTO_RETEST:-1}"
FAULT_BACKEND="${FAULT_BACKEND:-}"
FAULT_SLEEP_RANK="${FAULT_SLEEP_RANK:-}"
FAULT_SLEEP_SECONDS="${FAULT_SLEEP_SECONDS:-30}"
FAULT_NAN_RANK="${FAULT_NAN_RANK:-}"
FAULT_CORRUPT_RANK="${FAULT_CORRUPT_RANK:-}"
FAULT_SLEEP_POD="${FAULT_SLEEP_POD:-}"
FAULT_SLEEP_NODE="${FAULT_SLEEP_NODE:-}"
FAULT_NAN_POD="${FAULT_NAN_POD:-}"
FAULT_NAN_NODE="${FAULT_NAN_NODE:-}"
FAULT_CORRUPT_POD="${FAULT_CORRUPT_POD:-}"
FAULT_CORRUPT_NODE="${FAULT_CORRUPT_NODE:-}"
FAULT_JOIN_TIMEOUT_RANK="${FAULT_JOIN_TIMEOUT_RANK:-}"
FAULT_JOIN_TIMEOUT_POD="${FAULT_JOIN_TIMEOUT_POD:-}"
FAULT_JOIN_TIMEOUT_NODE="${FAULT_JOIN_TIMEOUT_NODE:-}"
FAULT_JOIN_TIMEOUT_SECONDS="${FAULT_JOIN_TIMEOUT_SECONDS:-300}"
FAULT_NET_SLOW_RANK="${FAULT_NET_SLOW_RANK:-}"
FAULT_NET_SLOW_POD="${FAULT_NET_SLOW_POD:-}"
FAULT_NET_SLOW_NODE="${FAULT_NET_SLOW_NODE:-}"
FAULT_NET_SLOW_SECONDS="${FAULT_NET_SLOW_SECONDS:-0.2}"
FAULT_RANK_EXIT_RANK="${FAULT_RANK_EXIT_RANK:-}"
FAULT_RANK_EXIT_POD="${FAULT_RANK_EXIT_POD:-}"
FAULT_RANK_EXIT_NODE="${FAULT_RANK_EXIT_NODE:-}"
FAULT_RANK_EXIT_CODE="${FAULT_RANK_EXIT_CODE:-17}"
FAULT_COMM_ENV_BAD_RANK="${FAULT_COMM_ENV_BAD_RANK:-}"
FAULT_COMM_ENV_BAD_POD="${FAULT_COMM_ENV_BAD_POD:-}"
FAULT_COMM_ENV_BAD_NODE="${FAULT_COMM_ENV_BAD_NODE:-}"
FAULT_ETH_FALLBACK_RANK="${FAULT_ETH_FALLBACK_RANK:-}"
FAULT_ETH_FALLBACK_POD="${FAULT_ETH_FALLBACK_POD:-}"
FAULT_ETH_FALLBACK_NODE="${FAULT_ETH_FALLBACK_NODE:-}"
FAULT_SLEEP_PODS="${FAULT_SLEEP_PODS:-}"
FAULT_SLEEP_NODES="${FAULT_SLEEP_NODES:-}"
FAULT_NAN_PODS="${FAULT_NAN_PODS:-}"
FAULT_NAN_NODES="${FAULT_NAN_NODES:-}"
FAULT_CORRUPT_PODS="${FAULT_CORRUPT_PODS:-}"
FAULT_CORRUPT_NODES="${FAULT_CORRUPT_NODES:-}"
FAULT_JOIN_TIMEOUT_PODS="${FAULT_JOIN_TIMEOUT_PODS:-}"
FAULT_JOIN_TIMEOUT_NODES="${FAULT_JOIN_TIMEOUT_NODES:-}"
FAULT_NET_SLOW_PODS="${FAULT_NET_SLOW_PODS:-}"
FAULT_NET_SLOW_NODES="${FAULT_NET_SLOW_NODES:-}"
FAULT_RANK_EXIT_PODS="${FAULT_RANK_EXIT_PODS:-}"
FAULT_RANK_EXIT_NODES="${FAULT_RANK_EXIT_NODES:-}"
FAULT_COMM_ENV_BAD_PODS="${FAULT_COMM_ENV_BAD_PODS:-}"
FAULT_COMM_ENV_BAD_NODES="${FAULT_COMM_ENV_BAD_NODES:-}"
FAULT_ETH_FALLBACK_PODS="${FAULT_ETH_FALLBACK_PODS:-}"
FAULT_ETH_FALLBACK_NODES="${FAULT_ETH_FALLBACK_NODES:-}"
COMM_PATH_DEBUG="${COMM_PATH_DEBUG:-0}"
DYNAMIC_FAULT_TYPE="${DYNAMIC_FAULT_TYPE:-}"
DYNAMIC_FAULT_POD="${DYNAMIC_FAULT_POD:-}"
DYNAMIC_FAULT_NODE="${DYNAMIC_FAULT_NODE:-}"
DYNAMIC_FAULT_LOCAL_RANK="${DYNAMIC_FAULT_LOCAL_RANK:-${DYNAMIC_FAULT_RANK:-}}"
DYNAMIC_FAULT_SLEEP_SECONDS="${DYNAMIC_FAULT_SLEEP_SECONDS:-300}"
DYNAMIC_FAULT_FRAME_BYTES="${DYNAMIC_FAULT_FRAME_BYTES:-512}"
STATIC_FAULT_TYPE="${STATIC_FAULT_TYPE:-}"
STATIC_FAULT_POD="${STATIC_FAULT_POD:-}"
STATIC_FAULT_RANK="${STATIC_FAULT_RANK:-}"
STATIC_FAULT_NODE="${STATIC_FAULT_NODE:-}"
STATIC_FAULT_SLEEP_SECONDS="${STATIC_FAULT_SLEEP_SECONDS:-600}"
PRE_CLEAN_CMD="${PRE_CLEAN_CMD:-ps -eo pid=,args= | awk '/[p]retrain_healthcheck.cli|[t]orchrun/ {print \$1}' | xargs -r kill -TERM || true}"
STATIC_OUTPUT_MODE="${STATIC_OUTPUT_MODE:-compact}"
STATIC_TMP_ROOT="${STATIC_TMP_ROOT:-/tmp}"
STATIC_KEEP_LOCAL_TMP="${STATIC_KEEP_LOCAL_TMP:-1}"
STATIC_COPY_RAW_OUTPUT="${STATIC_COPY_RAW_OUTPUT:-0}"
STATIC_STDOUT_MAX_BYTES="${STATIC_STDOUT_MAX_BYTES:-1048576}"
STATIC_FAULT_ENV="STATIC_FAULT_TYPE=\"${STATIC_FAULT_TYPE}\" STATIC_FAULT_POD=\"${STATIC_FAULT_POD}\" STATIC_FAULT_RANK=\"${STATIC_FAULT_RANK}\" STATIC_FAULT_NODE=\"${STATIC_FAULT_NODE}\" STATIC_FAULT_SLEEP_SECONDS=\"${STATIC_FAULT_SLEEP_SECONDS}\""
STATIC_CMD="${STATIC_CMD:-cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && ${STATIC_FAULT_ENV} OUT_DIR=\"\$HC_POD_RESULT_DIR\" STATIC_OUTPUT_MODE=\"${STATIC_OUTPUT_MODE}\" STATIC_TMP_ROOT=\"${STATIC_TMP_ROOT}\" STATIC_KEEP_LOCAL_TMP=\"${STATIC_KEEP_LOCAL_TMP}\" STATIC_COPY_RAW_OUTPUT=\"${STATIC_COPY_RAW_OUTPUT}\" STATIC_STDOUT_MAX_BYTES=\"${STATIC_STDOUT_MAX_BYTES}\" bash scripts/ascend/probe_pod_capabilities.sh}"
FAULT_ENV="FAULT_BACKEND=\"${FAULT_BACKEND}\" FAULT_SLEEP_RANK=\"${FAULT_SLEEP_RANK}\" FAULT_SLEEP_POD=\"${FAULT_SLEEP_POD}\" FAULT_SLEEP_NODE=\"${FAULT_SLEEP_NODE}\" FAULT_SLEEP_SECONDS=\"${FAULT_SLEEP_SECONDS}\" FAULT_NAN_RANK=\"${FAULT_NAN_RANK}\" FAULT_NAN_POD=\"${FAULT_NAN_POD}\" FAULT_NAN_NODE=\"${FAULT_NAN_NODE}\" FAULT_CORRUPT_RANK=\"${FAULT_CORRUPT_RANK}\" FAULT_CORRUPT_POD=\"${FAULT_CORRUPT_POD}\" FAULT_CORRUPT_NODE=\"${FAULT_CORRUPT_NODE}\" FAULT_JOIN_TIMEOUT_RANK=\"${FAULT_JOIN_TIMEOUT_RANK}\" FAULT_JOIN_TIMEOUT_POD=\"${FAULT_JOIN_TIMEOUT_POD}\" FAULT_JOIN_TIMEOUT_NODE=\"${FAULT_JOIN_TIMEOUT_NODE}\" FAULT_JOIN_TIMEOUT_SECONDS=\"${FAULT_JOIN_TIMEOUT_SECONDS}\" FAULT_NET_SLOW_RANK=\"${FAULT_NET_SLOW_RANK}\" FAULT_NET_SLOW_POD=\"${FAULT_NET_SLOW_POD}\" FAULT_NET_SLOW_NODE=\"${FAULT_NET_SLOW_NODE}\" FAULT_NET_SLOW_SECONDS=\"${FAULT_NET_SLOW_SECONDS}\" FAULT_RANK_EXIT_RANK=\"${FAULT_RANK_EXIT_RANK}\" FAULT_RANK_EXIT_POD=\"${FAULT_RANK_EXIT_POD}\" FAULT_RANK_EXIT_NODE=\"${FAULT_RANK_EXIT_NODE}\" FAULT_RANK_EXIT_CODE=\"${FAULT_RANK_EXIT_CODE}\" FAULT_COMM_ENV_BAD_RANK=\"${FAULT_COMM_ENV_BAD_RANK}\" FAULT_COMM_ENV_BAD_POD=\"${FAULT_COMM_ENV_BAD_POD}\" FAULT_COMM_ENV_BAD_NODE=\"${FAULT_COMM_ENV_BAD_NODE}\" FAULT_ETH_FALLBACK_RANK=\"${FAULT_ETH_FALLBACK_RANK}\" FAULT_ETH_FALLBACK_POD=\"${FAULT_ETH_FALLBACK_POD}\" FAULT_ETH_FALLBACK_NODE=\"${FAULT_ETH_FALLBACK_NODE}\" DYNAMIC_FAULT_TYPE=\"${DYNAMIC_FAULT_TYPE}\" DYNAMIC_FAULT_POD=\"${DYNAMIC_FAULT_POD}\" DYNAMIC_FAULT_NODE=\"${DYNAMIC_FAULT_NODE}\" DYNAMIC_FAULT_LOCAL_RANK=\"${DYNAMIC_FAULT_LOCAL_RANK}\" DYNAMIC_FAULT_SLEEP_SECONDS=\"${DYNAMIC_FAULT_SLEEP_SECONDS}\" DYNAMIC_FAULT_FRAME_BYTES=\"${DYNAMIC_FAULT_FRAME_BYTES}\" COMM_PATH_DEBUG=\"${COMM_PATH_DEBUG}\""
FAULT_ENV+=" FAULT_SLEEP_PODS=\"${FAULT_SLEEP_PODS}\" FAULT_SLEEP_NODES=\"${FAULT_SLEEP_NODES}\" FAULT_NAN_PODS=\"${FAULT_NAN_PODS}\" FAULT_NAN_NODES=\"${FAULT_NAN_NODES}\" FAULT_CORRUPT_PODS=\"${FAULT_CORRUPT_PODS}\" FAULT_CORRUPT_NODES=\"${FAULT_CORRUPT_NODES}\" FAULT_JOIN_TIMEOUT_PODS=\"${FAULT_JOIN_TIMEOUT_PODS}\" FAULT_JOIN_TIMEOUT_NODES=\"${FAULT_JOIN_TIMEOUT_NODES}\" FAULT_NET_SLOW_PODS=\"${FAULT_NET_SLOW_PODS}\" FAULT_NET_SLOW_NODES=\"${FAULT_NET_SLOW_NODES}\" FAULT_RANK_EXIT_PODS=\"${FAULT_RANK_EXIT_PODS}\" FAULT_RANK_EXIT_NODES=\"${FAULT_RANK_EXIT_NODES}\" FAULT_COMM_ENV_BAD_PODS=\"${FAULT_COMM_ENV_BAD_PODS}\" FAULT_COMM_ENV_BAD_NODES=\"${FAULT_COMM_ENV_BAD_NODES}\" FAULT_ETH_FALLBACK_PODS=\"${FAULT_ETH_FALLBACK_PODS}\" FAULT_ETH_FALLBACK_NODES=\"${FAULT_ETH_FALLBACK_NODES}\""
FAULT_ENV+=" DYNAMIC_COMPARE_MEASUREMENT_BATCHES=\"${DYNAMIC_COMPARE_MEASUREMENT_BATCHES}\""
FAULT_ENV+=" TOPOLOGY_OVERLAP_CANARY=\"${TOPOLOGY_OVERLAP_CANARY}\""
if [[ -z "${SINGLE_NODE_CMD:-}" ]]; then
  if [[ "${PROFILE}" =~ ^(smoke|quick|bandwidth|collective-bandwidth|dynamic-suite)$ ]]; then
    SINGLE_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PROJECT_REMOTE_DIR=\"${PROJECT_REMOTE_DIR}\" PYTHONPATH=\"${PROJECT_REMOTE_DIR}:\${PYTHONPATH:-}\" && STAGE_KIND=\"${PROFILE}\" GPUS_PER_NODE=\"${GPUS_PER_NODE}\" DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" DTYPE=\"${DTYPE}\" MESSAGE_SIZES=\"${MESSAGE_SIZES}\" MOE_PATTERNS=\"${MOE_PATTERNS}\" WARMUP=\"${WARMUP}\" ITERS=\"${ITERS}\" BANDWIDTH_MESSAGE_SIZES=\"${BANDWIDTH_MESSAGE_SIZES}\" BANDWIDTH_WARMUP=\"${BANDWIDTH_WARMUP}\" BANDWIDTH_ITERS=\"${BANDWIDTH_ITERS}\" BANDWIDTH_MIN_BUSBW=\"${BANDWIDTH_MIN_BUSBW}\" BANDWIDTH_AVG_BUSBW=\"${BANDWIDTH_AVG_BUSBW}\" COLLECTIVE_BANDWIDTH_OPS=\"${COLLECTIVE_BANDWIDTH_OPS}\" COLLECTIVE_BANDWIDTH_MESSAGE_SIZES=\"${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES}\" COLLECTIVE_BANDWIDTH_MOE_PATTERNS=\"${COLLECTIVE_BANDWIDTH_MOE_PATTERNS}\" COLLECTIVE_BANDWIDTH_EP_SIZE=\"${COLLECTIVE_BANDWIDTH_EP_SIZE}\" COLLECTIVE_BANDWIDTH_WARMUP=\"${COLLECTIVE_BANDWIDTH_WARMUP}\" COLLECTIVE_BANDWIDTH_ITERS=\"${COLLECTIVE_BANDWIDTH_ITERS}\" COLLECTIVE_BANDWIDTH_MIN_BUSBW=\"${COLLECTIVE_BANDWIDTH_MIN_BUSBW}\" COLLECTIVE_BANDWIDTH_AVG_BUSBW=\"${COLLECTIVE_BANDWIDTH_AVG_BUSBW}\" ${FAULT_ENV} bash scripts/ascend/run_single_node_dynamic_stage.sh"
  else
    SINGLE_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} torchrun --standalone --nproc-per-node=\"${GPUS_PER_NODE}\" -m pretrain_healthcheck.cli run-single-node --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${MESSAGE_SIZES}\" --moe-patterns \"${MOE_PATTERNS}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --seed \"${SEED}\"; rc=\$?; python3 -m pretrain_healthcheck.cli analyze --input-dir \"\$HC_POD_RESULT_DIR\" --output \"\$HC_POD_RESULT_DIR/report.md\"; exit \$rc"
  fi
fi
if [[ -z "${MULTI_NODE_CMD:-}" ]]; then
  DYNAMIC_SUITE_COMPACT='rc=$?; python3 tools/dynamic_compact.py --input-dir "$HC_POD_RESULT_DIR" --kind dynamic-suite --stage "$HC_RUN_STAGE" --returncode "$rc" --run-id "$HC_RUN_ID" --pod-name "$HC_POD_NAME" --node-name "$HC_NODE_NAME" --pod-ip "$HC_POD_IP" --host-ip "$HC_HOST_IP" --frame-output "$HC_POD_RESULT_DIR/.hc_dynamic_result.v2"; exit "$rc"'
  BANDWIDTH_COMPACT='rc=$?; python3 tools/dynamic_compact.py --input-dir "$HC_POD_RESULT_DIR" --kind bandwidth --stage "$HC_RUN_STAGE" --returncode "$rc" --run-id "$HC_RUN_ID" --pod-name "$HC_POD_NAME" --node-name "$HC_NODE_NAME" --pod-ip "$HC_POD_IP" --host-ip "$HC_HOST_IP" --frame-output "$HC_POD_RESULT_DIR/.hc_dynamic_result.v2"; exit "$rc"'
  COLLECTIVE_COMPACT='rc=$?; python3 tools/dynamic_compact.py --input-dir "$HC_POD_RESULT_DIR" --kind collective-bandwidth --stage "$HC_RUN_STAGE" --returncode "$rc" --run-id "$HC_RUN_ID" --pod-name "$HC_POD_NAME" --node-name "$HC_NODE_NAME" --pod-ip "$HC_POD_IP" --host-ip "$HC_HOST_IP" --frame-output "$HC_POD_RESULT_DIR/.hc_dynamic_result.v2"; exit "$rc"'
  if [[ "${PROFILE}" == "smoke" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"1\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli ping-group --output-dir \"\$HC_POD_RESULT_DIR\" --test-round smoke --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\""
  elif [[ "${PROFILE}" == "bandwidth" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-bandwidth --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${BANDWIDTH_MESSAGE_SIZES}\" --warmup \"${BANDWIDTH_WARMUP}\" --iters \"${BANDWIDTH_ITERS}\" --seed \"${SEED}\" --min-busbw \"${BANDWIDTH_MIN_BUSBW}\" --avg-busbw \"${BANDWIDTH_AVG_BUSBW}\" --test-round bandwidth --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; ${BANDWIDTH_COMPACT}"
  elif [[ "${PROFILE}" == "collective-bandwidth" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-collective-bandwidth --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES}\" --ops \"${COLLECTIVE_BANDWIDTH_OPS}\" --moe-patterns \"${COLLECTIVE_BANDWIDTH_MOE_PATTERNS}\" --ep-size \"${COLLECTIVE_BANDWIDTH_EP_SIZE}\" --warmup \"${COLLECTIVE_BANDWIDTH_WARMUP}\" --iters \"${COLLECTIVE_BANDWIDTH_ITERS}\" --seed \"${SEED}\" --min-busbw \"${COLLECTIVE_BANDWIDTH_MIN_BUSBW}\" --avg-busbw \"${COLLECTIVE_BANDWIDTH_AVG_BUSBW}\" --test-round collective_bandwidth --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; ${COLLECTIVE_COMPACT}"
  elif [[ "${PROFILE}" == "training-topology" ]]; then
    if [[ -z "${POD_TRAINING_TOPOLOGY_MANIFEST}" ]]; then
      echo "[vcctl-healthcheck] training-topology requires POD_TRAINING_TOPOLOGY_MANIFEST" >&2
      exit 2
    fi
    TOPOLOGY_COMPACT='rc=$?; python3 tools/dynamic_compact.py --input-dir "$HC_POD_RESULT_DIR" --kind training-topology --stage "$HC_RUN_STAGE" --returncode "$rc" --run-id "$HC_RUN_ID" --pod-name "$HC_POD_NAME" --node-name "$HC_NODE_NAME" --pod-ip "$HC_POD_IP" --host-ip "$HC_HOST_IP" --frame-output "$HC_POD_RESULT_DIR/.hc_dynamic_result.v2"; exit "$rc"'
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" TOPOLOGY_DTYPE=\"${DTYPE}\" TOPOLOGY_MANIFEST_SHA256=\"${TOPOLOGY_MANIFEST_SHA256}\" TOPOLOGY_RETEST_PLAN_B64=\"${TOPOLOGY_RETEST_PLAN_B64:-}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-training-topology --output-dir \"\$HC_POD_RESULT_DIR\" --manifest \"${POD_TRAINING_TOPOLOGY_MANIFEST}\" --ranks-per-node \"${GPUS_PER_NODE}\" --dtype \"${DTYPE}\" --warmup \"${TOPOLOGY_WARMUP}\" --iters \"${TOPOLOGY_ITERS}\" --seed \"${SEED}\" --test-round training_topology --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; ${TOPOLOGY_COMPACT}"
  elif [[ "${PROFILE}" == "dynamic-suite" ]]; then
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-dynamic-suite --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${MESSAGE_SIZES}\" --moe-patterns \"${MOE_PATTERNS}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --bandwidth-message-sizes \"${BANDWIDTH_MESSAGE_SIZES}\" --bandwidth-warmup \"${BANDWIDTH_WARMUP}\" --bandwidth-iters \"${BANDWIDTH_ITERS}\" --bandwidth-min-busbw \"${BANDWIDTH_MIN_BUSBW}\" --bandwidth-avg-busbw \"${BANDWIDTH_AVG_BUSBW}\" --collective-bandwidth-message-sizes \"${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES}\" --collective-bandwidth-ops \"${COLLECTIVE_BANDWIDTH_OPS}\" --collective-bandwidth-moe-patterns \"${COLLECTIVE_BANDWIDTH_MOE_PATTERNS}\" --collective-bandwidth-ep-size \"${COLLECTIVE_BANDWIDTH_EP_SIZE}\" --collective-bandwidth-warmup \"${COLLECTIVE_BANDWIDTH_WARMUP}\" --collective-bandwidth-iters \"${COLLECTIVE_BANDWIDTH_ITERS}\" --collective-bandwidth-min-busbw \"${COLLECTIVE_BANDWIDTH_MIN_BUSBW}\" --collective-bandwidth-avg-busbw \"${COLLECTIVE_BANDWIDTH_AVG_BUSBW}\" --seed \"${SEED}\" --test-round dynamic_suite --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; ${DYNAMIC_SUITE_COMPACT}"
  else
    MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${FAULT_ENV} HEALTHCHECK_GROUP_ID=\"\$HC_JOB_NAME-\$HC_RUN_ID\" torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-group --output-dir \"\$HC_POD_RESULT_DIR\" --dtype \"${DTYPE}\" --message-sizes \"${MESSAGE_SIZES}\" --moe-patterns \"${MOE_PATTERNS}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --seed \"${SEED}\" --test-round current_vcjob --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID\"; rc=\$?; if [ \"\$RANK\" = \"0\" ]; then python3 -m pretrain_healthcheck.cli analyze --input-dir \"\$HC_POD_RESULT_DIR\" --output \"\$HC_POD_RESULT_DIR/report.md\"; fi; exit \$rc"
  fi
fi
DYNAMIC_RETEST_FAULT_ENV="${FAULT_ENV} DYNAMIC_COMPARE_MEASUREMENT_BATCHES=\"${DYNAMIC_COMPARE_EFFECTIVE_RETEST_MEASUREMENT_BATCHES}\""
DYNAMIC_RETEST_COMPACT='rc=$?; python3 tools/dynamic_compact.py --input-dir "$HC_POD_RESULT_DIR/retest" --kind collective-bandwidth --stage "$HC_RUN_STAGE" --returncode "$rc" --run-id "$HC_RUN_ID" --pod-name "$HC_POD_NAME" --node-name "$HC_NODE_NAME" --pod-ip "$HC_POD_IP" --host-ip "$HC_HOST_IP" --frame-output "$HC_POD_RESULT_DIR/retest/.hc_dynamic_result.v2"; exit "$rc"'
DYNAMIC_RETEST_SINGLE_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${DYNAMIC_RETEST_FAULT_ENV} torchrun --standalone --nproc-per-node=\"${GPUS_PER_NODE}\" -m pretrain_healthcheck.cli run-case-retest --output-dir \"\$HC_POD_RESULT_DIR/retest\" --plan-b64 \"${DYNAMIC_RETEST_ONLY_PLAN_B64:-\$HC_DYNAMIC_RETEST_PLAN_B64}\" --dtype \"${DTYPE}\" --ep-size \"${COLLECTIVE_BANDWIDTH_EP_SIZE}\" --warmup \"${COLLECTIVE_BANDWIDTH_WARMUP}\" --iters \"${COLLECTIVE_BANDWIDTH_ITERS}\" --diagnostic-iters 3 --seed \"${SEED}\" --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID-retest\"; ${DYNAMIC_RETEST_COMPACT}"
DYNAMIC_RETEST_MULTI_NODE_CMD="cd ${PROJECT_REMOTE_DIR} && ${ASCEND_ENV_CMD} && export PYTHONPATH=\"${PROJECT_REMOTE_DIR}\" && DIST_BACKEND=\"${DIST_BACKEND}\" DEVICE_VENDOR=\"${DEVICE_VENDOR}\" COMM_RUNTIME=\"${COMM_RUNTIME}\" ${DYNAMIC_RETEST_FAULT_ENV} torchrun --nnodes=\"\$WORLD_SIZE\" --nproc-per-node=\"${GPUS_PER_NODE}\" --node-rank=\"\$RANK\" --master-addr=\"\$MASTER_ADDR\" --master-port=\"__HC_MASTER_PORT__\" -m pretrain_healthcheck.cli run-case-retest --output-dir \"\$HC_POD_RESULT_DIR/retest\" --plan-b64 \"${DYNAMIC_RETEST_ONLY_PLAN_B64:-\$HC_DYNAMIC_RETEST_PLAN_B64}\" --dtype \"${DTYPE}\" --ep-size \"${COLLECTIVE_BANDWIDTH_EP_SIZE}\" --warmup \"${COLLECTIVE_BANDWIDTH_WARMUP}\" --iters \"${COLLECTIVE_BANDWIDTH_ITERS}\" --diagnostic-iters 3 --seed \"${SEED}\" --group-id \"\$HC_JOB_NAME-\$HC_RUN_ID-retest\"; ${DYNAMIC_RETEST_COMPACT}"
if [[ -n "${DYNAMIC_RETEST_ONLY_PLAN_B64:-}" ]]; then
  MULTI_NODE_CMD="${DYNAMIC_RETEST_MULTI_NODE_CMD}"
  DYNAMIC_COMPARE_AUTO_RETEST=0
fi
if [[ -z "${EXEC_TIMEOUT_SECONDS:-}" && "${PROFILE}" == "collective-bandwidth" && "${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES}" == "${COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES}" ]]; then
  EXEC_TIMEOUT_SECONDS="1800"
fi
EXEC_TIMEOUT_SECONDS="${EXEC_TIMEOUT_SECONDS:-180}"
STATIC_EXEC_TIMEOUT_SECONDS="${STATIC_EXEC_TIMEOUT_SECONDS:-180}"
STATIC_DRIVER_TMP_ROOT="${STATIC_DRIVER_TMP_ROOT:-/tmp}"
VCCTL_TIMEOUT_SECONDS="${VCCTL_TIMEOUT_SECONDS:-120}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
STATIC_COMPARE="${STATIC_COMPARE:-1}"
STATIC_COMPARE_WORKERS="${STATIC_COMPARE_WORKERS:-0}"
STATIC_COMPARE_STRICT="${STATIC_COMPARE_STRICT:-1}"
STATIC_EXPECTED_GPUS="${STATIC_EXPECTED_GPUS:-${GPUS_PER_NODE}}"
STATIC_EXPECTED_XSCALE_PORTS="${STATIC_EXPECTED_XSCALE_PORTS:-0}"
STATIC_ECC_POLICY="${STATIC_ECC_POLICY:-alert}"
if [[ "${STATIC_ECC_POLICY}" != "alert" && "${STATIC_ECC_POLICY}" != "strict" ]]; then
  echo "[vcctl-healthcheck] STATIC_ECC_POLICY must be alert or strict, got: ${STATIC_ECC_POLICY}" >&2
  exit 2
fi
STATIC_KEEP_POD_FILES="${STATIC_KEEP_POD_FILES:-0}"
STATIC_KEEP_EXEC_LOGS="${STATIC_KEEP_EXEC_LOGS:-0}"
STATIC_FAILED_LOG_MODE="${STATIC_FAILED_LOG_MODE:-local-link}"
DYNAMIC_COMPARE="${DYNAMIC_COMPARE:-0}"
DYNAMIC_COMPARE_STRICT="${DYNAMIC_COMPARE_STRICT:-1}"
DYNAMIC_COMPARE_RATIO_THRESHOLD="${DYNAMIC_COMPARE_RATIO_THRESHOLD:-0.7}"
DYNAMIC_KEEP_EXEC_LOGS="${DYNAMIC_KEEP_EXEC_LOGS:-0}"
DYNAMIC_EXEC_LOG_ROOT="${DYNAMIC_EXEC_LOG_ROOT:-/tmp/pretrain_healthcheck_exec_logs/vcctl}"
DYNAMIC_FAILED_LOG_MODE="${DYNAMIC_FAILED_LOG_MODE:-local-link}"
DYNAMIC_FRAME_RECOVERY_DEADLINE_SECONDS="${DYNAMIC_FRAME_RECOVERY_DEADLINE_SECONDS:-60}"
DYNAMIC_FRAME_CHUNK_SIZE="${DYNAMIC_FRAME_CHUNK_SIZE:-2048}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/vcctl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
POD_RESULT_ROOT="${POD_RESULT_ROOT:-/tmp/pretrain_healthcheck_driver_${RUN_ID}}"
RUN_STAGE="${RUN_STAGE:-}"
DRY_RUN="${DRY_RUN:-1}"
POD_JSON_FILE="${POD_JSON_FILE:-}"

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<vcjob-name> [env ...] bash scripts/ascend/run_vcctl_healthcheck.sh

Required:
  JOB_NAME                 vcjob name used by "vcctl pod get --job". Default: grj-megatron-128-235b-moe0708

Common env:
  NAMESPACE                Kubernetes namespace. Default: default
  DRIVER_PYTHON            Developer-machine Python >=3.9. Default: auto-discover
  MODE                     static|single-node|multi-node|all. Default: all
  DEVICE_TYPE              Metadata only, for example gpu, npu, ascend. Default: ascend
  PROJECT_REMOTE_DIR       Project path inside target pods. Default: <project>.
  PROFILE                  quick|smoke|bandwidth|collective-bandwidth|dynamic-suite|training-topology. Default: quick
  PRE_CLEAN                1 runs cleanup before checks; 0 disables it. Default: 1
  PRE_CLEAN_CMD            Command used for cleanup. Default: pkill healthcheck torchrun/python.
  GPUS_PER_NODE            Local device count per pod. Default: 16
  DIST_BACKEND             PyTorch distributed backend name. Default: hccl
  HEALTHCHECK_MASTER_PORT  auto or explicit rendezvous port. Default: auto
  DEVICE_VENDOR            Result metadata. Default: ascend
  COMM_RUNTIME             Result metadata. Default: hccl
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
                           EP group size for all_to_allv. Default: 16
  FAULT_*                  Optional fault injection envs for backend/sleep/nan/corrupt tests.
  STATIC_CMD               Command executed in each pod for static checks. Default: Ascend static probe
  STATIC_OUTPUT_MODE       compact keeps only compact pod facts before run-level aggregation. Default: compact
  STATIC_TMP_ROOT          Pod-local temp root for static raw work files. Default: /tmp
  STATIC_KEEP_LOCAL_TMP    1 keeps pod-local static temp files. Default: 1
  STATIC_COPY_RAW_OUTPUT   1 copies raw static temp files to shared result dir. Default: 0
  STATIC_STDOUT_MAX_BYTES  Max bytes for one static stdout result frame. Default: 1048576
  SINGLE_NODE_CMD          Command executed in each pod for single-node checks. Default: Ascend 8-card torchrun
  MULTI_NODE_CMD           Command executed in every pod concurrently for multi-node checks. Default: Ascend current vcjob torchrun
  RESULT_ROOT              Shared result root. Default: <project>/results/vcctl
  POD_RESULT_ROOT          Temporary pod result root. Default: /tmp/pretrain_healthcheck_driver_<RUN_ID>
  RUN_ID                   Run id. Default: current timestamp
  RUN_STAGE                Override result stage directory. Default: derived from MODE
  EXEC_TIMEOUT_SECONDS     Per-pod exec timeout. Default: 180
  STATIC_EXEC_TIMEOUT_SECONDS
                           Per-pod static exec timeout. Default: 180
  STATIC_DRIVER_TMP_ROOT   Dev-machine local temp root for static exec stdout/stderr. Default: /tmp
  MAX_PARALLEL             0 means all pods concurrently. Default: 0
  STATIC_COMPARE           1 compares static outputs across pods. Default: 1
  STATIC_COMPARE_WORKERS   Parallel static parser workers; 0 means auto. Default: 0
  STATIC_COMPARE_STRICT    1 lets static outliers affect overall status. Default: 1
  STATIC_EXPECTED_GPUS     Expected visible accelerator count per pod. Default: GPUS_PER_NODE
  STATIC_EXPECTED_XSCALE_PORTS
                           Expected xscale/HCA port count per pod; 0 disables this gate. Default: 0
  STATIC_ECC_POLICY        alert keeps cumulative ECC counters as warnings; strict restores legacy gates. Default: alert
  STATIC_KEEP_POD_FILES    1 keeps pod_results/<pod>/static after aggregation. Default: 0
  STATIC_KEEP_EXEC_LOGS    1 keeps logs/<pod>.static.stdout/stderr after aggregation. Default: 0
  STATIC_FAILED_LOG_MODE   local-link keeps failed static exec logs under STATIC_DRIVER_TMP_ROOT and links them from shared results; shared restores old behavior. Default: local-link
  DYNAMIC_EXEC_LOG_ROOT    Dev-machine local root for dynamic exec stdout/stderr. Default: /tmp/pretrain_healthcheck_exec_logs/vcctl
  DYNAMIC_FAILED_LOG_MODE  local-link keeps failed dynamic exec logs under DYNAMIC_EXEC_LOG_ROOT and links them from shared results; shared restores old behavior. Default: local-link
  DRY_RUN                  1 prints generated vcctl exec commands without executing them. Default: 1
  POD_JSON_FILE            Optional fixture file instead of calling vcctl pod get.

Example:
  bash scripts/ascend/run_vcctl_healthcheck.sh

  DRY_RUN=0 bash scripts/ascend/run_vcctl_healthcheck.sh

  MODE=multi-node \
  DRY_RUN=1 \
  MULTI_NODE_CMD='cd /path/to/healthcheck && torchrun --nnodes=${WORLD_SIZE} --nproc-per-node=8 --node-rank=${RANK} --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} -m pretrain_healthcheck.cli run-group --output-dir ${HC_POD_RESULT_DIR}' \
  bash scripts/ascend/run_vcctl_healthcheck.sh
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

resolve_driver_python
print_driver_python

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
  local dynamic_retest_cmd="${DYNAMIC_RETEST_SINGLE_NODE_CMD}"
  if [[ "${stage_mode}" == "multi-node" ]]; then
    dynamic_retest_cmd="${DYNAMIC_RETEST_MULTI_NODE_CMD}"
  fi

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
    --static-ecc-policy "${STATIC_ECC_POLICY}"
    --static-failed-log-mode "${STATIC_FAILED_LOG_MODE}"
    --dynamic-compare-ratio-threshold "${DYNAMIC_COMPARE_RATIO_THRESHOLD}"
    --dynamic-compare-measurement-batches "${DYNAMIC_COMPARE_MEASUREMENT_BATCHES}"
    --dynamic-compare-retest-measurement-batches "${DYNAMIC_COMPARE_EFFECTIVE_RETEST_MEASUREMENT_BATCHES}"
    --dynamic-compare-busbw-ratio-threshold "${DYNAMIC_COMPARE_BUSBW_RATIO_THRESHOLD}"
    --dynamic-compare-latency-ratio-threshold "${DYNAMIC_COMPARE_LATENCY_RATIO_THRESHOLD}"
    --dynamic-compare-small-max-size "${DYNAMIC_COMPARE_SMALL_MAX_SIZE}"
    --dynamic-compare-large-min-size "${DYNAMIC_COMPARE_LARGE_MIN_SIZE}"
    --dynamic-compare-small-latency-abs-delta-ms "${DYNAMIC_COMPARE_SMALL_LATENCY_ABS_DELTA_MS}"
    --dynamic-compare-small-latency-mad-multiplier "${DYNAMIC_COMPARE_SMALL_LATENCY_MAD_MULTIPLIER}"
    --dynamic-compare-min-cohort "${DYNAMIC_COMPARE_MIN_COHORT}"
    --dynamic-retest-cmd "${dynamic_retest_cmd}"
    --dynamic-exec-log-root "${DYNAMIC_EXEC_LOG_ROOT}"
    --dynamic-failed-log-mode "${DYNAMIC_FAILED_LOG_MODE}"
    --dynamic-frame-recovery-deadline-seconds "${DYNAMIC_FRAME_RECOVERY_DEADLINE_SECONDS}"
    --dynamic-frame-chunk-size "${DYNAMIC_FRAME_CHUNK_SIZE}"
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

  if [[ "${DYNAMIC_COMPARE_AUTO_RETEST}" == "0" || "${DYNAMIC_COMPARE_AUTO_RETEST}" == "false" ]]; then
    args+=(--no-dynamic-compare-auto-retest)
  else
    args+=(--dynamic-compare-auto-retest)
  fi

  if [[ "${DYNAMIC_COMPARE_SMALL_LATENCY_WARN}" == "1" || "${DYNAMIC_COMPARE_SMALL_LATENCY_WARN}" == "true" ]]; then
    args+=(--dynamic-compare-small-latency-warn)
  else
    args+=(--no-dynamic-compare-small-latency-warn)
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

  "${DRIVER_PYTHON}" "${args[@]}"
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
