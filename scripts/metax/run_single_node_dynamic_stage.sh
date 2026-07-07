#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_REMOTE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STAGE_KIND="${STAGE_KIND:-quick}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
DEVICE_VENDOR="${DEVICE_VENDOR:-metax}"
COMM_RUNTIME="${COMM_RUNTIME:-mccl}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-20260623}"
STATIC_TMP_ROOT="${STATIC_TMP_ROOT:-/tmp}"
COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES="1K,2K,4K,8K,16K,32K,64K,128K,256K,512K,1M,2M,4M,8M,16M,32M,64M,128M,256M,512M,1G,2G"
COLLECTIVE_ACCEPTANCE_OPS="all_reduce,reduce_scatter,all_gather,broadcast,all_to_all,all_to_allv"

safe_run_id="$(printf "%s" "${HC_RUN_ID:-$(date +%Y%m%d_%H%M%S)}" | tr -c 'A-Za-z0-9_.-' '_')"
safe_pod_name="$(printf "%s" "${HC_POD_NAME:-pod}" | tr -c 'A-Za-z0-9_.-' '_')"
safe_stage="$(printf "%s" "${HC_RUN_STAGE:-single_node}" | tr -c 'A-Za-z0-9_.-' '_')"
WORK_ROOT="${DYNAMIC_WORK_ROOT:-${STATIC_TMP_ROOT%/}/pretrain_healthcheck_${safe_run_id}_${safe_pod_name}_$$}"
WORK_DIR="${DYNAMIC_WORK_DIR:-${WORK_ROOT}/${safe_stage}}"
mkdir -p "${WORK_DIR}"

echo "[dynamic-stage] workroot: ${WORK_ROOT}" >&2
echo "[dynamic-stage] workdir: ${WORK_DIR}" >&2
echo "[dynamic-stage] kind: ${STAGE_KIND}" >&2

cd "${PROJECT_DIR}" || exit 1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export DIST_BACKEND DEVICE_VENDOR COMM_RUNTIME

rc=0
case "${STAGE_KIND}" in
  smoke)
    torchrun --standalone --nproc-per-node="${GPUS_PER_NODE}" \
      -m pretrain_healthcheck.cli ping-group \
      --output-dir "${WORK_DIR}" \
      --test-round "${HC_RUN_STAGE:-single_node_smoke}" \
      --group-id "${HC_JOB_NAME:-job}-${HC_RUN_ID:-run}" || rc=$?
    ;;
  quick)
    torchrun --standalone --nproc-per-node="${GPUS_PER_NODE}" \
      -m pretrain_healthcheck.cli run-single-node \
      --output-dir "${WORK_DIR}" \
      --dtype "${DTYPE}" \
      --message-sizes "${MESSAGE_SIZES:-1M}" \
      --moe-patterns "${MOE_PATTERNS:-uniform,skewed,hot_expert,random,empty_expert}" \
      --warmup "${WARMUP:-1}" \
      --iters "${ITERS:-1}" \
      --seed "${SEED}" || rc=$?
    python3 -m pretrain_healthcheck.cli analyze \
      --input-dir "${WORK_DIR}" \
      --output "${WORK_DIR}/report.md" >/dev/null 2>&1 || true
    ;;
  bandwidth)
    torchrun --standalone --nproc-per-node="${GPUS_PER_NODE}" \
      -m pretrain_healthcheck.cli run-bandwidth \
      --output-dir "${WORK_DIR}" \
      --dtype "${DTYPE}" \
      --message-sizes "${BANDWIDTH_MESSAGE_SIZES:-1G}" \
      --warmup "${BANDWIDTH_WARMUP:-1}" \
      --iters "${BANDWIDTH_ITERS:-3}" \
      --seed "${SEED}" \
      --min-busbw "${BANDWIDTH_MIN_BUSBW:-0}" \
      --avg-busbw "${BANDWIDTH_AVG_BUSBW:-0}" \
      --test-round "${HC_RUN_STAGE:-single_node_bandwidth}" \
      --group-id "${HC_JOB_NAME:-job}-${HC_RUN_ID:-run}" || rc=$?
    ;;
  collective-bandwidth)
    torchrun --standalone --nproc-per-node="${GPUS_PER_NODE}" \
      -m pretrain_healthcheck.cli run-collective-bandwidth \
      --output-dir "${WORK_DIR}" \
      --dtype "${DTYPE}" \
      --message-sizes "${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES:-${COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES}}" \
      --ops "${COLLECTIVE_BANDWIDTH_OPS:-${COLLECTIVE_ACCEPTANCE_OPS}}" \
      --moe-patterns "${COLLECTIVE_BANDWIDTH_MOE_PATTERNS:-${MOE_PATTERNS:-uniform,skewed,hot_expert,random,empty_expert}}" \
      --ep-size "${COLLECTIVE_BANDWIDTH_EP_SIZE:-8}" \
      --warmup "${COLLECTIVE_BANDWIDTH_WARMUP:-1}" \
      --iters "${COLLECTIVE_BANDWIDTH_ITERS:-3}" \
      --seed "${SEED}" \
      --min-busbw "${COLLECTIVE_BANDWIDTH_MIN_BUSBW:-0}" \
      --avg-busbw "${COLLECTIVE_BANDWIDTH_AVG_BUSBW:-0}" \
      --test-round "${HC_RUN_STAGE:-single_node_collective_bandwidth}" \
      --group-id "${HC_JOB_NAME:-job}-${HC_RUN_ID:-run}" || rc=$?
    ;;
  dynamic-suite)
    torchrun --standalone --nproc-per-node="${GPUS_PER_NODE}" \
      -m pretrain_healthcheck.cli run-dynamic-suite \
      --output-dir "${WORK_DIR}" \
      --dtype "${DTYPE}" \
      --message-sizes "${MESSAGE_SIZES:-1M}" \
      --moe-patterns "${MOE_PATTERNS:-uniform,skewed,hot_expert,random,empty_expert}" \
      --warmup "${WARMUP:-1}" \
      --iters "${ITERS:-1}" \
      --bandwidth-message-sizes "${BANDWIDTH_MESSAGE_SIZES:-1G}" \
      --bandwidth-warmup "${BANDWIDTH_WARMUP:-1}" \
      --bandwidth-iters "${BANDWIDTH_ITERS:-3}" \
      --bandwidth-min-busbw "${BANDWIDTH_MIN_BUSBW:-0}" \
      --bandwidth-avg-busbw "${BANDWIDTH_AVG_BUSBW:-0}" \
      --collective-bandwidth-message-sizes "${COLLECTIVE_BANDWIDTH_MESSAGE_SIZES:-${COLLECTIVE_ACCEPTANCE_MESSAGE_SIZES}}" \
      --collective-bandwidth-ops "${COLLECTIVE_BANDWIDTH_OPS:-${COLLECTIVE_ACCEPTANCE_OPS}}" \
      --collective-bandwidth-moe-patterns "${COLLECTIVE_BANDWIDTH_MOE_PATTERNS:-${MOE_PATTERNS:-uniform,skewed,hot_expert,random,empty_expert}}" \
      --collective-bandwidth-ep-size "${COLLECTIVE_BANDWIDTH_EP_SIZE:-8}" \
      --collective-bandwidth-warmup "${COLLECTIVE_BANDWIDTH_WARMUP:-1}" \
      --collective-bandwidth-iters "${COLLECTIVE_BANDWIDTH_ITERS:-3}" \
      --collective-bandwidth-min-busbw "${COLLECTIVE_BANDWIDTH_MIN_BUSBW:-0}" \
      --collective-bandwidth-avg-busbw "${COLLECTIVE_BANDWIDTH_AVG_BUSBW:-0}" \
      --seed "${SEED}" \
      --test-round "${HC_RUN_STAGE:-dynamic_suite}" \
      --group-id "${HC_JOB_NAME:-job}-${HC_RUN_ID:-run}" || rc=$?
    ;;
  *)
    echo "[dynamic-stage] unsupported STAGE_KIND=${STAGE_KIND}" >&2
    rc=2
    ;;
esac

python3 "${PROJECT_DIR}/tools/dynamic_compact.py" \
  --input-dir "${WORK_DIR}" \
  --kind "${STAGE_KIND}" \
  --stage "${HC_RUN_STAGE:-single_node}" \
  --returncode "${rc}" \
  --run-id "${HC_RUN_ID:-}" \
  --pod-name "${HC_POD_NAME:-}" \
  --node-name "${HC_NODE_NAME:-}" \
  --pod-ip "${HC_POD_IP:-}" \
  --host-ip "${HC_HOST_IP:-}" || true

exit "${rc}"
