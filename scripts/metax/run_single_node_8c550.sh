#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
DEVICE_VENDOR="${DEVICE_VENDOR:-metax}"
COMM_RUNTIME="${COMM_RUNTIME:-mccl}"
DTYPE="${DTYPE:-bf16}"
MESSAGE_SIZES="${MESSAGE_SIZES:-1M,16M,64M}"
MOE_PATTERNS="${MOE_PATTERNS:-uniform,skewed,hot_expert,random,empty_expert}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-5}"
SEED="${SEED:-20260623}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results}"
RUN_ID="${RUN_ID:-metax_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${RESULT_ROOT}/${RUN_ID}"

mkdir -p "${OUT_DIR}"

echo "[metax-healthcheck] project      : ${PROJECT_DIR}"
echo "[metax-healthcheck] output       : ${OUT_DIR}"
echo "[metax-healthcheck] gpus         : ${GPUS_PER_NODE}"
echo "[metax-healthcheck] dist backend : ${DIST_BACKEND}"
echo "[metax-healthcheck] device vendor: ${DEVICE_VENDOR}"
echo "[metax-healthcheck] comm runtime : ${COMM_RUNTIME}"
echo "[metax-healthcheck] dtype        : ${DTYPE}"
echo "[metax-healthcheck] sizes        : ${MESSAGE_SIZES}"
echo "[metax-healthcheck] moe          : ${MOE_PATTERNS}"

cd "${PROJECT_DIR}"

echo "[metax-healthcheck] step 1/3 static checks"
OUT_DIR="${OUT_DIR}/static" bash scripts/metax/probe_pod_capabilities.sh 2>&1 | tee "${OUT_DIR}/static.log"

echo "[metax-healthcheck] step 2/3 single-node dynamic checks"
DIST_BACKEND="${DIST_BACKEND}" DEVICE_VENDOR="${DEVICE_VENDOR}" COMM_RUNTIME="${COMM_RUNTIME}" \
  torchrun --standalone --nproc-per-node="${GPUS_PER_NODE}" \
  -m pretrain_healthcheck.cli run-single-node \
  --output-dir "${OUT_DIR}/single_node" \
  --dtype "${DTYPE}" \
  --message-sizes "${MESSAGE_SIZES}" \
  --moe-patterns "${MOE_PATTERNS}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --seed "${SEED}" 2>&1 | tee "${OUT_DIR}/single_node.log"

echo "[metax-healthcheck] step 3/3 analyze"
python3 -m pretrain_healthcheck.cli analyze \
  --input-dir "${OUT_DIR}/single_node" \
  --output "${OUT_DIR}/report.md" | tee "${OUT_DIR}/analyze.log"

echo "[metax-healthcheck] done"
echo "[metax-healthcheck] report       : ${OUT_DIR}/report.md"
echo "[metax-healthcheck] static       : ${OUT_DIR}/static/summary.md"
echo "[metax-healthcheck] group summary: ${OUT_DIR}/single_node/group_summary.jsonl"
echo "[metax-healthcheck] rank detail  : ${OUT_DIR}/single_node/rank_detail.jsonl"
