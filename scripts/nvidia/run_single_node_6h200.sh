#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPUS_PER_NODE="${GPUS_PER_NODE:-6}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
DTYPE="${DTYPE:-bf16}"
MESSAGE_SIZES="${MESSAGE_SIZES:-1M,16M,64M}"
MOE_PATTERNS="${MOE_PATTERNS:-uniform,skewed,hot_expert,random,empty_expert}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-5}"
SEED="${SEED:-20260623}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${RESULT_ROOT}/${RUN_ID}"

mkdir -p "${OUT_DIR}"

echo "[pretrain-healthcheck] project: ${PROJECT_DIR}"
echo "[pretrain-healthcheck] output : ${OUT_DIR}"
echo "[pretrain-healthcheck] gpus   : ${GPUS_PER_NODE}"
echo "[pretrain-healthcheck] backend: ${DIST_BACKEND}"
echo "[pretrain-healthcheck] dtype  : ${DTYPE}"
echo "[pretrain-healthcheck] sizes  : ${MESSAGE_SIZES}"
echo "[pretrain-healthcheck] moe    : ${MOE_PATTERNS}"

cd "${PROJECT_DIR}"

echo "[pretrain-healthcheck] step 1/3 static checks"
python3 -m pretrain_healthcheck.cli static \
  --output "${OUT_DIR}/static.json" | tee "${OUT_DIR}/static.log"

echo "[pretrain-healthcheck] step 2/3 single-node dynamic checks"
DIST_BACKEND="${DIST_BACKEND}" torchrun --standalone --nproc-per-node="${GPUS_PER_NODE}" \
  -m pretrain_healthcheck.cli run-single-node \
  --output-dir "${OUT_DIR}/single_node" \
  --dtype "${DTYPE}" \
  --message-sizes "${MESSAGE_SIZES}" \
  --moe-patterns "${MOE_PATTERNS}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --seed "${SEED}" 2>&1 | tee "${OUT_DIR}/single_node.log"

echo "[pretrain-healthcheck] step 3/3 analyze"
python3 -m pretrain_healthcheck.cli analyze \
  --input-dir "${OUT_DIR}/single_node" \
  --output "${OUT_DIR}/report.md" | tee "${OUT_DIR}/analyze.log"

echo "[pretrain-healthcheck] done"
echo "[pretrain-healthcheck] report: ${OUT_DIR}/report.md"
echo "[pretrain-healthcheck] group summary: ${OUT_DIR}/single_node/group_summary.jsonl"
echo "[pretrain-healthcheck] rank detail : ${OUT_DIR}/single_node/rank_detail.jsonl"
