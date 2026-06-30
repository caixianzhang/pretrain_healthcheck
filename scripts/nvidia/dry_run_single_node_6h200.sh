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
RUN_ID="${RUN_ID:-YYYYMMDD_HHMMSS}"
OUT_DIR="${RESULT_ROOT}/${RUN_ID}"

cat <<EOF
cd "${PROJECT_DIR}"

python3 -m pretrain_healthcheck.cli static \\
  --output "${OUT_DIR}/static.json"

DIST_BACKEND="${DIST_BACKEND}" torchrun --standalone --nproc-per-node="${GPUS_PER_NODE}" \\
  -m pretrain_healthcheck.cli run-single-node \\
  --output-dir "${OUT_DIR}/single_node" \\
  --dtype "${DTYPE}" \\
  --message-sizes "${MESSAGE_SIZES}" \\
  --moe-patterns "${MOE_PATTERNS}" \\
  --warmup "${WARMUP}" \\
  --iters "${ITERS}" \\
  --seed "${SEED}"

python3 -m pretrain_healthcheck.cli analyze \\
  --input-dir "${OUT_DIR}/single_node" \\
  --output "${OUT_DIR}/report.md"
EOF
