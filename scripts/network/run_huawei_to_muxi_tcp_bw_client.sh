#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROBE="${PROBE:-${PROJECT_ROOT}/tools/tcp_bandwidth_probe.py}"

HOST="${HOST:-10.140.158.130}"
PORT="${PORT:-49175}"
DURATION="${DURATION:-60}"
PARALLEL="${PARALLEL:-8}"
BUFFER_SIZE="${BUFFER_SIZE:-1048576}"
DIRECTION="${DIRECTION:-huawei_to_muxi_8streams}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[tcp-bw-client] project     : ${PROJECT_ROOT}"
echo "[tcp-bw-client] probe       : ${PROBE}"
echo "[tcp-bw-client] target      : ${HOST}:${PORT}"
echo "[tcp-bw-client] duration    : ${DURATION}s"
echo "[tcp-bw-client] parallel    : ${PARALLEL}"
echo "[tcp-bw-client] buffer_size : ${BUFFER_SIZE}"
echo "[tcp-bw-client] direction   : ${DIRECTION}"
echo

if [[ ! -f "${PROBE}" ]]; then
  echo "[tcp-bw-client] ERROR: probe not found: ${PROBE}" >&2
  exit 2
fi

exec "${PYTHON_BIN}" "${PROBE}" client \
  --host "${HOST}" \
  --port "${PORT}" \
  --duration "${DURATION}" \
  --parallel "${PARALLEL}" \
  --buffer-size "${BUFFER_SIZE}" \
  --direction "${DIRECTION}"

