#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROBE="${PROBE:-${PROJECT_ROOT}/tools/tcp_bandwidth_probe.py}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-49162}"
DURATION="${DURATION:-120}"
PARALLEL="${PARALLEL:-8}"
BUFFER_SIZE="${BUFFER_SIZE:-1048576}"
DIRECTION="${DIRECTION:-huawei_to_muxi_8streams}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[tcp-bw-server] project     : ${PROJECT_ROOT}"
echo "[tcp-bw-server] probe       : ${PROBE}"
echo "[tcp-bw-server] listen      : ${HOST}:${PORT}"
echo "[tcp-bw-server] duration    : ${DURATION}s"
echo "[tcp-bw-server] parallel    : ${PARALLEL}"
echo "[tcp-bw-server] buffer_size : ${BUFFER_SIZE}"
echo "[tcp-bw-server] direction   : ${DIRECTION}"
echo

if [[ ! -f "${PROBE}" ]]; then
  echo "[tcp-bw-server] ERROR: probe not found: ${PROBE}" >&2
  exit 2
fi

echo "[tcp-bw-server] starting server; keep this terminal open until the client finishes"
exec "${PYTHON_BIN}" "${PROBE}" server \
  --host "${HOST}" \
  --port "${PORT}" \
  --duration "${DURATION}" \
  --parallel "${PARALLEL}" \
  --buffer-size "${BUFFER_SIZE}" \
  --direction "${DIRECTION}"

