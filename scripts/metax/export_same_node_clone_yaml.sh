#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JOB_NAME="${JOB_NAME:-muxi-2node1}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/job_clone}"
OUT_DIR="${OUT_DIR:-${RESULT_ROOT}/${JOB_NAME}_${RUN_ID}}"
CLONE_JOB_NAME="${CLONE_JOB_NAME:-${JOB_NAME}-clone-${RUN_ID}}"
ALLOW_EXTRA_NODE_MAP="${ALLOW_EXTRA_NODE_MAP:-0}"

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<vcjob-name> bash scripts/metax/export_same_node_clone_yaml.sh

Export the current vcctl/Volcano job, capture its pod-to-node map, and generate
a clone YAML that pins each task to the same physical node via
kubernetes.io/hostname nodeAffinity. A PyTorch worker task with replicas > 1 is
expanded into one task per original worker pod, for example worker-0, worker-1.

Common env:
  JOB_NAME              Source vcjob name. Default: muxi-2node1
  NAMESPACE             Kubernetes namespace. Default: default
  VCCTL_BIN             vcctl binary. Default: vcctl
  RUN_ID                Output run id. Default: current timestamp
  RESULT_ROOT           Output root. Default: <project>/results/job_clone
  OUT_DIR               Output dir. Default: RESULT_ROOT/JOB_NAME_RUN_ID
  CLONE_JOB_NAME        New job name in clone YAML. Default: JOB_NAME-clone-RUN_ID
  ALLOW_EXTRA_NODE_MAP  1 ignores node_map pods not represented by cloned tasks. Default: 0

Outputs:
  OUT_DIR/JOB_NAME.yaml
  OUT_DIR/JOB_NAME.json
  OUT_DIR/node_map.txt
  OUT_DIR/JOB_NAME_clone.yaml
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${JOB_NAME}" ]]; then
  echo "[same-node-clone] JOB_NAME is required" >&2
  usage >&2
  exit 2
fi

ORIGINAL_YAML="${OUT_DIR}/${JOB_NAME}.yaml"
ORIGINAL_JSON="${OUT_DIR}/${JOB_NAME}.json"
NODE_MAP="${OUT_DIR}/node_map.txt"
CLONE_YAML="${OUT_DIR}/${JOB_NAME}_clone.yaml"

mkdir -p "${OUT_DIR}"

echo "[same-node-clone] job          : ${JOB_NAME}"
echo "[same-node-clone] namespace    : ${NAMESPACE}"
echo "[same-node-clone] output dir   : ${OUT_DIR}"
echo "[same-node-clone] clone job    : ${CLONE_JOB_NAME}"

echo "[same-node-clone] exporting source YAML"
"${VCCTL_BIN}" job get "${JOB_NAME}" -n "${NAMESPACE}" -o yaml > "${ORIGINAL_YAML}"

echo "[same-node-clone] exporting source JSON"
"${VCCTL_BIN}" job get "${JOB_NAME}" -n "${NAMESPACE}" -o json > "${ORIGINAL_JSON}"

echo "[same-node-clone] exporting node map"
JOB_NAME="${JOB_NAME}" \
NAMESPACE="${NAMESPACE}" \
VCCTL_BIN="${VCCTL_BIN}" \
RESULT_FILE="${NODE_MAP}" \
bash "${SCRIPT_DIR}/print_vcctl_node_map.sh" >/dev/null

generator_args=(
  "${PROJECT_DIR}/tools/generate_same_node_clone_yaml.py"
  --job-json "${ORIGINAL_JSON}"
  --node-map "${NODE_MAP}"
  --output "${CLONE_YAML}"
  --clone-job-name "${CLONE_JOB_NAME}"
)

if [[ "${ALLOW_EXTRA_NODE_MAP}" == "1" || "${ALLOW_EXTRA_NODE_MAP}" == "true" ]]; then
  generator_args+=(--allow-extra-node-map)
fi

echo "[same-node-clone] generating clone YAML"
python3 "${generator_args[@]}"

echo "[same-node-clone] source yaml : ${ORIGINAL_YAML}"
echo "[same-node-clone] node map    : ${NODE_MAP}"
echo "[same-node-clone] clone yaml  : ${CLONE_YAML}"
