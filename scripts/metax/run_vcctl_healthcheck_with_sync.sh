#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LOCAL_PROJECT_PARENT="$(dirname "${LOCAL_PROJECT_DIR}")"
LOCAL_PROJECT_NAME="$(basename "${LOCAL_PROJECT_DIR}")"

JOB_NAME="${JOB_NAME:-}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-1}"

REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/afs-grj/pretrain_healthcheck}"
REMOTE_PROJECT_PARENT="$(dirname "${REMOTE_PROJECT_DIR}")"
REMOTE_PROJECT_NAME="$(basename "${REMOTE_PROJECT_DIR}")"
REMOTE_RESULT_ROOT="${REMOTE_RESULT_ROOT:-${REMOTE_PROJECT_DIR}/results/vcctl}"
LOCAL_RESULT_ROOT="${LOCAL_RESULT_ROOT:-${LOCAL_PROJECT_DIR}/results/vcctl}"

SYNC_PROJECT="${SYNC_PROJECT:-1}"
SYNC_RESULTS_BACK="${SYNC_RESULTS_BACK:-1}"
REMOTE_CLEAN_PROJECT="${REMOTE_CLEAN_PROJECT:-1}"
POD_JSON_FILE="${POD_JSON_FILE:-}"

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<vcjob-name> [env ...] bash scripts/metax/run_vcctl_healthcheck_with_sync.sh

This wrapper is for jobs whose pods cannot see the developer-machine project path.
It syncs the local project into a pod-visible shared path, runs the normal vcctl
healthcheck, then copies pod-side results back to the local result root.

Common env:
  JOB_NAME             vcjob name. Required unless POD_JSON_FILE is set.
  NAMESPACE            Kubernetes namespace. Default: default
  LOCAL_PROJECT_DIR    Local pretrain_healthcheck path. Default: current project
  REMOTE_PROJECT_DIR   Pod-visible project path. Default: /afs-grj/pretrain_healthcheck
  LOCAL_RESULT_ROOT    Local result root. Default: LOCAL_PROJECT_DIR/results/vcctl
  REMOTE_RESULT_ROOT   Pod-visible result root. Default: REMOTE_PROJECT_DIR/results/vcctl
  RUN_ID               Run id. Default: current timestamp
  SYNC_PROJECT         1 syncs project before running. Default: 1
  SYNC_RESULTS_BACK    1 copies remote results back after running. Default: 1
  REMOTE_CLEAN_PROJECT 1 removes REMOTE_PROJECT_DIR before syncing. Default: 1
  DRY_RUN              Passed to run_vcctl_healthcheck.sh. Default: 1

All healthcheck envs such as MODE, PROFILE, MESSAGE_SIZES, WARMUP, ITERS,
COLLECTIVE_BANDWIDTH_* are forwarded to run_vcctl_healthcheck.sh.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${JOB_NAME}" && -z "${POD_JSON_FILE}" ]]; then
  echo "[vcctl-sync] JOB_NAME is required unless POD_JSON_FILE is set" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "${LOCAL_PROJECT_DIR}" ]]; then
  echo "[vcctl-sync] LOCAL_PROJECT_DIR does not exist: ${LOCAL_PROJECT_DIR}" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d /tmp/pretrain_vcctl_sync.XXXXXX)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

PODS_JSON="${POD_JSON_FILE:-${TMP_DIR}/pods.raw.json}"
if [[ -z "${POD_JSON_FILE}" ]]; then
  "${VCCTL_BIN}" pod get --job "${JOB_NAME}" -n "${NAMESPACE}" -o json > "${PODS_JSON}"
fi

master_info="$(
  python3 - "${PODS_JSON}" "${CONTAINER_NAME}" <<'PY'
import json
import sys

path = sys.argv[1]
forced_container = sys.argv[2]
text = open(path, encoding="utf-8").read()
decoder = json.JSONDecoder()
idx = 0
pods = []
while idx < len(text):
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        break
    obj, idx = decoder.raw_decode(text, idx)
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        pods.extend(x for x in obj["items"] if isinstance(x, dict))
    elif isinstance(obj, dict):
        pods.append(obj)

def labels(pod):
    return pod.get("metadata", {}).get("labels", {}) or {}

def containers(pod):
    return pod.get("spec", {}).get("containers", []) or []

def env_value(container, name):
    for item in container.get("env", []) or []:
        if item.get("name") == name:
            return str(item.get("value", ""))
    return ""

def choose_container(pod):
    cs = containers(pod)
    if not cs:
        return ""
    if forced_container:
        for c in cs:
            if c.get("name") == forced_container:
                return forced_container
    task = labels(pod).get("volcano.sh/task-spec", "")
    for c in cs:
        if c.get("name") == task:
            return str(c.get("name", ""))
    return str(cs[0].get("name", ""))

def rank(pod):
    cname = choose_container(pod)
    for c in containers(pod):
        if c.get("name") == cname:
            return env_value(c, "RANK")
    return ""

pods = [p for p in pods if p.get("metadata", {}).get("name")]
if not pods:
    raise SystemExit("no pods found")
selected = None
for pod in pods:
    if rank(pod) == "0":
        selected = pod
        break
if selected is None:
    for pod in pods:
        if labels(pod).get("volcano.sh/task-spec") == "master":
            selected = pod
            break
if selected is None:
    selected = pods[0]
print(selected["metadata"]["name"] + "\t" + choose_container(selected))
PY
)"

MASTER_POD="${master_info%%$'\t'*}"
MASTER_CONTAINER="${master_info#*$'\t'}"

echo "[vcctl-sync] job                 : ${JOB_NAME:-<fixture>}"
echo "[vcctl-sync] namespace           : ${NAMESPACE}"
echo "[vcctl-sync] master pod          : ${MASTER_POD}"
echo "[vcctl-sync] master container    : ${MASTER_CONTAINER}"
echo "[vcctl-sync] local project       : ${LOCAL_PROJECT_DIR}"
echo "[vcctl-sync] remote project      : ${REMOTE_PROJECT_DIR}"
echo "[vcctl-sync] local result root   : ${LOCAL_RESULT_ROOT}"
echo "[vcctl-sync] remote result root  : ${REMOTE_RESULT_ROOT}"
echo "[vcctl-sync] run id              : ${RUN_ID}"
echo "[vcctl-sync] dry run             : ${DRY_RUN}"

if [[ "${SYNC_PROJECT}" == "1" || "${SYNC_PROJECT}" == "true" ]]; then
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "[vcctl-sync] DRY_RUN=1, skip project sync"
  else
    echo "[vcctl-sync] syncing project to ${MASTER_POD}:${REMOTE_PROJECT_DIR}"
    remote_prepare="mkdir -p ${REMOTE_PROJECT_PARENT}"
    if [[ "${REMOTE_CLEAN_PROJECT}" == "1" || "${REMOTE_CLEAN_PROJECT}" == "true" ]]; then
      remote_prepare="rm -rf ${REMOTE_PROJECT_DIR} && ${remote_prepare}"
    fi
    tar \
      --exclude="${LOCAL_PROJECT_NAME}/.git" \
      --exclude="${LOCAL_PROJECT_NAME}/results" \
      --exclude="${LOCAL_PROJECT_NAME}/__pycache__" \
      --exclude="${LOCAL_PROJECT_NAME}/.pytest_cache" \
      -C "${LOCAL_PROJECT_PARENT}" \
      -czf - "${LOCAL_PROJECT_NAME}" \
      | "${VCCTL_BIN}" pod exec "${MASTER_POD}" -n "${NAMESPACE}" -c "${MASTER_CONTAINER}" -i -- \
          bash -lc "${remote_prepare} && tar -xzf - -C ${REMOTE_PROJECT_PARENT} && if [ '${LOCAL_PROJECT_NAME}' != '${REMOTE_PROJECT_NAME}' ]; then rm -rf ${REMOTE_PROJECT_DIR} && mv ${REMOTE_PROJECT_PARENT}/${LOCAL_PROJECT_NAME} ${REMOTE_PROJECT_DIR}; fi"
  fi
fi

set +e
PROJECT_REMOTE_DIR="${REMOTE_PROJECT_DIR}" \
RESULT_ROOT="${LOCAL_RESULT_ROOT}" \
POD_RESULT_ROOT="${REMOTE_RESULT_ROOT}" \
RUN_ID="${RUN_ID}" \
POD_JSON_FILE="${PODS_JSON}" \
VCCTL_BIN="${VCCTL_BIN}" \
NAMESPACE="${NAMESPACE}" \
JOB_NAME="${JOB_NAME}" \
CONTAINER_NAME="${CONTAINER_NAME}" \
bash "${SCRIPT_DIR}/run_vcctl_healthcheck.sh"
healthcheck_rc=$?
set -e

sync_rc=0
if [[ "${SYNC_RESULTS_BACK}" == "1" || "${SYNC_RESULTS_BACK}" == "true" ]]; then
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "[vcctl-sync] DRY_RUN=1, skip result sync-back"
  else
    echo "[vcctl-sync] syncing results back to ${LOCAL_RESULT_ROOT}/${RUN_ID}"
    mkdir -p "${LOCAL_RESULT_ROOT}"
    remote_tar="${TMP_DIR}/${RUN_ID}.remote_results.tgz"
    set +e
    "${VCCTL_BIN}" pod exec "${MASTER_POD}" -n "${NAMESPACE}" -c "${MASTER_CONTAINER}" -- \
      bash -lc "cd ${REMOTE_RESULT_ROOT} && tar -czf - ${RUN_ID}" > "${remote_tar}"
    sync_rc=$?
    set -e
    if [[ "${sync_rc}" == "0" ]]; then
      tar -xzf "${remote_tar}" -C "${LOCAL_RESULT_ROOT}"
      echo "[vcctl-sync] synced results: ${LOCAL_RESULT_ROOT}/${RUN_ID}"
    else
      echo "[vcctl-sync] result sync-back failed with rc=${sync_rc}" >&2
    fi
  fi
fi

if [[ "${healthcheck_rc}" != "0" ]]; then
  exit "${healthcheck_rc}"
fi
exit "${sync_rc}"
