#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
source "${LOCAL_PROJECT_DIR}/scripts/common/driver_python.sh"
LOCAL_PROJECT_PARENT="$(dirname "${LOCAL_PROJECT_DIR}")"
LOCAL_PROJECT_NAME="$(basename "${LOCAL_PROJECT_DIR}")"

JOB_NAME="${JOB_NAME:-}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_STAGE="${RUN_STAGE:-}"
DRY_RUN="${DRY_RUN:-1}"

REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/tmp/pretrain_healthcheck_${RUN_ID}}"
REMOTE_PROJECT_PARENT="$(dirname "${REMOTE_PROJECT_DIR}")"
REMOTE_PROJECT_NAME="$(basename "${REMOTE_PROJECT_DIR}")"
REMOTE_RESULT_ROOT="${REMOTE_RESULT_ROOT:-${REMOTE_PROJECT_DIR}/results/vcctl}"
REMOTE_POD_RESULT_ROOT="${REMOTE_POD_RESULT_ROOT:-/tmp/pretrain_healthcheck_driver_${RUN_ID}}"
LOCAL_RESULT_ROOT="${LOCAL_RESULT_ROOT:-${LOCAL_PROJECT_DIR}/results/vcctl}"

SYNC_PROJECT="${SYNC_PROJECT:-1}"
SYNC_RESULTS_BACK="${SYNC_RESULTS_BACK:-1}"
REMOTE_CLEAN_PROJECT="${REMOTE_CLEAN_PROJECT:-0}"
SYNC_VALIDATE_IMPORT="${SYNC_VALIDATE_IMPORT:-1}"
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
  DRIVER_PYTHON        Developer-machine Python >=3.9. Default: auto-discover
  LOCAL_PROJECT_DIR    Local pretrain_healthcheck path. Default: current project
  REMOTE_PROJECT_DIR   Pod-visible project path. Default: /tmp/pretrain_healthcheck_<RUN_ID>
  LOCAL_RESULT_ROOT    Local result root. Default: LOCAL_PROJECT_DIR/results/vcctl
  REMOTE_RESULT_ROOT   Pod-visible result root. Default: REMOTE_PROJECT_DIR/results/vcctl
  REMOTE_POD_RESULT_ROOT
                       Pod/driver temporary result root. Default: /tmp/pretrain_healthcheck_driver_<RUN_ID>
  RUN_ID               Run id. Default: current timestamp
  RUN_STAGE            Stage directory override passed to run_vcctl_healthcheck.sh
  SYNC_PROJECT         1 syncs project before running. Default: 1
  SYNC_RESULTS_BACK    1 copies remote results back after running. Default: 1
  REMOTE_CLEAN_PROJECT 1 removes REMOTE_PROJECT_DIR before syncing. Default: 0
  SYNC_VALIDATE_IMPORT 1 validates Python import in every pod after sync. Default: 1
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

resolve_driver_python
print_driver_python

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
  "${DRIVER_PYTHON}" - "${PODS_JSON}" "${CONTAINER_NAME}" <<'PY'
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

pod_infos="$(
  "${DRIVER_PYTHON}" - "${PODS_JSON}" "${CONTAINER_NAME}" <<'PY'
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
            value = env_value(c, "RANK")
            if value != "":
                try:
                    return int(value)
                except ValueError:
                    return 999999
    return 999999

pods = [p for p in pods if p.get("metadata", {}).get("name")]
pods.sort(key=lambda p: (rank(p), p["metadata"]["name"]))
for pod in pods:
    print(pod["metadata"]["name"] + "\t" + choose_container(pod))
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
echo "[vcctl-sync] remote pod result   : ${REMOTE_POD_RESULT_ROOT}"
echo "[vcctl-sync] run id              : ${RUN_ID}"
echo "[vcctl-sync] dry run             : ${DRY_RUN}"

if [[ "${SYNC_PROJECT}" == "1" || "${SYNC_PROJECT}" == "true" ]]; then
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "[vcctl-sync] DRY_RUN=1, skip project sync"
  else
    echo "[vcctl-sync] syncing project to each pod:${REMOTE_PROJECT_DIR}"
    remote_prepare="mkdir -p ${REMOTE_PROJECT_PARENT} ${REMOTE_RESULT_ROOT}/${RUN_ID}"
    if [[ "${REMOTE_CLEAN_PROJECT}" == "1" || "${REMOTE_CLEAN_PROJECT}" == "true" ]]; then
      remote_prepare="if [ -e ${REMOTE_PROJECT_DIR} ] && [ ! -d ${REMOTE_PROJECT_DIR} ]; then rm -f ${REMOTE_PROJECT_DIR}; fi && ${remote_prepare}"
    fi
    while IFS=$'\t' read -r pod_name pod_container; do
      [[ -n "${pod_name}" ]] || continue
      echo "[vcctl-sync] syncing project to ${pod_name}:${REMOTE_PROJECT_DIR}"
      tar \
        --exclude="${LOCAL_PROJECT_NAME}/.git" \
        --exclude="${LOCAL_PROJECT_NAME}/results" \
        --exclude="${LOCAL_PROJECT_NAME}/__pycache__" \
        --exclude="${LOCAL_PROJECT_NAME}/.pytest_cache" \
        -C "${LOCAL_PROJECT_PARENT}" \
        -czf - "${LOCAL_PROJECT_NAME}" \
        | "${VCCTL_BIN}" pod exec "${pod_name}" -n "${NAMESPACE}" -c "${pod_container}" -i -- \
            bash -lc "${remote_prepare} && tar -xzf - -C ${REMOTE_PROJECT_PARENT} && if [ '${LOCAL_PROJECT_NAME}' != '${REMOTE_PROJECT_NAME}' ]; then rm -rf ${REMOTE_PROJECT_DIR} && mv ${REMOTE_PROJECT_PARENT}/${LOCAL_PROJECT_NAME} ${REMOTE_PROJECT_DIR}; fi && mkdir -p ${REMOTE_RESULT_ROOT}/${RUN_ID}"
    done <<< "${pod_infos}"
  fi
fi

if [[ "${SYNC_VALIDATE_IMPORT}" == "1" || "${SYNC_VALIDATE_IMPORT}" == "true" ]]; then
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "[vcctl-sync] DRY_RUN=1, skip import validation"
  else
    echo "[vcctl-sync] validating import on each pod"
    while IFS=$'\t' read -r pod_name pod_container; do
      [[ -n "${pod_name}" ]] || continue
      echo "[vcctl-sync] validating import on ${pod_name}"
      "${VCCTL_BIN}" pod exec "${pod_name}" -n "${NAMESPACE}" -c "${pod_container}" -- \
        bash -lc "cd ${REMOTE_PROJECT_DIR} && export PYTHONPATH=${REMOTE_PROJECT_DIR}:\${PYTHONPATH:-} && python3 -c 'import pretrain_healthcheck.cli; print(\"IMPORT_OK\")'"
    done <<< "${pod_infos}"
  fi
fi

set +e
PROJECT_REMOTE_DIR="${REMOTE_PROJECT_DIR}" \
RESULT_ROOT="${LOCAL_RESULT_ROOT}" \
POD_RESULT_ROOT="${REMOTE_POD_RESULT_ROOT}" \
RUN_ID="${RUN_ID}" \
RUN_STAGE="${RUN_STAGE}" \
POD_JSON_FILE="${PODS_JSON}" \
VCCTL_BIN="${VCCTL_BIN}" \
NAMESPACE="${NAMESPACE}" \
JOB_NAME="${JOB_NAME}" \
CONTAINER_NAME="${CONTAINER_NAME}" \
bash "${SCRIPT_DIR}/run_vcctl_healthcheck.sh"
healthcheck_rc=$?
set -e

sync_rc=0
post_sync_compare_rc=0
if [[ "${SYNC_RESULTS_BACK}" == "1" || "${SYNC_RESULTS_BACK}" == "true" ]]; then
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "[vcctl-sync] DRY_RUN=1, skip result sync-back"
  else
    echo "[vcctl-sync] syncing results back to ${LOCAL_RESULT_ROOT}/${RUN_ID}"
    mkdir -p "${LOCAL_RESULT_ROOT}"
    while IFS=$'\t' read -r pod_name pod_container; do
      [[ -n "${pod_name}" ]] || continue
      remote_tar="${TMP_DIR}/${RUN_ID}.${pod_name}.remote_results.tgz"
      set +e
      "${VCCTL_BIN}" pod exec "${pod_name}" -n "${NAMESPACE}" -c "${pod_container}" -- \
        bash -lc "if [ -d ${REMOTE_RESULT_ROOT}/${RUN_ID} ]; then cd ${REMOTE_RESULT_ROOT} && tar -czf - ${RUN_ID}; else tar -czf - --files-from /dev/null; fi" > "${remote_tar}"
      pod_sync_rc=$?
      set -e
      if [[ "${pod_sync_rc}" == "0" ]]; then
        tar -xzf "${remote_tar}" -C "${LOCAL_RESULT_ROOT}"
        echo "[vcctl-sync] synced pod results from ${pod_name}"
      else
        sync_rc="${pod_sync_rc}"
        echo "[vcctl-sync] result sync-back failed from ${pod_name} rc=${pod_sync_rc}" >&2
      fi
    done <<< "${pod_infos}"
    echo "[vcctl-sync] synced results: ${LOCAL_RESULT_ROOT}/${RUN_ID}"

    mode_value="${MODE:-all}"
    static_compare_value="${STATIC_COMPARE:-1}"
    if [[ "${static_compare_value}" != "0" && "${static_compare_value}" != "false" ]] \
      && [[ "${mode_value}" == "static" || "${mode_value}" == "all" ]]; then
      echo "[vcctl-sync] rerunning static compare after result sync-back"
      static_result_dir="${LOCAL_RESULT_ROOT}/${RUN_ID}/static"
      set +e
      "${DRIVER_PYTHON}" "${LOCAL_PROJECT_DIR}/tools/static_compare.py" \
        --result-dir "${static_result_dir}" \
        --workers "${STATIC_COMPARE_WORKERS:-0}" \
        --ecc-policy "${STATIC_ECC_POLICY:-alert}"
      post_sync_compare_rc=$?
      set -e
      if [[ -f "${static_result_dir}/summary.json" ]]; then
        final_status="$(
          "${DRIVER_PYTHON}" - "${static_result_dir}/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(summary.get("overall_status", "SUSPECT"))
PY
        )"
        echo "[vcctl-sync] static overall_status after sync-back compare: ${final_status}"
        if [[ "${mode_value}" == "static" ]]; then
          if [[ "${final_status}" == "PASS" || "${final_status}" == "DRY_RUN" ]]; then
            healthcheck_rc=0
          else
            healthcheck_rc=1
          fi
        elif [[ "${final_status}" != "PASS" && "${final_status}" != "DRY_RUN" ]]; then
          healthcheck_rc=1
        fi
      elif [[ "${post_sync_compare_rc}" != "0" ]]; then
        healthcheck_rc="${post_sync_compare_rc}"
      fi
    fi
  fi
fi

if [[ "${healthcheck_rc}" != "0" ]]; then
  exit "${healthcheck_rc}"
fi
exit "${sync_rc}"
