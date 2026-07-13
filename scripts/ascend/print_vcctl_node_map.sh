#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JOB_NAME="${JOB_NAME:-grj-megatron-muxi-0630-moe-30ba3b}"
NAMESPACE="${NAMESPACE:-default}"
VCCTL_BIN="${VCCTL_BIN:-vcctl}"
POD_JSON_FILE="${POD_JSON_FILE:-}"
RESULT_FILE="${RESULT_FILE:-${PROJECT_DIR}/results/node_map.txt}"

usage() {
  cat <<'EOF'
Usage:
  JOB_NAME=<vcjob-name> bash scripts/ascend/print_vcctl_node_map.sh

Print pod role to physical node mapping from vcctl pod metadata.

Common env:
  JOB_NAME       vcjob name. Default: grj-megatron-muxi-0630-moe-30ba3b
  NAMESPACE      Kubernetes namespace. Default: default
  VCCTL_BIN      vcctl binary. Default: vcctl
  POD_JSON_FILE  Optional saved "vcctl pod get --job ... -o json" output.
  RESULT_FILE    Output file. Default: <project>/results/node_map.txt

Output:
  master-0   -> nodeName host-...   hostIP ...
  worker-0   -> nodeName host-...   hostIP ...
  ...
  host-...,host-...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

parse_json() {
  python3 -c '
import json
import re
import sys

job_name = sys.argv[1]
raw = sys.stdin.read().strip()
if not raw:
    raise SystemExit("empty vcctl pod json")

decoder = json.JSONDecoder()
pos = 0
objects = []
while pos < len(raw):
    while pos < len(raw) and raw[pos].isspace():
        pos += 1
    if pos >= len(raw):
        break
    obj, end = decoder.raw_decode(raw, pos)
    objects.append(obj)
    pos = end

rows = []
for obj in objects:
    metadata = obj.get("metadata", {})
    spec = obj.get("spec", {})
    status = obj.get("status", {})

    pod_name = metadata.get("name", "")
    short_name = pod_name
    prefix = f"{job_name}-"
    if short_name.startswith(prefix):
        short_name = short_name[len(prefix):]

    node_name = spec.get("nodeName", "")
    host_ip = status.get("hostIP", "")
    if not pod_name or not node_name:
        continue

    if short_name.startswith("master-"):
        group = 0
    elif short_name.startswith("worker-"):
        group = 1
    else:
        group = 2
    match = re.search(r"-(\d+)$", short_name)
    index = int(match.group(1)) if match else 0
    rows.append((group, index, short_name, node_name, host_ip))

if not rows:
    raise SystemExit("no pod node mapping found")

rows.sort(key=lambda row: (row[0], row[1], row[2]))
name_width = max(len(row[2]) for row in rows)
node_width = max(len(row[3]) for row in rows)

for _, _, short_name, node_name, host_ip in rows:
    print(f"{short_name:<{name_width}}   -> nodeName {node_name:<{node_width}}   hostIP {host_ip}")

print(",".join(row[3] for row in rows))
  ' "${JOB_NAME}"
}

if [[ -n "${POD_JSON_FILE}" ]]; then
  output="$(parse_json < "${POD_JSON_FILE}")"
else
  output="$("${VCCTL_BIN}" pod get --job "${JOB_NAME}" -n "${NAMESPACE}" -o json | parse_json)"
fi

mkdir -p "$(dirname "${RESULT_FILE}")"
printf '%s\n' "${output}" | tee "${RESULT_FILE}"
