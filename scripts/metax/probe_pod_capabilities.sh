#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SHARED_OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/metax_pod_capabilities_$(date +%Y%m%d_%H%M%S)}"
STATIC_OUTPUT_MODE="${STATIC_OUTPUT_MODE:-compact}"
STATIC_TMP_ROOT="${STATIC_TMP_ROOT:-${TMPDIR:-/tmp}}"
STATIC_KEEP_LOCAL_TMP="${STATIC_KEEP_LOCAL_TMP:-1}"
STATIC_COPY_RAW_OUTPUT="${STATIC_COPY_RAW_OUTPUT:-0}"
STATIC_STDOUT_MAX_BYTES="${STATIC_STDOUT_MAX_BYTES:-1048576}"

safe_pod_name="$(printf "%s" "${HC_POD_NAME:-pod}" | tr -c 'A-Za-z0-9_.-' '_')"
safe_run_id="$(printf "%s" "${HC_RUN_ID:-$(date +%Y%m%d_%H%M%S)}" | tr -c 'A-Za-z0-9_.-' '_')"
safe_run_stage="$(printf "%s" "${HC_RUN_STAGE:-static}" | tr -c 'A-Za-z0-9_.-' '_')"
STATIC_WORK_ROOT="${STATIC_WORK_ROOT:-${STATIC_TMP_ROOT%/}/pretrain_healthcheck_${safe_run_id}_${safe_pod_name}_$$}"
STATIC_WORK_DIR="${STATIC_WORK_DIR:-${STATIC_WORK_ROOT}/${safe_run_stage}}"

mkdir -p "${STATIC_WORK_DIR}"

OUT_DIR="${STATIC_WORK_DIR}"

SUMMARY="${OUT_DIR}/summary.tsv"
DETAIL="${OUT_DIR}/details.log"
JSONL="${OUT_DIR}/checks.jsonl"

: > "${SUMMARY}"
: > "${DETAIL}"
: > "${JSONL}"

printf "category\titem\tstatus\tdetail\n" >> "${SUMMARY}"

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

record() {
  local category="$1"
  local item="$2"
  local status="$3"
  local detail="$4"
  printf "%s\t%s\t%s\t%s\n" "${category}" "${item}" "${status}" "${detail}" >> "${SUMMARY}"
  printf '{"category":%s,"item":%s,"status":%s,"detail":%s}\n' \
    "$(printf "%s" "${category}" | json_escape)" \
    "$(printf "%s" "${item}" | json_escape)" \
    "$(printf "%s" "${status}" | json_escape)" \
    "$(printf "%s" "${detail}" | json_escape)" >> "${JSONL}"
}

run_check() {
  local category="$1"
  local item="$2"
  shift 2
  local log="${OUT_DIR}/${category}_${item}.log"
  local err="${OUT_DIR}/${category}_${item}.err"
  mkdir -p "$(dirname "${log}")"
  echo "===== ${category}/${item}: $*" >> "${DETAIL}"
  if "$@" > "${log}" 2> "${err}"; then
    local first
    first="$(head -n 1 "${log}" | tr '\t' ' ' | cut -c1-180)"
    record "${category}" "${item}" "OK" "${first:-command succeeded}"
  else
    local rc=$?
    local reason
    reason="$(head -n 1 "${err}" | tr '\t' ' ' | cut -c1-180)"
    if [[ ${rc} -eq 127 ]]; then
      record "${category}" "${item}" "MISSING" "command not found"
    else
      record "${category}" "${item}" "FAIL" "rc=${rc} ${reason}"
    fi
  fi
  {
    echo "\$ $*"
    echo "--- stdout ---"
    sed -n '1,120p' "${log}" 2>/dev/null
    echo "--- stderr ---"
    sed -n '1,80p' "${err}" 2>/dev/null
    echo
  } >> "${DETAIL}"
}

path_check() {
  local category="$1"
  local item="$2"
  local path="$3"
  if [[ -e "${path}" ]]; then
    if [[ -r "${path}" ]]; then
      record "${category}" "${item}" "OK" "readable ${path}"
    else
      record "${category}" "${item}" "NO_READ" "exists but not readable ${path}"
    fi
  else
    record "${category}" "${item}" "MISSING" "missing ${path}"
  fi
}

collect_pci_numa_nodes() {
  local out="${OUT_DIR}/pci_numa_nodes.txt"
  local err="${OUT_DIR}/pci_numa_nodes.err"
  : > "${out}"
  : > "${err}"

  local path
  shopt -s nullglob
  for path in /sys/bus/pci/devices/*/numa_node; do
    if [[ -r "${path}" ]]; then
      printf "%s " "${path}" >> "${out}"
      cat "${path}" >> "${out}" 2>> "${err}" || true
    fi
  done
  shopt -u nullglob

  if [[ -s "${out}" ]]; then
    record sys pci_numa_nodes "OK" "wrote pci_numa_nodes.txt"
  else
    record sys pci_numa_nodes "EMPTY" "no readable /sys/bus/pci/devices/*/numa_node entries"
  fi
}

collect_infiniband_sysfs() {
  local out="${OUT_DIR}/infiniband_sysfs.txt"
  local err="${OUT_DIR}/infiniband_sysfs.err"
  : > "${out}"
  : > "${err}"

  local dev port name path
  shopt -s nullglob
  for dev in /sys/class/infiniband/*; do
    for port in "${dev}"/ports/*; do
      for name in state rate mtu lid link_layer; do
        path="${port}/${name}"
        if [[ -r "${path}" ]]; then
          printf "%s " "${path}" >> "${out}"
          cat "${path}" >> "${out}" 2>> "${err}" || true
        fi
      done
    done
  done
  shopt -u nullglob

  if [[ -s "${out}" ]]; then
    record sys infiniband_sysfs "OK" "wrote infiniband_sysfs.txt"
  else
    record sys infiniband_sysfs "EMPTY" "no readable /sys/class/infiniband/*/ports/* entries"
  fi
}

echo "[metax-probe] output: ${SHARED_OUT_DIR}" >&2
echo "[metax-probe] workroot: ${STATIC_WORK_ROOT}" >&2
echo "[metax-probe] workdir: ${OUT_DIR}" >&2

run_check basic uname uname -a
run_check basic date date -Is
run_check basic id id
run_check basic env env
run_check basic mount mount
run_check basic df df -h
run_check basic inode df -ih

run_check metax mx_smi mx-smi
run_check metax mx_smi_help mx-smi --help
run_check metax python_torch python3 -c 'import torch; import torch.distributed as dist; print("torch", torch.__version__); print("cuda_available", torch.cuda.is_available()); print("device_count", torch.cuda.device_count()); print("distributed_backends", sorted(dist.Backend.backend_type_map.keys()))'
run_check metax maca_env bash -lc 'env | sort | grep -E "^(MACA|MCCL|CUDA|NCCL|PYTORCH|RANK|WORLD_SIZE|MASTER_|LOCAL_|HC_)" || true'

run_check hca rdma_link rdma link
run_check hca rdma_dev rdma dev
run_check hca ibv_devinfo ibv_devinfo
run_check hca ip_link ip -br link
run_check hca ip_addr ip -br addr
run_check hca ip_route ip route
if command -v ip >/dev/null 2>&1; then
  run_check hca xscale_links bash -lc 'ip -br link | grep -E "xscale|net[0-9]|eth[0-9]"'
  run_check hca xscale_addr bash -lc 'ip -br addr | grep -E "xscale|net[0-9]|eth[0-9]"'
else
  record hca xscale_links "MISSING" "ip command not found"
  record hca xscale_addr "MISSING" "ip command not found"
fi

run_check numa lscpu lscpu
run_check numa numactl numactl --hardware

record logs dmesg "SKIP" "pod environment does not support dmesg; kernel log screening is owned by ops"
run_check logs journalctl journalctl -k -p warning --no-pager -n 100

path_check proc proc_cmdline /proc/cmdline
path_check proc proc_modules /proc/modules
path_check proc proc_interrupts /proc/interrupts
path_check proc proc_meminfo /proc/meminfo
path_check proc proc_cpuinfo /proc/cpuinfo

path_check sys sys_pci /sys/bus/pci/devices
path_check sys sys_infiniband /sys/class/infiniband
path_check sys sys_net /sys/class/net
path_check sys pci_switch_link /opt/pci_switch_link

if [[ -d /sys/bus/pci/devices ]]; then
  collect_pci_numa_nodes
fi

if [[ -d /sys/class/infiniband ]]; then
  collect_infiniband_sysfs
fi

python3 - <<'PY' "${SUMMARY}" "${OUT_DIR}/summary.md"
import csv, sys
tsv, md = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(tsv), delimiter='\t'))
with open(md, 'w', encoding='utf-8') as f:
    f.write('# MetaX Pod Capability Probe Summary\n\n')
    f.write('| category | item | status | detail |\n')
    f.write('| --- | --- | --- | --- |\n')
    for r in rows:
        f.write(f"| {r['category']} | {r['item']} | {r['status']} | {r['detail'].replace('|','/')} |\n")
PY

python3 - <<'PY' "${OUT_DIR}"
import json
import os
import re
import socket
import sys
from collections import Counter
from pathlib import Path

out_dir = Path(sys.argv[1])


def read_text(name: str) -> str:
    path = out_dir / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_jsonl(name: str) -> list[dict]:
    rows = []
    path = out_dir / name
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return normalize_ws(match.group(1)) if match else ""


def parse_mx_smi(text: str) -> dict:
    facts = {
        "mx_smi_version": first_match(text, r"mx-smi\s+version:\s*([^\n]+)"),
        "attached_gpus": first_match(text, r"Attached GPUs\s*:\s*(\d+)"),
        "driver_version": first_match(text, r"Kernel Mode Driver Version:\s*([^\s|]+)"),
        "maca_version": first_match(text, r"MACA Version:\s*([^\s|]+)"),
        "bios_version": first_match(text, r"BIOS Version:\s*([^\s|]+)"),
    }
    model_counts = Counter()
    for match in re.finditer(r"\|\s*\d+\s+([^|]+?)\s+\|\s*\d+\s+", text):
        model_counts[normalize_ws(match.group(1))] += 1
    if model_counts:
        facts["gpu_model_counts"] = dict(sorted(model_counts.items()))
    facts["gpu_available_count"] = len(re.findall(r"\bAvailable\b", text))
    facts["gpu_state_unavailable_count"] = len(re.findall(r"\bUnavailable\b", text))
    return {k: v for k, v in facts.items() if v is not None and v != ""}


def parse_python_torch(text: str) -> dict:
    fields = {
        "version": first_match(text, r"^torch\s+(.+)$"),
        "cuda_available": first_match(text, r"^cuda_available\s+(.+)$"),
        "device_count": first_match(text, r"^device_count\s+(.+)$"),
    }
    return {k: v for k, v in fields.items() if v}


def parse_uname(text: str) -> dict:
    parts = normalize_ws(text).split()
    return {"raw": normalize_ws(text), "kernel": parts[2] if len(parts) >= 3 else ""}


def parse_infiniband_sysfs(text: str) -> dict:
    ports = []
    by_port: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        match = re.match(r"/sys/class/infiniband/([^/]+)/ports/([^/]+)/([^ ]+)\s+(.+)$", line.strip())
        if not match:
            continue
        device, port, name, value = match.groups()
        key = f"{device}:{port}"
        by_port.setdefault(key, {"device": device, "port": port})[name] = normalize_ws(value)
    for key in sorted(by_port):
        ports.append(by_port[key])
    xscale_ports = [p for p in ports if str(p.get("device", "")).startswith("xscale_")]
    return {
        "port_count": len(ports),
        "xscale_port_count": len(xscale_ports),
        "ports": ports,
    }


def parse_ibv_devinfo(text: str) -> dict:
    hca_ids = sorted(set(re.findall(r"^\s*hca_id:\s*(\S+)", text, flags=re.MULTILINE)))
    return {"hca_ids": hca_ids, "hca_count": len(hca_ids)}


def parse_df(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 6:
            rows.append(
                {
                    "filesystem": parts[0],
                    "size": parts[1],
                    "used": parts[2],
                    "avail": parts[3],
                    "use_percent": parts[4],
                    "mount": parts[5],
                }
            )
    return rows


checks = read_jsonl("checks.jsonl")
status_by_check = {
    f"{row.get('category', '')}/{row.get('item', '')}": {
        "status": row.get("status", ""),
        "detail": row.get("detail", ""),
    }
    for row in checks
}

facts = {
    "schema_version": 1,
    "static_workdir": str(out_dir),
    "pod": {
        "name": os.environ.get("HC_POD_NAME", socket.gethostname()),
        "node_name": os.environ.get("HC_NODE_NAME", ""),
        "pod_ip": os.environ.get("HC_POD_IP", ""),
        "host_ip": os.environ.get("HC_HOST_IP", ""),
        "job_name": os.environ.get("HC_JOB_NAME", ""),
        "run_id": os.environ.get("HC_RUN_ID", ""),
        "mode": os.environ.get("HC_MODE", ""),
        "device_type": os.environ.get("HC_DEVICE_TYPE", ""),
    },
    "basic": {
        "uname": parse_uname(read_text("basic_uname.log")),
        "date": normalize_ws(read_text("basic_date.log")),
    },
    "container": {
        "env_keys": sorted(
            key
            for key in os.environ
            if key.startswith(("MACA", "MCCL", "CUDA", "NCCL", "PYTORCH", "RANK", "WORLD_SIZE", "MASTER_", "LOCAL_", "HC_"))
        ),
    },
    "gpu": {
        "metax": parse_mx_smi(read_text("metax_mx_smi.log")),
        "torch": parse_python_torch(read_text("metax_python_torch.log")),
    },
    "hca": {
        "ibv_devinfo": parse_ibv_devinfo(read_text("hca_ibv_devinfo.log")),
        "sysfs": parse_infiniband_sysfs(read_text("infiniband_sysfs.txt")),
    },
    "storage": {
        "df": parse_df(read_text("basic_df.log")),
        "inode": parse_df(read_text("basic_inode.log")),
    },
    "capability": {
        "dmesg": "skipped_in_pod",
        "checks": status_by_check,
    },
    "checks": checks,
}

statuses = [str(row.get("status", "")) for row in checks]
facts["status"] = {
    "check_count": len(checks),
    "fail_count": sum(1 for status in statuses if status == "FAIL"),
    "missing_count": sum(1 for status in statuses if status == "MISSING"),
    "warn_count": sum(1 for status in statuses if status in {"MISSING", "NO_READ", "EMPTY", "SKIP"}),
}
facts["status"]["overall"] = "FAIL" if facts["status"]["fail_count"] else "PASS"

(out_dir / "compact_facts.json").write_text(
    json.dumps(facts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

if [[ "${STATIC_OUTPUT_MODE}" != "compact" || "${STATIC_COPY_RAW_OUTPUT}" == "1" || "${STATIC_COPY_RAW_OUTPUT}" == "true" ]]; then
  mkdir -p "${SHARED_OUT_DIR}"
  cp -a "${OUT_DIR}/." "${SHARED_OUT_DIR}/"
fi

if ! python3 - <<'PY' "${OUT_DIR}/compact_facts.json" "${STATIC_STDOUT_MAX_BYTES}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_bytes = int(sys.argv[2])
payload = json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, sort_keys=True)
payload_bytes = len(payload.encode("utf-8"))
if payload_bytes > max_bytes:
    print(
        f"[metax-probe] compact payload too large: {payload_bytes} > {max_bytes}",
        file=sys.stderr,
    )
    raise SystemExit(3)
print("__HC_STATIC_RESULT_JSON__ " + payload)
PY
then
  echo "[metax-probe] failed to emit static result frame" >&2
  exit 3
fi

echo "[metax-probe] summary: ${OUT_DIR}/summary.md" >&2
echo "[metax-probe] compact: ${OUT_DIR}/compact_facts.json" >&2
if [[ "${STATIC_OUTPUT_MODE}" != "compact" ]]; then
  echo "[metax-probe] details: ${OUT_DIR}/details.log" >&2
fi

if [[ "${STATIC_KEEP_LOCAL_TMP}" != "1" && "${STATIC_KEEP_LOCAL_TMP}" != "true" ]]; then
  rm -rf "${OUT_DIR}"
fi
