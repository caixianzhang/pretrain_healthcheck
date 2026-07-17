#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SHARED_OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/ascend_pod_capabilities_$(date +%Y%m%d_%H%M%S)}"
STATIC_OUTPUT_MODE="${STATIC_OUTPUT_MODE:-compact}"
STATIC_TMP_ROOT="${STATIC_TMP_ROOT:-${TMPDIR:-/tmp}}"
STATIC_KEEP_LOCAL_TMP="${STATIC_KEEP_LOCAL_TMP:-1}"
STATIC_COPY_RAW_OUTPUT="${STATIC_COPY_RAW_OUTPUT:-0}"
STATIC_STDOUT_MAX_BYTES="${STATIC_STDOUT_MAX_BYTES:-1048576}"
ASCEND_LD_LIBRARY_PATH="${ASCEND_LD_LIBRARY_PATH:-/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64}"
export LD_LIBRARY_PATH="${ASCEND_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

safe_pod_name="$(printf "%s" "${HC_POD_NAME:-pod}" | tr -c 'A-Za-z0-9_.-' '_')"
safe_run_id="$(printf "%s" "${HC_RUN_ID:-$(date +%Y%m%d_%H%M%S)}" | tr -c 'A-Za-z0-9_.-' '_')"
safe_run_stage="$(printf "%s" "${HC_RUN_STAGE:-static}" | tr -c 'A-Za-z0-9_.-' '_')"
STATIC_WORK_ROOT="${STATIC_WORK_ROOT:-${STATIC_TMP_ROOT%/}/pretrain_healthcheck_${safe_run_id}_${safe_pod_name}_$$}"
STATIC_WORK_DIR="${STATIC_WORK_DIR:-${STATIC_WORK_ROOT}/${safe_run_stage}}"

mkdir -p "${STATIC_WORK_DIR}"

static_fault_target() {
  local target=0
  if [[ -n "${STATIC_FAULT_POD:-}" && "${STATIC_FAULT_POD}" == "${HC_POD_NAME:-}" ]]; then
    target=1
  fi
  if [[ -n "${STATIC_FAULT_RANK:-}" && "${STATIC_FAULT_RANK}" == "${RANK:-}" ]]; then
    target=1
  fi
  if [[ -n "${STATIC_FAULT_NODE:-}" && "${STATIC_FAULT_NODE}" == "${HC_NODE_NAME:-}" ]]; then
    target=1
  fi
  [[ "${target}" == "1" && -n "${STATIC_FAULT_TYPE:-}" ]]
}

if static_fault_target && [[ "${STATIC_FAULT_TYPE}" == "probe_timeout" ]]; then
  echo "[static-fault] target pod=${HC_POD_NAME:-} rank=${RANK:-} node=${HC_NODE_NAME:-} type=probe_timeout sleep=${STATIC_FAULT_SLEEP_SECONDS:-600}" >&2
  sleep "${STATIC_FAULT_SLEEP_SECONDS:-600}"
fi

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
    sed -n '1,160p' "${log}" 2>/dev/null
    echo "--- stderr ---"
    sed -n '1,100p' "${err}" 2>/dev/null
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

  local dev port name path value
  shopt -s nullglob
  for dev in /sys/class/infiniband/*; do
    for port in "${dev}"/ports/*; do
      for name in state rate mtu lid link_layer; do
        path="${port}/${name}"
        if [[ -r "${path}" ]]; then
          if value="$(cat "${path}" 2>> "${err}")"; then
            printf "%s %s\n" "${path}" "${value}" >> "${out}"
          fi
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

collect_net_sysfs() {
  local out="${OUT_DIR}/net_sysfs.txt"
  local err="${OUT_DIR}/net_sysfs.err"
  : > "${out}"
  : > "${err}"

  local dev name path value
  shopt -s nullglob
  for dev in /sys/class/net/*; do
    for name in operstate mtu speed address; do
      path="${dev}/${name}"
      if [[ -r "${path}" ]]; then
        if value="$(cat "${path}" 2>> "${err}")"; then
          printf "%s %s\n" "${path}" "${value}" >> "${out}"
        fi
      fi
    done
  done
  shopt -u nullglob

  if [[ -s "${out}" ]]; then
    record sys net_sysfs "OK" "wrote net_sysfs.txt"
  else
    record sys net_sysfs "EMPTY" "no readable /sys/class/net entries"
  fi
}

collect_npu_ecc() {
  local ids=()
  local id

  mapfile -t ids < <(
    awk -F'|' '
      NF > 2 {
        field = $2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", field)
        count = split(field, parts, /[[:space:]]+/)
        if (count >= 2 && parts[1] ~ /^[0-9]+$/ && parts[2] !~ /^[0-9]+$/) {
          print parts[1]
        }
      }
    ' "${OUT_DIR}/ascend_npu_smi.log" | sort -n -u
  )
  if [[ ${#ids[@]} -eq 0 ]]; then
    echo "no physical NPU IDs found in npu-smi info output" >&2
    return 2
  fi

  local rc=0
  for id in "${ids[@]}"; do
    printf "===== NPU %s =====\n" "${id}"
    if ! npu-smi info -t ecc -i "${id}"; then
      printf "npu-smi ECC query failed for NPU %s\n" "${id}" >&2
      rc=1
    fi
  done
  return "${rc}"
}

echo "[ascend-probe] output: ${SHARED_OUT_DIR}" >&2
echo "[ascend-probe] workroot: ${STATIC_WORK_ROOT}" >&2
echo "[ascend-probe] workdir: ${OUT_DIR}" >&2

run_check basic uname uname -a
run_check basic date date -Is
run_check basic id id
run_check basic env env
run_check basic mount mount
run_check basic df df -h
run_check basic inode df -ih

run_check ascend npu_smi npu-smi info
run_check ascend npu_smi_help npu-smi -h
run_check ascend npu_ecc collect_npu_ecc
run_check ascend cann_dirs bash -lc 'ls -ld /usr/local/Ascend /usr/local/Ascend/* /usr/local/Ascend/driver/* 2>/dev/null | head -120'
run_check ascend python_torch python3 -c 'import torch; import torch.distributed as dist; print("torch", torch.__version__);
try:
 import torch_npu; print("torch_npu", getattr(torch_npu, "__version__", "unknown")); print("npu_available", torch.npu.is_available()); print("device_count", torch.npu.device_count())
except Exception as exc:
 print("torch_npu_error", type(exc).__name__ + ":" + str(exc)); print("cuda_available", torch.cuda.is_available()); print("device_count", torch.cuda.device_count())
print("distributed_backends", sorted(dist.Backend.backend_type_map.keys()))'
run_check ascend ascend_env bash -lc 'env | sort | grep -E "^(ASCEND|HCCL|CANN|TE|MS|PYTORCH|RANK|WORLD_SIZE|MASTER_|LOCAL_|HC_)" || true'
run_check ascend ascend_visible_devices bash -lc 'printf "%s\n" "${ASCEND_VISIBLE_DEVICES:-}"'

run_check net ip_link ip -br link
run_check net ip_addr ip -br addr
run_check net ip_route ip route
run_check net resolv_conf cat /etc/resolv.conf
run_check rdma rdma_link rdma link
run_check rdma rdma_dev rdma dev
run_check rdma ibv_devinfo ibv_devinfo
run_check ascend hccn_tool_discovery bash -lc 'command -v hccn_tool || find /usr/local/Ascend -name hccn_tool -type f 2>/dev/null | head -1 || true'

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
path_check sys sys_net /sys/class/net
path_check sys sys_infiniband /sys/class/infiniband
path_check rdma dev_infiniband /dev/infiniband
path_check ascend ascend_driver /usr/local/Ascend/driver
path_check ascend ascend_cann /usr/local/Ascend/cann

if [[ -d /sys/bus/pci/devices ]]; then
  collect_pci_numa_nodes
fi
if [[ -d /sys/class/infiniband ]]; then
  collect_infiniband_sysfs
fi
if [[ -d /sys/class/net ]]; then
  collect_net_sysfs
fi

python3 - <<'PY' "${SUMMARY}" "${OUT_DIR}/summary.md"
import csv, sys
tsv, md = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(tsv), delimiter="\t"))
with open(md, "w", encoding="utf-8") as f:
    f.write("# Ascend Pod Capability Probe Summary\n\n")
    f.write("| category | item | status | detail |\n")
    f.write("| --- | --- | --- | --- |\n")
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

def parse_npu_smi(text: str) -> dict:
    facts = {
        "npu_smi_version": first_match(text, r"npu-smi\s+([^\s|]+)"),
        "version": first_match(text, r"Version:\s*([^\s|]+)"),
    }
    facts["chip_count"] = len(re.findall(r"^\|\s*\d+\s+\d+\s+\|", text, flags=re.MULTILINE))
    facts["health_counts"] = dict(Counter(re.findall(r"\|\s*\d+\s+Ascend910\s+\|\s*([A-Z]+)", text)))
    facts["process_count"] = len(re.findall(r"\|\s*\d+\s+\d+\s+\|\s*\d+\s+\|", text))
    hbm_totals = re.findall(r"(\d+)\s*/\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if hbm_totals:
        facts["hbm_total_mb_values"] = sorted(set(int(total) for _, total in hbm_totals))
    return {k: v for k, v in facts.items() if v not in ("", None, {}, [])}

ECC_FIELD_NAMES = {
    "HBM Single Bit Error Count": "hbm_single_bit_error_count",
    "HBM Double Bit Error Count": "hbm_double_bit_error_count",
    "HBM Single Bit Aggregate Total Err Cnt": "hbm_single_bit_aggregate_total_err_count",
    "HBM Double Bit Aggregate Total Err Cnt": "hbm_double_bit_aggregate_total_err_count",
    "HBM Single Bit Isolated Pages Count": "hbm_single_bit_isolated_pages_count",
    "HBM Double Bit Isolated Pages Count": "hbm_double_bit_isolated_pages_count",
    "HBM Single Bit Next-Isolated Pages Count": "hbm_single_bit_next_isolated_pages_count",
    "HBM Double Bit Next-Isolated Pages Count": "hbm_double_bit_next_isolated_pages_count",
}

def parse_npu_ecc(text: str, check_status: str) -> dict:
    totals = {field: 0 for field in ECC_FIELD_NAMES.values()}
    chip_counts_by_npu = {}
    nonzero_chips = []
    errors = []
    parsed_chip_count = 0

    chunks = re.split(r"^===== NPU (\d+) =====\s*$", text, flags=re.MULTILINE)
    if len(chunks) <= 1:
        errors.append("no NPU ECC sections found")
    for index in range(1, len(chunks), 2):
        requested_npu_id = chunks[index]
        body = chunks[index + 1] if index + 1 < len(chunks) else ""
        reported_npu_id = first_match(body, r"^\s*NPU ID\s*:\s*(\d+)\s*$")
        chip_count_raw = first_match(body, r"^\s*Chip Count\s*:\s*(\d+)\s*$")
        if reported_npu_id and reported_npu_id != requested_npu_id:
            errors.append(f"requested NPU {requested_npu_id} reported NPU {reported_npu_id}")
        if not chip_count_raw:
            errors.append(f"NPU {requested_npu_id} missing Chip Count")
            continue

        expected_chip_count = int(chip_count_raw)
        chip_counts_by_npu[requested_npu_id] = expected_chip_count
        current = {}
        parsed_for_npu = 0
        for raw_line in body.splitlines():
            if ":" not in raw_line:
                continue
            label, raw_value = (part.strip() for part in raw_line.split(":", 1))
            if label in ECC_FIELD_NAMES:
                if not re.fullmatch(r"\d+", raw_value):
                    errors.append(f"NPU {requested_npu_id} invalid {label}: {raw_value}")
                    continue
                current[ECC_FIELD_NAMES[label]] = int(raw_value)
            elif label == "Chip ID":
                if not re.fullmatch(r"\d+", raw_value):
                    errors.append(f"NPU {requested_npu_id} invalid Chip ID: {raw_value}")
                    current = {}
                    continue
                missing = sorted(set(ECC_FIELD_NAMES.values()) - set(current))
                if missing:
                    errors.append(
                        f"NPU {requested_npu_id} Chip {raw_value} missing fields: {','.join(missing)}"
                    )
                else:
                    for field, value in current.items():
                        totals[field] += value
                    if any(current.values()):
                        nonzero_chips.append(
                            {"npu_id": int(requested_npu_id), "chip_id": int(raw_value), **current}
                        )
                    parsed_chip_count += 1
                    parsed_for_npu += 1
                current = {}
        if parsed_for_npu != expected_chip_count:
            errors.append(
                f"NPU {requested_npu_id} parsed {parsed_for_npu} chips, expected {expected_chip_count}"
            )

    query_status = "OK" if check_status == "OK" and chip_counts_by_npu and not errors else "FAIL"
    topology_signature = ";".join(
        f"{npu_id}:{chip_counts_by_npu[npu_id]}"
        for npu_id in sorted(chip_counts_by_npu, key=int)
    )
    return {
        "query_status": query_status,
        "npu_count": len(chip_counts_by_npu),
        "chip_count": parsed_chip_count,
        "chip_counts_by_npu": chip_counts_by_npu,
        "topology_signature": topology_signature,
        "totals": totals,
        "nonzero_chips": nonzero_chips,
        "errors": errors,
    }

def parse_python_torch(text: str) -> dict:
    fields = {
        "version": first_match(text, r"^torch\s+(.+)$"),
        "torch_npu_version": first_match(text, r"^torch_npu\s+(.+)$"),
        "npu_available": first_match(text, r"^npu_available\s+(.+)$"),
        "cuda_available": first_match(text, r"^cuda_available\s+(.+)$"),
        "device_count": first_match(text, r"^device_count\s+(.+)$"),
        "torch_npu_error": first_match(text, r"^torch_npu_error\s+(.+)$"),
    }
    return {k: v for k, v in fields.items() if v}

def parse_uname(text: str) -> dict:
    parts = normalize_ws(text).split()
    return {"raw": normalize_ws(text), "kernel": parts[2] if len(parts) >= 3 else ""}

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

def parse_infiniband_sysfs(text: str) -> dict:
    by_port = {}
    for line in text.splitlines():
        match = re.match(r"/sys/class/infiniband/([^/]+)/ports/([^/]+)/([^ ]+)\s+(.+)$", line.strip())
        if not match:
            continue
        device, port, name, value = match.groups()
        key = f"{device}:{port}"
        by_port.setdefault(key, {"device": device, "port": port})[name] = normalize_ws(value)
    if not by_port:
        return {}
    state_rates = []
    for key in sorted(by_port):
        port = by_port[key]
        state_rates.append(
            f"{key}|state={port.get('state', '')}|rate={port.get('rate', '')}|"
            f"link={port.get('link_layer', '')}|mtu={port.get('mtu', '')}"
        )
    return {"port_count": len(by_port), "state_rates": state_rates}

def parse_net_sysfs(text: str) -> dict:
    by_name = {}
    for line in text.splitlines():
        match = re.match(r"/sys/class/net/([^/]+)/([^ ]+)\s+(.+)$", line.strip())
        if not match:
            continue
        iface, name, value = match.groups()
        by_name.setdefault(iface, {"name": iface})[name] = normalize_ws(value)
    if not by_name:
        return {}
    interfaces = []
    for iface in sorted(by_name):
        item = by_name[iface]
        interfaces.append(
            f"{item.get('name', '')}|state={item.get('operstate', '')}|"
            f"mtu={item.get('mtu', '')}|speed={item.get('speed', '')}"
        )
    return {"interface_count": len(by_name), "interfaces": interfaces}

def parse_ibv_devinfo(text: str) -> dict:
    hca_ids = sorted(set(re.findall(r"^\s*hca_id:\s*(\S+)", text, flags=re.MULTILINE)))
    return {"hca_ids": hca_ids, "hca_count": len(hca_ids)} if hca_ids else {}

def parse_hccn_tool_discovery(text: str) -> dict:
    tool = normalize_ws(text)
    return {"tool_path": tool, "available": bool(tool)}

checks = read_jsonl("checks.jsonl")
status_by_check = {
    f"{row.get('category', '')}/{row.get('item', '')}": {
        "status": row.get("status", ""),
        "detail": row.get("detail", ""),
    }
    for row in checks
}
ecc_facts = parse_npu_ecc(
    read_text("ascend_npu_ecc.log"),
    status_by_check.get("ascend/npu_ecc", {}).get("status", "MISSING"),
)

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
            if key.startswith(("ASCEND", "HCCL", "CANN", "TE", "MS", "PYTORCH", "RANK", "WORLD_SIZE", "MASTER_", "LOCAL_", "HC_"))
        ),
        "ascend_visible_devices": os.environ.get("ASCEND_VISIBLE_DEVICES", ""),
    },
    "npu": {
        "ascend": parse_npu_smi(read_text("ascend_npu_smi.log")),
        "ecc": ecc_facts,
        "torch": parse_python_torch(read_text("ascend_python_torch.log")),
        "network": {
            "hccn_tool": parse_hccn_tool_discovery(read_text("ascend_hccn_tool_discovery.log")),
        },
    },
    "rdma": {
        "ibv_devinfo": parse_ibv_devinfo(read_text("rdma_ibv_devinfo.log")),
        "sysfs": parse_infiniband_sysfs(read_text("infiniband_sysfs.txt")),
        "dev_infiniband_exists": os.path.exists("/dev/infiniband"),
    },
    "net": {
        "sysfs": parse_net_sysfs(read_text("net_sysfs.txt")),
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


def fault_target():
    if not os.environ.get("STATIC_FAULT_TYPE"):
        return False
    if os.environ.get("STATIC_FAULT_POD") and os.environ.get("STATIC_FAULT_POD") == facts["pod"].get("name", ""):
        return True
    if os.environ.get("STATIC_FAULT_RANK") and os.environ.get("STATIC_FAULT_RANK") == os.environ.get("RANK", ""):
        return True
    if os.environ.get("STATIC_FAULT_NODE") and os.environ.get("STATIC_FAULT_NODE") == facts["pod"].get("node_name", ""):
        return True
    return False

def set_check_fail(key, detail):
    category, item = key.split("/", 1)
    checks_map = facts.setdefault("capability", {}).setdefault("checks", {})
    checks_map[key] = {"status": "FAIL", "detail": detail}
    facts.setdefault("checks", []).append({"category": category, "item": item, "status": "FAIL", "detail": detail})

def recompute_status():
    statuses = [str(row.get("status", "")) for row in facts.get("checks", [])]
    facts["status"] = {
        "check_count": len(statuses),
        "fail_count": sum(1 for status in statuses if status == "FAIL"),
        "missing_count": sum(1 for status in statuses if status == "MISSING"),
        "warn_count": sum(1 for status in statuses if status in {"MISSING", "NO_READ", "EMPTY", "SKIP"}),
    }
    facts["status"]["overall"] = "FAIL" if facts["status"]["fail_count"] else "PASS"

if fault_target():
    fault_type = os.environ.get("STATIC_FAULT_TYPE", "")
    facts["fault_injection"] = {"enabled": True, "type": fault_type}
    if fault_type == "missing_device_cmd":
        set_check_fail("ascend/npu_smi", "fault injection: simulated npu-smi failure")
    elif fault_type == "device_count_mismatch":
        ascend_facts = facts.setdefault("npu", {}).setdefault("ascend", {})
        torch_facts = facts.setdefault("npu", {}).setdefault("torch", {})
        if "chip_count" in ascend_facts:
            try:
                ascend_facts["chip_count"] = max(0, int(ascend_facts["chip_count"]) - 1)
            except Exception:
                ascend_facts["chip_count"] = "fault_injected_mismatch"
        if "device_count" in torch_facts:
            try:
                torch_facts["device_count"] = str(max(0, int(torch_facts["device_count"]) - 1))
            except Exception:
                torch_facts["device_count"] = "fault_injected_mismatch"
    elif fault_type == "driver_version_mismatch":
        facts.setdefault("npu", {}).setdefault("ascend", {})["version"] = "fault_injected_ascend_version"
    elif fault_type == "hca_count_mismatch":
        set_check_fail("net/ip_link", "fault injection: simulated network device mismatch")
    elif fault_type == "env_missing":
        env_keys = facts.setdefault("container", {}).setdefault("env_keys", [])
        facts["container"]["env_keys"] = [k for k in env_keys if not k.startswith(("HCCL", "ASCEND", "MASTER_"))]
        facts["container"]["ascend_visible_devices"] = "fault_injected_missing"
        set_check_fail("ascend/ascend_env", "fault injection: simulated missing communication env")
    elif fault_type in {"ecc_single_bit", "ecc_double_bit", "ecc_aggregate_double_bit", "ecc_isolated_page", "ecc_query_failure"}:
        ecc = facts.setdefault("npu", {}).setdefault("ecc", {})
        totals = ecc.setdefault("totals", {})
        if fault_type == "ecc_single_bit":
            totals["hbm_single_bit_error_count"] = 1
        elif fault_type == "ecc_double_bit":
            totals["hbm_double_bit_error_count"] = 1
        elif fault_type == "ecc_aggregate_double_bit":
            totals["hbm_double_bit_aggregate_total_err_count"] = 1
        elif fault_type == "ecc_isolated_page":
            totals["hbm_single_bit_isolated_pages_count"] = 1
        else:
            ecc["query_status"] = "FAIL"
            ecc.setdefault("errors", []).append("fault injection: simulated ECC query failure")
            set_check_fail("ascend/npu_ecc", "fault injection: simulated ECC query failure")
    recompute_status()

(out_dir / "compact_facts.json").write_text(
    json.dumps(facts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

if [[ "${STATIC_OUTPUT_MODE}" != "compact" || "${STATIC_COPY_RAW_OUTPUT}" == "1" || "${STATIC_COPY_RAW_OUTPUT}" == "true" ]]; then
  mkdir -p "${SHARED_OUT_DIR}"
  cp -a "${OUT_DIR}/." "${SHARED_OUT_DIR}/"
fi

if static_fault_target && [[ "${STATIC_FAULT_TYPE}" == "frame_missing" ]]; then
  echo "[ascend-probe] fault injection: suppress static result frame" >&2
  exit 0
fi
if static_fault_target && [[ "${STATIC_FAULT_TYPE}" == "frame_corrupt" ]]; then
  echo "__HC_STATIC_RESULT_JSON__ {fault_injected_corrupt_json"
  echo "[ascend-probe] fault injection: emitted corrupt static result frame" >&2
  exit 0
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
        f"[ascend-probe] compact payload too large: {payload_bytes} > {max_bytes}",
        file=sys.stderr,
    )
    raise SystemExit(3)
print("__HC_STATIC_RESULT_JSON__ " + payload)
PY
then
  echo "[ascend-probe] failed to emit static result frame" >&2
  exit 3
fi

echo "[ascend-probe] summary: ${OUT_DIR}/summary.md" >&2
echo "[ascend-probe] compact: ${OUT_DIR}/compact_facts.json" >&2
if [[ "${STATIC_OUTPUT_MODE}" != "compact" ]]; then
  echo "[ascend-probe] details: ${OUT_DIR}/details.log" >&2
fi
