#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/pod_capabilities_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"

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
    sed -n '1,80p' "${log}" 2>/dev/null
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

echo "[probe] output: ${OUT_DIR}"

run_check basic uname uname -a
run_check basic date date -Is
run_check basic id id
run_check basic env env
run_check basic mount mount
run_check basic df df -h
run_check basic inode df -ih

run_check time timedatectl timedatectl status
run_check time chronyc_tracking chronyc tracking
run_check time chronyc_sources chronyc sources -v
run_check time ntpq ntpq -p
run_check time ptp4l ptp4l -v
run_check time phc2sys phc2sys -v

run_check gpu nvidia_smi_L nvidia-smi -L
run_check gpu nvidia_smi_query nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,vbios_version,temperature.gpu,power.draw,clocks.sm,memory.total,memory.used,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total --format=csv,noheader,nounits
run_check gpu nvidia_smi_topo nvidia-smi topo -m
run_check gpu nvidia_smi_q nvidia-smi -q
run_check gpu dcgmi dcgmi discovery -l

run_check hca ibv_devinfo ibv_devinfo
run_check hca ibstat ibstat
run_check hca ibstatus ibstatus
run_check hca rdma_link rdma link
run_check hca rdma_dev rdma dev
run_check hca devlink devlink dev show
run_check hca ethtool_i ethtool -i eth0
run_check hca ip_link ip -br link
run_check hca ip_addr ip -br addr
run_check hca ip_route ip route

run_check numa lscpu lscpu
run_check numa numactl numactl --hardware
run_check numa nvidia_topo nvidia-smi topo -m

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
path_check sys sys_nvidia /proc/driver/nvidia

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
    f.write('# Pod Capability Probe Summary\n\n')
    f.write('| category | item | status | detail |\n')
    f.write('| --- | --- | --- | --- |\n')
    for r in rows:
        f.write(f"| {r['category']} | {r['item']} | {r['status']} | {r['detail'].replace('|','/')} |\n")
PY

echo "[probe] summary: ${OUT_DIR}/summary.md"
echo "[probe] details: ${DETAIL}"
