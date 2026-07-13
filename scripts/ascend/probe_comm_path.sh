#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUT_DIR="${OUT_DIR:-${HC_POD_RESULT_DIR:-${PROJECT_DIR}/results/ascend_comm_path_$(date +%Y%m%d_%H%M%S)}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-16}"
DIST_BACKEND="${DIST_BACKEND:-hccl}"
DEVICE_VENDOR="${DEVICE_VENDOR:-ascend}"
COMM_RUNTIME="${COMM_RUNTIME:-hccl}"
DTYPE="${DTYPE:-bf16}"
MESSAGE_SIZES="${MESSAGE_SIZES:-1G}"
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-3}"
MIN_BUSBW="${MIN_BUSBW:-0}"
AVG_BUSBW="${AVG_BUSBW:-0}"
SEED="${SEED:-20260623}"
RUN_TORCH_DEBUG="${RUN_TORCH_DEBUG:-1}"
HEALTHCHECK_MASTER_PORT="${HEALTHCHECK_MASTER_PORT:-${MASTER_PORT:-29500}}"
ASCEND_LD_LIBRARY_PATH="${ASCEND_LD_LIBRARY_PATH:-/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64}"
export LD_LIBRARY_PATH="${ASCEND_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUT_DIR}"

log() {
  echo "[ascend-comm-probe] $*"
}

run_capture() {
  local name="$1"
  shift
  {
    echo "\$ $*"
    "$@"
  } > "${OUT_DIR}/${name}.log" 2> "${OUT_DIR}/${name}.err"
}

run_shell_capture() {
  local name="$1"
  local script="$2"
  {
    echo "\$ bash -lc ${script@Q}"
    bash -lc "${script}"
  } > "${OUT_DIR}/${name}.log" 2> "${OUT_DIR}/${name}.err"
}

collect_env() {
  env | sort | grep -E '^(HCCL|ASCEND|CANN|TE|MS|TORCH|PYTORCH|MASTER_|WORLD_SIZE|RANK|LOCAL_|HC_|VC_|JOB_NAME|POD_IP)=' \
    > "${OUT_DIR}/comm_env.log" 2> "${OUT_DIR}/comm_env.err" || true
}

collect_ib_sysfs() {
  local out="$1"
  : > "${out}"
  local dev port name path net
  shopt -s nullglob
  for dev in /sys/class/infiniband/*; do
    echo "## $(basename "${dev}")" >> "${out}"
    for name in fw_ver hca_type board_id node_guid sys_image_guid; do
      path="${dev}/${name}"
      [[ -r "${path}" ]] && printf "%s: %s\n" "${name}" "$(cat "${path}" 2>/dev/null)" >> "${out}"
    done
    if [[ -d "${dev}/device/net" ]]; then
      printf "netdevs:" >> "${out}"
      for net in "${dev}"/device/net/*; do
        printf " %s" "$(basename "${net}")" >> "${out}"
      done
      printf "\n" >> "${out}"
    fi
    for port in "${dev}"/ports/*; do
      echo "### port $(basename "${port}")" >> "${out}"
      for name in state phys_state rate mtu lid link_layer; do
        path="${port}/${name}"
        [[ -r "${path}" ]] && printf "%s: %s\n" "${name}" "$(cat "${path}" 2>/dev/null)" >> "${out}"
      done
      if [[ -d "${port}/gids" ]]; then
        echo "gids:" >> "${out}"
        for path in "${port}"/gids/*; do
          [[ -r "${path}" ]] && printf "  %s %s\n" "$(basename "${path}")" "$(cat "${path}" 2>/dev/null)" >> "${out}"
        done
      fi
    done
    echo >> "${out}"
  done
  shopt -u nullglob
}

collect_netdevs() {
  local out="$1"
  : > "${out}"
  local net path
  shopt -s nullglob
  for net in /sys/class/net/*; do
    case "$(basename "${net}")" in
      eth*|net*|xscale*|ib*|roce*)
        echo "## $(basename "${net}")" >> "${out}"
        for path in operstate mtu speed carrier address; do
          [[ -r "${net}/${path}" ]] && printf "%s: %s\n" "${path}" "$(cat "${net}/${path}" 2>/dev/null)" >> "${out}"
        done
        if [[ -e "${net}/device" ]]; then
          readlink -f "${net}/device" >> "${out}" 2>/dev/null || true
        fi
        echo >> "${out}"
        ;;
    esac
  done
  shopt -u nullglob
}

collect_ib_counters() {
  local out="$1"
  : > "${out}"
  local dev port counter path
  shopt -s nullglob
  for dev in /sys/class/infiniband/*; do
    for port in "${dev}"/ports/*; do
      for counter in port_xmit_data port_rcv_data port_xmit_packets port_rcv_packets symbol_error_counter link_error_recovery link_downed port_rcv_errors port_xmit_discards; do
        path="${port}/counters/${counter}"
        if [[ -r "${path}" ]]; then
          printf "%s\tport%s\t%s\t%s\n" "$(basename "${dev}")" "$(basename "${port}")" "${counter}" "$(cat "${path}" 2>/dev/null)" >> "${out}"
        fi
      done
    done
  done
  shopt -u nullglob
}

write_counter_delta() {
  local before="$1"
  local after="$2"
  local out="$3"
  python3 - "$before" "$after" "$out" <<'PY'
import sys
from pathlib import Path

def load(path):
    vals = {}
    p = Path(path)
    if not p.exists():
        return vals
    for line in p.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        key = tuple(parts[:3])
        try:
            vals[key] = int(parts[3])
        except ValueError:
            pass
    return vals

before = load(sys.argv[1])
after = load(sys.argv[2])
keys = sorted(set(before) | set(after))
with open(sys.argv[3], "w", encoding="utf-8") as f:
    f.write("hca\tport\tcounter\tbefore\tafter\tdelta\n")
    for key in keys:
        b = before.get(key, 0)
        a = after.get(key, 0)
        f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(*key, b, a, a - b))
PY
}

log "output: ${OUT_DIR}"
log "project: ${PROJECT_DIR}"

collect_env
run_capture uname uname -a
run_capture npu_smi npu-smi info
run_shell_capture torch_info 'python3 - <<PY
import torch
import torch.distributed as dist
print("torch", torch.__version__)
try:
    import torch_npu
    print("torch_npu", getattr(torch_npu, "__version__", "unknown"))
    print("npu_available", torch.npu.is_available())
    print("device_count", torch.npu.device_count())
except Exception as exc:
    print("torch_npu_error", type(exc).__name__ + ":" + str(exc))
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
print("distributed_backends", sorted(dist.Backend.backend_type_map.keys()))
PY'
run_shell_capture rdma_link 'command -v rdma >/dev/null 2>&1 && rdma link || true'
run_shell_capture rdma_dev 'command -v rdma >/dev/null 2>&1 && rdma dev || true'
run_shell_capture ibv_devinfo 'command -v ibv_devinfo >/dev/null 2>&1 && ibv_devinfo || true'
run_shell_capture ip_addr 'command -v ip >/dev/null 2>&1 && ip -br addr || true'
run_shell_capture ip_link 'command -v ip >/dev/null 2>&1 && ip -br link || true'

collect_ib_sysfs "${OUT_DIR}/ib_sysfs_before.txt"
collect_netdevs "${OUT_DIR}/netdev_sysfs_before.txt"
collect_ib_counters "${OUT_DIR}/ib_counters_before.tsv"

if [[ "${RUN_TORCH_DEBUG}" == "1" || "${RUN_TORCH_DEBUG}" == "true" ]]; then
  log "running debug all-reduce"
  (
    cd "${PROJECT_DIR}" || exit 1
    export DIST_BACKEND="${DIST_BACKEND}"
    export DEVICE_VENDOR="${DEVICE_VENDOR}"
    export COMM_RUNTIME="${COMM_RUNTIME}"
    export HCCL_DEBUG="${HCCL_DEBUG:-INFO}"
    export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
    torchrun \
      --nnodes="${WORLD_SIZE:-1}" \
      --nproc-per-node="${GPUS_PER_NODE}" \
      --node-rank="${RANK:-0}" \
      --master-addr="${MASTER_ADDR:-127.0.0.1}" \
      --master-port="${HEALTHCHECK_MASTER_PORT}" \
      -m pretrain_healthcheck.cli run-bandwidth \
      --output-dir "${OUT_DIR}/torch_bandwidth" \
      --dtype "${DTYPE}" \
      --message-sizes "${MESSAGE_SIZES}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      --seed "${SEED}" \
      --min-busbw "${MIN_BUSBW}" \
      --avg-busbw "${AVG_BUSBW}" \
      --test-round comm_path_probe \
      --group-id "${HEALTHCHECK_GROUP_ID:-comm-path-probe}"
  ) > "${OUT_DIR}/torch_debug.stdout" 2> "${OUT_DIR}/torch_debug.stderr"
  echo "$?" > "${OUT_DIR}/torch_debug.returncode"
fi

collect_ib_sysfs "${OUT_DIR}/ib_sysfs_after.txt"
collect_netdevs "${OUT_DIR}/netdev_sysfs_after.txt"
collect_ib_counters "${OUT_DIR}/ib_counters_after.tsv"
write_counter_delta "${OUT_DIR}/ib_counters_before.tsv" "${OUT_DIR}/ib_counters_after.tsv" "${OUT_DIR}/ib_counters_delta.tsv"

python3 - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {
    "out_dir": str(out),
    "torch_debug_returncode": None,
    "bandwidth_gate": None,
    "active_hca_counters": [],
}
rc = out / "torch_debug.returncode"
if rc.exists():
    summary["torch_debug_returncode"] = rc.read_text().strip()
gate = out / "torch_bandwidth" / "bandwidth_gate.json"
if gate.exists():
    summary["bandwidth_gate"] = json.loads(gate.read_text())
delta = out / "ib_counters_delta.tsv"
if delta.exists():
    for line in delta.read_text().splitlines()[1:]:
        hca, port, counter, before, after, d = line.split("\t")
        try:
            dv = int(d)
        except ValueError:
            continue
        if dv > 0 and counter in {"port_xmit_data", "port_rcv_data", "port_xmit_packets", "port_rcv_packets"}:
            summary["active_hca_counters"].append(
                {"hca": hca, "port": port, "counter": counter, "delta": dv}
            )
with open(out / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
with open(out / "summary.md", "w", encoding="utf-8") as f:
    f.write("# MetaX Communication Path Probe\n\n")
    f.write(f"- out_dir: `{out}`\n")
    f.write(f"- torch_debug_returncode: `{summary['torch_debug_returncode']}`\n")
    if summary["bandwidth_gate"]:
        f.write(f"- bandwidth_status: `{summary['bandwidth_gate'].get('status')}`\n")
    f.write("\n## Active HCA Counters\n\n")
    f.write("| hca | port | counter | delta |\n")
    f.write("| --- | --- | --- | ---: |\n")
    for row in summary["active_hca_counters"]:
        f.write(f"| {row['hca']} | {row['port']} | {row['counter']} | {row['delta']} |\n")
PY

log "summary: ${OUT_DIR}/summary.md"
log "torch debug stderr: ${OUT_DIR}/torch_debug.stderr"
log "counter delta: ${OUT_DIR}/ib_counters_delta.tsv"
