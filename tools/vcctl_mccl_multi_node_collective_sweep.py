#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


OP_BINARIES = {
    "all_reduce": "all_reduce_perf",
    "reduce_scatter": "reduce_scatter_perf",
    "all_gather": "all_gather_perf",
    "broadcast": "broadcast_perf",
    "all_to_all": "alltoall_perf",
    "all_to_allv": "alltoallv_perf",
}

ERROR_PATTERNS = {
    "QP_CREATE_RET5": re.compile(r"ibv_cmd_create_qp_ex failed\s*,?\s*ret\s*5", re.I),
    "QP_CREATE_FAILED": re.compile(r"create[_ ]qp|failed to create.*qp", re.I),
    "ATU_FAULT": re.compile(r"ATU\s+Fault|address translation", re.I),
    "ILLEGAL_ADDRESS": re.compile(r"mcErrorIllegalAddress|illegal memory access", re.I),
    "MCCL_ERROR": re.compile(r"mccl.*(?:error|fail)|mcError", re.I),
    "MPI_ABORT": re.compile(r"MPI_ABORT|mpirun.*terminated|process.*exited", re.I),
    "MPI_LAUNCH_ERROR": re.compile(r"plm_rsh_agent|ORTE was unable to reliably start", re.I),
    "OUT_OF_MEMORY": re.compile(r"out of memory|alloc.*failed|mcErrorMemoryAllocation", re.I),
    "SSH_ERROR": re.compile(r"ssh:|host key verification failed|permission denied", re.I),
}


@dataclass(frozen=True)
class Pod:
    pod_name: str
    container_name: str
    task_spec: str
    node_name: str
    host_ip: str
    pod_ip: str
    gpu_count: int


@dataclass
class OperationResult:
    op: str
    binary: str
    status: str
    returncode: int | None
    timeout: bool
    error_type: str
    signatures: list[str]
    row_count: int
    expected_row_count: int
    correctness_pass: bool
    elapsed_seconds: float
    stdout_path: str
    stderr_path: str
    command: str
    cleanup_status: str


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def parse_json_stream(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    objects: list[dict[str, Any]] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            objects.extend(item for item in obj["items"] if isinstance(item, dict))
        elif isinstance(obj, dict):
            objects.append(obj)
        else:
            raise ValueError(f"unsupported JSON value: {type(obj).__name__}")
    return objects


def nested(obj: dict[str, Any], *keys: str, default: Any = "") -> Any:
    value: Any = obj
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def choose_container(raw: dict[str, Any], forced: str) -> dict[str, Any]:
    containers = nested(raw, "spec", "containers", default=[])
    if not isinstance(containers, list) or not containers:
        raise ValueError(f"pod {nested(raw, 'metadata', 'name')} has no containers")
    if forced:
        for container in containers:
            if container.get("name") == forced:
                return container
        raise ValueError(f"container {forced!r} not found")
    task_spec = str(nested(raw, "metadata", "labels", "volcano.sh/task-spec"))
    for container in containers:
        if container.get("name") == task_spec:
            return container
    return containers[0]


def resource_gpu_count(container: dict[str, Any]) -> int:
    for section in ("limits", "requests"):
        resources = nested(container, "resources", section, default={})
        if not isinstance(resources, dict):
            continue
        for key in ("metax-tech.com/gpu", "metax-tech.com/vgpu"):
            value = resources.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return 0


def pod_sort_key(pod: Pod) -> tuple[int, int, str]:
    group = 0 if pod.task_spec == "master" else 1 if pod.task_spec == "worker" else 2
    suffix = pod.pod_name.rsplit("-", 1)[-1]
    return group, int(suffix) if suffix.isdigit() else 0, pod.pod_name


def load_pods(args: argparse.Namespace) -> list[Pod]:
    command = [args.vcctl_bin, "pod", "get", "--job", args.job_name, "-n", args.namespace, "-o", "json"]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"vcctl pod get failed rc={proc.returncode}: {proc.stderr.strip()}")
    pods: list[Pod] = []
    for raw in parse_json_stream(proc.stdout):
        pod_name = str(nested(raw, "metadata", "name"))
        node_name = str(nested(raw, "spec", "nodeName"))
        pod_ip = str(nested(raw, "status", "podIP"))
        if not pod_name or not node_name or not pod_ip:
            continue
        container = choose_container(raw, args.container_name)
        gpu_count = args.gpus_per_node or resource_gpu_count(container)
        pods.append(
            Pod(
                pod_name=pod_name,
                container_name=str(container.get("name", "")),
                task_spec=str(nested(raw, "metadata", "labels", "volcano.sh/task-spec")),
                node_name=node_name,
                host_ip=str(nested(raw, "status", "hostIP")),
                pod_ip=pod_ip,
                gpu_count=gpu_count,
            )
        )
    pods.sort(key=pod_sort_key)
    if not pods:
        raise RuntimeError(f"no scheduled pods found for job {args.job_name!r}")
    if pods[0].task_spec != "master":
        raise RuntimeError("first scheduled pod is not the master task")
    invalid = [pod.pod_name for pod in pods if pod.gpu_count <= 0]
    if invalid:
        raise RuntimeError(f"GPU count is unavailable for pods: {','.join(invalid)}; set GPUS_PER_NODE")
    return pods


def message_size_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([KMGT]?)B?\s*", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"invalid message size: {value!r}")
    unit = match.group(2).upper()
    return int(match.group(1)) * 1024 ** {"": 0, "K": 1, "M": 2, "G": 3, "T": 4}[unit]


def message_sizes(min_size: str, max_size: str, factor: int) -> list[int]:
    current = message_size_bytes(min_size)
    maximum = message_size_bytes(max_size)
    if current <= 0 or maximum < current or factor < 2:
        raise ValueError("invalid message-size range")
    sizes: list[int] = []
    while current <= maximum:
        sizes.append(current)
        current *= factor
    if sizes[-1] != maximum:
        raise ValueError("max message size is not reachable using step factor")
    return sizes


def human_size(value: int) -> str:
    for suffix, scale in (("G", 1024**3), ("M", 1024**2), ("K", 1024)):
        if value >= scale and value % scale == 0:
            return f"{value // scale}{suffix}"
    return str(value)


def parse_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def parse_wrong(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def parse_mccl_rows(text: str, op: str, rank_count: int, expected_sizes: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        size = int(fields[0])
        if size not in expected_sizes or len(fields) < 9:
            continue
        tail = fields[-8:]
        out_values = [parse_number(value) for value in tail[:3]]
        in_values = [parse_number(value) for value in tail[4:7]]
        out_wrong = parse_wrong(tail[3])
        in_wrong = parse_wrong(tail[7])
        if any(value is None for value in out_values + in_values):
            continue
        rows.append(
            {
                "op": op,
                "message_size": human_size(size),
                "message_size_bytes": size,
                "rank_count": rank_count,
                "out_of_place_latency_us": out_values[0],
                "out_of_place_algbw_gbps": out_values[1],
                "out_of_place_busbw_gbps": out_values[2],
                "out_of_place_wrong": out_wrong,
                "in_place_latency_us": in_values[0],
                "in_place_algbw_gbps": in_values[1],
                "in_place_busbw_gbps": in_values[2],
                "in_place_wrong": in_wrong,
                "correctness_pass": out_wrong == 0 and in_wrong in {0, None},
            }
        )
    return rows


def detect_signatures(text: str) -> list[str]:
    return [name for name, pattern in ERROR_PATTERNS.items() if pattern.search(text)]


def vcctl_exec(
    args: argparse.Namespace, pod: Pod, remote_command: str, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    command = [args.vcctl_bin, "pod", "exec", pod.pod_name, "-n", args.namespace]
    if pod.container_name:
        command.extend(["-c", pod.container_name])
    command.extend(["--", "bash", "-lc", remote_command])
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)


def run_on_pods(args: argparse.Namespace, pods: list[Pod], remote_command: str, timeout: int) -> list[dict[str, Any]]:
    def invoke(pod: Pod) -> dict[str, Any]:
        try:
            proc = vcctl_exec(args, pod, remote_command, timeout=timeout)
            return {"pod_name": pod.pod_name, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        except subprocess.TimeoutExpired:
            return {"pod_name": pod.pod_name, "returncode": 124, "stdout": "", "stderr": "timeout"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(pods))) as executor:
        return list(executor.map(invoke, pods))


def put_text(args: argparse.Namespace, pod: Pod, remote_path: str, content: str, mode: str) -> None:
    encoded = base64.b64encode(content.encode()).decode()
    parent = str(Path(remote_path).parent)
    command = (
        f"mkdir -p {shlex.quote(parent)} && "
        f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(remote_path)} && "
        f"chmod {shlex.quote(mode)} {shlex.quote(remote_path)}"
    )
    proc = vcctl_exec(args, pod, command, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to write {remote_path} in {pod.pod_name}: {proc.stderr.strip()}")


def inject_authorized_key(args: argparse.Namespace, pod: Pod, public_key: str, marker: str) -> None:
    command = (
        "mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
        "touch /root/.ssh/authorized_keys2 && "
        f"grep -Fv {shlex.quote(marker)} /root/.ssh/authorized_keys2 > /root/.ssh/authorized_keys2.hc.tmp || true; "
        "mv /root/.ssh/authorized_keys2.hc.tmp /root/.ssh/authorized_keys2; "
        f"printf '%s\\n' {shlex.quote(public_key.strip())} >> /root/.ssh/authorized_keys2; "
        "chmod 600 /root/.ssh/authorized_keys2"
    )
    proc = vcctl_exec(args, pod, command, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to authorize key in {pod.pod_name}: {proc.stderr.strip()}")


def remove_authorized_key(args: argparse.Namespace, pod: Pod, marker: str) -> None:
    command = (
        "test ! -f /root/.ssh/authorized_keys2 || { "
        f"grep -Fv {shlex.quote(marker)} /root/.ssh/authorized_keys2 > /root/.ssh/authorized_keys2.hc.tmp || true; "
        "mv /root/.ssh/authorized_keys2.hc.tmp /root/.ssh/authorized_keys2; "
        "chmod 600 /root/.ssh/authorized_keys2; }"
    )
    try:
        vcctl_exec(args, pod, command, timeout=120)
    except Exception:
        pass


def setup_ssh_mesh(
    args: argparse.Namespace, pods: list[Pod], master: Pod, master_tmp: str, local_tmp: Path, marker: str
) -> None:
    key_path = local_tmp / "id_ed25519"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", marker, "-f", str(key_path)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    public_key = key_path.with_suffix(".pub").read_text(encoding="utf-8")
    for pod in pods:
        inject_authorized_key(args, pod, public_key, marker)
    launcher = "\n".join(
        [
            "#!/usr/bin/env bash",
            f"exec /usr/bin/ssh -F /dev/null -p 22 -i {shlex.quote(master_tmp + '/id_ed25519')} "
            "-o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 \"$@\"",
            "",
        ]
    )
    hostfile = "".join(f"{pod.pod_ip}:{pod.gpu_count}\n" for pod in pods)
    private_key = key_path.read_text(encoding="utf-8")
    for pod in pods:
        put_text(args, pod, f"{master_tmp}/id_ed25519", private_key, "600")
        put_text(args, pod, f"{master_tmp}/ssh_launcher.sh", launcher, "700")
    put_text(args, master, f"{master_tmp}/hostfile", hostfile, "600")
    hosts = " ".join(shlex.quote(pod.pod_ip) for pod in pods)
    verify = f"set -e; for host in {hosts}; do {shlex.quote(master_tmp + '/ssh_launcher.sh')} \"$host\" true; done"
    proc = vcctl_exec(args, master, verify, timeout=max(120, len(pods) * 10))
    if proc.returncode != 0:
        raise RuntimeError(f"SSH mesh verification failed: {proc.stderr.strip()}")


def mpi_environment(args: argparse.Namespace) -> dict[str, str]:
    return {
        "MACA_PATH": args.maca_path,
        "LD_LIBRARY_PATH": f"{args.maca_path}/lib:{args.maca_path}/ompi/lib",
        "MCCL_IB_HCA": args.ib_hca,
        "MCCL_IB_GID_INDEX": args.ib_gid_index,
        "MCCL_IB_TC": args.ib_tc,
        "MCCL_ENABLE_VSWITCH": args.enable_vswitch,
        "MCCL_PCIE_BUFFER_MODE": args.pcie_buffer_mode,
        "MCCL_SOCKET_IFNAME": args.socket_ifname,
        "GLOO_SOCKET_IFNAME": args.socket_ifname,
        "MCCL_CROSS_NIC": args.cross_nic,
        "FORCE_ACTIVE_WAIT": args.force_active_wait,
    }


def build_mpi_command(args: argparse.Namespace, pods: list[Pod], op: str, master_tmp: str) -> str:
    binary = f"{args.test_bin_dir.rstrip('/')}/{OP_BINARIES[op]}"
    rank_count = sum(pod.gpu_count for pod in pods)
    host_spec = ",".join(f"{pod.pod_ip}:{pod.gpu_count}" for pod in pods)
    environment = mpi_environment(args)
    command = [
        args.mpi_bin,
        "--allow-run-as-root",
        "-np",
        str(rank_count),
        "-host",
        host_spec,
        "-mca",
        "plm_rsh_agent",
        f"{master_tmp}/ssh_launcher.sh",
        "-mca",
        "btl_tcp_if_include",
        args.socket_ifname,
        "-mca",
        "oob_tcp_if_include",
        args.socket_ifname,
        "-mca",
        "pml",
        "^ucx",
        "-mca",
        "osc",
        "^ucx",
        "-mca",
        "btl",
        "^openib",
    ]
    for key in environment:
        command.extend(["-x", key])
    command.extend(
        [
            binary,
            "-b",
            args.min_message_size,
            "-e",
            args.max_message_size,
            "-f",
            str(args.step_factor),
            "-d",
            args.dtype,
            "-g",
            "1",
            "-w",
            str(args.warmup),
            "-n",
            str(args.iters),
            "-c",
            "1",
            "-a",
            "1",
        ]
    )
    preamble = ["set -eo pipefail"]
    preamble.extend(f"export {key}={shlex.quote(value)}" for key, value in environment.items())
    preamble.extend([f"test -x {shlex.quote(binary)}", "exec " + shlex.join(command)])
    return "\n".join(preamble)


def idle_preflight(args: argparse.Namespace, pods: list[Pod]) -> None:
    names = "|".join(f"[{name[0]}]{name[1:]}" for name in OP_BINARIES.values())
    command = f"ps -eo pid=,args= | grep -E '[m]pirun|{names}|[t]orchrun|[p]retrain_healthcheck.cli' || true"
    rows = run_on_pods(args, pods, command, timeout=30)
    failed = [row for row in rows if row["returncode"] != 0]
    busy = [row for row in rows if row["stdout"].strip()]
    if failed or busy:
        details = [f"{row['pod_name']}:rc={row['returncode']}:{row['stderr'].strip()}" for row in failed]
        details.extend(f"{row['pod_name']}:{row['stdout'].strip()}" for row in busy)
        raise RuntimeError("MCCL idle preflight failed:\n" + "\n".join(details))
    binary_checks = " && ".join(
        f"test -x {shlex.quote(args.test_bin_dir.rstrip('/') + '/' + binary)}" for binary in OP_BINARIES.values()
    )
    rows = run_on_pods(args, pods, f"test -x {shlex.quote(args.mpi_bin)} && {binary_checks}", timeout=30)
    failed = [row for row in rows if row["returncode"] != 0]
    if failed:
        raise RuntimeError("MCCL binary preflight failed: " + ",".join(row["pod_name"] for row in failed))


def cleanup_residual_processes(args: argparse.Namespace, pods: list[Pod]) -> str:
    command = (
        "pids=$(ps -eo pid=,args= | awk '/[m]pirun|[m]ccl_perf\\// {print $1}'); "
        "if [ -n \"$pids\" ]; then kill -TERM $pids 2>/dev/null || true; sleep 5; "
        "kill -KILL $pids 2>/dev/null || true; fi"
    )
    rows = run_on_pods(args, pods, command, timeout=30)
    return "PASS" if all(row["returncode"] == 0 for row in rows) else "FAIL"


def kill_all_device_processes(args: argparse.Namespace, pods: list[Pod]) -> None:
    if not args.metax_kill_all_process_before_op:
        return
    command = f"command -v mx-smi >/dev/null && mx-smi --kill-all-process && sleep {args.kill_all_process_wait_seconds}"
    rows = run_on_pods(args, pods, command, timeout=max(60, args.kill_all_process_wait_seconds + 30))
    failed = [row for row in rows if row["returncode"] != 0]
    if failed:
        raise RuntimeError("mx-smi --kill-all-process failed: " + ",".join(row["pod_name"] for row in failed))


def operation_status(
    rows: list[dict[str, Any]], expected_sizes: set[int], returncode: int | None, timeout: bool
) -> tuple[str, str]:
    sizes = {int(row["message_size_bytes"]) for row in rows}
    if timeout:
        return "FAIL", "TIMEOUT"
    if returncode != 0:
        return "FAIL", "EXEC_FAILED"
    if sizes != expected_sizes:
        return "FAIL", "RESULT_MISSING"
    if not all(bool(row["correctness_pass"]) for row in rows):
        return "FAIL", "CORRECTNESS_FAILED"
    return "PASS", ""


def run_operation(
    args: argparse.Namespace,
    pods: list[Pod],
    master: Pod,
    op: str,
    master_tmp: str,
    raw_dir: Path,
    expected_sizes: set[int],
) -> tuple[OperationResult, list[dict[str, Any]], list[dict[str, Any]]]:
    kill_all_device_processes(args, pods)
    remote_command = build_mpi_command(args, pods, op, master_tmp)
    stdout_path = raw_dir / f"{op}.stdout"
    stderr_path = raw_dir / f"{op}.stderr"
    started = time.monotonic()
    timeout = False
    returncode: int | None = None
    stdout = ""
    stderr = ""
    try:
        proc = vcctl_exec(args, master, remote_command, timeout=args.op_timeout_seconds)
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timeout = True
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
    elapsed = time.monotonic() - started
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    rank_count = sum(pod.gpu_count for pod in pods)
    rows = parse_mccl_rows(stdout, op, rank_count, expected_sizes)
    status, error_type = operation_status(rows, expected_sizes, returncode, timeout)
    signatures = detect_signatures(stdout + "\n" + stderr)
    cleanup_status = "NOT_NEEDED"
    if status != "PASS":
        cleanup_status = cleanup_residual_processes(args, pods)
    result = OperationResult(
        op=op,
        binary=OP_BINARIES[op],
        status=status,
        returncode=returncode,
        timeout=timeout,
        error_type=error_type,
        signatures=signatures,
        row_count=len({row["message_size_bytes"] for row in rows}),
        expected_row_count=len(expected_sizes),
        correctness_pass=bool(rows) and all(bool(row["correctness_pass"]) for row in rows),
        elapsed_seconds=elapsed,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        command=remote_command,
        cleanup_status=cleanup_status,
    )
    failure_rows = [
        {"op": op, "signature": signature, "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
        for signature in signatures
    ]
    return result, rows, failure_rows


def write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    (output_dir / "collective_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    fields = [
        "op",
        "message_size",
        "message_size_bytes",
        "rank_count",
        "out_of_place_latency_us",
        "out_of_place_algbw_gbps",
        "out_of_place_busbw_gbps",
        "out_of_place_wrong",
        "in_place_latency_us",
        "in_place_algbw_gbps",
        "in_place_busbw_gbps",
        "in_place_wrong",
        "correctness_pass",
    ]
    with (output_dir / "collective_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_parameters(output_dir: Path, args: argparse.Namespace) -> None:
    values = {
        "JOB_NAME": args.job_name,
        "NAMESPACE": args.namespace,
        "RUN_ID": args.run_id,
        "GPUS_PER_NODE": args.gpus_per_node,
        "DTYPE": args.dtype,
        "MIN_MESSAGE_SIZE": args.min_message_size,
        "MAX_MESSAGE_SIZE": args.max_message_size,
        "STEP_FACTOR": args.step_factor,
        "WARMUP": args.warmup,
        "ITERS": args.iters,
        "COLLECTIVE_OPS": ",".join(args.ops),
        "OP_TIMEOUT_SECONDS": args.op_timeout_seconds,
        "CONTINUE_ON_FAILURE": int(args.continue_on_failure),
        "METAX_KILL_ALL_PROCESS_BEFORE_OP": int(args.metax_kill_all_process_before_op),
        "DRY_RUN": int(args.dry_run),
    }
    values.update(mpi_environment(args))
    (output_dir / "parameters.env").write_text(
        "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()), encoding="utf-8"
    )


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    pods: list[Pod],
    results: list[OperationResult],
    rows: list[dict[str, Any]],
    started_at: str,
    elapsed: float,
) -> dict[str, Any]:
    overall = "PASS" if len(results) == len(args.ops) and all(result.status == "PASS" for result in results) else "FAIL"
    rank_count = sum(pod.gpu_count for pod in pods)
    expected_case_count = len(args.ops) * len(args.expected_sizes)
    summary = {
        "run_id": args.run_id,
        "job_name": args.job_name,
        "status": overall,
        "started_at": started_at,
        "finished_at": iso_now(),
        "elapsed_seconds": elapsed,
        "node_count": len(pods),
        "rank_count": rank_count,
        "ordered_node_names_sha256": hashlib.sha256("\n".join(pod.node_name for pod in pods).encode()).hexdigest(),
        "workload": {
            "ops": args.ops,
            "min_message_size": args.min_message_size,
            "max_message_size": args.max_message_size,
            "step_factor": args.step_factor,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "correctness": True,
            "alltoallv_semantics": "vendor_builtin",
            "metax_kill_all_process_before_op": args.metax_kill_all_process_before_op,
        },
        "expected_case_count": expected_case_count,
        "result_case_count": len(rows),
        "correctness_pass_count": sum(bool(row["correctness_pass"]) for row in rows),
        "operations": [asdict(result) for result in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# MetaX Official MCCL Multi-Node Collective Sweep",
        "",
        f"- Status: **{overall}**",
        f"- Job: `{args.job_name}`",
        f"- Run ID: `{args.run_id}`",
        f"- Scale: {len(pods)} nodes, {rank_count} GPU ranks",
        f"- Range: {args.min_message_size}..{args.max_message_size}, factor={args.step_factor}",
        f"- Warmup/Iters: {args.warmup}/{args.iters}",
        f"- Cases: {len(rows)}/{expected_case_count}",
        f"- Correctness PASS: {summary['correctness_pass_count']}/{expected_case_count}",
        f"- Wall time: {elapsed:.3f}s",
        "- All-to-AllV semantics: vendor built-in traffic; not the Healthcheck EP/MoE pattern matrix.",
        "",
        "## Nodes",
        "",
        "| Pod | Node | Pod IP | GPU ranks |",
        "| --- | --- | --- | ---: |",
    ]
    for pod in pods:
        lines.append(f"| `{pod.pod_name}` | `{pod.node_name}` | `{pod.pod_ip}` | {pod.gpu_count} |")
    lines.extend(
        [
            "",
            "## Operations",
            "",
            "| Op | Status | Cases | Correctness | Elapsed (s) | Error | Signatures |",
            "| --- | --- | ---: | --- | ---: | --- | --- |",
        ]
    )
    for result in results:
        lines.append(
            f"| `{result.op}` | {result.status} | {result.row_count}/{result.expected_row_count} | "
            f"{'PASS' if result.correctness_pass else 'FAIL'} | {result.elapsed_seconds:.3f} | "
            f"{result.error_type or '-'} | {', '.join(result.signatures) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Large-Message Results",
            "",
            "| Op | Size | Out-of-place BusBW (GB/s) | In-place BusBW (GB/s) | Correctness |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        if int(row["message_size_bytes"]) >= 1024**3:
            lines.append(
                f"| `{row['op']}` | {row['message_size']} | {row['out_of_place_busbw_gbps']:.3f} | "
                f"{row['in_place_busbw_gbps']:.3f} | {'PASS' if row['correctness_pass'] else 'FAIL'} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MetaX official MCCL collective tests across all vcctl job pods")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--vcctl-bin", default="vcctl")
    parser.add_argument("--container-name", default="")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--maca-path", default="/opt/maca")
    parser.add_argument("--mpi-bin", required=True)
    parser.add_argument("--test-bin-dir", required=True)
    parser.add_argument("--gpus-per-node", type=int, default=0)
    parser.add_argument("--dtype", default="float")
    parser.add_argument("--min-message-size", default="1K")
    parser.add_argument("--max-message-size", default="2G")
    parser.add_argument("--step-factor", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--socket-ifname", default="eth0")
    parser.add_argument("--ib-hca", default="xscale_0,xscale_1,xscale_2,xscale_3")
    parser.add_argument("--ib-gid-index", default="5")
    parser.add_argument("--ib-tc", default="128")
    parser.add_argument("--enable-vswitch", default="1")
    parser.add_argument("--pcie-buffer-mode", default="0")
    parser.add_argument("--cross-nic", default="1")
    parser.add_argument("--force-active-wait", default="2")
    parser.add_argument("--ops", default=",".join(OP_BINARIES))
    parser.add_argument("--op-timeout-seconds", type=int, default=600)
    parser.add_argument("--continue-on-failure", type=int, choices=(0, 1), default=1)
    parser.add_argument("--metax-kill-all-process-before-op", type=int, choices=(0, 1), default=0)
    parser.add_argument("--allow-kill-all-process", type=int, choices=(0, 1), default=0)
    parser.add_argument("--kill-all-process-wait-seconds", type=int, default=5)
    parser.add_argument("--dry-run", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    args.dry_run = bool(args.dry_run)
    args.continue_on_failure = bool(args.continue_on_failure)
    args.metax_kill_all_process_before_op = bool(args.metax_kill_all_process_before_op)
    args.allow_kill_all_process = bool(args.allow_kill_all_process)
    args.ops = [item.strip() for item in args.ops.split(",") if item.strip()]
    unknown = sorted(set(args.ops) - set(OP_BINARIES))
    if unknown:
        parser.error(f"unsupported ops: {','.join(unknown)}")
    if len(args.ops) != len(set(args.ops)):
        parser.error("--ops contains duplicates")
    if args.gpus_per_node < 0:
        parser.error("--gpus-per-node must be >= 0")
    if args.op_timeout_seconds <= 0:
        parser.error("--op-timeout-seconds must be positive")
    if args.metax_kill_all_process_before_op and not args.allow_kill_all_process:
        parser.error("--metax-kill-all-process-before-op requires --allow-kill-all-process=1")
    args.expected_sizes = message_sizes(args.min_message_size, args.max_message_size, args.step_factor)
    return args


def main() -> int:
    args = parse_args()
    started_at = iso_now()
    started = time.monotonic()
    output_dir = args.result_root / args.run_id
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    write_parameters(output_dir, args)
    pods = load_pods(args)
    master = pods[0]
    rank_count = sum(pod.gpu_count for pod in pods)
    hostfile = "".join(f"{pod.pod_ip}:{pod.gpu_count}\n" for pod in pods)
    (output_dir / "hostfile").write_text(hostfile, encoding="utf-8")
    (output_dir / "pods.jsonl").write_text(
        "".join(json.dumps(asdict(pod), sort_keys=True) + "\n" for pod in pods), encoding="utf-8"
    )
    master_tmp = f"/tmp/pretrain_healthcheck_mccl_official/{args.run_id}"
    commands = [{"op": op, "command": build_mpi_command(args, pods, op, master_tmp)} for op in args.ops]
    (output_dir / "commands.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in commands), encoding="utf-8"
    )
    print(f"[mccl-multi-node] pods={len(pods)} ranks={rank_count} master={master.pod_name}")
    if args.dry_run:
        print("[mccl-multi-node] overall_status=DRY_RUN")
        print(f"[mccl-multi-node] output={output_dir}")
        return 0

    idle_preflight(args, pods)
    marker = f"pretrain-healthcheck-mccl-{args.run_id}"
    local_tmp = Path(tempfile.mkdtemp(prefix=f"mccl_official_{args.run_id}_", dir="/tmp"))
    results: list[OperationResult] = []
    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    try:
        print("[mccl-multi-node] setting up temporary SSH mesh")
        setup_ssh_mesh(args, pods, master, master_tmp, local_tmp, marker)
        print("[mccl-multi-node] SSH mesh verified")
        expected_sizes = set(args.expected_sizes)
        for op in args.ops:
            print(f"[mccl-multi-node] op_start={op}", flush=True)
            result, op_rows, op_failures = run_operation(
                args, pods, master, op, master_tmp, raw_dir, expected_sizes
            )
            results.append(result)
            rows.extend(op_rows)
            failure_rows.extend(op_failures)
            print(
                f"[mccl-multi-node] op_done={op} status={result.status} "
                f"cases={result.row_count}/{result.expected_row_count} elapsed={result.elapsed_seconds:.3f}s "
                f"signatures={','.join(result.signatures) or '-'}",
                flush=True,
            )
            if result.status != "PASS" and not args.continue_on_failure:
                break
    finally:
        print("[mccl-multi-node] cleaning temporary SSH mesh")
        for pod in pods:
            remove_authorized_key(args, pod, marker)
        run_on_pods(args, pods, f"rm -rf {shlex.quote(master_tmp)}", timeout=120)
        shutil.rmtree(local_tmp, ignore_errors=True)

    write_rows(output_dir, rows)
    (output_dir / "operation_results.jsonl").write_text(
        "".join(json.dumps(asdict(result), sort_keys=True) + "\n" for result in results), encoding="utf-8"
    )
    (output_dir / "failure_signatures.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failure_rows), encoding="utf-8"
    )
    summary = write_summary(output_dir, args, pods, results, rows, started_at, time.monotonic() - started)
    print(f"[mccl-multi-node] overall_status={summary['status']}")
    print(f"[mccl-multi-node] output={output_dir}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
