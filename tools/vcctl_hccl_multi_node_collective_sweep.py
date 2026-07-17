#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
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
    "all_reduce": "all_reduce_test",
    "reduce_scatter": "reduce_scatter_test",
    "all_gather": "all_gather_test",
    "broadcast": "broadcast_test",
    "all_to_all": "alltoall_test",
    "all_to_allv": "alltoallv_test",
}

HCCL_ROW_RE = re.compile(
    r"^\s*(?P<size>\d+)\s*\|\s*"
    r"(?P<latency>[0-9]+(?:\.[0-9]+)?)\s*\|\s*"
    r"(?P<algbw>[0-9]+(?:\.[0-9]+)?)\s*\|\s*"
    r"(?P<check>[A-Za-z_]+)\s*$"
)


@dataclass(frozen=True)
class Pod:
    pod_name: str
    container_name: str
    task_spec: str
    node_name: str
    host_ip: str
    pod_ip: str
    dns_name: str


@dataclass
class OperationResult:
    op: str
    binary: str
    status: str
    returncode: int | None
    timeout: bool
    error_type: str
    row_count: int
    expected_row_count: int
    correctness_pass: bool
    correctness_status: str
    elapsed_seconds: float
    stdout_path: str
    stderr_path: str
    command: str


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
        if not pod_name or not node_name:
            continue
        container = choose_container(raw, args.container_name)
        pods.append(
            Pod(
                pod_name=pod_name,
                container_name=str(container.get("name", "")),
                task_spec=str(nested(raw, "metadata", "labels", "volcano.sh/task-spec")),
                node_name=node_name,
                host_ip=str(nested(raw, "status", "hostIP")),
                pod_ip=str(nested(raw, "status", "podIP")),
                dns_name=f"{pod_name}.{args.job_name}",
            )
        )
    pods.sort(key=pod_sort_key)
    if not pods:
        raise RuntimeError(f"no scheduled pods found for job {args.job_name!r}")
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


def busbw_factor(op: str, rank_count: int) -> float:
    if op == "all_reduce":
        return 2 * max(0, rank_count - 1) / max(1, rank_count)
    if op in {"reduce_scatter", "all_gather", "all_to_all", "all_to_allv"}:
        return max(0, rank_count - 1) / max(1, rank_count)
    return 1.0


def parse_hccl_rows(text: str, op: str, rank_count: int, expected_sizes: set[int]) -> list[dict[str, Any]]:
    factor = busbw_factor(op, rank_count)
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = HCCL_ROW_RE.match(line)
        if not match:
            continue
        size = int(match.group("size"))
        if size not in expected_sizes:
            continue
        algbw = float(match.group("algbw"))
        rows.append(
            {
                "op": op,
                "message_size": human_size(size),
                "message_size_bytes": size,
                "rank_count": rank_count,
                "latency_us": float(match.group("latency")),
                "algbw_gbps": algbw,
                "busbw_factor": factor,
                "busbw_gbps": algbw * factor,
                "check": match.group("check").lower(),
            }
        )
    return rows


def vcctl_exec(args: argparse.Namespace, pod: Pod, remote_command: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    command = [args.vcctl_bin, "pod", "exec", pod.pod_name, "-n", args.namespace]
    if pod.container_name:
        command.extend(["-c", pod.container_name])
    command.extend(["--", "bash", "-lc", remote_command])
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)


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
    line = public_key.strip()
    command = (
        "mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
        "touch /root/.ssh/authorized_keys2 && "
        f"grep -Fv {shlex.quote(marker)} /root/.ssh/authorized_keys2 > /root/.ssh/authorized_keys2.hc.tmp || true; "
        "mv /root/.ssh/authorized_keys2.hc.tmp /root/.ssh/authorized_keys2; "
        f"printf '%s\\n' {shlex.quote(line)} >> /root/.ssh/authorized_keys2; "
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


def build_mpi_command(args: argparse.Namespace, op: str, master_tmp: str, rank_count: int) -> str:
    binary = f"{args.test_bin_dir.rstrip('/')}/{OP_BINARIES[op]}"
    command = [
        args.mpi_bin,
        "-f",
        f"{master_tmp}/hostfile",
        "-n",
        str(rank_count),
        "-launcher",
        "ssh",
        "-launcher-exec",
        f"{master_tmp}/ssh_launcher.sh",
        "-genvlist",
        "LD_LIBRARY_PATH,PYTHONPATH,ASCEND_AICPU_PATH,ASCEND_HOME_PATH,HCCL_SOCKET_IFNAME",
        binary,
        "-b",
        args.min_message_size,
        "-e",
        args.max_message_size,
        "-f",
        str(args.step_factor),
        "-d",
        args.dtype,
        "-o",
        "sum",
        "-w",
        str(args.warmup),
        "-n",
        str(args.iters),
        "-p",
        str(args.npus_per_node),
        "-c",
        "1",
    ]
    preamble = [
        "set -eo pipefail",
        f"source {shlex.quote(args.ascend_env_script)} >/dev/null 2>&1",
        f"export LD_LIBRARY_PATH={shlex.quote(args.mpi_lib_dir)}:\"${{LD_LIBRARY_PATH:-}}\"",
        f"export HCCL_SOCKET_IFNAME={shlex.quote(args.socket_ifname)}",
        "unset CPU_AFFINITY_CONF HCCL_DETERMINISTIC HCCL_OP_EXPANSION_MODE HCCL_BUFFSIZE HCCL_INTRA_ROCE_ENABLE",
        f"test -x {shlex.quote(binary)}",
        "exec " + shlex.join(command),
    ]
    return "\n".join(preamble)


def setup_ssh_mesh(args: argparse.Namespace, pods: list[Pod], master: Pod, master_tmp: str, local_tmp: Path, marker: str) -> None:
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
    hostfile = "".join(f"{pod.dns_name}:{args.npus_per_node}\n" for pod in pods)
    put_text(args, master, f"{master_tmp}/id_ed25519", key_path.read_text(encoding="utf-8"), "600")
    put_text(args, master, f"{master_tmp}/ssh_launcher.sh", launcher, "700")
    put_text(args, master, f"{master_tmp}/hostfile", hostfile, "600")
    hosts = " ".join(shlex.quote(pod.dns_name) for pod in pods)
    verify = f"set -e; for host in {hosts}; do {shlex.quote(master_tmp + '/ssh_launcher.sh')} \"$host\" true; done"
    proc = vcctl_exec(args, master, verify, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"SSH mesh verification failed: {proc.stderr.strip()}")


def correctness_status(rows: list[dict[str, Any]], overflow_warning: bool) -> str:
    checks = {str(row["check"]) for row in rows}
    if checks == {"success"}:
        return "PASS"
    if checks <= {"success", "null"} and "null" in checks and overflow_warning:
        return "SKIPPED_OVERFLOW"
    return "FAIL"


def operation_status(
    rows: list[dict[str, Any]],
    expected_sizes: set[int],
    returncode: int | None,
    timeout: bool,
    overflow_warning: bool = False,
) -> tuple[str, str]:
    sizes = {int(row["message_size_bytes"]) for row in rows}
    if timeout:
        return "FAIL", "TIMEOUT"
    if returncode != 0:
        return "FAIL", "EXEC_FAILED"
    if sizes != expected_sizes:
        return "FAIL", "RESULT_MISSING"
    if correctness_status(rows, overflow_warning) == "FAIL":
        return "FAIL", "CORRECTNESS_FAILED"
    return "PASS", ""


def run_operation(
    args: argparse.Namespace,
    master: Pod,
    op: str,
    master_tmp: str,
    raw_dir: Path,
    rank_count: int,
    expected_sizes: set[int],
) -> tuple[OperationResult, list[dict[str, Any]]]:
    remote_command = build_mpi_command(args, op, master_tmp, rank_count)
    stdout_path = raw_dir / f"{op}.stdout"
    stderr_path = raw_dir / f"{op}.stderr"
    started = time.monotonic()
    timeout = False
    returncode: int | None = None
    stdout = ""
    stderr = ""
    try:
        proc = vcctl_exec(args, master, remote_command, timeout=args.exec_timeout_seconds)
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
    rows = parse_hccl_rows(stdout, op, rank_count, expected_sizes)
    overflow_warning = "calculation result overflows" in stdout.lower()
    check_status = correctness_status(rows, overflow_warning)
    status, error_type = operation_status(rows, expected_sizes, returncode, timeout, overflow_warning)
    result = OperationResult(
        op=op,
        binary=OP_BINARIES[op],
        status=status,
        returncode=returncode,
        timeout=timeout,
        error_type=error_type,
        row_count=len({row["message_size_bytes"] for row in rows}),
        expected_row_count=len(expected_sizes),
        correctness_pass=bool(rows) and all(row["check"] == "success" for row in rows),
        correctness_status=check_status,
        elapsed_seconds=elapsed,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        command=remote_command,
    )
    return result, rows


def write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    (output_dir / "collective_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    fields = [
        "op",
        "message_size",
        "message_size_bytes",
        "rank_count",
        "latency_us",
        "algbw_gbps",
        "busbw_factor",
        "busbw_gbps",
        "check",
    ]
    with (output_dir / "collective_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_dir: Path, args: argparse.Namespace, pods: list[Pod], results: list[OperationResult], rows: list[dict[str, Any]], started_at: str, elapsed: float) -> dict[str, Any]:
    overall = "PASS" if results and all(result.status == "PASS" for result in results) else "FAIL"
    summary = {
        "run_id": args.run_id,
        "job_name": args.job_name,
        "status": overall,
        "started_at": started_at,
        "finished_at": iso_now(),
        "elapsed_seconds": elapsed,
        "node_count": len(pods),
        "rank_count": len(pods) * args.npus_per_node,
        "workload": {
            "ops": args.ops,
            "min_message_size": args.min_message_size,
            "max_message_size": args.max_message_size,
            "step_factor": args.step_factor,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "correctness": True,
            "alltoallv_semantics": "official_builtin",
            "tuning_overrides": False,
        },
        "expected_case_count": len(args.ops) * len(args.expected_sizes),
        "result_case_count": len(rows),
        "correctness_pass_count": sum(row["check"] == "success" for row in rows),
        "correctness_skipped_overflow_count": sum(row["check"] == "null" for row in rows),
        "operations": [asdict(result) for result in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Huawei Official HCCL Multi-Node Collective Sweep",
        "",
        f"- Status: **{overall}**",
        f"- Job: `{args.job_name}`",
        f"- Run ID: `{args.run_id}`",
        f"- Scale: {len(pods)} nodes, {summary['rank_count']} NPU ranks",
        f"- Range: {args.min_message_size}..{args.max_message_size}, factor={args.step_factor}",
        f"- Warmup/Iters: {args.warmup}/{args.iters}",
        f"- Cases: {len(rows)}/{summary['expected_case_count']}",
        f"- Correctness PASS: {summary['correctness_pass_count']}/{summary['expected_case_count']}",
        f"- Correctness skipped due to official SUM overflow guard: {summary['correctness_skipped_overflow_count']}",
        f"- Wall time: {elapsed:.3f}s",
        "- All-to-AllV semantics: official built-in traffic; not the EP8 five-pattern workload.",
        "- Tuning overrides: disabled.",
        "",
        "## Operations",
        "",
        "| Op | Status | Cases | Correctness | Elapsed (s) | Error |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| `{result.op}` | {result.status} | {result.row_count}/{result.expected_row_count} | "
            f"{result.correctness_status} | {result.elapsed_seconds:.3f} | {result.error_type or '-'} |"
        )
    lines.extend(["", "## Large-Message Results", "", "| Op | Size | Latency (us) | AlgBW (GB/s) | BusBW (GB/s) | Check |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for row in rows:
        if int(row["message_size_bytes"]) >= 1024**3:
            lines.append(
                f"| `{row['op']}` | {row['message_size']} | {row['latency_us']:.3f} | "
                f"{row['algbw_gbps']:.3f} | {row['busbw_gbps']:.3f} | {row['check']} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_parameters(output_dir: Path, args: argparse.Namespace) -> None:
    values = {
        "JOB_NAME": args.job_name,
        "NAMESPACE": args.namespace,
        "RUN_ID": args.run_id,
        "NPUS_PER_NODE": args.npus_per_node,
        "DTYPE": args.dtype,
        "MIN_MESSAGE_SIZE": args.min_message_size,
        "MAX_MESSAGE_SIZE": args.max_message_size,
        "STEP_FACTOR": args.step_factor,
        "WARMUP": args.warmup,
        "ITERS": args.iters,
        "COLLECTIVE_OPS": ",".join(args.ops),
        "HCCL_SOCKET_IFNAME": args.socket_ifname,
        "EXEC_TIMEOUT_SECONDS": args.exec_timeout_seconds,
        "DRY_RUN": int(args.dry_run),
    }
    (output_dir / "parameters.env").write_text(
        "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Huawei official HCCL collective tests across all vcctl job pods")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--vcctl-bin", default="vcctl")
    parser.add_argument("--container-name", default="")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ascend-env-script", required=True)
    parser.add_argument("--mpi-bin", required=True)
    parser.add_argument("--mpi-lib-dir", required=True)
    parser.add_argument("--test-bin-dir", required=True)
    parser.add_argument("--npus-per-node", type=int, default=16)
    parser.add_argument("--dtype", default="bfp16")
    parser.add_argument("--min-message-size", default="1K")
    parser.add_argument("--max-message-size", default="8G")
    parser.add_argument("--step-factor", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--socket-ifname", default="eth0")
    parser.add_argument("--ops", default=",".join(OP_BINARIES))
    parser.add_argument("--exec-timeout-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    args.dry_run = bool(args.dry_run)
    args.ops = [item.strip() for item in args.ops.split(",") if item.strip()]
    unknown = sorted(set(args.ops) - set(OP_BINARIES))
    if unknown:
        parser.error(f"unsupported ops: {','.join(unknown)}")
    if len(args.ops) != len(set(args.ops)):
        parser.error("--ops contains duplicates")
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
    if master.task_spec != "master":
        raise RuntimeError("first scheduled pod is not the master task")
    rank_count = len(pods) * args.npus_per_node
    hostfile = "".join(f"{pod.dns_name}:{args.npus_per_node}\n" for pod in pods)
    (output_dir / "hostfile").write_text(hostfile, encoding="utf-8")
    (output_dir / "pods.jsonl").write_text(
        "".join(json.dumps(asdict(pod), sort_keys=True) + "\n" for pod in pods), encoding="utf-8"
    )
    master_tmp = f"/tmp/pretrain_healthcheck_hccl_official/{args.run_id}"
    commands = [
        {"op": op, "command": build_mpi_command(args, op, master_tmp, rank_count)} for op in args.ops
    ]
    (output_dir / "commands.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in commands), encoding="utf-8"
    )
    print(f"[hccl-multi-node] pods={len(pods)} ranks={rank_count} master={master.pod_name}")
    if args.dry_run:
        print("[hccl-multi-node] overall_status=DRY_RUN")
        print(f"[hccl-multi-node] output={output_dir}")
        return 0

    marker = f"pretrain-healthcheck-hccl-{args.run_id}"
    local_tmp = Path(tempfile.mkdtemp(prefix=f"hccl_official_{args.run_id}_", dir="/tmp"))
    results: list[OperationResult] = []
    rows: list[dict[str, Any]] = []
    try:
        print("[hccl-multi-node] setting up temporary SSH mesh")
        setup_ssh_mesh(args, pods, master, master_tmp, local_tmp, marker)
        print("[hccl-multi-node] SSH mesh verified")
        expected_sizes = set(args.expected_sizes)
        for op in args.ops:
            print(f"[hccl-multi-node] op_start={op}")
            result, op_rows = run_operation(args, master, op, master_tmp, raw_dir, rank_count, expected_sizes)
            results.append(result)
            rows.extend(op_rows)
            print(
                f"[hccl-multi-node] op_done={op} status={result.status} "
                f"cases={result.row_count}/{result.expected_row_count} elapsed={result.elapsed_seconds:.3f}s"
            )
            if result.status != "PASS":
                break
    finally:
        print("[hccl-multi-node] cleaning temporary SSH mesh")
        for pod in pods:
            remove_authorized_key(args, pod, marker)
        try:
            vcctl_exec(args, master, f"rm -rf {shlex.quote(master_tmp)}", timeout=120)
        except Exception:
            pass
        shutil.rmtree(local_tmp, ignore_errors=True)

    write_rows(output_dir, rows)
    (output_dir / "operation_results.jsonl").write_text(
        "".join(json.dumps(asdict(result), sort_keys=True) + "\n" for result in results), encoding="utf-8"
    )
    summary = write_summary(output_dir, args, pods, results, rows, started_at, time.monotonic() - started)
    print(f"[hccl-multi-node] overall_status={summary['status']}")
    print(f"[hccl-multi-node] output={output_dir}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
