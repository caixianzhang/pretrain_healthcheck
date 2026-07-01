#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PodInfo:
    pod_name: str
    namespace: str
    container_name: str
    task_spec: str
    rank: str
    world_size: str
    master_addr: str
    master_port: str
    pod_ip: str
    host_ip: str
    node_name: str
    phase: str
    ready: bool
    restart_count: int


@dataclass
class ExecResult:
    pod_name: str
    container_name: str
    mode: str
    node_name: str
    pod_ip: str
    command: str
    returncode: int | None
    timeout: bool
    status: str
    reason: str
    stdout_path: str
    stderr_path: str
    started_at: str
    finished_at: str
    elapsed_seconds: float


@dataclass
class RunningExec:
    pod: PodInfo
    mode: str
    command: str
    process: subprocess.Popen[str]
    stdout_file: Any
    stderr_file: Any
    stdout_path: Path
    stderr_path: Path
    started_at: str
    start_time: float


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_json_stream(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    objs: list[dict[str, Any]] = []
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
            objs.extend(x for x in obj["items"] if isinstance(x, dict))
        elif isinstance(obj, dict):
            objs.append(obj)
        else:
            raise ValueError(f"unsupported JSON value at offset {idx}: {type(obj).__name__}")
        idx = next_idx
    return objs


def get_nested(obj: dict[str, Any], path: list[str], default: Any = "") -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def get_env(container: dict[str, Any], name: str) -> str:
    for item in container.get("env", []) or []:
        if item.get("name") == name:
            return str(item.get("value", ""))
    return ""


def container_ready_status(pod: dict[str, Any], container_name: str) -> tuple[bool, int]:
    statuses = get_nested(pod, ["status", "containerStatuses"], [])
    ready = False
    restart_count = 0
    if isinstance(statuses, list):
        for status in statuses:
            if status.get("name") == container_name:
                ready = bool(status.get("ready", False))
                restart_count = int(status.get("restartCount", 0))
                break
    return ready, restart_count


def choose_container(pod: dict[str, Any], forced: str) -> dict[str, Any]:
    containers = get_nested(pod, ["spec", "containers"], [])
    if not isinstance(containers, list) or not containers:
        raise ValueError(f"pod {get_nested(pod, ['metadata', 'name'])} has no containers")
    if forced:
        for container in containers:
            if container.get("name") == forced:
                return container
        raise ValueError(f"container {forced!r} not found in pod {get_nested(pod, ['metadata', 'name'])}")
    if len(containers) == 1:
        return containers[0]
    task_spec = str(get_nested(pod, ["metadata", "labels", "volcano.sh/task-spec"], ""))
    for container in containers:
        if container.get("name") == task_spec:
            return container
    return containers[0]


def pod_to_info(pod: dict[str, Any], forced_container: str) -> PodInfo:
    metadata = pod.get("metadata", {}) or {}
    labels = metadata.get("labels", {}) or {}
    container = choose_container(pod, forced_container)
    container_name = str(container.get("name", ""))
    ready, restart_count = container_ready_status(pod, container_name)
    return PodInfo(
        pod_name=str(metadata.get("name", "")),
        namespace=str(metadata.get("namespace", "default")),
        container_name=container_name,
        task_spec=str(labels.get("volcano.sh/task-spec", "")),
        rank=get_env(container, "RANK"),
        world_size=get_env(container, "WORLD_SIZE"),
        master_addr=get_env(container, "MASTER_ADDR"),
        master_port=get_env(container, "MASTER_PORT"),
        pod_ip=str(get_nested(pod, ["status", "podIP"], "")),
        host_ip=str(get_nested(pod, ["status", "hostIP"], "")),
        node_name=str(get_nested(pod, ["spec", "nodeName"], "")),
        phase=str(get_nested(pod, ["status", "phase"], "")),
        ready=ready,
        restart_count=restart_count,
    )


def run_capture(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)


def run_capture_to_files(
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[subprocess.Popen[str], Any, Any]:
    stdout_file = stdout_path.open("w", encoding="utf-8")
    stderr_file = stderr_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
    except Exception:
        stdout_file.close()
        stderr_file.close()
        raise
    return proc, stdout_file, stderr_file


def timeout_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def choose_master_pod(pods: list[PodInfo]) -> PodInfo:
    for pod in pods:
        if pod.rank == "0":
            return pod
    for pod in pods:
        if pod.task_spec == "master":
            return pod
    return pods[0]


def port_available_on_pod(pod: PodInfo, port: int, args: argparse.Namespace) -> bool:
    if args.dry_run or args.pod_json_file:
        return True
    code = (
        "import socket,sys;"
        "p=int(sys.argv[1]);"
        "s=socket.socket(socket.AF_INET, socket.SOCK_STREAM);"
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
        "s.bind(('0.0.0.0', p));"
        "s.close()"
    )
    cmd = [
        args.vcctl_bin,
        "pod",
        "exec",
        pod.pod_name,
        "-n",
        pod.namespace,
        "-c",
        pod.container_name,
        "--",
        "bash",
        "-lc",
        f"python3 -c {json.dumps(code)} {port}",
    ]
    proc = run_capture(cmd, timeout=args.vcctl_timeout_seconds)
    return proc.returncode == 0


def resolve_healthcheck_master_port(pods: list[PodInfo], args: argparse.Namespace) -> str:
    requested = str(args.healthcheck_master_port or "").strip().lower()
    if not requested:
        return ""
    master = choose_master_pod(pods)
    candidates: list[int]
    if requested == "auto":
        candidates = list(range(args.healthcheck_port_start, args.healthcheck_port_end + 1))
        random.Random(args.run_id).shuffle(candidates)
    else:
        try:
            first = int(requested)
        except ValueError as exc:
            raise ValueError(f"invalid healthcheck master port: {args.healthcheck_master_port}") from exc
        candidates = [first] + [
            port
            for port in range(args.healthcheck_port_start, args.healthcheck_port_end + 1)
            if port != first
        ]

    for port in candidates:
        if port_available_on_pod(master, port, args):
            if str(port) != requested:
                print(
                    f"[vcctl-healthcheck] selected healthcheck master port {port} "
                    f"on pod {master.pod_name}"
                )
            return str(port)
    raise RuntimeError(
        f"no free healthcheck master port found on pod {master.pod_name} "
        f"in range {args.healthcheck_port_start}-{args.healthcheck_port_end}"
    )


def load_pods(args: argparse.Namespace) -> tuple[str, list[PodInfo]]:
    if args.pod_json_file:
        raw = Path(args.pod_json_file).read_text(encoding="utf-8")
    else:
        cmd = [
            args.vcctl_bin,
            "pod",
            "get",
            "--job",
            args.job_name,
            "-n",
            args.namespace,
            "-o",
            "json",
        ]
        proc = run_capture(cmd, timeout=args.vcctl_timeout_seconds)
        raw = proc.stdout
        if proc.returncode != 0:
            raise RuntimeError(f"vcctl pod get failed rc={proc.returncode}: {proc.stderr.strip()}")
    pods = [pod_to_info(obj, args.container_name) for obj in parse_json_stream(raw)]
    pods = [pod for pod in pods if pod.pod_name]
    pods.sort(key=lambda p: (p.task_spec != "master", p.pod_name))
    return raw, pods


def validate_pod_for_mode(pod: PodInfo, mode: str) -> tuple[bool, str]:
    if pod.phase != "Running":
        return False, f"pod phase is {pod.phase}"
    if not pod.ready:
        return False, "container is not ready"
    if mode == "multi-node":
        missing = [
            name
            for name, value in [
                ("RANK", pod.rank),
                ("WORLD_SIZE", pod.world_size),
                ("MASTER_ADDR", pod.master_addr),
                ("MASTER_PORT", pod.master_port),
            ]
            if not value
        ]
        if missing:
            return False, "missing env: " + ",".join(missing)
    return True, ""


def command_with_exports(command: str, pod: PodInfo, args: argparse.Namespace, pod_result_dir: Path, mode: str) -> str:
    exports = {
        "HC_JOB_NAME": args.job_name,
        "HC_RUN_ID": args.run_id,
        "HC_MODE": mode,
        "HC_DEVICE_TYPE": args.device_type,
        "HC_POD_NAME": pod.pod_name,
        "HC_NODE_NAME": pod.node_name,
        "HC_POD_IP": pod.pod_ip,
        "HC_HOST_IP": pod.host_ip,
        "HC_POD_RESULT_DIR": str(pod_result_dir),
    }
    export_text = " ".join(f"{key}={json.dumps(value)}" for key, value in exports.items())
    return f"export {export_text}; {command}"


def run_exec_for_pod(pod: PodInfo, mode: str, command: str, args: argparse.Namespace) -> ExecResult:
    started = iso_now()
    start_time = time.monotonic()
    logs_dir = Path(args.output_dir) / "logs"
    pod_result_dir = Path(args.pod_output_dir) / "pod_results" / pod.pod_name / mode
    if args.pod_output_dir == args.output_dir:
        pod_result_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{pod.pod_name}.{mode}.stdout"
    stderr_path = logs_dir / f"{pod.pod_name}.{mode}.stderr"

    valid, reason = validate_pod_for_mode(pod, mode)
    if not valid:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(reason + "\n", encoding="utf-8")
        finished = iso_now()
        return ExecResult(
            pod_name=pod.pod_name,
            container_name=pod.container_name,
            mode=mode,
            node_name=pod.node_name,
            pod_ip=pod.pod_ip,
            command=command,
            returncode=None,
            timeout=False,
            status="SUSPECT",
            reason=reason,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            started_at=started,
            finished_at=finished,
            elapsed_seconds=round(time.monotonic() - start_time, 3),
        )

    inner_cmd = command_with_exports(command, pod, args, pod_result_dir, mode)
    exec_cmd = [
        args.vcctl_bin,
        "pod",
        "exec",
        pod.pod_name,
        "-n",
        pod.namespace,
        "-c",
        pod.container_name,
        "--",
        "bash",
        "-lc",
        inner_cmd,
    ]
    if args.dry_run:
        stdout_path.write_text("DRY_RUN: " + " ".join(exec_cmd) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        finished = iso_now()
        return ExecResult(
            pod_name=pod.pod_name,
            container_name=pod.container_name,
            mode=mode,
            node_name=pod.node_name,
            pod_ip=pod.pod_ip,
            command=command,
            returncode=0,
            timeout=False,
            status="DRY_RUN",
            reason="dry run",
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            started_at=started,
            finished_at=finished,
            elapsed_seconds=round(time.monotonic() - start_time, 3),
        )

    timeout = False
    returncode: int | None
    try:
        proc = run_capture(exec_cmd, timeout=args.exec_timeout_seconds)
        returncode = proc.returncode
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        timeout = True
        returncode = None
        stdout_path.write_text(timeout_output_to_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(
            timeout_output_to_text(exc.stderr) + f"\nTIMEOUT after {args.exec_timeout_seconds}s\n",
            encoding="utf-8",
        )

    finished = iso_now()
    if timeout:
        status = "FAIL"
        reason = "timeout"
    elif returncode == 0:
        status = "PASS"
        reason = ""
    else:
        status = "FAIL"
        reason = f"returncode={returncode}"

    return ExecResult(
        pod_name=pod.pod_name,
        container_name=pod.container_name,
        mode=mode,
        node_name=pod.node_name,
        pod_ip=pod.pod_ip,
        command=command,
        returncode=returncode,
        timeout=timeout,
        status=status,
        reason=reason,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        started_at=started,
        finished_at=finished,
        elapsed_seconds=round(time.monotonic() - start_time, 3),
    )


def build_exec_command(pod: PodInfo, mode: str, command: str, args: argparse.Namespace, pod_result_dir: Path) -> list[str]:
    inner_cmd = command_with_exports(command, pod, args, pod_result_dir, mode)
    return [
        args.vcctl_bin,
        "pod",
        "exec",
        pod.pod_name,
        "-n",
        pod.namespace,
        "-c",
        pod.container_name,
        "--",
        "bash",
        "-lc",
        inner_cmd,
    ]


def exec_result_from_completed(
    pod: PodInfo,
    mode: str,
    command: str,
    returncode: int | None,
    timeout: bool,
    reason: str,
    stdout_path: Path,
    stderr_path: Path,
    started_at: str,
    start_time: float,
) -> ExecResult:
    if reason:
        status = "SUSPECT" if timeout else "FAIL"
    elif returncode == 0:
        status = "PASS"
    else:
        status = "FAIL"
        reason = f"returncode={returncode}"
    return ExecResult(
        pod_name=pod.pod_name,
        container_name=pod.container_name,
        mode=mode,
        node_name=pod.node_name,
        pod_ip=pod.pod_ip,
        command=command,
        returncode=returncode,
        timeout=timeout,
        status=status,
        reason=reason,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        started_at=started_at,
        finished_at=iso_now(),
        elapsed_seconds=round(time.monotonic() - start_time, 3),
    )


def run_pod_aux_command(
    pod: PodInfo,
    args: argparse.Namespace,
    command: str,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int | None = None,
) -> int | None:
    cmd = [
        args.vcctl_bin,
        "pod",
        "exec",
        pod.pod_name,
        "-n",
        pod.namespace,
        "-c",
        pod.container_name,
        "--",
        "bash",
        "-lc",
        command,
    ]
    try:
        proc = run_capture(cmd, timeout=timeout or args.vcctl_timeout_seconds)
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        return proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(timeout_output_to_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(
            timeout_output_to_text(exc.stderr) + f"\nAUX_TIMEOUT after {timeout or args.vcctl_timeout_seconds}s\n",
            encoding="utf-8",
        )
        return None


def collect_hang_diagnostics(pods: list[PodInfo], mode: str, args: argparse.Namespace) -> None:
    logs_dir = Path(args.output_dir) / "logs"
    ps_cmd = (
        "date; hostname; "
        "ps -eo pid,ppid,stat,etime,cmd | "
        "grep -E '[t]orchrun|[p]retrain_healthcheck|[p]ython.*pretrain_healthcheck' || true"
    )
    for pod in pods:
        prefix = logs_dir / f"{pod.pod_name}.{mode}.hang"
        run_pod_aux_command(
            pod,
            args,
            ps_cmd,
            prefix.with_suffix(".ps.stdout"),
            prefix.with_suffix(".ps.stderr"),
            timeout=args.vcctl_timeout_seconds,
        )


def cleanup_hung_pods(pods: list[PodInfo], mode: str, args: argparse.Namespace) -> None:
    logs_dir = Path(args.output_dir) / "logs"
    cleanup_cmd = args.hang_cleanup_cmd or args.pre_clean_cmd
    if not cleanup_cmd:
        return
    for pod in pods:
        prefix = logs_dir / f"{pod.pod_name}.{mode}.hang-cleanup"
        run_pod_aux_command(
            pod,
            args,
            cleanup_cmd,
            prefix.with_suffix(".stdout"),
            prefix.with_suffix(".stderr"),
            timeout=args.vcctl_timeout_seconds,
        )


def run_mode_with_hang_timeout(
    pods: list[PodInfo],
    mode: str,
    command: str,
    args: argparse.Namespace,
) -> list[ExecResult]:
    logs_dir = Path(args.output_dir) / "logs"
    running: list[RunningExec] = []
    results: list[ExecResult] = []
    started = time.monotonic()
    deadline = started + args.hang_timeout_seconds

    for pod in pods:
        pod_result_dir = Path(args.pod_output_dir) / "pod_results" / pod.pod_name / mode
        if args.pod_output_dir == args.output_dir:
            pod_result_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = logs_dir / f"{pod.pod_name}.{mode}.stdout"
        stderr_path = logs_dir / f"{pod.pod_name}.{mode}.stderr"

        valid, reason = validate_pod_for_mode(pod, mode)
        started_at = iso_now()
        start_time = time.monotonic()
        if not valid:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(reason + "\n", encoding="utf-8")
            results.append(
                ExecResult(
                    pod_name=pod.pod_name,
                    container_name=pod.container_name,
                    mode=mode,
                    node_name=pod.node_name,
                    pod_ip=pod.pod_ip,
                    command=command,
                    returncode=None,
                    timeout=False,
                    status="SUSPECT",
                    reason=reason,
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                    started_at=started_at,
                    finished_at=iso_now(),
                    elapsed_seconds=round(time.monotonic() - start_time, 3),
                )
            )
            continue

        exec_cmd = build_exec_command(pod, mode, command, args, pod_result_dir)
        if args.dry_run:
            stdout_path.write_text("DRY_RUN: " + " ".join(exec_cmd) + "\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            results.append(
                ExecResult(
                    pod_name=pod.pod_name,
                    container_name=pod.container_name,
                    mode=mode,
                    node_name=pod.node_name,
                    pod_ip=pod.pod_ip,
                    command=command,
                    returncode=0,
                    timeout=False,
                    status="DRY_RUN",
                    reason="dry run",
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                    started_at=started_at,
                    finished_at=iso_now(),
                    elapsed_seconds=round(time.monotonic() - start_time, 3),
                )
            )
            continue

        proc, stdout_file, stderr_file = run_capture_to_files(exec_cmd, stdout_path, stderr_path)
        running.append(
            RunningExec(
                pod=pod,
                mode=mode,
                command=command,
                process=proc,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                started_at=started_at,
                start_time=start_time,
            )
        )

    hanging = False
    while running:
        remaining: list[RunningExec] = []
        for item in running:
            returncode = item.process.poll()
            if returncode is None:
                remaining.append(item)
                continue
            item.stdout_file.close()
            item.stderr_file.close()
            results.append(
                exec_result_from_completed(
                    item.pod,
                    item.mode,
                    item.command,
                    returncode,
                    timeout=False,
                    reason="",
                    stdout_path=item.stdout_path,
                    stderr_path=item.stderr_path,
                    started_at=item.started_at,
                    start_time=item.start_time,
                )
            )
        running = remaining
        if not running:
            break
        if time.monotonic() >= deadline:
            hanging = True
            break
        time.sleep(min(1.0, max(0.1, deadline - time.monotonic())))

    if hanging:
        hung_pods = [item.pod for item in running]
        print(
            "[vcctl-healthcheck] hang timeout reached: "
            f"mode={mode} timeout={args.hang_timeout_seconds}s "
            "pods=" + ",".join(pod.pod_name for pod in hung_pods),
            file=sys.stderr,
        )
        collect_hang_diagnostics(hung_pods, mode, args)
        cleanup_hung_pods(hung_pods, mode, args)
        for item in running:
            item.process.terminate()
        wait_deadline = time.monotonic() + max(1, args.hang_kill_grace_seconds)
        for item in running:
            while item.process.poll() is None and time.monotonic() < wait_deadline:
                time.sleep(0.2)
            if item.process.poll() is None:
                item.process.kill()
            returncode = item.process.wait(timeout=5)
            item.stdout_file.close()
            item.stderr_file.close()
            with item.stderr_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"\nHANG_TIMEOUT after {args.hang_timeout_seconds}s; "
                    "diagnostics collected and cleanup issued\n"
                )
            results.append(
                exec_result_from_completed(
                    item.pod,
                    item.mode,
                    item.command,
                    returncode,
                    timeout=True,
                    reason=f"hang_timeout={args.hang_timeout_seconds}s",
                    stdout_path=item.stdout_path,
                    stderr_path=item.stderr_path,
                    started_at=item.started_at,
                    start_time=item.start_time,
                )
            )

    return results


def run_mode(pods: list[PodInfo], mode: str, command: str, args: argparse.Namespace) -> list[ExecResult]:
    if not command:
        return [
            ExecResult(
                pod_name=pod.pod_name,
                container_name=pod.container_name,
                mode=mode,
                node_name=pod.node_name,
                pod_ip=pod.pod_ip,
                command="",
                returncode=None,
                timeout=False,
                status="SUSPECT",
                reason=f"{mode} command is empty",
                stdout_path="",
                stderr_path="",
                started_at=iso_now(),
                finished_at=iso_now(),
                elapsed_seconds=0.0,
            )
            for pod in pods
        ]
    if mode == "multi-node" and args.hang_timeout_seconds > 0:
        return run_mode_with_hang_timeout(pods, mode, command, args)
    workers = len(pods) if args.max_parallel <= 0 else min(args.max_parallel, len(pods))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_pod = {executor.submit(run_exec_for_pod, pod, mode, command, args): pod for pod in pods}
        results: list[ExecResult] = []
        for future in concurrent.futures.as_completed(future_to_pod):
            pod = future_to_pod[future]
            try:
                results.append(future.result())
            except Exception as exc:
                now = iso_now()
                logs_dir = Path(args.output_dir) / "logs"
                stdout_path = logs_dir / f"{pod.pod_name}.{mode}.stdout"
                stderr_path = logs_dir / f"{pod.pod_name}.{mode}.stderr"
                if not stdout_path.exists():
                    stdout_path.write_text("", encoding="utf-8")
                stderr_path.write_text(f"INTERNAL_ERROR: {type(exc).__name__}: {exc}\n", encoding="utf-8")
                results.append(
                    ExecResult(
                        pod_name=pod.pod_name,
                        container_name=pod.container_name,
                        mode=mode,
                        node_name=pod.node_name,
                        pod_ip=pod.pod_ip,
                        command=command,
                        returncode=None,
                        timeout=False,
                        status="FAIL",
                        reason=f"internal_error={type(exc).__name__}",
                        stdout_path=str(stdout_path),
                        stderr_path=str(stderr_path),
                        started_at=now,
                        finished_at=now,
                        elapsed_seconds=0.0,
                    )
                )
        return results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_pods_tsv(path: Path, pods: list[PodInfo]) -> None:
    fields = list(PodInfo.__dataclass_fields__)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for pod in pods:
            row = asdict(pod)
            f.write("\t".join(str(row[field]) for field in fields) + "\n")


def write_commands_env(path: Path, args: argparse.Namespace) -> None:
    lines = [
        f"JOB_NAME={args.job_name}",
        f"NAMESPACE={args.namespace}",
        f"MODE={args.mode}",
        f"DEVICE_TYPE={args.device_type}",
        f"RUN_ID={args.run_id}",
        f"RESULT_ROOT={args.result_root}",
        f"POD_RESULT_ROOT={args.pod_result_root or args.result_root}",
        f"OUTPUT_DIR={args.output_dir}",
        f"EXEC_TIMEOUT_SECONDS={args.exec_timeout_seconds}",
        f"HANG_TIMEOUT_SECONDS={args.hang_timeout_seconds}",
        f"HANG_KILL_GRACE_SECONDS={args.hang_kill_grace_seconds}",
        f"MAX_PARALLEL={args.max_parallel}",
        f"HEALTHCHECK_MASTER_PORT={args.healthcheck_master_port}",
        f"RESOLVED_HEALTHCHECK_MASTER_PORT={getattr(args, 'resolved_healthcheck_master_port', '')}",
        "PRE_CLEAN_CMD=" + args.pre_clean_cmd,
        "HANG_CLEANUP_CMD=" + args.hang_cleanup_cmd,
        "STATIC_CMD=" + args.static_cmd,
        "SINGLE_NODE_CMD=" + args.single_node_cmd,
        "MULTI_NODE_CMD=" + args.multi_node_cmd,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(results: list[ExecResult]) -> str:
    statuses = [result.status for result in results]
    if not results:
        return "SUSPECT"
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if any(status == "SUSPECT" for status in statuses):
        return "SUSPECT"
    if all(status == "DRY_RUN" for status in statuses):
        return "DRY_RUN"
    return "PASS"


def write_summary_md(path: Path, pods: list[PodInfo], results: list[ExecResult], overall: str) -> None:
    lines = [
        "# vcctl Healthcheck Summary",
        "",
        f"- overall_status: `{overall}`",
        f"- pod_count: `{len(pods)}`",
        f"- result_count: `{len(results)}`",
        "",
        "## Pods",
        "",
        "| pod | container | rank | world_size | phase | ready | node | pod_ip |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pod in pods:
        lines.append(
            f"| {pod.pod_name} | {pod.container_name} | {pod.rank} | {pod.world_size} | "
            f"{pod.phase} | {pod.ready} | {pod.node_name} | {pod.pod_ip} |"
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| mode | pod | status | returncode | timeout | reason | elapsed_seconds |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in sorted(results, key=lambda x: (x.mode, x.pod_name)):
        lines.append(
            f"| {result.mode} | {result.pod_name} | {result.status} | {result.returncode} | "
            f"{result.timeout} | {result.reason} | {result.elapsed_seconds} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vcctl healthcheck orchestrator")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--mode", choices=["static", "single-node", "multi-node", "all"], default="all")
    parser.add_argument("--device-type", default="gpu")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--pod-result-root", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--vcctl-bin", default="vcctl")
    parser.add_argument("--container-name", default="")
    parser.add_argument("--pre-clean-cmd", default="")
    parser.add_argument("--static-cmd", default="")
    parser.add_argument("--single-node-cmd", default="")
    parser.add_argument("--multi-node-cmd", default="")
    parser.add_argument("--healthcheck-master-port", default="")
    parser.add_argument("--healthcheck-port-start", type=int, default=29500)
    parser.add_argument("--healthcheck-port-end", type=int, default=29999)
    parser.add_argument("--exec-timeout-seconds", type=int, default=3600)
    parser.add_argument("--hang-timeout-seconds", type=int, default=0)
    parser.add_argument("--hang-kill-grace-seconds", type=int, default=10)
    parser.add_argument("--hang-cleanup-cmd", default="")
    parser.add_argument("--vcctl-timeout-seconds", type=int, default=120)
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pod-json-file", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.result_root) / args.run_id
    args.output_dir = str(output_dir)
    pod_result_root = args.pod_result_root or args.result_root
    args.pod_output_dir = str(Path(pod_result_root) / args.run_id)
    for subdir in ["logs", "pod_results"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    try:
        raw, pods = load_pods(args)
    except Exception as exc:
        print(f"[vcctl-healthcheck] failed to load pods: {exc}", file=sys.stderr)
        return 2

    (output_dir / "vcctl_pods.raw.json").write_text(raw, encoding="utf-8")
    write_jsonl(output_dir / "pods.jsonl", [asdict(pod) for pod in pods])
    write_pods_tsv(output_dir / "pods.tsv", pods)
    write_commands_env(output_dir / "commands.env", args)

    if not pods:
        print("[vcctl-healthcheck] no pods found", file=sys.stderr)
        return 2

    try:
        resolved_port = resolve_healthcheck_master_port(pods, args)
    except Exception as exc:
        print(f"[vcctl-healthcheck] failed to resolve healthcheck master port: {exc}", file=sys.stderr)
        return 2
    args.resolved_healthcheck_master_port = resolved_port
    if resolved_port:
        args.multi_node_cmd = args.multi_node_cmd.replace("__HC_MASTER_PORT__", resolved_port)
        args.single_node_cmd = args.single_node_cmd.replace("__HC_MASTER_PORT__", resolved_port)
        args.static_cmd = args.static_cmd.replace("__HC_MASTER_PORT__", resolved_port)
        args.pre_clean_cmd = args.pre_clean_cmd.replace("__HC_MASTER_PORT__", resolved_port)

    write_commands_env(output_dir / "commands.env", args)

    print(f"[vcctl-healthcheck] pods: {len(pods)}")
    if resolved_port:
        print(f"[vcctl-healthcheck] healthcheck master port: {resolved_port}")
    for pod in pods:
        print(
            f"[vcctl-healthcheck] pod={pod.pod_name} container={pod.container_name} "
            f"rank={pod.rank} world={pod.world_size} node={pod.node_name} ip={pod.pod_ip}"
        )

    results: list[ExecResult] = []
    if args.pre_clean_cmd:
        results.extend(run_mode(pods, "pre-clean", args.pre_clean_cmd, args))
    if args.mode in {"static", "all"}:
        results.extend(run_mode(pods, "static", args.static_cmd, args))
    if args.mode in {"single-node", "all"}:
        results.extend(run_mode(pods, "single-node", args.single_node_cmd, args))
    if args.mode in {"multi-node", "all"}:
        results.extend(run_mode(pods, "multi-node", args.multi_node_cmd, args))

    result_rows = [asdict(result) for result in results]
    write_jsonl(output_dir / "results.jsonl", result_rows)
    overall = summarize(results)
    summary = {
        "overall_status": overall,
        "job_name": args.job_name,
        "namespace": args.namespace,
        "mode": args.mode,
        "device_type": args.device_type,
        "run_id": args.run_id,
        "pod_count": len(pods),
        "result_count": len(results),
        "pods": [asdict(pod) for pod in pods],
        "results": result_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(output_dir / "summary.md", pods, results, overall)
    print(f"[vcctl-healthcheck] overall_status={overall}")
    print(f"[vcctl-healthcheck] output={output_dir}")
    return 0 if overall in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
