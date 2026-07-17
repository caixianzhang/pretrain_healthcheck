#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.dynamic_compare import compare_dynamic_results, write_report as write_dynamic_compare_report
from pretrain_healthcheck.common import parse_size
from tools.dynamic_frame import (
    CHUNK_MANIFEST_PREFIX,
    CHUNK_PREFIX,
    DEFAULT_CHUNK_SIZE,
    DynamicFrameError,
    V1_PREFIX,
    V2_PREFIX,
    decode_frame_line,
    sha256_hex,
)
from tools.static_compare import (
    compare_static_results,
    render_ecc_alert_section,
    render_node_environment_sample_section,
    write_static_compare_outputs,
)


STATIC_RESULT_PREFIX = "__HC_STATIC_RESULT_JSON__ "
DYNAMIC_RESULT_PREFIXES = (V2_PREFIX, V1_PREFIX)


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
    if timeout is not None and timeout <= 0:
        timeout = None
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)


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
        "HC_RUN_STAGE": args.run_stage,
        "HC_MODE": mode,
        "HC_DEVICE_TYPE": args.device_type,
        "HC_POD_NAME": pod.pod_name,
        "HC_NODE_NAME": pod.node_name,
        "HC_POD_IP": pod.pod_ip,
        "HC_HOST_IP": pod.host_ip,
        "HC_POD_RESULT_DIR": str(pod_result_dir),
        "RANK": pod.rank,
        "WORLD_SIZE": pod.world_size,
        "MASTER_ADDR": pod.master_addr,
        "MASTER_PORT": pod.master_port,
        "HC_DYNAMIC_RETEST_PLAN_B64": getattr(args, "dynamic_retest_plan_b64", ""),
    }
    export_text = " ".join(f"{key}={json.dumps(value)}" for key, value in exports.items())
    return f"export {export_text}; {command}"


def run_exec_for_pod(pod: PodInfo, mode: str, command: str, args: argparse.Namespace) -> ExecResult:
    started = iso_now()
    start_time = time.monotonic()
    if mode == "static":
        logs_dir = Path(args.static_driver_tmp_root) / f"pretrain_healthcheck_driver_{args.run_id}"
    elif (
        mode == "single-node"
        and args.dynamic_compare
        and args.dynamic_failed_log_mode == "local-link"
        and not args.dry_run
    ):
        logs_dir = Path(args.dynamic_exec_log_root) / args.run_id / args.run_stage / "logs"
    else:
        logs_dir = Path(args.output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pod_result_dir = Path(args.pod_output_dir) / "pod_results" / pod.pod_name / mode
    dynamic_compact_mode = args.dynamic_compare and mode in {"single-node", "multi-node"}
    if args.pod_output_dir == args.output_dir and mode != "static" and not dynamic_compact_mode:
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
        timeout_seconds = args.static_exec_timeout_seconds if mode == "static" else args.exec_timeout_seconds
        exec_timeout = timeout_seconds if timeout_seconds > 0 else None
        proc = run_capture(exec_cmd, timeout=exec_timeout)
        returncode = proc.returncode
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        timeout = True
        returncode = None
        stdout_path.write_text(timeout_output_to_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(
            timeout_output_to_text(exc.stderr) + f"\nTIMEOUT after {timeout_seconds}s\n",
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


def write_commands_env(path: Path, args: argparse.Namespace) -> None:
    lines = [
        f"JOB_NAME={args.job_name}",
        f"NAMESPACE={args.namespace}",
        f"MODE={args.mode}",
        f"DEVICE_TYPE={args.device_type}",
        f"RUN_ID={args.run_id}",
        f"RUN_STAGE={args.run_stage}",
        f"DRIVER_PYTHON={os.environ.get('DRIVER_PYTHON', sys.executable)}",
        f"DRIVER_PYTHON_VERSION={os.environ.get('DRIVER_PYTHON_VERSION', '.'.join(map(str, sys.version_info[:3])))}",
        f"RESULT_ROOT={args.result_root}",
        f"POD_RESULT_ROOT={args.pod_result_root or '/tmp/pretrain_healthcheck_driver_' + args.run_id}",
        f"OUTPUT_DIR={args.output_dir}",
        f"EXEC_TIMEOUT_SECONDS={args.exec_timeout_seconds}",
        f"STATIC_EXEC_TIMEOUT_SECONDS={args.static_exec_timeout_seconds}",
        f"STATIC_STDOUT_MAX_BYTES={args.static_stdout_max_bytes}",
        f"STATIC_DRIVER_TMP_ROOT={args.static_driver_tmp_root}",
        f"MAX_PARALLEL={args.max_parallel}",
        f"STATIC_COMPARE={args.static_compare}",
        f"STATIC_COMPARE_WORKERS={args.static_compare_workers}",
        f"STATIC_COMPARE_STRICT={args.static_compare_strict}",
        f"STATIC_EXPECTED_GPUS={args.static_expected_gpus}",
        f"STATIC_EXPECTED_XSCALE_PORTS={args.static_expected_xscale_ports}",
        f"STATIC_ECC_POLICY={args.static_ecc_policy}",
        f"STATIC_KEEP_POD_FILES={args.static_keep_pod_files}",
        f"STATIC_KEEP_EXEC_LOGS={args.static_keep_exec_logs}",
        f"STATIC_FAILED_LOG_MODE={args.static_failed_log_mode}",
        f"DYNAMIC_COMPARE={args.dynamic_compare}",
        f"DYNAMIC_COMPARE_STRICT={args.dynamic_compare_strict}",
        f"DYNAMIC_COMPARE_MEASUREMENT_BATCHES={args.dynamic_compare_measurement_batches}",
        f"DYNAMIC_COMPARE_RETEST_MEASUREMENT_BATCHES={args.dynamic_compare_retest_measurement_batches}",
        f"DYNAMIC_COMPARE_RATIO_THRESHOLD={args.dynamic_compare_ratio_threshold}",
        f"DYNAMIC_COMPARE_BUSBW_RATIO_THRESHOLD={args.dynamic_compare_busbw_ratio_threshold}",
        f"DYNAMIC_COMPARE_LATENCY_RATIO_THRESHOLD={args.dynamic_compare_latency_ratio_threshold}",
        f"DYNAMIC_COMPARE_SMALL_MAX_SIZE={args.dynamic_compare_small_max_size}",
        f"DYNAMIC_COMPARE_LARGE_MIN_SIZE={args.dynamic_compare_large_min_size}",
        f"DYNAMIC_COMPARE_SMALL_LATENCY_WARN={int(args.dynamic_compare_small_latency_warn)}",
        f"DYNAMIC_COMPARE_MIN_COHORT={args.dynamic_compare_min_cohort}",
        f"DYNAMIC_COMPARE_AUTO_RETEST={int(args.dynamic_compare_auto_retest)}",
        f"DYNAMIC_KEEP_EXEC_LOGS={args.dynamic_keep_exec_logs}",
        f"DYNAMIC_EXEC_LOG_ROOT={args.dynamic_exec_log_root}",
        f"DYNAMIC_FAILED_LOG_MODE={args.dynamic_failed_log_mode}",
        f"DYNAMIC_FRAME_RECOVERY_DEADLINE_SECONDS={args.dynamic_frame_recovery_deadline_seconds}",
        f"DYNAMIC_FRAME_CHUNK_SIZE={args.dynamic_frame_chunk_size}",
        f"HEALTHCHECK_MASTER_PORT={args.healthcheck_master_port}",
        f"RESOLVED_HEALTHCHECK_MASTER_PORT={getattr(args, 'resolved_healthcheck_master_port', '')}",
        "PRE_CLEAN_CMD=" + args.pre_clean_cmd,
        "STATIC_CMD=" + args.static_cmd,
        "SINGLE_NODE_CMD=" + args.single_node_cmd,
        "MULTI_NODE_CMD=" + args.multi_node_cmd,
        "DYNAMIC_RETEST_CMD=" + args.dynamic_retest_cmd,
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


def merge_static_compare_status(overall: str, static_compare_report: dict[str, Any] | None, strict: bool) -> str:
    if not static_compare_report or not strict:
        return overall
    static_status = static_compare_report.get("static_compare_status")
    if static_status == "FAIL":
        return "FAIL"
    if static_status == "SUSPECT" and overall == "PASS":
        return "SUSPECT"
    return overall


def merge_dynamic_compare_status(overall: str, dynamic_compare_report: dict[str, Any] | None, strict: bool) -> str:
    if not dynamic_compare_report or not strict:
        return overall
    dynamic_status = dynamic_compare_report.get("dynamic_compare_status")
    if dynamic_status == "FAIL":
        return "FAIL"
    if dynamic_status in {"SUSPECT", "INCONCLUSIVE", "RETEST_REQUIRED"} and overall == "PASS":
        return "SUSPECT"
    return overall


def read_file_if_exists(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def link_failed_static_log(path_text: str, result: ExecResult, suffix: str, args: argparse.Namespace) -> str:
    if args.static_failed_log_mode != "local-link":
        return ""
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return ""
    links_dir = Path(args.output_dir) / "failed_pod_logs"
    links_dir.mkdir(parents=True, exist_ok=True)
    link_path = links_dir / f"{result.pod_name}.static.{suffix}"
    if link_path.exists() or link_path.is_symlink():
        try:
            link_path.unlink()
        except OSError:
            pass
    try:
        link_path.symlink_to(path)
        return str(link_path)
    except OSError as exc:
        fallback = links_dir / f"{result.pod_name}.static.{suffix}.path"
        fallback.write_text(f"{path}\nsymlink_error={type(exc).__name__}:{exc}\n", encoding="utf-8")
        return ""


def failed_static_row(result: ExecResult, error_type: str, reason: str, args: argparse.Namespace) -> dict[str, Any]:
    stderr = read_file_if_exists(result.stderr_path)
    match = re.search(r"^\[metax-probe\] workdir:\s*(\S+)", stderr, flags=re.MULTILINE)
    shared_stdout_link = link_failed_static_log(result.stdout_path, result, "stdout", args)
    shared_stderr_link = link_failed_static_log(result.stderr_path, result, "stderr", args)
    return {
        "pod_name": result.pod_name,
        "node_name": result.node_name,
        "pod_ip": result.pod_ip,
        "mode": result.mode,
        "status": "FAIL",
        "error_type": error_type,
        "reason": reason,
        "returncode": result.returncode,
        "timeout": result.timeout,
        "stdout_path": result.stdout_path,
        "stderr_path": result.stderr_path,
        "shared_stdout_link": shared_stdout_link,
        "shared_stderr_link": shared_stderr_link,
        "static_workdir": match.group(1) if match else "",
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "elapsed_seconds": result.elapsed_seconds,
    }


def retain_static_exec_logs(result: ExecResult, args: argparse.Namespace) -> None:
    if args.static_failed_log_mode == "local-link":
        return
    shared_logs_dir = Path(args.output_dir) / "logs"
    shared_logs_dir.mkdir(parents=True, exist_ok=True)
    for attr, suffix in [("stdout_path", "stdout"), ("stderr_path", "stderr")]:
        old_path = Path(getattr(result, attr))
        new_path = shared_logs_dir / f"{result.pod_name}.static.{suffix}"
        if old_path.exists() and old_path.resolve() != new_path.resolve():
            shutil.copyfile(old_path, new_path)
            try:
                old_path.unlink()
            except OSError:
                pass
        setattr(result, attr, str(new_path))


def cleanup_static_tmp_exec_logs(results: list[ExecResult], args: argparse.Namespace) -> int:
    tmp_root = Path(args.static_driver_tmp_root) / f"pretrain_healthcheck_driver_{args.run_id}"
    removed = 0
    for result in results:
        if result.status != "PASS":
            continue
        for path_text in [result.stdout_path, result.stderr_path]:
            if not path_text:
                continue
            path = Path(path_text)
            try:
                if path.exists() and path.is_file() and path.is_relative_to(tmp_root):
                    path.unlink()
                    removed += 1
            except OSError:
                pass
    try:
        tmp_root.rmdir()
    except OSError:
        pass
    return removed


def cleanup_empty_dirs(root: Path) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    removed = 0
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not path.is_dir():
            continue
        try:
            path.rmdir()
            removed += 1
        except OSError:
            pass
    try:
        root.rmdir()
        removed += 1
    except OSError:
        pass
    return removed


def cleanup_driver_tmp_dirs(args: argparse.Namespace) -> int:
    removed = 0
    run_tmp_root = Path(args.static_driver_tmp_root) / f"pretrain_healthcheck_driver_{args.run_id}"
    removed += cleanup_empty_dirs(run_tmp_root)

    pod_result_root = Path(args.pod_result_root or f"/tmp/pretrain_healthcheck_driver_{args.run_id}")
    if pod_result_root.name == f"pretrain_healthcheck_driver_{args.run_id}" and str(pod_result_root).startswith("/tmp/"):
        removed += cleanup_empty_dirs(pod_result_root)
    return removed


def checks_from_static_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    checks = payload.get("checks")
    if isinstance(checks, list):
        return [row for row in checks if isinstance(row, dict)]
    capability = payload.get("capability", {}) if isinstance(payload.get("capability"), dict) else {}
    checks_map = capability.get("checks", {}) if isinstance(capability.get("checks"), dict) else {}
    rows: list[dict[str, Any]] = []
    for key, value in checks_map.items():
        category, _, item = str(key).partition("/")
        row = {
            "category": category,
            "item": item,
            "status": value.get("status", "") if isinstance(value, dict) else "",
            "detail": value.get("detail", "") if isinstance(value, dict) else "",
        }
        rows.append(row)
    return rows


def collect_static_stdout_results(results: list[ExecResult], args: argparse.Namespace) -> dict[str, Any]:
    facts_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for result in results:
        stdout = read_file_if_exists(result.stdout_path)
        frames = [line[len(STATIC_RESULT_PREFIX) :] for line in stdout.splitlines() if line.startswith(STATIC_RESULT_PREFIX)]

        if result.timeout:
            result.status = "FAIL"
            result.reason = "STATIC_TIMEOUT"
            retain_static_exec_logs(result, args)
            failed_rows.append(failed_static_row(result, "STATIC_TIMEOUT", "static probe timed out", args))
            continue
        if result.returncode != 0:
            result.status = "FAIL"
            result.reason = f"EXEC_FAIL returncode={result.returncode}"
            retain_static_exec_logs(result, args)
            failed_rows.append(failed_static_row(result, "EXEC_FAIL", result.reason, args))
            continue
        if len(frames) == 0:
            result.status = "FAIL"
            result.reason = "FRAME_MISSING"
            retain_static_exec_logs(result, args)
            failed_rows.append(failed_static_row(result, "FRAME_MISSING", "missing static result stdout frame", args))
            continue
        if len(frames) > 1:
            result.status = "FAIL"
            result.reason = "FRAME_PROTOCOL_ERROR"
            retain_static_exec_logs(result, args)
            failed_rows.append(failed_static_row(result, "FRAME_PROTOCOL_ERROR", "multiple static result stdout frames", args))
            continue

        frame = frames[0]
        frame_bytes = len(frame.encode("utf-8"))
        if frame_bytes > args.static_stdout_max_bytes:
            result.status = "FAIL"
            result.reason = "FRAME_TOO_LARGE"
            retain_static_exec_logs(result, args)
            failed_rows.append(
                failed_static_row(
                    result,
                    "FRAME_TOO_LARGE",
                    f"static result frame bytes {frame_bytes} exceed {args.static_stdout_max_bytes}",
                    args,
                )
            )
            continue
        try:
            payload = json.loads(frame)
        except json.JSONDecodeError as exc:
            result.status = "FAIL"
            result.reason = "FRAME_PARSE_FAIL"
            retain_static_exec_logs(result, args)
            failed_rows.append(failed_static_row(result, "FRAME_PARSE_FAIL", f"{exc.__class__.__name__}: {exc}", args))
            continue
        if not isinstance(payload, dict):
            result.status = "FAIL"
            result.reason = "FRAME_PROTOCOL_ERROR"
            retain_static_exec_logs(result, args)
            failed_rows.append(failed_static_row(result, "FRAME_PROTOCOL_ERROR", "static frame payload is not an object", args))
            continue

        pod_meta = payload.setdefault("pod", {})
        if not isinstance(pod_meta, dict):
            result.status = "FAIL"
            result.reason = "FRAME_PROTOCOL_ERROR"
            retain_static_exec_logs(result, args)
            failed_rows.append(failed_static_row(result, "FRAME_PROTOCOL_ERROR", "static frame pod metadata is invalid", args))
            continue
        if str(pod_meta.get("name", "")) != result.pod_name:
            result.status = "FAIL"
            result.reason = "FRAME_ID_MISMATCH"
            retain_static_exec_logs(result, args)
            failed_rows.append(
                failed_static_row(
                    result,
                    "FRAME_ID_MISMATCH",
                    f"pod_name mismatch: frame={pod_meta.get('name', '')} expected={result.pod_name}",
                    args,
                )
            )
            continue
        if str(pod_meta.get("run_id", "")) != args.run_id:
            result.status = "FAIL"
            result.reason = "FRAME_ID_MISMATCH"
            retain_static_exec_logs(result, args)
            failed_rows.append(
                failed_static_row(
                    result,
                    "FRAME_ID_MISMATCH",
                    f"run_id mismatch: frame={pod_meta.get('run_id', '')} expected={args.run_id}",
                    args,
                )
            )
            continue

        pod_meta.setdefault("node_name", result.node_name)
        pod_meta.setdefault("pod_ip", result.pod_ip)
        facts_rows.append(payload)

        for row in checks_from_static_payload(payload):
            enriched = dict(row)
            enriched.setdefault("pod_name", result.pod_name)
            enriched.setdefault("node_name", result.node_name)
            enriched.setdefault("pod_ip", result.pod_ip)
            check_rows.append(enriched)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "static_facts.jsonl", facts_rows)
    write_jsonl(output_dir / "static_checks.jsonl", check_rows)
    write_jsonl(output_dir / "static_failed_pods.jsonl", failed_rows)
    cleanup_static_tmp_exec_logs(results, args)
    return {
        "fact_count": len(facts_rows),
        "check_count": len(check_rows),
        "failed_count": len(failed_rows),
    }


def cleanup_static_exec_logs(output_dir: Path, results: list[ExecResult]) -> int:
    removed = 0
    for result in results:
        if result.mode not in {"pre-clean", "static"}:
            continue
        if result.status != "PASS":
            continue
        for path_text in [result.stdout_path, result.stderr_path]:
            if not path_text:
                continue
            path = Path(path_text)
            try:
                if path.exists() and path.is_file() and path.is_relative_to(output_dir):
                    path.unlink()
                    removed += 1
            except OSError:
                pass
    try:
        logs_dir = output_dir / "logs"
        logs_dir.rmdir()
    except OSError:
        pass
    cleanup_empty_dirs(output_dir / "pod_results")
    return removed


def cleanup_dynamic_exec_logs(output_dir: Path, results: list[ExecResult]) -> int:
    removed = 0
    for result in results:
        if result.mode not in {"pre-clean", "single-node", "multi-node"}:
            continue
        if result.status != "PASS":
            continue
        for path_text in [result.stdout_path, result.stderr_path]:
            if not path_text:
                continue
            path = Path(path_text)
            try:
                if path.exists() and path.is_file():
                    path.unlink()
                    removed += 1
            except OSError:
                pass
    try:
        logs_dir = output_dir / "logs"
        logs_dir.rmdir()
    except OSError:
        pass
    cleanup_empty_dirs(output_dir / "pod_results")
    return removed


def cleanup_dynamic_exec_log_root(args: argparse.Namespace) -> int:
    if args.dynamic_failed_log_mode != "local-link":
        return 0
    run_root = Path(args.dynamic_exec_log_root) / args.run_id
    stage_root = run_root / args.run_stage
    if not stage_root.exists():
        return 0
    removed = cleanup_empty_dirs(stage_root)
    try:
        run_root.rmdir()
        removed += 1
    except OSError:
        pass
    return removed


def link_failed_dynamic_log(path_text: str, result: ExecResult, suffix: str, args: argparse.Namespace) -> str:
    if args.dynamic_failed_log_mode != "local-link":
        return ""
    if result.mode != "single-node":
        return ""
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return ""
    links_dir = Path(args.output_dir) / "failed_pod_logs"
    links_dir.mkdir(parents=True, exist_ok=True)
    link_path = links_dir / f"{result.pod_name}.{suffix}"
    if link_path.exists() or link_path.is_symlink():
        try:
            link_path.unlink()
        except OSError:
            pass
    try:
        link_path.symlink_to(path)
        return str(link_path)
    except OSError as exc:
        fallback = links_dir / f"{result.pod_name}.{suffix}.path"
        fallback.write_text(f"{path}\nsymlink_error={type(exc).__name__}:{exc}\n", encoding="utf-8")
        return ""


def failed_dynamic_row(result: ExecResult, error_type: str, reason: str, args: argparse.Namespace) -> dict[str, Any]:
    stderr = read_file_if_exists(result.stderr_path)
    match = re.search(r"^\[dynamic-stage\] workdir:\s*(\S+)", stderr, flags=re.MULTILINE)
    shared_stdout_link = link_failed_dynamic_log(result.stdout_path, result, "stdout", args)
    shared_stderr_link = link_failed_dynamic_log(result.stderr_path, result, "stderr", args)
    return {
        "pod_name": result.pod_name,
        "node_name": result.node_name,
        "pod_ip": result.pod_ip,
        "mode": result.mode,
        "status": "FAIL",
        "error_type": error_type,
        "reason": reason,
        "returncode": result.returncode,
        "timeout": result.timeout,
        "stdout_path": result.stdout_path,
        "stderr_path": result.stderr_path,
        "shared_stdout_link": shared_stdout_link,
        "shared_stderr_link": shared_stderr_link,
        "local_workdir": match.group(1) if match else "",
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "elapsed_seconds": result.elapsed_seconds,
    }


def dynamic_frame_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith(DYNAMIC_RESULT_PREFIXES)]


def dynamic_sidecar_path(result: ExecResult) -> str:
    stderr = read_file_if_exists(result.stderr_path)
    matches = re.findall(r"^\[dynamic-compact\] sidecar:\s*(\S+)", stderr, flags=re.MULTILINE)
    if not matches:
        return ""
    path = matches[-1]
    if not path.startswith("/tmp/pretrain_healthcheck_") or "/../" in path or path.endswith("/.."):
        return ""
    return path


def validate_dynamic_identity(payload: dict[str, Any], result: ExecResult, args: argparse.Namespace) -> None:
    pod_meta = payload.get("pod")
    if not isinstance(pod_meta, dict):
        raise DynamicFrameError("payload pod metadata is missing")
    expected = {
        "name": result.pod_name,
        "run_id": args.run_id,
        "stage": args.run_stage,
    }
    for key, value in expected.items():
        if str(pod_meta.get(key, "")) != str(value):
            raise DynamicFrameError(f"payload identity mismatch for {key}: expected={value!r} actual={pod_meta.get(key)!r}")


def parse_dynamic_frame(stdout: str, result: ExecResult, args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    frames = dynamic_frame_lines(stdout)
    if not frames:
        raise DynamicFrameError("dynamic frame missing")
    if len(frames) != 1:
        raise DynamicFrameError(f"expected one dynamic frame, got {len(frames)}")
    payload, protocol = decode_frame_line(frames[0])
    validate_dynamic_identity(payload, result, args)
    return payload, protocol


def parse_recovery_output(
    stdout: str,
    manifest: dict[str, Any] | None,
    chunks: dict[int, bytes],
) -> tuple[dict[str, Any] | None, dict[int, bytes]]:
    for line in stdout.splitlines():
        if line.startswith(CHUNK_MANIFEST_PREFIX):
            try:
                candidate = json.loads(line[len(CHUNK_MANIFEST_PREFIX) :])
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            if manifest is not None and candidate != manifest:
                raise DynamicFrameError("sidecar manifest changed during recovery")
            manifest = candidate
        elif line.startswith(CHUNK_PREFIX):
            try:
                row = json.loads(line[len(CHUNK_PREFIX) :])
                index = int(row["index"])
                chunk = base64.b64decode(str(row["payload"]), validate=True)
                if len(chunk) != int(row["chunk_bytes"]):
                    continue
                if sha256_hex(chunk) != str(row["chunk_sha256"]):
                    continue
                chunks[index] = chunk
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return manifest, chunks


def recover_dynamic_sidecar(
    result: ExecResult,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    sidecar = dynamic_sidecar_path(result)
    event: dict[str, Any] = {
        "pod_name": result.pod_name,
        "node_name": result.node_name,
        "sidecar": sidecar,
        "attempts": 0,
        "chunks_received": 0,
        "status": "FAILED",
        "reason": "",
    }
    if not sidecar:
        event["reason"] = "trusted sidecar path missing"
        return None, event

    deadline = time.monotonic() + args.dynamic_frame_recovery_deadline_seconds
    backoff = 1.0
    manifest: dict[str, Any] | None = None
    chunks: dict[int, bytes] = {}
    while time.monotonic() < deadline:
        total = int(manifest.get("total_chunks", 0)) if manifest else 0
        missing = [index for index in range(total) if index not in chunks] if total else []
        indexes_arg = ",".join(map(str, missing))
        command = (
            f"python3 {shlex.quote(str(PROJECT_DIR / 'tools' / 'dynamic_frame.py'))} emit-chunks "
            f"--path {shlex.quote(sidecar)} --chunk-size {args.dynamic_frame_chunk_size}"
        )
        if indexes_arg:
            command += f" --indexes {shlex.quote(indexes_arg)}"
        exec_cmd = [
            args.vcctl_bin,
            "pod",
            "exec",
            result.pod_name,
            "-n",
            args.namespace,
            "-c",
            result.container_name,
            "--",
            "bash",
            "-lc",
            command,
        ]
        event["attempts"] += 1
        try:
            proc = run_capture(exec_cmd, timeout=min(args.vcctl_timeout_seconds, max(1, int(deadline - time.monotonic()))))
            if proc.returncode == 0:
                manifest, chunks = parse_recovery_output(proc.stdout, manifest, chunks)
            else:
                event["reason"] = f"sidecar read failed: returncode={proc.returncode} stderr={proc.stderr.strip()}"
                if "No such file" in proc.stderr or "not found" in proc.stderr:
                    break
        except (subprocess.TimeoutExpired, DynamicFrameError) as exc:
            event["reason"] = f"{type(exc).__name__}: {exc}"

        if manifest:
            total = int(manifest.get("total_chunks", 0))
            if total > 0 and all(index in chunks for index in range(total)):
                data = b"".join(chunks[index] for index in range(total))
                if len(data) != int(manifest.get("file_bytes", -1)):
                    event["reason"] = "reassembled sidecar length mismatch"
                elif sha256_hex(data) != str(manifest.get("file_sha256", "")):
                    event["reason"] = "reassembled sidecar SHA256 mismatch"
                else:
                    try:
                        payload, protocol = decode_frame_line(data.decode("utf-8").rstrip("\n"))
                        validate_dynamic_identity(payload, result, args)
                    except (UnicodeDecodeError, DynamicFrameError) as exc:
                        event["reason"] = f"{type(exc).__name__}: {exc}"
                    else:
                        event.update(
                            status="RECOVERED",
                            protocol=protocol,
                            chunks_received=len(chunks),
                            sidecar_bytes=len(data),
                            sidecar_sha256=sha256_hex(data),
                            reason="",
                        )
                        return payload, event
        if time.monotonic() >= deadline:
            break
        time.sleep(min(backoff, max(0.0, deadline - time.monotonic())))
        backoff = min(backoff * 2.0, 8.0)

    event["chunks_received"] = len(chunks)
    if not event["reason"]:
        event["reason"] = "recovery deadline exceeded"
    return None, event


def collect_dynamic_stdout_results(
    results: list[ExecResult],
    args: argparse.Namespace,
    *,
    facts_filename: str = "dynamic_facts.jsonl",
    failed_filename: str = "dynamic_failed_pods.jsonl",
    transport_filename: str = "dynamic_transport.json",
) -> dict[str, Any]:
    facts_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    transport_events: list[dict[str, Any]] = []
    resolved: dict[str, tuple[dict[str, Any] | None, str, dict[str, Any], bool]] = {}
    pending: list[tuple[ExecResult, DynamicFrameError]] = []
    for result in results:
        if result.mode not in {"single-node", "multi-node"}:
            continue
        if result.timeout:
            continue
        stdout = read_file_if_exists(result.stdout_path)
        try:
            payload, protocol = parse_dynamic_frame(stdout, result, args)
            event = {
                "pod_name": result.pod_name,
                "node_name": result.node_name,
                "status": "INITIAL_OK",
                "protocol": protocol,
                "attempts": 0,
            }
        except DynamicFrameError as initial_exc:
            pending.append((result, initial_exc))
        else:
            resolved[result.pod_name] = (payload, protocol, event, False)

    if pending:
        workers = len(pending) if args.max_parallel <= 0 else min(args.max_parallel, len(pending))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(recover_dynamic_sidecar, result, args): (result, initial_exc) for result, initial_exc in pending}
            for future in concurrent.futures.as_completed(futures):
                result, initial_exc = futures[future]
                try:
                    payload, event = future.result()
                except Exception as exc:
                    payload = None
                    event = {
                        "pod_name": result.pod_name,
                        "node_name": result.node_name,
                        "status": "FAILED",
                        "attempts": 0,
                        "reason": f"internal recovery error: {type(exc).__name__}: {exc}",
                    }
                event["initial_error"] = str(initial_exc)
                protocol = str(event.get("protocol", "v2-gzip-base64"))
                resolved[result.pod_name] = (payload, protocol, event, payload is not None)

    for result in results:
        if result.mode not in {"single-node", "multi-node"}:
            continue
        if result.timeout:
            result.status = "FAIL"
            result.reason = "DYNAMIC_TIMEOUT"
            failed_rows.append(failed_dynamic_row(result, "DYNAMIC_TIMEOUT", "dynamic probe timed out", args))
            transport_events.append(
                {"pod_name": result.pod_name, "node_name": result.node_name, "status": "NOT_ATTEMPTED", "reason": "probe timeout"}
            )
            continue
        payload, protocol, transport_event, recovered = resolved[result.pod_name]
        if payload is None:
            result.status = "FAIL"
            result.reason = "RESULT_TRANSPORT_FAIL"
            failed_rows.append(
                failed_dynamic_row(result, "RESULT_TRANSPORT_FAIL", str(transport_event.get("reason", "recovery failed")), args)
            )
            transport_events.append(transport_event)
            continue

        coverage = payload.get("coverage")
        if result.returncode == 0 and isinstance(coverage, dict) and not coverage.get("complete", False):
            result.status = "FAIL"
            result.reason = "DATA_INCOMPLETE"
            failed_rows.append(
                failed_dynamic_row(result, "DATA_INCOMPLETE", "; ".join(map(str, coverage.get("errors", []))), args)
            )
            transport_event["status"] = "DATA_INCOMPLETE"
            transport_events.append(transport_event)
            continue

        pod_meta = payload.get("pod", {})
        pod_meta.setdefault("node_name", result.node_name)
        pod_meta.setdefault("pod_ip", result.pod_ip)
        driver_result = asdict(result)
        driver_result.update(frame_protocol=protocol, frame_recovered=recovered, frame_recovery_attempts=transport_event.get("attempts", 0))
        payload["driver_result"] = driver_result
        facts_rows.append(payload)
        transport_events.append(transport_event)
        summary = payload.get("summary", {})
        if result.returncode != 0 or not isinstance(summary, dict) or not summary.get("correctness_pass", False):
            result.status = "FAIL"
            result.reason = str(summary.get("error_type", "")) if isinstance(summary, dict) else f"returncode={result.returncode}"
            failed_rows.append(
                failed_dynamic_row(result, "DYNAMIC_CHECK_FAILED", result.reason or f"returncode={result.returncode}", args)
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / facts_filename, facts_rows)
    write_jsonl(output_dir / failed_filename, failed_rows)
    transport = {
        "schema_version": 1,
        "expected_pods": len([result for result in results if result.mode in {"single-node", "multi-node"}]),
        "accepted_facts": len(facts_rows),
        "initial_ok": sum(row.get("status") == "INITIAL_OK" for row in transport_events),
        "recovery_attempts": sum(int(row.get("attempts", 0) or 0) for row in transport_events),
        "recovery_success": sum(row.get("status") == "RECOVERED" for row in transport_events),
        "recovery_failed": sum(row.get("status") == "FAILED" for row in transport_events),
        "data_incomplete": sum(row.get("status") == "DATA_INCOMPLETE" for row in transport_events),
        "events": transport_events,
    }
    (output_dir / transport_filename).write_text(json.dumps(transport, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"fact_count": len(facts_rows), "failed_count": len(failed_rows), **transport}


def compare_dynamic_with_one_retest(
    pods: list[PodInfo],
    mode: str,
    initial_results: list[ExecResult],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[ExecResult]]:
    collect_dynamic_stdout_results(initial_results, args)
    compare_kwargs = {
        "ratio_threshold": args.dynamic_compare_busbw_ratio_threshold,
        "latency_ratio_threshold": args.dynamic_compare_latency_ratio_threshold,
        "min_cohort": args.dynamic_compare_min_cohort,
        "small_max_bytes": args.dynamic_compare_small_max_bytes,
        "large_min_bytes": args.dynamic_compare_large_min_bytes,
        "small_latency_warn": args.dynamic_compare_small_latency_warn,
        "small_latency_abs_delta_seconds": args.dynamic_compare_small_latency_abs_delta_ms / 1000.0,
        "small_latency_mad_multiplier": args.dynamic_compare_small_latency_mad_multiplier,
    }
    report = compare_dynamic_results(Path(args.output_dir), **compare_kwargs)
    retest_results: list[ExecResult] = []
    if (
        report.get("retest_required")
        and args.dynamic_compare_auto_retest
        and args.dynamic_retest_cmd
        and report.get("retest_plan")
    ):
        plan_json = json.dumps(report["retest_plan"], ensure_ascii=False, separators=(",", ":"))
        args.dynamic_retest_plan_b64 = base64.b64encode(plan_json.encode("utf-8")).decode("ascii")
        print(
            f"[vcctl-healthcheck] dynamic_retest_start mode={mode} cases={len(report['retest_plan'])}",
            flush=True,
        )
        retest_results = run_mode(pods, mode, args.dynamic_retest_cmd, args)
        collect_dynamic_stdout_results(
            retest_results,
            args,
            facts_filename="dynamic_retest_facts.jsonl",
            failed_filename="dynamic_retest_failed_pods.jsonl",
            transport_filename="dynamic_retest_transport.json",
        )
        report = compare_dynamic_results(
            Path(args.output_dir),
            retest_facts_path=Path(args.output_dir) / "dynamic_retest_facts.jsonl",
            **compare_kwargs,
        )
        print(
            f"[vcctl-healthcheck] dynamic_retest_done mode={mode} status={report.get('dynamic_compare_status')}",
            flush=True,
        )
    report["initial_measurement_batches"] = args.dynamic_compare_measurement_batches
    report["retest_measurement_batches"] = args.dynamic_compare_retest_measurement_batches
    write_dynamic_compare_report(Path(args.output_dir), report)
    return report, retest_results


def write_summary_md(
    path: Path,
    pods: list[PodInfo],
    results: list[ExecResult],
    overall: str,
    static_compare_report: dict[str, Any] | None = None,
    dynamic_compare_report: dict[str, Any] | None = None,
) -> None:
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
    transport_path = path.parent / "dynamic_transport.json"
    if transport_path.exists():
        try:
            transport = json.loads(transport_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            transport = {}
        lines.extend(
            [
                "",
                "## Dynamic Result Transport",
                "",
                f"- expected_pods: `{transport.get('expected_pods', 0)}`",
                f"- accepted_facts: `{transport.get('accepted_facts', 0)}`",
                f"- initial_ok: `{transport.get('initial_ok', 0)}`",
                f"- recovery_attempts: `{transport.get('recovery_attempts', 0)}`",
                f"- recovery_success: `{transport.get('recovery_success', 0)}`",
                f"- recovery_failed: `{transport.get('recovery_failed', 0)}`",
                f"- data_incomplete: `{transport.get('data_incomplete', 0)}`",
            ]
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
    if static_compare_report:
        lines.extend(
            [
                "",
                "## Static Compare",
                "",
                f"- static_compare_status: `{static_compare_report.get('static_compare_status')}`",
                f"- issue_count: `{static_compare_report.get('issue_count', 0)}`",
                f"- warning_count: `{static_compare_report.get('warning_count', 0)}`",
                "- report: `static_compare.md`",
            ]
        )
        ecc_alert_section = render_ecc_alert_section(static_compare_report)
        if ecc_alert_section:
            lines.extend(["", ecc_alert_section.rstrip()])
        node_sample_section = render_node_environment_sample_section(path.parent, static_compare_report)
        if node_sample_section:
            lines.extend(["", node_sample_section.rstrip()])
    if dynamic_compare_report:
        lines.extend(
            [
                "",
                "## Dynamic Compare",
                "",
                f"- dynamic_compare_status: `{dynamic_compare_report.get('dynamic_compare_status')}`",
                f"- issue_count: `{dynamic_compare_report.get('issue_count', 0)}`",
                f"- outlier_count: `{dynamic_compare_report.get('outlier_count', 0)}`",
                "- report: `dynamic_compare.md`",
            ]
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
    parser.add_argument("--run-stage", default="")
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
    parser.add_argument("--static-exec-timeout-seconds", type=int, default=180)
    parser.add_argument("--static-stdout-max-bytes", type=int, default=1048576)
    parser.add_argument("--static-driver-tmp-root", default="/tmp")
    parser.add_argument("--vcctl-timeout-seconds", type=int, default=120)
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--static-compare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--static-compare-workers", type=int, default=0)
    parser.add_argument("--static-compare-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--static-expected-gpus", type=int, default=0)
    parser.add_argument("--static-expected-xscale-ports", type=int, default=0)
    parser.add_argument("--static-ecc-policy", choices=["alert", "strict"], default="alert")
    parser.add_argument("--static-keep-pod-files", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static-keep-exec-logs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--static-failed-log-mode", choices=["local-link", "shared"], default="local-link")
    parser.add_argument("--dynamic-compare", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic-compare-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-compare-measurement-batches", type=int, default=1)
    parser.add_argument("--dynamic-compare-retest-measurement-batches", type=int, default=3)
    parser.add_argument("--dynamic-compare-ratio-threshold", type=float, default=0.7)
    parser.add_argument("--dynamic-compare-busbw-ratio-threshold", type=float, default=0.7)
    parser.add_argument("--dynamic-compare-latency-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--dynamic-compare-small-max-size", type=parse_size, default=parse_size("1M"))
    parser.add_argument("--dynamic-compare-large-min-size", type=parse_size, default=parse_size("1G"))
    parser.add_argument("--dynamic-compare-small-latency-warn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic-compare-small-latency-abs-delta-ms", type=float, default=0.2)
    parser.add_argument("--dynamic-compare-small-latency-mad-multiplier", type=float, default=6.0)
    parser.add_argument("--dynamic-compare-min-cohort", type=int, default=3)
    parser.add_argument("--dynamic-compare-auto-retest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-retest-cmd", default="")
    parser.add_argument("--dynamic-keep-exec-logs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic-exec-log-root", default="/tmp/pretrain_healthcheck_exec_logs/vcctl")
    parser.add_argument("--dynamic-failed-log-mode", choices=["local-link", "shared"], default="local-link")
    parser.add_argument("--dynamic-frame-recovery-deadline-seconds", type=int, default=60)
    parser.add_argument("--dynamic-frame-chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pod-json-file", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for attr in ("dynamic_compare_measurement_batches", "dynamic_compare_retest_measurement_batches"):
        if getattr(args, attr) < 1:
            raise SystemExit(f"{attr.replace('_', '-')} must be >= 1")
    if args.dynamic_frame_recovery_deadline_seconds < 1:
        raise SystemExit("dynamic-frame-recovery-deadline-seconds must be >= 1")
    if args.dynamic_frame_chunk_size < 256:
        raise SystemExit("dynamic-frame-chunk-size must be >= 256")
    args.dynamic_compare_small_max_bytes = args.dynamic_compare_small_max_size
    args.dynamic_compare_large_min_bytes = args.dynamic_compare_large_min_size
    if args.dynamic_compare_busbw_ratio_threshold == 0.7 and args.dynamic_compare_ratio_threshold != 0.7:
        args.dynamic_compare_busbw_ratio_threshold = args.dynamic_compare_ratio_threshold
    args.dynamic_retest_plan_b64 = ""
    if not args.run_stage:
        args.run_stage = args.mode.replace("-", "_")
    output_dir = Path(args.result_root) / args.run_id / args.run_stage
    args.output_dir = str(output_dir)
    pod_result_root = args.pod_result_root or f"/tmp/pretrain_healthcheck_driver_{args.run_id}"
    args.pod_output_dir = str(Path(pod_result_root) / args.run_id / args.run_stage)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _raw, pods = load_pods(args)
    except Exception as exc:
        print(f"[vcctl-healthcheck] failed to load pods: {exc}", file=sys.stderr)
        return 2

    write_jsonl(output_dir / "pods.jsonl", [asdict(pod) for pod in pods])
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
        args.dynamic_retest_cmd = args.dynamic_retest_cmd.replace("__HC_MASTER_PORT__", resolved_port)
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
    static_compare_report: dict[str, Any] | None = None
    dynamic_compare_report: dict[str, Any] | None = None
    if args.pre_clean_cmd:
        results.extend(run_mode(pods, "pre-clean", args.pre_clean_cmd, args))
    if args.mode in {"static", "all"}:
        static_results = run_mode(pods, "static", args.static_cmd, args)
        results.extend(static_results)
        if args.static_compare and not args.dry_run:
            collect_static_stdout_results(static_results, args)
            static_compare_report = compare_static_results(
                output_dir,
                workers=args.static_compare_workers,
                expected_gpus=args.static_expected_gpus,
                expected_xscale_ports=args.static_expected_xscale_ports,
                ecc_policy=args.static_ecc_policy,
            )
            write_static_compare_outputs(output_dir, static_compare_report)
        if not args.static_keep_exec_logs and not args.dry_run:
            removed_static_logs = cleanup_static_exec_logs(output_dir, results)
            if removed_static_logs:
                print(f"[vcctl-healthcheck] removed static exec logs: {removed_static_logs}")
    if args.mode in {"single-node", "all"}:
        single_node_results = run_mode(pods, "single-node", args.single_node_cmd, args)
        results.extend(single_node_results)
        if args.dynamic_compare and not args.dry_run:
            dynamic_compare_report, retest_results = compare_dynamic_with_one_retest(
                pods, "single-node", single_node_results, args
            )
            results.extend(retest_results)
            if not args.dynamic_keep_exec_logs:
                removed_dynamic_logs = cleanup_dynamic_exec_logs(output_dir, results)
                if removed_dynamic_logs:
                    print(f"[vcctl-healthcheck] removed dynamic exec logs: {removed_dynamic_logs}")
    if args.mode in {"multi-node", "all"}:
        multi_node_results = run_mode(pods, "multi-node", args.multi_node_cmd, args)
        results.extend(multi_node_results)
        if args.dynamic_compare and not args.dry_run:
            dynamic_compare_report, retest_results = compare_dynamic_with_one_retest(
                pods, "multi-node", multi_node_results, args
            )
            results.extend(retest_results)
            if not args.dynamic_keep_exec_logs:
                removed_dynamic_logs = cleanup_dynamic_exec_logs(output_dir, results)
                if removed_dynamic_logs:
                    print(f"[vcctl-healthcheck] removed dynamic exec logs: {removed_dynamic_logs}")

    result_rows = [asdict(result) for result in results]
    write_jsonl(output_dir / "results.jsonl", result_rows)
    overall = merge_static_compare_status(summarize(results), static_compare_report, args.static_compare_strict)
    overall = merge_dynamic_compare_status(overall, dynamic_compare_report, args.dynamic_compare_strict)
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
        "static_compare": static_compare_report,
        "dynamic_compare": dynamic_compare_report,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(output_dir / "summary.md", pods, results, overall, static_compare_report, dynamic_compare_report)
    removed_tmp_dirs = cleanup_driver_tmp_dirs(args)
    if removed_tmp_dirs:
        print(f"[vcctl-healthcheck] removed empty driver tmp dirs: {removed_tmp_dirs}")
    removed_dynamic_log_dirs = cleanup_dynamic_exec_log_root(args)
    if removed_dynamic_log_dirs:
        print(f"[vcctl-healthcheck] removed empty dynamic exec log dirs: {removed_dynamic_log_dirs}")
    print(f"[vcctl-healthcheck] overall_status={overall}")
    print(f"[vcctl-healthcheck] output={output_dir}")
    return 0 if overall in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
