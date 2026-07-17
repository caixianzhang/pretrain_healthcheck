#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HISTORICAL_DYNAMIC_SUITE_BUSBW = [
    176.600,
    176.608,
    167.291,
    159.948,
    176.615,
    178.897,
    163.305,
    176.460,
]

HCCL_ROW_RE = re.compile(
    r"^\s*(?P<size>\d+)\s*\|\s*"
    r"(?P<latency>[0-9]+(?:\.[0-9]+)?)\s*\|\s*"
    r"(?P<algbw>[0-9]+(?:\.[0-9]+)?)\s*\|\s*"
    r"(?P<check>[A-Za-z_]+)\s*$"
)


@dataclass(frozen=True)
class Pod:
    pod_name: str
    namespace: str
    container_name: str
    task_spec: str
    node_name: str
    host_ip: str
    pod_ip: str


@dataclass
class Result:
    run_id: str
    variant: str
    pod_name: str
    container_name: str
    node_name: str
    host_ip: str
    pod_ip: str
    status: str
    returncode: int | None
    timeout: bool
    error_type: str
    rank_count: int
    dtype: str
    message_size: str
    message_size_bytes: int | None
    warmup: int
    iters: int
    result_row_count: int
    correctness_pass: bool
    max_avg_latency_us: float | None
    printed_algbw_min_gbps: float | None
    algbw_gbps: float | None
    busbw_gbps: float | None
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
        raise ValueError(f"container {forced!r} not found in pod {nested(raw, 'metadata', 'name')}")
    task_spec = str(nested(raw, "metadata", "labels", "volcano.sh/task-spec"))
    for container in containers:
        if container.get("name") == task_spec:
            return container
    return containers[0]


def pod_from_raw(raw: dict[str, Any], forced_container: str) -> Pod | None:
    pod_name = str(nested(raw, "metadata", "name"))
    node_name = str(nested(raw, "spec", "nodeName"))
    if not pod_name or not node_name:
        return None
    container = choose_container(raw, forced_container)
    return Pod(
        pod_name=pod_name,
        namespace=str(nested(raw, "metadata", "namespace", default="default")),
        container_name=str(container.get("name", "")),
        task_spec=str(nested(raw, "metadata", "labels", "volcano.sh/task-spec")),
        node_name=node_name,
        host_ip=str(nested(raw, "status", "hostIP")),
        pod_ip=str(nested(raw, "status", "podIP")),
    )


def pod_sort_key(pod: Pod) -> tuple[int, int, str]:
    group = 0 if pod.task_spec == "master" else 1 if pod.task_spec == "worker" else 2
    suffix = pod.pod_name.rsplit("-", 1)[-1]
    return group, int(suffix) if suffix.isdigit() else 0, pod.pod_name


def select_pods(pods: list[Pod], requested_names: list[str]) -> list[Pod]:
    if not requested_names:
        return pods
    if len(requested_names) != len(set(requested_names)):
        raise ValueError("pod filter contains duplicate names")
    by_name = {pod.pod_name: pod for pod in pods}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"requested pods not found: {','.join(missing)}")
    return [by_name[name] for name in requested_names]


def message_size_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([KMGT]?)B?\s*", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"invalid message size: {value!r}")
    number = int(match.group(1))
    unit = match.group(2).upper()
    power = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4}[unit]
    return number * (1024**power)


def parse_hccl_output(text: str, expected_size: int, rank_count: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = HCCL_ROW_RE.match(line)
        if not match:
            continue
        size = int(match.group("size"))
        if size != expected_size:
            continue
        rows.append(
            {
                "size": size,
                "latency_us": float(match.group("latency")),
                "algbw_gbps": float(match.group("algbw")),
                "check": match.group("check").lower(),
            }
        )
    correctness_pass = bool(rows) and all(row["check"] == "success" for row in rows)
    max_latency = max((row["latency_us"] for row in rows), default=None)
    printed_min = min((row["algbw_gbps"] for row in rows), default=None)
    algbw = None
    busbw = None
    if max_latency is not None and max_latency > 0:
        algbw = expected_size / (max_latency / 1_000_000) / 1_000_000_000
        busbw = algbw * 2 * (rank_count - 1) / rank_count
    return {
        "rows": rows,
        "result_row_count": len(rows),
        "correctness_pass": correctness_pass,
        "max_avg_latency_us": max_latency,
        "printed_algbw_min_gbps": printed_min,
        "algbw_gbps": algbw,
        "busbw_gbps": busbw,
    }


def shell_assign(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def build_remote_command(args: argparse.Namespace) -> str:
    lines = [
        "set -eo pipefail",
        f"test -r {shlex.quote(args.ascend_env_script)}",
        f"source {shlex.quote(args.ascend_env_script)} >/dev/null 2>&1",
        f"test -x {shlex.quote(args.mpi_bin)}",
        f"test -x {shlex.quote(args.test_bin)}",
        f"export LD_LIBRARY_PATH={shlex.quote(args.mpi_lib_dir)}:\"${{LD_LIBRARY_PATH:-}}\"",
        shell_assign("HCCL_SOCKET_IFNAME", args.socket_ifname),
    ]
    command = [
        args.mpi_bin,
        "-n",
        str(args.npus_per_node),
        args.test_bin,
        "-b",
        args.message_size,
        "-e",
        args.message_size,
        "-f",
        "2",
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
    lines.append("exec " + shlex.join(command))
    return "\n".join(lines)


def vcctl_exec_command(args: argparse.Namespace, pod: Pod, remote_command: str) -> list[str]:
    command = [args.vcctl_bin, "pod", "exec", pod.pod_name, "-n", args.namespace]
    if pod.container_name:
        command.extend(["-c", pod.container_name])
    command.extend(["--", "bash", "-lc", remote_command])
    return command


def load_pods(args: argparse.Namespace) -> list[Pod]:
    command = [args.vcctl_bin, "pod", "get", "--job", args.job_name, "-n", args.namespace, "-o", "json"]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"vcctl pod get failed rc={proc.returncode}: {proc.stderr.strip()}")
    pods = [pod for pod in (pod_from_raw(raw, args.container_name) for raw in parse_json_stream(proc.stdout)) if pod]
    pods.sort(key=pod_sort_key)
    if not pods:
        raise RuntimeError(f"no scheduled pods found for job {args.job_name!r}")
    return select_pods(pods, args.pod_names)


def run_one(args: argparse.Namespace, pod: Pod, variant: str, log_root: Path) -> Result:
    remote_command = build_remote_command(args)
    command = vcctl_exec_command(args, pod, remote_command)
    stdout_path = log_root / f"{pod.pod_name}.{variant}.stdout"
    stderr_path = log_root / f"{pod.pod_name}.{variant}.stderr"
    started = time.monotonic()
    timeout = False
    returncode: int | None = None
    stdout = ""
    stderr = ""
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=None if args.exec_timeout_seconds <= 0 else args.exec_timeout_seconds,
        )
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
    parsed = parse_hccl_output(stdout, args.message_size_bytes, args.npus_per_node)
    if timeout:
        error_type = "TIMEOUT"
    elif returncode != 0:
        error_type = "EXEC_FAILED"
    elif not parsed["rows"]:
        error_type = "RESULT_MISSING"
    elif not parsed["correctness_pass"]:
        error_type = "CORRECTNESS_FAILED"
    else:
        error_type = ""
    status = "PASS" if not error_type else "FAIL"
    return Result(
        run_id=args.run_id,
        variant=variant,
        pod_name=pod.pod_name,
        container_name=pod.container_name,
        node_name=pod.node_name,
        host_ip=pod.host_ip,
        pod_ip=pod.pod_ip,
        status=status,
        returncode=returncode,
        timeout=timeout,
        error_type=error_type,
        rank_count=args.npus_per_node,
        dtype=args.dtype,
        message_size=args.message_size,
        message_size_bytes=args.message_size_bytes,
        warmup=args.warmup,
        iters=args.iters,
        result_row_count=parsed["result_row_count"],
        correctness_pass=parsed["correctness_pass"],
        max_avg_latency_us=parsed["max_avg_latency_us"],
        printed_algbw_min_gbps=parsed["printed_algbw_min_gbps"],
        algbw_gbps=parsed["algbw_gbps"],
        busbw_gbps=parsed["busbw_gbps"],
        elapsed_seconds=elapsed,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        command=shlex.join(command),
    )


def fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None, "median": None, "cv": None}
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "cv": statistics.pstdev(values) / mean if mean else None,
    }


def build_summary(args: argparse.Namespace, pods: list[Pod], results: list[Result], started_at: str, elapsed: float) -> dict[str, Any]:
    busbw = [result.busbw_gbps for result in results if result.status == "PASS" and result.busbw_gbps is not None]
    baseline = {
        "status": "PASS" if len(busbw) == len(pods) else "FAIL",
        "pass_count": sum(result.status == "PASS" for result in results),
        "fail_count": sum(result.status != "PASS" for result in results),
        "busbw_gbps": distribution([float(value) for value in busbw]),
    }
    baseline_mean = baseline["busbw_gbps"]["mean"]
    historical = distribution(HISTORICAL_DYNAMIC_SUITE_BUSBW)
    baseline_vs_historical = None
    if isinstance(baseline_mean, (int, float)) and historical["mean"]:
        baseline_vs_historical = (baseline_mean - float(historical["mean"])) / float(historical["mean"]) * 100
    return {
        "run_id": args.run_id,
        "job_name": args.job_name,
        "namespace": args.namespace,
        "status": baseline["status"],
        "started_at": started_at,
        "finished_at": iso_now(),
        "elapsed_seconds": elapsed,
        "pod_count": len(pods),
        "workload": {
            "op": "all_reduce",
            "reduce_op": "sum",
            "rank_count": args.npus_per_node,
            "dtype": args.dtype,
            "message_size": args.message_size,
            "message_size_bytes": args.message_size_bytes,
            "warmup": args.warmup,
            "iters": args.iters,
            "correctness": True,
            "busbw_factor": 2 * (args.npus_per_node - 1) / args.npus_per_node,
        },
        "baseline": baseline,
        "historical_dynamic_suite_busbw_gbps": historical,
        "aligned_vs_historical_percent": baseline_vs_historical,
    }


def write_summary_md(path: Path, summary: dict[str, Any], results: list[Result]) -> None:
    workload = summary["workload"]
    lines = [
        "# Huawei Single-Node HCCL All-Reduce Summary",
        "",
        f"- Status: **{summary['status']}**",
        f"- Job: `{summary['job_name']}`",
        f"- Run ID: `{summary['run_id']}`",
        f"- Pods: {summary['pod_count']}",
        f"- Workload: {workload['rank_count']} ranks, All-Reduce SUM, {workload['message_size']}, {workload['dtype']}, warmup={workload['warmup']}, iters={workload['iters']}",
        f"- BusBW factor: {workload['busbw_factor']:.3f}",
        f"- Wall time: {summary['elapsed_seconds']:.3f}s",
        "",
        "## Per-node Results",
        "",
        "| Variant | Pod | Node | Status | Rows | Correctness | Max avg latency (us) | AlgBW (GB/s) | BusBW (GB/s) |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for result in sorted(results, key=lambda item: (item.variant, item.pod_name)):
        lines.append(
            f"| `{result.variant}` | `{result.pod_name}` | `{result.node_name}` | {result.status} | "
            f"{result.result_row_count} | {str(result.correctness_pass).lower()} | "
            f"{fmt(result.max_avg_latency_us)} | {fmt(result.algbw_gbps)} | {fmt(result.busbw_gbps)} |"
        )
    lines.extend(["", "## Distribution", "", "| Variant | Status | Mean BusBW | Min | Max | CV |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    item = summary["baseline"]
    stats = item["busbw_gbps"]
    lines.append(
        f"| `aligned_baseline` | {item['status']} | {fmt(stats['mean'])} | {fmt(stats['min'])} | "
        f"{fmt(stats['max'])} | {fmt(stats['cv'], 4)} |"
    )
    historical = summary["historical_dynamic_suite_busbw_gbps"]
    lines.append(
        f"| `historical_dynamic_suite` | reference | {fmt(historical['mean'])} | {fmt(historical['min'])} | "
        f"{fmt(historical['max'])} | {fmt(historical['cv'], 4)} |"
    )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- Aligned baseline vs historical dynamic-suite mean: {fmt(summary['aligned_vs_historical_percent'])}%.",
            "- HCCL test reports average latency per result row. The group metric uses the maximum row latency, then converts AlgBW to 16-rank All-Reduce BusBW.",
            "- A difference above 10% requires environment and topology analysis; it is not by itself a hardware failure verdict.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_parameters(path: Path, args: argparse.Namespace) -> None:
    values = {
        "JOB_NAME": args.job_name,
        "NAMESPACE": args.namespace,
        "POD_NAMES": ",".join(args.pod_names),
        "RUN_ID": args.run_id,
        "NPUS_PER_NODE": args.npus_per_node,
        "DTYPE": args.dtype,
        "MESSAGE_SIZE": args.message_size,
        "WARMUP": args.warmup,
        "ITERS": args.iters,
        "ASCEND_ENV_SCRIPT": args.ascend_env_script,
        "MPI_BIN": args.mpi_bin,
        "MPI_LIB_DIR": args.mpi_lib_dir,
        "HCCL_TEST_BIN": args.test_bin,
        "HCCL_SOCKET_IFNAME": args.socket_ifname,
        "EXEC_TIMEOUT_SECONDS": args.exec_timeout_seconds,
        "MAX_PARALLEL": args.max_parallel,
        "DRY_RUN": int(args.dry_run),
    }
    path.write_text("".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official HCCL all_reduce_test independently in every vcctl pod")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--vcctl-bin", default="vcctl")
    parser.add_argument("--container-name", default="")
    parser.add_argument("--pod-names", default="")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ascend-env-script", required=True)
    parser.add_argument("--mpi-bin", required=True)
    parser.add_argument("--mpi-lib-dir", required=True)
    parser.add_argument("--test-bin", required=True)
    parser.add_argument("--npus-per-node", type=int, default=16)
    parser.add_argument("--dtype", default="bfp16")
    parser.add_argument("--message-size", default="1G")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--socket-ifname", default="eth0")
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--exec-timeout-seconds", type=int, default=300)
    parser.add_argument("--dry-run", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    args.dry_run = bool(args.dry_run)
    args.pod_names = [name.strip() for name in args.pod_names.split(",") if name.strip()]
    args.message_size_bytes = message_size_bytes(args.message_size)
    if args.npus_per_node < 2:
        parser.error("--npus-per-node must be at least 2")
    return args


def main() -> int:
    args = parse_args()
    started_at = iso_now()
    started = time.monotonic()
    output_dir = args.result_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path("/tmp") / "pretrain_healthcheck_hccl_official" / args.run_id
    log_root.mkdir(parents=True, exist_ok=True)
    write_parameters(output_dir / "parameters.env", args)
    pods = load_pods(args)
    (output_dir / "pods.jsonl").write_text(
        "".join(json.dumps(asdict(pod), sort_keys=True) + "\n" for pod in pods), encoding="utf-8"
    )
    print(f"[hccl-single-node] pods: {len(pods)}")
    for pod in pods:
        print(f"[hccl-single-node] pod={pod.pod_name} container={pod.container_name} node={pod.node_name} host_ip={pod.host_ip}")

    if args.dry_run:
        commands = []
        for pod in pods:
            command = vcctl_exec_command(args, pod, build_remote_command(args))
            commands.append({"variant": "aligned_baseline", "pod": pod.pod_name, "command": shlex.join(command)})
        (output_dir / "dry_run_commands.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in commands), encoding="utf-8"
        )
        print(f"[hccl-single-node] overall_status=DRY_RUN")
        print(f"[hccl-single-node] output={output_dir}")
        return 0

    results: list[Result] = []
    workers = args.max_parallel if args.max_parallel > 0 else len(pods)
    variant = "aligned_baseline"
    print(f"[hccl-single-node] variant_start={variant} pods={len(pods)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(run_one, args, pod, variant, log_root): pod for pod in pods}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[hccl-single-node] pod_done variant={variant} pod={result.pod_name} "
                f"status={result.status} busbw={fmt(result.busbw_gbps)} elapsed={result.elapsed_seconds:.3f}s"
            )
    print(f"[hccl-single-node] variant_done={variant}")

    (output_dir / "node_results.jsonl").write_text(
        "".join(json.dumps(asdict(result), sort_keys=True) + "\n" for result in results), encoding="utf-8"
    )
    summary = build_summary(args, pods, results, started_at, time.monotonic() - started)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_md(output_dir / "summary.md", summary, results)
    failed = [result for result in results if result.status != "PASS"]
    if failed:
        links = output_dir / "failed_pod_logs"
        links.mkdir(exist_ok=True)
        for result in failed:
            for source in (Path(result.stdout_path), Path(result.stderr_path)):
                link = links / source.name
                if not link.exists():
                    link.symlink_to(source)
    print(f"[hccl-single-node] overall_status={summary['status']}")
    print(f"[hccl-single-node] output={output_dir}")
    print(f"[hccl-single-node] raw_logs={log_root}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
