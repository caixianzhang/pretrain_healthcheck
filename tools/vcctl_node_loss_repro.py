#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pretrain_healthcheck.training_topology import load_training_topology_manifest
from tools.vcctl_multi_node_batch import (
    GroupTask,
    evaluate_job_liveness,
    group_pod_json,
    job_liveness_record,
    parse_json_stream,
    pod_from_raw,
    pod_sort_key,
)


EXIT_PASS = 0
EXIT_NODE_LOSS_REPRODUCED = 10
EXIT_JOB_INFRA_LOSS = 11
EXIT_PREFLIGHT_FAILED = 20
EXIT_WORKLOAD_FAILED = 30
EXIT_CONTROLLER_ERROR = 40

ERROR_PATTERNS = re.compile(
    r"NET/IB|Got completion|vendor err|ncclRemoteError|remote process exiting|"
    r"connection (?:reset|closed)|ibv_cmd_create_qp_ex|QP.*fail|ATU.*Fault|"
    r"illegal memory access|UnexpectedAdmissionError|unhealthy devices",
    re.IGNORECASE,
)


class PreflightError(RuntimeError):
    pass


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def run(
    command: list[str],
    *,
    timeout: int = 120,
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if check and process.returncode != 0:
        raise RuntimeError(
            f"command failed rc={process.returncode}: {' '.join(command)}\n"
            f"{process.stderr.strip()}"
        )
    return process


def parse_excluded_nodes(value: str) -> list[str]:
    return sorted({token for token in re.split(r"[\s,]+", value.strip()) if token})


def valid_record(record: dict[str, Any]) -> bool:
    return bool(
        record.get("node_name")
        and record.get("phase") == "Running"
        and record.get("ready")
        and record.get("pod_ip")
    )


def issue_node_name(issue: dict[str, Any]) -> str:
    for key in ("current", "baseline"):
        value = issue.get(key)
        if isinstance(value, dict) and value.get("node_name"):
            return str(value["node_name"])
    return str(issue.get("node_name", ""))


def issue_pod_name(issue: dict[str, Any]) -> str:
    if issue.get("pod_name"):
        return str(issue["pod_name"])
    for key in ("current", "baseline"):
        value = issue.get(key)
        if isinstance(value, dict) and value.get("pod_name"):
            return str(value["pod_name"])
    return ""


def natural_pod_key(name: str) -> tuple[int, int, str]:
    task = 0 if "-master-" in name else 1 if "-worker-" in name else 2
    suffix = name.rsplit("-", 1)[-1]
    return task, int(suffix) if suffix.isdigit() else 0, name


def workload_shapes(world_size: int) -> dict[str, list[dict[str, Any]]]:
    shapes: list[dict[str, Any]] = [
        {
            "case_id": "model_tp_all_reduce_32m",
            "family": "tp",
            "op": "all_reduce",
            "message_bytes": 32 * 1024 * 1024,
        },
        {
            "case_id": "model_tp_reduce_scatter_32m",
            "family": "tp",
            "op": "reduce_scatter",
            "message_bytes": 32 * 1024 * 1024,
        },
        {
            "case_id": "model_tp_all_gather_32m",
            "family": "tp",
            "op": "all_gather",
            "message_bytes": 32 * 1024 * 1024,
        },
        {
            "case_id": "model_pp_activation_32m",
            "family": "pp",
            "op": "send_recv",
            "message_bytes": 32 * 1024 * 1024,
        },
        {
            "case_id": "model_ep_all_to_all_64m",
            "family": "ep",
            "op": "all_to_all",
            "message_bytes": 64 * 1024 * 1024,
        },
    ]
    for pattern in ("uniform", "skewed", "hot_expert", "random", "empty_expert"):
        shapes.append(
            {
                "case_id": f"model_ep_all_to_allv_64m_{pattern}",
                "family": "ep",
                "op": "all_to_allv",
                "message_bytes": 64 * 1024 * 1024,
                "payload_pattern": pattern,
            }
        )
    return {str(world_size): shapes}


class ReproductionRun:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.started_monotonic = time.monotonic()
        self.shared_dir = args.result_root / args.run_id
        self.local_dir = args.local_output_root / args.run_id
        self.inputs_dir = self.shared_dir / "inputs"
        self.controller_logs = self.local_dir / "controller_logs"
        self.events_path = self.shared_dir / "liveness_events.jsonl"
        self.healthcheck = args.project_dir / "scripts/metax/run_vcctl_healthcheck.sh"
        self.selected_node_names: set[str] = set()
        self.excluded_nodes: set[str] = set()
        self.excluded_pod_names: set[str] = set()
        self.baseline: list[dict[str, Any]] = []
        self.process: subprocess.Popen[Any] | None = None
        self.console_handle: Any = None
        self.first_issues: list[dict[str, Any]] = []
        self.issue_first_seen: dict[tuple[str, str], str] = {}
        self.initialized = False
        self.state: dict[str, Any] = {
            "run_id": args.run_id,
            "job_name": args.job_name,
            "namespace": args.namespace,
            "target_nodes": args.target_nodes,
            "started_at": now(),
            "overall_status": "RUNNING",
            "exit_code": None,
            "selected_nodes": [],
            "excluded_nodes": [],
            "liveness_issues": [],
        }

    def vcctl_raw(self) -> str:
        command = [
            self.args.vcctl_bin,
            "pod",
            "get",
            "--job",
            self.args.job_name,
            "-n",
            self.args.namespace,
            "-o",
            "json",
        ]
        return run(command, timeout=30).stdout

    def current_records(self) -> list[dict[str, Any]]:
        return [job_liveness_record(raw) for raw in parse_json_stream(self.vcctl_raw())]

    def filtered_current_records(self) -> list[dict[str, Any]]:
        return [
            record
            for record in self.current_records()
            if str(record.get("node_name", "")) not in self.excluded_nodes
            and str(record.get("pod_name", "")) not in self.excluded_pod_names
        ]

    def initialize_directories(self) -> None:
        if self.shared_dir.exists():
            raise PreflightError(f"result directory already exists: {self.shared_dir}")
        self.inputs_dir.mkdir(parents=True)
        self.controller_logs.mkdir(parents=True, exist_ok=True)
        self.initialized = True

    def select_nodes(self) -> list[Any]:
        raw_objects = parse_json_stream(self.vcctl_raw())
        pods = [pod for pod in (pod_from_raw(raw) for raw in raw_objects) if pod is not None]
        pods.sort(key=pod_sort_key)
        records = [job_liveness_record(raw) for raw in raw_objects]
        by_node = {pod.node_name: pod for pod in pods}
        known_nodes = {str(record.get("node_name", "")) for record in records if record.get("node_name")}

        requested = parse_excluded_nodes(self.args.excluded_nodes)
        unknown = sorted(set(requested) - known_nodes)
        if unknown:
            raise PreflightError(f"unknown EXCLUDED_NODES: {','.join(unknown)}")
        self.excluded_nodes = set(requested)
        self.excluded_pod_names = {
            str(record.get("pod_name", ""))
            for record in records
            if str(record.get("node_name", "")) in self.excluded_nodes
        }
        self.state["excluded_nodes"] = requested

        write_lines(self.shared_dir / "excluded_nodes_requested.txt", requested)
        matched_rows = ["node_name\tpod_name\tphase\tready\treason\tmessage"]
        for record in sorted(records, key=lambda row: natural_pod_key(str(row.get("pod_name", "")))):
            if str(record.get("node_name", "")) in self.excluded_nodes:
                matched_rows.append(
                    "\t".join(
                        [
                            str(record.get("node_name", "")),
                            str(record.get("pod_name", "")),
                            str(record.get("phase", "")),
                            str(bool(record.get("ready"))).lower(),
                            str(record.get("reason", "")).replace("\t", " "),
                            str(record.get("message", "")).replace("\t", " ").replace("\n", " "),
                        ]
                    )
                )
        (self.shared_dir / "excluded_nodes_matched.tsv").write_text(
            "\n".join(matched_rows) + "\n", encoding="utf-8"
        )

        nonexcluded = [record for record in records if str(record.get("node_name", "")) not in self.excluded_nodes]
        invalid = [record for record in nonexcluded if not valid_record(record)]
        atomic_json(self.shared_dir / "job_preflight_invalid_nodes.json", invalid)
        if invalid:
            raise PreflightError(
                f"{len(invalid)} non-excluded pods are not Running/Ready with Pod IP; "
                "exclude confirmed failed hostnames or use a fresh job"
            )

        eligible = [
            by_node[str(record["node_name"])]
            for record in nonexcluded
            if str(record.get("node_name", "")) in by_node
        ]
        eligible.sort(key=pod_sort_key)
        if len({pod.node_name for pod in eligible}) != len(eligible):
            raise PreflightError("eligible pool contains duplicate hostnames")
        write_lines(self.shared_dir / "eligible_nodes.txt", [pod.node_name for pod in eligible])
        if len(eligible) < self.args.target_nodes:
            raise PreflightError(
                f"insufficient eligible nodes: required_nodes={self.args.target_nodes} "
                f"eligible_nodes={len(eligible)} excluded_nodes={len(self.excluded_nodes)}"
            )

        selected = eligible[: self.args.target_nodes]
        standby = eligible[self.args.target_nodes :]
        self.selected_node_names = {pod.node_name for pod in selected}
        write_lines(self.inputs_dir / "selected_nodes.txt", [pod.node_name for pod in selected])
        write_lines(self.shared_dir / "standby_nodes.txt", [pod.node_name for pod in standby])
        atomic_json(self.inputs_dir / "selected_pods.json", group_pod_json(
            GroupTask("node_loss_repro", "node_loss_repro", "node_loss_repro", selected)
        ))
        self.baseline = nonexcluded
        atomic_json(self.shared_dir / "job_liveness_baseline.json", self.baseline)
        atomic_json(self.shared_dir / "job_state_before.json", records)

        self.state["selected_nodes"] = [pod.node_name for pod in selected]
        self.state["excluded_nodes"] = requested
        self.state["standby_nodes"] = [pod.node_name for pod in standby]
        return selected

    def pod_probe(self, pod: Any) -> dict[str, Any]:
        pattern = "[t]orchrun|pretrain_healthcheck[.]cli|[s]creen[.]py|[t]rain_qwen"
        shell = (
            f"processes=$(ps -eo pid=,args= | grep -E '{pattern}' || true); "
            f"devices=$({self.args.pod_python} -c "
            "\"import torch; print(torch.cuda.device_count())\" 2>&1); "
            "printf 'DEVICE_COUNT=%s\\n' \"$devices\"; printf 'PROCESSES_BEGIN\\n%s\\nPROCESSES_END\\n' \"$processes\""
        )
        command = [
            self.args.vcctl_bin,
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
            shell,
        ]
        try:
            process = run(command, timeout=30, check=False)
            match = re.search(r"DEVICE_COUNT=(\d+)", process.stdout)
            count = int(match.group(1)) if match else -1
            process_text = process.stdout.partition("PROCESSES_BEGIN\n")[2].partition("\nPROCESSES_END")[0].strip()
            return {
                "pod_name": pod.pod_name,
                "node_name": pod.node_name,
                "returncode": process.returncode,
                "device_count": count,
                "processes": process_text,
                "stderr": process.stderr.strip(),
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "pod_name": pod.pod_name,
                "node_name": pod.node_name,
                "returncode": 124,
                "device_count": -1,
                "processes": "",
                "stderr": f"TimeoutExpired: {exc}",
            }

    def probe_selected_pods(self, selected: list[Any]) -> None:
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(128, len(selected))) as pool:
            futures = [pool.submit(self.pod_probe, pod) for pod in selected]
            for future in as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda row: natural_pod_key(str(row["pod_name"])))
        atomic_json(self.shared_dir / "process_preflight.json", rows)
        failed = [
            row
            for row in rows
            if row["returncode"] != 0
            or row["device_count"] != self.args.gpus_per_node
            or row["processes"]
        ]
        if failed:
            raise PreflightError(f"{len(failed)} selected pods failed device/process preflight")

    def create_manifest(self) -> tuple[Path, str]:
        world_size = self.args.target_nodes * self.args.gpus_per_node
        if not self.args.megatron_path.is_dir():
            raise PreflightError(f"Megatron path not found: {self.args.megatron_path}")
        python_probe = run(
            [
                str(self.args.driver_python),
                "-c",
                "import torch; print(torch.__version__)",
            ],
            timeout=30,
            check=False,
        )
        if python_probe.returncode != 0:
            raise PreflightError(
                f"DRIVER_PYTHON cannot import torch: {self.args.driver_python}: "
                f"{python_probe.stderr.strip()}"
            )
        shapes_path = self.inputs_dir / "workload_shapes.json"
        model_path = self.inputs_dir / "model.json"
        manifest_path = self.inputs_dir / "training_topology_manifest.json"
        atomic_json(shapes_path, workload_shapes(world_size))
        atomic_json(
            model_path,
            {
                "name": "qwen3_1tb_moe_reproduction",
                "tp": 4,
                "ep": 32,
                "etp": 1,
                "pp": 8,
                "cp": 1,
                "mbs": 1,
                "gbs": 1024,
            },
        )
        command = [
            str(self.args.driver_python),
            str(self.args.project_dir / "tools/export_megatron_training_topology.py"),
            "--megatron-path",
            str(self.args.megatron_path),
            "--world-size",
            str(world_size),
            "--ranks-per-node",
            str(self.args.gpus_per_node),
            "--tp",
            "4",
            "--ep",
            "32",
            "--etp",
            "1",
            "--pp",
            "8",
            "--cp",
            "1",
            "--mbs",
            "1",
            "--gbs",
            "1024",
            "--rank-order",
            "tp-cp-ep-dp-pp",
            "--model-json",
            str(model_path),
            "--workload-shapes-json",
            str(shapes_path),
            "--output",
            str(manifest_path),
        ]
        try:
            run(command, timeout=120, cwd=self.args.project_dir)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            raise PreflightError(f"training topology export failed: {exc}") from exc
        manifest = load_training_topology_manifest(manifest_path)
        profile = manifest.profiles[world_size]
        atomic_json(
            self.shared_dir / "topology_profile_summary.json",
            {
                "manifest_sha256": manifest.sha256,
                "parallelism": profile.parallelism,
                "group_counts": {family: len(groups) for family, groups in profile.groups.items()},
                "group_sizes": {
                    family: sorted({len(group.ranks) for group in groups})
                    for family, groups in profile.groups.items()
                },
                "workload_shapes": list(profile.workload_shapes),
            },
        )
        return manifest_path, manifest.sha256

    def liveness_issues(self) -> list[dict[str, Any]]:
        return evaluate_job_liveness(self.baseline, self.filtered_current_records())

    def record_liveness(self, stage: str, issues: list[dict[str, Any]]) -> None:
        append_jsonl(
            self.events_path,
            {
                "timestamp": now(),
                "stage": stage,
                "issues": issues,
            },
        )

    def idle_baseline(self) -> None:
        started = time.monotonic()
        while time.monotonic() - started < self.args.idle_seconds:
            issues = self.liveness_issues()
            self.record_liveness("idle_baseline", issues)
            if issues:
                raise PreflightError(f"job changed during idle baseline: {len(issues)} issues")
            elapsed = int(time.monotonic() - started)
            print(
                f"[node-loss-repro] progress stage=idle_baseline "
                f"elapsed={elapsed}/{self.args.idle_seconds}s",
                flush=True,
            )
            time.sleep(
                min(
                    self.args.poll_seconds,
                    max(0.1, self.args.idle_seconds - (time.monotonic() - started)),
                )
            )

    def workload_env(self, manifest_path: Path, manifest_sha256: str) -> dict[str, str]:
        environment = os.environ.copy()
        stage_run_id = f"{self.args.run_id}_training_topology"
        environment.update(
            {
                "JOB_NAME": self.args.job_name,
                "NAMESPACE": self.args.namespace,
                "RUN_ID": stage_run_id,
                "RUN_STAGE": "node_loss_repro/training_topology/multi_node_training_topology",
                "POD_JSON_FILE": str(self.inputs_dir / "selected_pods.json"),
                "PRESERVE_POD_JSON_ORDER": "1",
                "MODE": "multi-node",
                "PROFILE": "training-topology",
                "DRY_RUN": "0",
                "PRE_CLEAN": "1",
                "PRE_CLEAN_STRICT": "1",
                "DYNAMIC_COMPARE": "1",
                "DYNAMIC_COMPARE_AUTO_RETEST": "0",
                "EXEC_TIMEOUT_SECONDS": str(self.args.exec_timeout_seconds),
                "GPUS_PER_NODE": str(self.args.gpus_per_node),
                "DRIVER_PYTHON": str(self.args.driver_python),
                "PROJECT_REMOTE_DIR": str(self.args.pod_project_dir),
                "POD_PROJECT_DIR": str(self.args.pod_project_dir),
                "TRAINING_TOPOLOGY_MANIFEST": str(manifest_path),
                "POD_TRAINING_TOPOLOGY_MANIFEST": str(manifest_path),
                "TOPOLOGY_MANIFEST_SHA256": manifest_sha256,
                "TOPOLOGY_WARMUP": "1",
                "TOPOLOGY_ITERS": "1",
                "TOPOLOGY_OVERLAP_CANARY": "0",
                "HEALTHCHECK_MASTER_PORT": str(self.args.master_port),
                "MAX_PARALLEL": "0",
                "RESULT_ROOT": str(self.local_dir),
                "POD_RESULT_ROOT": f"/tmp/pretrain_healthcheck_driver_{self.args.run_id}",
                "DYNAMIC_KEEP_EXEC_LOGS": "1",
                "DYNAMIC_FAILED_LOG_MODE": "local-link",
            }
        )
        return environment

    def start_workload(self, manifest_path: Path, manifest_sha256: str) -> None:
        console_path = self.controller_logs / "training_topology.console.log"
        self.console_handle = console_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            ["bash", str(self.healthcheck)],
            cwd=str(self.args.project_dir),
            env=self.workload_env(manifest_path, manifest_sha256),
            stdout=self.console_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        print(f"[node-loss-repro] workload_start pid={self.process.pid}", flush=True)

    def stop_workload(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def targeted_cleanup(self, selected: list[Any]) -> None:
        marker = self.args.run_id
        shell = (
            f"marker={json.dumps(marker)}; "
            "ps -eo pid=,args= | awk -v marker=\"$marker\" "
            "'index($0,marker) && "
            "($0 ~ /[p]retrain_healthcheck[.]cli/ || $0 ~ /[t]orchrun/) {print $1}' "
            "| xargs -r kill -TERM"
        )

        def cleanup_one(pod: Any) -> dict[str, Any]:
            command = [
                self.args.vcctl_bin,
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
                shell,
            ]
            try:
                process = run(command, timeout=20, check=False)
                return {
                    "pod_name": pod.pod_name,
                    "node_name": pod.node_name,
                    "returncode": process.returncode,
                    "stderr": process.stderr.strip(),
                }
            except subprocess.TimeoutExpired as exc:
                return {
                    "pod_name": pod.pod_name,
                    "node_name": pod.node_name,
                    "returncode": None,
                    "stderr": f"TimeoutExpired: {exc}",
                }

        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(128, len(selected))) as pool:
            futures = [pool.submit(cleanup_one, pod) for pod in selected]
            for future in as_completed(futures):
                rows.append(future.result())
        atomic_json(
            self.shared_dir / "targeted_cleanup.json",
            {"timestamp": now(), "marker": marker, "results": sorted(rows, key=lambda row: row["pod_name"])},
        )

    def snapshot(self, filename: str) -> list[dict[str, Any]]:
        records = self.current_records()
        atomic_json(self.shared_dir / filename, records)
        return records

    def observe_after_failure(self) -> list[dict[str, Any]]:
        self.snapshot("job_state_first_failure.json")
        started = time.monotonic()
        latest_issues = self.first_issues[:]
        wrote_30 = False
        while time.monotonic() - started < self.args.post_failure_observe_seconds:
            latest_issues = self.liveness_issues()
            self.record_liveness("post_failure_observe", latest_issues)
            elapsed = time.monotonic() - started
            if elapsed >= 30 and not wrote_30:
                self.snapshot("job_state_after_30s.json")
                wrote_30 = True
            time.sleep(
                min(
                    self.args.poll_seconds,
                    max(0.1, self.args.post_failure_observe_seconds - elapsed),
                )
            )
        if not wrote_30:
            self.snapshot("job_state_after_30s.json")
        self.snapshot("job_state_after_120s.json")
        return latest_issues

    def classify_and_write_loss(self, issues: list[dict[str, Any]]) -> int:
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for issue in issues:
            node_name = issue_node_name(issue)
            pod_name = issue_pod_name(issue)
            if not node_name and not pod_name:
                continue
            unique[(node_name, pod_name)] = issue
        baseline_by_pod = {
            str(record.get("pod_name", "")): record for record in self.baseline
        }
        detected_at = now()
        rows = [
            "node_name\tpod_name\toriginal_pod_ip\tfirst_detected_at\tselected\t"
            "issue_type\treason\tmessage"
        ]
        suggested: list[str] = []
        selected_loss = False
        for (node_name, pod_name), issue in sorted(unique.items()):
            issue_key = (node_name, pod_name)
            if issue_key not in self.issue_first_seen:
                self.issue_first_seen[issue_key] = detected_at
            current = issue.get("current") if isinstance(issue.get("current"), dict) else {}
            baseline = issue.get("baseline") if isinstance(issue.get("baseline"), dict) else {}
            if not baseline:
                baseline = baseline_by_pod.get(pod_name, {})
            selected = node_name in self.selected_node_names
            selected_loss = selected_loss or selected
            if node_name:
                suggested.append(node_name)
            rows.append(
                "\t".join(
                    [
                        node_name,
                        pod_name,
                        str(baseline.get("pod_ip", "")),
                        self.issue_first_seen[issue_key],
                        str(selected).lower(),
                        str(issue.get("type", "")),
                        str(current.get("reason", "")).replace("\t", " "),
                        str(current.get("message", "")).replace("\t", " ").replace("\n", " "),
                    ]
                )
            )
        (self.shared_dir / "lost_nodes.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        write_lines(self.shared_dir / "suggested_excluded_nodes.txt", sorted(set(suggested)))
        return EXIT_NODE_LOSS_REPRODUCED if selected_loss else EXIT_JOB_INFRA_LOSS

    def monitor_workload(self) -> tuple[int, list[dict[str, Any]]]:
        assert self.process is not None
        started = time.monotonic()
        deadline = started + self.args.controller_timeout_seconds
        while self.process.poll() is None:
            issues = self.liveness_issues()
            self.record_liveness("training_topology", issues)
            if issues:
                self.first_issues = issues
                atomic_json(
                    self.shared_dir / "job_liveness_alert.json",
                    {"timestamp": now(), "stage": "training_topology", "issues": issues},
                )
                return self.classify_and_write_loss(issues), issues
            if time.monotonic() >= deadline:
                return EXIT_WORKLOAD_FAILED, []
            elapsed = int(time.monotonic() - started)
            print(
                f"[node-loss-repro] progress stage=training_topology elapsed={elapsed}s",
                flush=True,
            )
            time.sleep(self.args.poll_seconds)
        returncode = int(self.process.returncode or 0)
        issues = self.liveness_issues()
        self.record_liveness("training_topology_done", issues)
        if issues:
            self.first_issues = issues
            atomic_json(
                self.shared_dir / "job_liveness_alert.json",
                {"timestamp": now(), "stage": "training_topology_done", "issues": issues},
            )
            return self.classify_and_write_loss(issues), issues
        return (EXIT_PASS if returncode == 0 else EXIT_WORKLOAD_FAILED), []

    def cooldown(self) -> tuple[int, list[dict[str, Any]]]:
        started = time.monotonic()
        while time.monotonic() - started < self.args.cooldown_seconds:
            issues = self.liveness_issues()
            self.record_liveness("final_cooldown", issues)
            if issues:
                self.first_issues = issues
                atomic_json(
                    self.shared_dir / "job_liveness_alert.json",
                    {"timestamp": now(), "stage": "final_cooldown", "issues": issues},
                )
                return self.classify_and_write_loss(issues), issues
            time.sleep(
                min(
                    self.args.poll_seconds,
                    max(0.1, self.args.cooldown_seconds - (time.monotonic() - started)),
                )
            )
        return EXIT_PASS, []

    def collect_error_signatures(self) -> None:
        evidence_dir = self.shared_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        excerpts: list[str] = []
        for path in sorted(self.local_dir.rglob("*")):
            if not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    matched_for_file = 0
                    for line_number, line in enumerate(handle, 1):
                        if ERROR_PATTERNS.search(line):
                            text = line.rstrip()
                            rows.append(
                                {
                                    "source": str(path),
                                    "line": line_number,
                                    "message": text,
                                }
                            )
                            excerpts.append(f"{path}:{line_number}: {text}")
                            matched_for_file += 1
                            if matched_for_file >= 200 or len(rows) >= 2000:
                                break
            except OSError:
                continue
            if len(rows) >= 2000:
                break
        for record in self.current_records():
            combined = f"{record.get('reason', '')} {record.get('message', '')}"
            if ERROR_PATTERNS.search(combined):
                rows.append(
                    {
                        "source": "vcctl_pod_status",
                        "pod_name": record.get("pod_name", ""),
                        "node_name": record.get("node_name", ""),
                        "message": combined.strip(),
                    }
                )
                excerpts.append(
                    f"vcctl_pod_status:{record.get('pod_name', '')}: {combined.strip()}"
                )
        with (self.shared_dir / "error_signatures.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (evidence_dir / "error_excerpt.log").write_text(
            "\n".join(excerpts) + ("\n" if excerpts else ""),
            encoding="utf-8",
        )

    def write_manifest(self, manifest_sha256: str = "") -> None:
        atomic_json(
            self.shared_dir / "reproduction_manifest.json",
            {
                "schema_version": 1,
                "job_name": self.args.job_name,
                "namespace": self.args.namespace,
                "run_id": self.args.run_id,
                "target_nodes": self.args.target_nodes,
                "gpus_per_node": self.args.gpus_per_node,
                "world_size": self.args.target_nodes * self.args.gpus_per_node,
                "excluded_nodes": sorted(self.excluded_nodes),
                "selected_nodes": sorted(self.selected_node_names),
                "training_parallelism": {
                    "tp": 4,
                    "ep": 32,
                    "etp": 1,
                    "pp": 8,
                    "cp": 1,
                    "mbs": 1,
                    "gbs": 1024,
                    "rank_order": "tp-cp-ep-dp-pp",
                },
                "payloads": {"tp_pp": "32M", "ep": "64M"},
                "dtype": "bf16",
                "warmup": 1,
                "iters": 1,
                "overlap_canary": False,
                "manifest_sha256": manifest_sha256,
                "idle_seconds": self.args.idle_seconds,
                "poll_seconds": self.args.poll_seconds,
                "post_failure_observe_seconds": self.args.post_failure_observe_seconds,
            },
        )

    def write_summary(self) -> None:
        exit_code = self.state.get("exit_code")
        descriptions = {
            EXIT_PASS: "communication completed without node loss",
            EXIT_NODE_LOSS_REPRODUCED: "selected communication node loss reproduced",
            EXIT_JOB_INFRA_LOSS: "non-selected job infrastructure node loss",
            EXIT_PREFLIGHT_FAILED: "preflight rejected",
            EXIT_WORKLOAD_FAILED: "workload failed without platform node loss",
            EXIT_CONTROLLER_ERROR: "controller error",
        }
        meaning = (
            "preflight passed without starting communication"
            if self.state.get("overall_status") == "PREFLIGHT_PASS"
            else descriptions.get(exit_code, "unknown")
        )
        lines = [
            f"# Muxi {self.args.target_nodes}-Node Loss Reproduction Summary",
            "",
            f"- job: `{self.args.job_name}`",
            f"- run_id: `{self.args.run_id}`",
            f"- status: `{self.state.get('overall_status', '')}`",
            f"- exit_code: `{exit_code}`",
            f"- meaning: `{meaning}`",
            f"- started_at: `{self.state.get('started_at', '')}`",
            f"- finished_at: `{self.state.get('finished_at', '')}`",
            f"- elapsed_seconds: `{self.state.get('elapsed_seconds', 0)}`",
            f"- target_nodes: `{self.args.target_nodes}`",
            f"- excluded_nodes: `{','.join(self.state.get('excluded_nodes', []))}`",
            "",
            "## Artifacts",
            "",
            "- `lost_nodes.tsv`",
            "- `suggested_excluded_nodes.txt`",
            "- `job_liveness_alert.json`",
            "- `liveness_events.jsonl`",
            "- `error_signatures.jsonl`",
            "- `evidence/error_excerpt.log`",
            "- `topology_profile_summary.json`",
        ]
        if self.state.get("error"):
            lines.extend(["", "## Error", "", f"`{self.state['error']}`"])
        (self.shared_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def finalize(self) -> None:
        if self.console_handle is not None:
            self.console_handle.close()
        self.state["finished_at"] = now()
        self.state["elapsed_seconds"] = round(time.monotonic() - self.started_monotonic, 3)
        atomic_json(self.shared_dir / "run_summary.json", self.state)
        self.write_summary()
        raw_link = self.shared_dir / "raw_output"
        if not raw_link.exists() and not raw_link.is_symlink():
            try:
                raw_link.symlink_to(self.local_dir)
            except OSError:
                (self.shared_dir / "raw_output.path").write_text(
                    str(self.local_dir) + "\n", encoding="utf-8"
                )
        print(
            f"[node-loss-repro] done status={self.state['overall_status']} "
            f"exit_code={self.state['exit_code']} result={self.shared_dir}",
            flush=True,
        )

    def execute(self) -> int:
        selected: list[Any] = []
        manifest_sha256 = ""
        exit_code = EXIT_CONTROLLER_ERROR
        try:
            self.initialize_directories()
            selected = self.select_nodes()
            self.write_manifest()
            self.probe_selected_pods(selected)
            manifest_path, manifest_sha256 = self.create_manifest()
            self.write_manifest(manifest_sha256)
            if self.args.preflight_only:
                self.state["overall_status"] = "PREFLIGHT_PASS"
                exit_code = EXIT_PASS
                return exit_code
            if self.args.confirmation != "YES":
                raise PreflightError(
                    "formal execution requires CONFIRM_NODE_LOSS_REPRO=YES"
                )
            self.idle_baseline()
            self.start_workload(manifest_path, manifest_sha256)
            exit_code, issues = self.monitor_workload()
            if issues:
                self.stop_workload()
                self.targeted_cleanup(selected)
                delayed_issues = self.observe_after_failure()
                exit_code = self.classify_and_write_loss(self.first_issues + delayed_issues)
            elif exit_code == EXIT_PASS:
                exit_code, issues = self.cooldown()
                if issues:
                    self.stop_workload()
                    self.targeted_cleanup(selected)
                    delayed_issues = self.observe_after_failure()
                    exit_code = self.classify_and_write_loss(self.first_issues + delayed_issues)
            else:
                self.stop_workload()
                self.targeted_cleanup(selected)
                delayed_issues = self.observe_after_failure()
                if delayed_issues:
                    self.first_issues = delayed_issues
                    exit_code = self.classify_and_write_loss(delayed_issues)
            self.state["overall_status"] = {
                EXIT_PASS: "PASS",
                EXIT_NODE_LOSS_REPRODUCED: "NODE_LOSS_REPRODUCED",
                EXIT_JOB_INFRA_LOSS: "JOB_INFRA_LOSS",
                EXIT_WORKLOAD_FAILED: "WORKLOAD_FAILED",
            }.get(exit_code, "FAILED")
            return exit_code
        except PreflightError as exc:
            exit_code = EXIT_PREFLIGHT_FAILED
            self.state["overall_status"] = "PREFLIGHT_FAILED"
            self.state["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[node-loss-repro] PRECHECK ERROR {exc}", file=sys.stderr, flush=True)
            return exit_code
        except Exception as exc:  # noqa: BLE001
            self.stop_workload()
            if selected:
                self.targeted_cleanup(selected)
            exit_code = EXIT_CONTROLLER_ERROR
            self.state["overall_status"] = "CONTROLLER_ERROR"
            self.state["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[node-loss-repro] ERROR {self.state['error']}", file=sys.stderr, flush=True)
            return exit_code
        finally:
            self.state["exit_code"] = exit_code
            self.state["liveness_issues"] = self.first_issues
            if self.initialized:
                try:
                    self.collect_error_signatures()
                except Exception as exc:  # noqa: BLE001
                    self.state["evidence_error"] = f"{type(exc).__name__}: {exc}"
                self.finalize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="reproduce Muxi 96/128-node topology node loss")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--pod-project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--vcctl-bin", default="vcctl")
    parser.add_argument("--target-nodes", type=int, choices=(96, 128), required=True)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--excluded-nodes", default="")
    parser.add_argument("--result-root", type=Path, default=PROJECT_DIR / "results/vcctl")
    parser.add_argument(
        "--local-output-root",
        type=Path,
        default=Path("/tmp/pretrain_healthcheck_group_outputs/vcctl"),
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--megatron-path", type=Path, required=True)
    parser.add_argument("--driver-python", type=Path, required=True)
    parser.add_argument("--pod-python", default="/opt/conda/bin/python3")
    parser.add_argument("--idle-seconds", type=int, default=120)
    parser.add_argument("--cooldown-seconds", type=int, default=120)
    parser.add_argument("--post-failure-observe-seconds", type=int, default=120)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--exec-timeout-seconds", type=int, default=300)
    parser.add_argument("--controller-timeout-seconds", type=int, default=420)
    parser.add_argument("--master-port", type=int, default=29741)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.project_dir = args.project_dir.expanduser().resolve()
    args.pod_project_dir = args.pod_project_dir.expanduser().resolve()
    args.result_root = args.result_root.expanduser().resolve()
    args.local_output_root = args.local_output_root.expanduser().resolve()
    args.megatron_path = args.megatron_path.expanduser().resolve()
    args.driver_python = args.driver_python.expanduser().resolve()
    if not args.run_id:
        args.run_id = (
            f"muxi_{args.target_nodes}node_loss_repro_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )
    if args.target_nodes * args.gpus_per_node % (32 * 8):
        raise SystemExit("world size must be divisible by EP*PP=256")
    for name in (
        "idle_seconds",
        "cooldown_seconds",
        "post_failure_observe_seconds",
        "poll_seconds",
        "exec_timeout_seconds",
        "controller_timeout_seconds",
    ):
        if int(getattr(args, name)) <= 0:
            raise SystemExit(f"{name} must be positive")
    return ReproductionRun(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())
