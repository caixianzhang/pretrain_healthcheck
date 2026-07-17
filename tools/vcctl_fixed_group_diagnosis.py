#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TOOLS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.vcctl_multi_node_batch import choose_container, ensure_env, parse_json_stream, pod_from_raw


DEFAULT_PHASES = "pairwise,ep8,scale16,scale32,scale64"
DEFAULT_GROUP_IDS = (
    "scale64_r1_group_0000,scale64_r1_group_0001,"
    "final_all_group_0000_split_0,final_all_group_0000_split_1"
)


@dataclass(frozen=True)
class FixedGroup:
    group_id: str
    members: tuple[dict[str, Any], ...]
    ordered_node_names_sha256: str

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(str(member["node_name"]) for member in self.members)


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def load_fixed_groups(path: Path, requested_ids: list[str], expected_size: int) -> list[FixedGroup]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_groups = payload.get("failed_groups")
    if manifest_groups is None:
        manifest_groups = payload.get("groups", [])
    available = {str(item.get("group_id", "")): item for item in manifest_groups}
    missing = [group_id for group_id in requested_ids if group_id not in available]
    if missing:
        raise ValueError(f"groups absent from manifest: {','.join(missing)}")

    groups: list[FixedGroup] = []
    for group_id in requested_ids:
        item = available[group_id]
        members = tuple(sorted(item.get("members", []), key=lambda member: int(member["node_rank"])))
        nodes = [str(member["node_name"]) for member in members]
        if len(nodes) != expected_size or len(set(nodes)) != expected_size:
            raise ValueError(f"{group_id}: expected {expected_size} unique nodes, got {len(nodes)}")
        digest = hashlib.sha256("\n".join(nodes).encode("utf-8")).hexdigest()
        expected_digest = str(item.get("ordered_node_names_sha256", ""))
        if expected_digest and digest != expected_digest:
            raise ValueError(f"{group_id}: ordered node digest mismatch")
        groups.append(FixedGroup(group_id, members, digest))
    return groups


def schedule_disjoint_rounds(groups: list[FixedGroup]) -> list[list[FixedGroup]]:
    rounds: list[list[FixedGroup]] = []
    used_nodes: list[set[str]] = []
    for group in groups:
        nodes = set(group.nodes)
        for index, occupied in enumerate(used_nodes):
            if occupied.isdisjoint(nodes):
                rounds[index].append(group)
                occupied.update(nodes)
                break
        else:
            rounds.append([group])
            used_nodes.append(set(nodes))
    return rounds


def load_current_pods(vcctl_bin: str, job_name: str, namespace: str) -> dict[str, Any]:
    proc = subprocess.run(
        [vcctl_bin, "pod", "get", "--job", job_name, "-n", namespace, "-o", "json"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vcctl pod get failed rc={proc.returncode}: {proc.stderr.strip()}")
    pods = [pod_from_raw(obj) for obj in parse_json_stream(proc.stdout)]
    return {pod.node_name: pod for pod in pods if pod is not None}


def write_group_pod_json(group: FixedGroup, current_pods: dict[str, Any], path: Path) -> None:
    selected = []
    for member in group.members:
        node_name = str(member["node_name"])
        pod = current_pods.get(node_name)
        if pod is None:
            raise ValueError(f"{group.group_id}: node absent from current job: {node_name}")
        expected_pod = str(member.get("pod_name", ""))
        expected_host_ip = str(member.get("host_ip", ""))
        if expected_pod and pod.pod_name != expected_pod:
            raise ValueError(
                f"{group.group_id}: pod changed for {node_name}: expected={expected_pod} actual={pod.pod_name}"
            )
        if expected_host_ip and pod.host_ip != expected_host_ip:
            raise ValueError(
                f"{group.group_id}: host IP changed for {node_name}: expected={expected_host_ip} actual={pod.host_ip}"
            )
        selected.append(pod)

    master_addr = selected[0].pod_ip or selected[0].pod_name
    items = []
    for rank, pod in enumerate(selected):
        raw = copy.deepcopy(pod.raw)
        container = choose_container(raw)
        ensure_env(container, "RANK", str(rank))
        ensure_env(container, "WORLD_SIZE", str(len(selected)))
        ensure_env(container, "MASTER_ADDR", master_addr)
        ensure_env(container, "MASTER_PORT", "29500")
        items.append(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")


def stream_output(proc: subprocess.Popen[str], group_id: str, log_path: Path) -> None:
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            print(f"[{group_id}] {line}", end="", flush=True)


def classify_batch_summary(summary: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    phase_counts = summary.get("phase_status_counts", [])
    failed_phases: dict[str, int] = {}
    passed_phases: dict[str, int] = {}
    for item in phase_counts:
        phase = str(item.get("phase", ""))
        status = str(item.get("status", ""))
        count = int(item.get("count", 0) or 0)
        if status in {"FAIL", "TIMEOUT"}:
            failed_phases[phase] = failed_phases.get(phase, 0) + count
        elif status == "PASS":
            passed_phases[phase] = passed_phases.get(phase, 0) + count

    suspect_path = result_dir / "suspect_nodes.txt"
    suspect_nodes = []
    if suspect_path.exists():
        suspect_nodes = [line.strip() for line in suspect_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    scale64_failed = failed_phases.get("scale64", 0) > 0
    final_all_failed = failed_phases.get("final_all", 0) > 0
    lower_scale_failures = {
        phase: count
        for phase, count in failed_phases.items()
        if phase not in {"scale64", "final_all"}
    }
    if scale64_failed and final_all_failed:
        diagnosis_status = "SCALE_REPRODUCED"
    elif scale64_failed or final_all_failed:
        diagnosis_status = "SCALE_FAILURE"
    elif lower_scale_failures:
        diagnosis_status = "LOWER_SCALE_FAILURE"
    elif suspect_nodes or summary.get("overall_status") == "SUSPECT":
        diagnosis_status = "SUSPECT"
    elif summary.get("overall_status") == "PASS":
        diagnosis_status = "PASS"
    else:
        diagnosis_status = str(summary.get("overall_status", "MISSING"))

    return {
        "diagnosis_status": diagnosis_status,
        "batch_overall_status": summary.get("overall_status", "MISSING"),
        "failed_phases": failed_phases,
        "passed_phases": passed_phases,
        "lower_scale_failures": lower_scale_failures,
        "suspect_nodes": suspect_nodes,
    }


def run_child(group: FixedGroup, args: argparse.Namespace, diag_dir: Path, work_dir: Path) -> dict[str, Any]:
    group_json = work_dir / "group_pods" / f"{safe_name(group.group_id)}.json"
    child_result_root = diag_dir
    child_run_id = safe_name(group.group_id)
    log_path = work_dir / "driver_logs" / f"{child_run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "JOB_NAME": args.job_name,
            "NAMESPACE": args.namespace,
            "VCCTL_BIN": args.vcctl_bin,
            "POD_JSON_FILE": str(group_json),
            "PRESERVE_POD_JSON_ORDER": "1",
            "RESULT_ROOT": str(child_result_root),
            "BATCH_RUN_ID": child_run_id,
            "TARGET_SCALE": str(args.expected_group_size),
            "PHASES": args.phases,
            "GROUP_SEED": str(args.group_seed),
            "PAIRWISE_MESSAGE_SIZES": args.message_sizes,
            "GROUP_TIMEOUT_SECONDS": str(args.group_timeout_seconds),
            "FINAL_ALL_TIMEOUT_SECONDS": str(args.group_timeout_seconds),
            "PROGRESS_INTERVAL_SECONDS": str(args.progress_interval_seconds),
            "PHASE_GROUP_CONCURRENCY": "0",
            "DRY_RUN": args.dry_run,
            "PRE_CLEAN": args.pre_clean,
            "DYNAMIC_COMPARE": args.dynamic_compare,
            "DYNAMIC_COMPARE_AUTO_RETEST": "1",
            "BATCH_RUNTIME_WARN_SECONDS": str(args.runtime_warn_seconds),
            "DISABLE_FINAL_SUPERSET_SKIP": "1",
            "COMM_PATH_DEBUG": args.comm_path_debug,
            "POD_PROJECT_DIR": args.pod_project_dir,
            "GROUP_OUTPUT_ROOT": str(Path(args.group_output_root) / args.diag_run_id),
            "FAILED_GROUP_OUTPUT_MODE": "local-link",
        }
    )
    started = time.monotonic()
    print(f"[fixed-group-diagnosis] group_start group={group.group_id} nodes={len(group.nodes)}", flush=True)
    proc = subprocess.Popen(
        ["bash", str(Path(args.project_dir) / "scripts/metax/run_vcctl_multi_node_batch_healthcheck.sh")],
        cwd=args.project_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_thread = threading.Thread(target=stream_output, args=(proc, group.group_id, log_path), daemon=True)
    output_thread.start()
    while proc.poll() is None:
        time.sleep(args.progress_interval_seconds)
        print(
            f"[fixed-group-diagnosis] progress group={group.group_id} "
            f"elapsed={int(time.monotonic() - started)}s",
            flush=True,
        )
    output_thread.join()
    elapsed = round(time.monotonic() - started, 3)
    summary_path = diag_dir / child_run_id / "batch_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result_dir = diag_dir / child_run_id
    classification = classify_batch_summary(summary, result_dir)
    result = {
        "group_id": group.group_id,
        "returncode": int(proc.returncode or 0),
        "elapsed_seconds": elapsed,
        "ordered_node_names_sha256": group.ordered_node_names_sha256,
        "result_dir": str(result_dir),
        "driver_log": str(log_path),
        "overall_status": classification["diagnosis_status"],
        "batch_summary": summary,
        **classification,
    }
    print(
        f"[fixed-group-diagnosis] group_done group={group.group_id} "
        f"status={result['overall_status']} rc={result['returncode']} elapsed={elapsed}s",
        flush=True,
    )
    return result


def write_aggregate(diag_dir: Path, args: argparse.Namespace, rounds: list[list[FixedGroup]], results: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "job_name": args.job_name,
        "diag_run_id": args.diag_run_id,
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "phases": csv_values(args.phases),
        "message_sizes": csv_values(args.message_sizes),
        "rounds": [[group.group_id for group in round_groups] for round_groups in rounds],
        "results": results,
    }
    (diag_dir / "diagnosis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Fixed 64-Node Group Diagnosis",
        "",
        f"- job: `{args.job_name}`",
        f"- diag_run_id: `{args.diag_run_id}`",
        f"- phases: `{args.phases}`",
        f"- message_sizes: `{args.message_sizes}`",
        f"- group_timeout_seconds: `{args.group_timeout_seconds}`",
        "",
        "## Scheduling",
        "",
    ]
    for index, round_groups in enumerate(rounds, start=1):
        lines.append(f"- round {index}: " + ", ".join(f"`{group.group_id}`" for group in round_groups))
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| source group | diagnosis | batch status | failed phases | suspect nodes | elapsed seconds | result directory |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for result in results:
        lines.append(
            f"| `{result['group_id']}` | {result['diagnosis_status']} | {result['batch_overall_status']} | "
            f"{', '.join(f'{phase}:{count}' for phase, count in result['failed_phases'].items()) or '-'} | "
            f"{', '.join(result['suspect_nodes']) or '-'} | {result['elapsed_seconds']} | `{result['result_dir']}` |"
        )
    lines.append("")
    (diag_dir / "diagnosis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run layered diagnosis inside fixed failed node groups")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--vcctl-bin", default="vcctl")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--group-ids", default=DEFAULT_GROUP_IDS)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--diag-run-id", default="")
    parser.add_argument("--expected-group-size", type=int, default=64)
    parser.add_argument("--phases", default=DEFAULT_PHASES)
    parser.add_argument("--message-sizes", default="1M,128M,1G")
    parser.add_argument("--group-seed", type=int, default=20260706)
    parser.add_argument("--group-timeout-seconds", type=int, default=180)
    parser.add_argument("--progress-interval-seconds", type=int, default=10)
    parser.add_argument("--runtime-warn-seconds", type=int, default=900)
    parser.add_argument("--dry-run", default="1")
    parser.add_argument("--pre-clean", default="1")
    parser.add_argument("--dynamic-compare", default="1")
    parser.add_argument("--comm-path-debug", default="1")
    parser.add_argument("--pod-project-dir", default="")
    parser.add_argument("--group-output-root", default="/tmp/pretrain_healthcheck_fixed_group_outputs/vcctl")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.diag_run_id:
        args.diag_run_id = "fixed64_diagnosis_" + time.strftime("%Y%m%d_%H%M%S")
    if args.group_timeout_seconds <= 0 or args.progress_interval_seconds <= 0:
        raise ValueError("timeouts and progress interval must be positive")

    group_ids = csv_values(args.group_ids)
    groups = load_fixed_groups(Path(args.source_manifest), group_ids, args.expected_group_size)
    rounds = schedule_disjoint_rounds(groups)
    current_pods = load_current_pods(args.vcctl_bin, args.job_name, args.namespace)

    diag_dir = Path(args.result_root) / args.diag_run_id
    work_dir = Path("/tmp") / "pretrain_healthcheck_fixed_group_diagnosis" / args.diag_run_id
    diag_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        write_group_pod_json(group, current_pods, work_dir / "group_pods" / f"{safe_name(group.group_id)}.json")

    source_payload = {
        "job_name": args.job_name,
        "diag_run_id": args.diag_run_id,
        "groups": [
            {
                "group_id": group.group_id,
                "ordered_node_names_sha256": group.ordered_node_names_sha256,
                "members": list(group.members),
            }
            for group in groups
        ],
        "rounds": [[group.group_id for group in round_groups] for round_groups in rounds],
    }
    (diag_dir / "source_groups.json").write_text(
        json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[fixed-group-diagnosis] job={args.job_name}", flush=True)
    print(f"[fixed-group-diagnosis] diag_run_id={args.diag_run_id}", flush=True)
    print(f"[fixed-group-diagnosis] result_dir={diag_dir}", flush=True)
    print(f"[fixed-group-diagnosis] rounds={len(rounds)} groups={len(groups)}", flush=True)

    results: list[dict[str, Any]] = []
    for round_index, round_groups in enumerate(rounds, start=1):
        print(
            f"[fixed-group-diagnosis] round_start round={round_index} "
            f"groups={','.join(group.group_id for group in round_groups)}",
            flush=True,
        )
        round_results: list[dict[str, Any]] = []
        threads: list[threading.Thread] = []
        lock = threading.Lock()

        def worker(group: FixedGroup) -> None:
            try:
                result = run_child(group, args, diag_dir, work_dir)
            except Exception as exc:
                result = {
                    "group_id": group.group_id,
                    "returncode": 1,
                    "elapsed_seconds": 0.0,
                    "ordered_node_names_sha256": group.ordered_node_names_sha256,
                    "result_dir": str(diag_dir / safe_name(group.group_id)),
                    "overall_status": "DRIVER_ERROR",
                    "diagnosis_status": "DRIVER_ERROR",
                    "batch_overall_status": "MISSING",
                    "failed_phases": {},
                    "passed_phases": {},
                    "lower_scale_failures": {},
                    "suspect_nodes": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(
                    f"[fixed-group-diagnosis] group_error group={group.group_id} error={result['error']}",
                    flush=True,
                )
            with lock:
                round_results.append(result)

        for group in round_groups:
            thread = threading.Thread(target=worker, args=(group,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        results.extend(sorted(round_results, key=lambda item: group_ids.index(item["group_id"])))
        write_aggregate(diag_dir, args, rounds, results)
        print(f"[fixed-group-diagnosis] round_done round={round_index}", flush=True)

    write_aggregate(diag_dir, args, rounds, results)
    failed = [result for result in results if result["overall_status"] != "PASS"]
    print(
        f"[fixed-group-diagnosis] done groups={len(results)} non_pass={len(failed)} "
        f"summary={diag_dir / 'diagnosis_summary.md'}",
        flush=True,
    )
    return 0 if not failed or args.dry_run == "1" else 1


if __name__ == "__main__":
    raise SystemExit(main())
