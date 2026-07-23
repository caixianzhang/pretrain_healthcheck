#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from tools.dynamic_frame import atomic_write_frame, encode_v2_frame


def csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def source_file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    record: dict[str, Any] = {"path": relative, "exists": path.is_file(), "bytes": 0, "sha256": "", "rows": 0}
    if not path.is_file():
        return record
    data = path.read_bytes()
    record["bytes"] = len(data)
    record["sha256"] = hashlib.sha256(data).hexdigest()
    if relative.endswith(".jsonl"):
        record["rows"] = sum(1 for line in data.splitlines() if line.strip())
    return record


def source_glob_record(root: Path, relative_pattern: str) -> dict[str, Any]:
    paths = sorted(root.glob(relative_pattern))
    digest = hashlib.sha256()
    total_bytes = 0
    total_rows = 0
    for path in paths:
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        total_bytes += len(data)
        total_rows += sum(1 for line in data.splitlines() if line.strip())
    return {
        "path": relative_pattern,
        "exists": bool(paths),
        "files": len(paths),
        "bytes": total_bytes,
        "sha256": digest.hexdigest() if paths else "",
        "rows": total_rows,
    }


def source_manifest(input_dir: Path, kind: str) -> list[dict[str, Any]]:
    by_kind = {
        "smoke": ["ping_summary.json"],
        "quick": ["rank_detail.jsonl", "group_summary.jsonl"],
        "bandwidth": ["bandwidth_summary.jsonl"],
        "collective-bandwidth": ["collective_bandwidth_summary.jsonl"],
        "dynamic-suite": [
            "dynamic_suite_plan.json",
            "smoke/ping_summary.json",
            "quick/rank_detail.jsonl",
            "quick/group_summary.jsonl",
            "bandwidth/bandwidth_summary.jsonl",
            "collective_bandwidth/collective_bandwidth_summary.jsonl",
        ],
        "training-topology": [
            "training_topology_plan.json",
            "training_topology_gate.json",
        ],
    }
    records = [source_file_record(input_dir, relative) for relative in by_kind[kind]]
    if kind == "training-topology":
        records.append(source_glob_record(input_dir, "training_topology_rank_summaries/rank_*.jsonl"))
    return records


def coverage_manifest(args: argparse.Namespace, compact: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ops = csv_values(args.expected_collective_ops)
    expected_sizes = csv_values(args.expected_collective_message_sizes)
    expected_patterns = csv_values(args.expected_collective_moe_patterns)
    expected_bandwidth_sizes = csv_values(args.expected_bandwidth_message_sizes)
    expected_ranks = args.expected_ranks
    suite_plan = read_json(args.input_dir / "dynamic_suite_plan.json") if args.kind == "dynamic-suite" else {}
    topology_plan = read_json(args.input_dir / "training_topology_plan.json") if args.kind == "training-topology" else {}
    if suite_plan:
        expected_ranks = int(suite_plan.get("expected_world_size", expected_ranks) or 0)
        expected_bandwidth_sizes = [str(value) for value in suite_plan.get("bandwidth_message_sizes", [])]
        expected_sizes = [str(value) for value in suite_plan.get("collective_message_sizes", [])]
        expected_ops = [str(value) for value in suite_plan.get("collective_ops", [])]
        expected_patterns = [str(value) for value in suite_plan.get("collective_moe_patterns", [])]
    elif topology_plan:
        expected_ranks = int(topology_plan.get("world_size", expected_ranks) or 0)
    actual_cases = compact.get("case_metrics", []) if isinstance(compact.get("case_metrics"), list) else []
    summary_owner = compact.get("summary_owner") is not False
    errors: list[str] = []
    required_missing = [str(row["path"]) for row in sources if not row.get("exists")]
    if summary_owner and required_missing:
        errors.append("missing source files: " + ",".join(required_missing))

    expected_case_count = 0
    if args.kind == "dynamic-suite" and suite_plan:
        expected_case_count = int(suite_plan.get("expected_case_count", 0) or 0)
    elif args.kind == "training-topology" and topology_plan:
        expected_case_count = int(topology_plan.get("case_count", 0) or 0)
    elif args.kind == "dynamic-suite" and expected_sizes and expected_ops:
        collective_per_size = sum(len(expected_patterns) if op == "all_to_allv" else 1 for op in expected_ops)
        expected_case_count = len(expected_bandwidth_sizes) + len(expected_sizes) * collective_per_size
    elif args.kind == "collective-bandwidth" and expected_sizes and expected_ops:
        per_size = sum(len(expected_patterns) if op == "all_to_allv" else 1 for op in expected_ops)
        expected_case_count = len(expected_sizes) * per_size
    elif args.kind == "bandwidth" and expected_bandwidth_sizes:
        expected_case_count = len(expected_bandwidth_sizes)

    if summary_owner and args.kind != "training-topology" and expected_case_count and len(actual_cases) != expected_case_count:
        errors.append(f"case count mismatch: expected={expected_case_count} actual={len(actual_cases)}")
    if summary_owner and expected_ranks > 0 and args.kind in {"smoke", "dynamic-suite", "training-topology"}:
        if args.kind == "training-topology":
            actual_ranks = int(compact.get("rank_count", 0) or 0)
        else:
            smoke = compact.get("sub_summaries", {}).get("smoke", {}) if args.kind == "dynamic-suite" else compact
            actual_ranks = int(smoke.get("rank_count", 0) or 0) if isinstance(smoke, dict) else 0
        if actual_ranks != expected_ranks:
            errors.append(f"rank count mismatch: expected={expected_ranks} actual={actual_ranks}")

    case_ids = [
        "/".join(
            [
                str(row.get("stage", "")),
                str(row.get("topology_family", "")),
                str(row.get("topology_group_id", "")),
                str(row.get("op_type", "")),
                str(row.get("requested_message_bytes", row.get("message_bytes", ""))),
                str(row.get("payload_pattern", "none")),
                str(row.get("collective_group_size", "")),
            ]
        )
        for row in actual_cases
        if isinstance(row, dict)
    ]
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate case identifiers")
    return {
        "complete": not errors,
        "summary_owner": summary_owner,
        "errors": errors,
        "expected": {
            "ranks": expected_ranks,
            "bandwidth_message_sizes": expected_bandwidth_sizes,
            "collective_message_sizes": expected_sizes,
            "collective_ops": expected_ops,
            "collective_moe_patterns": expected_patterns,
            "case_count": expected_case_count,
        },
        "actual": {"case_count": len(actual_cases), "unique_case_count": len(set(case_ids))},
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def number_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def max_int(rows: list[dict[str, Any]], key: str) -> int:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int):
            values.append(value)
        elif isinstance(value, float):
            values.append(int(value))
    return max(values) if values else 0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def compact_case_metrics(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    fields = [
        "op_type",
        "message_size",
        "message_bytes",
        "requested_message_bytes",
        "payload_pattern",
        "collective_group_size",
        "dtype",
        "timing_mode",
        "measurement_batches",
        "iterations_per_batch",
        "latency_p50",
        "latency_p95",
        "latency_p99",
        "avg_busbw",
        "second_lowest_busbw",
        "correctness_pass",
        "performance_pass",
        "error_type",
        "diagnostic_timing_mode",
        "diagnostic_latency_p50",
        "diagnostic_latency_p95",
        "diagnostic_latency_p99",
        "diagnostic_slowest_ranks",
        "topology_family",
        "topology_group_id",
        "subgroup_count",
        "aggregation",
        "case_source",
    ]
    compact: list[dict[str, Any]] = []
    for row in rows:
        item = {"stage": str(row.get("source_stage", stage))}
        for field in fields:
            if field in row:
                item[field] = row[field]
        item.setdefault("payload_pattern", "none")
        item.setdefault("collective_group_size", 0)
        compact.append(item)
    return compact


def summarize_quick(input_dir: Path) -> dict[str, Any]:
    ranks = read_jsonl(input_dir / "rank_detail.jsonl")
    groups = read_jsonl(input_dir / "group_summary.jsonl")
    latencies = number_values(groups, "latency_p50") or number_values(ranks, "latency")
    gemm = number_values(ranks, "gemm_tflops")
    memory = number_values(ranks, "memory_bandwidth")
    failed_groups = [row for row in groups if not row.get("correctness_pass", True)]
    failed_ranks = [row for row in ranks if row.get("error_type") or row.get("nan_count", 0) or row.get("inf_count", 0)]
    return {
        "kind": "quick",
        "rank_count": len(ranks),
        "group_count": len(groups),
        "op_types": sorted({str(row.get("op_type", "")) for row in groups if row.get("op_type")}),
        "correctness_pass": not failed_groups and not failed_ranks,
        "performance_pass": all(row.get("performance_pass", True) for row in groups),
        "nan_count": max_int(ranks, "nan_count"),
        "inf_count": max_int(ranks, "inf_count"),
        "error_type": ",".join(sorted({str(row.get("error_type")) for row in failed_ranks if row.get("error_type")})),
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "latency_p99": percentile(latencies, 0.99),
        "gemm_tflops_avg": statistics.fmean(gemm) if gemm else None,
        "memory_bandwidth_avg": statistics.fmean(memory) if memory else None,
    }


def summarize_smoke(input_dir: Path) -> dict[str, Any]:
    summary = read_json(input_dir / "ping_summary.json")
    return {
        "kind": "smoke",
        "correctness_pass": summary.get("status") == "PASS",
        "performance_pass": True,
        "rank_count": len(summary.get("ranks", []) or []),
        "expected_all_reduce_value": summary.get("expected_all_reduce_value"),
    }


def summarize_bandwidth(input_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(input_dir / "bandwidth_summary.jsonl")
    return {
        "kind": "bandwidth",
        "summary_count": len(rows),
        "correctness_pass": all(row.get("correctness_pass", True) for row in rows),
        "performance_pass": all(row.get("performance_pass", True) for row in rows),
        "op_types": sorted({str(row.get("op_type", "")) for row in rows if row.get("op_type")}),
        "message_sizes": sorted({str(row.get("message_size", "")) for row in rows if row.get("message_size")}),
        "second_lowest_busbw_min": min(number_values(rows, "second_lowest_busbw") or [0.0]),
        "avg_busbw_min": min(number_values(rows, "avg_busbw") or [0.0]),
        "latency_p50": percentile(number_values(rows, "latency_p50"), 0.50),
        "latency_p95": percentile(number_values(rows, "latency_p95"), 0.95),
        "latency_p99": percentile(number_values(rows, "latency_p99"), 0.99),
        "case_metrics": compact_case_metrics(rows, "bandwidth"),
    }


def summarize_collective_bandwidth(input_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(input_dir / "collective_bandwidth_summary.jsonl")
    return {
        "kind": "collective_bandwidth",
        "summary_count": len(rows),
        "correctness_pass": all(row.get("correctness_pass", True) for row in rows),
        "performance_pass": all(row.get("performance_pass", True) for row in rows),
        "op_types": sorted({str(row.get("op_type", "")) for row in rows if row.get("op_type")}),
        "message_sizes": sorted({str(row.get("message_size", "")) for row in rows if row.get("message_size")}),
        "payload_patterns": sorted({str(row.get("payload_pattern", "")) for row in rows if row.get("payload_pattern")}),
        "second_lowest_busbw_min": min(number_values(rows, "second_lowest_busbw") or [0.0]),
        "avg_busbw_min": min(number_values(rows, "avg_busbw") or [0.0]),
        "latency_p50": percentile(number_values(rows, "latency_p50"), 0.50),
        "latency_p95": percentile(number_values(rows, "latency_p95"), 0.95),
        "latency_p99": percentile(number_values(rows, "latency_p99"), 0.99),
        "case_metrics": compact_case_metrics(rows, "collective_bandwidth"),
    }


def summarize_dynamic_suite(input_dir: Path) -> dict[str, Any]:
    sub_summaries = {
        "smoke": summarize_smoke(input_dir / "smoke"),
        "quick": summarize_quick(input_dir / "quick"),
        "bandwidth": summarize_bandwidth(input_dir / "bandwidth"),
        "collective_bandwidth": summarize_collective_bandwidth(input_dir / "collective_bandwidth"),
    }
    correctness_failed = [
        name
        for name, summary in sub_summaries.items()
        if not summary.get("correctness_pass", False)
    ]
    performance_failed = [
        name
        for name, summary in sub_summaries.items()
        if not summary.get("performance_pass", True)
    ]
    error_types = sorted(
        {
            str(summary.get("error_type"))
            for summary in sub_summaries.values()
            if summary.get("error_type")
        }
    )
    quick = sub_summaries["quick"]
    bandwidth = sub_summaries["bandwidth"]
    collective = sub_summaries["collective_bandwidth"]
    return {
        "kind": "dynamic-suite",
        "sub_summaries": sub_summaries,
        "failed_stages": sorted(set(correctness_failed + performance_failed)),
        "correctness_failed_stages": correctness_failed,
        "performance_failed_stages": performance_failed,
        "correctness_pass": not correctness_failed,
        "performance_pass": not performance_failed,
        "error_type": ",".join(error_types),
        "rank_count": max(int(summary.get("rank_count", 0) or 0) for summary in sub_summaries.values()),
        "gemm_tflops_avg": quick.get("gemm_tflops_avg"),
        "memory_bandwidth_avg": quick.get("memory_bandwidth_avg"),
        "second_lowest_busbw_min": min(
            [
                float(value)
                for value in [
                    bandwidth.get("second_lowest_busbw_min"),
                    collective.get("second_lowest_busbw_min"),
                ]
                if isinstance(value, (int, float))
            ]
            or [0.0]
        ),
        "avg_busbw_min": min(
            [
                float(value)
                for value in [
                    bandwidth.get("avg_busbw_min"),
                    collective.get("avg_busbw_min"),
                ]
                if isinstance(value, (int, float))
            ]
            or [0.0]
        ),
        "case_metrics": list(bandwidth.get("case_metrics", [])) + list(collective.get("case_metrics", [])),
    }


def summarize_training_topology(input_dir: Path) -> dict[str, Any]:
    plan = read_json(input_dir / "training_topology_plan.json")
    gate = read_json(input_dir / "training_topology_gate.json")
    rows: list[dict[str, Any]] = []
    for path in sorted((input_dir / "training_topology_rank_summaries").glob("rank_*.jsonl")):
        rows.extend(read_jsonl(path))
    correctness_pass = gate.get("status") == "PASS" and all(row.get("correctness_pass", False) for row in rows)
    return {
        "kind": "training-topology",
        "rank_count": int(plan.get("world_size", 0) or 0),
        "summary_count": len(rows),
        "correctness_pass": correctness_pass,
        "performance_pass": all(row.get("performance_pass", True) for row in rows),
        "error_type": "" if correctness_pass else "TOPOLOGY_CORRECTNESS_FAIL",
        "manifest_sha256": str(plan.get("manifest_sha256", "")),
        "topology_group_counts": plan.get("group_counts", {}),
        "case_metrics": compact_case_metrics(rows, "training_topology"),
        "summary_owner": bool(rows),
    }


def summarize_comm_path(input_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(input_dir / "comm_path_summary.jsonl")
    if not rows:
        summary_json = input_dir / "comm_path_summary.json"
        if summary_json.exists():
            try:
                data = json.loads(summary_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict):
                rows = [row for row in data.get("rows", []) if isinstance(row, dict)]
    if not rows:
        return {}
    env_keys = sorted({key for row in rows for key in (row.get("env") or {}).keys()})
    return {
        "rank_count": len(rows),
        "nodes": sorted({str(row.get("node_name", "")) for row in rows if row.get("node_name")}),
        "pods": sorted({str(row.get("pod_name", "")) for row in rows if row.get("pod_name")}),
        "dist_backends": sorted({str(row.get("dist_backend", "")) for row in rows if row.get("dist_backend")}),
        "comm_runtimes": sorted({str(row.get("comm_runtime", "")) for row in rows if row.get("comm_runtime")}),
        "env_keys": env_keys,
        "rows": rows,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir
    if args.kind == "smoke":
        compact = summarize_smoke(input_dir)
    elif args.kind == "quick":
        compact = summarize_quick(input_dir)
    elif args.kind == "bandwidth":
        compact = summarize_bandwidth(input_dir)
    elif args.kind == "collective-bandwidth":
        compact = summarize_collective_bandwidth(input_dir)
    elif args.kind == "dynamic-suite":
        compact = summarize_dynamic_suite(input_dir)
    elif args.kind == "training-topology":
        compact = summarize_training_topology(input_dir)
    else:
        raise ValueError(f"unsupported kind: {args.kind}")

    compact["returncode"] = args.returncode
    no_owned_summary = (
        int(compact.get("rank_count", 0) or 0) == 0
        and int(compact.get("summary_count", 0) or 0) == 0
        and not compact.get("case_metrics")
    )
    if args.returncode == 0 and "multi_node" in args.stage and no_owned_summary:
        compact["correctness_pass"] = True
        compact["performance_pass"] = True
        compact["error_type"] = ""
        compact["summary_owner"] = False
    if args.returncode != 0:
        compact["correctness_pass"] = False
        compact["performance_pass"] = False
        compact["error_type"] = compact.get("error_type") or f"returncode={args.returncode}"

    sources = source_manifest(input_dir, args.kind)
    coverage = coverage_manifest(args, compact, sources)
    if args.returncode == 0 and not coverage["complete"]:
        compact["correctness_pass"] = False
        compact["performance_pass"] = False
        compact["error_type"] = "DATA_INCOMPLETE"
    payload = {
        "schema_version": 2,
        "pod": {
            "name": args.pod_name,
            "node_name": args.node_name,
            "pod_ip": args.pod_ip,
            "host_ip": args.host_ip,
            "run_id": args.run_id,
            "stage": args.stage,
        },
        "local_workdir": str(input_dir),
        "summary": compact,
        "coverage": coverage,
        "source_files": sources,
    }
    comm_path = summarize_comm_path(input_dir)
    if comm_path:
        payload["comm_path_summary"] = comm_path
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="emit compact dynamic healthcheck result")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=["smoke", "quick", "bandwidth", "collective-bandwidth", "dynamic-suite", "training-topology"],
        required=True,
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--returncode", type=int, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--pod-name", default="")
    parser.add_argument("--node-name", default="")
    parser.add_argument("--pod-ip", default="")
    parser.add_argument("--host-ip", default="")
    parser.add_argument("--expected-ranks", type=int, default=0)
    parser.add_argument("--expected-bandwidth-message-sizes", default="")
    parser.add_argument("--expected-collective-message-sizes", default="")
    parser.add_argument("--expected-collective-ops", default="")
    parser.add_argument("--expected-collective-moe-patterns", default="")
    parser.add_argument("--frame-output", type=Path)
    parser.add_argument("--no-stdout", action="store_true")
    args = parser.parse_args()
    frame = encode_v2_frame(build_payload(args))
    if args.frame_output:
        atomic_write_frame(args.frame_output, frame)
        print(f"[dynamic-compact] sidecar: {args.frame_output}", file=sys.stderr)
    if not args.no_stdout:
        print(frame)


if __name__ == "__main__":
    main()
