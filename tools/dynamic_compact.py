#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PREFIX = "__HC_DYNAMIC_RESULT_JSON__ "


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
    }


def summarize_dynamic_suite(input_dir: Path) -> dict[str, Any]:
    sub_summaries = {
        "smoke": summarize_smoke(input_dir / "smoke"),
        "quick": summarize_quick(input_dir / "quick"),
        "bandwidth": summarize_bandwidth(input_dir / "bandwidth"),
        "collective_bandwidth": summarize_collective_bandwidth(input_dir / "collective_bandwidth"),
    }
    failed = [
        name
        for name, summary in sub_summaries.items()
        if not summary.get("correctness_pass", False) or not summary.get("performance_pass", False)
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
        "failed_stages": failed,
        "correctness_pass": not failed,
        "performance_pass": not failed,
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
    else:
        raise ValueError(f"unsupported kind: {args.kind}")

    compact["returncode"] = args.returncode
    if args.returncode != 0:
        compact["correctness_pass"] = False
        compact["performance_pass"] = False
        compact["error_type"] = compact.get("error_type") or f"returncode={args.returncode}"

    return {
        "schema_version": 1,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="emit compact dynamic healthcheck result")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=["smoke", "quick", "bandwidth", "collective-bandwidth", "dynamic-suite"],
        required=True,
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--returncode", type=int, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--pod-name", default="")
    parser.add_argument("--node-name", default="")
    parser.add_argument("--pod-ip", default="")
    parser.add_argument("--host-ip", default="")
    args = parser.parse_args()
    print(PREFIX + json.dumps(build_payload(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
