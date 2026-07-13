#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def metric_value(row: dict[str, Any], key: str) -> float | None:
    value = (row.get("summary") or {}).get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def compare_dynamic_results(result_dir: Path, ratio_threshold: float) -> dict[str, Any]:
    facts = read_jsonl(result_dir / "dynamic_facts.jsonl")
    failed = read_jsonl(result_dir / "dynamic_failed_pods.jsonl")
    issues: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []

    for row in failed:
        issue = {
            "severity": "FAIL",
            "pod_name": row.get("pod_name", ""),
            "node_name": row.get("node_name", ""),
            "check": "dynamic_failed_pod",
            "reason": row.get("reason", ""),
            "actual": row.get("error_type", ""),
            "expected": "successful dynamic compact frame",
        }
        issues.append(issue)
        outliers.append(issue)

    for row in facts:
        summary = row.get("summary") or {}
        if summary.get("summary_owner") is False:
            continue
        if not summary.get("correctness_pass", False):
            issue = {
                "severity": "FAIL",
                "pod_name": row.get("pod", {}).get("name", ""),
                "node_name": row.get("pod", {}).get("node_name", ""),
                "check": "correctness_pass",
                "reason": summary.get("error_type", "correctness failed"),
                "actual": False,
                "expected": True,
            }
            issues.append(issue)
            outliers.append(issue)
        for stage in summary.get("failed_stages", []) or []:
            issue = {
                "severity": "FAIL",
                "pod_name": row.get("pod", {}).get("name", ""),
                "node_name": row.get("pod", {}).get("node_name", ""),
                "check": f"dynamic_suite/{stage}",
                "reason": "dynamic suite sub-stage failed",
                "actual": "FAIL",
                "expected": "PASS",
            }
            issues.append(issue)
            outliers.append(issue)

    for key in [
        "gemm_tflops_avg",
        "memory_bandwidth_avg",
        "second_lowest_busbw_min",
        "avg_busbw_min",
    ]:
        values = [value for value in (metric_value(row, key) for row in facts) if value is not None and value > 0]
        if len(values) < 3:
            continue
        median = statistics.median(values)
        if median <= 0:
            continue
        cutoff = median * ratio_threshold
        for row in facts:
            value = metric_value(row, key)
            if value is None or value >= cutoff:
                continue
            issue = {
                "severity": "SUSPECT",
                "pod_name": row.get("pod", {}).get("name", ""),
                "node_name": row.get("pod", {}).get("node_name", ""),
                "check": key,
                "reason": f"value below {ratio_threshold:.2f}x median",
                "actual": value,
                "expected": f">= {cutoff:.6f}",
                "median": median,
            }
            issues.append(issue)
            outliers.append(issue)

    status = "FAIL" if any(row["severity"] == "FAIL" for row in issues) else "PASS"
    report = {
        "dynamic_compare_status": status,
        "pod_count": len(facts),
        "failed_pod_count": len(failed),
        "issue_count": len(issues),
        "outlier_count": len(outliers),
        "issues": issues,
        "ratio_threshold": ratio_threshold,
    }
    return report


def write_report(result_dir: Path, report: dict[str, Any]) -> None:
    (result_dir / "dynamic_compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(result_dir / "dynamic_outliers.jsonl", report.get("issues", []))

    lines = [
        "# Dynamic Compare Summary",
        "",
        f"- dynamic_compare_status: `{report.get('dynamic_compare_status')}`",
        f"- pod_count: `{report.get('pod_count')}`",
        f"- failed_pod_count: `{report.get('failed_pod_count')}`",
        f"- issue_count: `{report.get('issue_count')}`",
        "",
        "## Issues",
        "",
        "| severity | pod | node | check | expected | actual | reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    issues = report.get("issues") or []
    if not issues:
        lines.append("|  |  |  |  |  |  | no issues |")
    else:
        for row in issues:
            lines.append(
                f"| {row.get('severity', '')} | {row.get('pod_name', '')} | {row.get('node_name', '')} | "
                f"{row.get('check', '')} | {row.get('expected', '')} | {row.get('actual', '')} | "
                f"{row.get('reason', '')} |"
            )
    (result_dir / "dynamic_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="compare compact dynamic healthcheck outputs")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--ratio-threshold", type=float, default=0.7)
    args = parser.parse_args()
    report = compare_dynamic_results(args.result_dir, args.ratio_threshold)
    write_report(args.result_dir, report)
    print(f"[dynamic-compare] status={report['dynamic_compare_status']} result_dir={args.result_dir}")
    raise SystemExit(0 if report["dynamic_compare_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
