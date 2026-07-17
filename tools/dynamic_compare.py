#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SMALL_MAX_BYTES = 1 << 20
LARGE_MIN_BYTES = 1 << 30
SMALL_LATENCY_ABS_DELTA_SECONDS = 0.0002
SMALL_LATENCY_MAD_MULTIPLIER = 6.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _case_id(row: dict[str, Any]) -> str:
    return "/".join(
        str(row.get(key, ""))
        for key in ["stage", "op_type", "message_bytes", "payload_pattern", "collective_group_size"]
    )


def extract_case_metrics(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact in facts:
        summary = fact.get("summary") if isinstance(fact.get("summary"), dict) else {}
        if summary.get("summary_owner") is False:
            continue
        pod = fact.get("pod") if isinstance(fact.get("pod"), dict) else {}
        for metric in summary.get("case_metrics", []) or []:
            if not isinstance(metric, dict):
                continue
            row = dict(metric)
            row["pod_name"] = str(pod.get("name", ""))
            row["node_name"] = str(pod.get("node_name", ""))
            row["run_id"] = str(pod.get("run_id", ""))
            row.setdefault("payload_pattern", "none")
            row.setdefault("collective_group_size", int(summary.get("rank_count", 0) or 0))
            row["case_id"] = _case_id(row)
            rows.append(row)
    return rows


def hard_failure_issues(facts: list[dict[str, Any]], failed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in failed:
        issues.append(
            {
                "severity": "FAIL",
                "classification": "HARD_FAILURE",
                "pod_name": row.get("pod_name", ""),
                "node_name": row.get("node_name", ""),
                "check": "dynamic_failed_pod",
                "reason": row.get("reason", ""),
                "actual": row.get("error_type", ""),
                "expected": "successful dynamic compact frame",
            }
        )
    for fact in facts:
        summary = fact.get("summary") if isinstance(fact.get("summary"), dict) else {}
        if summary.get("summary_owner") is False or summary.get("correctness_pass", False):
            continue
        pod = fact.get("pod") if isinstance(fact.get("pod"), dict) else {}
        issues.append(
            {
                "severity": "FAIL",
                "classification": "HARD_FAILURE",
                "pod_name": pod.get("name", ""),
                "node_name": pod.get("node_name", ""),
                "check": "correctness_pass",
                "reason": summary.get("error_type", "correctness failed"),
                "actual": False,
                "expected": True,
            }
        )
    return issues


def _metric_policy(
    row: dict[str, Any],
    small_max_bytes: int,
    large_min_bytes: int,
    small_latency_warn: bool,
) -> tuple[str, str] | None:
    size = int(row.get("requested_message_bytes", row.get("message_bytes", 0)) or 0)
    if small_latency_warn and size <= small_max_bytes:
        return "latency_p50", "higher"
    if size >= large_min_bytes:
        return "avg_busbw", "lower"
    return None


def _message_class(row: dict[str, Any], small_max_bytes: int, large_min_bytes: int) -> str:
    size = int(row.get("requested_message_bytes", row.get("message_bytes", 0)) or 0)
    if size <= small_max_bytes:
        return "small"
    if size >= large_min_bytes:
        return "large"
    return "medium"


def candidate_performance_issues(
    rows: list[dict[str, Any]],
    *,
    latency_ratio_threshold: float,
    busbw_ratio_threshold: float,
    min_cohort: int,
    small_max_bytes: int,
    large_min_bytes: int,
    small_latency_warn: bool = False,
    small_latency_abs_delta_seconds: float = SMALL_LATENCY_ABS_DELTA_SECONDS,
    small_latency_mad_multiplier: float = SMALL_LATENCY_MAD_MULTIPLIER,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    candidates: list[dict[str, Any]] = []
    cohorts: list[dict[str, Any]] = []
    for case_id, case_rows in sorted(grouped.items()):
        policy = _metric_policy(case_rows[0], small_max_bytes, large_min_bytes, small_latency_warn)
        if policy is None:
            cohorts.append({"case_id": case_id, "status": "OBSERVATION_ONLY", "sample_count": len(case_rows)})
            continue
        metric, direction = policy
        samples = [row for row in case_rows if isinstance(row.get(metric), (int, float)) and float(row[metric]) > 0]
        if len(samples) < min_cohort:
            cohorts.append({"case_id": case_id, "status": "INSUFFICIENT_COHORT", "sample_count": len(samples)})
            continue
        values = [float(row[metric]) for row in samples]
        median = statistics.median(values)
        message_class = _message_class(case_rows[0], small_max_bytes, large_min_bytes)
        mad = statistics.median(abs(value - median) for value in values)
        if direction == "higher":
            threshold = max(
                median * latency_ratio_threshold,
                median + small_latency_abs_delta_seconds,
                median + small_latency_mad_multiplier * mad,
            )
        else:
            threshold = median * busbw_ratio_threshold
        cohorts.append(
            {
                "case_id": case_id,
                "status": "COMPARED",
                "metric": metric,
                "median": median,
                "mad": mad,
                "threshold": threshold,
                "sample_count": len(samples),
                "message_class": message_class,
            }
        )
        for row in samples:
            value = float(row[metric])
            abnormal = value > threshold if direction == "higher" else value < threshold
            if not abnormal:
                continue
            candidates.append(
                {
                    "severity": "WARN" if message_class == "small" else "RETEST",
                    "classification": "SMALL_LATENCY_WARN" if message_class == "small" else "LARGE_BUSBW_CANDIDATE",
                    "pod_name": row.get("pod_name", ""),
                    "node_name": row.get("node_name", ""),
                    "check": metric,
                    "case_id": case_id,
                    "stage": row.get("stage", ""),
                    "op_type": row.get("op_type", ""),
                    "message_size": row.get("message_size", ""),
                    "message_bytes": int(row.get("requested_message_bytes", row.get("message_bytes", 0)) or 0),
                    "payload_pattern": row.get("payload_pattern", "none"),
                    "collective_group_size": row.get("collective_group_size", 0),
                    "actual": value,
                    "median": median,
                    "mad": mad,
                    "message_class": message_class,
                    "expected": f"{'<=' if direction == 'higher' else '>='} {threshold:.6f}",
                    "reason": f"{metric} outside cohort threshold",
                }
            )
    return candidates, cohorts


def _family_key(issue: dict[str, Any]) -> tuple[Any, ...]:
    return (
        issue.get("pod_name", ""),
        issue.get("stage", ""),
        issue.get("op_type", ""),
        issue.get("payload_pattern", "none"),
        issue.get("collective_group_size", 0),
        issue.get("check", ""),
    )


def confirm_adjacent_candidates(
    candidates: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    small_max_bytes: int,
    large_min_bytes: int,
    small_latency_warn: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in case_rows:
        policy = _metric_policy(row, small_max_bytes, large_min_bytes, small_latency_warn)
        if policy is None:
            continue
        key = (
            row.get("pod_name", ""), row.get("stage", ""), row.get("op_type", ""),
            row.get("payload_pattern", "none"), row.get("collective_group_size", 0), policy[0],
        )
        available[key].append(int(row.get("requested_message_bytes", row.get("message_bytes", 0)) or 0))
    by_family: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for issue in candidates:
        by_family[_family_key(issue)].append(issue)
    confirmed_ids: set[tuple[str, str]] = set()
    for family, issues in by_family.items():
        sizes = sorted(set(available.get(family, [])))
        abnormal = {int(issue["message_bytes"]): issue for issue in issues}
        for left, right in zip(sizes, sizes[1:]):
            if left in abnormal and right in abnormal:
                confirmed_ids.add((str(abnormal[left].get("pod_name", "")), str(abnormal[left]["case_id"])))
                confirmed_ids.add((str(abnormal[right].get("pod_name", "")), str(abnormal[right]["case_id"])))
    confirmed: list[dict[str, Any]] = []
    isolated: list[dict[str, Any]] = []
    for issue in candidates:
        item = dict(issue)
        identity = (str(item.get("pod_name", "")), str(item.get("case_id", "")))
        if identity in confirmed_ids:
            if item.get("message_class") == "small":
                item.update(severity="WARN", classification="SMALL_RELATED_CASES_WARN")
            else:
                item.update(severity="RETEST", classification="LARGE_RELATED_CASES_RETEST")
            confirmed.append(item)
        else:
            isolated.append(item)
    return confirmed, isolated


def build_retest_plan(
    isolated: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    large_min_bytes: int = LARGE_MIN_BYTES,
) -> list[dict[str, Any]]:
    plan: dict[str, dict[str, Any]] = {}
    for issue in isolated:
        family = _family_key(issue)
        family_rows = [
            row for row in case_rows
            if (
                row.get("pod_name", ""), row.get("stage", ""), row.get("op_type", ""),
                row.get("payload_pattern", "none"), row.get("collective_group_size", 0), issue.get("check", ""),
            ) == family
        ]
        sizes = sorted(
            {
                int(row.get("requested_message_bytes", row.get("message_bytes", 0)) or 0)
                for row in family_rows
                if int(row.get("requested_message_bytes", row.get("message_bytes", 0)) or 0) >= large_min_bytes
            }
        )
        target = int(issue["message_bytes"])
        selected = {target}
        if target in sizes:
            index = sizes.index(target)
            if index > 0:
                selected.add(sizes[index - 1])
            if index + 1 < len(sizes):
                selected.add(sizes[index + 1])
        for size in selected:
            key = "/".join(map(str, [issue.get("stage", ""), issue.get("op_type", ""), size, issue.get("payload_pattern", "none"), issue.get("collective_group_size", 0)]))
            plan[key] = {
                "stage": issue.get("stage", ""),
                "op_type": issue.get("op_type", ""),
                "message_bytes": size,
                "payload_pattern": issue.get("payload_pattern", "none"),
                "collective_group_size": issue.get("collective_group_size", 0),
            }
    return [plan[key] for key in sorted(plan)]


def compare_dynamic_results(
    result_dir: Path,
    ratio_threshold: float = 0.7,
    *,
    latency_ratio_threshold: float = 1.5,
    min_cohort: int = 3,
    small_max_bytes: int = SMALL_MAX_BYTES,
    large_min_bytes: int = LARGE_MIN_BYTES,
    small_latency_warn: bool = False,
    small_latency_abs_delta_seconds: float = SMALL_LATENCY_ABS_DELTA_SECONDS,
    small_latency_mad_multiplier: float = SMALL_LATENCY_MAD_MULTIPLIER,
    retest_facts_path: Path | None = None,
) -> dict[str, Any]:
    facts = read_jsonl(result_dir / "dynamic_facts.jsonl")
    failed = read_jsonl(result_dir / "dynamic_failed_pods.jsonl")
    if retest_facts_path and retest_facts_path.exists():
        failed.extend(read_jsonl(result_dir / "dynamic_retest_failed_pods.jsonl"))
    case_rows = extract_case_metrics(facts)
    write_jsonl(result_dir / "dynamic_case_metrics.jsonl", case_rows)
    hard = hard_failure_issues(facts, failed)
    candidates, cohorts = candidate_performance_issues(
        case_rows,
        latency_ratio_threshold=latency_ratio_threshold,
        busbw_ratio_threshold=ratio_threshold,
        min_cohort=min_cohort,
        small_max_bytes=small_max_bytes,
        large_min_bytes=large_min_bytes,
        small_latency_warn=small_latency_warn,
        small_latency_abs_delta_seconds=small_latency_abs_delta_seconds,
        small_latency_mad_multiplier=small_latency_mad_multiplier,
    )
    related, isolated = confirm_adjacent_candidates(
        candidates,
        case_rows,
        small_max_bytes,
        large_min_bytes,
        small_latency_warn,
    )
    performance_warnings = [item for item in related + isolated if item.get("message_class") == "small"]
    initial_large_candidates = [item for item in candidates if item.get("message_class") == "large"]
    retest_plan = build_retest_plan(initial_large_candidates, case_rows, large_min_bytes)
    confirmed: list[dict[str, Any]] = []
    transient: list[dict[str, Any]] = []
    retest_only: list[dict[str, Any]] = []
    inconclusive: list[dict[str, Any]] = []
    if retest_facts_path and retest_facts_path.exists() and initial_large_candidates:
        retest_rows = extract_case_metrics(read_jsonl(retest_facts_path))
        observed_retest_candidates, _ = candidate_performance_issues(
            retest_rows,
            latency_ratio_threshold=latency_ratio_threshold,
            busbw_ratio_threshold=ratio_threshold,
            min_cohort=min_cohort,
            small_max_bytes=small_max_bytes,
            large_min_bytes=large_min_bytes,
            small_latency_warn=small_latency_warn,
            small_latency_abs_delta_seconds=small_latency_abs_delta_seconds,
            small_latency_mad_multiplier=small_latency_mad_multiplier,
        )
        retest_large = [item for item in observed_retest_candidates if item.get("message_class") == "large"]
        initial_ids = {
            (str(item.get("pod_name", "")), str(item.get("case_id", "")))
            for item in initial_large_candidates
        }
        retest_by_id = {(str(item.get("pod_name", "")), str(item.get("case_id", ""))): item for item in retest_large}
        for issue in initial_large_candidates:
            identity = (str(issue.get("pod_name", "")), str(issue.get("case_id", "")))
            if identity in retest_by_id:
                item = dict(retest_by_id[identity])
                item.update(severity="SUSPECT", classification="CONFIRMED_RETEST")
                confirmed.append(item)
            else:
                item = dict(issue)
                item.update(severity="WARN", classification="TRANSIENT_RECOVERED")
                transient.append(item)
        for identity, issue in retest_by_id.items():
            if identity in initial_ids:
                continue
            item = dict(issue)
            item.update(severity="WARN", classification="RETEST_ONLY_OBSERVATION")
            retest_only.append(item)
    issues = hard + confirmed + performance_warnings + transient + retest_only + inconclusive
    if hard:
        status = "FAIL"
    elif confirmed or inconclusive:
        status = "SUSPECT"
    elif initial_large_candidates and not (retest_facts_path and retest_facts_path.exists()):
        status = "RETEST_REQUIRED"
    else:
        status = "PASS"
    return {
        "dynamic_compare_status": status,
        "pod_count": len(facts),
        "failed_pod_count": len(failed),
        "issue_count": len(issues),
        "outlier_count": len(confirmed),
        "candidate_count": len(candidates),
        "small_warning_count": len(performance_warnings),
        "large_candidate_count": len(initial_large_candidates),
        "retest_required": status == "RETEST_REQUIRED",
        "retest_plan": retest_plan,
        "issues": issues,
        "confirmed_suspects": confirmed,
        "performance_warnings": performance_warnings,
        "transient_observations": transient,
        "retest_only_observations": retest_only,
        "inconclusive_cohorts": inconclusive,
        "cohorts": cohorts,
        "busbw_ratio_threshold": ratio_threshold,
        "latency_ratio_threshold": latency_ratio_threshold,
        "small_max_bytes": small_max_bytes,
        "large_min_bytes": large_min_bytes,
        "small_latency_warn": small_latency_warn,
        "small_latency_abs_delta_seconds": small_latency_abs_delta_seconds,
        "small_latency_mad_multiplier": small_latency_mad_multiplier,
        "min_cohort": min_cohort,
    }


def write_report(result_dir: Path, report: dict[str, Any]) -> None:
    (result_dir / "dynamic_compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(result_dir / "dynamic_outliers.jsonl", report.get("issues", []))
    write_jsonl(result_dir / "dynamic_retest.jsonl", report.get("retest_plan", []))
    lines = [
        "# Dynamic Compare Summary", "",
        f"- dynamic_compare_status: `{report.get('dynamic_compare_status')}`",
        f"- pod_count: `{report.get('pod_count')}`",
        f"- issue_count: `{report.get('issue_count', 0)}`",
        f"- candidate_count: `{report.get('candidate_count', 0)}`",
        f"- performance_gate: `avg_busbw for message_bytes >= {report.get('large_min_bytes')}`",
        f"- small_latency_warn: `{str(bool(report.get('small_latency_warn', False))).lower()}`",
        f"- initial_measurement_batches: `{report.get('initial_measurement_batches', 'unknown')}`",
        f"- retest_measurement_batches: `{report.get('retest_measurement_batches', 'unknown')}`",
    ]
    sections = [
        ("Hard Failures", [row for row in report.get("issues", []) if row.get("severity") == "FAIL"]),
        ("Performance Warnings", report.get("performance_warnings", [])),
        ("Confirmed Performance Suspects", report.get("confirmed_suspects", [])),
        ("Transient Recoveries", report.get("transient_observations", [])),
        ("Retest-only Observations", report.get("retest_only_observations", [])),
        ("Inconclusive Cohorts", report.get("inconclusive_cohorts", [])),
    ]
    for title, rows in sections:
        lines.extend(["", f"## {title}", "", "| severity | pod | node | case | check | actual | reason |", "| --- | --- | --- | --- | --- | ---: | --- |"])
        if not rows:
            lines.append("|  |  |  |  |  |  | none |")
        for row in rows:
            lines.append(
                f"| {row.get('severity', '')} | {row.get('pod_name', '')} | {row.get('node_name', '')} | "
                f"{row.get('case_id', '')} | {row.get('check', '')} | {row.get('actual', '')} | {row.get('reason', '')} |"
            )
    (result_dir / "dynamic_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="compare compact dynamic healthcheck outputs")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--ratio-threshold", type=float, default=0.7)
    parser.add_argument("--latency-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--min-cohort", type=int, default=3)
    parser.add_argument("--small-max-bytes", type=int, default=SMALL_MAX_BYTES)
    parser.add_argument("--large-min-bytes", type=int, default=LARGE_MIN_BYTES)
    parser.add_argument("--small-latency-warn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--small-latency-abs-delta-seconds", type=float, default=SMALL_LATENCY_ABS_DELTA_SECONDS)
    parser.add_argument("--small-latency-mad-multiplier", type=float, default=SMALL_LATENCY_MAD_MULTIPLIER)
    parser.add_argument("--retest-facts", type=Path)
    args = parser.parse_args()
    report = compare_dynamic_results(
        args.result_dir,
        args.ratio_threshold,
        latency_ratio_threshold=args.latency_ratio_threshold,
        min_cohort=args.min_cohort,
        small_max_bytes=args.small_max_bytes,
        large_min_bytes=args.large_min_bytes,
        small_latency_warn=args.small_latency_warn,
        small_latency_abs_delta_seconds=args.small_latency_abs_delta_seconds,
        small_latency_mad_multiplier=args.small_latency_mad_multiplier,
        retest_facts_path=args.retest_facts,
    )
    write_report(args.result_dir, report)
    print(f"[dynamic-compare] status={report['dynamic_compare_status']} result_dir={args.result_dir}")
    raise SystemExit(0 if report["dynamic_compare_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
