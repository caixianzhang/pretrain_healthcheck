#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SKIP_STATUS_ITEMS = {
    "logs/dmesg",
}

CRITICAL_STATUS_ITEMS = {
    "metax/mx_smi",
    "metax/python_torch",
    "metax/maca_env",
    "hca/ibv_devinfo",
    "sys/sys_infiniband",
    "sys/infiniband_sysfs",
}

HARD_FACT_KEYS = {
    "metax.attached_gpus",
    "metax.driver_version",
    "metax.maca_version",
    "metax.bios_version",
    "metax.gpu_model_counts",
    "metax.gpu_available_count",
    "torch.version",
    "torch.device_count",
    "hca.ibv_hca_ids",
    "hca.sysfs_xscale_count",
    "hca.sysfs_xscale_state_rates",
}

SOFT_FACT_KEYS = {
    "basic.uname_kernel",
    "hca.sysfs_all_device_count",
    "hca.sysfs_all_state_rates",
}


@dataclass
class PodStaticFacts:
    pod_name: str
    node_name: str
    pod_ip: str
    status_rows: dict[str, dict[str, str]]
    facts: dict[str, str]
    errors: list[str]


@dataclass
class StaticIssue:
    severity: str
    pod_name: str
    node_name: str
    check: str
    expected: str
    actual: str
    reason: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return normalize_ws(match.group(1)) if match else ""


def parse_mx_smi(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    if not text:
        return facts

    facts["metax.mx_smi_version"] = first_match(text, r"mx-smi\s+version:\s*([^\n]+)")
    facts["metax.attached_gpus"] = first_match(text, r"Attached GPUs\s*:\s*(\d+)")
    facts["metax.driver_version"] = first_match(text, r"Kernel Mode Driver Version:\s*([^\s|]+)")
    facts["metax.maca_version"] = first_match(text, r"MACA Version:\s*([^\s|]+)")
    facts["metax.bios_version"] = first_match(text, r"BIOS Version:\s*([^\s|]+)")

    model_counts = Counter()
    for match in re.finditer(r"\|\s*\d+\s+([^|]+?)\s+\|\s*\d+\s+", text):
        model_counts[normalize_ws(match.group(1))] += 1
    if model_counts:
        facts["metax.gpu_model_counts"] = json.dumps(dict(sorted(model_counts.items())), sort_keys=True)

    available_count = len(re.findall(r"\bAvailable\b", text))
    if available_count:
        facts["metax.gpu_available_count"] = str(available_count)
    return facts


def parse_python_torch(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    if not text:
        return facts
    facts["torch.version"] = first_match(text, r"^torch\s+(.+)$")
    facts["torch.cuda_available"] = first_match(text, r"^cuda_available\s+(.+)$")
    facts["torch.device_count"] = first_match(text, r"^device_count\s+(.+)$")
    return {key: value for key, value in facts.items() if value}


def parse_uname(text: str) -> dict[str, str]:
    if not text:
        return {}
    parts = normalize_ws(text).split()
    if len(parts) >= 3:
        return {"basic.uname_kernel": parts[2]}
    return {}


def parse_ibv_devinfo(text: str) -> dict[str, str]:
    hca_ids = sorted(set(re.findall(r"^\s*hca_id:\s*(\S+)", text, flags=re.MULTILINE)))
    if not hca_ids:
        return {}
    return {"hca.ibv_hca_ids": ",".join(hca_ids)}


def parse_infiniband_sysfs(text: str) -> dict[str, str]:
    device_state_rates: dict[str, dict[str, str]] = defaultdict(dict)
    for line in text.splitlines():
        match = re.match(r"/sys/class/infiniband/([^/]+)/ports/([^/]+)/([^ ]+)\s+(.+)$", line.strip())
        if not match:
            continue
        device, port, name, value = match.groups()
        key = f"{device}:{port}"
        if name in {"state", "rate", "link_layer"}:
            device_state_rates[key][name] = normalize_ws(value)

    if not device_state_rates:
        return {}

    all_items = []
    xscale_items = []
    for key in sorted(device_state_rates):
        fields = device_state_rates[key]
        item = f"{key}|state={fields.get('state', '')}|rate={fields.get('rate', '')}|link={fields.get('link_layer', '')}"
        all_items.append(item)
        if key.startswith("xscale_"):
            xscale_items.append(item)

    facts = {
        "hca.sysfs_all_device_count": str(len(device_state_rates)),
        "hca.sysfs_all_state_rates": ";".join(all_items),
        "hca.sysfs_xscale_count": str(len(xscale_items)),
        "hca.sysfs_xscale_state_rates": ";".join(xscale_items),
    }
    return facts


def compact_to_flat_facts(row: dict[str, Any]) -> dict[str, str]:
    facts: dict[str, str] = {}

    basic = row.get("basic", {}) if isinstance(row.get("basic"), dict) else {}
    uname = basic.get("uname", {}) if isinstance(basic.get("uname"), dict) else {}
    if uname.get("kernel"):
        facts["basic.uname_kernel"] = str(uname["kernel"])

    gpu = row.get("gpu", {}) if isinstance(row.get("gpu"), dict) else {}
    metax = gpu.get("metax", {}) if isinstance(gpu.get("metax"), dict) else {}
    torch = gpu.get("torch", {}) if isinstance(gpu.get("torch"), dict) else {}
    for source, target in [
        ("attached_gpus", "metax.attached_gpus"),
        ("driver_version", "metax.driver_version"),
        ("maca_version", "metax.maca_version"),
        ("bios_version", "metax.bios_version"),
        ("gpu_available_count", "metax.gpu_available_count"),
    ]:
        if source in metax:
            facts[target] = str(metax[source])
    if "gpu_model_counts" in metax:
        facts["metax.gpu_model_counts"] = json.dumps(metax["gpu_model_counts"], sort_keys=True)
    for source, target in [
        ("version", "torch.version"),
        ("device_count", "torch.device_count"),
        ("cuda_available", "torch.cuda_available"),
    ]:
        if source in torch:
            facts[target] = str(torch[source])

    hca = row.get("hca", {}) if isinstance(row.get("hca"), dict) else {}
    ibv = hca.get("ibv_devinfo", {}) if isinstance(hca.get("ibv_devinfo"), dict) else {}
    if isinstance(ibv.get("hca_ids"), list):
        facts["hca.ibv_hca_ids"] = ",".join(str(x) for x in ibv["hca_ids"])

    sysfs = hca.get("sysfs", {}) if isinstance(hca.get("sysfs"), dict) else {}
    ports = sysfs.get("ports", []) if isinstance(sysfs.get("ports"), list) else []
    all_items = []
    xscale_items = []
    for port in sorted(ports, key=lambda item: (str(item.get("device", "")), str(item.get("port", "")))):
        key = f"{port.get('device', '')}:{port.get('port', '')}"
        item = (
            f"{key}|state={port.get('state', '')}|rate={port.get('rate', '')}|"
            f"link={port.get('link_layer', '')}"
        )
        all_items.append(item)
        if str(port.get("device", "")).startswith("xscale_"):
            xscale_items.append(item)
    if ports:
        facts["hca.sysfs_all_device_count"] = str(len(ports))
        facts["hca.sysfs_all_state_rates"] = ";".join(all_items)
        facts["hca.sysfs_xscale_count"] = str(len(xscale_items))
        facts["hca.sysfs_xscale_state_rates"] = ";".join(xscale_items)
    return facts


def load_compact_fact_row(row: dict[str, Any], pods_by_name: dict[str, dict[str, Any]]) -> PodStaticFacts:
    pod_meta = row.get("pod", {}) if isinstance(row.get("pod"), dict) else {}
    pod_name = str(pod_meta.get("name", ""))
    node_name = str(pod_meta.get("node_name", ""))
    pod_ip = str(pod_meta.get("pod_ip", ""))
    if pod_name and (not node_name or not pod_ip):
        meta_node, meta_ip = pod_metadata(pod_name, pods_by_name)
        node_name = node_name or meta_node
        pod_ip = pod_ip or meta_ip

    capability = row.get("capability", {}) if isinstance(row.get("capability"), dict) else {}
    checks = capability.get("checks", {}) if isinstance(capability.get("checks"), dict) else {}
    status_rows = {
        str(key): {
            "status": str(value.get("status", "")) if isinstance(value, dict) else "",
            "detail": str(value.get("detail", "")) if isinstance(value, dict) else "",
        }
        for key, value in checks.items()
    }
    return PodStaticFacts(
        pod_name=pod_name,
        node_name=node_name,
        pod_ip=pod_ip,
        status_rows=status_rows,
        facts=compact_to_flat_facts(row),
        errors=[] if pod_name else ["missing pod.name in compact facts"],
    )


def pod_metadata(pod_name: str, pods_by_name: dict[str, dict[str, Any]]) -> tuple[str, str]:
    pod = pods_by_name.get(pod_name, {})
    return str(pod.get("node_name", "")), str(pod.get("pod_ip", ""))


def load_pod_static_result(pod_dir: Path, pods_by_name: dict[str, dict[str, Any]]) -> PodStaticFacts:
    pod_name = pod_dir.parent.name
    node_name, pod_ip = pod_metadata(pod_name, pods_by_name)
    errors: list[str] = []

    rows = read_jsonl(pod_dir / "checks.jsonl")
    if not rows:
        errors.append("missing or empty checks.jsonl")

    status_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        category = str(row.get("category", ""))
        item = str(row.get("item", ""))
        key = f"{category}/{item}"
        status_rows[key] = {
            "status": str(row.get("status", "")),
            "detail": str(row.get("detail", "")),
        }

    facts: dict[str, str] = {}
    facts.update(parse_uname(read_text(pod_dir / "basic_uname.log")))
    facts.update(parse_mx_smi(read_text(pod_dir / "metax_mx_smi.log")))
    facts.update(parse_python_torch(read_text(pod_dir / "metax_python_torch.log")))
    facts.update(parse_ibv_devinfo(read_text(pod_dir / "hca_ibv_devinfo.log")))
    facts.update(parse_infiniband_sysfs(read_text(pod_dir / "infiniband_sysfs.txt")))

    return PodStaticFacts(
        pod_name=pod_name,
        node_name=node_name,
        pod_ip=pod_ip,
        status_rows=status_rows,
        facts=facts,
        errors=errors,
    )


def aggregate_static_outputs(result_dir: Path, keep_pod_files: bool = True) -> dict[str, Any]:
    pods_jsonl = result_dir / "pods.jsonl"
    pods_by_name = {
        str(row.get("pod_name", "")): row
        for row in read_jsonl(pods_jsonl)
        if row.get("pod_name")
    }
    static_dirs = sorted((result_dir / "pod_results").glob("*/static"))
    facts_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    missing_compact: list[str] = []

    for pod_dir in static_dirs:
        pod_name = pod_dir.parent.name
        compact_path = pod_dir / "compact_facts.json"
        if compact_path.exists():
            compact = json.loads(compact_path.read_text(encoding="utf-8"))
        else:
            legacy = load_pod_static_result(pod_dir, pods_by_name)
            missing_compact.append(pod_name)
            compact = {
                "schema_version": 0,
                "pod": {
                    "name": legacy.pod_name,
                    "node_name": legacy.node_name,
                    "pod_ip": legacy.pod_ip,
                },
                "capability": {"checks": legacy.status_rows},
                "legacy_flat_facts": legacy.facts,
            }
        pod_meta = compact.setdefault("pod", {})
        if not pod_meta.get("name"):
            pod_meta["name"] = pod_name
        if not pod_meta.get("node_name") or not pod_meta.get("pod_ip"):
            node_name, pod_ip = pod_metadata(str(pod_meta.get("name", "")), pods_by_name)
            pod_meta["node_name"] = pod_meta.get("node_name") or node_name
            pod_meta["pod_ip"] = pod_meta.get("pod_ip") or pod_ip
        facts_rows.append(compact)

        for row in read_jsonl(pod_dir / "checks.jsonl"):
            enriched = dict(row)
            enriched.setdefault("pod_name", pod_meta.get("name", pod_name))
            enriched.setdefault("node_name", pod_meta.get("node_name", ""))
            enriched.setdefault("pod_ip", pod_meta.get("pod_ip", ""))
            check_rows.append(enriched)

    if facts_rows:
        write_jsonl(result_dir / "static_facts.jsonl", facts_rows)
    if check_rows:
        write_jsonl(result_dir / "static_checks.jsonl", check_rows)

    removed_dirs = 0
    if facts_rows and not keep_pod_files:
        for pod_dir in static_dirs:
            shutil.rmtree(pod_dir, ignore_errors=True)
            removed_dirs += 1

    return {
        "fact_count": len(facts_rows),
        "check_count": len(check_rows),
        "removed_pod_static_dirs": removed_dirs,
        "missing_compact_pods": missing_compact,
    }


def load_aggregated_static_results(result_dir: Path, pods_by_name: dict[str, dict[str, Any]]) -> list[PodStaticFacts]:
    rows = read_jsonl(result_dir / "static_facts.jsonl")
    pods: list[PodStaticFacts] = []
    for row in rows:
        if "legacy_flat_facts" in row:
            pod_meta = row.get("pod", {}) if isinstance(row.get("pod"), dict) else {}
            checks = row.get("capability", {}).get("checks", {}) if isinstance(row.get("capability"), dict) else {}
            pods.append(
                PodStaticFacts(
                    pod_name=str(pod_meta.get("name", "")),
                    node_name=str(pod_meta.get("node_name", "")),
                    pod_ip=str(pod_meta.get("pod_ip", "")),
                    status_rows={
                        str(key): {
                            "status": str(value.get("status", "")) if isinstance(value, dict) else "",
                            "detail": str(value.get("detail", "")) if isinstance(value, dict) else "",
                        }
                        for key, value in checks.items()
                    },
                    facts={str(k): str(v) for k, v in row.get("legacy_flat_facts", {}).items()},
                    errors=[],
                )
            )
        else:
            pods.append(load_compact_fact_row(row, pods_by_name))
    return pods


def majority(values: dict[str, str]) -> str:
    counter = Counter(values.values())
    if not counter:
        return ""
    value, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
    if count <= len(values) / 2:
        return ""
    return value


def compare_status_rows(pods: list[PodStaticFacts]) -> tuple[list[StaticIssue], list[dict[str, Any]]]:
    issues: list[StaticIssue] = []
    warnings: list[dict[str, Any]] = []
    all_keys = sorted({key for pod in pods for key in pod.status_rows})
    for key in all_keys:
        if key in SKIP_STATUS_ITEMS:
            continue
        values = {
            pod.pod_name: pod.status_rows.get(key, {}).get("status", "MISSING_ROW")
            for pod in pods
        }
        baseline = majority(values)
        if not baseline:
            for pod in pods:
                actual = values[pod.pod_name]
                issues.append(
                    StaticIssue(
                        severity="SUSPECT",
                        pod_name=pod.pod_name,
                        node_name=pod.node_name,
                        check=f"status:{key}",
                        expected="majority baseline",
                        actual=actual,
                        reason="no majority baseline; cannot isolate one pod from static data alone",
                    )
                )
            continue

        if baseline != "OK":
            if len(set(values.values())) == 1:
                warnings.append(
                    {
                        "check": key,
                        "status": baseline,
                        "reason": "all pods report the same non-OK status",
                    }
                )
                continue

        for pod in pods:
            actual = values[pod.pod_name]
            if actual == baseline:
                continue
            severity = "FAIL" if key in CRITICAL_STATUS_ITEMS or baseline == "OK" else "SUSPECT"
            detail = pod.status_rows.get(key, {}).get("detail", "")
            issues.append(
                StaticIssue(
                    severity=severity,
                    pod_name=pod.pod_name,
                    node_name=pod.node_name,
                    check=f"status:{key}",
                    expected=baseline,
                    actual=actual,
                    reason=detail or "status differs from majority baseline",
                )
            )
    return issues, warnings


def compare_facts(pods: list[PodStaticFacts]) -> list[StaticIssue]:
    issues: list[StaticIssue] = []
    all_keys = sorted({key for pod in pods for key in pod.facts})
    for key in all_keys:
        values = {pod.pod_name: pod.facts.get(key, "") for pod in pods}
        if len(set(values.values())) <= 1:
            continue
        baseline = majority(values)
        if not baseline:
            for pod in pods:
                issues.append(
                    StaticIssue(
                        severity="SUSPECT",
                        pod_name=pod.pod_name,
                        node_name=pod.node_name,
                        check=f"fact:{key}",
                        expected="majority baseline",
                        actual=values[pod.pod_name],
                        reason="no majority baseline; cannot isolate one pod from static data alone",
                    )
                )
            continue
        if key in HARD_FACT_KEYS:
            severity = "FAIL"
        elif key in SOFT_FACT_KEYS:
            severity = "SUSPECT"
        else:
            severity = "SUSPECT"
        for pod in pods:
            actual = values[pod.pod_name]
            if actual == baseline:
                continue
            issues.append(
                StaticIssue(
                    severity=severity,
                    pod_name=pod.pod_name,
                    node_name=pod.node_name,
                    check=f"fact:{key}",
                    expected=baseline,
                    actual=actual,
                    reason="value differs from majority baseline",
                )
            )
    return issues


def compare_rule_gates(
    pods: list[PodStaticFacts],
    expected_gpus: int = 0,
    expected_xscale_ports: int = 0,
) -> list[StaticIssue]:
    issues: list[StaticIssue] = []
    if expected_gpus > 0:
        for pod in pods:
            for key in ["metax.attached_gpus", "metax.gpu_available_count", "torch.device_count"]:
                actual = pod.facts.get(key, "")
                if actual != str(expected_gpus):
                    issues.append(
                        StaticIssue(
                            severity="FAIL",
                            pod_name=pod.pod_name,
                            node_name=pod.node_name,
                            check=f"rule:{key}",
                            expected=str(expected_gpus),
                            actual=actual,
                            reason="value does not match expected GPU count",
                        )
                    )
    if expected_xscale_ports > 0:
        for pod in pods:
            actual = pod.facts.get("hca.sysfs_xscale_count", "")
            if actual != str(expected_xscale_ports):
                issues.append(
                    StaticIssue(
                        severity="FAIL",
                        pod_name=pod.pod_name,
                        node_name=pod.node_name,
                        check="rule:hca.sysfs_xscale_count",
                        expected=str(expected_xscale_ports),
                        actual=actual,
                        reason="value does not match expected xscale/HCA port count",
                    )
                )
    return issues


def compare_static_results(
    result_dir: Path,
    workers: int = 0,
    expected_gpus: int = 0,
    expected_xscale_ports: int = 0,
) -> dict[str, Any]:
    pods_jsonl = result_dir / "pods.jsonl"
    pods_by_name = {
        str(row.get("pod_name", "")): row
        for row in read_jsonl(pods_jsonl)
        if row.get("pod_name")
    }
    if not (result_dir / "static_facts.jsonl").exists():
        aggregate_static_outputs(result_dir, keep_pod_files=True)

    pods = load_aggregated_static_results(result_dir, pods_by_name)
    failed_pods = read_jsonl(result_dir / "static_failed_pods.jsonl")
    if not pods:
        static_dirs = sorted((result_dir / "pod_results").glob("*/static"))
        max_workers = workers if workers > 0 else min(32, len(static_dirs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            pods = list(executor.map(lambda path: load_pod_static_result(path, pods_by_name), static_dirs))

    if not pods:
        issues = [
            asdict(
                StaticIssue(
                    severity="FAIL",
                    pod_name=str(row.get("pod_name", "")),
                    node_name=str(row.get("node_name", "")),
                    check="static_failed_pod",
                    expected="successful static probe stdout frame",
                    actual=str(row.get("error_type", "")),
                    reason=str(row.get("reason", "")),
                )
            )
            for row in failed_pods
        ]
        return {
            "static_compare_status": "FAIL" if failed_pods else "SUSPECT",
            "pod_count": 0,
            "issue_count": len(issues),
            "warning_count": 0,
            "failed_pod_count": len(failed_pods),
            "failed_pods": failed_pods,
            "issues": issues,
            "warnings": [{"reason": "no aggregated static facts or pod static result directories found"}],
            "pods": [],
        }

    issues: list[StaticIssue] = []
    warnings: list[dict[str, Any]] = []
    for row in failed_pods:
        issues.append(
            StaticIssue(
                severity="FAIL",
                pod_name=str(row.get("pod_name", "")),
                node_name=str(row.get("node_name", "")),
                check="static_failed_pod",
                expected="successful static probe stdout frame",
                actual=str(row.get("error_type", "")),
                reason=str(row.get("reason", "")),
            )
        )
    for pod in pods:
        for error in pod.errors:
            issues.append(
                StaticIssue(
                    severity="FAIL",
                    pod_name=pod.pod_name,
                    node_name=pod.node_name,
                    check="static_result",
                    expected="readable static probe output",
                    actual=error,
                    reason=error,
                )
            )

    status_issues, status_warnings = compare_status_rows(pods)
    issues.extend(status_issues)
    warnings.extend(status_warnings)
    issues.extend(compare_rule_gates(pods, expected_gpus=expected_gpus, expected_xscale_ports=expected_xscale_ports))
    issues.extend(compare_facts(pods))

    status = "PASS"
    if any(issue.severity == "FAIL" for issue in issues):
        status = "FAIL"
    elif issues:
        status = "SUSPECT"

    return {
        "static_compare_status": status,
        "pod_count": len(pods),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "failed_pod_count": len(failed_pods),
        "rule_gate": {
            "expected_gpus": expected_gpus,
            "expected_xscale_ports": expected_xscale_ports,
        },
        "failed_pods": failed_pods,
        "issues": [asdict(issue) for issue in sorted(issues, key=lambda x: (x.severity, x.pod_name, x.check))],
        "warnings": warnings,
        "pods": [
            {
                "pod_name": pod.pod_name,
                "node_name": pod.node_name,
                "pod_ip": pod.pod_ip,
                "fact_count": len(pod.facts),
                "status_row_count": len(pod.status_rows),
            }
            for pod in pods
        ],
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Static Compare Summary",
        "",
        f"- static_compare_status: `{report['static_compare_status']}`",
        f"- pod_count: `{report['pod_count']}`",
        f"- issue_count: `{report.get('issue_count', 0)}`",
        f"- warning_count: `{report.get('warning_count', 0)}`",
        f"- failed_pod_count: `{report.get('failed_pod_count', 0)}`",
        "",
        "## Issues",
        "",
    ]
    issues = report.get("issues", [])
    if issues:
        lines.extend(
            [
                "| severity | pod | node | check | expected | actual | reason |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for issue in issues:
            lines.append(
                "| {severity} | {pod_name} | {node_name} | {check} | {expected} | {actual} | {reason} |".format(
                    severity=issue.get("severity", ""),
                    pod_name=issue.get("pod_name", ""),
                    node_name=issue.get("node_name", ""),
                    check=str(issue.get("check", "")).replace("|", "/"),
                    expected=str(issue.get("expected", "")).replace("|", "/"),
                    actual=str(issue.get("actual", "")).replace("|", "/"),
                    reason=str(issue.get("reason", "")).replace("|", "/"),
                )
            )
    else:
        lines.append("No static outliers detected.")

    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["| check | status | reason |", "| --- | --- | --- |"])
        for warning in warnings:
            lines.append(
                "| {check} | {status} | {reason} |".format(
                    check=str(warning.get("check", "")).replace("|", "/"),
                    status=str(warning.get("status", "")).replace("|", "/"),
                    reason=str(warning.get("reason", "")).replace("|", "/"),
                )
            )
    else:
        lines.append("No global capability warnings.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_static_compare_outputs(result_dir: Path, report: dict[str, Any]) -> None:
    write_json(result_dir / "static_compare.json", report)
    write_jsonl(result_dir / "static_outliers.jsonl", report.get("issues", []))
    write_markdown(result_dir / "static_compare.md", report)


def summarize_result_rows(rows: list[dict[str, Any]]) -> str:
    statuses = [str(row.get("status", "")) for row in rows]
    if not rows:
        return "SUSPECT"
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if any(status == "SUSPECT" for status in statuses):
        return "SUSPECT"
    if all(status == "DRY_RUN" for status in statuses):
        return "DRY_RUN"
    return "PASS"


def merge_static_status(base_status: str, static_status: str) -> str:
    if static_status == "FAIL":
        return "FAIL"
    if static_status == "SUSPECT" and base_status == "PASS":
        return "SUSPECT"
    return base_status


def format_summary_value(value: Any) -> str:
    if value in ("", None, [], {}):
        return "unknown"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def md_cell(value: Any) -> str:
    return format_summary_value(value).replace("|", "/")


def render_node_environment_sample_section(result_dir: Path, report: dict[str, Any] | None) -> str:
    if not report or report.get("static_compare_status") != "PASS":
        return ""
    facts_rows = read_jsonl(result_dir / "static_facts.jsonl")
    if not facts_rows:
        return ""
    sample = next((row for row in facts_rows if isinstance(row, dict)), None)
    if not sample:
        return ""

    pod = sample.get("pod", {}) if isinstance(sample.get("pod"), dict) else {}
    gpu = sample.get("gpu", {}) if isinstance(sample.get("gpu"), dict) else {}
    metax = gpu.get("metax", {}) if isinstance(gpu.get("metax"), dict) else {}
    torch = gpu.get("torch", {}) if isinstance(gpu.get("torch"), dict) else {}
    hca = sample.get("hca", {}) if isinstance(sample.get("hca"), dict) else {}
    ibv = hca.get("ibv_devinfo", {}) if isinstance(hca.get("ibv_devinfo"), dict) else {}
    sysfs = hca.get("sysfs", {}) if isinstance(hca.get("sysfs"), dict) else {}
    basic = sample.get("basic", {}) if isinstance(sample.get("basic"), dict) else {}
    uname = basic.get("uname", {}) if isinstance(basic.get("uname"), dict) else {}

    rows = [
        ("pod", pod.get("name", "unknown")),
        ("node", pod.get("node_name", "unknown")),
        ("pod_ip", pod.get("pod_ip", "unknown")),
        ("host_ip", pod.get("host_ip", "unknown")),
        ("kernel", uname.get("kernel", "unknown")),
        ("pod_time", basic.get("date", "unknown")),
        ("gpu_model_counts", metax.get("gpu_model_counts", "unknown")),
        ("attached_gpus", metax.get("attached_gpus", "unknown")),
        ("gpu_available_count", metax.get("gpu_available_count", "unknown")),
        ("mx_smi_version", metax.get("mx_smi_version", "unknown")),
        ("driver_version", metax.get("driver_version", "unknown")),
        ("maca_version", metax.get("maca_version", "unknown")),
        ("bios_version", metax.get("bios_version", "unknown")),
        ("torch_version", torch.get("version", "unknown")),
        ("torch_device_count", torch.get("device_count", "unknown")),
        ("hca_count", ibv.get("hca_count", "unknown")),
        ("hca_ids", ibv.get("hca_ids", "unknown")),
        ("xscale_port_count", sysfs.get("xscale_port_count", "unknown")),
    ]

    lines = [
        "## Node Environment Sample",
        "",
        "Static compare passed. The following is one representative node's software and hardware summary from `static_facts.jsonl`.",
        "",
        "| item | value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {item} | {md_cell(value)} |" for item, value in rows)
    return "\n".join(lines) + "\n"


def update_summary_files(result_dir: Path, report: dict[str, Any]) -> None:
    summary_json = result_dir / "summary.json"
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        base_status = summarize_result_rows(summary.get("results", []))
        summary["static_compare"] = report
        summary["overall_status"] = merge_static_status(base_status, str(report.get("static_compare_status", "")))
        write_json(summary_json, summary)

    summary_md = result_dir / "summary.md"
    if summary_md.exists():
        text = summary_md.read_text(encoding="utf-8")
        if summary_json.exists():
            overall = json.loads(summary_json.read_text(encoding="utf-8")).get("overall_status", "")
            text = re.sub(r"- overall_status: `[^`]+`", f"- overall_status: `{overall}`", text, count=1)
        static_section = "\n".join(
            [
                "## Static Compare",
                "",
                f"- static_compare_status: `{report.get('static_compare_status')}`",
                f"- issue_count: `{report.get('issue_count', 0)}`",
                f"- warning_count: `{report.get('warning_count', 0)}`",
                "- report: `static_compare.md`",
                "",
            ]
        )
        node_sample_section = render_node_environment_sample_section(result_dir, report)
        text = re.sub(r"\n## Node Environment Sample\n.*?(?=\n## |\Z)", "\n", text, flags=re.S)
        if "## Static Compare" in text:
            text = re.sub(r"\n## Static Compare\n.*?(?=\n## |\Z)", "\n" + static_section, text, flags=re.S)
        else:
            text = text.rstrip() + "\n\n" + static_section
        if node_sample_section:
            text = text.rstrip() + "\n\n" + node_sample_section
        summary_md.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare static probe outputs across vcctl pods.")
    parser.add_argument("--result-dir", required=True, type=Path, help="results/vcctl/<run_id> directory")
    parser.add_argument("--workers", type=int, default=0, help="Parallel parser workers. 0 means auto.")
    parser.add_argument("--expected-gpus", type=int, default=0, help="Expected GPU count per pod. 0 disables this rule gate.")
    parser.add_argument(
        "--expected-xscale-ports",
        type=int,
        default=0,
        help="Expected xscale/HCA port count per pod. 0 disables this rule gate.",
    )
    args = parser.parse_args()

    report = compare_static_results(
        args.result_dir,
        workers=args.workers,
        expected_gpus=args.expected_gpus,
        expected_xscale_ports=args.expected_xscale_ports,
    )
    write_static_compare_outputs(args.result_dir, report)
    update_summary_files(args.result_dir, report)
    print(f"[static-compare] status={report['static_compare_status']} result_dir={args.result_dir}")
    return 0 if report["static_compare_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
