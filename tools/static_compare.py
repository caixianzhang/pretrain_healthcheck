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
    "metax/mx_ecc_state",
    "metax/mx_ras_count",
    "metax/mx_events",
    "metax/python_torch",
    "metax/maca_env",
    "ascend/npu_smi",
    "ascend/npu_ecc",
    "ascend/python_torch",
    "ascend/ascend_env",
    "net/ip_link",
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
    "metax.ecc.gpu_count",
    "metax.ecc.topology_signature",
    "ascend.chip_count",
    "ascend.version",
    "ascend.health_counts",
    "ascend.ecc.npu_count",
    "ascend.ecc.chip_count",
    "ascend.ecc.topology_signature",
    "ascend.visible_devices",
    "ascend.hccn_tool_available",
    "ascend.rdma_dev_infiniband_exists",
    "ascend.rdma_hca_ids",
    "ascend.rdma_sysfs_port_count",
    "ascend.rdma_sysfs_state_rates",
    "ascend.net_interface_count",
    "ascend.net_interfaces",
    "torch.version",
    "torch.device_count",
    "torch_npu.version",
    "torch_npu.available",
    "hca.ibv_hca_ids",
    "hca.sysfs_xscale_count",
    "hca.sysfs_xscale_state_rates",
}

SOFT_FACT_KEYS = {
    "basic.uname_kernel",
    "hca.sysfs_all_device_count",
    "hca.sysfs_all_state_rates",
}

ECC_SINGLE_BIT_KEYS = {
    "ascend.ecc.hbm_single_bit_error_count",
    "ascend.ecc.hbm_single_bit_aggregate_total_err_count",
}

ECC_CRITICAL_KEYS = {
    "ascend.ecc.hbm_double_bit_error_count",
    "ascend.ecc.hbm_double_bit_aggregate_total_err_count",
    "ascend.ecc.hbm_single_bit_isolated_pages_count",
    "ascend.ecc.hbm_double_bit_isolated_pages_count",
    "ascend.ecc.hbm_single_bit_next_isolated_pages_count",
    "ascend.ecc.hbm_double_bit_next_isolated_pages_count",
}

ASCEND_ECC_AGGREGATE_KEYS = {
    "ascend.ecc.hbm_single_bit_aggregate_total_err_count",
    "ascend.ecc.hbm_double_bit_aggregate_total_err_count",
}

ASCEND_ECC_CURRENT_WARNING_KEYS = {
    "ascend.ecc.hbm_single_bit_error_count",
}

ASCEND_ECC_CURRENT_CRITICAL_KEYS = {
    "ascend.ecc.hbm_double_bit_error_count",
    "ascend.ecc.hbm_single_bit_isolated_pages_count",
    "ascend.ecc.hbm_double_bit_isolated_pages_count",
    "ascend.ecc.hbm_single_bit_next_isolated_pages_count",
    "ascend.ecc.hbm_double_bit_next_isolated_pages_count",
}

ECC_RULE_ONLY_FACT_KEYS = {"ascend.ecc.query_status", *ECC_SINGLE_BIT_KEYS, *ECC_CRITICAL_KEYS}

METAX_ECC_CORRECTED_KEYS = {
    "metax.ecc.corrected_error_gpu_count",
    "metax.ecc.corrected_event_count",
}

METAX_ECC_CRITICAL_KEYS = {
    "metax.ecc.uncorrected_error_gpu_count",
    "metax.ecc.critical_event_count",
}

METAX_ECC_RULE_ONLY_FACT_KEYS = {
    "metax.ecc.query_status",
    "metax.ecc.all_enabled",
    *METAX_ECC_CORRECTED_KEYS,
    *METAX_ECC_CRITICAL_KEYS,
}


@dataclass
class PodStaticFacts:
    pod_name: str
    node_name: str
    pod_ip: str
    status_rows: dict[str, dict[str, str]]
    facts: dict[str, str]
    errors: list[str]
    raw: dict[str, Any] | None = None


@dataclass
class StaticIssue:
    severity: str
    pod_name: str
    node_name: str
    check: str
    expected: str
    actual: str
    reason: str


def ecc_alert(
    pod: PodStaticFacts,
    vendor: str,
    severity: str,
    source: str,
    category: str,
    reason: str,
    *,
    device_id: Any = "",
    chip_id: Any = "",
    counter_name: str = "",
    counter_value: Any = "",
    event_detail: Any = "",
    action: str = "observe",
) -> dict[str, Any]:
    return {
        "vendor": vendor,
        "severity": severity,
        "pod_name": pod.pod_name,
        "node_name": pod.node_name,
        "pod_ip": pod.pod_ip,
        "device_id": device_id,
        "chip_id": chip_id,
        "source": source,
        "category": category,
        "counter_name": counter_name,
        "counter_value": counter_value,
        "event_detail": event_detail,
        "reason": reason,
        "action": action,
    }


def issue_from_ecc_alert(alert: dict[str, Any], check: str, expected: str) -> StaticIssue:
    actual = alert.get("event_detail") or {
        "counter_name": alert.get("counter_name", ""),
        "counter_value": alert.get("counter_value", ""),
        "device_id": alert.get("device_id", ""),
        "chip_id": alert.get("chip_id", ""),
    }
    return StaticIssue(
        severity=str(alert["severity"]),
        pod_name=str(alert["pod_name"]),
        node_name=str(alert["node_name"]),
        check=check,
        expected=expected,
        actual=json.dumps(actual, ensure_ascii=False, sort_keys=True) if not isinstance(actual, str) else actual,
        reason=str(alert["reason"]),
    )


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
    metax_ecc = gpu.get("ecc", {}) if isinstance(gpu.get("ecc"), dict) else {}
    for source, target in [
        ("query_status", "metax.ecc.query_status"),
        ("gpu_count", "metax.ecc.gpu_count"),
        ("topology_signature", "metax.ecc.topology_signature"),
        ("all_enabled", "metax.ecc.all_enabled"),
        ("corrected_error_gpu_count", "metax.ecc.corrected_error_gpu_count"),
        ("uncorrected_error_gpu_count", "metax.ecc.uncorrected_error_gpu_count"),
        ("corrected_event_count", "metax.ecc.corrected_event_count"),
        ("critical_event_count", "metax.ecc.critical_event_count"),
    ]:
        if source in metax_ecc:
            facts[target] = str(metax_ecc[source])
    for source, target in [
        ("version", "torch.version"),
        ("device_count", "torch.device_count"),
        ("cuda_available", "torch.cuda_available"),
    ]:
        if source in torch:
            facts[target] = str(torch[source])

    npu = row.get("npu", {}) if isinstance(row.get("npu"), dict) else {}
    ascend = npu.get("ascend", {}) if isinstance(npu.get("ascend"), dict) else {}
    npu_torch = npu.get("torch", {}) if isinstance(npu.get("torch"), dict) else {}
    for source, target in [
        ("chip_count", "ascend.chip_count"),
        ("version", "ascend.version"),
        ("npu_smi_version", "ascend.npu_smi_version"),
    ]:
        if source in ascend:
            facts[target] = str(ascend[source])
    if "health_counts" in ascend:
        facts["ascend.health_counts"] = json.dumps(ascend["health_counts"], sort_keys=True)
    ecc = npu.get("ecc", {}) if isinstance(npu.get("ecc"), dict) else {}
    for source, target in [
        ("query_status", "ascend.ecc.query_status"),
        ("npu_count", "ascend.ecc.npu_count"),
        ("chip_count", "ascend.ecc.chip_count"),
        ("topology_signature", "ascend.ecc.topology_signature"),
    ]:
        if source in ecc:
            facts[target] = str(ecc[source])
    ecc_totals = ecc.get("totals", {}) if isinstance(ecc.get("totals"), dict) else {}
    for source in sorted(ECC_SINGLE_BIT_KEYS | ECC_CRITICAL_KEYS):
        raw_name = source.removeprefix("ascend.ecc.")
        if raw_name in ecc_totals:
            facts[source] = str(ecc_totals[raw_name])
    container = row.get("container", {}) if isinstance(row.get("container"), dict) else {}
    if "ascend_visible_devices" in container:
        raw_visible = str(container["ascend_visible_devices"])
        parts = [part.strip() for part in raw_visible.split(",") if part.strip()]
        try:
            parts = [str(x) for x in sorted(int(part) for part in parts)]
        except ValueError:
            parts = sorted(parts)
        facts["ascend.visible_devices"] = ",".join(parts) if parts else raw_visible
    for source, target in [
        ("version", "torch.version"),
        ("device_count", "torch.device_count"),
        ("torch_npu_version", "torch_npu.version"),
        ("npu_available", "torch_npu.available"),
    ]:
        if source in npu_torch:
            facts[target] = str(npu_torch[source])

    npu_network = npu.get("network", {}) if isinstance(npu.get("network"), dict) else {}
    hccn_tool = npu_network.get("hccn_tool", {}) if isinstance(npu_network.get("hccn_tool"), dict) else {}
    if "available" in hccn_tool:
        facts["ascend.hccn_tool_available"] = str(hccn_tool["available"])

    rdma = row.get("rdma", {}) if isinstance(row.get("rdma"), dict) else {}
    if "dev_infiniband_exists" in rdma:
        facts["ascend.rdma_dev_infiniband_exists"] = str(rdma["dev_infiniband_exists"])
    ascend_ibv = rdma.get("ibv_devinfo", {}) if isinstance(rdma.get("ibv_devinfo"), dict) else {}
    if isinstance(ascend_ibv.get("hca_ids"), list):
        facts["ascend.rdma_hca_ids"] = ",".join(str(x) for x in ascend_ibv["hca_ids"])
    ascend_sysfs = rdma.get("sysfs", {}) if isinstance(rdma.get("sysfs"), dict) else {}
    if "port_count" in ascend_sysfs:
        facts["ascend.rdma_sysfs_port_count"] = str(ascend_sysfs["port_count"])
    state_rates = ascend_sysfs.get("state_rates", []) if isinstance(ascend_sysfs.get("state_rates"), list) else []
    if state_rates:
        facts["ascend.rdma_sysfs_state_rates"] = ";".join(str(x) for x in state_rates)

    net = row.get("net", {}) if isinstance(row.get("net"), dict) else {}
    net_sysfs = net.get("sysfs", {}) if isinstance(net.get("sysfs"), dict) else {}
    if "interface_count" in net_sysfs:
        facts["ascend.net_interface_count"] = str(net_sysfs["interface_count"])
    interfaces = net_sysfs.get("interfaces", []) if isinstance(net_sysfs.get("interfaces"), list) else []
    if interfaces:
        facts["ascend.net_interfaces"] = ";".join(str(x) for x in interfaces)

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
        raw=row,
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
        raw=None,
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
                    raw=row,
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
        if key in ECC_RULE_ONLY_FACT_KEYS or key in METAX_ECC_RULE_ONLY_FACT_KEYS:
            continue
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


def compare_metax_ecc_gates(
    pods: list[PodStaticFacts], ecc_policy: str = "alert"
) -> tuple[list[StaticIssue], list[dict[str, Any]]]:
    issues: list[StaticIssue] = []
    alerts: list[dict[str, Any]] = []
    required_keys = sorted(METAX_ECC_CORRECTED_KEYS | METAX_ECC_CRITICAL_KEYS)
    for pod in pods:
        if "metax.attached_gpus" not in pod.facts:
            continue

        raw = pod.raw or {}
        gpu = raw.get("gpu", {}) if isinstance(raw.get("gpu"), dict) else {}
        ecc = gpu.get("ecc", {}) if isinstance(gpu.get("ecc"), dict) else {}

        query_status = pod.facts.get("metax.ecc.query_status", "")
        if query_status != "OK":
            alert = ecc_alert(
                pod, "metax", "FAIL", "status", "query_failure",
                "MetaX ECC/RAS query failed, is missing, or produced incomplete output",
                event_detail=ecc.get("errors", query_status or "missing"), action="ops_check",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_query", "OK"))
            continue

        missing_keys = [key for key in required_keys if key not in pod.facts]
        if missing_keys:
            alert = ecc_alert(
                pod, "metax", "FAIL", "status", "missing_counters",
                "MetaX ECC/RAS output is missing required counters",
                event_detail=missing_keys, action="ops_check",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_counters", "all ECC/RAS counters present"))
            continue

        try:
            ecc_gpu_count = int(pod.facts.get("metax.ecc.gpu_count", ""))
            attached_gpu_count = int(pod.facts.get("metax.attached_gpus", ""))
        except ValueError:
            alert = ecc_alert(
                pod, "metax", "FAIL", "status", "invalid_topology",
                "MetaX ECC topology counters are invalid",
                event_detail={
                    "ecc_gpu_count": pod.facts.get("metax.ecc.gpu_count", ""),
                    "attached_gpu_count": pod.facts.get("metax.attached_gpus", ""),
                }, action="ops_check",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_topology", "numeric GPU counts"))
            continue

        if ecc_gpu_count <= 0 or ecc_gpu_count != attached_gpu_count:
            alert = ecc_alert(
                pod, "metax", "FAIL", "status", "topology_mismatch",
                "MetaX ECC/RAS query did not cover every attached GPU",
                counter_name="ecc_gpu_count", counter_value=ecc_gpu_count,
                event_detail={"attached_gpu_count": attached_gpu_count}, action="ops_check",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_topology", f"ecc_gpu_count={attached_gpu_count}"))

        if pod.facts.get("metax.ecc.all_enabled", "").lower() != "true":
            disabled = ecc.get("disabled_gpus", []) or [""]
            for gpu_id in disabled:
                alert = ecc_alert(
                    pod, "metax", "FAIL", "status", "ecc_disabled",
                    "ECC is disabled on one or more MetaX GPUs",
                    device_id=gpu_id, event_detail=ecc.get("states", {}), action="quarantine",
                )
                alerts.append(alert)
                issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_enabled", "True"))

        try:
            critical = {
                key: int(pod.facts[key])
                for key in sorted(METAX_ECC_CRITICAL_KEYS)
                if int(pod.facts[key]) > 0
            }
            corrected = {
                key: int(pod.facts[key])
                for key in sorted(METAX_ECC_CORRECTED_KEYS)
                if int(pod.facts[key]) > 0
            }
        except ValueError:
            alert = ecc_alert(
                pod, "metax", "FAIL", "status", "invalid_counters",
                "MetaX ECC/RAS counter parsing failed", event_detail="invalid counter value", action="ops_check",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_counters", "non-negative integer counters"))
            continue

        corrected_details = ecc.get("corrected_error_details", {}) if isinstance(ecc.get("corrected_error_details"), dict) else {}
        uncorrected_details = ecc.get("uncorrected_error_details", {}) if isinstance(ecc.get("uncorrected_error_details"), dict) else {}
        counter_specs = [
            (corrected_details, "corrected_error_details", "corrected_count", "corrected MetaX ECC/RAS cumulative count detected"),
            (uncorrected_details, "uncorrected_error_details", "uncorrected_count", "uncorrected MetaX ECC/RAS cumulative count detected without a critical event"),
        ]
        for details, detail_key, category, reason in counter_specs:
            for gpu_id, values in sorted(details.items(), key=lambda item: str(item[0])):
                severity = "SUSPECT" if ecc_policy == "strict" and category == "corrected_count" else "WARN"
                if ecc_policy == "strict" and category == "uncorrected_count":
                    severity = "FAIL"
                action = "quarantine" if severity == "FAIL" else "retest" if severity == "SUSPECT" else "observe"
                alert = ecc_alert(
                    pod, "metax", severity, "counter", category, reason,
                    device_id=gpu_id, counter_name=detail_key, counter_value=len(values) if isinstance(values, list) else 1,
                    event_detail=values, action=action,
                )
                alerts.append(alert)
                if severity in {"SUSPECT", "FAIL"}:
                    issues.append(issue_from_ecc_alert(alert, f"rule:metax_ecc_{category}", "0"))

        corrected_events = ecc.get("corrected_events", []) if isinstance(ecc.get("corrected_events"), list) else []
        critical_events = ecc.get("critical_events", []) if isinstance(ecc.get("critical_events"), list) else []
        for event in corrected_events:
            severity = "SUSPECT" if ecc_policy == "strict" else "WARN"
            alert = ecc_alert(
                pod, "metax", severity, "event", str(event.get("event_type", "corrected_event")),
                "corrected MetaX RAS event detected", device_id=event.get("gpu_id", ""),
                event_detail=event.get("details", []), action="retest",
            )
            alerts.append(alert)
            if severity == "SUSPECT":
                issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_corrected_event", "no corrected RAS events"))

        for event in critical_events:
            alert = ecc_alert(
                pod, "metax", "FAIL", "event", str(event.get("event_type", "critical_event")),
                "critical MetaX RAS event detected", device_id=event.get("gpu_id", ""),
                event_detail=event.get("details", []), action="quarantine",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_critical_event", "no critical RAS events"))

        # Legacy compact facts may contain aggregate counters without device details.
        if int(pod.facts.get("metax.ecc.corrected_error_gpu_count", "0")) > 0 and not corrected_details:
            severity = "SUSPECT" if ecc_policy == "strict" else "WARN"
            alert = ecc_alert(pod, "metax", severity, "counter", "corrected_count", "corrected MetaX ECC/RAS cumulative count detected", counter_name="corrected_error_gpu_count", counter_value=pod.facts["metax.ecc.corrected_error_gpu_count"], action="retest")
            alerts.append(alert)
            if severity == "SUSPECT":
                issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_corrected", "0"))
        if int(pod.facts.get("metax.ecc.uncorrected_error_gpu_count", "0")) > 0 and not uncorrected_details:
            severity = "FAIL" if ecc_policy == "strict" else "WARN"
            alert = ecc_alert(pod, "metax", severity, "counter", "uncorrected_count", "uncorrected MetaX ECC/RAS cumulative count detected without a critical event", counter_name="uncorrected_error_gpu_count", counter_value=pod.facts["metax.ecc.uncorrected_error_gpu_count"], action="quarantine" if severity == "FAIL" else "observe")
            alerts.append(alert)
            if severity == "FAIL":
                issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_critical", "0"))
        if int(pod.facts.get("metax.ecc.corrected_event_count", "0")) > 0 and not corrected_events:
            severity = "SUSPECT" if ecc_policy == "strict" else "WARN"
            alert = ecc_alert(pod, "metax", severity, "event", "corrected_event", "corrected MetaX RAS event detected", counter_name="corrected_event_count", counter_value=pod.facts["metax.ecc.corrected_event_count"], action="retest")
            alerts.append(alert)
            if severity == "SUSPECT":
                issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_corrected_event", "0"))
        if int(pod.facts.get("metax.ecc.critical_event_count", "0")) > 0 and not critical_events:
            alert = ecc_alert(pod, "metax", "FAIL", "event", "critical_event", "critical MetaX RAS event detected", counter_name="critical_event_count", counter_value=pod.facts["metax.ecc.critical_event_count"], action="quarantine")
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:metax_ecc_critical_event", "0"))
    return issues, alerts


def compare_ascend_ecc_gates(
    pods: list[PodStaticFacts], ecc_policy: str = "alert"
) -> tuple[list[StaticIssue], list[dict[str, Any]]]:
    issues: list[StaticIssue] = []
    alerts: list[dict[str, Any]] = []
    required_counter_keys = sorted(ECC_SINGLE_BIT_KEYS | ECC_CRITICAL_KEYS)
    for pod in pods:
        if "ascend.chip_count" not in pod.facts:
            continue

        raw = pod.raw or {}
        npu = raw.get("npu", {}) if isinstance(raw.get("npu"), dict) else {}
        ecc = npu.get("ecc", {}) if isinstance(npu.get("ecc"), dict) else {}

        query_status = pod.facts.get("ascend.ecc.query_status", "")
        if query_status != "OK":
            alert = ecc_alert(
                pod, "ascend", "FAIL", "status", "query_failure",
                "ECC query failed, is missing, or produced incomplete output",
                event_detail=ecc.get("errors", query_status or "missing"), action="ops_check",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:ascend_ecc_query", "OK"))
            continue

        missing_keys = [key for key in required_counter_keys if key not in pod.facts]
        if missing_keys:
            alert = ecc_alert(
                pod, "ascend", "FAIL", "status", "missing_counters",
                "ECC output is missing required counters", event_detail=missing_keys, action="ops_check",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:ascend_ecc_counters", "all ECC counters present"))
            continue

        try:
            ecc_npu_count = int(pod.facts.get("ascend.ecc.npu_count", ""))
            visible_npu_count = int(pod.facts.get("ascend.chip_count", ""))
            ecc_chip_count = int(pod.facts.get("ascend.ecc.chip_count", ""))
        except ValueError:
            alert = ecc_alert(
                pod, "ascend", "FAIL", "status", "invalid_topology",
                "ECC topology counters are invalid", event_detail={
                    "npu_count": pod.facts.get("ascend.ecc.npu_count", ""),
                    "visible_chip_count": pod.facts.get("ascend.chip_count", ""),
                    "ecc_chip_count": pod.facts.get("ascend.ecc.chip_count", ""),
                }, action="ops_check",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:ascend_ecc_topology", "numeric NPU and Chip counts"))
            continue

        if ecc_npu_count <= 0 or ecc_chip_count <= 0 or ecc_chip_count != visible_npu_count:
            alert = ecc_alert(
                pod, "ascend", "FAIL", "status", "topology_mismatch",
                "ECC query did not cover every visible logical NPU chip",
                counter_name="ecc_chip_count", counter_value=ecc_chip_count,
                event_detail={"physical_npu_count": ecc_npu_count, "visible_chip_count": visible_npu_count}, action="ops_check",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:ascend_ecc_topology", f"ecc_chip_count={visible_npu_count}, physical_npu_count>0"))

        try:
            critical = {
                key: int(pod.facts[key])
                for key in sorted(ECC_CRITICAL_KEYS)
                if int(pod.facts[key]) > 0
            }
            single_bit = {
                key: int(pod.facts[key])
                for key in sorted(ECC_SINGLE_BIT_KEYS)
                if int(pod.facts[key]) > 0
            }
        except ValueError:
            alert = ecc_alert(
                pod, "ascend", "FAIL", "status", "invalid_counters",
                "ECC counter parsing failed", event_detail="invalid counter value", action="ops_check",
            )
            alerts.append(alert)
            issues.append(issue_from_ecc_alert(alert, "rule:ascend_ecc_counters", "non-negative integer counters"))
            continue

        nonzero_chips = ecc.get("nonzero_chips", []) if isinstance(ecc.get("nonzero_chips"), list) else []
        emitted_keys: set[str] = set()
        for chip in nonzero_chips:
            if not isinstance(chip, dict):
                continue
            for raw_name, raw_value in chip.items():
                if raw_name in {"npu_id", "chip_id"} or not isinstance(raw_value, int) or raw_value <= 0:
                    continue
                key = f"ascend.ecc.{raw_name}"
                emitted_keys.add(key)
                if key in ASCEND_ECC_CURRENT_CRITICAL_KEYS:
                    severity, action = "FAIL", "quarantine"
                    reason = "current uncorrectable HBM ECC or isolated-page condition detected"
                elif ecc_policy == "strict":
                    severity = "FAIL" if key in ECC_CRITICAL_KEYS else "SUSPECT"
                    action = "quarantine" if severity == "FAIL" else "retest"
                    reason = "Ascend ECC counter detected under strict policy"
                else:
                    severity, action = "WARN", "observe"
                    reason = "historical or correctable Ascend ECC counter detected"
                alert = ecc_alert(
                    pod, "ascend", severity, "counter", raw_name, reason,
                    device_id=chip.get("npu_id", ""), chip_id=chip.get("chip_id", ""),
                    counter_name=raw_name, counter_value=raw_value, action=action,
                )
                alerts.append(alert)
                if severity in {"SUSPECT", "FAIL"}:
                    issues.append(issue_from_ecc_alert(alert, f"rule:ascend_ecc_{raw_name}", "0"))

        # Legacy compact facts only have aggregate totals; preserve their policy semantics.
        for key, value in sorted({**single_bit, **critical}.items()):
            if key in emitted_keys:
                continue
            if key in ASCEND_ECC_CURRENT_CRITICAL_KEYS:
                severity, action = "FAIL", "quarantine"
                reason = "current uncorrectable HBM ECC or isolated-page condition detected"
            elif ecc_policy == "strict":
                severity = "FAIL" if key in ECC_CRITICAL_KEYS else "SUSPECT"
                action = "quarantine" if severity == "FAIL" else "retest"
                reason = "Ascend ECC counter detected under strict policy"
            else:
                severity, action = "WARN", "observe"
                reason = "historical or correctable Ascend ECC counter detected"
            alert = ecc_alert(
                pod, "ascend", severity, "counter", key.removeprefix("ascend.ecc."), reason,
                counter_name=key.removeprefix("ascend.ecc."), counter_value=value, action=action,
            )
            alerts.append(alert)
            if severity in {"SUSPECT", "FAIL"}:
                issues.append(issue_from_ecc_alert(alert, f"rule:{key.replace('.', '_')}", "0"))
    return issues, alerts


def compare_rule_gates(
    pods: list[PodStaticFacts],
    expected_gpus: int = 0,
    expected_xscale_ports: int = 0,
) -> list[StaticIssue]:
    issues: list[StaticIssue] = []
    if expected_gpus > 0:
        for pod in pods:
            present_count_keys = [
                key
                for key in [
                    "metax.attached_gpus",
                    "metax.gpu_available_count",
                    "ascend.chip_count",
                    "torch.device_count",
                ]
                if pod.facts.get(key, "") != ""
            ]
            if not present_count_keys:
                issues.append(
                    StaticIssue(
                        severity="FAIL",
                        pod_name=pod.pod_name,
                        node_name=pod.node_name,
                        check="rule:device_count",
                        expected=str(expected_gpus),
                        actual="",
                        reason="no device count fact found for expected device count gate",
                    )
                )
                continue
            for key in present_count_keys:
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
                            reason="value does not match expected device count",
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
    ecc_policy: str = "alert",
) -> dict[str, Any]:
    if ecc_policy not in {"alert", "strict"}:
        raise ValueError(f"unsupported ECC policy: {ecc_policy}")
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
            "ecc_policy": ecc_policy,
            "ecc_alerts": [],
            "ecc_summary": summarize_ecc_alerts([], ecc_policy),
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
    metax_ecc_issues, metax_ecc_alerts = compare_metax_ecc_gates(pods, ecc_policy=ecc_policy)
    ascend_ecc_issues, ascend_ecc_alerts = compare_ascend_ecc_gates(pods, ecc_policy=ecc_policy)
    ecc_alerts = metax_ecc_alerts + ascend_ecc_alerts
    issues.extend(metax_ecc_issues)
    issues.extend(ascend_ecc_issues)
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
        "warning_count": len(warnings) + sum(1 for alert in ecc_alerts if alert.get("severity") == "WARN"),
        "failed_pod_count": len(failed_pods),
        "rule_gate": {
            "expected_gpus": expected_gpus,
            "expected_xscale_ports": expected_xscale_ports,
        },
        "failed_pods": failed_pods,
        "issues": [asdict(issue) for issue in sorted(issues, key=lambda x: (x.severity, x.pod_name, x.check))],
        "warnings": warnings,
        "ecc_policy": ecc_policy,
        "ecc_alerts": sorted(
            ecc_alerts,
            key=lambda item: (
                str(item.get("severity", "")), str(item.get("vendor", "")),
                str(item.get("node_name", "")), str(item.get("device_id", "")),
                str(item.get("chip_id", "")), str(item.get("category", "")),
            ),
        ),
        "ecc_summary": summarize_ecc_alerts(ecc_alerts, ecc_policy),
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


def summarize_ecc_alerts(alerts: list[dict[str, Any]], ecc_policy: str) -> dict[str, Any]:
    severity_counts = Counter(str(alert.get("severity", "")) for alert in alerts)
    vendor_counts = Counter(str(alert.get("vendor", "")) for alert in alerts)
    affected_nodes = sorted({str(alert.get("node_name", "")) for alert in alerts if alert.get("node_name")})
    affected_pods = sorted({str(alert.get("pod_name", "")) for alert in alerts if alert.get("pod_name")})
    status = "PASS"
    if severity_counts.get("FAIL", 0):
        status = "FAIL"
    elif severity_counts.get("SUSPECT", 0):
        status = "SUSPECT"
    elif severity_counts.get("WARN", 0):
        status = "WARN"
    return {
        "status": status,
        "policy": ecc_policy,
        "alert_count": len(alerts),
        "warning_count": severity_counts.get("WARN", 0),
        "suspect_count": severity_counts.get("SUSPECT", 0),
        "fail_count": severity_counts.get("FAIL", 0),
        "affected_node_count": len(affected_nodes),
        "affected_pod_count": len(affected_pods),
        "affected_nodes": affected_nodes,
        "affected_pods": affected_pods,
        "vendor_counts": dict(sorted(vendor_counts.items())),
        "jsonl_report": "static_ecc_alerts.jsonl",
        "markdown_report": "static_ecc_alerts.md",
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    ecc_summary = report.get("ecc_summary", {})
    lines = [
        "# Static Compare Summary",
        "",
        f"- static_compare_status: `{report['static_compare_status']}`",
        f"- pod_count: `{report['pod_count']}`",
        f"- issue_count: `{report.get('issue_count', 0)}`",
        f"- warning_count: `{report.get('warning_count', 0)}`",
        f"- failed_pod_count: `{report.get('failed_pod_count', 0)}`",
        f"- ecc_alert_status: `{ecc_summary.get('status', 'PASS')}`",
        f"- ecc_affected_node_count: `{ecc_summary.get('affected_node_count', 0)}`",
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

    lines.extend(["", "## ECC/RAS Alerts", ""])
    if report.get("ecc_alerts"):
        lines.extend(
            [
                f"- policy: `{report.get('ecc_policy', 'alert')}`",
                f"- status: `{ecc_summary.get('status', 'PASS')}`",
                f"- warning_count: `{ecc_summary.get('warning_count', 0)}`",
                f"- fail_count: `{ecc_summary.get('fail_count', 0)}`",
                f"- affected_nodes: `{','.join(ecc_summary.get('affected_nodes', []))}`",
                "- detailed_report: `static_ecc_alerts.md`",
            ]
        )
    else:
        lines.append("No ECC/RAS alerts detected.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ecc_alert_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report.get("ecc_summary", {})
    alerts = report.get("ecc_alerts", [])
    lines = [
        "# Static ECC/RAS Alerts",
        "",
        f"- status: `{summary.get('status', 'PASS')}`",
        f"- policy: `{summary.get('policy', report.get('ecc_policy', 'alert'))}`",
        f"- alert_count: `{summary.get('alert_count', 0)}`",
        f"- warning_count: `{summary.get('warning_count', 0)}`",
        f"- suspect_count: `{summary.get('suspect_count', 0)}`",
        f"- fail_count: `{summary.get('fail_count', 0)}`",
        f"- affected_node_count: `{summary.get('affected_node_count', 0)}`",
        "",
    ]
    if not alerts:
        lines.append("No ECC/RAS alerts detected.")
    else:
        lines.extend(
            [
                "| severity | vendor | pod | node | device | chip | source | category | counter | value | detail | action |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for alert in alerts:
            lines.append(
                "| {severity} | {vendor} | {pod_name} | {node_name} | {device_id} | {chip_id} | "
                "{source} | {category} | {counter_name} | {counter_value} | {event_detail} | {action} |".format(
                    **{key: md_cell(alert.get(key, "")) for key in [
                        "severity", "vendor", "pod_name", "node_name", "device_id", "chip_id", "source",
                        "category", "counter_name", "counter_value", "event_detail", "action",
                    ]}
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_static_compare_outputs(result_dir: Path, report: dict[str, Any]) -> None:
    write_json(result_dir / "static_compare.json", report)
    write_jsonl(result_dir / "static_outliers.jsonl", report.get("issues", []))
    write_jsonl(result_dir / "static_ecc_alerts.jsonl", report.get("ecc_alerts", []))
    write_markdown(result_dir / "static_compare.md", report)
    write_ecc_alert_markdown(result_dir / "static_ecc_alerts.md", report)


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


def render_ecc_alert_section(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    summary = report.get("ecc_summary", {})
    alerts = report.get("ecc_alerts", [])
    lines = [
        "## ECC/RAS Alerts",
        "",
        f"- status: `{summary.get('status', 'PASS')}`",
        f"- policy: `{summary.get('policy', report.get('ecc_policy', 'alert'))}`",
        f"- warning_count: `{summary.get('warning_count', 0)}`",
        f"- fail_count: `{summary.get('fail_count', 0)}`",
        f"- affected_node_count: `{summary.get('affected_node_count', 0)}`",
        "- detailed_report: `static_ecc_alerts.md`",
        "",
    ]
    if not alerts:
        lines.append("No ECC/RAS alerts detected.")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| severity | vendor | pod | node | device | chip | category | value | action |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for alert in alerts:
        lines.append(
            "| {severity} | {vendor} | {pod_name} | {node_name} | {device_id} | {chip_id} | "
            "{category} | {counter_value} | {action} |".format(
                **{key: md_cell(alert.get(key, "")) for key in [
                    "severity", "vendor", "pod_name", "node_name", "device_id", "chip_id",
                    "category", "counter_value", "action",
                ]}
            )
        )
    return "\n".join(lines) + "\n"


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
    basic = sample.get("basic", {}) if isinstance(sample.get("basic"), dict) else {}
    uname = basic.get("uname", {}) if isinstance(basic.get("uname"), dict) else {}
    container = sample.get("container", {}) if isinstance(sample.get("container"), dict) else {}
    storage = sample.get("storage", {}) if isinstance(sample.get("storage"), dict) else {}

    common_rows = [
        ("pod", pod.get("name", "unknown")),
        ("node", pod.get("node_name", "unknown")),
        ("pod_ip", pod.get("pod_ip", "unknown")),
        ("host_ip", pod.get("host_ip", "unknown")),
        ("device_type", pod.get("device_type", "unknown")),
        ("kernel", uname.get("kernel", "unknown")),
        ("pod_time", basic.get("date", "unknown")),
        ("df_mount_count", len(storage.get("df", [])) if isinstance(storage.get("df"), list) else "unknown"),
        ("inode_mount_count", len(storage.get("inode", [])) if isinstance(storage.get("inode"), list) else "unknown"),
    ]

    gpu = sample.get("gpu", {}) if isinstance(sample.get("gpu"), dict) else {}
    metax = gpu.get("metax", {}) if isinstance(gpu.get("metax"), dict) else {}
    metax_torch = gpu.get("torch", {}) if isinstance(gpu.get("torch"), dict) else {}
    npu = sample.get("npu", {}) if isinstance(sample.get("npu"), dict) else {}
    ascend = npu.get("ascend", {}) if isinstance(npu.get("ascend"), dict) else {}
    npu_torch = npu.get("torch", {}) if isinstance(npu.get("torch"), dict) else {}

    platform_title = "Platform Specific Fields"
    platform_rows: list[tuple[str, Any]] = []
    if ascend:
        network = npu.get("network", {}) if isinstance(npu.get("network"), dict) else {}
        hccn_tool = network.get("hccn_tool", {}) if isinstance(network.get("hccn_tool"), dict) else {}
        rdma = sample.get("rdma", {}) if isinstance(sample.get("rdma"), dict) else {}
        ibv = rdma.get("ibv_devinfo", {}) if isinstance(rdma.get("ibv_devinfo"), dict) else {}
        rdma_sysfs = rdma.get("sysfs", {}) if isinstance(rdma.get("sysfs"), dict) else {}
        net = sample.get("net", {}) if isinstance(sample.get("net"), dict) else {}
        net_sysfs = net.get("sysfs", {}) if isinstance(net.get("sysfs"), dict) else {}
        ecc = npu.get("ecc", {}) if isinstance(npu.get("ecc"), dict) else {}
        ecc_totals = ecc.get("totals", {}) if isinstance(ecc.get("totals"), dict) else {}
        platform_title = "Ascend / NPU Fields"
        platform_rows = [
            ("chip_count", ascend.get("chip_count", "unknown")),
            ("health_counts", ascend.get("health_counts", "unknown")),
            ("npu_smi_version", ascend.get("npu_smi_version", "unknown")),
            ("ascend_version", ascend.get("version", "unknown")),
            ("hbm_total_mb_values", ascend.get("hbm_total_mb_values", "unknown")),
            ("ecc_query_status", ecc.get("query_status", "not_collected")),
            ("ecc_npu_count", ecc.get("npu_count", "not_collected")),
            ("ecc_chip_count", ecc.get("chip_count", "not_collected")),
            ("ecc_topology_signature", ecc.get("topology_signature", "not_collected")),
            ("ecc_single_bit_current", ecc_totals.get("hbm_single_bit_error_count", "not_collected")),
            ("ecc_double_bit_current", ecc_totals.get("hbm_double_bit_error_count", "not_collected")),
            ("ecc_single_bit_aggregate", ecc_totals.get("hbm_single_bit_aggregate_total_err_count", "not_collected")),
            ("ecc_double_bit_aggregate", ecc_totals.get("hbm_double_bit_aggregate_total_err_count", "not_collected")),
            ("ecc_single_bit_isolated_pages", ecc_totals.get("hbm_single_bit_isolated_pages_count", "not_collected")),
            ("ecc_double_bit_isolated_pages", ecc_totals.get("hbm_double_bit_isolated_pages_count", "not_collected")),
            ("ecc_single_bit_next_isolated_pages", ecc_totals.get("hbm_single_bit_next_isolated_pages_count", "not_collected")),
            ("ecc_double_bit_next_isolated_pages", ecc_totals.get("hbm_double_bit_next_isolated_pages_count", "not_collected")),
            ("ecc_nonzero_chips", ecc.get("nonzero_chips", [])),
            ("ecc_errors", ecc.get("errors", [])),
            ("torch_version", npu_torch.get("version", "unknown")),
            ("torch_npu_version", npu_torch.get("torch_npu_version", "unknown")),
            ("npu_available", npu_torch.get("npu_available", "unknown")),
            ("torch_npu_device_count", npu_torch.get("device_count", "unknown")),
            ("ASCEND_VISIBLE_DEVICES", container.get("ascend_visible_devices", "unknown")),
            ("hccn_tool_available", hccn_tool.get("available", "not_collected")),
            ("hccn_tool_path", hccn_tool.get("tool_path", "missing_tool")),
            ("dev_infiniband_exists", rdma.get("dev_infiniband_exists", "not_collected")),
            ("rdma_hca_count", ibv.get("hca_count", "missing_tool_or_no_device")),
            ("rdma_hca_ids", ibv.get("hca_ids", "missing_tool_or_no_device")),
            ("rdma_sysfs_port_count", rdma_sysfs.get("port_count", "not_collected")),
            ("rdma_sysfs_state_rates", rdma_sysfs.get("state_rates", "not_collected")),
            ("net_interface_count", net_sysfs.get("interface_count", "not_collected")),
            ("net_interfaces", net_sysfs.get("interfaces", "not_collected")),
        ]
    elif metax:
        hca = sample.get("hca", {}) if isinstance(sample.get("hca"), dict) else {}
        ibv = hca.get("ibv_devinfo", {}) if isinstance(hca.get("ibv_devinfo"), dict) else {}
        sysfs = hca.get("sysfs", {}) if isinstance(hca.get("sysfs"), dict) else {}
        metax_ecc = gpu.get("ecc", {}) if isinstance(gpu.get("ecc"), dict) else {}
        platform_title = "MetaX / GPU Fields"
        platform_rows = [
            ("gpu_model_counts", metax.get("gpu_model_counts", "unknown")),
            ("attached_gpus", metax.get("attached_gpus", "unknown")),
            ("gpu_available_count", metax.get("gpu_available_count", "unknown")),
            ("mx_smi_version", metax.get("mx_smi_version", "unknown")),
            ("driver_version", metax.get("driver_version", "unknown")),
            ("maca_version", metax.get("maca_version", "unknown")),
            ("bios_version", metax.get("bios_version", "unknown")),
            ("ecc_query_status", metax_ecc.get("query_status", "not_collected")),
            ("ecc_gpu_count", metax_ecc.get("gpu_count", "not_collected")),
            ("ecc_topology_signature", metax_ecc.get("topology_signature", "not_collected")),
            ("ecc_all_enabled", metax_ecc.get("all_enabled", "not_collected")),
            ("ecc_disabled_gpus", metax_ecc.get("disabled_gpus", [])),
            ("ecc_corrected_error_gpu_count", metax_ecc.get("corrected_error_gpu_count", "not_collected")),
            ("ecc_uncorrected_error_gpu_count", metax_ecc.get("uncorrected_error_gpu_count", "not_collected")),
            ("ecc_corrected_event_count", metax_ecc.get("corrected_event_count", "not_collected")),
            ("ecc_critical_event_count", metax_ecc.get("critical_event_count", "not_collected")),
            ("ecc_corrected_error_details", metax_ecc.get("corrected_error_details", {})),
            ("ecc_uncorrected_error_details", metax_ecc.get("uncorrected_error_details", {})),
            ("ecc_corrected_events", metax_ecc.get("corrected_events", [])),
            ("ecc_critical_events", metax_ecc.get("critical_events", [])),
            ("ecc_errors", metax_ecc.get("errors", [])),
            ("torch_version", metax_torch.get("version", "unknown")),
            ("torch_device_count", metax_torch.get("device_count", "unknown")),
            ("hca_count", ibv.get("hca_count", "unknown")),
            ("hca_ids", ibv.get("hca_ids", "unknown")),
            ("xscale_port_count", sysfs.get("xscale_port_count", "unknown")),
            ("rdma_port_count", sysfs.get("port_count", "unknown")),
        ]

    lines = [
        "## Node Environment Sample",
        "",
        "Static compare passed. The following is one representative node's software and hardware summary from `static_facts.jsonl`.",
        "",
        "### Common Fields",
        "",
        "| item | value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {item} | {md_cell(value)} |" for item, value in common_rows)
    lines.extend(["", f"### {platform_title}", "", "| item | value |", "| --- | --- |"])
    if platform_rows:
        lines.extend(f"| {item} | {md_cell(value)} |" for item, value in platform_rows)
    else:
        lines.append("| platform | unknown |")
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
        ecc_section = render_ecc_alert_section(report)
        node_sample_section = render_node_environment_sample_section(result_dir, report)
        text = re.sub(r"\n## ECC/RAS Alerts\n.*?(?=\n## |\Z)", "\n", text, flags=re.S)
        text = re.sub(r"\n## Node Environment Sample\n.*?(?=\n## |\Z)", "\n", text, flags=re.S)
        if "## Static Compare" in text:
            text = re.sub(r"\n## Static Compare\n.*?(?=\n## |\Z)", "\n" + static_section, text, flags=re.S)
        else:
            text = text.rstrip() + "\n\n" + static_section
        if ecc_section:
            text = text.rstrip() + "\n\n" + ecc_section
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
    parser.add_argument(
        "--ecc-policy",
        choices=["alert", "strict"],
        default="alert",
        help="ECC policy: alert keeps cumulative counters as warnings; strict restores legacy gates.",
    )
    args = parser.parse_args()

    report = compare_static_results(
        args.result_dir,
        workers=args.workers,
        expected_gpus=args.expected_gpus,
        expected_xscale_ports=args.expected_xscale_ports,
        ecc_policy=args.ecc_policy,
    )
    write_static_compare_outputs(args.result_dir, report)
    update_summary_files(args.result_dir, report)
    print(f"[static-compare] status={report['static_compare_status']} result_dir={args.result_dir}")
    return 0 if report["static_compare_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
