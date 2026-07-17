#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.static_compare import (
    ECC_CRITICAL_KEYS,
    ECC_SINGLE_BIT_KEYS,
    METAX_ECC_CORRECTED_KEYS,
    METAX_ECC_CRITICAL_KEYS,
    PodStaticFacts,
    compare_ascend_ecc_gates,
    compare_metax_ecc_gates,
    compare_static_results,
    summarize_ecc_alerts,
    write_static_compare_outputs,
)


def metax_pod(ecc: dict) -> PodStaticFacts:
    facts = {
        "metax.attached_gpus": "8",
        "metax.ecc.query_status": "OK",
        "metax.ecc.gpu_count": "8",
        "metax.ecc.all_enabled": "True",
    }
    for key in METAX_ECC_CORRECTED_KEYS | METAX_ECC_CRITICAL_KEYS:
        facts[key] = "0"
    for source, target in [
        ("corrected_error_gpu_count", "metax.ecc.corrected_error_gpu_count"),
        ("uncorrected_error_gpu_count", "metax.ecc.uncorrected_error_gpu_count"),
        ("corrected_event_count", "metax.ecc.corrected_event_count"),
        ("critical_event_count", "metax.ecc.critical_event_count"),
    ]:
        if source in ecc:
            facts[target] = str(ecc[source])
    return PodStaticFacts(
        pod_name="metax-worker-0",
        node_name="host-metax-0",
        pod_ip="10.0.0.1",
        status_rows={},
        facts=facts,
        errors=[],
        raw={"gpu": {"ecc": ecc}},
    )


def ascend_pod(ecc: dict) -> PodStaticFacts:
    totals = ecc.get("totals", {})
    facts = {
        "ascend.chip_count": "16",
        "ascend.ecc.query_status": "OK",
        "ascend.ecc.npu_count": "8",
        "ascend.ecc.chip_count": "16",
    }
    for key in ECC_SINGLE_BIT_KEYS | ECC_CRITICAL_KEYS:
        facts[key] = str(totals.get(key.removeprefix("ascend.ecc."), 0))
    return PodStaticFacts(
        pod_name="ascend-worker-0",
        node_name="host-ascend-0",
        pod_ip="10.0.0.2",
        status_rows={},
        facts=facts,
        errors=[],
        raw={"npu": {"ecc": ecc}},
    )


class StaticEccPolicyTest(unittest.TestCase):
    def test_alert_policy_keeps_static_status_pass(self) -> None:
        compact = {
            "pod": {"name": "metax-worker-0", "node_name": "host-metax-0", "pod_ip": "10.0.0.1"},
            "capability": {"checks": {}},
            "gpu": {
                "metax": {"attached_gpus": 8},
                "ecc": {
                    "query_status": "OK",
                    "gpu_count": 8,
                    "topology_signature": "0,1,2,3,4,5,6,7",
                    "all_enabled": True,
                    "corrected_error_gpu_count": 0,
                    "uncorrected_error_gpu_count": 1,
                    "corrected_event_count": 0,
                    "critical_event_count": 0,
                    "corrected_error_details": {},
                    "uncorrected_error_details": {"0": ["MCCTL2 : 1"]},
                    "corrected_events": [],
                    "critical_events": [],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "static_facts.jsonl").write_text(json.dumps(compact) + "\n")
            report = compare_static_results(out, ecc_policy="alert")
        self.assertEqual("PASS", report["static_compare_status"])
        self.assertEqual("WARN", report["ecc_summary"]["status"])
        self.assertEqual(["host-metax-0"], report["ecc_summary"]["affected_nodes"])

    def test_metax_cumulative_counts_are_warnings_in_alert_policy(self) -> None:
        pod = metax_pod(
            {
                "corrected_error_gpu_count": 1,
                "uncorrected_error_gpu_count": 1,
                "corrected_event_count": 0,
                "critical_event_count": 0,
                "corrected_error_details": {"0": ["MCCTL2 : 1"]},
                "uncorrected_error_details": {"0": ["MCCTL2 : 1"]},
                "corrected_events": [],
                "critical_events": [],
            }
        )
        issues, alerts = compare_metax_ecc_gates([pod], ecc_policy="alert")
        self.assertEqual([], issues)
        self.assertEqual({"corrected_count", "uncorrected_count"}, {row["category"] for row in alerts})
        self.assertTrue(all(row["severity"] == "WARN" for row in alerts))
        self.assertTrue(all(row["device_id"] == "0" for row in alerts))

    def test_metax_critical_event_fails(self) -> None:
        pod = metax_pod(
            {
                "corrected_error_gpu_count": 0,
                "uncorrected_error_gpu_count": 0,
                "corrected_event_count": 0,
                "critical_event_count": 1,
                "corrected_error_details": {},
                "uncorrected_error_details": {},
                "corrected_events": [],
                "critical_events": [{"gpu_id": 3, "event_type": "dbe", "details": ["DBE"]}],
            }
        )
        issues, alerts = compare_metax_ecc_gates([pod], ecc_policy="alert")
        self.assertEqual("FAIL", issues[0].severity)
        self.assertEqual("dbe", alerts[0]["category"])
        self.assertEqual(3, alerts[0]["device_id"])

    def test_metax_strict_restores_uncorrected_failure(self) -> None:
        pod = metax_pod(
            {
                "corrected_error_gpu_count": 0,
                "uncorrected_error_gpu_count": 1,
                "corrected_event_count": 0,
                "critical_event_count": 0,
                "corrected_error_details": {},
                "uncorrected_error_details": {"2": ["MCCTL2 : 1"]},
                "corrected_events": [],
                "critical_events": [],
            }
        )
        issues, alerts = compare_metax_ecc_gates([pod], ecc_policy="strict")
        self.assertEqual("FAIL", issues[0].severity)
        self.assertEqual("FAIL", alerts[0]["severity"])

    def test_ascend_aggregate_double_bit_is_warning(self) -> None:
        field = "hbm_double_bit_aggregate_total_err_count"
        pod = ascend_pod(
            {
                "totals": {field: 2},
                "nonzero_chips": [{"npu_id": 1, "chip_id": 0, field: 2}],
            }
        )
        issues, alerts = compare_ascend_ecc_gates([pod], ecc_policy="alert")
        self.assertEqual([], issues)
        self.assertEqual("WARN", alerts[0]["severity"])
        self.assertEqual(1, alerts[0]["device_id"])
        self.assertEqual(0, alerts[0]["chip_id"])

    def test_ascend_current_double_bit_and_isolated_page_fail(self) -> None:
        fields = {
            "hbm_double_bit_error_count": 1,
            "hbm_single_bit_isolated_pages_count": 1,
        }
        pod = ascend_pod(
            {
                "totals": fields,
                "nonzero_chips": [{"npu_id": 4, "chip_id": 1, **fields}],
            }
        )
        issues, alerts = compare_ascend_ecc_gates([pod], ecc_policy="alert")
        self.assertEqual(2, len(issues))
        self.assertTrue(all(issue.severity == "FAIL" for issue in issues))
        self.assertTrue(all(alert["severity"] == "FAIL" for alert in alerts))

    def test_ecc_report_files_contain_node_and_device_details(self) -> None:
        alerts = [
            {
                "vendor": "metax",
                "severity": "WARN",
                "pod_name": "pod-0",
                "node_name": "node-0",
                "pod_ip": "10.0.0.1",
                "device_id": 0,
                "chip_id": "",
                "source": "counter",
                "category": "corrected_count",
                "counter_name": "MCCTL2",
                "counter_value": 1,
                "event_detail": ["MCCTL2 : 1"],
                "reason": "cumulative counter",
                "action": "observe",
            }
        ]
        report = {
            "static_compare_status": "PASS",
            "pod_count": 1,
            "issue_count": 0,
            "warning_count": 1,
            "failed_pod_count": 0,
            "issues": [],
            "warnings": [],
            "ecc_policy": "alert",
            "ecc_alerts": alerts,
            "ecc_summary": summarize_ecc_alerts(alerts, "alert"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_static_compare_outputs(out, report)
            rows = [json.loads(line) for line in (out / "static_ecc_alerts.jsonl").read_text().splitlines()]
            self.assertEqual("node-0", rows[0]["node_name"])
            self.assertEqual(0, rows[0]["device_id"])
            md = (out / "static_ecc_alerts.md").read_text()
            self.assertIn("node-0", md)
            self.assertIn("corrected_count", md)


if __name__ == "__main__":
    unittest.main()
