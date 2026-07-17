from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.dynamic_compare import compare_dynamic_results


def case(size: int, *, latency: float = 0.001, busbw: float = 100.0) -> dict:
    return {
        "stage": "collective_bandwidth",
        "op_type": "all_reduce",
        "message_size": str(size),
        "message_bytes": size,
        "requested_message_bytes": size,
        "payload_pattern": "none",
        "collective_group_size": 16,
        "latency_p50": latency,
        "latency_p95": latency,
        "latency_p99": latency,
        "avg_busbw": busbw,
        "second_lowest_busbw": busbw,
        "correctness_pass": True,
        "performance_pass": True,
    }


def fact(pod: str, cases: list[dict]) -> dict:
    return {
        "schema_version": 2,
        "pod": {"name": pod, "node_name": f"node-{pod}", "run_id": "run"},
        "summary": {"summary_owner": True, "correctness_pass": True, "case_metrics": cases},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class DynamicCompareCasePolicyTest(unittest.TestCase):
    def run_compare(
        self,
        facts: list[dict],
        retest: list[dict] | None = None,
        *,
        small_latency_warn: bool = False,
    ) -> dict:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        write_jsonl(root / "dynamic_facts.jsonl", facts)
        write_jsonl(root / "dynamic_failed_pods.jsonl", [])
        retest_path = None
        if retest is not None:
            retest_path = root / "dynamic_retest_facts.jsonl"
            write_jsonl(retest_path, retest)
        return compare_dynamic_results(
            root,
            retest_facts_path=retest_path,
            small_latency_warn=small_latency_warn,
        )

    def test_small_latency_is_observation_only_by_default(self) -> None:
        size = 1 << 20
        report = self.run_compare([
            fact("a", [case(size, latency=0.003)]),
            fact("b", [case(size, latency=0.001)]),
            fact("c", [case(size, latency=0.001)]),
        ])
        self.assertEqual(report["dynamic_compare_status"], "PASS")
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["small_warning_count"], 0)
        self.assertFalse(report["retest_required"])
        self.assertEqual(report["cohorts"][0]["status"], "OBSERVATION_ONLY")

    def test_optional_small_latency_diagnostics_remain_non_blocking(self) -> None:
        sizes = [512 << 10, 1 << 20]
        report = self.run_compare([
            fact("a", [case(size, latency=0.003) for size in sizes]),
            fact("b", [case(size, latency=0.001) for size in sizes]),
            fact("c", [case(size, latency=0.001) for size in sizes]),
        ], small_latency_warn=True)
        self.assertEqual(report["dynamic_compare_status"], "PASS")
        self.assertEqual(len(report["performance_warnings"]), 2)
        self.assertEqual(len(report["confirmed_suspects"]), 0)

    def test_small_relative_spike_below_absolute_delta_is_ignored(self) -> None:
        size = 32 << 10
        report = self.run_compare([
            fact("a", [case(size, latency=0.00016)]),
            fact("b", [case(size, latency=0.00010)]),
            fact("c", [case(size, latency=0.00010)]),
        ], small_latency_warn=True)
        self.assertEqual(report["dynamic_compare_status"], "PASS")
        self.assertEqual(report["candidate_count"], 0)

    def test_transition_band_does_not_create_candidate(self) -> None:
        size = 512 << 20
        report = self.run_compare([
            fact("a", [case(size, latency=9.0, busbw=0.01)]),
            fact("b", [case(size)]),
            fact("c", [case(size)]),
        ])
        self.assertEqual(report["dynamic_compare_status"], "PASS")
        self.assertEqual(report["candidate_count"], 0)

    def test_large_message_uses_average_busbw(self) -> None:
        size = 1 << 30
        report = self.run_compare([
            fact("a", [case(size, latency=0.001, busbw=40.0)]),
            fact("b", [case(size, latency=5.0, busbw=100.0)]),
            fact("c", [case(size, latency=5.0, busbw=100.0)]),
        ])
        self.assertEqual(report["dynamic_compare_status"], "RETEST_REQUIRED")
        self.assertEqual(report["retest_plan"][0]["message_bytes"], size)

    def test_retest_candidate_shift_is_inconclusive_and_bounded(self) -> None:
        size = 1 << 30
        first = [
            fact("a", [case(size, busbw=40.0)]),
            fact("b", [case(size, busbw=100.0)]),
            fact("c", [case(size, busbw=100.0)]),
        ]
        retest = [
            fact("a", [case(size, busbw=100.0)]),
            fact("b", [case(size, busbw=40.0)]),
            fact("c", [case(size, busbw=100.0)]),
        ]
        report = self.run_compare(first, retest)
        self.assertEqual(report["dynamic_compare_status"], "PASS")
        self.assertFalse(report["retest_required"])
        self.assertEqual(len(report["inconclusive_cohorts"]), 0)
        self.assertEqual(len(report["retest_only_observations"]), 1)

    def test_retest_repeat_confirms_same_node(self) -> None:
        size = 1 << 30
        rows = [
            fact("a", [case(size, busbw=40.0)]),
            fact("b", [case(size, busbw=100.0)]),
            fact("c", [case(size, busbw=100.0)]),
        ]
        report = self.run_compare(rows, rows)
        self.assertEqual(report["dynamic_compare_status"], "SUSPECT")
        self.assertEqual(report["confirmed_suspects"][0]["pod_name"], "a")

    def test_retest_plan_does_not_include_observation_sizes(self) -> None:
        sizes = [512 << 20, 1 << 30, 2 << 30]
        report = self.run_compare([
            fact("a", [case(size, busbw=40.0 if size == 1 << 30 else 100.0) for size in sizes]),
            fact("b", [case(size, busbw=100.0) for size in sizes]),
            fact("c", [case(size, busbw=100.0) for size in sizes]),
        ])
        self.assertEqual(report["dynamic_compare_status"], "RETEST_REQUIRED")
        self.assertEqual(
            {row["message_bytes"] for row in report["retest_plan"]},
            {1 << 30, 2 << 30},
        )

    def test_small_message_correctness_failure_remains_hard_failure(self) -> None:
        broken = fact("a", [case(1 << 10)])
        broken["summary"]["correctness_pass"] = False
        broken["summary"]["error_type"] = "injected correctness failure"
        report = self.run_compare([broken, fact("b", [case(1 << 10)]), fact("c", [case(1 << 10)])])
        self.assertEqual(report["dynamic_compare_status"], "FAIL")
        self.assertEqual(report["issues"][0]["classification"], "HARD_FAILURE")


if __name__ == "__main__":
    unittest.main()
