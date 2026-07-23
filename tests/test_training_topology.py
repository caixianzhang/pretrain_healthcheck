from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pretrain_healthcheck.training_topology import (
    load_training_topology_manifest,
    require_profile,
    topology_case_plan,
)
from tools.dynamic_compact import build_payload
from tools.dynamic_compare import extract_case_metrics, topology_coverage_issues
from tools.export_megatron_training_topology import create_generator, groups


def manifest_payload(world_size: int = 8, ranks_per_node: int = 4) -> dict[str, object]:
    ranks = list(range(world_size))

    def partition(size: int) -> list[dict[str, object]]:
        return [
            {"group_id": f"g{index:02d}", "ranks": ranks[offset : offset + size]}
            for index, offset in enumerate(range(0, world_size, size))
        ]

    return {
        "schema_version": 1,
        "ranks_per_node": ranks_per_node,
        "framework": {
            "name": "Megatron-test",
            "code_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "rank_order": "tp-cp-ep-dp-pp",
        },
        "model": {"name": "unit-test"},
        "profiles": {
            str(world_size): {
                "world_size": world_size,
                "parallelism": {"tp": 2, "ep": 2, "pp": 1, "cp": 1, "dp": 2},
                "groups": {
                    "tp": partition(2),
                    "dense_dp": partition(4),
                    "expert_dp": partition(2),
                    "ep": partition(4),
                    "pp": partition(2),
                },
                "workload_shapes": [
                    {
                        "case_id": "real_ep_payload",
                        "family": "ep",
                        "op": "all_to_allv",
                        "message_bytes": 3 << 20,
                        "payload_pattern": "hot_expert",
                    },
                    {
                        "family": "tp",
                        "op": "all_reduce",
                        "message_bytes": 1 << 20,
                    },
                ],
            }
        },
    }


class TrainingTopologyManifestTests(unittest.TestCase):
    def write_manifest(self, root: Path, payload: dict[str, object]) -> Path:
        path = root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_manifest_is_validated_and_case_plan_adds_only_new_training_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_manifest(Path(tmp), manifest_payload())
            manifest = load_training_topology_manifest(path)
            profile = require_profile(manifest, 8, 4)
            cases = topology_case_plan(profile)

            self.assertEqual(len(manifest.sha256), 64)
            self.assertEqual(len(cases), 43)
            self.assertIn("real_ep_payload", {case["case_id"] for case in cases})
            representative = [
                case
                for case in cases
                if case["family"] == "tp"
                and case["op"] == "all_reduce"
                and case["message_bytes"] == 1 << 20
            ]
            self.assertEqual(len(representative), 1)

    def test_duplicate_or_missing_family_rank_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = manifest_payload()
            groups = payload["profiles"]["8"]["groups"]  # type: ignore[index]
            groups["tp"][1]["ranks"] = [1, 3]  # type: ignore[index]
            path = self.write_manifest(Path(tmp), payload)
            with self.assertRaisesRegex(ValueError, "appears in both|do not cover all ranks"):
                load_training_topology_manifest(path)

    def test_runtime_profile_and_ranks_per_node_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_training_topology_manifest(
                self.write_manifest(Path(tmp), manifest_payload())
            )
            with self.assertRaisesRegex(ValueError, "ranks_per_node"):
                require_profile(manifest, 8, 8)
            with self.assertRaisesRegex(ValueError, "world_size=16"):
                require_profile(manifest, 16, 4)


class TrainingTopologyCompactTests(unittest.TestCase):
    def test_complete_topology_result_passes_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "world_size": 8,
                "case_count": 1,
                "manifest_sha256": "c" * 64,
                "group_counts": {"tp": 4},
            }
            row = {
                "topology_family": "tp",
                "topology_group_id": "cohort",
                "op_type": "all_reduce",
                "message_size": "1M",
                "message_bytes": 1 << 20,
                "requested_message_bytes": 1 << 20,
                "payload_pattern": "none",
                "collective_group_size": 2,
                "measurement_batches": 1,
                "iterations_per_batch": 1,
                "latency_p50": 0.001,
                "latency_p95": 0.001,
                "latency_p99": 0.001,
                "avg_busbw": 1.0,
                "correctness_pass": True,
                "performance_pass": True,
            }
            (root / "training_topology_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            summaries = root / "training_topology_rank_summaries"
            summaries.mkdir()
            (summaries / "rank_000000.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (root / "training_topology_gate.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            args = argparse.Namespace(
                input_dir=root,
                kind="training-topology",
                returncode=0,
                stage="multi_node_training_topology",
                run_id="test",
                pod_name="pod-0",
                node_name="node-0",
                pod_ip="198.51.100.1",
                host_ip="192.0.2.1",
                expected_ranks=0,
                expected_bandwidth_message_sizes="",
                expected_collective_message_sizes="",
                expected_collective_ops="",
                expected_collective_moe_patterns="",
            )

            payload = build_payload(args)

            self.assertTrue(payload["coverage"]["complete"])
            self.assertTrue(payload["summary"]["correctness_pass"])
            self.assertEqual(payload["summary"]["rank_count"], 8)
            self.assertEqual(payload["summary"]["case_metrics"][0]["topology_family"], "tp")

    def test_subgroup_leader_metrics_are_aggregated_without_global_collective(self) -> None:
        def fact(pod: str, start: int) -> dict[str, object]:
            metrics = []
            for index in range(start, start + 2):
                metrics.append(
                    {
                        "stage": "training_topology",
                        "topology_family": "tp",
                        "topology_group_id": f"tp-{index}",
                        "op_type": "all_reduce",
                        "requested_message_bytes": 1 << 30,
                        "message_bytes": 1 << 30,
                        "message_size": "1G",
                        "payload_pattern": "none",
                        "collective_group_size": 4,
                        "latency_p50": 0.01 + index * 0.001,
                        "latency_p95": 0.01 + index * 0.001,
                        "latency_p99": 0.01 + index * 0.001,
                        "avg_busbw": 100.0 - index,
                        "second_lowest_busbw": 100.0 - index,
                        "correctness_pass": True,
                        "performance_pass": True,
                    }
                )
            return {
                "pod": {"name": pod, "node_name": pod},
                "summary": {
                    "summary_owner": True,
                    "topology_group_counts": {"tp": 4},
                    "case_metrics": metrics,
                },
                "coverage": {"expected": {"case_count": 1}},
            }

        facts = [fact("pod-0", 0), fact("pod-1", 2)]
        rows = extract_case_metrics(facts)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["topology_group_id"], "cohort")
        self.assertEqual(rows[0]["subgroup_count"], 4)
        self.assertEqual(rows[0]["avg_busbw"], 97.0)
        self.assertEqual(topology_coverage_issues(facts, rows), [])

    def test_missing_subgroup_summary_is_a_hard_coverage_failure(self) -> None:
        facts = [
            {
                "pod": {"name": "pod-0", "node_name": "node-0"},
                "summary": {
                    "summary_owner": True,
                    "topology_group_counts": {"tp": 2},
                    "case_metrics": [
                        {
                            "stage": "training_topology",
                            "topology_family": "tp",
                            "topology_group_id": "tp-0",
                            "op_type": "all_reduce",
                            "requested_message_bytes": 1 << 30,
                            "message_bytes": 1 << 30,
                            "payload_pattern": "none",
                            "collective_group_size": 4,
                            "latency_p50": 0.01,
                            "avg_busbw": 100.0,
                            "correctness_pass": True,
                            "performance_pass": True,
                        }
                    ],
                },
                "coverage": {"expected": {"case_count": 1}},
            }
        ]
        rows = extract_case_metrics(facts)
        issues = topology_coverage_issues(facts, rows)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["check"], "topology_subgroup_count")


class MegatronExporterTests(unittest.TestCase):
    def test_exporter_uses_separate_dense_and_expert_generators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fake_parallel.py").write_text(
                """
class RankGenerator:
    def __init__(self, tp, ep, dp, pp, cp, order, rank_offset=0):
        self.tp, self.ep, self.dp, self.pp, self.cp = tp, ep, dp, pp, cp
        self.world_size = tp * ep * dp * pp * cp

    def get_ranks(self, token, independent_ep=False):
        size = {"tp": self.tp, "ep": self.ep, "dp": self.dp, "pp": self.pp}[token]
        return [list(range(offset, offset + size)) for offset in range(0, self.world_size, size)]
""".lstrip(),
                encoding="utf-8",
            )
            output = root / "manifest.json"
            exporter = Path(__file__).resolve().parents[1] / "tools/export_megatron_training_topology.py"
            subprocess.run(
                [
                    sys.executable,
                    str(exporter),
                    "--megatron-path", str(root),
                    "--rank-generator-module", "fake_parallel",
                    "--world-size", "64",
                    "--ranks-per-node", "8",
                    "--tp", "4",
                    "--ep", "8",
                    "--etp", "1",
                    "--pp", "2",
                    "--mbs", "1",
                    "--gbs", "64",
                    "--output", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            parallelism = json.loads(output.read_text(encoding="utf-8"))["profiles"]["64"]["parallelism"]
            self.assertEqual(parallelism["dense_dp"], 8)
            self.assertEqual(parallelism["expert_dp"], 4)

    def test_exporter_uses_framework_rank_generator_api(self) -> None:
        class FakeRankGenerator:
            def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str, rank_offset: int = 0):
                self.values = (tp, ep, dp, pp, cp, order, rank_offset)

            def get_ranks(self, token: str, independent_ep: bool = False) -> list[list[int]]:
                self.last_call = (token, independent_ep)
                return [[0, 2], [1, 3]] if independent_ep else [[0, 1], [2, 3]]

        generator = create_generator(FakeRankGenerator, 2, 2, 1, 1, 1, "tp-ep-dp-pp")
        self.assertEqual(generator.values, (2, 2, 1, 1, 1, "tp-ep-dp-pp", 0))
        self.assertEqual(groups(generator, "ep", independent_ep=True), [[0, 2], [1, 3]])
        self.assertEqual(generator.last_call, ("ep", True))


if __name__ == "__main__":
    unittest.main()
