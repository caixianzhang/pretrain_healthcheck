from __future__ import annotations

import base64
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.dynamic_frame import (
    CHUNK_MANIFEST_PREFIX,
    CHUNK_PREFIX,
    DynamicFrameError,
    chunk_manifest,
    decode_frame_line,
    encode_v2_frame,
    iter_chunks,
)
from tools.dynamic_compact import build_payload
from tools.vcctl_healthcheck_driver import ExecResult, collect_dynamic_stdout_results


def chunk_output(data: bytes, chunk_size: int = 256) -> str:
    lines = [CHUNK_MANIFEST_PREFIX + json.dumps(chunk_manifest(data, chunk_size), separators=(",", ":"))]
    for row in iter_chunks(data, chunk_size):
        lines.append(CHUNK_PREFIX + json.dumps(row, separators=(",", ":")))
    return "\n".join(lines) + "\n"


class DynamicFrameProtocolTest(unittest.TestCase):
    def _dynamic_suite_fixture(self, root: Path) -> Namespace:
        for name in ["smoke", "quick", "bandwidth", "collective_bandwidth"]:
            (root / name).mkdir(parents=True)
        (root / "dynamic_suite_plan.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "expected_world_size": 2,
                    "bandwidth_message_sizes": [1 << 30],
                    "collective_message_sizes": [1 << 30],
                    "collective_ops": ["all_reduce", "all_to_allv"],
                    "collective_moe_patterns": ["uniform", "hot_expert"],
                    "expected_case_count": 4,
                }
            ),
            encoding="utf-8",
        )
        (root / "smoke" / "ping_summary.json").write_text(
            json.dumps({"status": "PASS", "ranks": [{"rank": 0}, {"rank": 1}]}), encoding="utf-8"
        )
        (root / "quick" / "rank_detail.jsonl").write_text(json.dumps({"rank": 0}) + "\n", encoding="utf-8")
        (root / "quick" / "group_summary.jsonl").write_text(
            json.dumps({"op_type": "all_reduce", "correctness_pass": True, "performance_pass": True}) + "\n",
            encoding="utf-8",
        )
        bandwidth = {
            "op_type": "all_reduce", "message_size": "1G", "message_bytes": 1 << 30,
            "correctness_pass": True, "performance_pass": True,
        }
        (root / "bandwidth" / "bandwidth_summary.jsonl").write_text(json.dumps(bandwidth) + "\n", encoding="utf-8")
        collective = []
        for op, pattern in [("all_reduce", "none"), ("all_to_allv", "uniform"), ("all_to_allv", "hot_expert")]:
            collective.append(
                {
                    "op_type": op, "payload_pattern": pattern, "message_size": "1G", "message_bytes": 1 << 30,
                    "correctness_pass": True, "performance_pass": True,
                }
            )
        (root / "collective_bandwidth" / "collective_bandwidth_summary.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in collective), encoding="utf-8"
        )
        return Namespace(
            input_dir=root, kind="dynamic-suite", returncode=0, stage="dynamic_suite", run_id="run-1",
            pod_name="pod-0", node_name="node-0", pod_ip="10.0.0.1", host_ip="10.0.0.2",
            expected_ranks=2, expected_bandwidth_message_sizes="1G", expected_collective_message_sizes="1G",
            expected_collective_ops="all_reduce,all_to_allv", expected_collective_moe_patterns="uniform,hot_expert",
        )

    def test_coverage_manifest_proves_expected_cases_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self._dynamic_suite_fixture(Path(temp))
            payload = build_payload(args)
            self.assertTrue(payload["coverage"]["complete"])
            self.assertEqual(payload["coverage"]["expected"]["case_count"], 4)
            self.assertEqual(payload["coverage"]["actual"]["case_count"], 4)
            self.assertTrue(all(row["sha256"] for row in payload["source_files"]))

    def test_missing_source_file_is_data_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = self._dynamic_suite_fixture(Path(temp))
            (Path(temp) / "collective_bandwidth" / "collective_bandwidth_summary.jsonl").unlink()
            payload = build_payload(args)
            self.assertFalse(payload["coverage"]["complete"])
            self.assertEqual(payload["summary"]["error_type"], "DATA_INCOMPLETE")

    def test_v2_round_trip_and_tamper_detection(self) -> None:
        payload = {"pod": {"name": "pod-0"}, "summary": {"case_metrics": [{"value": index} for index in range(100)]}}
        frame = encode_v2_frame(payload)
        decoded, protocol = decode_frame_line(frame)
        self.assertEqual(decoded, payload)
        self.assertEqual(protocol, "v2-gzip-base64")
        with self.assertRaises(DynamicFrameError):
            decode_frame_line(frame[:-17])

    def test_chunks_reassemble_exact_sidecar(self) -> None:
        data = (encode_v2_frame({"payload": "x" * 5000}) + "\n").encode()
        rows = list(iter_chunks(data, 256))
        reconstructed = b"".join(base64.b64decode(row["payload"]) for row in reversed(list(reversed(rows))))
        self.assertEqual(reconstructed, data)
        self.assertEqual(chunk_manifest(data, 256)["file_bytes"], len(data))

    def test_truncated_stdout_recovers_from_verified_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {
                "schema_version": 2,
                "pod": {"name": "pod-0", "run_id": "run-1", "stage": "dynamic_suite"},
                "summary": {"correctness_pass": True, "performance_pass": True, "case_metrics": []},
                "coverage": {"complete": True},
            }
            sidecar_data = (encode_v2_frame(payload) + "\n").encode()
            sidecar = Path("/tmp/pretrain_healthcheck_run-1_pod-0_1/dynamic_suite/.hc_dynamic_result.v2")
            stdout = root / "pod.stdout"
            stderr = root / "pod.stderr"
            stdout.write_text(sidecar_data.decode()[:128], encoding="utf-8")
            stderr.write_text(f"[dynamic-compact] sidecar: {sidecar}\n", encoding="utf-8")
            result = ExecResult(
                pod_name="pod-0",
                container_name="worker",
                mode="single-node",
                node_name="node-0",
                pod_ip="10.0.0.1",
                command="check",
                returncode=0,
                timeout=False,
                status="PASS",
                reason="",
                stdout_path=str(stdout),
                stderr_path=str(stderr),
                started_at="start",
                finished_at="finish",
                elapsed_seconds=1.0,
            )
            args = SimpleNamespace(
                output_dir=str(root / "output"),
                run_id="run-1",
                run_stage="dynamic_suite",
                namespace="default",
                vcctl_bin="vcctl",
                vcctl_timeout_seconds=10,
                dynamic_frame_recovery_deadline_seconds=5,
                dynamic_frame_chunk_size=256,
                dynamic_failed_log_mode="shared",
                max_parallel=0,
            )
            completed = subprocess.CompletedProcess([], 0, chunk_output(sidecar_data), "")
            with mock.patch("tools.vcctl_healthcheck_driver.run_capture", return_value=completed):
                report = collect_dynamic_stdout_results([result], args)

            self.assertEqual(report["fact_count"], 1)
            self.assertEqual(report["failed_count"], 0)
            self.assertEqual(report["recovery_success"], 1)
            facts = [json.loads(line) for line in (root / "output" / "dynamic_facts.jsonl").read_text().splitlines()]
            self.assertTrue(facts[0]["driver_result"]["frame_recovered"])

    def test_wrong_identity_is_not_accepted(self) -> None:
        payload = {"pod": {"name": "other", "run_id": "run-1", "stage": "dynamic_suite"}}
        frame = encode_v2_frame(payload)
        decoded, _ = decode_frame_line(frame)
        self.assertEqual(decoded["pod"]["name"], "other")


if __name__ == "__main__":
    unittest.main()
