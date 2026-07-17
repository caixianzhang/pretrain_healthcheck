from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.dynamic_compact import summarize_dynamic_suite


class DynamicCompactPolicyTest(unittest.TestCase):
    def test_performance_gate_failure_is_not_a_correctness_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ["smoke", "quick", "bandwidth", "collective_bandwidth"]:
                (root / name).mkdir(parents=True)
            (root / "smoke" / "ping_summary.json").write_text(
                json.dumps({"status": "PASS", "ranks": [{"rank": 0}]}), encoding="utf-8"
            )
            bandwidth = {
                "op_type": "all_reduce",
                "message_size": "64M",
                "message_bytes": 64 << 20,
                "collective_group_size": 16,
                "latency_p50": 0.01,
                "avg_busbw": 50.0,
                "second_lowest_busbw": 49.0,
                "correctness_pass": True,
                "performance_pass": False,
                "error_type": "BandwidthGateFailed",
            }
            (root / "bandwidth" / "bandwidth_summary.jsonl").write_text(
                json.dumps(bandwidth) + "\n", encoding="utf-8"
            )

            summary = summarize_dynamic_suite(root)

            self.assertTrue(summary["correctness_pass"])
            self.assertFalse(summary["performance_pass"])
            self.assertEqual(summary["correctness_failed_stages"], [])
            self.assertEqual(summary["performance_failed_stages"], ["bandwidth"])
            self.assertEqual(summary["case_metrics"][0]["stage"], "bandwidth")


if __name__ == "__main__":
    unittest.main()
