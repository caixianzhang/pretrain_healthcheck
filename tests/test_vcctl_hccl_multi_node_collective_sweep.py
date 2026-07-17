import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "vcctl_hccl_multi_node_collective_sweep.py"
SPEC = importlib.util.spec_from_file_location("vcctl_hccl_multi_node_collective_sweep", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OfficialHcclMultiNodeSweepTest(unittest.TestCase):
    def test_message_sizes_cover_1k_through_8g(self) -> None:
        sizes = MODULE.message_sizes("1K", "8G", 2)
        self.assertEqual(len(sizes), 24)
        self.assertEqual(sizes[0], 1024)
        self.assertEqual(sizes[-1], 8 * 1024**3)

    def test_parse_rows_uses_healthcheck_busbw_factor(self) -> None:
        text = """data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
1024 | 60.70 | 0.01687 | success
1073741824 | 10442.38 | 102.82543 | success
"""
        rows = MODULE.parse_hccl_rows(text, "all_reduce", 128, {1024, 1024**3})
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[1]["busbw_factor"], 2 * 127 / 128)
        self.assertAlmostEqual(rows[1]["busbw_gbps"], 102.82543 * 2 * 127 / 128)

    def test_operation_status_rejects_missing_or_bad_rows(self) -> None:
        good = [
            {"message_size_bytes": 1024, "check": "success"},
            {"message_size_bytes": 2048, "check": "success"},
        ]
        self.assertEqual(MODULE.operation_status(good, {1024, 2048}, 0, False), ("PASS", ""))
        self.assertEqual(MODULE.operation_status(good[:1], {1024, 2048}, 0, False)[1], "RESULT_MISSING")
        bad = [dict(good[0]), dict(good[1])]
        bad[1]["check"] = "failed"
        self.assertEqual(MODULE.operation_status(bad, {1024, 2048}, 0, False)[1], "CORRECTNESS_FAILED")
        self.assertEqual(MODULE.operation_status(good, {1024, 2048}, None, True)[1], "TIMEOUT")

    def test_null_check_is_only_accepted_with_official_overflow_warning(self) -> None:
        rows = [{"message_size_bytes": 1024, "check": "null"}]
        self.assertEqual(MODULE.operation_status(rows, {1024}, 0, False)[1], "CORRECTNESS_FAILED")
        self.assertEqual(MODULE.operation_status(rows, {1024}, 0, False, True), ("PASS", ""))
        self.assertEqual(MODULE.correctness_status(rows, True), "SKIPPED_OVERFLOW")

    def test_mpi_command_has_baseline_environment_and_official_binary(self) -> None:
        args = SimpleNamespace(
            mpi_bin="/opt/mpi/mpirun",
            test_bin_dir="/opt/hccl",
            min_message_size="1K",
            max_message_size="8G",
            step_factor=2,
            dtype="bfp16",
            warmup=5,
            iters=30,
            npus_per_node=16,
            ascend_env_script="/opt/ascend/set_env.sh",
            mpi_lib_dir="/opt/mpi/lib",
            socket_ifname="eth0",
        )
        command = MODULE.build_mpi_command(args, "all_to_allv", "/tmp/run", 128)
        self.assertIn("alltoallv_test", command)
        self.assertIn("-n 128", command)
        self.assertIn("-w 5", command)
        self.assertIn("-n 30", command)
        self.assertIn("unset CPU_AFFINITY_CONF", command)
        self.assertNotIn("HCCL_OP_EXPANSION_MODE=AIV", command)


if __name__ == "__main__":
    unittest.main()
