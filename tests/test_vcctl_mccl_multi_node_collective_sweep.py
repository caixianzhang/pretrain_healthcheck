import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "vcctl_mccl_multi_node_collective_sweep.py"
SPEC = importlib.util.spec_from_file_location("vcctl_mccl_multi_node_collective_sweep", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OfficialMcclMultiNodeSweepTest(unittest.TestCase):
    def test_message_sizes_cover_1k_through_2g(self) -> None:
        sizes = MODULE.message_sizes("1K", "2G", 2)
        self.assertEqual(len(sizes), 22)
        self.assertEqual(sizes[0], 1024)
        self.assertEqual(sizes[-1], 2 * 1024**3)

    def test_parse_nccl_style_rows(self) -> None:
        text = """#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
        1024           256     float     sum      -1    12.50    0.08    0.15      0    11.50    0.09    0.17      0
  1073741824     268435456     float     sum      -1 10000.00  107.37  201.32      0  9900.00  108.45  203.34      0
"""
        rows = MODULE.parse_mccl_rows(text, "all_reduce", 16, {1024, 1024**3})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["message_size"], "1G")
        self.assertAlmostEqual(rows[1]["out_of_place_busbw_gbps"], 201.32)
        self.assertTrue(rows[1]["correctness_pass"])

    def test_parse_rows_rejects_bad_correctness(self) -> None:
        text = "1024 256 float sum -1 12.5 0.08 0.15 1 11.5 0.09 0.17 0\n"
        rows = MODULE.parse_mccl_rows(text, "all_reduce", 16, {1024})
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["correctness_pass"])
        self.assertEqual(MODULE.operation_status(rows, {1024}, 0, False)[1], "CORRECTNESS_FAILED")

    def test_in_place_na_is_valid_when_out_of_place_passes(self) -> None:
        text = "1024 16 float -1 113.08 0.01 0.01 0 107.38 0.01 0.01 N/A\n"
        rows = MODULE.parse_mccl_rows(text, "all_to_all", 16, {1024})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["out_of_place_wrong"], 0)
        self.assertIsNone(rows[0]["in_place_wrong"])
        self.assertTrue(rows[0]["correctness_pass"])

    def test_operation_status_rejects_missing_failed_or_timeout(self) -> None:
        good = [
            {"message_size_bytes": 1024, "correctness_pass": True},
            {"message_size_bytes": 2048, "correctness_pass": True},
        ]
        self.assertEqual(MODULE.operation_status(good, {1024, 2048}, 0, False), ("PASS", ""))
        self.assertEqual(MODULE.operation_status(good[:1], {1024, 2048}, 0, False)[1], "RESULT_MISSING")
        self.assertEqual(MODULE.operation_status(good, {1024, 2048}, 1, False)[1], "EXEC_FAILED")
        self.assertEqual(MODULE.operation_status(good, {1024, 2048}, None, True)[1], "TIMEOUT")

    def test_detect_known_metax_failure_chain(self) -> None:
        text = "ibv_cmd_create_qp_ex failed,ret 5\nATU Fault\nmcErrorIllegalAddress: illegal memory access"
        signatures = MODULE.detect_signatures(text)
        self.assertIn("QP_CREATE_RET5", signatures)
        self.assertIn("ATU_FAULT", signatures)
        self.assertIn("ILLEGAL_ADDRESS", signatures)

    def test_detect_mpi_launcher_failure(self) -> None:
        signatures = MODULE.detect_signatures('plm_rsh_agent path missing; ORTE was unable to reliably start')
        self.assertIn("MPI_LAUNCH_ERROR", signatures)

    def test_mpi_command_has_vendor_environment_and_binary(self) -> None:
        args = SimpleNamespace(
            mpi_bin="/opt/maca/ompi/bin/mpirun",
            test_bin_dir="/opt/maca/mccl_perf",
            min_message_size="1K",
            max_message_size="2G",
            step_factor=2,
            dtype="float",
            warmup=5,
            iters=10,
            maca_path="/opt/maca",
            socket_ifname="eth0",
            ib_hca="xscale_0,xscale_1,xscale_2,xscale_3",
            ib_gid_index="5",
            ib_tc="128",
            enable_vswitch="1",
            pcie_buffer_mode="0",
            cross_nic="1",
            force_active_wait="2",
        )
        pods = [
            MODULE.Pod("master-0", "master", "master", "host-a", "1.1.1.1", "10.0.0.1", 8),
            MODULE.Pod("worker-0", "worker", "worker", "host-b", "1.1.1.2", "10.0.0.2", 8),
        ]
        command = MODULE.build_mpi_command(args, pods, "all_to_allv", "/tmp/run")
        self.assertIn("alltoallv_perf", command)
        self.assertIn("-np 16", command)
        self.assertIn("-host 10.0.0.1:8,10.0.0.2:8", command)
        self.assertIn("-b 1K -e 2G -f 2", command)
        self.assertIn("-w 5 -n 10 -c 1 -a 1", command)
        self.assertIn("MCCL_IB_HCA=xscale_0,xscale_1,xscale_2,xscale_3", command)
        self.assertIn("MCCL_ENABLE_VSWITCH=1", command)

    def test_gpu_count_uses_metax_limit(self) -> None:
        container = {"resources": {"limits": {"metax-tech.com/gpu": "8"}}}
        self.assertEqual(MODULE.resource_gpu_count(container), 8)


if __name__ == "__main__":
    unittest.main()
