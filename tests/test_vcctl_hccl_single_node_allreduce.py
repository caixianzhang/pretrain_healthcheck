from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "vcctl_hccl_single_node_allreduce.py"
SPEC = importlib.util.spec_from_file_location("vcctl_hccl_single_node_allreduce", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_hccl_output_uses_slowest_rank_latency() -> None:
    output = """
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
1073741824 | 5000.00 | 214.74836 | success
1073741824 | 6000.00 | 178.95697 | success
"""
    parsed = MODULE.parse_hccl_output(output, 1073741824, 16)

    assert parsed["result_row_count"] == 2
    assert parsed["correctness_pass"] is True
    assert parsed["max_avg_latency_us"] == 6000.0
    assert round(parsed["algbw_gbps"], 6) == 178.956971
    assert round(parsed["busbw_gbps"], 6) == 335.54432


def test_parse_hccl_output_rejects_failed_correctness() -> None:
    output = "1073741824 | 6000.00 | 178.95697 | failed\n"
    parsed = MODULE.parse_hccl_output(output, 1073741824, 16)

    assert parsed["result_row_count"] == 1
    assert parsed["correctness_pass"] is False


def test_parse_hccl_output_ignores_other_message_sizes() -> None:
    output = "536870912 | 3000.00 | 178.95697 | success\n"
    parsed = MODULE.parse_hccl_output(output, 1073741824, 16)

    assert parsed["result_row_count"] == 0
    assert parsed["busbw_gbps"] is None


def test_message_size_bytes() -> None:
    assert MODULE.message_size_bytes("1G") == 1073741824
    assert MODULE.message_size_bytes("1024M") == 1073741824


def test_remote_command_uses_portable_baseline_only() -> None:
    args = Namespace(
        ascend_env_script="/opt/ascend/set_env.sh",
        mpi_bin="/opt/mpich/bin/mpirun",
        mpi_lib_dir="/opt/mpich/lib",
        test_bin="/opt/hccl/all_reduce_test",
        npus_per_node=16,
        message_size="1G",
        dtype="bfp16",
        warmup=1,
        iters=3,
        socket_ifname="eth0",
    )

    aligned = MODULE.build_remote_command(args)

    assert "-n 16" in aligned
    assert "-b 1G -e 1G" in aligned
    assert "-d bfp16" in aligned
    assert "HCCL_OP_EXPANSION_MODE" not in aligned
    assert "HCCL_BUFFSIZE" not in aligned
    assert "CPU_AFFINITY_CONF" not in aligned


def test_select_pods_filters_exact_names_in_requested_order() -> None:
    pod_a = MODULE.Pod("pod-a", "default", "worker", "worker", "node-a", "10.0.0.1", "10.1.0.1")
    pod_b = MODULE.Pod("pod-b", "default", "worker", "worker", "node-b", "10.0.0.2", "10.1.0.2")

    selected = MODULE.select_pods([pod_a, pod_b], ["pod-b"])

    assert selected == [pod_b]
