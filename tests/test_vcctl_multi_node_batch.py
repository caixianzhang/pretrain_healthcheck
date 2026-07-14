from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/vcctl_multi_node_batch.py"
if not MODULE_PATH.exists():
    MODULE_PATH = Path("/tmp/vcctl_multi_node_batch.py")
SPEC = importlib.util.spec_from_file_location("vcctl_multi_node_batch", MODULE_PATH)
assert SPEC and SPEC.loader
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


def pod(name: str) -> object:
    return batch.Pod(
        pod_name=f"pod-{name}",
        namespace="default",
        container_name="worker",
        task_spec="worker",
        node_name=name,
        host_ip=f"192.0.2.{ord(name) - 64}",
        pod_ip=f"198.51.100.{ord(name) - 64}",
        raw={},
    )


class BatchTargetTests(unittest.TestCase):
    def test_plural_targets_are_merged_and_propagated(self) -> None:
        args = argparse.Namespace(
            batch_fault_node="node-a",
            batch_fault_nodes="node-b,node-a",
            batch_fault_pod="",
            batch_fault_pods="pod-a,pod-b",
        )
        self.assertEqual(batch._fault_target_nodes(args), ["node-a", "node-b"])
        self.assertEqual(batch._fault_target_pods(args), ["pod-a", "pod-b"])
        env = batch._target_fault_env("FAULT_JOIN_TIMEOUT", args)
        self.assertEqual(env["FAULT_JOIN_TIMEOUT_NODES"], "node-a,node-b")
        self.assertEqual(env["FAULT_JOIN_TIMEOUT_PODS"], "pod-a,pod-b")


class LocalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.con = batch.init_db(Path(self.tmp.name) / "batch.sqlite")

    def tearDown(self) -> None:
        self.con.close()
        self.tmp.cleanup()

    def add_pods(self, names: str) -> dict[str, object]:
        pods = {name: pod(name) for name in names}
        batch.upsert_pods(self.con, list(pods.values()))
        return pods

    def add_group(self, group_id: str, round_id: str, status: str, *pods: object) -> None:
        task = batch.GroupTask("pairwise", round_id, group_id, list(pods))
        batch.table_group(self.con, task, status)

    def statuses(self) -> dict[str, str]:
        return dict(self.con.execute("select node_name,status from nodes"))

    def test_single_persistent_node_isolated_from_healthy_partners(self) -> None:
        pods = self.add_pods("ABCDEF")
        self.add_group("g1", "pairwise_r1", "FAIL", pods["A"], pods["B"])
        self.add_group("g2", "pairwise_r2", "FAIL", pods["A"], pods["C"])
        self.add_group("g3", "pairwise_r1", "PASS", pods["B"], pods["D"])
        self.add_group("g4", "pairwise_r2", "PASS", pods["C"], pods["E"])
        self.add_group("g1_retest_A", "pairwise_r1_retest", "FAIL", pods["A"], pods["D"])
        self.add_group("g1_retest_B", "pairwise_r1_retest", "PASS", pods["B"], pods["E"])
        self.add_group("g2_retest_C", "pairwise_r2_retest", "PASS", pods["C"], pods["F"])

        suspects = batch.finalize_pairwise_localization(self.con, "pairwise")

        self.assertEqual(suspects, ["A"])
        self.assertEqual(self.statuses()["A"], "SUSPECT")
        self.assertTrue(all(self.statuses()[name] == "PASS" for name in "BCDEF"))

    def test_retest_anchors_use_pass_evidence_not_last_status(self) -> None:
        pods = self.add_pods("ABCDE")
        failed = batch.GroupTask("pairwise", "pairwise_r1", "failed", [pods["A"], pods["B"]])
        self.add_group("pass_before_fail", "pairwise_r1", "PASS", pods["C"], pods["D"])
        self.add_group("later_failure", "pairwise_r2", "FAIL", pods["C"], pods["E"])
        batch.set_node_status(self.con, "C", "SUSPECT", "pairwise", "later failure")
        self.con.commit()

        retests = batch.retest_pairwise(failed, list(pods.values()), self.con)

        self.assertEqual(len(retests), 2)
        anchors = [task.pods[1].node_name for task in retests]
        self.assertEqual(anchors, ["C", "D"])

    def test_two_persistent_nodes_are_both_isolated(self) -> None:
        pods = self.add_pods("ABCDEFGH")
        self.add_group("g1", "pairwise_r1", "FAIL", pods["A"], pods["B"])
        self.add_group("g2", "pairwise_r2", "FAIL", pods["A"], pods["C"])
        self.add_group("g3", "pairwise_r2", "FAIL", pods["B"], pods["D"])
        self.add_group("g4", "pairwise_r1", "PASS", pods["C"], pods["E"])
        self.add_group("g5", "pairwise_r1", "PASS", pods["D"], pods["F"])
        self.add_group("g6", "pairwise_r2", "PASS", pods["G"], pods["H"])
        self.add_group("g2_retest_A", "pairwise_r2_retest", "FAIL", pods["A"], pods["G"])
        self.add_group("g3_retest_B", "pairwise_r2_retest", "FAIL", pods["B"], pods["H"])
        self.add_group("g2_retest_C", "pairwise_r2_retest", "PASS", pods["C"], pods["E"])
        self.add_group("g3_retest_D", "pairwise_r2_retest", "PASS", pods["D"], pods["F"])

        suspects = batch.finalize_pairwise_localization(self.con, "pairwise")

        self.assertEqual(suspects, ["A", "B"])
        statuses = self.statuses()
        self.assertEqual(statuses["A"], "SUSPECT")
        self.assertEqual(statuses["B"], "SUSPECT")
        self.assertTrue(all(statuses[name] == "PASS" for name in "CDEFGH"))

    def test_pass_evidence_prevents_healthy_anchor_false_positive(self) -> None:
        pods = self.add_pods("ABCDEFGH")
        self.add_group("a1", "pairwise_r1", "FAIL", pods["A"], pods["C"])
        self.add_group("a2", "pairwise_r2", "FAIL", pods["A"], pods["D"])
        self.add_group("b1", "pairwise_r1", "FAIL", pods["B"], pods["C"])
        self.add_group("b2", "pairwise_r2", "FAIL", pods["B"], pods["E"])
        self.add_group("a_retest", "pairwise_r1_retest", "FAIL", pods["A"], pods["F"])
        self.add_group("b_retest", "pairwise_r1_retest", "FAIL", pods["B"], pods["F"])
        self.add_group("c_pass", "pairwise_r2_retest", "PASS", pods["C"], pods["G"])
        self.add_group("d_pass", "pairwise_r2_retest", "PASS", pods["D"], pods["H"])
        self.add_group("e_pass", "pairwise_r2_retest", "PASS", pods["E"], pods["G"])
        self.add_group("f_pass", "pairwise_r2_retest", "PASS", pods["F"], pods["H"])

        suspects = batch.finalize_pairwise_localization(self.con, "pairwise")

        self.assertEqual(suspects, ["A", "B"])
        statuses = self.statuses()
        self.assertTrue(all(statuses[name] == "PASS" for name in "CDEFGH"))


if __name__ == "__main__":
    unittest.main()
