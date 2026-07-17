from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


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


class MatrixAndRetestPolicyTests(unittest.TestCase):
    def test_pairwise_and_final_use_full_matrix(self) -> None:
        pairwise = batch.task_matrix_env(batch.GroupTask("pairwise", "r1", "g1", []))
        final_all = batch.task_matrix_env(batch.GroupTask("final_all", "r1", "g2", []))
        scale = batch.task_matrix_env(batch.GroupTask("scale64", "r1", "g3", []))
        self.assertEqual(pairwise["COLLECTIVE_BANDWIDTH_MESSAGE_SIZES"], batch.PAIRWISE_MESSAGE_SIZES)
        self.assertEqual(batch.PAIRWISE_MESSAGE_SIZES, batch.FULL_MESSAGE_SIZES)
        self.assertEqual(pairwise["COLLECTIVE_BANDWIDTH_ITERS"], "1")
        self.assertEqual(final_all["COLLECTIVE_BANDWIDTH_MESSAGE_SIZES"], batch.FULL_MESSAGE_SIZES)
        self.assertEqual(final_all["COLLECTIVE_BANDWIDTH_ITERS"], "3")
        self.assertEqual(scale["COLLECTIVE_BANDWIDTH_MESSAGE_SIZES"], batch.FAST_MESSAGE_SIZES)

    def test_small_cohort_retests_all_groups(self) -> None:
        tasks = [batch.GroupTask("pairwise", "r1", f"g{i}", []) for i in range(8)]
        args = argparse.Namespace(dynamic_compare_retest_max_groups=32, group_seed=7)
        batches = batch.select_performance_retest_batches(tasks, {"g0"}, args)
        self.assertEqual(len(batches), 1)
        selected, controls = batches[0]
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(controls), 7)

    def test_large_cohort_uses_candidates_and_adaptive_controls(self) -> None:
        tasks = [batch.GroupTask("pairwise", "r1", f"g{i:03d}", []) for i in range(512)]
        args = argparse.Namespace(dynamic_compare_retest_max_groups=32, group_seed=7)
        batches = batch.select_performance_retest_batches(tasks, {"g000"}, args)
        self.assertEqual(len(batches), 1)
        selected, controls = batches[0]
        self.assertEqual(len(selected), 9)
        self.assertEqual(len(controls), 8)
        self.assertIn("g000", {task.group_id for task in selected})
        selected_again, controls_again = batch.select_performance_retest_batches(tasks, {"g000"}, args)[0]
        self.assertEqual([task.group_id for task in selected], [task.group_id for task in selected_again])
        self.assertEqual(controls, controls_again)

    def test_all_candidates_are_retested_in_bounded_batches(self) -> None:
        tasks = [batch.GroupTask("pairwise", "r1", f"g{i:03d}", []) for i in range(512)]
        candidates = {f"g{i:03d}" for i in range(40)}
        args = argparse.Namespace(dynamic_compare_retest_max_groups=32, group_seed=7)
        batches = batch.select_performance_retest_batches(tasks, candidates, args)
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(selected) <= 32 for selected, _controls in batches))
        observed = [
            task.group_id
            for selected, _controls in batches
            for task in selected
            if task.group_id in candidates
        ]
        self.assertEqual(sorted(observed), sorted(candidates))
        self.assertEqual(len(observed), len(set(observed)))
        self.assertTrue(all(len(controls) == 8 for _selected, controls in batches))


class RuntimeWarningTests(unittest.TestCase):
    def test_runtime_warning_records_event_without_stopping_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            con = batch.init_db(Path(tmp) / "batch.sqlite")
            args = argparse.Namespace(
                batch_run_id="runtime-warning-test",
                batch_runtime_warn_seconds=1,
                batch_started_monotonic=time.monotonic() - 2,
                _runtime_warning_emitted=False,
                _runtime_warning_lock=threading.Lock(),
            )
            batch.maybe_emit_runtime_warning(args, con)
            self.assertTrue(args._runtime_warning_emitted)
            row = con.execute(
                "select event_type from events where event_type='runtime_target_exceeded'"
            ).fetchone()
            self.assertIsNotNone(row)
            con.close()


class PhaseCandidateTests(unittest.TestCase):
    def test_direct_final_all_uses_all_unknown_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            con = batch.init_db(Path(tmp) / "batch.sqlite")
            pods = [pod("A"), pod("B")]
            batch.upsert_pods(con, pods)
            candidates, direct = batch.phase_candidates("final_all", con, pods)
            self.assertTrue(direct)
            self.assertEqual([item.node_name for item in candidates], ["A", "B"])
            con.close()

    def test_direct_final_all_does_not_bypass_known_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            con = batch.init_db(Path(tmp) / "batch.sqlite")
            pods = [pod("A"), pod("B")]
            batch.upsert_pods(con, pods)
            con.execute("update nodes set status='FAIL' where node_name='A'")
            con.commit()
            candidates, direct = batch.phase_candidates("final_all", con, pods)
            self.assertFalse(direct)
            self.assertEqual(candidates, [])
            con.close()


class FinalAllGateTests(unittest.TestCase):
    def test_failed_root_is_not_hidden_by_passing_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            con = batch.init_db(Path(tmp) / "batch.sqlite")
            pods = [pod(name) for name in "ABCD"]
            root = batch.GroupTask("final_all", "final_all_r1", "final_all_group_0000", pods)
            batch.table_group(con, root, "FAIL")
            for index, part in enumerate((pods[:2], pods[2:])):
                split = batch.GroupTask(
                    "final_all",
                    "final_all_r1_split",
                    f"final_all_group_0000_split_{index}",
                    part,
                    parent_group_id=root.group_id,
                    attempt=1,
                )
                batch.table_group(con, split, "PASS")
            self.assertTrue(batch.final_all_root_failed(con))
            con.close()

    def test_passing_or_reused_root_is_healthy(self) -> None:
        for status in ("PASS", "REUSED_PASS"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                con = batch.init_db(Path(tmp) / "batch.sqlite")
                root = batch.GroupTask("final_all", "final_all_r1", "final_all_group_0000", [])
                batch.table_group(con, root, status)
                self.assertFalse(batch.final_all_root_failed(con))
                con.close()


class PerformanceCandidateTests(unittest.TestCase):
    def test_systemic_event_retests_and_does_not_mark_nodes_as_hard_suspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            con = batch.init_db(root / "batch.sqlite")
            pods = [pod(chr(65 + index)) for index in range(26)]
            tasks = [
                batch.GroupTask("pairwise", "pairwise_r2", f"g{index:02d}", [pods[index % 26]])
                for index in range(64)
            ]
            batch.upsert_pods(con, pods)
            for task in tasks:
                batch.table_group(con, task, "PASS")
            args = argparse.Namespace(
                dynamic_compare_latency_ratio_threshold=1.5,
                dynamic_compare_busbw_ratio_threshold=0.7,
                dynamic_compare_min_cohort=3,
                dynamic_compare_small_max_bytes=1024 * 1024,
                dynamic_compare_large_min_bytes=1024**3,
                dynamic_compare_small_latency_warn=False,
                dynamic_compare_small_latency_abs_delta_seconds=0.0002,
                dynamic_compare_small_latency_mad_multiplier=6.0,
                dynamic_compare_retest_max_groups=32,
                dynamic_compare_systemic_candidate_fraction=0.05,
                dynamic_compare_auto_retest=True,
                group_seed=7,
                phase_group_concurrency=0,
            )
            initial = [
                {
                    "pod_name": f"g{index:02d}",
                    "case_id": "collective_bandwidth/all_reduce/1G/none/32",
                    "message_class": "large",
                }
                for index in range(4)
            ]
            with (
                mock.patch.object(batch, "_task_case_metrics", return_value=[{"initial": True}]),
                mock.patch.object(batch, "candidate_performance_issues", side_effect=[(initial, {}), ([], {})]),
                mock.patch.object(batch, "build_retest_plan", return_value=[{"case": "all_reduce-1G"}]),
                mock.patch.object(batch, "_run_performance_retest_round", return_value=([], [])) as run_retest,
            ):
                confirmed, advisory = batch.compare_round_performance(tasks, args, con, root)

            self.assertEqual(confirmed, [])
            self.assertTrue(advisory)
            run_retest.assert_called_once()
            event = con.execute(
                "select payload_json from events where event_type='systemic_performance_event'"
            ).fetchone()
            self.assertIsNotNone(event)
            self.assertTrue(json.loads(event[0])["execution_continued"])
            statuses = {status for _node, status in con.execute("select node_name,status from nodes")}
            self.assertEqual(statuses, {"UNKNOWN"})
            candidate_statuses = {
                status for (status,) in con.execute("select status from performance_candidates")
            }
            self.assertEqual(candidate_statuses, {"RECOVERED"})
            con.close()

    def test_unresolved_performance_candidates_are_reported_without_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            con = batch.init_db(root / "batch.sqlite")
            pods = [pod("A"), pod("B"), pod("C")]
            batch.upsert_pods(con, pods)
            for item in pods:
                batch.set_node_status(con, item.node_name, "PASS", "pairwise")
            task = batch.GroupTask("pairwise", "pairwise_r1", "g1", pods[:2])
            batch.table_group(con, task, "PASS")
            candidate = {
                "pod_name": "g1",
                "case_id": "collective_bandwidth/all_reduce/1G/none/32",
                "message_class": "large",
            }
            batch.record_performance_candidates(
                con, "pairwise", "pairwise_r1", [candidate], "CONFIRMED"
            )

            nodes = batch.apply_performance_candidate_statuses(con)
            batch.write_performance_candidate_files(con, root)
            batch.write_node_files(con, root)

            self.assertEqual(nodes, ["A", "B"])
            statuses = dict(con.execute("select node_name,status from nodes"))
            self.assertEqual(statuses, {"A": "SUSPECT", "B": "SUSPECT", "C": "PASS"})
            self.assertEqual((root / "suspect_nodes.txt").read_text().splitlines(), ["A", "B"])
            self.assertEqual((root / "performance_candidate_nodes.txt").read_text().splitlines(), ["A", "B"])
            self.assertIn("pod-C", (root / "node_map.txt").read_text())
            self.assertNotIn("pod-A", (root / "node_map.txt").read_text())
            con.close()

    def test_recovered_candidate_does_not_mark_nodes_suspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            con = batch.init_db(Path(tmp) / "batch.sqlite")
            pods = [pod("A"), pod("B")]
            batch.upsert_pods(con, pods)
            for item in pods:
                batch.set_node_status(con, item.node_name, "PASS", "pairwise")
            batch.table_group(con, batch.GroupTask("pairwise", "pairwise_r1", "g1", pods), "PASS")
            batch.record_performance_candidates(
                con,
                "pairwise",
                "pairwise_r1",
                [{"pod_name": "g1", "case_id": "case-1"}],
                "RECOVERED",
            )
            self.assertEqual(batch.apply_performance_candidate_statuses(con), [])
            self.assertTrue(all(status == "PASS" for _node, status in con.execute("select node_name,status from nodes")))
            con.close()


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


class FinalAllReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.con = batch.init_db(Path(self.tmp.name) / "batch.sqlite")
        self.args = argparse.Namespace(
            healthcheck_script="/tmp/run_vcctl_healthcheck.sh",
            pod_project_dir="/tmp/pretrain_healthcheck",
            dynamic_compare="1",
            dynamic_compare_busbw_ratio_threshold=0.7,
            dynamic_compare_latency_ratio_threshold=1.5,
            dynamic_compare_small_max_size="1M",
            dynamic_compare_large_min_size="1G",
            dynamic_compare_small_latency_warn=False,
            dynamic_compare_min_cohort=3,
            dynamic_compare_auto_retest=True,
            batch_fault_type="",
            job_name="test-job",
            batch_run_id="test-run",
        )

    def tearDown(self) -> None:
        self.con.close()
        self.tmp.cleanup()

    def prepare_source(self, names: str, groups: list[str], *, signature: str | None = None) -> list[object]:
        pods = [pod(name) for name in names]
        batch.upsert_pods(self.con, pods)
        by_name = {item.node_name: item for item in pods}
        for index, group_names in enumerate(groups):
            task = batch.GroupTask(
                "ep8",
                "ep8_r1",
                f"ep8_r1_group_{index:04d}",
                [by_name[name] for name in group_names],
            )
            batch.table_group(self.con, task, "PASS")
            metrics = {
                "execution_signature": signature or batch.execution_signature(self.args),
                "case_metrics": [{"op_type": "all_reduce", "message_size": "1M"}],
            }
            self.con.execute(
                """
                insert into group_results(group_id,status,error_type,metrics_json,elapsed_seconds,local_workdirs_json,created_at)
                values(?,?,?,?,?,?,?)
                """,
                (task.group_id, "PASS", "", json.dumps(metrics), 12.5, "{}", batch.iso_now()),
            )
        for item in pods:
            batch.set_node_status(self.con, item.node_name, "PASS", "ep8")
        self.con.commit()
        return pods

    def test_reuses_single_full_group_with_matching_signature(self) -> None:
        pods = self.prepare_source("ABCDEFGH", ["ABCDEFGH"])

        reused = batch.reuse_final_all_from_phase("ep8", pods, self.args, self.con)

        self.assertTrue(reused)
        group = self.con.execute(
            "select status,parent_group_id,nodes_json from groups where group_id='final_all_group_0000'"
        ).fetchone()
        self.assertEqual(group[0], "REUSED_PASS")
        self.assertEqual(group[1], "ep8_r1_group_0000")
        self.assertEqual(json.loads(group[2]), list("ABCDEFGH"))
        metrics = json.loads(
            self.con.execute(
                "select metrics_json from group_results where group_id='final_all_group_0000'"
            ).fetchone()[0]
        )
        self.assertTrue(metrics["reused"])
        self.assertEqual(metrics["source_elapsed_seconds"], 12.5)
        output_dir = Path(self.tmp.name) / "summary"
        output_dir.mkdir()
        batch.write_batch_summary(self.con, output_dir, self.args, "PASS")
        summary = json.loads((output_dir / "batch_summary.json").read_text())
        self.assertEqual(summary["final_all_reuse"]["status"], "REUSED_PASS")
        self.assertEqual(summary["final_all_reuse"]["avoided_group_executions"], 1)
        self.assertIn(
            "execution_continued_after_warning: `false`",
            (output_dir / "batch_summary.md").read_text(),
        )

    def test_does_not_reuse_multiple_groups_with_same_total_size(self) -> None:
        pods = self.prepare_source("ABCDEFGHIJKLMNOP", ["ABCDEFGH", "IJKLMNOP"])

        reused = batch.reuse_final_all_from_phase("ep8", pods, self.args, self.con)

        self.assertFalse(reused)
        self.assertIsNone(
            self.con.execute("select status from groups where group_id='final_all_group_0000'").fetchone()
        )

    def test_does_not_reuse_mismatched_execution_signature(self) -> None:
        pods = self.prepare_source("ABCDEFGH", ["ABCDEFGH"], signature="stale-signature")

        self.assertFalse(batch.reuse_final_all_from_phase("ep8", pods, self.args, self.con))

    def test_fault_injection_disables_reuse(self) -> None:
        pods = self.prepare_source("ABCDEFGH", ["ABCDEFGH"])
        self.args.batch_fault_type = "corrupt"

        self.assertFalse(batch.reuse_final_all_from_phase("ep8", pods, self.args, self.con))

    def test_preserves_existing_executed_final_all_pass(self) -> None:
        pods = self.prepare_source("ABCDEFGH", ["ABCDEFGH"])
        final_task = batch.GroupTask(
            "final_all",
            "final_all_r1",
            "final_all_group_0000",
            pods,
        )
        batch.table_group(self.con, final_task, "PASS")
        self.con.execute(
            """
            insert into group_results(group_id,status,error_type,metrics_json,elapsed_seconds,local_workdirs_json,created_at)
            values(?,?,?,?,?,?,?)
            """,
            (final_task.group_id, "PASS", "", "{}", 21.0, "{}", batch.iso_now()),
        )
        self.con.commit()

        self.assertFalse(batch.reuse_final_all_from_phase("ep8", pods, self.args, self.con))
        group = self.con.execute(
            "select status,parent_group_id from groups where group_id='final_all_group_0000'"
        ).fetchone()
        self.assertEqual(group[0], "PASS")
        self.assertEqual(group[1], "")


class FinalAllSupersetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.con = batch.init_db(Path(self.tmp.name) / "batch.sqlite")
        self.args = argparse.Namespace(
            disable_final_superset_skip=False,
            batch_fault_type="",
            group_seed=20260706,
        )

    def tearDown(self) -> None:
        self.con.close()
        self.tmp.cleanup()

    def prepare(self, names: str) -> list[object]:
        pods = [pod(name) for name in names]
        batch.upsert_pods(self.con, pods)
        for item in pods:
            batch.set_node_status(self.con, item.node_name, "PASS", "pairwise")
        self.con.commit()
        return pods

    def test_ep8_is_superseded_for_exactly_eight_pass_nodes(self) -> None:
        pods = self.prepare("ABCDEFGH")
        self.assertTrue(batch.supersede_phase_with_final_all("ep8", pods, self.args, self.con))
        status = self.con.execute("select status from groups where phase='ep8'").fetchone()[0]
        self.assertEqual(status, "SUPERSEDED")

    def test_ep8_is_not_superseded_when_it_has_multiple_groups(self) -> None:
        pods = self.prepare("ABCDEFGHIJKLMNOP")
        self.assertFalse(batch.supersede_phase_with_final_all("ep8", pods, self.args, self.con))

    def test_fault_injection_disables_superset_skip(self) -> None:
        pods = self.prepare("ABCDEFGH")
        self.args.batch_fault_type = "corrupt"
        self.assertFalse(batch.supersede_phase_with_final_all("ep8", pods, self.args, self.con))


if __name__ == "__main__":
    unittest.main()
