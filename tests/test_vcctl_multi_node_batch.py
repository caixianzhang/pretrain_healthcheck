from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import types
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


def numbered_pods(count: int) -> list[object]:
    return [
        batch.Pod(
            pod_name=f"pod-{index:04d}",
            namespace="default",
            container_name="worker",
            task_spec="worker",
            node_name=f"node-{index:04d}",
            host_ip=f"192.0.2.{index % 250 + 1}",
            pod_ip=f"198.51.100.{index % 250 + 1}",
            raw={},
        )
        for index in range(count)
    ]


def live_raw(
    name: str,
    node: str,
    *,
    uid: str = "uid-1",
    phase: str = "Running",
    ready: bool = True,
    pod_ip: str = "198.51.100.10",
    reason: str = "",
) -> dict[str, object]:
    return {
        "metadata": {"name": name, "uid": uid, "namespace": "default"},
        "spec": {"nodeName": node, "containers": [{"name": "worker"}]},
        "status": {
            "phase": phase,
            "podIP": pod_ip,
            "reason": reason,
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


class JobLivenessTests(unittest.TestCase):
    def test_healthy_snapshot_has_no_issues(self) -> None:
        baseline = [batch.job_liveness_record(live_raw("pod-a", "node-a"))]
        self.assertEqual(batch.evaluate_job_liveness(baseline, list(baseline)), [])

    def test_failed_pod_is_reported_immediately(self) -> None:
        baseline = [batch.job_liveness_record(live_raw("pod-a", "node-a"))]
        current = [
            batch.job_liveness_record(
                live_raw(
                    "pod-a",
                    "node-a",
                    phase="Failed",
                    ready=False,
                    pod_ip="",
                    reason="UnexpectedAdmissionError",
                )
            )
        ]
        issue_types = {item["type"] for item in batch.evaluate_job_liveness(baseline, current)}
        self.assertEqual(issue_types, {"POD_NOT_RUNNING", "POD_NOT_READY", "POD_IP_MISSING"})

    def test_missing_replaced_or_moved_pod_is_reported(self) -> None:
        baseline = [batch.job_liveness_record(live_raw("pod-a", "node-a"))]
        replaced = [batch.job_liveness_record(live_raw("pod-a", "node-b", uid="uid-2"))]
        issue_types = {item["type"] for item in batch.evaluate_job_liveness(baseline, replaced)}
        self.assertEqual(issue_types, {"POD_UID_CHANGED", "POD_NODE_CHANGED"})
        self.assertEqual(
            {item["type"] for item in batch.evaluate_job_liveness(baseline, [])},
            {"POD_MISSING"},
        )

    def test_three_query_failures_abort_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = live_raw("pod-a", "node-a")
            target = batch.pod_from_raw(raw)
            self.assertIsNotNone(target)
            args = argparse.Namespace(
                vcctl_bin="vcctl",
                job_name="test-job",
                namespace="default",
                batch_run_id="test-run",
                job_liveness_check_interval_seconds=1,
                _job_abort_event=threading.Event(),
                _job_abort_payload={},
            )
            monitor = batch.JobLivenessMonitor(args, [target], Path(tmp))
            monitor.stop_event.wait = mock.Mock(return_value=False)
            with mock.patch.object(batch, "query_job_liveness", side_effect=RuntimeError("api down")):
                monitor._run()
            self.assertTrue(args._job_abort_event.is_set())
            self.assertEqual(args._job_abort_payload["reason"], "JOB_LIVENESS_QUERY_UNAVAILABLE")
            self.assertTrue((Path(tmp) / "job_liveness_alert.json").is_file())


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
    def test_pairwise_scale32_and_topology_matrix_modes(self) -> None:
        pairwise = batch.task_matrix_env(batch.GroupTask("pairwise", "r1", "g1", []))
        scale = batch.task_matrix_env(batch.GroupTask("scale32_crosscheck", "r1", "g2", []))
        topology = batch.task_matrix_env(batch.GroupTask("scale64_topology", "r1", "g3", []))
        self.assertEqual(pairwise["COLLECTIVE_BANDWIDTH_MESSAGE_SIZES"], batch.PAIRWISE_MESSAGE_SIZES)
        self.assertEqual(batch.PAIRWISE_MESSAGE_SIZES, batch.FULL_MESSAGE_SIZES)
        self.assertEqual(pairwise["COLLECTIVE_BANDWIDTH_ITERS"], "1")
        self.assertEqual(scale["COLLECTIVE_BANDWIDTH_MESSAGE_SIZES"], batch.FAST_MESSAGE_SIZES)
        self.assertEqual(
            topology,
            {
                "TOPOLOGY_WARMUP": "1",
                "TOPOLOGY_ITERS": "1",
                "TOPOLOGY_OVERLAP_CANARY": "0",
            },
        )

    def test_topology_overlap_canary_is_explicit_opt_in(self) -> None:
        task = batch.GroupTask("scale64_topology", "r1", "g1", [])
        with mock.patch.dict(os.environ, {"TOPOLOGY_OVERLAP_CANARY": "1"}):
            self.assertEqual(batch.task_matrix_env(task)["TOPOLOGY_OVERLAP_CANARY"], "1")

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
    def test_final_topology_validates_prequalified_world_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "manifest.json"
            source.write_text("{}\n", encoding="utf-8")
            group = types.SimpleNamespace(ranks=(0, 1, 2, 3))
            profile = types.SimpleNamespace(
                parallelism={"tp": 4},
                groups={"tp": [group]},
            )
            manifest = types.SimpleNamespace(
                sha256="a" * 64,
                ranks_per_node=8,
                framework={"name": "test", "rank_order": "tp-cp-ep-dp-pp"},
                profiles={256: profile},
            )
            args = argparse.Namespace(
                training_topology_manifest=str(source),
                pod_training_topology_manifest="",
                prequalified_node_count=32,
                gpus_per_node=8,
            )
            with mock.patch.object(batch, "load_training_topology_manifest", return_value=manifest), mock.patch.object(
                batch, "require_profile", return_value=profile
            ) as require:
                batch.prepare_training_topology(
                    args,
                    numbered_pods(128),
                    ["final_training_topology"],
                    root,
                )

            require.assert_called_once_with(manifest, 256, 8)

    def test_prequalified_nodes_seed_topology_candidates_and_archive_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            con = batch.init_db(root / "batch.sqlite")
            pods = [pod("A"), pod("B")]
            batch.upsert_pods(con, pods)
            source = root / "nodes.txt"
            source.write_text("A\nB\n", encoding="utf-8")
            args = argparse.Namespace(
                prequalified_nodes_file=str(source),
                batch_run_id="prequalified-test",
            )
            output = root / "result"
            output.mkdir()

            batch.prepare_prequalified_nodes(con, args, pods, output)
            candidates, direct = batch.phase_candidates("scale64_topology", con, pods)

            self.assertFalse(direct)
            self.assertEqual([item.node_name for item in candidates], ["A", "B"])
            self.assertEqual(args.prequalified_node_count, 2)
            self.assertEqual((output / "prequalified_nodes.txt").read_text(), "A\nB\n")
            self.assertEqual(
                con.execute("select distinct last_phase from nodes").fetchone()[0],
                "external_prequalified",
            )
            con.close()

    def test_prequalified_nodes_reject_unknown_or_changed_resume_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            con = batch.init_db(root / "batch.sqlite")
            pods = [pod("A"), pod("B")]
            batch.upsert_pods(con, pods)
            source = root / "nodes.txt"
            source.write_text("A\nC\n", encoding="utf-8")
            args = argparse.Namespace(prequalified_nodes_file=str(source), batch_run_id="test")
            output = root / "result"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "not present"):
                batch.prepare_prequalified_nodes(con, args, pods, output)
            source.write_text("A\nB\n", encoding="utf-8")
            batch.prepare_prequalified_nodes(con, args, pods, output)
            source.write_text("A\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                batch.prepare_prequalified_nodes(con, args, pods, output)
            con.close()

    def test_direct_final_topology_uses_all_unknown_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            con = batch.init_db(Path(tmp) / "batch.sqlite")
            pods = [pod("A"), pod("B")]
            batch.upsert_pods(con, pods)
            candidates, direct = batch.phase_candidates("final_training_topology", con, pods)
            self.assertTrue(direct)
            self.assertEqual([item.node_name for item in candidates], ["A", "B"])
            con.close()

    def test_direct_final_topology_does_not_bypass_known_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            con = batch.init_db(Path(tmp) / "batch.sqlite")
            pods = [pod("A"), pod("B")]
            batch.upsert_pods(con, pods)
            con.execute("update nodes set status='FAIL' where node_name='A'")
            con.commit()
            candidates, direct = batch.phase_candidates("final_training_topology", con, pods)
            self.assertFalse(direct)
            self.assertEqual(candidates, [])
            con.close()


class FinalTopologyGateTests(unittest.TestCase):
    def test_failed_root_is_not_hidden_by_passing_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            con = batch.init_db(Path(tmp) / "batch.sqlite")
            pods = [pod(name) for name in "ABCD"]
            root = batch.GroupTask(
                "final_training_topology",
                "final_training_topology_r1",
                "final_training_topology_group_0000",
                pods,
            )
            batch.table_group(con, root, "FAIL")
            for index, part in enumerate((pods[:2], pods[2:])):
                split = batch.GroupTask(
                    "scale32_crosscheck",
                    "final_training_topology_r1_split",
                    f"final_training_topology_group_0000_split_{index}",
                    part,
                    parent_group_id=root.group_id,
                    attempt=1,
                )
                batch.table_group(con, split, "PASS")
            self.assertTrue(batch.final_training_topology_root_failed(con))
            con.close()

    def test_passing_or_reused_root_is_healthy(self) -> None:
        for status in ("PASS",):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                con = batch.init_db(Path(tmp) / "batch.sqlite")
                root = batch.GroupTask(
                    "final_training_topology",
                    "final_training_topology_r1",
                    "final_training_topology_group_0000",
                    [],
                )
                batch.table_group(con, root, status)
                self.assertFalse(batch.final_training_topology_root_failed(con))
                con.close()


class TopologyPhasePlanTests(unittest.TestCase):
    def test_auto_phases_replace_legacy_global_scale_phases(self) -> None:
        self.assertEqual(
            batch.auto_phases(128, 128),
            [
                "pairwise",
                "ep8",
                "scale32_crosscheck",
                "scale64_topology",
                "final_training_topology",
            ],
        )
        with self.assertRaisesRegex(ValueError, "legacy global scale phases"):
            batch.parse_phases("pairwise,scale64,final_all", 128, 128)

    def test_scale32_has_two_complete_deterministic_rounds(self) -> None:
        pods = numbered_pods(128)
        first = batch.make_phase_groups("scale32_crosscheck", pods, 17)
        second = batch.make_phase_groups("scale32_crosscheck", pods, 17)
        self.assertEqual(
            [[pod.node_name for pod in task.pods] for task in first],
            [[pod.node_name for pod in task.pods] for task in second],
        )
        self.assertEqual(len(first), 8)
        for round_id in ("scale32_crosscheck_r1", "scale32_crosscheck_r2"):
            round_tasks = [task for task in first if task.round_id == round_id]
            self.assertEqual(len(round_tasks), 4)
            self.assertEqual({len(task.pods) for task in round_tasks}, {32})
            self.assertEqual(
                sorted(pod.node_name for task in round_tasks for pod in task.pods),
                sorted(pod.node_name for pod in pods),
            )

    def test_scale64_crosses_halves_in_two_rounds(self) -> None:
        pods = numbered_pods(128)
        tasks = batch.make_phase_groups("scale64_topology", pods, 19)
        self.assertEqual(len(tasks), 4)
        rounds = {
            round_id: [task for task in tasks if task.round_id == round_id]
            for round_id in ("scale64_topology_r1", "scale64_topology_r2")
        }
        for round_tasks in rounds.values():
            self.assertEqual(len(round_tasks), 2)
            self.assertEqual({len(task.pods) for task in round_tasks}, {64})
            self.assertEqual(
                sorted(pod.node_name for task in round_tasks for pod in task.pods),
                sorted(pod.node_name for pod in pods),
            )
        round1_sets = {frozenset(pod.node_name for pod in task.pods) for task in rounds["scale64_topology_r1"]}
        round2_sets = {frozenset(pod.node_name for pod in task.pods) for task in rounds["scale64_topology_r2"]}
        self.assertTrue(round1_sets.isdisjoint(round2_sets))

    def test_final_topology_is_one_full_candidate_group(self) -> None:
        pods = numbered_pods(128)
        tasks = batch.make_phase_groups("final_training_topology", pods, 23)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].phase, "final_training_topology")
        self.assertEqual(tasks[0].pods, pods)

    def test_scale64_performance_requires_cross_round_node_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            con = batch.init_db(Path(tmp) / "batch.sqlite")
            pods = numbered_pods(128)
            batch.upsert_pods(con, pods)
            for item in pods:
                batch.set_node_status(con, item.node_name, "PASS", "scale32_crosscheck")
            tasks = batch.make_phase_groups("scale64_topology", pods, 19)
            for task in tasks:
                batch.table_group(con, task, "PASS")
            round1 = next(task for task in tasks if task.round_id == "scale64_topology_r1")
            round2 = next(task for task in tasks if task.round_id == "scale64_topology_r2")
            case = {
                "pod_name": round1.group_id,
                "case_id": "training_topology/ep/all_to_allv/1G/hot_expert/32",
            }
            batch.record_performance_candidates(
                con, round1.phase, round1.round_id, [case], "CONFIRMED"
            )
            self.assertEqual(batch.apply_performance_candidate_statuses(con), [])

            batch.record_performance_candidates(
                con,
                round2.phase,
                round2.round_id,
                [{**case, "pod_name": round2.group_id}],
                "CONFIRMED",
            )
            suspects = batch.apply_performance_candidate_statuses(con)
            expected = sorted(
                {pod.node_name for pod in round1.pods}
                & {pod.node_name for pod in round2.pods}
            )
            self.assertEqual(suspects, expected)
            self.assertEqual(len(suspects), 32)
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


if __name__ == "__main__":
    unittest.main()
