from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import vcctl_node_loss_repro as repro


def raw_pod(index: int, *, ready: bool = True, phase: str = "Running") -> dict[str, object]:
    task = "master" if index == 0 else "worker"
    ordinal = 0 if index == 0 else index - 1
    return {
        "metadata": {
            "name": f"test-job-{task}-{ordinal}",
            "uid": f"uid-{index}",
            "namespace": "default",
            "labels": {"volcano.sh/task-spec": task},
        },
        "spec": {
            "nodeName": f"node-{index:03d}",
            "containers": [{"name": task}],
        },
        "status": {
            "phase": phase,
            "podIP": f"198.51.100.{index % 250 + 1}" if ready else "",
            "hostIP": f"192.0.2.{index % 250 + 1}",
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "reason": "" if ready else "UnexpectedAdmissionError",
            "message": "" if ready else "unhealthy devices",
        },
    }


def make_args(root: Path, *, target: int = 96, excluded: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        project_dir=root,
        pod_project_dir=root,
        job_name="test-job",
        namespace="default",
        vcctl_bin="vcctl",
        target_nodes=target,
        gpus_per_node=8,
        excluded_nodes=excluded,
        result_root=root / "results",
        local_output_root=root / "local",
        run_id="run-1",
        megatron_path=root,
        driver_python=Path("/usr/bin/python3"),
        pod_python="/opt/conda/bin/python3",
        idle_seconds=1,
        cooldown_seconds=1,
        post_failure_observe_seconds=1,
        poll_seconds=1,
        exec_timeout_seconds=1,
        controller_timeout_seconds=1,
        master_port=29741,
        preflight_only=True,
        confirmation="",
    )


class ExclusionParsingTests(unittest.TestCase):
    def test_comma_space_newline_and_duplicates(self) -> None:
        self.assertEqual(
            repro.parse_excluded_nodes("node-c,node-a node-b\nnode-a"),
            ["node-a", "node-b", "node-c"],
        )


class SelectionTests(unittest.TestCase):
    def run_selection(
        self,
        raws: list[dict[str, object]],
        *,
        target: int,
        excluded: str,
    ) -> tuple[repro.ReproductionRun, list[object]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        controller = repro.ReproductionRun(make_args(root, target=target, excluded=excluded))
        controller.initialize_directories()
        stream = "".join(json.dumps(raw) for raw in raws)
        with mock.patch.object(controller, "vcctl_raw", return_value=stream):
            selected = controller.select_nodes()
        return controller, selected

    def test_excluded_failed_node_is_ignored_and_pool_is_stably_filled(self) -> None:
        raws = [raw_pod(index) for index in range(100)]
        raws[10] = raw_pod(10, ready=False, phase="Failed")
        controller, selected = self.run_selection(
            raws,
            target=96,
            excluded="node-010",
        )
        selected_nodes = [pod.node_name for pod in selected]
        self.assertEqual(len(selected_nodes), 96)
        self.assertNotIn("node-010", selected_nodes)
        self.assertEqual(selected_nodes[:3], ["node-000", "node-001", "node-002"])
        self.assertEqual(selected_nodes[-1], "node-096")
        self.assertEqual(controller.excluded_nodes, {"node-010"})
        self.assertEqual(controller.state["excluded_nodes"], ["node-010"])

    def test_unknown_excluded_hostname_is_rejected(self) -> None:
        raws = [raw_pod(index) for index in range(96)]
        with self.assertRaisesRegex(repro.PreflightError, "unknown EXCLUDED_NODES"):
            self.run_selection(raws, target=96, excluded="node-missing")

    def test_unexcluded_failed_node_blocks_preflight(self) -> None:
        raws = [raw_pod(index) for index in range(96)]
        raws[8] = raw_pod(8, ready=False, phase="Failed")
        with self.assertRaisesRegex(repro.PreflightError, "not Running/Ready"):
            self.run_selection(raws, target=96, excluded="")

    def test_insufficient_eligible_nodes_does_not_downgrade(self) -> None:
        raws = [raw_pod(index) for index in range(128)]
        with self.assertRaisesRegex(
            repro.PreflightError,
            "required_nodes=128 eligible_nodes=127 excluded_nodes=1",
        ):
            self.run_selection(raws, target=128, excluded="node-127")


class LossClassificationTests(unittest.TestCase):
    def make_controller(self) -> repro.ReproductionRun:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        controller = repro.ReproductionRun(make_args(root))
        controller.initialize_directories()
        controller.selected_node_names = {"node-selected"}
        return controller

    def test_selected_node_loss_uses_exit_10(self) -> None:
        controller = self.make_controller()
        issues = [
            {
                "type": "POD_NOT_READY",
                "pod_name": "pod-selected",
                "current": {
                    "pod_name": "pod-selected",
                    "node_name": "node-selected",
                    "reason": "UnexpectedAdmissionError",
                    "message": "unhealthy devices",
                },
            }
        ]
        self.assertEqual(
            controller.classify_and_write_loss(issues),
            repro.EXIT_NODE_LOSS_REPRODUCED,
        )

    def test_standby_node_loss_uses_exit_11(self) -> None:
        controller = self.make_controller()
        issues = [
            {
                "type": "POD_NOT_READY",
                "pod_name": "pod-standby",
                "current": {
                    "pod_name": "pod-standby",
                    "node_name": "node-standby",
                    "reason": "UnexpectedAdmissionError",
                    "message": "unhealthy devices",
                },
            }
        ]
        self.assertEqual(
            controller.classify_and_write_loss(issues),
            repro.EXIT_JOB_INFRA_LOSS,
        )

    def test_first_seen_time_survives_final_rewrite(self) -> None:
        controller = self.make_controller()
        first = {
            "type": "POD_NOT_READY",
            "pod_name": "pod-selected",
            "current": {
                "pod_name": "pod-selected",
                "node_name": "node-selected",
                "reason": "",
                "message": "",
            },
        }
        delayed = {
            "type": "POD_NOT_READY",
            "pod_name": "pod-delayed",
            "current": {
                "pod_name": "pod-delayed",
                "node_name": "node-delayed",
                "reason": "",
                "message": "",
            },
        }
        with mock.patch.object(repro, "now", side_effect=["time-1", "time-2"]):
            controller.classify_and_write_loss([first])
            controller.classify_and_write_loss([first, delayed])
        rows = (controller.shared_dir / "lost_nodes.tsv").read_text(encoding="utf-8")
        self.assertIn("pod-selected\t\ttime-1\t", rows)
        self.assertIn("pod-delayed\t\ttime-2\t", rows)


if __name__ == "__main__":
    unittest.main()
