import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools import vcctl_fixed_group_diagnosis as diagnosis
from tools import vcctl_multi_node_batch as batch


def member(rank: int, node: str, pod: str) -> dict:
    return {
        "node_rank": rank,
        "global_rank_start": rank * 8,
        "global_rank_end": rank * 8 + 7,
        "node_name": node,
        "pod_name": pod,
        "host_ip": node.removeprefix("host-").replace("-", "."),
    }


def manifest_group(group_id: str, nodes: list[str]) -> dict:
    members = [member(rank, node, f"job-worker-{node[-1]}") for rank, node in enumerate(nodes)]
    digest = hashlib.sha256("\n".join(nodes).encode("utf-8")).hexdigest()
    return {
        "group_id": group_id,
        "group_size": len(nodes),
        "ordered_node_names_sha256": digest,
        "members": members,
    }


def raw_pod(name: str, node: str) -> dict:
    return {
        "metadata": {
            "name": name,
            "namespace": "default",
            "labels": {"volcano.sh/task-spec": "worker"},
        },
        "spec": {"nodeName": node, "containers": [{"name": "worker", "env": []}]},
        "status": {"hostIP": "10.0.0.1", "podIP": "10.1.0.1"},
    }


class FixedGroupDiagnosisTest(unittest.TestCase):
    def test_scale_failure_is_not_hidden_by_split_passes(self) -> None:
        summary = {
            "overall_status": "PASS",
            "phase_status_counts": [
                {"phase": "pairwise", "status": "PASS", "count": 96},
                {"phase": "scale64", "status": "FAIL", "count": 1},
                {"phase": "scale64", "status": "PASS", "count": 2},
                {"phase": "final_all", "status": "FAIL", "count": 1},
                {"phase": "final_all", "status": "PASS", "count": 2},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = diagnosis.classify_batch_summary(summary, Path(tmp))
        self.assertEqual(result["diagnosis_status"], "SCALE_REPRODUCED")
        self.assertEqual(result["batch_overall_status"], "PASS")
        self.assertEqual(result["failed_phases"], {"scale64": 1, "final_all": 1})

    def test_suspect_nodes_are_preserved(self) -> None:
        summary = {"overall_status": "SUSPECT", "phase_status_counts": []}
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            (result_dir / "suspect_nodes.txt").write_text("node-a\nnode-b\n", encoding="utf-8")
            result = diagnosis.classify_batch_summary(summary, result_dir)
        self.assertEqual(result["diagnosis_status"], "SUSPECT")
        self.assertEqual(result["suspect_nodes"], ["node-a", "node-b"])

    def test_load_and_schedule_disjoint_rounds(self) -> None:
        groups = [
            manifest_group("g0", ["node-a", "node-b"]),
            manifest_group("g1", ["node-c", "node-d"]),
            manifest_group("g2", ["node-a", "node-c"]),
            manifest_group("g3", ["node-b", "node-d"]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"failed_groups": groups}), encoding="utf-8")
            loaded = diagnosis.load_fixed_groups(path, ["g0", "g1", "g2", "g3"], 2)
        rounds = diagnosis.schedule_disjoint_rounds(loaded)
        self.assertEqual([[group.group_id for group in item] for item in rounds], [["g0", "g1"], ["g2", "g3"]])

    def test_manifest_digest_mismatch_is_rejected(self) -> None:
        group = manifest_group("g0", ["node-a", "node-b"])
        group["ordered_node_names_sha256"] = "bad"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"failed_groups": [group]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                diagnosis.load_fixed_groups(path, ["g0"], 2)

    def test_persisted_source_groups_can_be_reused_as_manifest(self) -> None:
        group = manifest_group("g0", ["node-a", "node-b"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source_groups.json"
            path.write_text(json.dumps({"groups": [group]}), encoding="utf-8")
            loaded = diagnosis.load_fixed_groups(path, ["g0"], 2)
        self.assertEqual(loaded[0].nodes, ("node-a", "node-b"))

    def test_pod_json_order_can_be_preserved(self) -> None:
        payload = {"items": [raw_pod("job-worker-9", "node-z"), raw_pod("job-worker-1", "node-a")]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pods.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _, preserved = batch.load_pods(SimpleNamespace(pod_json_file=str(path), preserve_pod_json_order=True))
            _, sorted_pods = batch.load_pods(SimpleNamespace(pod_json_file=str(path), preserve_pod_json_order=False))
        self.assertEqual([pod.pod_name for pod in preserved], ["job-worker-9", "job-worker-1"])
        self.assertEqual([pod.pod_name for pod in sorted_pods], ["job-worker-1", "job-worker-9"])


if __name__ == "__main__":
    unittest.main()
