#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


NODE_MAP_RE = re.compile(r"^(\S+)\s+->\s+nodeName\s+(\S+)\s+hostIP\s+(\S+)\s*$")


def parse_node_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = NODE_MAP_RE.match(line)
        if not match:
            continue
        pod_short_name, node_name, _host_ip = match.groups()
        mapping[pod_short_name] = node_name
    if not mapping:
        raise ValueError(f"no pod node mapping found in {path}")
    return mapping


def clean_job_metadata(job: dict[str, Any], clone_job_name: str) -> None:
    metadata = job.setdefault("metadata", {})
    metadata["name"] = clone_job_name
    for key in [
        "uid",
        "resourceVersion",
        "generation",
        "creationTimestamp",
        "managedFields",
        "selfLink",
        "deletionGracePeriodSeconds",
        "deletionTimestamp",
    ]:
        metadata.pop(key, None)
    metadata.pop("generateName", None)
    job.pop("status", None)


def set_hostname_affinity(pod_spec: dict[str, Any], node_name: str) -> None:
    pod_spec.pop("nodeName", None)
    affinity = pod_spec.setdefault("affinity", {})
    node_affinity = affinity.setdefault("nodeAffinity", {})
    required = node_affinity.setdefault("requiredDuringSchedulingIgnoredDuringExecution", {})
    terms = required.setdefault("nodeSelectorTerms", [])
    if not terms:
        terms.append({"matchExpressions": []})

    for term in terms:
        expressions = term.setdefault("matchExpressions", [])
        for expr in expressions:
            if expr.get("key") == "kubernetes.io/hostname":
                expr["operator"] = "In"
                expr["values"] = [node_name]
                return

    terms[0].setdefault("matchExpressions", []).append(
        {
            "key": "kubernetes.io/hostname",
            "operator": "In",
            "values": [node_name],
        }
    )


def get_pytorch_worker_names(job: dict[str, Any]) -> list[str]:
    pytorch_args = job.get("spec", {}).get("plugins", {}).get("pytorch", [])
    if not isinstance(pytorch_args, list):
        return []

    worker_names: list[str] = []
    for arg in pytorch_args:
        if isinstance(arg, str) and arg.startswith("--worker="):
            value = arg.split("=", 1)[1].strip()
            if value:
                worker_names.extend(name.strip() for name in value.split(",") if name.strip())
    return worker_names


def update_pytorch_workers(job: dict[str, Any], split_workers: dict[str, list[str]]) -> None:
    if not split_workers:
        return

    pytorch_args = job.get("spec", {}).get("plugins", {}).get("pytorch")
    if not isinstance(pytorch_args, list):
        raise ValueError("worker replicas were split, but spec.plugins.pytorch is missing or invalid")

    new_args: list[Any] = []
    replaced_workers: set[str] = set()
    for arg in pytorch_args:
        if isinstance(arg, str) and arg.startswith("--worker="):
            worker_names = [name.strip() for name in arg.split("=", 1)[1].split(",") if name.strip()]
            expanded: list[str] = []
            for worker_name in worker_names:
                replacement = split_workers.get(worker_name)
                if replacement:
                    expanded.extend(replacement)
                    replaced_workers.add(worker_name)
                else:
                    expanded.append(worker_name)
            new_args.extend(f"--worker={name}" for name in expanded)
        else:
            new_args.append(arg)

    missing = sorted(set(split_workers) - replaced_workers)
    if missing:
        raise ValueError(
            "worker replicas were split, but spec.plugins.pytorch has no matching --worker entry for: "
            + ",".join(missing)
        )

    job["spec"]["plugins"]["pytorch"] = new_args


def set_task_min_available(task: dict[str, Any]) -> None:
    if "minAvailable" in task:
        task["minAvailable"] = 1


def bind_task_to_node(task: dict[str, Any], task_name: str, pod_short_name: str, node_map: dict[str, str]) -> None:
    node_name = node_map.get(pod_short_name)
    if not node_name:
        raise ValueError(f"missing node_map entry for {pod_short_name}")

    pod_spec = task.get("template", {}).get("spec")
    if not isinstance(pod_spec, dict):
        raise ValueError(f"task {task_name!r} template.spec is missing")
    set_hostname_affinity(pod_spec, node_name)


def bind_tasks_to_nodes(job: dict[str, Any], node_map: dict[str, str], allow_extra_node_map: bool) -> None:
    spec = job.get("spec", {})
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("job spec.tasks is empty or missing")

    pytorch_worker_names = set(get_pytorch_worker_names(job))
    split_workers: dict[str, list[str]] = {}
    new_tasks: list[dict[str, Any]] = []
    expected_pods: set[str] = set()
    for task in tasks:
        task_name = task.get("name")
        replicas = task.get("replicas", 1)
        if not task_name:
            raise ValueError("task without name is unsupported")
        if not isinstance(replicas, int) or replicas < 1:
            raise ValueError(f"task {task_name!r} has invalid replicas={replicas!r}")

        if replicas == 1:
            pod_short_name = f"{task_name}-0"
            expected_pods.add(pod_short_name)
            cloned_task = copy.deepcopy(task)
            bind_task_to_node(cloned_task, task_name, pod_short_name, node_map)
            new_tasks.append(cloned_task)
            continue

        if task_name not in pytorch_worker_names:
            raise ValueError(
                f"task {task_name!r} has replicas={replicas}; only PyTorch worker tasks can be "
                "split for exact same-node clone generation"
            )

        split_names: list[str] = []
        for replica_index in range(replicas):
            pod_short_name = f"{task_name}-{replica_index}"
            split_task_name = pod_short_name
            expected_pods.add(pod_short_name)

            cloned_task = copy.deepcopy(task)
            cloned_task["name"] = split_task_name
            cloned_task["replicas"] = 1
            set_task_min_available(cloned_task)
            bind_task_to_node(cloned_task, split_task_name, pod_short_name, node_map)
            new_tasks.append(cloned_task)
            split_names.append(split_task_name)

        split_workers[task_name] = split_names

    spec["tasks"] = new_tasks
    if "minAvailable" in spec:
        spec["minAvailable"] = sum(int(task.get("replicas", 1)) for task in new_tasks)
    update_pytorch_workers(job, split_workers)

    extra = sorted(set(node_map) - expected_pods)
    if extra and not allow_extra_node_map:
        raise ValueError(
            "node_map contains pods not represented by cloned tasks: "
            + ",".join(extra)
            + "; pass --allow-extra-node-map to ignore them"
        )


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def dump_yaml(value: Any, indent: int = 0) -> list[str]:
    pad = " " * indent
    lines: list[str] = []

    if isinstance(value, dict):
        if not value:
            return [pad + "{}"]
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, str) and "\n" in item:
                lines.append(f"{pad}{key_text}: |-")
                lines.extend((" " * (indent + 2)) + part for part in item.splitlines())
            elif isinstance(item, (dict, list)):
                if item:
                    lines.append(f"{pad}{key_text}:")
                    lines.extend(dump_yaml(item, indent + 2))
                else:
                    lines.append(f"{pad}{key_text}: {'{}' if isinstance(item, dict) else '[]'}")
            else:
                lines.append(f"{pad}{key_text}: {yaml_scalar(item)}")
        return lines

    if isinstance(value, list):
        if not value:
            return [pad + "[]"]
        for item in value:
            if isinstance(item, str) and "\n" in item:
                lines.append(f"{pad}- |-")
                lines.extend((" " * (indent + 2)) + part for part in item.splitlines())
            elif isinstance(item, (dict, list)):
                if item:
                    nested = dump_yaml(item, indent + 2)
                    first = nested[0]
                    lines.append(f"{pad}- {first[indent + 2:]}")
                    lines.extend(nested[1:])
                else:
                    lines.append(f"{pad}- {'{}' if isinstance(item, dict) else '[]'}")
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
        return lines

    return [pad + yaml_scalar(value)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a same-node clone YAML for a vcctl/Volcano job.")
    parser.add_argument("--job-json", required=True, type=Path, help="Original job JSON from vcctl job get -o json.")
    parser.add_argument("--node-map", required=True, type=Path, help="node_map.txt from print_vcctl_node_map.sh.")
    parser.add_argument("--output", required=True, type=Path, help="Output clone YAML path.")
    parser.add_argument("--clone-job-name", required=True, help="New job name for the clone YAML.")
    parser.add_argument(
        "--allow-extra-node-map",
        action="store_true",
        help="Ignore node_map entries that are not represented by cloned tasks.",
    )
    args = parser.parse_args()

    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    node_map = parse_node_map(args.node_map)
    clean_job_metadata(job, args.clone_job_name)
    bind_tasks_to_nodes(job, node_map, args.allow_extra_node_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(dump_yaml(job)) + "\n", encoding="utf-8")
    print(f"[same-node-clone] wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[same-node-clone] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
