#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import hashlib
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
PROJECT_DIR = TOOLS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pretrain_healthcheck.common import parse_size
from dynamic_compare import (
    build_retest_plan,
    candidate_performance_issues,
)


PASS_STATUSES = {"PASS", "DRY_RUN", "REUSED_PASS"}
FULL_MESSAGE_SIZES = "1K,2K,4K,8K,16K,32K,64K,128K,256K,512K,1M,2M,4M,8M,16M,32M,64M,128M,256M,512M,1G,2G"
PAIRWISE_MESSAGE_SIZES = os.environ.get("PAIRWISE_MESSAGE_SIZES", FULL_MESSAGE_SIZES)
FAST_MESSAGE_SIZES = "1M,128M,1G"
FULL_OPS = "all_reduce,reduce_scatter,all_gather,broadcast,all_to_all,all_to_allv"
FULL_PATTERNS = "uniform,skewed,hot_expert,random,empty_expert"


@dataclass
class Pod:
    pod_name: str
    namespace: str
    container_name: str
    task_spec: str
    node_name: str
    host_ip: str
    pod_ip: str
    raw: dict[str, Any]


@dataclass
class GroupTask:
    phase: str
    round_id: str
    group_id: str
    pods: list[Pod]
    parent_group_id: str = ""
    attempt: int = 0
    fault_env: dict[str, str] = field(default_factory=dict)
    performance_retest_plan: list[dict[str, Any]] = field(default_factory=list)


def task_matrix_env(task: GroupTask) -> dict[str, str]:
    if task.performance_retest_plan:
        return {}
    if task.phase == "pairwise":
        sizes, iters = PAIRWISE_MESSAGE_SIZES, "1"
    elif task.phase == "final_all":
        sizes, iters = FULL_MESSAGE_SIZES, "3"
    else:
        sizes, iters = FAST_MESSAGE_SIZES, "1"
    return {
        "COLLECTIVE_BANDWIDTH_MESSAGE_SIZES": sizes,
        "COLLECTIVE_BANDWIDTH_OPS": FULL_OPS,
        "COLLECTIVE_BANDWIDTH_MOE_PATTERNS": FULL_PATTERNS,
        "COLLECTIVE_BANDWIDTH_WARMUP": "1",
        "COLLECTIVE_BANDWIDTH_ITERS": iters,
    }


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def parse_json_stream(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    objs: list[dict[str, Any]] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            objs.extend(item for item in obj["items"] if isinstance(item, dict))
        elif isinstance(obj, dict):
            objs.append(obj)
        else:
            raise ValueError(f"unsupported JSON value: {type(obj).__name__}")
        idx = next_idx
    return objs


def get_nested(obj: dict[str, Any], path: list[str], default: Any = "") -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def choose_container(pod: dict[str, Any]) -> dict[str, Any]:
    containers = get_nested(pod, ["spec", "containers"], [])
    if not isinstance(containers, list) or not containers:
        return {}
    task_spec = str(get_nested(pod, ["metadata", "labels", "volcano.sh/task-spec"], ""))
    for container in containers:
        if container.get("name") == task_spec:
            return container
    return containers[0]


def pod_sort_key(pod: Pod) -> tuple[int, int, str]:
    if pod.task_spec == "master":
        group = 0
    elif pod.task_spec == "worker":
        group = 1
    else:
        group = 2
    suffix = pod.pod_name.rsplit("-", 1)[-1]
    index = int(suffix) if suffix.isdigit() else 0
    return group, index, pod.pod_name


def pod_from_raw(raw: dict[str, Any]) -> Pod | None:
    metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}
    status = raw.get("status", {}) if isinstance(raw.get("status"), dict) else {}
    spec = raw.get("spec", {}) if isinstance(raw.get("spec"), dict) else {}
    labels = metadata.get("labels", {}) if isinstance(metadata.get("labels"), dict) else {}
    container = choose_container(raw)
    pod_name = str(metadata.get("name", ""))
    node_name = str(spec.get("nodeName", ""))
    if not pod_name or not node_name:
        return None
    return Pod(
        pod_name=pod_name,
        namespace=str(metadata.get("namespace", "default")),
        container_name=str(container.get("name", "")),
        task_spec=str(labels.get("volcano.sh/task-spec", "")),
        node_name=node_name,
        host_ip=str(status.get("hostIP", "")),
        pod_ip=str(status.get("podIP", "")),
        raw=raw,
    )


def load_pods(args: argparse.Namespace) -> tuple[str, list[Pod]]:
    if args.pod_json_file:
        raw = Path(args.pod_json_file).read_text(encoding="utf-8")
        pods = [pod for pod in (pod_from_raw(obj) for obj in parse_json_stream(raw)) if pod is not None]
        if not args.preserve_pod_json_order:
            pods.sort(key=pod_sort_key)
        return raw, pods
    cmd = [args.vcctl_bin, "pod", "get", "--job", args.job_name, "-n", args.namespace, "-o", "json"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"vcctl pod get failed rc={proc.returncode}: {proc.stderr.strip()}")
    pods = [pod for pod in (pod_from_raw(obj) for obj in parse_json_stream(proc.stdout)) if pod is not None]
    pods.sort(key=pod_sort_key)
    return proc.stdout, pods


def init_db(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=120)
    con.execute("pragma journal_mode=WAL")
    con.executescript(
        """
        create table if not exists pods (
          pod_name text primary key,
          container_name text,
          node_name text,
          host_ip text,
          pod_ip text,
          task_spec text
        );
        create table if not exists nodes (
          node_name text primary key,
          pod_name text,
          host_ip text,
          status text,
          suspect_score integer default 0,
          fail_reason text default '',
          last_phase text default ''
        );
        create table if not exists groups (
          group_id text primary key,
          phase text,
          round_id text,
          group_size integer,
          nodes_json text,
          status text,
          parent_group_id text default '',
          attempt integer default 0
        );
        create table if not exists group_results (
          id integer primary key autoincrement,
          group_id text,
          status text,
          error_type text,
          metrics_json text,
          elapsed_seconds real,
          local_workdirs_json text,
          created_at text
        );
        create table if not exists retest_tasks (
          task_id text primary key,
          source_group_id text,
          phase text,
          nodes_json text,
          reason text,
          status text
        );
        create table if not exists events (
          id integer primary key autoincrement,
          timestamp text,
          event_type text,
          message text,
          payload_json text
        );
        create table if not exists node_localization (
          node_name text,
          phase text,
          failed_group_count integer default 0,
          passed_group_count integer default 0,
          distinct_failed_partners integer default 0,
          distinct_passed_partners integer default 0,
          retest_failed_count integer default 0,
          retest_passed_count integer default 0,
          failed_partners_json text default '[]',
          passed_partners_json text default '[]',
          classification text,
          reason text default '',
          primary key(node_name, phase)
        );
        create table if not exists performance_candidates (
          phase text,
          round_id text,
          group_id text,
          case_id text,
          status text,
          details_json text default '{}',
          updated_at text,
          primary key(phase, round_id, group_id, case_id)
        );
        """
    )
    return con


def record_event(con: sqlite3.Connection, event_type: str, message: str, payload: dict[str, Any] | None = None) -> None:
    con.execute(
        "insert into events(timestamp,event_type,message,payload_json) values(?,?,?,?)",
        (iso_now(), event_type, message, json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)),
    )
    con.commit()


def upsert_pods(con: sqlite3.Connection, pods: list[Pod]) -> None:
    for pod in pods:
        con.execute(
            """
            insert into pods(pod_name,container_name,node_name,host_ip,pod_ip,task_spec)
            values(?,?,?,?,?,?)
            on conflict(pod_name) do update set
              container_name=excluded.container_name,
              node_name=excluded.node_name,
              host_ip=excluded.host_ip,
              pod_ip=excluded.pod_ip,
              task_spec=excluded.task_spec
            """,
            (pod.pod_name, pod.container_name, pod.node_name, pod.host_ip, pod.pod_ip, pod.task_spec),
        )
        con.execute(
            """
            insert into nodes(node_name,pod_name,host_ip,status,suspect_score,last_phase)
            values(?,?,?,'UNKNOWN',0,'')
            on conflict(node_name) do update set
              pod_name=excluded.pod_name,
              host_ip=excluded.host_ip
            """,
            (pod.node_name, pod.pod_name, pod.host_ip),
        )
    con.commit()


def node_status(con: sqlite3.Connection, node_name: str) -> str:
    row = con.execute("select status from nodes where node_name=?", (node_name,)).fetchone()
    return str(row[0]) if row else "UNKNOWN"


def set_node_status(con: sqlite3.Connection, node_name: str, status: str, phase: str, reason: str = "") -> None:
    con.execute(
        """
        update nodes
        set status=?,
            last_phase=?,
            fail_reason=case when ? != '' then ? else fail_reason end
        where node_name=?
        """,
        (status, phase, reason, reason, node_name),
    )


def add_suspect(con: sqlite3.Connection, node_name: str, phase: str, reason: str) -> None:
    con.execute(
        """
        update nodes
        set status=case when status='FAIL' then status else 'SUSPECT' end,
            suspect_score=suspect_score+1,
            last_phase=?,
            fail_reason=case when fail_reason='' then ? else fail_reason end
        where node_name=?
        """,
        (phase, reason, node_name),
    )


def phase_scale(phase: str) -> int:
    if phase == "pairwise":
        return 2
    if phase == "ep8":
        return 8
    if phase == "final_all":
        return 0
    if phase.startswith("scale"):
        return int(phase.removeprefix("scale"))
    raise ValueError(f"unsupported phase: {phase}")


def phase_order_key(phase: str) -> tuple[int, str]:
    if phase == "pairwise":
        return (10, phase)
    if phase == "ep8":
        return (20, phase)
    if phase.startswith("scale"):
        try:
            return (100 + int(phase.removeprefix("scale")), phase)
        except ValueError:
            return (500, phase)
    if phase == "final_all":
        return (10_000, phase)
    return (500, phase)


def auto_phases(node_count: int, target_scale: int) -> list[str]:
    if node_count < 2:
        raise ValueError("at least 2 nodes are required for multi-node checks")
    max_scale = target_scale if target_scale > 0 else node_count
    phases = ["pairwise"]
    if node_count >= 8 and max_scale >= 8:
        phases.append("ep8")
    if node_count >= 64 and max_scale >= 64:
        phases.append("scale64")
    if node_count >= 128 and max_scale >= 128:
        phases.append("scale128")
    if node_count >= 256 and max_scale >= 256:
        phases.append("scale256")
    return phases


def parse_phases(value: str, node_count: int, target_scale: int) -> list[str]:
    if value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return auto_phases(node_count, target_scale)


def chunks(items: list[Pod], size: int) -> list[list[Pod]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size) if len(items[idx : idx + size]) == size]


def make_pairwise_rounds(pods: list[Pod], seed: int) -> list[GroupTask]:
    shuffled = pods[:]
    random.Random(seed).shuffle(shuffled)
    tasks: list[GroupTask] = []
    seen: set[tuple[str, str]] = set()
    for idx, pair in enumerate(chunks(shuffled, 2)):
        key = tuple(sorted(pod.node_name for pod in pair))
        seen.add(key)
        tasks.append(GroupTask("pairwise", "pairwise_r1", f"pairwise_r1_pair_{idx:04d}", pair))
    if len(pods) >= 4:
        shifted = shuffled[1:] + shuffled[:1]
        round2: list[list[Pod]] = []
        used: set[str] = set()
        for pod in shifted:
            if pod.node_name in used:
                continue
            partner = next(
                (
                    other
                    for other in shifted
                    if other.node_name not in used
                    and other.node_name != pod.node_name
                    and tuple(sorted([pod.node_name, other.node_name])) not in seen
                ),
                None,
            )
            if partner is None:
                continue
            used.add(pod.node_name)
            used.add(partner.node_name)
            round2.append([pod, partner])
        for idx, pair in enumerate(round2):
            tasks.append(GroupTask("pairwise", "pairwise_r2", f"pairwise_r2_pair_{idx:04d}", pair))
    return tasks


def make_phase_groups(phase: str, pods: list[Pod], seed: int) -> list[GroupTask]:
    if phase == "pairwise":
        return make_pairwise_rounds(pods, seed)
    if phase == "final_all":
        return [GroupTask("final_all", "final_all_r1", "final_all_group_0000", pods[:])]
    size = phase_scale(phase)
    shuffled = pods[:]
    random.Random(f"{seed}:{phase}").shuffle(shuffled)
    round_id = f"{phase}_r1"
    return [
        GroupTask(phase, round_id, f"{round_id}_group_{idx:04d}", group)
        for idx, group in enumerate(chunks(shuffled, size))
    ]


def ensure_env(container: dict[str, Any], name: str, value: str) -> None:
    env = container.setdefault("env", [])
    if not isinstance(env, list):
        env = []
        container["env"] = env
    for item in env:
        if item.get("name") == name:
            item["value"] = value
            item.pop("valueFrom", None)
            return
    env.append({"name": name, "value": value})


def group_pod_json(task: GroupTask) -> dict[str, Any]:
    master_addr = task.pods[0].pod_ip or task.pods[0].pod_name
    items: list[dict[str, Any]] = []
    for rank, pod in enumerate(task.pods):
        raw = copy.deepcopy(pod.raw)
        container = choose_container(raw)
        ensure_env(container, "RANK", str(rank))
        ensure_env(container, "WORLD_SIZE", str(len(task.pods)))
        ensure_env(container, "MASTER_ADDR", master_addr)
        ensure_env(container, "MASTER_PORT", "29500")
        items.append(raw)
    return {"items": items}


def write_group_json(task: GroupTask, args: argparse.Namespace) -> Path:
    path = Path(args.batch_tmp_dir) / "group_pods" / f"{task.group_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(group_pod_json(task), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def table_group(con: sqlite3.Connection, task: GroupTask, status: str) -> None:
    con.execute(
        """
        insert into groups(group_id,phase,round_id,group_size,nodes_json,status,parent_group_id,attempt)
        values(?,?,?,?,?,?,?,?)
        on conflict(group_id) do update set status=excluded.status
        """,
        (
            task.group_id,
            task.phase,
            task.round_id,
            len(task.pods),
            json.dumps([pod.node_name for pod in task.pods], ensure_ascii=False),
            status,
            task.parent_group_id,
            task.attempt,
        ),
    )
    con.commit()


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def copy_failed_summary(task: GroupTask, output_dir: Path, batch_dir: Path, reason: str) -> None:
    failed_dir = batch_dir / "failed_groups" / task.group_id
    failed_dir.mkdir(parents=True, exist_ok=True)
    src = output_dir / "summary.md"
    if src.exists():
        shutil.copyfile(src, failed_dir / "summary.md")
    else:
        (failed_dir / "summary.md").write_text(
            f"# Failed Group\n\n- group_id: `{task.group_id}`\n- reason: `{reason}`\n",
            encoding="utf-8",
        )


def link_failed_group_output(task: GroupTask, output_dir: Path, batch_dir: Path, args: argparse.Namespace) -> str:
    if args.failed_group_output_mode != "local-link":
        return ""
    if args.dry_run == "1" or args.dry_run.lower() == "true":
        return ""
    if not output_dir.exists():
        return ""
    links_dir = batch_dir / "failed_group_outputs"
    links_dir.mkdir(parents=True, exist_ok=True)
    link_path = links_dir / task.group_id
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path, ignore_errors=True)
        else:
            try:
                link_path.unlink()
            except OSError:
                pass
    try:
        link_path.symlink_to(output_dir)
        return str(link_path)
    except OSError as exc:
        (links_dir / f"{task.group_id}.path").write_text(
            f"{output_dir}\nsymlink_error={type(exc).__name__}:{exc}\n",
            encoding="utf-8",
        )
        return ""


def cleanup_pass_output(output_dir: Path, args: argparse.Namespace) -> None:
    if args.keep_group_outputs == "1" or args.keep_group_outputs.lower() == "true":
        return
    if args.dry_run == "1" or args.dry_run.lower() == "true":
        return
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
        parent = output_dir.parent
        while parent.name != args.batch_run_id and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def status_from_summary(summary: dict[str, Any], returncode: int) -> tuple[str, str]:
    if returncode != 0:
        return "TIMEOUT" if returncode == 124 else "FAIL", f"returncode={returncode}"
    status = str(summary.get("overall_status", "FAIL"))
    if status in PASS_STATUSES:
        return status, ""
    return "FAIL", f"overall_status={status}"


def group_result_root(args: argparse.Namespace) -> str:
    if args.failed_group_output_mode == "local-link" and args.dry_run not in {"1", "true"}:
        return args.group_output_root
    return args.result_root


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def execution_signature(args: argparse.Namespace) -> str:
    script_path = Path(args.healthcheck_script).resolve()
    try:
        script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
    except OSError:
        script_sha256 = "unavailable"
    exact_env_names = {
        "GPUS_PER_NODE",
        "DIST_BACKEND",
        "DEVICE_VENDOR",
        "COMM_RUNTIME",
        "DTYPE",
        "MESSAGE_SIZES",
        "MOE_PATTERNS",
        "WARMUP",
        "ITERS",
        "MULTI_NODE_CMD",
        "PROJECT_REMOTE_DIR",
    }
    env_prefixes = (
        "ASCEND_",
        "BANDWIDTH_",
        "COLLECTIVE_BANDWIDTH_",
        "COMM_PATH_",
        "DYNAMIC_COMPARE_",
        "HCCL_",
        "MCCL_",
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in exact_env_names or key.startswith(env_prefixes)
    }
    payload = {
        "healthcheck_script": str(script_path),
        "healthcheck_script_sha256": script_sha256,
        "pod_project_dir": str(args.pod_project_dir or ""),
        "dynamic_compare": str(args.dynamic_compare),
        "dynamic_compare_busbw_ratio_threshold": args.dynamic_compare_busbw_ratio_threshold,
        "dynamic_compare_latency_ratio_threshold": args.dynamic_compare_latency_ratio_threshold,
        "dynamic_compare_small_max_size": str(args.dynamic_compare_small_max_size),
        "dynamic_compare_large_min_size": str(args.dynamic_compare_large_min_size),
        "dynamic_compare_small_latency_warn": bool(args.dynamic_compare_small_latency_warn),
        "dynamic_compare_min_cohort": args.dynamic_compare_min_cohort,
        "dynamic_compare_auto_retest": bool(args.dynamic_compare_auto_retest),
        "environment": environment,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase_matches_fault(task: GroupTask, args: argparse.Namespace) -> bool:
    phase = str(args.batch_fault_phase or "all").strip()
    return phase in {"", "all"} or phase == task.phase or phase == task.round_id


def _csv_values(*values: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in str(value or "").split(","):
            item = item.strip()
            if item and item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _fault_target_nodes(args: argparse.Namespace) -> list[str]:
    return _csv_values(args.batch_fault_node, args.batch_fault_nodes)


def _fault_target_pods(args: argparse.Namespace) -> list[str]:
    return _csv_values(args.batch_fault_pod, args.batch_fault_pods)


def _task_matches_fault_target(task: GroupTask, args: argparse.Namespace) -> bool:
    target_nodes = set(_fault_target_nodes(args))
    target_pods = set(_fault_target_pods(args))
    if not target_nodes and not target_pods:
        return True
    return any(pod.node_name in target_nodes or pod.pod_name in target_pods for pod in task.pods)


def _target_fault_env(prefix: str, args: argparse.Namespace) -> dict[str, str]:
    target_pods = _fault_target_pods(args)
    target_nodes = _fault_target_nodes(args)
    env: dict[str, str] = {}
    if target_pods:
        env[f"{prefix}_PODS"] = ",".join(target_pods)
    if target_nodes:
        env[f"{prefix}_NODES"] = ",".join(target_nodes)
    return env or {f"{prefix}_RANK": "0"}


def build_batch_fault_env(task: GroupTask, args: argparse.Namespace) -> dict[str, str]:
    fault_type = str(args.batch_fault_type or "").strip().lower()
    if not fault_type:
        return {}
    if not _phase_matches_fault(task, args) or not _task_matches_fault_target(task, args):
        return {}
    max_hits = int(args.batch_fault_max_hits)
    if max_hits > 0 and args._batch_fault_hit_count >= max_hits:
        return {}

    env: dict[str, str] = {}
    if fault_type == "nan":
        env.update(_target_fault_env("FAULT_NAN", args))
    elif fault_type == "corrupt":
        env.update(_target_fault_env("FAULT_CORRUPT", args))
    elif fault_type == "sleep":
        env.update(_target_fault_env("FAULT_SLEEP", args))
        env["FAULT_SLEEP_SECONDS"] = str(args.batch_fault_sleep_seconds)
    elif fault_type == "join_timeout":
        env.update(_target_fault_env("FAULT_JOIN_TIMEOUT", args))
        env["FAULT_JOIN_TIMEOUT_SECONDS"] = str(args.batch_fault_sleep_seconds)
    elif fault_type == "net_slow":
        env.update(_target_fault_env("FAULT_NET_SLOW", args))
        env["FAULT_NET_SLOW_SECONDS"] = f"{float(args.batch_fault_delay_ms) / 1000.0:.6f}"
    elif fault_type == "rank_exit":
        env.update(_target_fault_env("FAULT_RANK_EXIT", args))
        env["FAULT_RANK_EXIT_CODE"] = "17"
    elif fault_type == "backend":
        env["FAULT_BACKEND"] = "1"
    elif fault_type == "comm_env_bad":
        env.update(_target_fault_env("FAULT_COMM_ENV_BAD", args))
        env["COMM_PATH_DEBUG"] = "1"
    elif fault_type == "eth_fallback":
        env.update(_target_fault_env("FAULT_ETH_FALLBACK", args))
        env["COMM_PATH_DEBUG"] = "1"
    else:
        raise ValueError(f"unsupported BATCH_FAULT_TYPE: {fault_type}")

    if _truthy(args.comm_path_debug):
        env["COMM_PATH_DEBUG"] = "1"
    args._batch_fault_hit_count += 1
    return env


def prepare_task_fault_env(task: GroupTask, args: argparse.Namespace, con: sqlite3.Connection) -> None:
    if task.fault_env:
        return
    env = build_batch_fault_env(task, args)
    if not env:
        return
    task.fault_env = env
    record_event(
        con,
        "batch_fault_applied",
        task.group_id,
        {
            "fault_type": args.batch_fault_type,
            "fault_phase": args.batch_fault_phase,
            "fault_nodes": _fault_target_nodes(args),
            "fault_pods": _fault_target_pods(args),
            "env_keys": sorted(env.keys()),
            "nodes": [pod.node_name for pod in task.pods],
        },
    )


def comm_path_rows_from_facts(task: GroupTask, status: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact in facts:
        summary = fact.get("comm_path_summary")
        if not isinstance(summary, dict):
            continue
        for row in summary.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            enriched = dict(row)
            enriched["group_id"] = task.group_id
            enriched["phase"] = task.phase
            enriched["round_id"] = task.round_id
            enriched["group_status"] = status
            rows.append(enriched)
    return rows


def task_timeout_seconds(task: GroupTask, args: argparse.Namespace) -> int:
    if task.phase == "final_all":
        return args.final_all_timeout_seconds
    if task.performance_retest_plan:
        return min(args.group_timeout_seconds, args.dynamic_compare_retest_time_budget_seconds)
    return args.group_timeout_seconds


def maybe_emit_runtime_warning(args: argparse.Namespace, con: sqlite3.Connection) -> None:
    target = args.batch_runtime_warn_seconds
    if target <= 0 or args._runtime_warning_emitted:
        return
    elapsed = time.monotonic() - args.batch_started_monotonic
    if elapsed < target:
        return
    with args._runtime_warning_lock:
        if args._runtime_warning_emitted:
            return
        args._runtime_warning_emitted = True
        payload = {
            "elapsed_seconds": round(elapsed, 3),
            "target_seconds": target,
            "execution_continues": True,
        }
        print(
            "[batch-healthcheck] WARNING runtime_target_exceeded "
            f"elapsed={int(elapsed)}s target={target}s execution_continues=1",
            flush=True,
        )
        record_event(con, "runtime_target_exceeded", args.batch_run_id, payload)


def run_group(task: GroupTask, args: argparse.Namespace, con: sqlite3.Connection, batch_dir: Path) -> str:
    table_group(con, task, "RUNNING")
    group_json = write_group_json(task, args)
    stage_suffix = "multi_node_performance_retest" if task.performance_retest_plan else "multi_node_dynamic_suite"
    run_stage = f"{task.phase}/{task.group_id}/{stage_suffix}"
    run_result_root = group_result_root(args)
    output_dir = Path(run_result_root) / args.batch_run_id / run_stage
    cmd = [
        "bash",
        str(Path(args.healthcheck_script)),
    ]
    env = os.environ.copy()
    env.update(
        {
            "JOB_NAME": args.job_name,
            "NAMESPACE": args.namespace,
            "RUN_ID": args.batch_run_id,
            "RUN_STAGE": run_stage,
            "POD_JSON_FILE": str(group_json),
            "MODE": "multi-node",
            "PROFILE": "dynamic-suite",
            "DRY_RUN": args.dry_run,
            "PRE_CLEAN": args.pre_clean,
            "DYNAMIC_COMPARE": args.dynamic_compare,
            "DYNAMIC_COMPARE_BUSBW_RATIO_THRESHOLD": str(args.dynamic_compare_busbw_ratio_threshold),
            "DYNAMIC_COMPARE_LATENCY_RATIO_THRESHOLD": str(args.dynamic_compare_latency_ratio_threshold),
            "DYNAMIC_COMPARE_SMALL_MAX_SIZE": str(args.dynamic_compare_small_max_size),
            "DYNAMIC_COMPARE_LARGE_MIN_SIZE": str(args.dynamic_compare_large_min_size),
            "DYNAMIC_COMPARE_SMALL_LATENCY_WARN": "1" if args.dynamic_compare_small_latency_warn else "0",
            "DYNAMIC_COMPARE_MIN_COHORT": str(args.dynamic_compare_min_cohort),
            "EXEC_TIMEOUT_SECONDS": str(task_timeout_seconds(task, args)),
            "RESULT_ROOT": run_result_root,
        }
    )
    env.update(task_matrix_env(task))
    if task.performance_retest_plan:
        plan_json = json.dumps(task.performance_retest_plan, ensure_ascii=False, separators=(",", ":"))
        env["DYNAMIC_RETEST_ONLY_PLAN_B64"] = base64.b64encode(plan_json.encode("utf-8")).decode("ascii")
        env["DYNAMIC_COMPARE_AUTO_RETEST"] = "0"
    if args.pod_project_dir:
        env["PROJECT_REMOTE_DIR"] = args.pod_project_dir
    if _truthy(args.comm_path_debug):
        env["COMM_PATH_DEBUG"] = "1"
    if task.fault_env:
        env.update(task.fault_env)

    started = time.monotonic()
    print(
        f"[batch-healthcheck] group_start phase={task.round_id} "
        f"group={task.group_id} size={len(task.pods)} "
        f"nodes={','.join(pod.node_name for pod in task.pods)}",
        flush=True,
    )
    record_event(con, "group_start", task.group_id, {"nodes": [pod.node_name for pod in task.pods]})
    proc = subprocess.Popen(cmd, cwd=args.project_dir, env=env)
    next_progress = time.monotonic() + max(1, args.progress_interval_seconds)
    timeout_seconds = task_timeout_seconds(task, args)
    driver_deadline = started + max(1, timeout_seconds) + 120
    batch_timeout = False
    while proc.poll() is None:
        now = time.monotonic()
        if now >= driver_deadline:
            batch_timeout = True
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            break
        if now >= next_progress:
            elapsed = int(now - started)
            print(
                f"[batch-healthcheck] progress phase={task.round_id} "
                f"group={task.group_id} elapsed={elapsed}s",
                flush=True,
            )
            next_progress = now + max(1, args.progress_interval_seconds)
        maybe_emit_runtime_warning(args, con)
        time.sleep(0.5)
    returncode = proc.returncode
    elapsed = round(time.monotonic() - started, 3)
    summary = read_summary(output_dir / "summary.json")
    if batch_timeout:
        status, reason = "TIMEOUT", f"BATCH_GROUP_TIMEOUT>{timeout_seconds + 120}s"
    else:
        status, reason = status_from_summary(summary, returncode)
    facts = read_jsonl(output_dir / "dynamic_facts.jsonl")
    local_workdirs = {
        str(row.get("pod", {}).get("node_name", "")): str(row.get("local_workdir", ""))
        for row in facts
        if isinstance(row.get("pod"), dict)
    }
    comm_path_rows = comm_path_rows_from_facts(task, status, facts)
    metrics = {
        "summary": summary.get("dynamic_compare") or {},
        "pod_count": summary.get("pod_count"),
        "result_count": summary.get("result_count"),
        "driver_group_output_dir": str(output_dir),
        "case_metrics": read_jsonl(output_dir / "dynamic_case_metrics.jsonl"),
        "performance_retest": bool(task.performance_retest_plan),
        "execution_signature": execution_signature(args),
    }
    if task.fault_env:
        metrics["batch_fault"] = {
            "type": args.batch_fault_type,
            "phase": args.batch_fault_phase,
            "nodes": _fault_target_nodes(args),
            "pods": _fault_target_pods(args),
            "env_keys": sorted(task.fault_env.keys()),
        }
    if comm_path_rows:
        metrics["comm_path_summary"] = comm_path_rows
    con.execute("update groups set status=? where group_id=?", (status, task.group_id))
    con.execute(
        """
        insert into group_results(group_id,status,error_type,metrics_json,elapsed_seconds,local_workdirs_json,created_at)
        values(?,?,?,?,?,?,?)
        """,
        (
            task.group_id,
            status,
            reason,
            json.dumps(metrics, ensure_ascii=False, sort_keys=True),
            elapsed,
            json.dumps(local_workdirs, ensure_ascii=False, sort_keys=True),
            iso_now(),
        ),
    )
    con.commit()
    if status in PASS_STATUSES:
        for pod in task.pods:
            set_node_status(con, pod.node_name, "PASS", task.phase)
        con.commit()
        cleanup_pass_output(output_dir, args)
    else:
        for pod in task.pods:
            add_suspect(con, pod.node_name, task.phase, f"{task.group_id}:{reason}")
        con.commit()
        copy_failed_summary(task, output_dir, batch_dir, reason)
        failed_output_link = link_failed_group_output(task, output_dir, batch_dir, args)
        if failed_output_link:
            metrics["shared_failed_output_link"] = failed_output_link
            con.execute(
                """
                update group_results
                set metrics_json=?
                where id=(select max(id) from group_results where group_id=?)
                """,
                (json.dumps(metrics, ensure_ascii=False, sort_keys=True), task.group_id),
            )
            con.commit()
    if args.keep_group_outputs not in {"1", "true"} and args.dry_run not in {"1", "true"}:
        try:
            group_json.unlink()
        except OSError:
            pass
    print(
        f"[batch-healthcheck] group_done phase={task.round_id} "
        f"group={task.group_id} status={status} elapsed={elapsed}s reason={reason}",
        flush=True,
    )
    record_event(con, "group_done", task.group_id, {"status": status, "reason": reason, "elapsed": elapsed})
    return status


def known_good_pods(con: sqlite3.Connection, pods: list[Pod], exclude: set[str]) -> list[Pod]:
    passed_nodes: set[str] = set()
    for nodes_json, status in con.execute("select nodes_json,status from groups"):
        if str(status) not in PASS_STATUSES:
            continue
        try:
            passed_nodes.update(str(node) for node in json.loads(nodes_json or "[]"))
        except (json.JSONDecodeError, TypeError):
            continue
    return [pod for pod in pods if pod.node_name not in exclude and pod.node_name in passed_nodes]


def retest_pairwise(task: GroupTask, pods: list[Pod], con: sqlite3.Connection) -> list[GroupTask]:
    goods = known_good_pods(con, pods, {pod.node_name for pod in task.pods})
    retests: list[GroupTask] = []
    for idx, pod in enumerate(task.pods):
        if idx >= len(goods):
            break
        pair = [pod, goods[idx]]
        retests.append(
            GroupTask(
                phase="pairwise",
                round_id=f"{task.round_id}_retest",
                group_id=f"{task.group_id}_retest_{idx:02d}",
                pods=pair,
                parent_group_id=task.group_id,
                attempt=task.attempt + 1,
            )
        )
    return retests


def split_retests(task: GroupTask) -> list[GroupTask]:
    if len(task.pods) <= 2:
        return []
    mid = len(task.pods) // 2
    parts = [task.pods[:mid], task.pods[mid:]]
    return [
        GroupTask(
            phase=task.phase,
            round_id=f"{task.round_id}_split",
            group_id=f"{task.group_id}_split_{idx}",
            pods=part,
            parent_group_id=task.group_id,
            attempt=task.attempt + 1,
        )
        for idx, part in enumerate(parts)
        if len(part) >= 2
    ]


def run_retests(
    failed: list[GroupTask],
    all_pods: list[Pod],
    args: argparse.Namespace,
    con: sqlite3.Connection,
    batch_dir: Path,
) -> bool:
    queue: list[GroupTask] = []
    for task in failed:
        queue.extend(retest_pairwise(task, all_pods, con) if len(task.pods) == 2 else split_retests(task))
    if not queue:
        return False
    any_pass = False
    while queue:
        task = queue.pop(0)
        con.execute(
            "insert or replace into retest_tasks(task_id,source_group_id,phase,nodes_json,reason,status) values(?,?,?,?,?,?)",
            (
                task.group_id,
                task.parent_group_id,
                task.phase,
                json.dumps([pod.node_name for pod in task.pods], ensure_ascii=False),
                "auto_retest",
                "RUNNING",
            ),
        )
        con.commit()
        prepare_task_fault_env(task, args, con)
        status = run_group(task, args, con, batch_dir)
        con.execute("update retest_tasks set status=? where task_id=?", (status, task.group_id))
        con.commit()
        if status in PASS_STATUSES:
            any_pass = True
            continue
        if len(task.pods) > 2:
            queue.extend(split_retests(task))
    return any_pass


def finalize_pairwise_localization(con: sqlite3.Connection, phase: str) -> list[str]:
    rows = con.execute(
        "select group_id,round_id,nodes_json,status from groups where phase=? order by group_id",
        (phase,),
    ).fetchall()
    evidence: dict[str, dict[str, Any]] = {}

    def node_evidence(node_name: str) -> dict[str, Any]:
        return evidence.setdefault(
            node_name,
            {
                "failed_groups": [],
                "passed_groups": [],
                "failed_partners": set(),
                "passed_partners": set(),
                "retest_failed_count": 0,
                "retest_passed_count": 0,
            },
        )

    for group_id, round_id, nodes_json, status in rows:
        try:
            nodes = [str(node) for node in json.loads(nodes_json or "[]")]
        except (json.JSONDecodeError, TypeError):
            nodes = []
        passed = str(status) in PASS_STATUSES
        is_retest = "retest" in str(round_id)
        for node_name in nodes:
            item = node_evidence(node_name)
            partners = {partner for partner in nodes if partner != node_name}
            if passed:
                item["passed_groups"].append(str(group_id))
                item["passed_partners"].update(partners)
                if is_retest:
                    item["retest_passed_count"] += 1
            else:
                item["failed_groups"].append(str(group_id))
                item["failed_partners"].update(partners)
                if is_retest:
                    item["retest_failed_count"] += 1

    primary_suspects = {
        node_name
        for node_name, item in evidence.items()
        if len(item["failed_groups"]) >= 2
        and len(item["failed_partners"]) >= 2
        and item["retest_failed_count"] >= 1
        and not item["passed_groups"]
        and item["retest_passed_count"] == 0
    }

    for node_name, item in evidence.items():
        failed_groups = list(item["failed_groups"])
        passed_groups = list(item["passed_groups"])
        failed_partners = sorted(item["failed_partners"])
        passed_partners = sorted(item["passed_partners"])
        if not failed_groups:
            classification = "PASS"
            reason = "all_observed_pairwise_groups_passed"
        elif item["retest_passed_count"] > 0 or passed_groups:
            classification = "PASS"
            reason = "independent_pairwise_or_retest_passed"
        elif node_name in primary_suspects:
            classification = "SUSPECT"
            reason = (
                f"persistent_pairwise_failure groups={len(failed_groups)} "
                f"partners={len(failed_partners)} retest_fail={item['retest_failed_count']}"
            )
        elif set(failed_partners).issubset(primary_suspects):
            classification = "PASS"
            reason = "failures_explained_by_primary_suspect_partner"
        else:
            classification = "SUSPECT"
            reason = "ambiguous_pairwise_failure_insufficient_independent_evidence"

        con.execute(
            """
            insert into node_localization(
              node_name,phase,failed_group_count,passed_group_count,
              distinct_failed_partners,distinct_passed_partners,
              retest_failed_count,retest_passed_count,
              failed_partners_json,passed_partners_json,classification,reason
            ) values(?,?,?,?,?,?,?,?,?,?,?,?)
            on conflict(node_name,phase) do update set
              failed_group_count=excluded.failed_group_count,
              passed_group_count=excluded.passed_group_count,
              distinct_failed_partners=excluded.distinct_failed_partners,
              distinct_passed_partners=excluded.distinct_passed_partners,
              retest_failed_count=excluded.retest_failed_count,
              retest_passed_count=excluded.retest_passed_count,
              failed_partners_json=excluded.failed_partners_json,
              passed_partners_json=excluded.passed_partners_json,
              classification=excluded.classification,
              reason=excluded.reason
            """,
            (
                node_name,
                phase,
                len(failed_groups),
                len(passed_groups),
                len(failed_partners),
                len(passed_partners),
                item["retest_failed_count"],
                item["retest_passed_count"],
                json.dumps(failed_partners, ensure_ascii=False),
                json.dumps(passed_partners, ensure_ascii=False),
                classification,
                reason,
            ),
        )
        if classification == "PASS":
            con.execute(
                "update nodes set status='PASS',suspect_score=0,fail_reason='',last_phase=? where node_name=?",
                (phase, node_name),
            )
        else:
            con.execute(
                """
                update nodes
                set status='SUSPECT',suspect_score=?,fail_reason=?,last_phase=?
                where node_name=?
                """,
                (len(failed_groups), reason, phase, node_name),
            )
    con.commit()
    suspects = sorted(
        node_name
        for node_name, item in evidence.items()
        if con.execute("select status from nodes where node_name=?", (node_name,)).fetchone()[0] == "SUSPECT"
    )
    record_event(
        con,
        "localization_done",
        phase,
        {"primary_suspects": sorted(primary_suspects), "suspect_nodes": suspects},
    )
    return suspects


def run_group_with_own_connection(
    task: GroupTask,
    args: argparse.Namespace,
    batch_dir: Path,
) -> tuple[GroupTask, str]:
    con = init_db(Path(args.db_path))
    try:
        return task, run_group(task, args, con, batch_dir)
    finally:
        con.close()


def phase_pass_nodes(con: sqlite3.Connection, pods: list[Pod]) -> list[Pod]:
    return [pod for pod in pods if node_status(con, pod.node_name) == "PASS"]


def phase_candidates(phase: str, con: sqlite3.Connection, pods: list[Pod]) -> tuple[list[Pod], bool]:
    candidates = phase_pass_nodes(con, pods)
    if phase == "pairwise":
        return [pod for pod in pods if node_status(con, pod.node_name) not in {"FAIL", "EXCLUDED"}], False
    if phase == "final_all" and not candidates:
        statuses = [node_status(con, pod.node_name) for pod in pods]
        if statuses and all(status == "UNKNOWN" for status in statuses):
            return list(pods), True
    return candidates, False


def record_performance_candidates(
    con: sqlite3.Connection,
    phase: str,
    round_id: str,
    candidates: list[dict[str, Any]],
    status: str = "DETECTED",
) -> None:
    for item in candidates:
        group_id = str(item.get("pod_name", ""))
        case_id = str(item.get("case_id", ""))
        if not group_id or not case_id:
            continue
        con.execute(
            """
            insert into performance_candidates(
              phase,round_id,group_id,case_id,status,details_json,updated_at
            ) values(?,?,?,?,?,?,?)
            on conflict(phase,round_id,group_id,case_id) do update set
              status=excluded.status,
              details_json=excluded.details_json,
              updated_at=excluded.updated_at
            """,
            (
                phase,
                round_id,
                group_id,
                case_id,
                status,
                json.dumps(item, ensure_ascii=False, sort_keys=True),
                iso_now(),
            ),
        )
    con.commit()


def update_performance_candidate_statuses(
    con: sqlite3.Connection,
    phase: str,
    round_id: str,
    statuses: dict[tuple[str, str], str],
) -> None:
    for (group_id, case_id), status in statuses.items():
        con.execute(
            """
            update performance_candidates
            set status=?,updated_at=?
            where phase=? and round_id=? and group_id=? and case_id=?
            """,
            (status, iso_now(), phase, round_id, group_id, case_id),
        )
    con.commit()


def _task_case_metrics(con: sqlite3.Connection, tasks: list[GroupTask]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        record = con.execute(
            "select metrics_json from group_results where group_id=? order by id desc limit 1",
            (task.group_id,),
        ).fetchone()
        if not record:
            continue
        try:
            metrics = json.loads(record[0] or "{}")
        except json.JSONDecodeError:
            continue
        for raw in metrics.get("case_metrics", []) or []:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item["pod_name"] = task.group_id
            item["node_name"] = task.group_id
            item["case_id"] = "/".join(
                str(item.get(key, ""))
                for key in ["stage", "op_type", "message_bytes", "payload_pattern", "collective_group_size"]
            )
            rows.append(item)
    return rows


def _run_performance_retest_round(
    tasks: list[GroupTask],
    plan: list[dict[str, Any]],
    args: argparse.Namespace,
    con: sqlite3.Connection,
    batch_dir: Path,
    batch_index: int = 0,
) -> tuple[list[GroupTask], list[dict[str, Any]]]:
    suffix = f"performance_retest_b{batch_index:03d}"
    retest_tasks = [
        GroupTask(
            phase=task.phase,
            round_id=f"{task.round_id}_{suffix}",
            group_id=f"{task.group_id}_{suffix}",
            pods=task.pods,
            parent_group_id=task.group_id,
            attempt=task.attempt + 1,
            performance_retest_plan=plan,
        )
        for task in tasks
    ]
    max_workers = len(retest_tasks) if args.phase_group_concurrency <= 0 else min(args.phase_group_concurrency, len(retest_tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_group_with_own_connection, task, args, batch_dir) for task in retest_tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return retest_tasks, _task_case_metrics(con, retest_tasks)


def select_performance_retest_batches(
    tasks: list[GroupTask],
    candidate_group_ids: set[str],
    args: argparse.Namespace,
) -> list[tuple[list[GroupTask], list[str]]]:
    if len(tasks) <= args.dynamic_compare_retest_max_groups:
        selected = tasks[:]
        controls = [task.group_id for task in selected if task.group_id not in candidate_group_ids]
        return [(selected, controls)]

    candidates = sorted(
        (task for task in tasks if task.group_id in candidate_group_ids),
        key=lambda task: task.group_id,
    )
    normal = [task for task in tasks if task.group_id not in candidate_group_ids]
    rng = random.Random(f"{args.group_seed}:{tasks[0].round_id}:performance-retest")
    control_count = min(8, len(normal), max(0, args.dynamic_compare_retest_max_groups - 1))
    controls = rng.sample(normal, control_count) if control_count else []
    candidate_capacity = max(1, args.dynamic_compare_retest_max_groups - len(controls))
    batches: list[tuple[list[GroupTask], list[str]]] = []
    for offset in range(0, len(candidates), candidate_capacity):
        candidate_batch = candidates[offset : offset + candidate_capacity]
        selected = sorted(candidate_batch + controls, key=lambda task: task.group_id)
        batches.append((selected, [task.group_id for task in controls]))
    return batches


def compare_round_performance(
    tasks: list[GroupTask],
    args: argparse.Namespace,
    con: sqlite3.Connection,
    batch_dir: Path,
) -> tuple[list[GroupTask], bool]:
    """Return confirmed slow groups and whether performance produced an advisory."""
    case_rows = _task_case_metrics(con, tasks)
    candidates, _cohorts = candidate_performance_issues(
        case_rows,
        latency_ratio_threshold=args.dynamic_compare_latency_ratio_threshold,
        busbw_ratio_threshold=args.dynamic_compare_busbw_ratio_threshold,
        min_cohort=args.dynamic_compare_min_cohort,
        small_max_bytes=args.dynamic_compare_small_max_bytes,
        large_min_bytes=args.dynamic_compare_large_min_bytes,
        small_latency_warn=args.dynamic_compare_small_latency_warn,
        small_latency_abs_delta_seconds=args.dynamic_compare_small_latency_abs_delta_seconds,
        small_latency_mad_multiplier=args.dynamic_compare_small_latency_mad_multiplier,
    )
    task_by_id = {task.group_id: task for task in tasks}
    small_warnings = [item for item in candidates if item.get("message_class") == "small"]
    large_candidates = [item for item in candidates if item.get("message_class") == "large"]
    if small_warnings:
        record_event(
            con,
            "performance_warning",
            tasks[0].round_id if tasks else "",
            {"warning_count": len(small_warnings), "warnings": small_warnings},
        )
    if not large_candidates:
        return [], False

    phase = tasks[0].phase
    round_id = tasks[0].round_id
    record_performance_candidates(con, phase, round_id, large_candidates)
    confirmed_ids: set[str] = set()
    candidate_group_ids = {str(item.get("pod_name", "")) for item in large_candidates}
    advisory = False
    systemic = len(tasks) > args.dynamic_compare_retest_max_groups and (
        len(candidate_group_ids) >= args.dynamic_compare_retest_max_groups
        or len(candidate_group_ids) / max(1, len(tasks)) > args.dynamic_compare_systemic_candidate_fraction
    )
    if systemic:
        advisory = True
        record_event(
            con,
            "systemic_performance_event",
            round_id,
            {
                "candidate_groups": sorted(candidate_group_ids),
                "candidate_group_count": len(candidate_group_ids),
                "phase_group_count": len(tasks),
                "candidate_fraction": len(candidate_group_ids) / max(1, len(tasks)),
                "threshold": args.dynamic_compare_systemic_candidate_fraction,
                "execution_continued": True,
                "automatic_retest": bool(args.dynamic_compare_auto_retest),
            },
        )

    if args.dynamic_compare_auto_retest:
        plan = build_retest_plan(large_candidates, case_rows, args.dynamic_compare_large_min_bytes)
        initial_ids = {(str(item.get("pod_name", "")), str(item.get("case_id", ""))) for item in large_candidates}
        repeated_ids: set[tuple[str, str]] = set()
        new_ids: set[tuple[str, str]] = set()
        retest_only_items: list[dict[str, Any]] = []
        all_control_group_ids: set[str] = set()
        batches = select_performance_retest_batches(tasks, candidate_group_ids, args)
        for batch_index, (selected_tasks, control_group_ids) in enumerate(batches, start=1):
            all_control_group_ids.update(control_group_ids)
            selected_candidate_ids = sorted(
                task.group_id for task in selected_tasks if task.group_id in candidate_group_ids
            )
            record_event(
                con,
                "performance_retest_start",
                round_id,
                {
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                    "phase_group_count": len(tasks),
                    "retest_group_count": len(selected_tasks),
                    "candidate_groups": selected_candidate_ids,
                    "control_groups": control_group_ids,
                    "cases": plan,
                },
            )
            retest_tasks, retest_rows = _run_performance_retest_round(
                selected_tasks, plan, args, con, batch_dir, batch_index
            )
            retest_candidates, _ = candidate_performance_issues(
                retest_rows,
                latency_ratio_threshold=args.dynamic_compare_latency_ratio_threshold,
                busbw_ratio_threshold=args.dynamic_compare_busbw_ratio_threshold,
                min_cohort=args.dynamic_compare_min_cohort,
                small_max_bytes=args.dynamic_compare_small_max_bytes,
                large_min_bytes=args.dynamic_compare_large_min_bytes,
                small_latency_warn=args.dynamic_compare_small_latency_warn,
                small_latency_abs_delta_seconds=args.dynamic_compare_small_latency_abs_delta_seconds,
                small_latency_mad_multiplier=args.dynamic_compare_small_latency_mad_multiplier,
            )
            retest_candidates = [item for item in retest_candidates if item.get("message_class") == "large"]
            parent_by_retest = {task.group_id: task.parent_group_id for task in retest_tasks}
            for item in retest_candidates:
                parent = parent_by_retest.get(str(item.get("pod_name", "")), "")
                identity = (parent, str(item.get("case_id", "")))
                if identity in initial_ids:
                    repeated_ids.add(identity)
                    confirmed_ids.add(parent)
                else:
                    new_ids.add(identity)
                    parent_item = dict(item)
                    parent_item["pod_name"] = parent
                    parent_item["node_name"] = parent
                    retest_only_items.append(parent_item)
        recovered = initial_ids - repeated_ids
        update_performance_candidate_statuses(
            con,
            phase,
            round_id,
            {identity: "CONFIRMED" for identity in repeated_ids}
            | {identity: "RECOVERED" for identity in recovered},
        )
        if retest_only_items:
            record_performance_candidates(con, phase, round_id, retest_only_items, "RETEST_ONLY")
        new_control_groups = {group_id for group_id, _case_id in new_ids if group_id in all_control_group_ids}
        broad_control_threshold = max(2, (len(all_control_group_ids) + 4) // 5)
        advisory = advisory or len(new_control_groups) >= broad_control_threshold
        record_event(
            con,
            "performance_retest_done",
            round_id,
            {
                "batch_count": len(batches),
                "confirmed_groups": sorted(confirmed_ids),
                "confirmed_cases": sorted([list(item) for item in repeated_ids]),
                "recovered_count": len(recovered),
                "retest_only_count": len(new_ids),
                "new_control_groups": sorted(new_control_groups),
                "control_group_count": len(all_control_group_ids),
                "advisory": advisory,
                "execution_continued": True,
            },
        )
    else:
        advisory = True
        initial_ids = {(str(item.get("pod_name", "")), str(item.get("case_id", ""))) for item in large_candidates}
        update_performance_candidate_statuses(
            con,
            phase,
            round_id,
            {identity: "UNCONFIRMED" for identity in initial_ids},
        )
        record_event(
            con,
            "performance_retest_required",
            round_id,
            {"candidate_count": len(large_candidates), "auto_retest": False, "execution_continued": True},
        )
    confirmed_tasks = [task_by_id[group_id] for group_id in sorted(confirmed_ids) if group_id in task_by_id]
    return confirmed_tasks, advisory


def reuse_final_all_from_phase(
    source_phase: str,
    pods: list[Pod],
    args: argparse.Namespace,
    con: sqlite3.Connection,
) -> bool:
    candidates = phase_pass_nodes(con, pods)
    if len(candidates) <= 2 or not source_phase or args.batch_fault_type:
        return False
    existing_final = con.execute(
        "select status from groups where group_id='final_all_group_0000'"
    ).fetchone()
    if existing_final and str(existing_final[0]) in PASS_STATUSES:
        return False
    source_rows = con.execute(
        """
        select group_id,round_id,nodes_json,status
        from groups
        where phase=? and parent_group_id=''
        order by group_id
        """,
        (source_phase,),
    ).fetchall()
    if len(source_rows) != 1:
        return False
    source_group_id, source_round_id, source_nodes_json, source_status = source_rows[0]
    if str(source_status) != "PASS":
        return False
    try:
        source_nodes = [str(node) for node in json.loads(source_nodes_json or "[]")]
    except (json.JSONDecodeError, TypeError):
        return False
    candidate_nodes = [pod.node_name for pod in candidates]
    if len(source_nodes) != len(candidate_nodes) or set(source_nodes) != set(candidate_nodes):
        return False
    result_row = con.execute(
        """
        select metrics_json,elapsed_seconds,local_workdirs_json,status
        from group_results
        where group_id=?
        order by id desc
        limit 1
        """,
        (source_group_id,),
    ).fetchone()
    if not result_row or str(result_row[3]) != "PASS":
        return False
    try:
        source_metrics = json.loads(result_row[0] or "{}")
    except json.JSONDecodeError:
        return False
    if source_metrics.get("execution_signature") != execution_signature(args):
        return False

    pods_by_node = {pod.node_name: pod for pod in candidates}
    source_pods = [pods_by_node[node] for node in source_nodes]
    task = GroupTask(
        phase="final_all",
        round_id="final_all_r1",
        group_id="final_all_group_0000",
        pods=source_pods,
        parent_group_id=str(source_group_id),
    )
    table_group(con, task, "REUSED_PASS")
    con.execute(
        """
        update groups
        set phase=?,round_id=?,group_size=?,nodes_json=?,status=?,parent_group_id=?,attempt=?
        where group_id=?
        """,
        (
            task.phase,
            task.round_id,
            len(task.pods),
            json.dumps(source_nodes, ensure_ascii=False),
            "REUSED_PASS",
            str(source_group_id),
            task.attempt,
            task.group_id,
        ),
    )
    source_elapsed = float(result_row[1] or 0.0)
    reused_metrics = dict(source_metrics)
    reused_metrics.update(
        {
            "reused": True,
            "reused_from_group_id": str(source_group_id),
            "source_phase": source_phase,
            "source_round": str(source_round_id),
            "source_elapsed_seconds": source_elapsed,
            "saved_execution": True,
        }
    )
    con.execute(
        """
        insert into group_results(group_id,status,error_type,metrics_json,elapsed_seconds,local_workdirs_json,created_at)
        values(?,?,?,?,?,?,?)
        """,
        (
            task.group_id,
            "REUSED_PASS",
            "",
            json.dumps(reused_metrics, ensure_ascii=False, sort_keys=True),
            0.0,
            str(result_row[2] or "{}"),
            iso_now(),
        ),
    )
    for pod in source_pods:
        set_node_status(con, pod.node_name, "PASS", "final_all")
    payload = {
        "source_phase": source_phase,
        "source_group_id": str(source_group_id),
        "node_count": len(source_pods),
        "source_elapsed_seconds": source_elapsed,
    }
    record_event(con, "phase_reused", "final_all", payload)
    record_event(con, "phase_done", "final_all", {"pass": 1, "fail": 0, "counts": {"REUSED_PASS": 1}})
    con.commit()
    print(
        f"[batch-healthcheck] phase_reused phase=final_all source_phase={source_phase} "
        f"source_group={source_group_id} nodes={len(source_pods)} "
        f"saved_estimate={source_elapsed}s",
        flush=True,
    )
    return True


def supersede_phase_with_final_all(
    phase: str,
    pods: list[Pod],
    args: argparse.Namespace,
    con: sqlite3.Connection,
) -> bool:
    if phase not in {"ep8", "scale64", "scale128", "scale256"}:
        return False
    if args.disable_final_superset_skip or args.batch_fault_type:
        return False
    candidates = phase_pass_nodes(con, pods)
    if len(candidates) != phase_scale(phase):
        return False
    tasks = make_phase_groups(phase, candidates, args.group_seed)
    if len(tasks) != 1 or {pod.node_name for pod in tasks[0].pods} != {pod.node_name for pod in candidates}:
        return False
    task = tasks[0]
    table_group(con, task, "SUPERSEDED")
    payload = {
        "phase": phase,
        "group_id": task.group_id,
        "node_count": len(candidates),
        "reason": "final_all_matrix_superset",
        "superseded_by": "final_all",
    }
    record_event(con, "phase_superseded", phase, payload)
    con.commit()
    print(
        f"[batch-healthcheck] phase_superseded phase={phase} group={task.group_id} "
        f"nodes={len(candidates)} superseded_by=final_all reason=final_all_matrix_superset",
        flush=True,
    )
    return True


def run_phase(
    phase: str,
    pods: list[Pod],
    args: argparse.Namespace,
    con: sqlite3.Connection,
    batch_dir: Path,
) -> bool:
    candidates, direct_final_all = phase_candidates(phase, con, pods)
    if direct_final_all:
        print(
            f"[batch-healthcheck] phase_direct phase=final_all candidates={len(candidates)} "
            "reason=no_prior_node_status",
            flush=True,
        )
        record_event(
            con,
            "phase_direct",
            phase,
            {"reason": "no_prior_node_status", "candidates": len(candidates)},
        )
    if phase == "final_all" and not candidates:
        print("[batch-healthcheck] phase_skip phase=final_all reason=no_pass_nodes", flush=True)
        record_event(con, "phase_skip", phase, {"reason": "no_pass_nodes", "candidates": 0})
        return False
    if phase != "final_all":
        size = phase_scale(phase)
        if len(candidates) < size:
            print(
                f"[batch-healthcheck] phase_skip phase={phase} reason=not_enough_pass_nodes "
                f"need={size} have={len(candidates)}",
                flush=True,
            )
            record_event(
                con,
                "phase_skip",
                phase,
                {"reason": "not_enough_pass_nodes", "need": size, "candidates": len(candidates)},
            )
            return False
    tasks = make_phase_groups(phase, candidates, args.group_seed)
    print(f"[batch-healthcheck] phase_start phase={phase} groups={len(tasks)} candidates={len(candidates)}", flush=True)
    record_event(con, "phase_start", phase, {"groups": len(tasks), "candidates": len(candidates)})
    failed: list[GroupTask] = []
    phase_performance_advisory = False
    passed = 0
    tasks_by_round: dict[str, list[GroupTask]] = {}
    for task in tasks:
        tasks_by_round.setdefault(task.round_id, []).append(task)
    for round_id, round_tasks in tasks_by_round.items():
        runnable: list[GroupTask] = []
        skipped_pass = 0
        for task in round_tasks:
            existing = con.execute("select status from groups where group_id=?", (task.group_id,)).fetchone()
            if args.resume and existing and existing[0] in PASS_STATUSES:
                skipped_pass += 1
                passed += 1
                continue
            prepare_task_fault_env(task, args, con)
            runnable.append(task)
        if not runnable:
            print(
                f"[batch-healthcheck] round_skip phase={phase} round={round_id} "
                f"reason=all_pass groups={skipped_pass}",
                flush=True,
            )
            continue
        max_workers = len(runnable) if args.phase_group_concurrency <= 0 else min(args.phase_group_concurrency, len(runnable))
        print(
            f"[batch-healthcheck] round_start phase={phase} round={round_id} "
            f"groups={len(runnable)} concurrency={max_workers} skipped_pass={skipped_pass}",
            flush=True,
        )
        round_started = time.monotonic()
        round_pass = 0
        round_fail = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_group_with_own_connection, task, args, batch_dir) for task in runnable]
            for future in concurrent.futures.as_completed(futures):
                task, status = future.result()
                if status in PASS_STATUSES:
                    passed += 1
                    round_pass += 1
                else:
                    failed.append(task)
                    round_fail += 1
        print(
            f"[batch-healthcheck] round_done phase={phase} round={round_id} "
            f"pass={round_pass + skipped_pass} fail={round_fail} "
            f"elapsed={round(time.monotonic() - round_started, 3)}s",
            flush=True,
        )
        if _truthy(args.dynamic_compare):
            performance_suspects, round_advisory = compare_round_performance(
                round_tasks, args, con, batch_dir
            )
            phase_performance_advisory = phase_performance_advisory or round_advisory
            if performance_suspects:
                print(
                    f"[batch-healthcheck] performance_suspects phase={phase} round={round_id} "
                    f"groups={len(performance_suspects)} execution_continues=1",
                    flush=True,
                )
    if phase_performance_advisory:
        print(
            f"[batch-healthcheck] phase_warning phase={phase} "
            "reason=performance_candidates_require_vendor_review execution_continues=1",
            flush=True,
        )
        record_event(
            con,
            "phase_warning",
            phase,
            {
                "reason": "performance_candidates_require_vendor_review",
                "execution_continued": True,
            },
        )
    if failed:
        print(f"[batch-healthcheck] phase_retest phase={phase} failed_groups={len(failed)}", flush=True)
        run_retests(failed, pods, args, con, batch_dir)
    if phase == "pairwise":
        finalize_pairwise_localization(con, phase)
    if failed:
        unresolved_suspects = con.execute(
            """
            select node_name,pod_name,fail_reason
            from nodes
            where status='SUSPECT' and last_phase=?
            order by node_name
            """,
            (phase,),
        ).fetchall()
        if unresolved_suspects:
            suspect_nodes = [str(row[0]) for row in unresolved_suspects]
            print(
                f"[batch-healthcheck] phase_blocked phase={phase} "
                f"reason=unresolved_suspects suspect_nodes={','.join(suspect_nodes)}",
                flush=True,
            )
            record_event(
                con,
                "phase_blocked",
                phase,
                {
                    "reason": "unresolved_suspects",
                    "suspect_nodes": suspect_nodes,
                    "suspects": [
                        {"node_name": row[0], "pod_name": row[1], "fail_reason": row[2]}
                        for row in unresolved_suspects
                    ],
                },
            )
            return False
    status_rows = con.execute(
        "select status,count(*) from groups where phase=? group by status",
        (phase,),
    ).fetchall()
    counts = {str(status): int(count) for status, count in status_rows}
    pass_count = sum(count for status, count in counts.items() if status in PASS_STATUSES)
    fail_count = sum(count for status, count in counts.items() if status not in PASS_STATUSES)
    if pass_count == 0:
        print(f"[batch-healthcheck] phase_blocked phase={phase} reason=no_pass_groups", flush=True)
        record_event(con, "phase_blocked", phase, {"reason": "no_pass_groups", "counts": counts})
        return False
    print(
        f"[batch-healthcheck] phase_done phase={phase} pass={pass_count} "
        f"fail={fail_count} timeout={counts.get('TIMEOUT', 0)}",
        flush=True,
    )
    record_event(con, "phase_done", phase, {"pass": pass_count, "fail": fail_count, "counts": counts})
    return True


def write_node_files(con: sqlite3.Connection, batch_dir: Path) -> None:
    rows = con.execute("select node_name,status from nodes order by node_name").fetchall()
    for status, filename in [("PASS", "pass_nodes.txt"), ("SUSPECT", "suspect_nodes.txt"), ("FAIL", "fail_nodes.txt")]:
        values = [node for node, row_status in rows if row_status == status]
        (batch_dir / filename).write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")
    pod_rows = con.execute(
        """
        select p.pod_name,p.node_name,p.host_ip
        from pods p join nodes n on n.node_name=p.node_name
        where n.status='PASS'
        order by case when p.pod_name like '%master-%' then 0 else 1 end,p.pod_name
        """
    ).fetchall()
    lines = []
    names = []
    name_width = max([len(row[0]) for row in pod_rows] or [1])
    node_width = max([len(row[1]) for row in pod_rows] or [1])
    for pod_name, node_name, host_ip in pod_rows:
        short = pod_name
        lines.append(f"{short:<{name_width}}   -> nodeName {node_name:<{node_width}}   hostIP {host_ip}")
        names.append(node_name)
    lines.append(",".join(names))
    (batch_dir / "node_map.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def performance_candidate_rows(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query = con.execute(
        """
        select p.phase,p.round_id,p.group_id,p.case_id,p.status,p.details_json,g.nodes_json
        from performance_candidates p
        left join groups g on g.group_id=p.group_id
        order by p.phase,p.round_id,p.group_id,p.case_id
        """
    )
    for phase, round_id, group_id, case_id, status, details_json, nodes_json in query:
        try:
            details = json.loads(details_json or "{}")
        except json.JSONDecodeError:
            details = {}
        try:
            nodes = [str(node) for node in json.loads(nodes_json or "[]")]
        except (json.JSONDecodeError, TypeError):
            nodes = []
        rows.append(
            {
                "phase": str(phase),
                "round_id": str(round_id),
                "group_id": str(group_id),
                "case_id": str(case_id),
                "status": str(status),
                "nodes": nodes,
                "details": details,
            }
        )
    return rows


def apply_performance_candidate_statuses(con: sqlite3.Connection) -> list[str]:
    unresolved = {"DETECTED", "CONFIRMED", "UNCONFIRMED", "RETEST_ONLY"}
    evidence: dict[str, set[str]] = {}
    for row in performance_candidate_rows(con):
        if row["status"] not in unresolved:
            continue
        for node_name in row["nodes"]:
            evidence.setdefault(node_name, set()).add(f"{row['round_id']}:{row['group_id']}:{row['case_id']}")
    for node_name, reasons in evidence.items():
        if node_status(con, node_name) == "FAIL":
            continue
        con.execute(
            """
            update nodes
            set status='SUSPECT',suspect_score=?,fail_reason=?,last_phase='performance_gate'
            where node_name=?
            """,
            (len(reasons), f"unresolved_performance_candidates={len(reasons)}", node_name),
        )
    con.commit()
    return sorted(evidence)


def write_performance_candidate_files(con: sqlite3.Connection, batch_dir: Path) -> None:
    rows = performance_candidate_rows(con)
    (batch_dir / "performance_candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    unresolved = {"DETECTED", "CONFIRMED", "UNCONFIRMED", "RETEST_ONLY"}
    node_evidence: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] not in unresolved:
            continue
        for node_name in row["nodes"]:
            node_evidence.setdefault(node_name, []).append(row)
    (batch_dir / "performance_candidate_nodes.txt").write_text(
        "\n".join(sorted(node_evidence)) + ("\n" if node_evidence else ""),
        encoding="utf-8",
    )
    lines = [
        "# Performance Candidates",
        "",
        f"- detected_cases: `{len(rows)}`",
        f"- unresolved_nodes: `{len(node_evidence)}`",
        "- note: performance candidates do not stop later phases; unresolved nodes require vendor review.",
        "",
        "## Candidate Cases",
        "",
        "| phase | round | group | status | case | nodes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['round_id']} | {row['group_id']} | {row['status']} | "
            f"{row['case_id']} | {', '.join(row['nodes'])} |"
        )
    lines.extend(["", "## Nodes Requiring Review", ""])
    if not node_evidence:
        lines.append("No unresolved performance candidate nodes.")
    else:
        lines.extend(["| node | unresolved cases |", "| --- | ---: |"])
        for node_name in sorted(node_evidence):
            lines.append(f"| {node_name} | {len(node_evidence[node_name])} |")
    (batch_dir / "performance_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_localization_files(con: sqlite3.Connection, batch_dir: Path) -> None:
    rows = con.execute(
        """
        select l.node_name,p.pod_name,l.phase,l.classification,
               l.failed_group_count,l.passed_group_count,
               l.distinct_failed_partners,l.distinct_passed_partners,
               l.retest_failed_count,l.retest_passed_count,
               l.failed_partners_json,l.passed_partners_json,l.reason
        from node_localization l
        left join pods p on p.node_name=l.node_name
        order by case l.classification when 'SUSPECT' then 0 else 1 end,l.node_name,l.phase
        """
    ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            failed_partners = json.loads(row[10] or "[]")
        except json.JSONDecodeError:
            failed_partners = []
        try:
            passed_partners = json.loads(row[11] or "[]")
        except json.JSONDecodeError:
            passed_partners = []
        items.append(
            {
                "node_name": row[0],
                "pod_name": row[1] or "",
                "phase": row[2],
                "classification": row[3],
                "failed_group_count": row[4],
                "passed_group_count": row[5],
                "distinct_failed_partners": row[6],
                "distinct_passed_partners": row[7],
                "retest_failed_count": row[8],
                "retest_passed_count": row[9],
                "failed_partners": failed_partners,
                "passed_partners": passed_partners,
                "reason": row[12],
            }
        )
    summary = {
        "suspect_nodes": [item["node_name"] for item in items if item["classification"] == "SUSPECT"],
        "nodes": items,
    }
    (batch_dir / "localization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_lines = [
        "# Node Localization Summary",
        "",
        f"- suspect_nodes: `{','.join(summary['suspect_nodes'])}`",
        "",
        "| node | pod | phase | classification | failed groups | passed groups | failed partners | retest fail | retest pass | reason |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in items:
        md_lines.append(
            "| {node} | {pod} | {phase} | {classification} | {failed} | {passed} | {partners} | {retest_failed} | {retest_passed} | {reason} |".format(
                node=item["node_name"],
                pod=item["pod_name"],
                phase=item["phase"],
                classification=item["classification"],
                failed=item["failed_group_count"],
                passed=item["passed_group_count"],
                partners=item["distinct_failed_partners"],
                retest_failed=item["retest_failed_count"],
                retest_passed=item["retest_passed_count"],
                reason=item["reason"],
            )
        )
    (batch_dir / "localization_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_group_plan_files(con: sqlite3.Connection, batch_dir: Path) -> None:
    rows = con.execute(
        """
        select phase,round_id,group_id,group_size,nodes_json,status,parent_group_id,attempt
        from groups
        order by round_id,group_id
        """
    ).fetchall()
    rows = sorted(rows, key=lambda row: (*phase_order_key(str(row[0])), str(row[1]), str(row[2])))
    jsonl_rows: list[dict[str, Any]] = []
    md_lines = [
        "# Batch Group Plan",
        "",
        "| phase | round | group_id | size | status | nodes | parent_group_id | attempt |",
        "| --- | --- | --- | ---: | --- | --- | --- | ---: |",
    ]
    for phase, round_id, group_id, group_size, nodes_json, status, parent_group_id, attempt in rows:
        try:
            nodes = json.loads(nodes_json)
        except json.JSONDecodeError:
            nodes = []
        if not isinstance(nodes, list):
            nodes = []
        row = {
            "phase": phase,
            "round_id": round_id,
            "group_id": group_id,
            "group_size": group_size,
            "status": status,
            "nodes": nodes,
            "parent_group_id": parent_group_id or "",
            "attempt": attempt,
        }
        jsonl_rows.append(row)
        md_lines.append(
            "| {phase} | {round_id} | {group_id} | {group_size} | {status} | {nodes} | {parent} | {attempt} |".format(
                phase=phase,
                round_id=round_id,
                group_id=group_id,
                group_size=group_size,
                status=status,
                nodes=", ".join(str(node) for node in nodes),
                parent=parent_group_id or "",
                attempt=attempt,
            )
        )
    (batch_dir / "group_plan.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in jsonl_rows),
        encoding="utf-8",
    )
    (batch_dir / "group_plan.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_comm_path_summary_files(con: sqlite3.Connection, batch_dir: Path) -> None:
    rows = con.execute(
        """
        select group_id,status,metrics_json,created_at
        from group_results
        order by id
        """
    ).fetchall()
    comm_rows: list[dict[str, Any]] = []
    for group_id, status, metrics_json, created_at in rows:
        try:
            metrics = json.loads(metrics_json or "{}")
        except json.JSONDecodeError:
            metrics = {}
        for row in metrics.get("comm_path_summary", []) or []:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item.setdefault("group_id", group_id)
            item.setdefault("group_status", status)
            item["created_at"] = created_at
            comm_rows.append(item)
    if not comm_rows:
        return
    (batch_dir / "comm_path_summary.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in comm_rows),
        encoding="utf-8",
    )
    md_lines = [
        "# Communication Path Summary",
        "",
        "| group | status | rank | pod | node | backend | runtime | key envs |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in comm_rows:
        env = row.get("env") if isinstance(row.get("env"), dict) else {}
        key_envs = []
        for key in [
            "MCCL_IB_HCA",
            "MCCL_IB_GID_INDEX",
            "MCCL_SOCKET_IFNAME",
            "MCCL_IB_DISABLE",
            "NCCL_IB_DISABLE",
            "NCCL_SOCKET_IFNAME",
            "HCCL_SOCKET_IFNAME",
            "GLOO_SOCKET_IFNAME",
        ]:
            if key in env:
                key_envs.append(f"{key}={env[key]}")
        md_lines.append(
            "| {group} | {status} | {rank} | {pod} | {node} | {backend} | {runtime} | {envs} |".format(
                group=row.get("group_id", ""),
                status=row.get("group_status", ""),
                rank=row.get("rank", ""),
                pod=row.get("pod_name", ""),
                node=row.get("node_name", ""),
                backend=row.get("dist_backend", ""),
                runtime=row.get("comm_runtime", ""),
                envs="<br>".join(key_envs),
            )
        )
    (batch_dir / "comm_path_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_gate_files(con: sqlite3.Connection, batch_dir: Path, args: argparse.Namespace) -> None:
    rows = con.execute(
        "select timestamp,event_type,message,payload_json from events order by id"
    ).fetchall()
    warnings: list[dict[str, Any]] = []
    retests: list[dict[str, Any]] = []
    systemic: list[dict[str, Any]] = []
    for timestamp, event_type, message, payload_json in rows:
        try:
            payload = json.loads(payload_json or "{}")
        except json.JSONDecodeError:
            payload = {}
        base = {"timestamp": timestamp, "event_type": event_type, "scope": message}
        if event_type == "performance_warning":
            for warning in payload.get("warnings", []) or []:
                warnings.append({**base, **warning})
        elif event_type == "performance_retest_start":
            retests.append({**base, **payload})
        elif event_type == "systemic_performance_event":
            systemic.append({**base, **payload})
    (batch_dir / "performance_warnings.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in warnings),
        encoding="utf-8",
    )
    (batch_dir / "retest_plan.json").write_text(
        json.dumps(retests, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_exceeded = bool(args._runtime_warning_emitted)
    runtime_target = args.batch_runtime_warn_seconds
    elapsed = float(args.batch_elapsed_seconds)
    summary = {
        "performance_warning_count": len(warnings),
        "performance_retest_count": len(retests),
        "systemic_performance_event_count": len(systemic),
        "runtime_sla_status": "WARN" if runtime_exceeded else "PASS",
        "runtime_target_seconds": runtime_target,
        "elapsed_seconds": elapsed,
        "runtime_target_exceeded": runtime_exceeded,
        "runtime_exceeded_seconds": max(0.0, elapsed - runtime_target) if runtime_target > 0 else 0.0,
        "systemic_performance_events": systemic,
    }
    (batch_dir / "gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Healthcheck Gate Summary",
        "",
        f"- performance_warning_count: `{len(warnings)}`",
        f"- performance_retest_count: `{len(retests)}`",
        f"- systemic_performance_event_count: `{len(systemic)}`",
        f"- runtime_sla_status: `{summary['runtime_sla_status']}`",
        f"- runtime_target_seconds: `{runtime_target}`",
        f"- elapsed_seconds: `{elapsed}`",
        f"- runtime_target_exceeded: `{str(runtime_exceeded).lower()}`",
        "",
        "Runtime SLA warnings do not stop execution or change the healthcheck result.",
        "Systemic performance events are advisory; all candidate groups are retested in bounded batches and later phases continue.",
    ]
    (batch_dir / "gate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_batch_summary(con: sqlite3.Connection, batch_dir: Path, args: argparse.Namespace, overall: str) -> None:
    runtime_warning_emitted = bool(getattr(args, "_runtime_warning_emitted", False))
    runtime_target_seconds = int(getattr(args, "batch_runtime_warn_seconds", 900))
    batch_elapsed_seconds = float(getattr(args, "batch_elapsed_seconds", 0.0))
    phase_rows = con.execute(
        "select phase,status,count(*) from groups group by phase,status order by status"
    ).fetchall()
    phase_rows = sorted(phase_rows, key=lambda row: (*phase_order_key(str(row[0])), str(row[1])))
    node_rows = con.execute("select status,count(*) from nodes group by status order by status").fetchall()
    reuse_row = con.execute(
        """
        select g.group_id,g.parent_group_id,r.metrics_json
        from groups g
        left join group_results r on r.id=(
          select max(r2.id) from group_results r2 where r2.group_id=g.group_id
        )
        where g.phase='final_all' and g.status='REUSED_PASS'
        limit 1
        """
    ).fetchone()
    final_all_reuse: dict[str, Any] | None = None
    if reuse_row:
        try:
            reuse_metrics = json.loads(reuse_row[2] or "{}")
        except json.JSONDecodeError:
            reuse_metrics = {}
        final_all_reuse = {
            "status": "REUSED_PASS",
            "group_id": str(reuse_row[0]),
            "reused_from_group_id": str(reuse_row[1]),
            "source_phase": str(reuse_metrics.get("source_phase", "")),
            "source_elapsed_seconds": float(reuse_metrics.get("source_elapsed_seconds", 0.0) or 0.0),
            "avoided_group_executions": 1,
        }
    summary = {
        "overall_status": overall,
        "job_name": args.job_name,
        "batch_run_id": args.batch_run_id,
        "phase_status_counts": [{"phase": p, "status": s, "count": c} for p, s, c in phase_rows],
        "node_status_counts": [{"status": s, "count": c} for s, c in node_rows],
        "final_all_reuse": final_all_reuse,
        "runtime_sla_status": "WARN" if runtime_warning_emitted else "PASS",
        "runtime_target_seconds": runtime_target_seconds,
        "elapsed_seconds": batch_elapsed_seconds,
        "runtime_target_exceeded": runtime_warning_emitted,
    }
    (batch_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Batch Health Check Summary",
        "",
        f"- overall_status: `{overall}`",
        f"- job_name: `{args.job_name}`",
        f"- batch_run_id: `{args.batch_run_id}`",
        f"- elapsed_seconds: `{batch_elapsed_seconds}`",
        "",
        "## Phase Status",
        "",
        "| phase | status | count |",
        "| --- | --- | ---: |",
    ]
    for phase, status, count in phase_rows:
        lines.append(f"| {phase} | {status} | {count} |")
    lines.extend(["", "## Node Status", "", "| status | count |", "| --- | ---: |"])
    for status, count in node_rows:
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Runtime SLA",
            "",
            f"- target_seconds: `{runtime_target_seconds}`",
            f"- actual_seconds: `{batch_elapsed_seconds}`",
            f"- status: `{'WARN' if runtime_warning_emitted else 'PASS'}`",
            f"- execution_continued_after_warning: `{str(runtime_warning_emitted).lower()}`",
        ]
    )
    if final_all_reuse:
        lines.extend(
            [
                "",
                "## Final-All Reuse",
                "",
                "- final_all_status: `REUSED_PASS`",
                f"- reused_from_group_id: `{final_all_reuse['reused_from_group_id']}`",
                f"- source_phase: `{final_all_reuse['source_phase']}`",
                f"- avoided_group_executions: `{final_all_reuse['avoided_group_executions']}`",
                f"- estimated_saved_seconds: `{final_all_reuse['source_elapsed_seconds']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `group_plan.md`",
            "- `group_plan.jsonl`",
            "- `pass_nodes.txt`",
            "- `suspect_nodes.txt`",
            "- `fail_nodes.txt`",
            "- `node_map.txt`",
            "- `localization_summary.md`",
            "- `localization_summary.json`",
            "- `comm_path_summary.md` when COMM_PATH_DEBUG=1 produced data",
        ]
    )
    (batch_dir / "batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_all_root_failed(con: sqlite3.Connection) -> bool:
    rows = con.execute(
        """
        select status
        from groups
        where phase='final_all' and coalesce(parent_group_id, '')=''
        """
    ).fetchall()
    return any(str(row[0]) not in PASS_STATUSES for row in rows)


def cleanup_batch_tmp(args: argparse.Namespace) -> None:
    if args.dry_run in {"1", "true"}:
        return
    root = Path(args.batch_tmp_dir)
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not path.is_dir():
            continue
        try:
            path.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def cleanup_empty_group_output_root(args: argparse.Namespace) -> None:
    if args.failed_group_output_mode != "local-link":
        return
    if args.dry_run in {"1", "true"}:
        return
    root = Path(args.group_output_root) / args.batch_run_id
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not path.is_dir():
            continue
        try:
            path.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.batch_run_id:
        args.batch_run_id = time.strftime("%Y%m%d_%H%M%S")
    args.target_scale = int(args.target_scale or 0)
    args.group_seed = int(args.group_seed)
    args.group_timeout_seconds = int(args.group_timeout_seconds)
    args.final_all_timeout_seconds = int(args.final_all_timeout_seconds)
    args.progress_interval_seconds = int(args.progress_interval_seconds)
    args.phase_group_concurrency = int(args.phase_group_concurrency)
    args.batch_fault_type = str(args.batch_fault_type or "").strip().lower()
    args.batch_fault_phase = str(args.batch_fault_phase or "all").strip()
    args.batch_fault_nodes = ",".join(_csv_values(args.batch_fault_nodes))
    args.batch_fault_pods = ",".join(_csv_values(args.batch_fault_pods))
    args.batch_fault_max_hits = int(args.batch_fault_max_hits)
    args.batch_fault_sleep_seconds = float(args.batch_fault_sleep_seconds)
    args.batch_fault_delay_ms = float(args.batch_fault_delay_ms)
    args.comm_path_debug = str(args.comm_path_debug or "0").strip()
    args.dynamic_compare_small_max_bytes = parse_size(str(args.dynamic_compare_small_max_size))
    args.dynamic_compare_large_min_bytes = parse_size(str(args.dynamic_compare_large_min_size))
    args.dynamic_compare_small_latency_abs_delta_seconds = (
        float(args.dynamic_compare_small_latency_abs_delta_ms) / 1000.0
    )
    args.dynamic_compare_retest_max_groups = int(args.dynamic_compare_retest_max_groups)
    args.dynamic_compare_retest_time_budget_seconds = int(args.dynamic_compare_retest_time_budget_seconds)
    args.dynamic_compare_systemic_candidate_fraction = float(args.dynamic_compare_systemic_candidate_fraction)
    args.batch_runtime_warn_seconds = int(args.batch_runtime_warn_seconds)
    if args.group_timeout_seconds <= 0:
        raise ValueError("group_timeout_seconds must be positive")
    if args.final_all_timeout_seconds <= 0:
        raise ValueError("final_all_timeout_seconds must be positive")
    if args.dynamic_compare_small_latency_abs_delta_seconds < 0:
        raise ValueError("dynamic_compare_small_latency_abs_delta_ms must be non-negative")
    if args.dynamic_compare_small_latency_mad_multiplier < 0:
        raise ValueError("dynamic_compare_small_latency_mad_multiplier must be non-negative")
    if args.dynamic_compare_retest_max_groups < 1:
        raise ValueError("dynamic_compare_retest_max_groups must be positive")
    if args.dynamic_compare_retest_time_budget_seconds <= 0:
        raise ValueError("dynamic_compare_retest_time_budget_seconds must be positive")
    if not 0 < args.dynamic_compare_systemic_candidate_fraction <= 1:
        raise ValueError("dynamic_compare_systemic_candidate_fraction must be in (0, 1]")
    if args.batch_runtime_warn_seconds < 0:
        raise ValueError("batch_runtime_warn_seconds must be non-negative")
    args._batch_fault_hit_count = 0
    args.failed_group_output_mode = str(args.failed_group_output_mode or "local-link").strip()
    if args.failed_group_output_mode not in {"local-link", "shared"}:
        raise ValueError("failed_group_output_mode must be local-link or shared")
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vcctl multi-node grouped healthcheck batch runner")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--healthcheck-script", default="")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--vcctl-bin", default="vcctl")
    parser.add_argument("--pod-json-file", default="")
    parser.add_argument(
        "--preserve-pod-json-order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep POD_JSON_FILE item order for exact fixed-rank reproduction.",
    )
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--batch-run-id", default="")
    parser.add_argument("--target-scale", default="0")
    parser.add_argument("--phases", default="")
    parser.add_argument("--group-seed", default="20260706")
    parser.add_argument("--group-timeout-seconds", default="180")
    parser.add_argument("--final-all-timeout-seconds", default="300")
    parser.add_argument("--progress-interval-seconds", default="10")
    parser.add_argument("--phase-group-concurrency", default="0")
    parser.add_argument("--dry-run", default="1")
    parser.add_argument("--pre-clean", default="1")
    parser.add_argument("--dynamic-compare", default="1")
    parser.add_argument("--dynamic-compare-busbw-ratio-threshold", type=float, default=0.7)
    parser.add_argument("--dynamic-compare-latency-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--dynamic-compare-small-max-size", default="1M")
    parser.add_argument("--dynamic-compare-large-min-size", default="1G")
    parser.add_argument("--dynamic-compare-small-latency-warn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic-compare-small-latency-abs-delta-ms", type=float, default=0.2)
    parser.add_argument("--dynamic-compare-small-latency-mad-multiplier", type=float, default=6.0)
    parser.add_argument("--dynamic-compare-min-cohort", type=int, default=3)
    parser.add_argument("--dynamic-compare-auto-retest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-compare-retest-max-groups", default="32")
    parser.add_argument("--dynamic-compare-retest-time-budget-seconds", default="120")
    parser.add_argument("--dynamic-compare-systemic-candidate-fraction", type=float, default=0.05)
    parser.add_argument("--batch-runtime-warn-seconds", default="900")
    parser.add_argument("--disable-final-superset-skip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-group-outputs", default="0")
    parser.add_argument("--pod-project-dir", default="")
    parser.add_argument("--group-output-root", default="/tmp/pretrain_healthcheck_group_outputs/vcctl")
    parser.add_argument("--failed-group-output-mode", default="local-link", choices=["local-link", "shared"])
    parser.add_argument("--batch-fault-type", default="")
    parser.add_argument("--batch-fault-node", default="")
    parser.add_argument("--batch-fault-pod", default="")
    parser.add_argument("--batch-fault-nodes", default="")
    parser.add_argument("--batch-fault-pods", default="")
    parser.add_argument("--batch-fault-phase", default="all")
    parser.add_argument("--batch-fault-max-hits", default="0")
    parser.add_argument("--batch-fault-sleep-seconds", default="300")
    parser.add_argument("--batch-fault-delay-ms", default="200")
    parser.add_argument("--comm-path-debug", default="0")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    args.batch_started_monotonic = time.monotonic()
    args._runtime_warning_lock = threading.Lock()
    args._runtime_warning_emitted = False
    if not args.healthcheck_script:
        args.healthcheck_script = str(Path(args.project_dir) / "scripts/metax/run_vcctl_healthcheck.sh")
    result_root = Path(args.result_root)
    batch_dir = result_root / args.batch_run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    args.batch_tmp_dir = str(Path("/tmp") / f"pretrain_healthcheck_batch_{args.batch_run_id}")
    if args.failed_group_output_mode == "local-link" and args.dry_run not in {"1", "true"}:
        Path(args.group_output_root, args.batch_run_id).mkdir(parents=True, exist_ok=True)
    db_path = batch_dir / "batch_results.sqlite"
    args.db_path = str(db_path)

    _raw, pods = load_pods(args)
    con = init_db(db_path)
    upsert_pods(con, pods)
    phases = parse_phases(args.phases, len(pods), args.target_scale)
    run_phases = phases[:] if "final_all" in phases else phases + ["final_all"]

    print(f"[batch-healthcheck] job          : {args.job_name}")
    print(f"[batch-healthcheck] batch_run_id : {args.batch_run_id}")
    print(f"[batch-healthcheck] node_count   : {len(pods)}")
    print(f"[batch-healthcheck] phases       : {','.join(run_phases)}")
    print(f"[batch-healthcheck] target_scale : {args.target_scale or 'auto'}")
    print(f"[batch-healthcheck] group_timeout: {args.group_timeout_seconds}s")
    print(f"[batch-healthcheck] final_timeout: {args.final_all_timeout_seconds}s")
    print(f"[batch-healthcheck] runtime_warn : {args.batch_runtime_warn_seconds}s")
    print(f"[batch-healthcheck] concurrency  : {args.phase_group_concurrency or 'all groups per round'}")
    print(f"[batch-healthcheck] group outputs: {args.failed_group_output_mode}")
    if args.failed_group_output_mode == "local-link":
        print(f"[batch-healthcheck] group output root: {Path(args.group_output_root) / args.batch_run_id}")
    if args.batch_fault_type:
        print(
            f"[batch-healthcheck] batch_fault : type={args.batch_fault_type} "
            f"phase={args.batch_fault_phase} nodes={','.join(_fault_target_nodes(args)) or '-'} "
            f"pods={','.join(_fault_target_pods(args)) or '-'} max_hits={args.batch_fault_max_hits}"
        )
    if _truthy(args.comm_path_debug):
        print("[batch-healthcheck] comm_path_debug: 1")
    print(f"[batch-healthcheck] result_dir   : {batch_dir}")
    print(f"[batch-healthcheck] sqlite       : {db_path}")
    record_event(con, "batch_start", args.batch_run_id, {"phases": run_phases, "node_count": len(pods)})

    overall = "PASS"
    last_completed_phase = ""
    for phase in run_phases:
        maybe_emit_runtime_warning(args, con)
        if phase != "final_all" and supersede_phase_with_final_all(phase, pods, args, con):
            continue
        if not run_phase(phase, pods, args, con, batch_dir):
            overall = "SUSPECT"
            break
        last_completed_phase = phase

    apply_performance_candidate_statuses(con)
    fail_count = con.execute("select count(*) from nodes where status='FAIL'").fetchone()[0]
    suspect_count = con.execute("select count(*) from nodes where status='SUSPECT'").fetchone()[0]
    if fail_count or final_all_root_failed(con):
        overall = "FAIL"
    elif suspect_count and overall == "PASS":
        overall = "SUSPECT"
    write_node_files(con, batch_dir)
    write_performance_candidate_files(con, batch_dir)
    write_localization_files(con, batch_dir)
    write_group_plan_files(con, batch_dir)
    write_comm_path_summary_files(con, batch_dir)
    args.batch_elapsed_seconds = round(time.monotonic() - args.batch_started_monotonic, 3)
    maybe_emit_runtime_warning(args, con)
    write_gate_files(con, batch_dir, args)
    write_batch_summary(con, batch_dir, args, overall)
    cleanup_batch_tmp(args)
    cleanup_empty_group_output_root(args)
    record_event(con, "batch_done", args.batch_run_id, {"overall_status": overall})
    print(f"[batch-healthcheck] batch_done overall_status={overall}")
    print(f"[batch-healthcheck] pass_nodes={batch_dir / 'pass_nodes.txt'}")
    print(f"[batch-healthcheck] suspect_nodes={batch_dir / 'suspect_nodes.txt'}")
    print(f"[batch-healthcheck] fail_nodes={batch_dir / 'fail_nodes.txt'}")
    print(f"[batch-healthcheck] node_map={batch_dir / 'node_map.txt'}")
    print(f"[batch-healthcheck] sqlite={db_path}")
    return 0 if overall == "PASS" or args.dry_run == "1" else 1


if __name__ == "__main__":
    raise SystemExit(main())
