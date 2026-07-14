#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PASS_STATUSES = {"PASS", "DRY_RUN"}


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


def run_group(task: GroupTask, args: argparse.Namespace, con: sqlite3.Connection, batch_dir: Path) -> str:
    table_group(con, task, "RUNNING")
    group_json = write_group_json(task, args)
    run_stage = f"{task.phase}/{task.group_id}/multi_node_dynamic_suite"
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
            "EXEC_TIMEOUT_SECONDS": str(args.group_timeout_seconds),
            "RESULT_ROOT": run_result_root,
        }
    )
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
    driver_deadline = started + max(1, args.group_timeout_seconds) + 120
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
        time.sleep(0.5)
    returncode = proc.returncode
    elapsed = round(time.monotonic() - started, 3)
    summary = read_summary(output_dir / "summary.json")
    if batch_timeout:
        status, reason = "TIMEOUT", f"BATCH_GROUP_TIMEOUT>{args.group_timeout_seconds + 120}s"
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


def run_phase(
    phase: str,
    pods: list[Pod],
    args: argparse.Namespace,
    con: sqlite3.Connection,
    batch_dir: Path,
) -> bool:
    candidates = phase_pass_nodes(con, pods)
    if phase == "pairwise":
        candidates = [pod for pod in pods if node_status(con, pod.node_name) not in {"FAIL", "EXCLUDED"}]
    if phase == "final_all":
        if len(candidates) <= 2:
            print(
                f"[batch-healthcheck] phase_skip phase={phase} "
                f"reason=covered_by_pairwise have={len(candidates)}",
                flush=True,
            )
            record_event(con, "phase_skip", phase, {"reason": "covered_by_pairwise", "candidates": len(candidates)})
            return True
    else:
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
    pass_count = counts.get("PASS", 0) + counts.get("DRY_RUN", 0)
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
        "select pod_name,node_name,host_ip from pods order by case when pod_name like '%master-%' then 0 else 1 end, pod_name"
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


def write_batch_summary(con: sqlite3.Connection, batch_dir: Path, args: argparse.Namespace, overall: str) -> None:
    phase_rows = con.execute(
        "select phase,status,count(*) from groups group by phase,status order by status"
    ).fetchall()
    phase_rows = sorted(phase_rows, key=lambda row: (*phase_order_key(str(row[0])), str(row[1])))
    node_rows = con.execute("select status,count(*) from nodes group by status order by status").fetchall()
    summary = {
        "overall_status": overall,
        "job_name": args.job_name,
        "batch_run_id": args.batch_run_id,
        "phase_status_counts": [{"phase": p, "status": s, "count": c} for p, s, c in phase_rows],
        "node_status_counts": [{"status": s, "count": c} for s, c in node_rows],
    }
    (batch_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Batch Health Check Summary",
        "",
        f"- overall_status: `{overall}`",
        f"- job_name: `{args.job_name}`",
        f"- batch_run_id: `{args.batch_run_id}`",
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
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--batch-run-id", default="")
    parser.add_argument("--target-scale", default="0")
    parser.add_argument("--phases", default="")
    parser.add_argument("--group-seed", default="20260706")
    parser.add_argument("--group-timeout-seconds", default="180")
    parser.add_argument("--progress-interval-seconds", default="10")
    parser.add_argument("--phase-group-concurrency", default="0")
    parser.add_argument("--dry-run", default="1")
    parser.add_argument("--pre-clean", default="1")
    parser.add_argument("--dynamic-compare", default="1")
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
    for phase in run_phases:
        if not run_phase(phase, pods, args, con, batch_dir):
            overall = "SUSPECT"
            break

    fail_count = con.execute("select count(*) from nodes where status='FAIL'").fetchone()[0]
    suspect_count = con.execute("select count(*) from nodes where status='SUSPECT'").fetchone()[0]
    if fail_count:
        overall = "FAIL"
    elif suspect_count and overall == "PASS":
        overall = "SUSPECT"
    write_node_files(con, batch_dir)
    write_localization_files(con, batch_dir)
    write_group_plan_files(con, batch_dir)
    write_comm_path_summary_files(con, batch_dir)
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
