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
from dataclasses import dataclass
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
    if phase.startswith("scale"):
        return int(phase.removeprefix("scale"))
    raise ValueError(f"unsupported phase: {phase}")


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


def run_group(task: GroupTask, args: argparse.Namespace, con: sqlite3.Connection, batch_dir: Path) -> str:
    table_group(con, task, "RUNNING")
    group_json = write_group_json(task, args)
    run_stage = f"{task.phase}/{task.group_id}/multi_node_dynamic_suite"
    output_dir = Path(args.result_root) / args.batch_run_id / run_stage
    cmd = [
        "bash",
        str(Path(args.project_dir) / "scripts/metax/run_vcctl_healthcheck.sh"),
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
            "RESULT_ROOT": args.result_root,
        }
    )
    if args.pod_project_dir:
        env["PROJECT_REMOTE_DIR"] = args.pod_project_dir

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
    metrics = {
        "summary": summary.get("dynamic_compare") or {},
        "pod_count": summary.get("pod_count"),
        "result_count": summary.get("result_count"),
    }
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
    return [pod for pod in pods if pod.node_name not in exclude and node_status(con, pod.node_name) == "PASS"]


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
        status = run_group(task, args, con, batch_dir)
        con.execute("update retest_tasks set status=? where task_id=?", (status, task.group_id))
        con.commit()
        if status in PASS_STATUSES:
            any_pass = True
            continue
        if len(task.pods) > 2:
            queue.extend(split_retests(task))
    return any_pass


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
    size = phase_scale(phase)
    if len(candidates) < size:
        print(f"[batch-healthcheck] phase_skip phase={phase} reason=not_enough_pass_nodes need={size} have={len(candidates)}")
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


def write_group_plan_files(con: sqlite3.Connection, batch_dir: Path) -> None:
    rows = con.execute(
        """
        select phase,round_id,group_id,group_size,nodes_json,status,parent_group_id,attempt
        from groups
        order by phase,round_id,group_id
        """
    ).fetchall()
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


def write_batch_summary(con: sqlite3.Connection, batch_dir: Path, args: argparse.Namespace, overall: str) -> None:
    phase_rows = con.execute(
        "select phase,status,count(*) from groups group by phase,status order by phase,status"
    ).fetchall()
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


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.batch_run_id:
        args.batch_run_id = time.strftime("%Y%m%d_%H%M%S")
    args.target_scale = int(args.target_scale or 0)
    args.group_seed = int(args.group_seed)
    args.group_timeout_seconds = int(args.group_timeout_seconds)
    args.progress_interval_seconds = int(args.progress_interval_seconds)
    args.phase_group_concurrency = int(args.phase_group_concurrency)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vcctl multi-node grouped healthcheck batch runner")
    parser.add_argument("--project-dir", required=True)
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
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    result_root = Path(args.result_root)
    batch_dir = result_root / args.batch_run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    args.batch_tmp_dir = str(Path("/tmp") / f"pretrain_healthcheck_batch_{args.batch_run_id}")
    db_path = batch_dir / "batch_results.sqlite"
    args.db_path = str(db_path)

    _raw, pods = load_pods(args)
    con = init_db(db_path)
    upsert_pods(con, pods)
    phases = parse_phases(args.phases, len(pods), args.target_scale)

    print(f"[batch-healthcheck] job          : {args.job_name}")
    print(f"[batch-healthcheck] batch_run_id : {args.batch_run_id}")
    print(f"[batch-healthcheck] node_count   : {len(pods)}")
    print(f"[batch-healthcheck] phases       : {','.join(phases)}")
    print(f"[batch-healthcheck] target_scale : {args.target_scale or 'auto'}")
    print(f"[batch-healthcheck] group_timeout: {args.group_timeout_seconds}s")
    print(f"[batch-healthcheck] concurrency  : {args.phase_group_concurrency or 'all groups per round'}")
    print(f"[batch-healthcheck] result_dir   : {batch_dir}")
    print(f"[batch-healthcheck] sqlite       : {db_path}")
    record_event(con, "batch_start", args.batch_run_id, {"phases": phases, "node_count": len(pods)})

    overall = "PASS"
    for phase in phases:
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
    write_group_plan_files(con, batch_dir)
    write_batch_summary(con, batch_dir, args, overall)
    cleanup_batch_tmp(args)
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
